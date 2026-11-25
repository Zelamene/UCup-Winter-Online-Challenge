#!/usr/bin/env python3
"""Stage A wide grid search for bSB on a single (N, R, SNR) case."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from bsb_vectorized import BatchedBSB  # noqa: E402
from real_bench import build_case  # noqa: E402
from decode_v3_python import select_case, decode_frame  # noqa: E402

SCORE_SCALE = 100_000.0


def eval_v12_baseline(case_val: dict, v12_manifest: dict, max_frames: int = 150) -> tuple[float, float, int]:
    """Evaluate current v12 baseline on the validation set."""
    from decode_v2_python import decode_frame as decode_v2_frame, select_case as select_v2_case
    from phase3_fixed_dsb import CheckTable

    cfg = select_v2_case(v12_manifest, case_val["n"], case_val["k"], case_val["snr_db"])
    checks = CheckTable(case_val["h"])
    k = case_val["k"]
    errors = 0
    frames = min(max_frames, case_val["b"])
    total = frames * k
    for row in range(frames):
        llr = case_val["llr"][row]
        hard = decode_v2_frame(llr, case_val["h"], checks, cfg, TOOLS_DIR.parent)
        errors += int(np.count_nonzero(np.asarray(hard[:k], dtype=np.int8) != case_val["answers"][row, :k]))
    ber = max(errors / total, 1.0 / total)
    score = -SCORE_SCALE * math.log10(ber)
    return score, ber, errors


def run_stage_a(
    n: int,
    rate: float,
    snr: float,
    train_samples: int = 300,
    val_samples: int = 500,
    train_seed: int = 911,
    val_seed: int = 2024,
    noise_scale: float = 1.001120,
    random_trials: int = 300,
    polish_trials: int = 50,
) -> dict:
    k = int(round(n * rate))
    print(f"\n========================================================")
    print(f"Stage A Tuning: N={n}, K={k}, Rate={rate:.3f}, SNR={snr:.1f} dB")
    print(f"Train: {train_samples} frames (seed {train_seed}), Val: {val_samples} frames (seed {val_seed})")
    print(f"========================================================")

    # Build datasets
    t0 = time.perf_counter()
    train_case = build_case(n, k, train_samples, snr, train_seed, noise_scale=noise_scale)
    val_case = build_case(n, k, val_samples, snr, val_seed, noise_scale=noise_scale)
    sim_train = BatchedBSB(train_case["h"])
    sim_val = BatchedBSB(val_case["h"])
    print(f"Cases built and topology pre-computed in {time.perf_counter() - t0:.2f} s")

    # Evaluate v12 baseline
    v12_path = TOOLS_DIR.parent / "artifacts" / "phase7_case_manifest_v12_safe.json"
    if v12_path.exists():
        v12_manifest = json.loads(v12_path.read_text(encoding="utf-8"))
        # Using decode_frame with v12 probe params
        base_score, base_ber, base_errs = eval_v12_baseline(val_case, v12_manifest)
        print(f"v12 Baseline on Val: Score={base_score:.1f}, BER={base_ber:.5f} ({base_errs}/{val_samples * k})")
    else:
        base_score, base_ber, base_errs = 90000.0, 0.125, int(0.125 * val_samples * k)
        print(f"v12 Baseline estimated: Score={base_score:.1f}")

    # Stage A Wide Search Space
    # tau in [1.0, 4.0], xi in [0.3, 3.0], lambda in [1.0, 12.0], delta in [2.0, 8.0], dt in [0.1, 0.3]
    rng = np.random.default_rng(train_seed)
    candidates = []

    # Include anchors known to perform well
    known_anchors = [
        {"tau": 2.0, "xi": 1.2, "lambd": 3.0, "delta": 5.0, "dt": 0.2},
        {"tau": 2.1, "xi": 1.2, "lambd": 3.0, "delta": 5.0, "dt": 0.24},
        {"tau": 2.0, "xi": 1.0, "lambd": 3.5, "delta": 5.5, "dt": 0.2},
        {"tau": 2.2, "xi": 1.4, "lambd": 2.5, "delta": 6.0, "dt": 0.2},
        {"tau": 1.8, "xi": 1.2, "lambd": 3.0, "delta": 6.5, "dt": 0.2},
        {"tau": 2.5, "xi": 1.0, "lambd": 4.0, "delta": 5.0, "dt": 0.2},
        {"tau": 2.0, "xi": 1.2, "lambd": 3.0, "delta": 4.0, "dt": 0.2},
    ]

    configs_to_eval = list(known_anchors)
    for _ in range(random_trials):
        configs_to_eval.append({
            "tau": float(rng.uniform(1.0, 4.0)),
            "xi": float(rng.uniform(0.4, 2.8)),
            "lambd": float(rng.uniform(1.0, 10.0)),
            "delta": float(rng.uniform(2.5, 8.5)),
            "dt": float(rng.choice([0.15, 0.18, 0.20, 0.22, 0.24, 0.26])),
        })

    print(f"Evaluating {len(configs_to_eval)} configurations on 6 worker threads...", flush=True)
    t_start = time.perf_counter()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futs = [
            executor.submit(
                sim_train.evaluate,
                train_case["llr"], train_case["answers"], k,
                tau=cfg["tau"], xi=cfg["xi"], lambd=cfg["lambd"], delta=cfg["delta"], dt=cfg["dt"]
            )
            for cfg in configs_to_eval
        ]
        for cfg, fut in zip(configs_to_eval, futs):
            score, ber, errs = fut.result()
            candidates.append((score, cfg))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cfg = candidates[0]
    t_rand = time.perf_counter() - t_start
    print(f"Search done in {t_rand:.2f} s. Top score on train: {best_score:.1f}", flush=True)
    print(f"Top config: tau={best_cfg['tau']:.4f}, xi={best_cfg['xi']:.4f}, lambd={best_cfg['lambd']:.4f}, delta={best_cfg['delta']:.4f}, dt={best_cfg['dt']:.4f}", flush=True)

    # Coordinate polishing around top candidate
    print(f"Polishing top candidates with {polish_trials} coordinate steps...")
    cur_cfg = dict(best_cfg)
    cur_score = best_score
    keys = ["tau", "xi", "lambd", "delta", "dt"]

    for step in range(polish_trials):
        cand = dict(cur_cfg)
        key = str(rng.choice(keys))
        if key == "dt":
            cand[key] = float(np.clip(cand[key] + rng.choice([-0.02, 0.02]), 0.1, 0.3))
        else:
            scale = math.exp(rng.normal(0.0, 0.08))
            cand[key] = float(cand[key] * scale)

        score, ber, errs = sim_train.evaluate(
            train_case["llr"], train_case["answers"], k,
            tau=cand["tau"], xi=cand["xi"], lambd=cand["lambd"], delta=cand["delta"], dt=cand["dt"]
        )
        if score > cur_score:
            cur_score = score
            cur_cfg = cand
            candidates.append((cur_score, cur_cfg))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cfg = candidates[0]
    print(f"Polished train score: {best_score:.1f}")
    print(f"Polished config: tau={best_cfg['tau']:.4f}, xi={best_cfg['xi']:.4f}, lambd={best_cfg['lambd']:.4f}, delta={best_cfg['delta']:.4f}, dt={best_cfg['dt']:.4f}")

    # Validation on held-out 500 frames
    val_score, val_ber, val_errs = sim_val.evaluate(
        val_case["llr"], val_case["answers"], k,
        tau=best_cfg["tau"], xi=best_cfg["xi"], lambd=best_cfg["lambd"], delta=best_cfg["delta"], dt=best_cfg["dt"]
    )
    delta_score = val_score - base_score

    # Confidence Interval on BER (Wilson score interval / normal approximation)
    p = val_ber
    total_bits = val_samples * k
    se = math.sqrt(p * (1 - p) / total_bits)
    ci_p_low = max(1e-7, p - 1.96 * se)
    ci_p_high = p + 1.96 * se
    ci_score_high = -SCORE_SCALE * math.log10(ci_p_low)
    ci_score_low = -SCORE_SCALE * math.log10(ci_p_high)

    print(f"\n---------------- Validation Results ----------------")
    print(f"Validation Score: {val_score:.1f} (95% CI: [{ci_score_low:.1f}, {ci_score_high:.1f}])")
    print(f"Validation BER:   {val_ber:.5f} ({val_errs}/{total_bits})")
    print(f"v12 Baseline:     {base_score:.1f} (BER {base_ber:.5f})")
    print(f"Gain vs Baseline: {delta_score:+.1f} points")
    print(f"Gate (>= +3000):  {'PASSED' if delta_score >= 3000 else 'FAILED'}")
    print(f"----------------------------------------------------")

    result = {
        "n": n,
        "k": k,
        "rate": rate,
        "snr": snr,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "noise_scale": noise_scale,
        "base_score": base_score,
        "base_ber": base_ber,
        "train_score": best_score,
        "val_score": val_score,
        "val_ber": val_ber,
        "gain": delta_score,
        "gate_passed": bool(delta_score >= 3000),
        "ci_95": [ci_score_low, ci_score_high],
        "best_config": best_cfg,
        "top_5": [c[1] for c in candidates[:5]],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=911)
    parser.add_argument("--val-seed", type=int, default=2024)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--polish", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    res = run_stage_a(
        n=args.n,
        rate=args.rate,
        snr=args.snr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        random_trials=args.trials,
        polish_trials=args.polish,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"Saved result to {args.out}")


if __name__ == "__main__":
    main()
