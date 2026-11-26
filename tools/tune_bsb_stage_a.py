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

from baseline_cache import DEFAULT_CACHE_DIR, load_baseline_errors  # noqa: E402
from bsb_vectorized import BatchedBSB  # noqa: E402
from real_bench import build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0

R12_ANCHORS = [
    {"tau": 2.0, "xi": 1.2, "lambd": 3.0, "delta": 5.0, "dt": 0.2},
    {"tau": 2.1, "xi": 1.2, "lambd": 3.0, "delta": 5.0, "dt": 0.24},
    {"tau": 2.0, "xi": 1.0, "lambd": 3.5, "delta": 5.5, "dt": 0.2},
    {"tau": 2.2, "xi": 1.4, "lambd": 2.5, "delta": 6.0, "dt": 0.2},
    {"tau": 1.8, "xi": 1.2, "lambd": 3.0, "delta": 6.5, "dt": 0.2},
    {"tau": 2.5, "xi": 1.0, "lambd": 4.0, "delta": 5.0, "dt": 0.2},
    {"tau": 2.0, "xi": 1.2, "lambd": 3.0, "delta": 4.0, "dt": 0.2},
]
R23_ANCHORS = [
    {"tau": 3.0, "xi": 0.6, "lambd": 6.0, "delta": 3.0, "dt": 0.2},
    {"tau": 1.5, "xi": 0.6, "lambd": 6.0, "delta": 3.0, "dt": 0.2},
    {"tau": 3.0, "xi": 1.2, "lambd": 6.0, "delta": 5.0, "dt": 0.2},
]
R34_ANCHORS = [
    {"tau": 4.0, "xi": 0.4, "lambd": 8.0, "delta": 3.0, "dt": 0.2},
    {"tau": 2.0, "xi": 0.8, "lambd": 8.0, "delta": 5.0, "dt": 0.2},
    {"tau": 4.0, "xi": 0.8, "lambd": 8.0, "delta": 5.0, "dt": 0.2},
]

R12_BOUNDS = {
    "tau": (1.0, 4.0),
    "xi": (0.4, 2.8),
    "lambd": (1.0, 10.0),
    "delta": (2.5, 8.5),
    "dt": (0.1, 0.3),
}
HIGH_RATE_BOUNDS = {
    "tau": (1.0, 6.0),
    "xi": (0.15, 1.5),
    "lambd": (3.0, 20.0),
    "delta": (1.5, 6.0),
    "dt": (0.15, 0.35),
}


def rate_tag(rate: float) -> str:
    """Return a stable filename tag for the configured contest rates."""
    for numerator, denominator, label in ((1, 2, "0.5"), (2, 3, "2-3"), (3, 4, "3-4")):
        if abs(float(rate) - numerator / denominator) < 1e-4:
            return label
    return f"{float(rate):.6g}".replace(".", "-")


def search_bounds(rate: float) -> dict[str, tuple[float, float]]:
    """Return anchors and continuous ranges appropriate for the code rate."""
    if abs(float(rate) - 0.5) < 1e-4:
        return R12_BOUNDS
    return HIGH_RATE_BOUNDS


def scaled_bounds(rate: float, scale: float) -> dict[str, tuple[float, float]]:
    """Widen rate-specific ranges once when the best config hits an edge."""
    bounds = search_bounds(rate)
    if abs(float(rate) - 0.5) < 1e-4 or abs(float(scale) - 1.0) < 1e-9:
        return bounds
    if scale <= 0.0:
        raise ValueError("bounds scale must be positive")
    widened: dict[str, tuple[float, float]] = {}
    for key, (low, high) in bounds.items():
        center = 0.5 * (low + high)
        half = 0.5 * (high - low) * float(scale)
        widened[key] = (max(1e-6, center - half), center + half)
    return widened


def anchors_for(rate: float) -> list[dict]:
    if abs(float(rate) - 0.5) < 1e-4:
        return [dict(item) for item in R12_ANCHORS]
    if abs(float(rate) - 2.0 / 3.0) < 1e-4:
        return [dict(item) for item in R23_ANCHORS]
    if abs(float(rate) - 0.75) < 1e-4:
        return [dict(item) for item in R34_ANCHORS]
    raise ValueError(f"no Stage A anchors configured for rate {rate}")


def eval_v12_baseline(
    case_val: dict,
    v12_manifest: dict,
    max_frames: int | None = None,
    baseline_errors: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Evaluate the v12/v13c baseline on the same validation frames."""
    frames = case_val["b"] if max_frames is None else min(int(max_frames), case_val["b"])
    k = case_val["k"]
    if baseline_errors is None:
        from decode_v2_python import decode_frame as decode_v2_frame, select_case as select_v2_case
        from phase3_fixed_dsb import CheckTable

        cfg = select_v2_case(v12_manifest, case_val["n"], k, case_val["snr_db"])
        checks = CheckTable(case_val["h"])
        errors = 0
        for row in range(frames):
            llr = case_val["llr"][row]
            hard = decode_v2_frame(llr, case_val["h"], checks, cfg, ROOT)
            errors += int(np.count_nonzero(
                np.asarray(hard[:k], dtype=np.int8) != case_val["answers"][row, :k]
            ))
    else:
        if len(baseline_errors) < frames:
            raise ValueError(f"baseline cache has {len(baseline_errors)} frames, expected {frames}")
        errors = int(np.asarray(baseline_errors[:frames], dtype=np.int64).sum())
    total = frames * k
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
    baseline_cache_dir: Path = DEFAULT_CACHE_DIR,
    bounds_scale: float = 1.0,
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

    # Evaluate the inherited v12/v13c baseline on every validation frame.
    v12_path = ROOT / "artifacts" / "phase7_case_manifest_v12_safe.json"
    baseline_errors: np.ndarray | None = None
    baseline_source = "v12_python"
    if rate > 0.5:
        baseline_errors = load_baseline_errors(
            baseline_cache_dir, n, k, val_seed, expected_frames=val_samples
        )
        baseline_source = "v13c_binary_cache"
    if v12_path.exists():
        v12_manifest = json.loads(v12_path.read_text(encoding="utf-8"))
        base_score, base_ber, base_errs = eval_v12_baseline(
            val_case, v12_manifest, max_frames=val_samples,
            baseline_errors=baseline_errors,
        )
        print(
            f"Baseline on Val [{baseline_source}]: Score={base_score:.1f}, "
            f"BER={base_ber:.5f} ({base_errs}/{val_samples * k})"
        )
    else:
        base_score, base_ber, base_errs = 90000.0, 0.125, int(0.125 * val_samples * k)
        baseline_source = "estimated"
        print(f"Baseline estimated: Score={base_score:.1f}")

    # Stage A search space. R=1/2 keeps the historical ranges so the accepted
    # Stage A result stays reproducible; higher rates need weaker xi and much
    # larger check strength because their info columns have weight 3 or 2.
    bounds = scaled_bounds(rate, bounds_scale)
    is_r12 = abs(float(rate) - 0.5) < 1e-4
    rng = np.random.default_rng(train_seed)
    candidates = []

    configs_to_eval = [dict(item) for item in anchors_for(rate)]
    for _ in range(random_trials):
        configs_to_eval.append({
            "tau": float(rng.uniform(*bounds["tau"])),
            "xi": float(rng.uniform(*bounds["xi"])),
            "lambd": float(rng.uniform(*bounds["lambd"])),
            "delta": float(rng.uniform(*bounds["delta"])),
            "dt": float(
                rng.choice([0.15, 0.18, 0.20, 0.22, 0.24, 0.26])
                if is_r12 else rng.uniform(*bounds["dt"])
            ),
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
            low, high = bounds["dt"]
            cand[key] = float(np.clip(cand[key] + rng.choice([-0.02, 0.02]), low, high))
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
    edge_params = sorted(
        name for name, (low, high) in bounds.items()
        if abs(best_cfg[name] - low) < 1e-6 or abs(best_cfg[name] - high) < 1e-6
    )
    if edge_params:
        print(f"WARNING: best config is on a search bound for {', '.join(edge_params)}")

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
        "baseline_source": baseline_source,
        "bounds_scale": float(bounds_scale),
        "search_bounds": {key: list(value) for key, value in bounds.items()},
        "base_score": base_score,
        "base_ber": base_ber,
        "train_score": best_score,
        "val_score": val_score,
        "val_ber": val_ber,
        "gain": delta_score,
        "gate_passed": bool(delta_score >= 3000),
        "edge_params": edge_params,
        "needs_boundary_rerun": bool(edge_params),
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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--bounds-scale", type=float, default=1.0,
        help="widen high-rate search ranges by this factor after an edge hit",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="default is stage_a_N<N>_R<rate>.json using the requested --n/--rate",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = ROOT / "artifacts" / f"stage_a_N{args.n}_R{rate_tag(args.rate)}.json"

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
        baseline_cache_dir=args.cache_dir,
        bounds_scale=args.bounds_scale,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"Saved result to {args.out}")


if __name__ == "__main__":
    main()
