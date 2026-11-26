#!/usr/bin/env python3
"""Stage B: piecewise-linear bSB schedules, tuned on one seed and validated on two others.

Starting from a Stage A constant configuration, replace xi / lambd / delta / dt
with piecewise-linear schedules defined by KNOTS knots over the 50 steps, then
run coordinate pattern search in log space. The accepted objective is the train
score; the reported numbers are held-out gains measured on the Stage A
validation seed (2024) and on an independent third seed, each paired
frame-for-frame against the v12 baseline, with a paired bootstrap CI.
"""

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
from tune_bsb_stage_a import rate_tag  # noqa: E402

SCORE_SCALE = 100_000.0
STEPS = 50
KNOT_STEPS = (0, 12, 25, 37, 49)
KNOTS = len(KNOT_STEPS)
GROUPS = ("xi", "lambd", "delta", "dt")
PARAM_NAMES = tuple(f"{g}[{i}]" for g in GROUPS for i in range(KNOTS))


def interp_schedule(knots: list[float]) -> np.ndarray:
    """Piecewise-linear interpolation of knot values onto the 50 steps."""
    return np.interp(np.arange(STEPS), np.asarray(KNOT_STEPS, dtype=float), np.asarray(knots, dtype=float))


def expand(params: np.ndarray) -> dict[str, np.ndarray]:
    """Turn a 20-vector of knot values into per-key schedules."""
    out: dict[str, np.ndarray] = {}
    for index, group in enumerate(GROUPS):
        block = params[index * KNOTS:(index + 1) * KNOTS]
        out[group] = interp_schedule(block.tolist())
    return out


def flat_from_config(config: dict) -> np.ndarray:
    """Replicate a Stage A scalar configuration across every knot."""
    params = []
    for group in GROUPS:
        value = float(config.get(group, 0.2 if group == "dt" else 1.0))
        params.extend([value] * KNOTS)
    return np.asarray(params, dtype=float)


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def per_frame_errors(hard: np.ndarray, answers: np.ndarray, k: int) -> np.ndarray:
    return np.count_nonzero(hard[:, :k] != answers[:, :k], axis=1)


def paired_bootstrap(
    base_errs: np.ndarray,
    cand_errs: np.ndarray,
    k: int,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI for the score delta, resampling frames in pairs."""
    frames = len(base_errs)
    total = frames * k
    idx = rng.integers(0, frames, size=(samples, frames))
    base_scores = np.empty(samples)
    cand_scores = np.empty(samples)
    for row in range(samples):
        base_scores[row] = score_of(int(base_errs[idx[row]].sum()), total)
        cand_scores[row] = score_of(int(cand_errs[idx[row]].sum()), total)
    deltas = cand_scores - base_scores
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def evaluate_case(
    case: dict,
    sim: BatchedBSB,
    params: np.ndarray,
    tau: float,
    k: int,
) -> tuple[float, np.ndarray, int]:
    schedules = expand(params)
    hard = sim.decode_batch(
        case["llr"],
        tau=tau,
        xi=schedules["xi"],
        lambd=schedules["lambd"],
        delta=schedules["delta"],
        dt=schedules["dt"],
        steps=STEPS,
    )
    errs = per_frame_errors(hard, case["answers"], k)
    total = case["llr"].shape[0] * k
    return score_of(int(errs.sum()), total), errs, int(errs.sum())


def baseline_per_frame(
    case: dict,
    v12_manifest: dict,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """v12/v13c baseline error count per frame on the same frames."""
    if case["k"] / case["n"] > 0.5:
        return load_baseline_errors(
            cache_dir, case["n"], case["k"], int(case["seed"]),
            expected_frames=case["b"],
        )
    from decode_v2_python import decode_frame as decode_v2_frame, select_case as select_v2_case
    from phase3_fixed_dsb import CheckTable

    cfg = select_v2_case(v12_manifest, case["n"], case["k"], case["snr_db"])
    checks = CheckTable(case["h"])
    k = case["k"]
    errs = np.empty(case["llr"].shape[0], dtype=np.int64)
    for row in range(case["llr"].shape[0]):
        hard = decode_v2_frame(case["llr"][row], case["h"], checks, cfg, TOOLS_DIR.parent)
        errs[row] = np.count_nonzero(np.asarray(hard[:k], dtype=np.int8) != case["answers"][row, :k])
    return errs


def pattern_search(
    sim: BatchedBSB,
    case: dict,
    k: int,
    tau: float,
    start: np.ndarray,
    passes: int,
    steps: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Coordinate pattern search in log space on the 20 knot values."""
    rng = np.random.default_rng(seed)
    best = start.copy()
    best_score, _, _ = evaluate_case(case, sim, best, tau, k)
    evals = 1
    for pass_index, step in enumerate(steps):
        improved = 0
        order = rng.permutation(len(best))
        for index in order:
            for direction in (1.0, -1.0):
                cand = best.copy()
                cand[index] = max(cand[index] * math.exp(direction * step), 1e-6)
                score, _, _ = evaluate_case(case, sim, cand, tau, k)
                evals += 1
                if score > best_score + 1e-9:
                    best, best_score = cand, score
                    improved += 1
                    break
        print(
            f"  pass {pass_index + 1}/{len(steps)} (step {step:.3f}): "
            f"train {best_score:.1f}, {improved} params moved, {evals} evals",
            flush=True,
        )
        if improved == 0 and pass_index >= passes - 1:
            break
    return best, best_score, evals


def run(
    n: int,
    rate: float,
    snr: float | None,
    stage_a_path: Path,
    train_samples: int,
    val_samples: int,
    train_seed: int,
    val_seeds: list[int],
    seed: int,
    max_passes: int,
    quick: bool,
    baseline_cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    k = int(round(n * rate))
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    for key, expected in (("n", n), ("k", k), ("rate", rate)):
        if key in stage_a and abs(float(stage_a[key]) - float(expected)) > 1e-4:
            raise ValueError(
                f"{stage_a_path}: {key}={stage_a[key]} does not match requested {expected}"
            )
    if snr is None:
        if "snr" not in stage_a:
            raise ValueError(f"{stage_a_path}: no snr field; pass --snr explicitly")
        snr = float(stage_a["snr"])
    elif abs(float(stage_a.get("snr", snr)) - float(snr)) > 1e-4:
        raise ValueError(
            f"{stage_a_path}: snr={stage_a['snr']} does not match requested {snr}"
        )
    noise_scale = float(stage_a.get("noise_scale", 1.001120))
    base_cfg = stage_a["best_config"]
    tau = float(base_cfg["tau"])
    print(f"Stage B: N={n} K={k} R={rate:.3f} SNR={snr:.1f}")
    print(
        f"Stage A start: tau={tau:.4f} xi={base_cfg['xi']:.4f} lambd={base_cfg['lambd']:.4f} "
        f"delta={base_cfg['delta']:.4f} dt={base_cfg['dt']:.4f}"
    )

    t0 = time.perf_counter()
    train_case = build_case(n, k, train_samples, snr, train_seed, noise_scale=noise_scale)
    val_cases = {
        seed_value: build_case(n, k, val_samples, snr, seed_value, noise_scale=noise_scale)
        for seed_value in val_seeds
    }
    sim_train = BatchedBSB(train_case["h"])
    sims = {seed_value: BatchedBSB(case["h"]) for seed_value, case in val_cases.items()}
    print(f"cases built in {time.perf_counter() - t0:.1f} s")

    start = flat_from_config(base_cfg)
    flat_train_score, _, _ = evaluate_case(train_case, sim_train, start, tau, k)
    print(f"flat Stage A config, train score: {flat_train_score:.1f}")

    steps = (0.10, 0.05) if quick else (0.12, 0.06, 0.03)
    t1 = time.perf_counter()
    best, best_train_score, evals = pattern_search(
        sim_train, train_case, k, tau, start, max_passes, steps, seed
    )
    elapsed = time.perf_counter() - t1
    print(f"search: {evals} evals in {elapsed:.1f} s, train {flat_train_score:.1f} -> {best_train_score:.1f}")

    v12_manifest = json.loads(
        (TOOLS_DIR.parent / "artifacts" / "phase7_case_manifest_v12_safe.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(seed + 1)
    per_seed = {}
    for seed_value, case in val_cases.items():
        base_errs = baseline_per_frame(case, v12_manifest, baseline_cache_dir)
        base_score = score_of(int(base_errs.sum()), case["llr"].shape[0] * k)
        cand_score, cand_errs, cand_errors = evaluate_case(case, sims[seed_value], best, tau, k)
        low, high = paired_bootstrap(base_errs, cand_errs, k, 2000, rng)
        per_seed[str(seed_value)] = {
            "base_score": base_score,
            "candidate_score": cand_score,
            "gain": cand_score - base_score,
            "gain_ci_95": [low, high],
            "candidate_errors": int(cand_errors),
            "base_errors": int(base_errs.sum()),
        }
        print(
            f"val seed {seed_value}: base {base_score:.1f} -> {cand_score:.1f} "
            f"(gain {cand_score - base_score:+.1f}, 95% CI [{low:+.1f}, {high:+.1f}])"
        )

    schedules = expand(best)
    # Stage A flat-schedule gain on the same validation frames, for comparison.
    flat_gain = {}
    for seed_value, case in val_cases.items():
        flat_score, _, _ = evaluate_case(case, sims[seed_value], start, tau, k)
        flat_gain[str(seed_value)] = flat_score - per_seed[str(seed_value)]["base_score"]

    result = {
        "n": n,
        "k": k,
        "rate": rate,
        "snr": snr,
        "stage_a_config": base_cfg,
        "train_seed": train_seed,
        "val_seeds": val_seeds,
        "val_samples": val_samples,
        "noise_scale": noise_scale,
        "baseline_source": "v13c_binary_cache" if rate > 0.5 else "v12_python",
        "knot_steps": list(KNOT_STEPS),
        "parameters": list(PARAM_NAMES),
        "flat_train_score": flat_train_score,
        "train_score": best_train_score,
        "evaluations": evals,
        "search_seconds": elapsed,
        "param_vector": best.tolist(),
        "schedules": {key: value.tolist() for key, value in schedules.items()},
        "validation": per_seed,
        "flat_stage_a_gain": flat_gain,
        "gate": {
            "train_gain": best_train_score - flat_train_score,
            "min_val_gain": min(item["gain"] for item in per_seed.values()),
            "all_seeds_ge_3000": all(item["gain"] >= 3000 for item in per_seed.values()),
            "schedules_beat_flat": all(
                per_seed[str(seed)]["gain"] > flat_gain[str(seed)] for seed in val_seeds
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=648)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument(
        "--snr", type=float, default=None,
        help="defaults to the snr recorded in the Stage A artifact",
    )
    parser.add_argument("--stage-a", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=911)
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[2024, 31337])
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="two coarse passes only")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.stage_a is None:
        args.stage_a = (
            TOOLS_DIR.parent / "artifacts" / f"stage_a_N{args.n}_R{rate_tag(args.rate)}.json"
        )
    if args.out is None:
        args.out = (
            TOOLS_DIR.parent / "artifacts" / f"stage_b_N{args.n}_R{rate_tag(args.rate)}.json"
        )

    result = run(
        n=args.n,
        rate=args.rate,
        snr=args.snr,
        stage_a_path=args.stage_a,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        train_seed=args.train_seed,
        val_seeds=args.val_seeds,
        seed=args.seed,
        max_passes=args.max_passes,
        quick=args.quick,
        baseline_cache_dir=args.cache_dir,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.out}")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
