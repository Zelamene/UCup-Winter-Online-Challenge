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
from tune_tap_local import min_gain_for_rate  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
DEFAULT_CACHE_DIR_V15 = ROOT / "artifacts" / "baseline_cache_v15"
STEPS = 50
KNOT_STEPS = (0, 12, 25, 37, 49)
KNOTS = len(KNOT_STEPS)
BASE_GROUPS = ("xi", "lambd", "delta", "dt")


def groups_for(beta_knots: bool) -> tuple[str, ...]:
    """Stage B' can add a beta schedule; Stage B keeps the four v14 groups."""
    return BASE_GROUPS + ("beta",) if beta_knots else BASE_GROUPS


def parameter_names(groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{group}[{index}]" for group in groups for index in range(KNOTS))


def interp_schedule(knots: list[float]) -> np.ndarray:
    """Piecewise-linear interpolation of knot values onto the 50 steps."""
    return np.interp(np.arange(STEPS), np.asarray(KNOT_STEPS, dtype=float), np.asarray(knots, dtype=float))


def expand(params: np.ndarray, groups: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Turn a knot vector into per-key per-step schedules."""
    out: dict[str, np.ndarray] = {}
    for index, group in enumerate(groups):
        block = params[index * KNOTS:(index + 1) * KNOTS]
        out[group] = interp_schedule(block.tolist())
    return out


def flat_from_config(config: dict, groups: tuple[str, ...]) -> np.ndarray:
    """Replicate a Stage A scalar configuration across every knot."""
    params = []
    for group in groups:
        default = 0.2 if group == "dt" else (0.0 if group == "beta" else 1.0)
        value = float(config.get(group, default))
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
    groups: tuple[str, ...],
    beta: float = 0.0,
    parity_scale: float = 1.0,
) -> tuple[float, np.ndarray, int]:
    schedules = expand(params, groups)
    hard = sim.decode_batch(
        case["llr"],
        tau=tau,
        xi=schedules["xi"],
        lambd=schedules["lambd"],
        delta=schedules["delta"],
        dt=schedules["dt"],
        steps=STEPS,
        beta=schedules.get("beta", beta),
        parity_scale=parity_scale,
    )
    errs = per_frame_errors(hard, case["answers"], k)
    total = case["llr"].shape[0] * k
    return score_of(int(errs.sum()), total), errs, int(errs.sum())


def baseline_per_frame(
    case: dict,
    v12_manifest: dict,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    baseline_source: str = "v13c_binary_cache",
) -> np.ndarray:
    """Baseline error count per frame on the same frames."""
    if baseline_source == "v15_binary_cache" or case["k"] / case["n"] > 0.5:
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
    groups: tuple[str, ...],
    beta: float = 0.0,
    parity_scale: float = 1.0,
    workers: int = 6,
) -> tuple[np.ndarray, float, int]:
    """Coordinate pattern search in log space on the knot values.

    Both directions for every knot are evaluated in parallel per pass. Moves are
    accepted against the pass-start score, so simultaneous moves do not
    double-count the same gain.
    """
    import concurrent.futures

    rng = np.random.default_rng(seed)
    best = start.copy()
    best_score, _, _ = evaluate_case(case, sim, best, tau, k, groups, beta, parity_scale)
    evals = 1
    for pass_index, step in enumerate(steps):
        order = rng.permutation(len(best))
        probes: list[tuple[int, np.ndarray]] = []
        for index in order:
            for direction in (1.0, -1.0):
                cand = best.copy()
                cand[index] = max(cand[index] * math.exp(direction * step), 1e-6)
                probes.append((int(index), cand))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    evaluate_case, case, sim, cand, tau, k, groups, beta, parity_scale
                )
                for _, cand in probes
            ]
            scores = [future.result()[0] for future in futures]
        evals += len(probes)

        pass_reference = best_score
        by_index: dict[int, list[tuple[float, np.ndarray]]] = {}
        for (index, cand), score in zip(probes, scores):
            by_index.setdefault(index, []).append((score, cand))
        improved = 0
        for index in order:
            index = int(index)
            score, cand = max(by_index[index], key=lambda item: item[0])
            if score > pass_reference + 1e-9:
                best, best_score = cand, score
                improved += 1
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
    baseline_cache_dir: Path | None = None,
    tap_from_path: Path | None = None,
    beta_knots: bool = False,
    min_gain: float | None = None,
    workers: int = 6,
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
    base_cfg = dict(stage_a["best_config"])

    if tap_from_path is not None:
        tap_artifact = json.loads(tap_from_path.read_text(encoding="utf-8"))
        for key, expected in (("n", n), ("k", k), ("rate", rate)):
            if key in tap_artifact and abs(float(tap_artifact[key]) - float(expected)) > 1e-4:
                raise ValueError(
                    f"{tap_from_path}: {key}={tap_artifact[key]} does not match requested {expected}"
                )
        tap_cfg = dict(tap_artifact.get("best_config", tap_artifact))
        required = ("tau", "xi", "lambd", "delta", "dt", "beta", "parity_scale")
        missing = [key for key in required if key not in tap_cfg]
        if missing:
            raise ValueError(f"{tap_from_path}: missing TAP fields {missing}")
        base_cfg = {key: float(tap_cfg[key]) for key in required}

    tau = float(base_cfg["tau"])
    beta = float(base_cfg.get("beta", 0.0))
    parity_scale = float(base_cfg.get("parity_scale", 1.0))
    groups = groups_for(beta_knots)
    if beta_knots and "beta" not in base_cfg:
        raise ValueError("--beta-knots requires --tap-from with an explicit beta field")
    baseline_source = "v15_binary_cache" if tap_from_path is not None else (
        "v13c_binary_cache" if rate > 0.5 else "v12_python"
    )
    if baseline_cache_dir is None:
        baseline_cache_dir = (
            DEFAULT_CACHE_DIR_V15 if tap_from_path is not None else DEFAULT_CACHE_DIR
        )
    phase = "B_prime" if tap_from_path is not None else "B"
    print(f"Stage {phase}: N={n} K={k} R={rate:.3f} SNR={snr:.1f}")
    print("  " + " ".join(f"{key}={base_cfg[key]:.5f}" for key in base_cfg))

    t0 = time.perf_counter()
    train_case = build_case(n, k, train_samples, snr, train_seed, noise_scale=noise_scale)
    val_cases = {
        seed_value: build_case(n, k, val_samples, snr, seed_value, noise_scale=noise_scale)
        for seed_value in val_seeds
    }
    sim_train = BatchedBSB(train_case["h"])
    sims = {seed_value: BatchedBSB(case["h"]) for seed_value, case in val_cases.items()}
    print(f"cases built in {time.perf_counter() - t0:.1f} s")

    start = flat_from_config(base_cfg, groups)
    flat_train_score, _, _ = evaluate_case(
        train_case, sim_train, start, tau, k, groups, beta, parity_scale
    )
    print(f"flat Stage A config, train score: {flat_train_score:.1f}")

    steps = (0.10, 0.05) if quick else (0.12, 0.06, 0.03)
    t1 = time.perf_counter()
    best, best_train_score, evals = pattern_search(
        sim_train, train_case, k, tau, start, max_passes, steps, seed,
        groups, beta, parity_scale, workers,
    )
    elapsed = time.perf_counter() - t1
    print(f"search: {evals} evals in {elapsed:.1f} s, train {flat_train_score:.1f} -> {best_train_score:.1f}")

    v12_manifest = json.loads(
        (TOOLS_DIR.parent / "artifacts" / "phase7_case_manifest_v12_safe.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(seed + 1)
    per_seed = {}
    for seed_value, case in val_cases.items():
        base_errs = baseline_per_frame(
            case, v12_manifest, baseline_cache_dir, baseline_source
        )
        base_score = score_of(int(base_errs.sum()), case["llr"].shape[0] * k)
        cand_score, cand_errs, cand_errors = evaluate_case(
            case, sims[seed_value], best, tau, k, groups, beta, parity_scale
        )
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
            f"val seed {seed_value}: {baseline_source} {base_score:.1f} -> {cand_score:.1f} "
            f"(gain {cand_score - base_score:+.1f}, 95% CI [{low:+.1f}, {high:+.1f}])"
        )

    schedules = expand(best, groups)
    # Flat Stage A'/A gain on the same validation frames, for comparison.
    flat_gain = {}
    for seed_value, case in val_cases.items():
        flat_score, _, _ = evaluate_case(
            case, sims[seed_value], start, tau, k, groups, beta, parity_scale
        )
        flat_gain[str(seed_value)] = flat_score - per_seed[str(seed_value)]["base_score"]

    threshold = min_gain_for_rate(rate) if min_gain is None else float(min_gain)
    min_val_gain = min(item["gain"] for item in per_seed.values())
    all_ge = all(item["gain"] >= threshold for item in per_seed.values())
    all_ci_positive = all(item["gain_ci_95"][0] > 0.0 for item in per_seed.values())
    beats_flat = all(
        per_seed[str(seed_value)]["gain"] > flat_gain[str(seed_value)]
        for seed_value in val_seeds
    )
    gate_passed = all_ge and all_ci_positive and beats_flat

    result = {
        "phase": phase,
        "n": n,
        "k": k,
        "rate": rate,
        "snr": snr,
        "stage_a_config": base_cfg,
        "tap_from": str(tap_from_path) if tap_from_path is not None else None,
        "beta_knots": bool(beta_knots),
        "train_seed": train_seed,
        "val_seeds": val_seeds,
        "val_samples": val_samples,
        "noise_scale": noise_scale,
        "baseline_source": baseline_source,
        "baseline_cache_dir": str(baseline_cache_dir),
        "knot_steps": list(KNOT_STEPS),
        "parameters": list(parameter_names(groups)),
        "flat_train_score": flat_train_score,
        "train_score": best_train_score,
        "evaluations": evals,
        "search_seconds": elapsed,
        "workers": int(workers),
        "param_vector": best.tolist(),
        "schedules": {key: value.tolist() for key, value in schedules.items()},
        "validation": per_seed,
        "flat_stage_a_gain": flat_gain,
        "gate": {
            "threshold": threshold,
            "train_gain": best_train_score - flat_train_score,
            "min_val_gain": min_val_gain,
            "all_seeds_ge_threshold": all_ge,
            "all_ci_lower_positive": all_ci_positive,
            "schedules_beat_flat": beats_flat,
            "gate_passed": gate_passed,
            # Kept for older validation tooling.
            "all_seeds_ge_3000": all(item["gain"] >= 3000 for item in per_seed.values()),
        },
        "gate_passed": gate_passed,
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
    parser.add_argument(
        "--tap-from", type=Path, default=None,
        help="Stage A' TAP artifact (or raw config); enables Stage B' with v15 baseline",
    )
    parser.add_argument(
        "--beta-knots", action="store_true",
        help="also schedule beta over the five Stage B knots (requires --tap-from)",
    )
    parser.add_argument("--min-gain", type=float, default=None)
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=911)
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[2024, 31337])
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="two coarse passes only")
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="baseline cache; defaults to v15 cache for --tap-from, v12/v13c otherwise",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.stage_a is None:
        args.stage_a = (
            TOOLS_DIR.parent / "artifacts" / f"stage_a_N{args.n}_R{rate_tag(args.rate)}.json"
        )
    if args.out is None:
        stem = "tap_b_prime" if args.tap_from is not None else "stage_b"
        args.out = (
            TOOLS_DIR.parent / "artifacts" / f"{stem}_N{args.n}_R{rate_tag(args.rate)}.json"
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
        tap_from_path=args.tap_from,
        beta_knots=args.beta_knots,
        min_gain=args.min_gain,
        workers=args.workers,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.out}")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
