#!/usr/bin/env python3
"""Stage A': local TAP/bSB retune from the accepted v15 per-rate settings.

The search keeps the QAIA dynamics unchanged and only retunes the seven scalar
trajectory fields ``tau``, ``xi``, ``lambd``, ``delta``, ``dt``, ``beta`` and
``parity_scale``. It starts from the exact v15 case configuration, runs a
coordinate pattern search on the seed-911 training frames, and validates each
candidate frame-for-frame against the compiled v15 decoder on seeds 2024 and
31337.
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
from real_bench import RATE_SNR_DB, build_case  # noqa: E402
from tune_bsb_stage_a import rate_tag  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
STEPS = 50
TAP_KEYS = ("tau", "xi", "lambd", "delta", "dt", "beta", "parity_scale")
TAP_BOUNDS: dict[str, tuple[float, float]] = {
    "tau": (1.0, 8.0),
    "xi": (0.2, 3.0),
    "lambd": (2.0, 20.0),
    "delta": (1.5, 8.0),
    "dt": (0.12, 0.5),
    "beta": (0.05, 0.45),
    "parity_scale": (0.15, 1.2),
}
DEFAULT_V15_MANIFEST = ROOT / "submissions" / "v15_tap_bsb_candidate" / "manifest.json"
DEFAULT_CACHE_DIR_V15 = ROOT / "artifacts" / "baseline_cache_v15"
NOISE_SCALE = 1.001120


def configured_snr(rate: float) -> float:
    for configured_rate, snr in RATE_SNR_DB:
        if abs(float(configured_rate) - float(rate)) < 1e-4:
            return float(snr)
    raise ValueError(f"no configured SNR for rate {rate}")


def min_gain_for_rate(rate: float) -> float:
    """Phase 12 per-case gate: +2k at R=1/2, +1.5k at R=2/3."""
    if abs(float(rate) - 0.5) < 1e-4:
        return 2000.0
    if abs(float(rate) - 2.0 / 3.0) < 1e-4:
        return 1500.0
    return 0.0


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
    """95% paired-bootstrap CI for the score delta, resampling whole frames."""
    frames = len(base_errs)
    if len(cand_errs) != frames:
        raise ValueError(f"paired bootstrap frame mismatch: {len(base_errs)} vs {len(cand_errs)}")
    total = frames * k
    idx = rng.integers(0, frames, size=(samples, frames))
    base_scores = np.empty(samples)
    cand_scores = np.empty(samples)
    for row in range(samples):
        base_scores[row] = score_of(int(base_errs[idx[row]].sum()), total)
        cand_scores[row] = score_of(int(cand_errs[idx[row]].sum()), total)
    deltas = cand_scores - base_scores
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def widen_bounds(
    bounds: dict[str, tuple[float, float]], scale: float
) -> dict[str, tuple[float, float]]:
    """Widen every bound once around its centre by ``scale``."""
    if abs(float(scale) - 1.0) < 1e-12:
        return {key: tuple(value) for key, value in bounds.items()}  # type: ignore[dict-item]
    if scale <= 0.0:
        raise ValueError("bounds scale must be positive")
    widened: dict[str, tuple[float, float]] = {}
    for key, (low, high) in bounds.items():
        center = 0.5 * (low + high)
        half = 0.5 * (high - low) * float(scale)
        widened[key] = (max(1e-6, center - half), center + half)
    return widened


def edge_params(
    cfg: dict[str, float], bounds: dict[str, tuple[float, float]], tol: float = 1e-6
) -> list[str]:
    hits = []
    for key in TAP_KEYS:
        low, high = bounds[key]
        if abs(cfg[key] - low) <= tol or abs(cfg[key] - high) <= tol:
            hits.append(key)
    return hits


def load_start_config(
    manifest_path: Path, n: int, k: int, snr: float
) -> dict[str, float]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    matches = [
        case for case in manifest.get("cases", [])
        if int(case.get("n", -1)) == n
        and int(case.get("k", -1)) == k
        and abs(float(case.get("snr", float("nan"))) - snr) < 1e-4
    ]
    if len(matches) != 1:
        raise ValueError(f"{manifest_path}: expected one case for {(n, k, snr)}, got {len(matches)}")
    trajectories = matches[0].get("trajectories", [])
    if len(trajectories) != 1:
        raise ValueError(f"{manifest_path}: expected one trajectory for {(n, k, snr)}")
    traj = trajectories[0]
    missing = [key for key in TAP_KEYS if key not in traj]
    if missing:
        raise ValueError(f"{manifest_path}: trajectory is missing {missing}")
    return {key: float(traj[key]) for key in TAP_KEYS}


def evaluate(case: dict, sim: BatchedBSB, cfg: dict[str, float]) -> tuple[float, np.ndarray, int]:
    hard = sim.decode_batch(
        case["llr"],
        tau=cfg["tau"],
        xi=cfg["xi"],
        lambd=cfg["lambd"],
        delta=cfg["delta"],
        dt=cfg["dt"],
        steps=STEPS,
        beta=cfg["beta"],
        parity_scale=cfg["parity_scale"],
    )
    errs = per_frame_errors(hard, case["answers"], case["k"])
    total = case["llr"].shape[0] * case["k"]
    return score_of(int(errs.sum()), total), errs, int(errs.sum())


def coordinate_search(
    case: dict,
    sim: BatchedBSB,
    start: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    seed: int,
    min_move_gain: float,
    max_cycles: int,
    workers: int = 6,
) -> tuple[dict[str, float], float, float, int, int]:
    """Pattern search with coarse (0.75, 1.33) and fine (0.9, 1.11) multipliers.

    All two-way probes for one cycle are evaluated in parallel. Moves are still
    accepted one parameter at a time, using the score at the cycle start as the
    reference so two accepted moves cannot double-count the same gain.
    """
    import concurrent.futures

    rng = np.random.default_rng(seed)
    best = dict(start)
    best_score, _, _ = evaluate(case, sim, best)
    start_score = best_score
    evals = 1
    moves = 0

    for multipliers in ((0.75, 1.33), (0.9, 1.11)):
        for cycle in range(max_cycles):
            order = list(TAP_KEYS)
            rng.shuffle(order)
            probes: list[tuple[str, dict[str, float]]] = []
            for key in order:
                low, high = bounds[key]
                for multiplier in multipliers:
                    cand = dict(best)
                    cand[key] = float(np.clip(best[key] * multiplier, low, high))
                    if abs(cand[key] - best[key]) >= 1e-12:
                        probes.append((key, cand))

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(evaluate, case, sim, cand) for _, cand in probes
                ]
                scores = [future.result()[0] for future in futures]
            evals += len(probes)

            cycle_reference = best_score
            by_key: dict[str, list[tuple[float, dict[str, float]]]] = {}
            for (key, cand), score in zip(probes, scores):
                by_key.setdefault(key, []).append((score, cand))
            improved = 0
            for key in order:
                if key not in by_key:
                    continue
                score, cand = max(by_key[key], key=lambda item: item[0])
                if score > cycle_reference + min_move_gain:
                    best, best_score = cand, score
                    improved += 1
                    moves += 1
            print(
                f"  multipliers {multipliers}, cycle {cycle + 1}/{max_cycles}: "
                f"train {best_score:.1f}, {improved} params moved, {evals} evals",
                flush=True,
            )
            if improved == 0:
                break

    return best, best_score, start_score, evals, moves


def run(
    n: int,
    rate: float,
    snr: float,
    start_config: dict[str, float],
    start_source: str,
    train_samples: int,
    val_samples: int,
    train_seed: int,
    val_seeds: list[int],
    seed: int,
    cache_dir: Path,
    bounds_scale: float,
    min_move_gain: float,
    max_cycles: int,
    min_gain: float | None,
    workers: int = 6,
) -> dict:
    k = int(round(n * rate))
    threshold = min_gain_for_rate(rate) if min_gain is None else float(min_gain)
    bounds = widen_bounds(TAP_BOUNDS, bounds_scale)
    print(f"Stage A': N={n} K={k} R={rate:.4f} SNR={snr:.1f} dB")
    print(f"Start: {start_source}")
    print("  " + " ".join(f"{key}={start_config[key]:.5f}" for key in TAP_KEYS))

    t0 = time.perf_counter()
    train_case = build_case(n, k, train_samples, snr, train_seed, noise_scale=NOISE_SCALE)
    sim_train = BatchedBSB(train_case["h"])
    val_cases = {
        seed_value: build_case(n, k, val_samples, snr, seed_value, noise_scale=NOISE_SCALE)
        for seed_value in val_seeds
    }
    sims = {seed_value: BatchedBSB(case["h"]) for seed_value, case in val_cases.items()}
    print(f"cases built in {time.perf_counter() - t0:.1f} s", flush=True)

    t1 = time.perf_counter()
    best, best_score, start_score, evals, moves = coordinate_search(
        train_case, sim_train, start_config, bounds, seed,
        min_move_gain, max_cycles, workers,
    )
    elapsed = time.perf_counter() - t1
    print(
        f"search: {evals} evals in {elapsed:.1f} s, train {start_score:.1f} -> "
        f"{best_score:.1f} ({best_score - start_score:+.1f}), {moves} accepted moves",
        flush=True,
    )
    print("  " + " ".join(f"{key}={best[key]:.5f}" for key in TAP_KEYS), flush=True)

    edges = edge_params(best, bounds)
    needs_boundary_rerun = bool(edges)
    if edges:
        print(f"WARNING: best config is on a search bound for {', '.join(edges)}", flush=True)

    rng = np.random.default_rng(seed + 1)
    per_seed: dict[str, dict] = {}
    for seed_value, case in val_cases.items():
        base_errs = load_baseline_errors(
            cache_dir, n, k, seed_value, expected_frames=val_samples
        )
        base_score = score_of(int(base_errs.sum()), val_samples * k)
        cand_score, cand_errs, cand_errors = evaluate(case, sims[seed_value], best)
        low, high = paired_bootstrap(base_errs, cand_errs, k, 2000, rng)
        gain = cand_score - base_score
        per_seed[str(seed_value)] = {
            "baseline_score": base_score,
            "candidate_score": cand_score,
            "gain": gain,
            "gain_ci_95": [low, high],
            "baseline_errors": int(base_errs.sum()),
            "candidate_errors": int(cand_errors),
        }
        print(
            f"val seed {seed_value}: v15 {base_score:.1f} -> {cand_score:.1f} "
            f"(gain {gain:+.1f}, 95% CI [{low:+.1f}, {high:+.1f}])",
            flush=True,
        )

    gains = [item["gain"] for item in per_seed.values()]
    ci_lows = [item["gain_ci_95"][0] for item in per_seed.values()]
    gate_ok = (
        all(gain >= threshold for gain in gains)
        and all(low > 0.0 for low in ci_lows)
        and not needs_boundary_rerun
    )
    primary = per_seed[str(val_seeds[0])]
    result = {
        "phase": "A_prime",
        "n": n,
        "k": k,
        "rate": rate,
        "snr": snr,
        "noise_scale": NOISE_SCALE,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_seed": train_seed,
        "val_seeds": val_seeds,
        "start_config": dict(start_config),
        "start_source": start_source,
        "best_config": best,
        "baseline_source": "v15_binary_cache",
        "search_bounds": {key: list(value) for key, value in bounds.items()},
        "bounds_scale": float(bounds_scale),
        "search_seed": seed,
        "workers": int(workers),
        "min_move_gain": float(min_move_gain),
        "evaluations": evals,
        "accepted_moves": moves,
        "search_seconds": round(elapsed, 3),
        "start_train_score": start_score,
        "train_score": best_score,
        "train_gain": best_score - start_score,
        "val_score": primary["candidate_score"],
        "val_ber": primary["candidate_errors"] / (val_samples * k),
        "gain": min(gains),
        "ci_95": [ci_lows[0], max(item["gain_ci_95"][1] for item in per_seed.values())],
        "validation": per_seed,
        "edge_params": edges,
        "needs_boundary_rerun": needs_boundary_rerun,
        "gate": {
            "threshold": threshold,
            "min_val_gain": min(gains),
            "all_seeds_ge_threshold": all(gain >= threshold for gain in gains),
            "all_ci_lower_positive": all(low > 0.0 for low in ci_lows),
            "no_boundary_hit": not needs_boundary_rerun,
            "gate_passed": gate_ok,
        },
        "gate_passed": gate_ok,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--start-manifest", type=Path, default=DEFAULT_V15_MANIFEST)
    parser.add_argument("--start-config", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=911)
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[2024, 31337])
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR_V15)
    parser.add_argument("--bounds-scale", type=float, default=1.0)
    parser.add_argument("--min-move-gain", type=float, default=80.0)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-gain", type=float, default=None)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="default is artifacts/tap_a_prime_N<N>_R<rate>.json",
    )
    args = parser.parse_args()

    snr = args.snr if args.snr is not None else configured_snr(args.rate)
    k = int(round(args.n * args.rate))
    if args.start_config is not None:
        raw = json.loads(args.start_config.read_text(encoding="utf-8"))
        start_config = raw.get("best_config", raw)
        start_source = str(args.start_config)
        missing = [key for key in TAP_KEYS if key not in start_config]
        if missing:
            raise SystemExit(f"{args.start_config}: missing TAP fields {missing}")
        start_config = {key: float(start_config[key]) for key in TAP_KEYS}
    else:
        start_config = load_start_config(args.start_manifest, args.n, k, snr)
        start_source = str(args.start_manifest)

    result = run(
        n=args.n,
        rate=args.rate,
        snr=snr,
        start_config=start_config,
        start_source=start_source,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        train_seed=args.train_seed,
        val_seeds=args.val_seeds,
        seed=args.seed,
        cache_dir=args.cache_dir,
        bounds_scale=args.bounds_scale,
        min_move_gain=args.min_move_gain,
        max_cycles=args.max_cycles,
        min_gain=args.min_gain,
        workers=args.workers,
    )
    out = args.out or ROOT / "artifacts" / f"tap_a_prime_N{args.n}_R{rate_tag(args.rate)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
