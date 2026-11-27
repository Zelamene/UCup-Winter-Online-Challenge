#!/usr/bin/env python3
"""Phase 15 workstream C: rate-2/3 joint-pair side bet on case 11.

Starts from the accepted v17 case-11 trajectory, enables exact Gray-pair
reaction terms with ``pair_gamma=0.15``, and tunes only ``pair_gamma``, ``beta``,
``parity_scale`` and ``dt`` with the local CMA-ES. Tuning uses the first
``--tune-frames`` frames of the cache-matched seed-911 case, so every gain is
paired against the compiled v17 on identical frames.

Gate: case 11 >= +1,500 on seeds 2024 and 31337 with both 95% paired-bootstrap
lower bounds above zero. If either seed fails, workstream C is dropped.
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
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402
from cmaes_lite import CMAESLite  # noqa: E402

TUNE_KEYS = ("pair_gamma", "beta", "parity_scale", "dt")
BASE_OVERRIDES = {"joint_pairs": 1, "pair_gamma": 0.15, "pair_scale": 1.0}
DEFAULT_OUT = p.PHASE15_ARTIFACTS / "workstream_c_case11.json"


def make_config(base: dict, multipliers: np.ndarray) -> dict:
    result = dict(base)
    for key, multiplier in zip(TUNE_KEYS, multipliers):
        result[key] = float(base[key]) * float(multiplier)
    return result


def evaluate(workspace: p.CaseWorkspace, config: dict) -> tuple[float, np.ndarray]:
    hard = p.decode_hard(workspace, config)
    errors = p.per_frame_errors(hard, workspace.case)
    score = p.score_from_errors(errors, workspace.k)
    return score, errors


def validation_report(base: dict, best_multipliers: np.ndarray, seeds: list[int], frames: int) -> dict:
    config = make_config(base, best_multipliers)
    report: dict[str, dict] = {}
    for seed in seeds:
        workspace = p.CaseWorkspace(3888, 2592, seed, 500, snr=9.0, frame_slice=frames)
        score, errors = evaluate(workspace, config)
        report[str(seed)] = {
            "workspace": workspace.summary(),
            "config": config,
            "score": score,
            "gain": p.gain_report(workspace, errors),
        }
    return report


def decision(report: dict) -> dict:
    gains = []
    for seed, payload in report.items():
        gain = payload["gain"]["gain"]
        low = payload["gain"]["gain_ci_95"][0]
        gains.append(gain > 0.0 and low > 0.0 and gain >= 1500.0)
    passed = len(gains) == len(report) and all(gains)
    return {
        "gate": "promote_C" if passed else "drop_C",
        "required_gain": 1500.0,
        "seeds_passed": [seed for seed, payload in report.items()
                         if payload["gain"]["gain"] >= 1500.0
                         and payload["gain"]["gain_ci_95"][0] > 0.0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--tune-frames", type=int, default=100)
    parser.add_argument("--val-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=911, help="training seed (cache must exist)")
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[2024, 31337])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cache_frames = 300 if args.seed == 911 else 500
    if args.tune_frames > cache_frames:
        raise ValueError(f"tune-frames must be <= {cache_frames} for seed {args.seed}")
    base = p.with_overrides(p.load_v17_config(3888, 2592), **BASE_OVERRIDES)
    train = p.CaseWorkspace(
        3888, 2592, args.seed, cache_frames, snr=9.0, frame_slice=args.tune_frames
    )
    holdout = p.CaseWorkspace(
        3888, 2592, args.val_seeds[0], 500, snr=9.0, frame_slice=args.tune_frames
    )

    optimizer = CMAESLite(
        len(TUNE_KEYS), mean=np.ones(len(TUNE_KEYS)), sigma=0.15,
        population_size=12, mu=6, seed=5150, bounds=(0.5, 2.0),
    )
    history: list[dict] = []
    best_multipliers = np.ones(len(TUNE_KEYS))
    best_fitness = math.inf
    started = time.time()
    for generation in range(args.generations):
        candidates = optimizer.ask()
        fitness: list[float] = []
        for candidate in candidates:
            score, _ = evaluate(train, make_config(base, candidate))
            fitness.append(-score)
        optimizer.tell(candidates, fitness)
        if optimizer.best_fitness < best_fitness:
            best_fitness = float(optimizer.best_fitness)
            best_multipliers = np.asarray(optimizer.best_solution).copy()
        checkpoint = {
            "generation": generation + 1,
            "evaluations": optimizer.evaluations,
            "best_fitness": best_fitness,
            "best_multipliers": best_multipliers.tolist(),
            "best_config": make_config(base, best_multipliers),
            "sigma": optimizer.sigma,
        }
        if (generation + 1) % 2 == 0 or generation + 1 == args.generations:
            hold_score, hold_errors = evaluate(holdout, make_config(base, best_multipliers))
            checkpoint["holdout_score"] = hold_score
            checkpoint["holdout_gain"] = p.gain_report(holdout, hold_errors)
        history.append(checkpoint)
        optimizer.save(Path(args.out).with_name(Path(args.out).stem + "_cma.json"))
        payload = {
            "decision": None, "base": base, "tune_keys": list(TUNE_KEYS),
            "train": train.summary(), "history": history,
            "best_config": make_config(base, best_multipliers),
            "elapsed_seconds": time.time() - started,
        }
        p.write_json(args.out, payload)
        print(f"gen {generation + 1}/{args.generations}: best fitness {best_fitness:.4f}, sigma {optimizer.sigma:.4f}, elapsed {time.time() - started:.0f}s", flush=True)

    val = validation_report(base, best_multipliers, args.val_seeds, args.val_frames)
    payload = {
        "decision": decision(val),
        "base": base,
        "tune_keys": list(TUNE_KEYS),
        "bounds": [0.5, 2.0],
        "sigma0": 0.15,
        "population": 12,
        "mu": 6,
        "train": train.summary(),
        "tune_frames": args.tune_frames,
        "val_frames": args.val_frames,
        "history": history,
        "best_config": make_config(base, best_multipliers),
        "validation": val,
        "elapsed_seconds": time.time() - started,
    }
    p.write_json(args.out, payload)
    print(json.dumps({
        "out": str(args.out),
        "decision": payload["decision"],
        "best_config": payload["best_config"],
        "validation": {
            seed: {
                "gain": data["gain"]["gain"],
                "ci": data["gain"]["gain_ci_95"],
            }
            for seed, data in val.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
