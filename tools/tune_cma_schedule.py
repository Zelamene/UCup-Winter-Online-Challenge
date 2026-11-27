#!/usr/bin/env python3
"""Phase 15 workstream B: CMA-ES learned per-step QAIA schedules.

The 33 parameters are log-multipliers in [0.5, 2]: five knots for each of tau,
xi, lambd, delta, dt and beta (30), plus scalar pair_gamma, parity_scale and
energy_lambda. Only tau knot 0 affects the C++/Python initialization, matching
the exporter; the other four tau knots are recorded but inert.

The local CMA-ES uses sigma0=0.15, lambda=12, mu=6, a held-out watch every two
generations, JSON checkpoints and an SVG progress plot. Validation gains are
paired against cached v17 on identical frames with 1,000-resample bootstraps.
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

KNOT_STEPS = (0, 12, 25, 37, 49)
KNOTS = len(KNOT_STEPS)
GROUPS = ("tau", "xi", "lambd", "delta", "dt", "beta")
SCALARS = ("pair_gamma", "parity_scale", "energy_lambda")
DIM = len(GROUPS) * KNOTS + len(SCALARS)
LOG_LOW, LOG_HIGH = math.log(0.5), math.log(2.0)
DEFAULT_OUT = p.PHASE15_ARTIFACTS / "workstream_b_case10_schedule.json"


def parameter_names() -> list[str]:
    return [f"{group}[{step}]" for group in GROUPS for step in KNOT_STEPS] + list(SCALARS)


def config_from_log_params(base: dict, log_params: np.ndarray) -> dict:
    if len(log_params) != DIM:
        raise ValueError(f"expected {DIM} parameters, got {len(log_params)}")
    mult = np.exp(np.asarray(log_params, dtype=np.float64))
    config = dict(base)
    for group_index, group in enumerate(GROUPS):
        block = mult[group_index * KNOTS:(group_index + 1) * KNOTS]
        values = np.interp(
            np.arange(p.STEPS), np.asarray(KNOT_STEPS, dtype=float),
            float(base.get(group, 0.0)) * block,
        )
        if group == "tau":
            config[group] = float(values[0])
        else:
            config[group] = values
    for scalar_index, key in enumerate(SCALARS):
        config[key] = float(base.get(key, 0.0)) * float(mult[len(GROUPS) * KNOTS + scalar_index])
    if config["energy_lambda"] < 0.0:
        config["energy_lambda"] = 0.0
    return config


def evaluate(workspace: p.CaseWorkspace, config: dict) -> tuple[float, np.ndarray]:
    hard = p.decode_hard_schedules(workspace, config)
    errors = p.per_frame_errors(hard, workspace.case)
    return p.score_from_errors(errors, workspace.k), errors


def gate_for_rate(rate: float) -> float:
    return 2000.0 if abs(rate - 0.5) < 1e-4 else 1500.0


def validation_report(base: dict, best: np.ndarray, case_index: int, seeds: list[int], frames: int) -> dict:
    config = config_from_log_params(base, best)
    _, n, k, rate, snr = p.CASE_SPECS[case_index - 1]
    report: dict[str, dict] = {}
    for seed in seeds:
        workspace = p.CaseWorkspace(n, k, seed, 500, snr=snr, frame_slice=frames)
        score, errors = evaluate(workspace, config)
        report[str(seed)] = {
            "workspace": workspace.summary(),
            "score": score,
            "gain": p.gain_report(workspace, errors),
        }
    required = gate_for_rate(rate)
    passed = all(
        payload["gain"]["gain"] >= required and payload["gain"]["gain_ci_95"][0] > 0.0
        for payload in report.values()
    )
    return {
        "config": config,
        "rate": rate,
        "required_gain": required,
        "decision": "promote_B_case" if passed else "reject_B_case",
        "seeds": report,
    }


def write_plot(path: Path, history: list[dict]) -> None:
    """Write a dependency-free SVG of fitness and held-out gain by generation."""
    width, height = 720, 320
    fitness = [item["best_fitness"] for item in history]
    holds = [item.get("holdout_gain", {}).get("gain") for item in history]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if fitness:
        lo, hi = min(fitness), max(fitness)
        span = hi - lo or 1.0
        for index, value in enumerate(fitness):
            x = 60 + index * (width - 100) / max(1, len(fitness) - 1)
            y = 40 + (value - lo) / span * 100
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2563eb"/>')
        lines.append('<text x="10" y="30" font-size="12" fill="#2563eb">best fitness</text>')
    if any(value is not None for value in holds):
        valid = [value for value in holds if value is not None]
        lo, hi = min(valid), max(valid)
        span = hi - lo or 1.0
        points = []
        for index, value in enumerate(holds):
            if value is None:
                continue
            x = 60 + index * (width - 100) / max(1, len(holds) - 1)
            y = 200 + (value - lo) / span * 80
            points.append(f"{x:.1f},{y:.1f}")
        points_text = " ".join(points)
        lines.append(f'<polyline points="{points_text}" fill="none" stroke="#dc2626" stroke-width="2"/>')
        lines.append('<text x="10" y="190" font-size="12" fill="#dc2626">held-out gain</text>')
    lines.append(f'<text x="10" y="305" font-size="11" fill="#444">generations: {len(history)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--tune-frames", type=int, default=100)
    parser.add_argument("--val-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[2024, 31337])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    case_index = int(args.case)
    _, n, k, rate, snr = p.CASE_SPECS[case_index - 1]
    out = args.out or p.PHASE15_ARTIFACTS / f"workstream_b_case{case_index}_schedule.json"
    cache_frames = 300 if args.seed == 911 else 500
    if args.tune_frames > cache_frames:
        raise ValueError(f"tune-frames must be <= {cache_frames} for seed {args.seed}")
    base = p.load_v17_config(n, k)
    train = p.CaseWorkspace(n, k, args.seed, cache_frames, snr=snr, frame_slice=args.tune_frames)
    holdout = p.CaseWorkspace(n, k, args.val_seeds[0], 500, snr=snr, frame_slice=args.tune_frames)

    optimizer = CMAESLite(
        DIM, mean=np.zeros(DIM), sigma=0.15, population_size=12, mu=6,
        seed=5150 + case_index, bounds=(LOG_LOW, LOG_HIGH),
    )
    history: list[dict] = []
    best = np.zeros(DIM)
    best_fitness = math.inf
    started = time.time()
    for generation in range(args.generations):
        candidates = optimizer.ask()
        fitness: list[float] = []
        for candidate in candidates:
            score, _ = evaluate(train, config_from_log_params(base, candidate))
            fitness.append(-score)
        optimizer.tell(candidates, fitness)
        if optimizer.best_fitness < best_fitness:
            best_fitness = float(optimizer.best_fitness)
            best = np.asarray(optimizer.best_solution).copy()
        checkpoint = {
            "generation": generation + 1,
            "evaluations": optimizer.evaluations,
            "best_fitness": best_fitness,
            "best_log_params": best.tolist(),
            "best_config": config_from_log_params(base, best),
            "sigma": optimizer.sigma,
        }
        if (generation + 1) % 2 == 0 or generation + 1 == args.generations:
            score, errors = evaluate(holdout, config_from_log_params(base, best))
            checkpoint["holdout_score"] = score
            checkpoint["holdout_gain"] = p.gain_report(holdout, errors)
        history.append(checkpoint)
        optimizer.save(Path(out).with_name(Path(out).stem + "_cma.json"))
        write_plot(Path(out).with_name(Path(out).stem + "_progress.svg"), history)
        payload = {
            "case": case_index, "n": n, "k": k, "rate": rate,
            "decision": None, "base": base, "parameter_names": parameter_names(),
            "bounds_multiplier": [0.5, 2.0], "sigma0": 0.15,
            "population": 12, "mu": 6,
            "train": train.summary(), "tune_frames": args.tune_frames,
            "history": history, "best_config": config_from_log_params(base, best),
            "elapsed_seconds": time.time() - started,
        }
        p.write_json(out, payload)
        print(f"case {case_index} gen {generation + 1}/{args.generations}: best fitness {best_fitness:.4f}, sigma {optimizer.sigma:.4f}, elapsed {time.time() - started:.0f}s", flush=True)

    val = validation_report(base, best, case_index, args.val_seeds, args.val_frames)
    payload.update({
        "decision": val["decision"],
        "required_gain": val["required_gain"],
        "val_frames": args.val_frames,
        "validation": val["seeds"],
        "best_config": val["config"],
        "history": history,
        "elapsed_seconds": time.time() - started,
    })
    p.write_json(out, payload)
    print(json.dumps({
        "out": str(out),
        "decision": val["decision"],
        "validation": {
            seed: {
                "gain": data["gain"]["gain"],
                "ci": data["gain"]["gain_ci_95"],
            }
            for seed, data in val["seeds"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
