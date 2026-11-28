#!/usr/bin/env python3
"""Coordinate search using several frame-matched validation seeds at once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as pc  # noqa: E402
from baseline_cache import cache_path  # noqa: E402

CASE_DIMS = {1: (648, 324), 4: (1296, 648), 7: (1944, 972), 10: (3888, 1944)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True, choices=sorted(CASE_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 31337])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=5150)
    parser.add_argument("--objective", choices=("min", "mean"), default="min")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n, k = CASE_DIMS[args.case]
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(grid, dict) or not grid:
        raise ValueError("base config and grid must be JSON objects")
    config.setdefault("mode", "kspin")
    config.setdefault("combine", "energy")
    config.setdefault("joint_pairs", 1)

    workspaces = []
    for seed in args.seeds:
        cache = cache_path(pc.V18_CACHE, n, k, seed)
        cached_frames = int(np.load(cache).size)
        if args.frames > cached_frames:
            raise ValueError(f"seed {seed} has only {cached_frames} cached frames")
        workspaces.append(
            pc.CaseWorkspace(
                n=n, k=k, seed=seed, frames=cached_frames,
                snr=pc.configured_snr(k / n), cache_dir=pc.V18_CACHE,
                frame_slice=args.frames,
            )
        )

    def evaluate(candidate: dict) -> tuple[float, list[dict]]:
        rows = []
        for workspace in workspaces:
            started = time.perf_counter()
            hard = pc.decode_hard(workspace, candidate, batch_size=args.batch_size)
            elapsed = time.perf_counter() - started
            errors = pc.per_frame_errors(hard, workspace.case)
            report = pc.gain_report(workspace, errors, args.bootstrap_seed)
            rows.append({
                "seed": workspace.seed, "elapsed_s": round(elapsed, 3), **report,
            })
        gains = [row["gain"] for row in rows]
        objective = min(gains) if args.objective == "min" else float(np.mean(gains))
        return objective, rows

    best_objective, rows = evaluate(config)
    print(
        f"case {args.case} start objective={best_objective:+.1f} " +
        " ".join(f"s{row['seed']}:{row['gain']:+.0f}" for row in rows),
        flush=True,
    )
    history = [{"round": 0, "param": "start", "value": None, "objective": best_objective, "config": dict(config), "rows": rows}]

    for round_index in range(1, args.rounds + 1):
        for param, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid[{param!r}] must be a non-empty list")
            trials = []
            for value in values:
                if value == config.get(param):
                    continue
                candidate = dict(config)
                candidate[param] = value
                objective, rows = evaluate(candidate)
                trials.append((objective, value, rows))
                print(
                    f"  r{round_index} {param}={value}: objective={objective:+.1f} " +
                    " ".join(f"s{row['seed']}:{row['gain']:+.0f}" for row in rows),
                    flush=True,
                )
            if trials:
                objective, value, rows = max(trials, key=lambda row: row[0])
                if objective > best_objective:
                    best_objective = objective
                    config[param] = value
            history.append({
                "round": round_index, "param": param, "value": config.get(param),
                "objective": best_objective, "config": dict(config), "rows": rows,
            })
            print(
                f"  best after {param}: objective={best_objective:+.1f} config={config}",
                flush=True,
            )

    final_objective, final_rows = evaluate(config)
    payload = {
        "case": args.case,
        "seeds": args.seeds,
        "frames": args.frames,
        "objective": args.objective,
        "best_config": config,
        "best_objective": final_objective,
        "best_rows": final_rows,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pc.write_json(args.output, payload)
    print(f"Saved {args.output}: objective={final_objective:+.1f} config={config}")


if __name__ == "__main__":
    main()
