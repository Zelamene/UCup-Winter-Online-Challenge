#!/usr/bin/env python3
"""Coordinate-descent tuning for one K-spin case against v18b."""

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
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=5150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n, k = CASE_DIMS[args.case]
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(grid, dict) or not grid:
        raise ValueError("base config and grid must be JSON objects")
    config.setdefault("mode", "kspin")
    config.setdefault("joint_pairs", 1)

    cache = cache_path(pc.V18_CACHE, n, k, args.seed)
    if not cache.exists():
        raise FileNotFoundError(f"missing v18 baseline cache {cache}")
    cached_frames = int(np.load(cache).size)
    workspace = pc.CaseWorkspace(
        n=n,
        k=k,
        seed=args.seed,
        frames=cached_frames,
        snr=pc.configured_snr(k / n),
        cache_dir=pc.V18_CACHE,
        frame_slice=args.frames,
    )

    history = []

    def evaluate(candidate: dict) -> tuple[float, np.ndarray, float]:
        started = time.perf_counter()
        hard = pc.decode_hard(workspace, candidate, batch_size=args.batch_size)
        elapsed = time.perf_counter() - started
        errors = pc.per_frame_errors(hard, workspace.case)
        report = pc.gain_report(workspace, errors, args.bootstrap_seed)
        return report["gain"], errors, elapsed

    best_gain, best_errors, elapsed = evaluate(config)
    print(
        f"case {args.case} start gain={best_gain:+.1f} "
        f"errors={int(best_errors.sum())} {elapsed:.1f}s",
        flush=True,
    )
    history.append({"round": 0, "param": "start", "value": None, "gain": best_gain, "config": dict(config)})

    for round_index in range(1, args.rounds + 1):
        for param, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid[{param!r}] must be a non-empty list")
            current = config.get(param)
            trials = []
            for value in values:
                if value == current:
                    continue
                candidate = dict(config)
                candidate[param] = value
                gain, errors, elapsed = evaluate(candidate)
                trials.append((gain, value, errors, elapsed))
                print(
                    f"  r{round_index} {param}={value}: gain={gain:+.1f} "
                    f"errors={int(errors.sum())} {elapsed:.1f}s",
                    flush=True,
                )
            if trials:
                gain, value, errors, _elapsed = max(trials, key=lambda row: row[0])
                if gain > best_gain:
                    best_gain, best_errors = gain, errors
                    config[param] = value
            history.append({
                "round": round_index,
                "param": param,
                "value": config.get(param),
                "gain": best_gain,
                "config": dict(config),
            })
            print(
                f"  best after {param}: gain={best_gain:+.1f} config={config}",
                flush=True,
            )

    payload = {
        "case": args.case,
        "seed": args.seed,
        "frames": args.frames,
        "batch_size": args.batch_size,
        "best_config": config,
        "best_gain": best_gain,
        "best_errors": int(best_errors.sum()),
        "history": history,
    }
    pc.write_json(args.output, payload)
    print(f"Saved {args.output}: gain={best_gain:+.1f} config={config}")


if __name__ == "__main__":
    main()
