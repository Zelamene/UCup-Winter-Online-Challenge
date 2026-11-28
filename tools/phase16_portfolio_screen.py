#!/usr/bin/env python3
"""Screen K-spin portfolios against the accepted v18b per-frame cache."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as pc  # noqa: E402
from phase16_case_gate import decode_case  # noqa: E402
from baseline_cache import cache_path  # noqa: E402

CASE_DIMS = {1: (648, 324), 4: (1296, 648), 7: (1944, 972), 10: (3888, 1944)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=5150)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.configs.read_text(encoding="utf-8"))
    results = []
    for index in args.cases:
        if index not in CASE_DIMS:
            raise ValueError(f"case {index} is not a rate-1/2 K-spin candidate")
        n, k = CASE_DIMS[index]
        raw = payload.get(str(index), payload.get(index))
        if raw is None:
            raise ValueError(f"no configs for case {index}")
        configs = raw.get("configs", []) if isinstance(raw, dict) else raw
        if isinstance(configs, dict):
            configs = [configs]
        if not configs or len(configs) > 8:
            raise ValueError(f"case {index} needs 1..8 configs")
        cache = cache_path(pc.V18_CACHE, n, k, args.seed)
        cached_frames = int(np.load(cache).size)
        workspace = pc.CaseWorkspace(
            n=n, k=k, seed=args.seed, frames=cached_frames,
            snr=pc.configured_snr(k / n), cache_dir=pc.V18_CACHE,
            frame_slice=args.frames,
        )
        case = {
            "n": n, "k": k, "snr": workspace.snr, "mode": "kspin",
            "combine": "energy", "dt": 0.47,
        }
        print(
            f"case {index} seed {args.seed} frames {workspace.frames}: "
            f"{len(configs)} trajectories",
            flush=True,
        )
        for size in range(1, min(args.max_size, len(configs)) + 1):
            for indices in itertools.combinations(range(len(configs)), size):
                sub_case = dict(case)
                sub_case["trajectories"] = [configs[value] for value in indices]
                started = time.perf_counter()
                hard = decode_case(workspace, sub_case, args.batch_size)
                elapsed = time.perf_counter() - started
                errors = pc.per_frame_errors(hard, workspace.case)
                report = pc.gain_report(workspace, errors, args.bootstrap_seed)
                names = [
                    str(configs[value].get("name", configs[value].get("source", value)))
                    for value in indices
                ]
                row = {
                    "case": index,
                    "seed": args.seed,
                    "frames": workspace.frames,
                    "indices": list(indices),
                    "names": names,
                    "configs": [configs[value] for value in indices],
                    "elapsed_s": round(elapsed, 3),
                    **report,
                }
                results.append(row)
                print(
                    f"  {names} gain={report['gain']:+9.1f} "
                    f"CI=[{report['gain_ci_95'][0]:+.1f},{report['gain_ci_95'][1]:+.1f}]",
                    flush=True,
                )

    output = {
        "baseline": "artifacts/baseline_cache_v18",
        "seed": args.seed,
        "frames": args.frames,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pc.write_json(args.output, output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
