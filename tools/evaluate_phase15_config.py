#!/usr/bin/env python3
"""Evaluate Phase 15 candidate configurations against cached v17 on identical frames.

The input JSON maps case indices to one to six trajectory configs, optionally
with ``energy_lambda`` and per-step schedules. Energy selection uses the exact
C++ energy so the reported gain matches a compiled submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402

DEFAULT_OUT = p.PHASE15_ARTIFACTS / "evaluation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    selection = json.loads(args.configs.read_text(encoding="utf-8"))
    configs_by_case = selection.get("cases", selection)
    cache_frames = 300 if args.seed == 911 else 500
    if args.frames > cache_frames:
        raise ValueError(f"frames must be <= {cache_frames} for seed {args.seed}")

    results: dict[str, dict] = {}
    for case_index in args.cases:
        _, n, k, rate, snr = p.CASE_SPECS[case_index - 1]
        key = str(case_index)
        if key not in configs_by_case:
            raise ValueError(f"case {case_index} missing from {args.configs}")
        configs = configs_by_case[key]
        if isinstance(configs, dict):
            configs = configs.get("trajectories", [])
        workspace = p.CaseWorkspace(n, k, args.seed, cache_frames, snr=snr, frame_slice=args.frames)
        outcome = p.portfolio_errors(workspace, configs)
        results[key] = {
            "n": n, "k": k, "rate": rate,
            "selection": p.gain_report(workspace, outcome["selected_errors"]),
            "oracle": p.gain_report(workspace, outcome["oracle_errors"]),
            "selection_accuracy": outcome["selection_accuracy"],
            "trajectories": len(configs),
        }
        print(f"case {case_index}: gain {results[key]['selection']['gain']:+.1f}, "
              f"CI {results[key]['selection']['gain_ci_95']}, oracle {results[key]['oracle']['gain']:+.1f}", flush=True)

    p.write_json(args.out, {
        "seed": args.seed, "frames": args.frames,
        "configs": str(args.configs.resolve()),
        "bootstrap_resamples": 1000,
        "cases": results,
    })
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
