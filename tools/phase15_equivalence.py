#!/usr/bin/env python3
"""Phase 15 three-way bit identity for energy-combined portfolios.

The Phase 14 equivalence tool compares only the first trajectory, so it cannot
validate a multi-trajectory v18 case. This tool compares, frame by frame:

1. scalar Python trajectories and scalar Python energy selection,
2. batched Python trajectories and batched Python energy selection,
3. the compiled C++ submission.

It also compares every trajectory before selection, which catches schedule and
LCG-state differences that an energy tie could otherwise hide.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402
from binary_io import read_hard, run_binary, write_input  # noqa: E402
from decode_v3_python import decode_frame, decode_trajectory  # noqa: E402

ROOT = TOOLS_DIR.parent


def trajectory_config(case: dict, traj: dict) -> dict:
    config = dict(traj)
    config.setdefault("mode", case.get("mode", "bsb"))
    config.setdefault("combine", case.get("combine", "energy"))
    config.setdefault("dt", case.get("dt", 0.2))
    config["n"] = int(case["n"])
    config["k"] = int(case["k"])
    config["snr"] = float(case["snr"])
    return config


def check_case(
    manifest_case: dict,
    binary: Path,
    frames: int,
    seed: int,
    workdir: Path,
) -> dict:
    n = int(manifest_case["n"])
    k = int(manifest_case["k"])
    snr = float(manifest_case["snr"])
    cache_frames = 300 if int(seed) == 911 else 500
    if not 1 <= frames <= cache_frames:
        raise ValueError(f"frames must be in 1..{cache_frames}, got {frames}")
    workspace = p.CaseWorkspace(
        n, k, seed, cache_frames, snr=snr, frame_slice=frames
    )
    built = workspace.case
    configs = [trajectory_config(manifest_case, traj) for traj in manifest_case["trajectories"]]
    steps = int(manifest_case.get("steps", 50))

    scalar_trajectories: list[np.ndarray] = []
    for traj in manifest_case["trajectories"]:
        scalar_trajectories.append(np.asarray([
            decode_trajectory(
                built["llr"][index], built["h"], traj,
                mode=str(manifest_case.get("mode", "bsb")),
                dt=float(traj.get("dt", manifest_case.get("dt", 0.2))),
                steps=steps, snr=snr,
            )[0]
            for index in range(frames)
        ], dtype=np.int8))
    scalar_selected = np.asarray([
        decode_frame(built["llr"][index], built["h"], manifest_case)
        for index in range(frames)
    ], dtype=np.int8)

    batched_trajectories = [p.decode_hard(workspace, config) for config in configs]
    portfolio = p.portfolio_from_hards(workspace, batched_trajectories, configs)
    selected_index = portfolio["selected_index"]
    batched_selected = np.stack(batched_trajectories)[selected_index, np.arange(frames)]

    input_path = workdir / "phase15_equiv.in"
    output_path = workdir / "phase15_equiv.out"
    frame_case = dict(built)
    frame_case["b"] = frames
    write_input(frame_case, input_path)
    run_binary(binary, input_path, output_path)
    cpp_hard = read_hard(output_path, frames, n)

    per_trajectory_equal = [
        bool(np.array_equal(scalar, batched))
        for scalar, batched in zip(scalar_trajectories, batched_trajectories)
    ]
    scalar_batched = bool(np.array_equal(scalar_selected, batched_selected))
    batched_cpp = bool(np.array_equal(batched_selected, cpp_hard))
    scalar_cpp = bool(np.array_equal(scalar_selected, cpp_hard))
    all_equal = all(per_trajectory_equal) and scalar_batched and batched_cpp and scalar_cpp
    return {
        "n": n,
        "k": k,
        "snr": snr,
        "frames": frames,
        "trajectory_count": len(configs),
        "selected_indices": selected_index.tolist(),
        "per_trajectory_scalar_equals_batched": per_trajectory_equal,
        "scalar_equals_batched_selected": scalar_batched,
        "batched_equals_cpp_selected": batched_cpp,
        "scalar_equals_cpp_selected": scalar_cpp,
        "all_equal": all_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "phase15" / "equivalence.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="phase15_equiv_", dir=args.output.parent) as tmp_name:
        workdir = Path(tmp_name)
        for case_index in args.cases:
            if not 1 <= case_index <= len(cases):
                raise ValueError(f"case index {case_index} out of range 1..{len(cases)}")
            result = check_case(cases[case_index - 1], args.binary.resolve(), args.frames, args.seed, workdir)
            results.append(result)
            print(
                f"case {case_index:>2d} n={result['n']} "
                f"trajectories={result['trajectory_count']}: "
                f"scalar==batched {result['scalar_equals_batched_selected']}, "
                f"batched==cpp {result['batched_equals_cpp_selected']}, "
                f"scalar==cpp {result['scalar_equals_cpp_selected']}",
                flush=True,
            )
    payload = {
        "manifest": str(args.manifest.resolve()),
        "binary": str(args.binary.resolve()),
        "seed": args.seed,
        "frames": args.frames,
        "results": results,
        "all_passed": all(item["all_equal"] for item in results),
    }
    p.write_json(args.output, payload)
    print(f"Saved {args.output}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
