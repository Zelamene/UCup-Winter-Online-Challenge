#!/usr/bin/env python3
"""Phase 14 A2 three-way Python / batched / C++ joint-pair equivalence check."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from bsb_vectorized import BatchedBSB  # noqa: E402
from decode_v3_python import decode_frame  # noqa: E402
from qam_pairs import pair_tables as build_pair_tables  # noqa: E402
from real_bench import build_case, write_input  # noqa: E402
from binary_io import read_hard, run_binary  # noqa: E402

ROOT = TOOLS_DIR.parent


def select_cases(manifest: dict, indices: list[int]) -> list[tuple[int, dict]]:
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest has no cases")
    selected = []
    for index in indices:
        if not 1 <= index <= len(cases):
            raise ValueError(f"case index {index} out of range 1..{len(cases)}")
        selected.append((index, cases[index - 1]))
    return selected


def check_case(
    binary: Path,
    case: dict,
    frames: int,
    seed: int,
    noise_scale: float,
    workdir: Path,
) -> dict:
    n = int(case["n"])
    k = int(case["k"])
    snr = float(case["snr"])
    built = build_case(
        n, k, frames, snr, seed, noise_scale=noise_scale
    )
    traj = case["trajectories"][0]
    tables = build_pair_tables(built["llr"], snr)
    batched = BatchedBSB(built["h"]).decode_batch(
        built["llr"],
        tau=float(traj["tau"]),
        xi=float(traj["xi"]),
        lambd=float(traj["lambd"]),
        delta=float(traj["delta"]),
        dt=float(traj.get("dt", case.get("dt", 0.2))),
        steps=int(case.get("steps", 50)),
        beta=float(traj.get("beta", 0.0)),
        parity_scale=float(traj.get("parity_scale", 1.0)),
        joint_pairs=int(traj.get("joint_pairs", 0)),
        pair_gamma=float(traj.get("pair_gamma", 0.0)),
        pair_scale=float(traj.get("pair_scale", 1.0)),
        pair_tables=tables if int(traj.get("joint_pairs", 0)) else None,
    )
    python_hard = np.asarray(
        [decode_frame(built["llr"][i], built["h"], case) for i in range(frames)],
        dtype=np.int8,
    )
    input_path = workdir / "equiv.in"
    output_path = workdir / "equiv.out"
    write_input(built, input_path)
    run_binary(binary, input_path, output_path)
    cpp_hard = read_hard(output_path, frames, n)

    python_batched = bool(np.array_equal(python_hard, batched))
    batched_cpp = bool(np.array_equal(batched, cpp_hard))
    return {
        "n": n,
        "k": k,
        "snr": snr,
        "joint_pairs": int(traj.get("joint_pairs", 0)),
        "frames": frames,
        "python_equals_batched": python_batched,
        "batched_equals_cpp": batched_cpp,
        "all_equal": python_batched and batched_cpp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--noise-scale", type=float, default=1.001120)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "phase14" / "equivalence.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    binary = args.binary.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="equiv_", dir=args.output.parent) as tmp_name:
        workdir = Path(tmp_name)
        for index, case in select_cases(manifest, args.cases):
            result = check_case(
                binary, case, args.frames, args.seed, args.noise_scale, workdir
            )
            result["case_index"] = index
            results.append(result)
            print(
                f"case {index:>2d} N={result['n']} joint={result['joint_pairs']}: "
                f"python==batched {result['python_equals_batched']}, "
                f"batched==cpp {result['batched_equals_cpp']}",
                flush=True,
            )

    payload = {
        "manifest": str(args.manifest),
        "binary": str(binary),
        "seed": args.seed,
        "noise_scale": args.noise_scale,
        "results": results,
        "all_passed": all(item["all_equal"] for item in results),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
