#!/usr/bin/env python3
"""Phase 14 A4 guard check: swapped Gray pairs must fall back to per-bit v16."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import run_binary  # noqa: E402

ROOT = TOOLS_DIR.parent


def swap_pair_llrs(source: Path, target: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty input file")
    header = lines[0].split()
    if len(header) != 4:
        raise ValueError(f"bad header: {lines[0]!r}")
    n = int(header[0])
    if n % 4 != 0:
        raise ValueError(f"n={n} is not divisible by 4")
    h_rows = n - int(header[1])
    frame_start = 1 + h_rows
    frame_end = frame_start + int(header[2])
    for index in range(frame_start, frame_end):
        values = lines[index].split()
        if len(values) != n:
            raise ValueError(f"line {index + 1}: expected {n} LLRs, got {len(values)}")
        for pair in range(n // 2):
            first = 2 * pair
            values[first], values[first + 1] = values[first + 1], values[first]
        lines[index] = " ".join(values)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_diff(first: Path, second: Path) -> int:
    a = first.read_bytes().splitlines()
    b = second.read_bytes().splitlines()
    if len(a) != len(b):
        return abs(len(a) - len(b))
    return int(sum(left != right for left, right in zip(a, b)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--joint-binary", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "phase14" / "guard_check.json",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="guard_", dir=args.output.parent) as tmp_name:
        workdir = Path(tmp_name)
        swapped = workdir / "swapped.in"
        swap_pair_llrs(args.public_input, swapped)

        joint_orig = workdir / "joint_orig.out"
        ref_orig = workdir / "ref_orig.out"
        joint_swap = workdir / "joint_swap.out"
        ref_swap = workdir / "ref_swap.out"
        run_binary(args.joint_binary, args.public_input, joint_orig)
        run_binary(args.reference_binary, args.public_input, ref_orig)
        run_binary(args.joint_binary, swapped, joint_swap)
        run_binary(args.reference_binary, swapped, ref_swap)

        active_diff = output_diff(joint_orig, ref_orig)
        fallback_diff = output_diff(joint_swap, ref_swap)
        joint_swap_diff = output_diff(joint_orig, joint_swap)
        ref_swap_diff = output_diff(ref_orig, ref_swap)
        identical_bytes = joint_swap.read_bytes() == ref_swap.read_bytes()
        gate = {
            "joint_active_on_original": active_diff > 0,
            "fallback_matches_reference_on_swapped": identical_bytes and fallback_diff == 0,
            "original_diff_lines": active_diff,
            "swapped_diff_lines": fallback_diff,
            "joint_output_swap_diff_lines": joint_swap_diff,
            "reference_output_swap_diff_lines": ref_swap_diff,
        }

    payload = {
        "public_input": str(args.public_input),
        "joint_binary": str(args.joint_binary.resolve()),
        "reference_binary": str(args.reference_binary.resolve()),
        "gate": gate,
        "all_passed": all(
            value for key, value in gate.items() if isinstance(value, bool)
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
