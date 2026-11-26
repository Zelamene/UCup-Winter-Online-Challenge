#!/usr/bin/env python3
"""Evaluate decoder binaries on the labeled publish_20261125 case set."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import read_hard, run_binary  # noqa: E402


def parse_header(path: Path) -> tuple[int, int, int, float]:
    with path.open("r", encoding="utf-8") as handle:
        parts = handle.readline().split()
    return int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])


def score_from_errors(errors: int, bits: int) -> float:
    ber = max(errors / bits, 1.0 / bits)
    return -100_000.0 * math.log10(ber)


def evaluate(binary: Path, case: Path, n: int, k: int, frames: int, workdir: Path) -> dict:
    out_path = workdir / "out.txt"
    seconds = run_binary(binary, case, out_path)
    hard = read_hard(out_path, frames, n)
    truth = read_hard(case.with_suffix(".a"), frames, n)
    errors = int(np.count_nonzero(hard[:, :k] != truth[:, :k]))
    return {
        "errors": errors,
        "bits": frames * k,
        "ber": errors / (frames * k),
        "score": score_from_errors(errors, frames * k),
        "seconds": round(seconds, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--binaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory(prefix="publish_eval_") as tmp_name:
        workdir = Path(tmp_name)
        for case in sorted(args.directory.glob("*.in")):
            n, k, frames, snr = parse_header(case)
            entry = {"case": case.stem, "n": n, "k": k, "frames": frames, "snr": snr}
            for binary in args.binaries:
                entry[binary.stem] = evaluate(binary, case, n, k, frames, workdir)
            rows.append(entry)
            scores = " ".join(
                f"{b.stem}={entry[b.stem]['score']:.0f}" for b in args.binaries
            )
            print(
                f"{case.stem} N={n} R={k/n:.3f} snr={snr:g}: {scores}",
                flush=True,
            )

    payload = {"directory": str(args.directory), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for binary in args.binaries:
        total = sum(entry[binary.stem]["score"] for entry in rows)
        print(f"TOTAL {binary.stem}: {total:,.0f}", flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
