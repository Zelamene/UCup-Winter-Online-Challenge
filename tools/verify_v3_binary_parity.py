#!/usr/bin/env python3
"""Verify that a v3 Python manifest and its compiled C++ export agree bit-for-bit."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import read_hard, run_binary, write_input  # noqa: E402
from decode_v3_python import decode_frame, select_case  # noqa: E402
from real_bench import build_case  # noqa: E402

ROOT = TOOLS_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--noise-scale", type=float, default=1.001120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = []
    for case_meta in manifest.get("cases", []):
        n = int(case_meta["n"])
        if n == 0:
            continue
        k = int(case_meta["k"])
        snr = float(case_meta["snr"])
        case = build_case(
            n, k, args.samples, snr, args.seed, noise_scale=args.noise_scale
        )
        with tempfile.TemporaryDirectory(prefix="parity_", dir=ROOT / "artifacts") as tmp_name:
            workdir = Path(tmp_name)
            input_path = workdir / "case.in"
            output_path = workdir / "case.out"
            write_input(case, input_path)
            run_binary(args.binary, input_path, output_path)
            cpp_hard = read_hard(output_path, case["b"], case["n"])

        active = select_case(manifest, n, k, snr)
        mismatches = 0
        for row in range(case["b"]):
            py_hard = np.asarray(
                decode_frame(case["llr"][row], case["h"], active), dtype=np.int8
            )
            mismatches += int(np.count_nonzero(py_hard != cpp_hard[row]))
        entry = {
            "n": n,
            "k": k,
            "snr": snr,
            "frames": case["b"],
            "mismatched_bits": mismatches,
            "match": mismatches == 0,
        }
        results.append(entry)
        print(
            f"N={n} K={k} snr={snr:g}: "
            f"{'MATCH' if mismatches == 0 else f'{mismatches} mismatched bits'}",
            flush=True,
        )

    payload = {
        "manifest": str(args.manifest),
        "binary": str(args.binary),
        "samples_per_case": args.samples,
        "seed": args.seed,
        "noise_scale": args.noise_scale,
        "cases": results,
        "all_match": all(item["match"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: all_match={payload['all_match']}", flush=True)


if __name__ == "__main__":
    main()
