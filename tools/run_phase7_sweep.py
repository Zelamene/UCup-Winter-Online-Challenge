#!/usr/bin/env python3
"""Run the Phase 7 per-case search across selected operating points."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "phase7_per_case.py"
N_VALUES = (648, 1296, 1944, 3888)
RATE_SNR = {0.5: 6.0, 2.0 / 3.0: 9.0, 0.75: 10.5}
LABELS = {0.5: "1_2", 2.0 / 3.0: "2_3", 0.75: "3_4"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", default="0.5,0.6666666666666666,0.75")
    parser.add_argument("--n-values", default="648,1296,1944,3888")
    parser.add_argument("--tag", default="sweep")
    parser.add_argument("--val-codes", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=48)
    parser.add_argument("--local-rounds", type=int, default=2)
    parser.add_argument("--top-grid", type=int, default=8)
    parser.add_argument("--no-train", action="store_true", default=True)
    args = parser.parse_args()

    rates = [float(value) for value in args.rates.split(",")]
    n_values = [int(value) for value in args.n_values.split(",")]

    for rate in rates:
        if rate not in RATE_SNR:
            raise ValueError(f"Unsupported rate {rate}")
        for n in n_values:
            label = f"N{n}_R{LABELS[rate]}_SNR{RATE_SNR[rate]:g}".replace(".", "p")
            command = [
                sys.executable,
                str(SCRIPT),
                "--n", str(n),
                "--rate", repr(rate),
                "--snr", str(RATE_SNR[rate]),
                "--val-codes", str(args.val_codes),
                "--val-samples", str(args.val_samples),
                "--local-rounds", str(args.local_rounds),
                "--top-grid", str(args.top_grid),
                "--tag", args.tag,
            ]
            if args.no_train:
                command.append("--no-train")
            print(f"\n=== {label} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
