#!/usr/bin/env python3
"""Create a synthetic judge-format input for parity checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_answers, parse_instance  # noqa: E402
from phase5_generalization import empirical_margins, make_case  # noqa: E402

ROOT = TOOLS_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=648)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "synthetic_r12.in")
    args = parser.parse_args()

    instance = parse_instance(
        (ROOT / "Problem statements" / "publish.in").read_text(encoding="utf-8"), "public"
    )
    answers = parse_answers(
        (ROOT / "Problem statements" / "publish.ans").read_text(encoding="utf-8"),
        "public",
        instance.n,
        instance.b,
    )
    margins = empirical_margins(instance, answers)
    case = make_case(
        args.n, args.rate, args.samples, args.snr, margins, instance.snr,
        np.random.default_rng(args.seed),
    )
    lines = [f"{case['n']} {case['k']} {args.samples} {args.snr:g}"]
    lines.extend(" ".join(map(str, row.tolist())) for row in case["h"])
    lines.extend(" ".join(f"{value:.17g}" for value in row) for row in case["llr"])
    lines.extend(" ".join(map(str, row.tolist())) for row in case["g"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
