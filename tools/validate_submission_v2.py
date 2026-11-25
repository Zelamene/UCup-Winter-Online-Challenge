#!/usr/bin/env python3
"""Validate the Phase 7 per-case submission manifest and public output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_answers, parse_instance  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase7_case_manifest.json"
DEFAULT_SUBMISSION = ROOT / "submission_v2.cpp"
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_ACCEPTED = ROOT / "artifacts" / "submission_accepted_public.out"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--public-output", type=Path, default=ROOT / "artifacts" / "phase7_v2_public.out")
    parser.add_argument("--accepted-output", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts" / "phase7_v2_validation.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = args.submission.read_text(encoding="utf-8")
    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )

    cases = manifest["cases"]
    guess_counts = [
        (1 if case.get("use_learned", True) else 0) + len(case["fixed_candidates"])
        for case in cases
    ]
    fallback = manifest["fallback"]
    guess_counts.append(
        (1 if fallback.get("use_learned", True) else 0) + len(fallback["fixed_candidates"])
    )
    size = args.submission.stat().st_size

    lines = args.public_output.read_text(encoding="utf-8").splitlines()
    if len(lines) != instance.b:
        raise ValueError(f"Expected {instance.b} output lines, got {len(lines)}")
    hard = np.asarray([[int(value) for value in line.split()] for line in lines], dtype=np.int8)
    if hard.shape != (instance.b, instance.n):
        raise ValueError(f"Unexpected output shape {hard.shape}")
    ber = float(np.mean(hard[:, : instance.k] != answers[:, : instance.k]))
    syndromes = 0
    for row in range(instance.m):
        products = 1.0 - 2.0 * hard[:, np.flatnonzero(instance.h[row])]
        syndromes += int(np.count_nonzero(np.prod(products, axis=1) < 0))

    accepted_ber = None
    accepted_bit_mismatches = None
    if args.accepted_output.exists():
        accepted_lines = args.accepted_output.read_text(encoding="utf-8").splitlines()
        accepted = np.asarray(
            [[int(value) for value in line.split()] for line in accepted_lines], dtype=np.int8
        )
        accepted_ber = float(np.mean(accepted[:, : instance.k] != answers[:, : instance.k]))
        accepted_bit_mismatches = int(np.count_nonzero(hard != accepted))

    gate = {
        "source_under_512k": size < 512 * 1024,
        "at_most_six_guesses": max(guess_counts) <= 6,
        "at_most_twelve_cases_dispatched": len(cases) <= 12,
        "public_ber_not_worse": accepted_ber is None or ber <= accepted_ber + 1e-12,
        "no_unreplaced_tokens": "__" not in source,
        "all_output_bits_binary": bool(np.all((hard == 0) | (hard == 1))),
    }

    summary = {
        "source_bytes": size,
        "case_count": len(cases),
        "max_guess_count": max(guess_counts),
        "public_ber": ber,
        "accepted_public_ber": accepted_ber,
        "accepted_bit_mismatches": accepted_bit_mismatches,
        "public_syndrome_ones": syndromes,
        "gate": gate,
        "all_passed": all(gate.values()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit("Phase 7 validation failed")


if __name__ == "__main__":
    main()
