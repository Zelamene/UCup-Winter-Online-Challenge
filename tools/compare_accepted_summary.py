#!/usr/bin/env python3
"""Compare an accepted manifest portfolio with a tuned summary on fresh synthetic cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_answers, parse_instance  # noqa: E402
from phase5_generalization import empirical_margins  # noqa: E402
from phase7_per_case import ber_of, load_params, make_cases, select_energies  # noqa: E402
from refine_accepted_case import build_accepted_candidates, select_manifest_case  # noqa: E402
from tune_portfolio_energy import build_candidates  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase7_case_manifest_v9.json"
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_GLOBAL_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--global-params", type=Path, default=DEFAULT_GLOBAL_PARAMS)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--val-codes", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    args.manifest = args.manifest.resolve()
    args.summary = args.summary.resolve()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer),
        instance.n, instance.b,
    )
    margins = empirical_margins(instance, answers)
    cases = make_cases(
        args.n, args.rate, args.snr, args.val_codes, args.val_samples,
        margins, instance.snr, args.seed + 1000,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = select_manifest_case(manifest, args.n, args.rate, args.snr)
    accepted, _ = build_accepted_candidates(case, args.manifest, cases)
    accepted_ber = ber_of(cases, select_energies(cases, accepted, list(range(len(accepted)))))

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    params_path = args.summary.with_name(
        args.summary.name.replace("summary", "params")
    )
    params = load_params(params_path)
    global_params = load_params(args.global_params.resolve())
    tuned = build_candidates(summary, cases, params, global_params, {})
    tuned_ber = ber_of(cases, select_energies(cases, tuned, list(range(len(tuned)))))

    print(f"seed={args.seed} accepted={accepted_ber:.9f} tuned={tuned_ber:.9f} "
          f"ratio={accepted_ber / tuned_ber:.4f}")
    print(f"tuned names: {summary['optimized_portfolio']['names']}")


if __name__ == "__main__":
    main()
