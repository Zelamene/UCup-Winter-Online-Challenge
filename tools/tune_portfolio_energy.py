#!/usr/bin/env python3
"""Tune energy-selection lambdas for an existing dSB portfolio.

Trajectories are unchanged; only the coefficients used to choose among them
are optimized. This keeps the QAIA/dSB decoder and six-guess limit intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_answers, parse_instance  # noqa: E402
from phase5_generalization import empirical_margins  # noqa: E402
from phase7_per_case import (  # noqa: E402
    DSBConfig,
    ber_of,
    decode_config,
    decode_learned,
    load_params,
    make_cases,
    select_energies,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"
DEFAULT_GLOBAL_PARAMS = DEFAULT_OUTDIR / "phase4_unfolded_dsb_params.json"


def parse_config(name: str) -> tuple[float, float, float, float] | None:
    parts = name.split("_")
    if not parts or parts[0] not in ("fixed",) and not parts[0].startswith("grid"):
        return None
    values = parts[1:]
    if len(values) != 4:
        return None
    return tuple(float(value) for value in values)


def build_candidates(
    summary: dict,
    val_cases: list[dict],
    params: dict,
    global_params: dict,
    energy_lambdas: dict[str, float],
) -> list[dict]:
    by_name = summary["optimized_portfolio"].get("energy_lambdas", {})
    candidates = []
    for name in summary["optimized_portfolio"]["names"]:
        config = parse_config(name)
        if name == "per_case_learned":
            hards = decode_learned(val_cases, params)
            lambd = float(summary["best_fixed_config"][2])
        elif name == "global_learned":
            hards = decode_learned(val_cases, global_params)
            lambd = 6.0
        elif config is not None:
            tau, xi, lam, delta = config
            hards = decode_config(
                val_cases,
                DSBConfig(tau=tau, xi=xi, lambd=lam, delta=delta, dt=0.2, steps=50,
                          gradient_sign=1),
            )
            lambd = lam
        else:
            raise ValueError(f"Unknown portfolio candidate: {name}")
        candidates.append({
            "name": name,
            "lambd": lambd,
            "energy_lambda": float(by_name.get(name, lambd)),
            "hards": hards,
        })

    for candidate in candidates:
        if candidate["name"] in energy_lambdas:
            candidate["energy_lambda"] = energy_lambdas[candidate["name"]]
    return candidates


def portfolio_ber(val_cases: list[dict], candidates: list[dict]) -> float:
    selected = select_energies(val_cases, candidates, list(range(len(candidates))))
    return ber_of(val_cases, selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--global-params", type=Path, default=DEFAULT_GLOBAL_PARAMS)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--lambdas",
        default="1,2,3,4,5,6,8,10,12,16,24",
    )
    args = parser.parse_args()

    args.summary = args.summary.resolve()
    if args.output is None:
        args.output = args.summary.with_name(
            args.summary.name.replace("summary", "summary_energy")
        )
    args.output = args.output.resolve()
    args.global_params = args.global_params.resolve()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    params_path = args.summary.with_name(args.summary.name.replace("summary", "params"))
    params = load_params(params_path)
    global_params = load_params(args.global_params)

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"),
        str(args.answer),
        instance.n,
        instance.b,
    )
    margins = empirical_margins(instance, answers)
    data = summary["data"]
    val_cases = make_cases(
        int(summary["case"]["n"]),
        float(summary["case"]["rate"]),
        float(summary["case"]["snr"]),
        int(data["val_codes"]),
        int(data["val_samples_per_code"]),
        margins,
        instance.snr,
        int(data["seed"]) + 1000,
    )

    candidates = build_candidates(summary, val_cases, params, global_params, {})
    best_ber = portfolio_ber(val_cases, candidates)
    print(f"Starting BER: {best_ber:.9f}")

    values = tuple(float(value) for value in args.lambdas.split(","))
    for round_index in range(args.rounds):
        improved = False
        for index, candidate in enumerate(candidates):
            original = candidate["energy_lambda"]
            current_best = original
            current_ber = best_ber
            for value in values:
                if value == original:
                    continue
                candidate["energy_lambda"] = value
                ber = portfolio_ber(val_cases, candidates)
                if ber < current_ber:
                    current_ber = ber
                    current_best = value
            candidate["energy_lambda"] = current_best
            if current_ber < best_ber:
                best_ber = current_ber
                improved = True
        print(f"Round {round_index + 1}: BER {best_ber:.9f}")
        if not improved:
            break

    energy_lambdas = {
        candidate["name"]: float(candidate["energy_lambda"])
        for candidate in candidates
    }
    print("Energy lambdas:", json.dumps(energy_lambdas, sort_keys=True))

    result = dict(summary)
    portfolio = dict(summary["optimized_portfolio"])
    portfolio["ber"] = best_ber
    portfolio["energy_lambdas"] = energy_lambdas
    portfolio["energy_tuning"] = {
        "rounds": args.rounds,
        "lambdas": list(values),
    }
    result["optimized_portfolio"] = portfolio
    result["gate"]["portfolio_beats_accepted_portfolio"] = (
        best_ber < float(summary["accepted_portfolio_ber"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    output_params = args.output.with_name(args.output.name.replace("summary", "params"))
    output_params.write_text(params_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {output_params}")


if __name__ == "__main__":
    main()
