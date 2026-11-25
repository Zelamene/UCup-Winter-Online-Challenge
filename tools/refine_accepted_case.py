#!/usr/bin/env python3
"""Refine one accepted per-case portfolio against its own synthetic validation set.

Unlike phase7_per_case.py, the baseline here is the currently accepted portfolio
from a manifest, so every reported improvement is relative to the submission we
are actually trying to beat.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    grid_search,
    load_params,
    local_search,
    make_cases,
    select_energies,
    syndrome_ones,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase7_case_manifest_v9.json"
DEFAULT_OUTDIR = ROOT / "artifacts"


def parse_list(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def select_manifest_case(manifest: dict, n: int, rate: float, snr: float) -> dict:
    for case in manifest["cases"]:
        if (
            int(case["n"]) == n
            and abs(int(case["k"]) / int(case["n"]) - rate) < 1e-9
            and abs(float(case["snr"]) - snr) < 1e-9
        ):
            return case
    raise ValueError(f"No manifest case for n={n}, rate={rate}, snr={snr}")


def config_name(prefix: str, config: tuple[float, float, float, float]) -> str:
    tau, xi, lambd, delta = config
    return f"{prefix}_{tau:g}_{xi:g}_{lambd:g}_{delta:g}"


def build_accepted_candidates(
    case: dict,
    manifest: Path,
    val_cases: list[dict],
) -> tuple[list[dict], Path]:
    params_path = Path(case["params"])
    if not params_path.is_absolute():
        params_path = (manifest.resolve().parent.parent / params_path).resolve()

    candidates: list[dict] = []
    if case.get("use_learned", False):
        params = load_params(params_path)
        hards = decode_learned(val_cases, params)
        lambd = float(case["learned_lambda"])
        name = "global_learned" if params_path.name == "phase4_unfolded_dsb_params.json" else "per_case_learned"
        candidates.append({
            "name": name,
            "lambd": lambd,
            "energy_lambda": lambd,
            "hards": hards,
        })

    for values in case["fixed_candidates"]:
        tau, xi, lambd, delta = [float(value) for value in values[:4]]
        energy_lambda = float(values[4]) if len(values) >= 5 else lambd
        config = DSBConfig(
            tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50,
            gradient_sign=1,
        )
        candidates.append({
            "name": config_name("fixed", (tau, xi, lambd, delta)),
            "lambd": lambd,
            "energy_lambda": energy_lambda,
            "hards": decode_config(val_cases, config),
        })
    return candidates, params_path


def add_grid_candidates(
    candidates: list[dict],
    grid_results: list[dict],
    top_grid: int,
) -> None:
    for rank, item in enumerate(grid_results[:top_grid]):
        tau, xi, lambd, delta = item["config"]
        candidates.append({
            "name": config_name(f"grid{rank}", item["config"]),
            "lambd": float(lambd),
            "energy_lambda": float(lambd),
            "hards": item["hards"],
        })


def seeded_portfolio(
    cases: list[dict],
    candidates: list[dict],
    accepted_indices: list[int],
    limit: int = 6,
    max_swaps: int = 6,
) -> dict:
    selected = list(accepted_indices)
    if len(selected) > limit:
        raise ValueError(f"Accepted portfolio already has {len(selected)} trajectories")

    while len(selected) < limit:
        best_index = None
        best_ber = ber_of(cases, select_energies(cases, candidates, selected))
        for index in range(len(candidates)):
            if index in selected:
                continue
            value = ber_of(cases, select_energies(cases, candidates, selected + [index]))
            if value < best_ber:
                best_ber = value
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)

    current = ber_of(cases, select_energies(cases, candidates, selected))
    history = [{"stage": "accepted", "ber": current, "subset": list(selected)}]
    for _ in range(max_swaps):
        best_value = current
        best_position = None
        best_index = None
        for position in range(len(selected)):
            for index in range(len(candidates)):
                if index in selected:
                    continue
                trial = list(selected)
                trial[position] = index
                value = ber_of(cases, select_energies(cases, candidates, trial))
                if value < best_value - 1e-12:
                    best_value = value
                    best_position = position
                    best_index = index
        if best_position is None:
            break
        selected[best_position] = best_index
        current = best_value
        history.append({
            "stage": "swap",
            "added": candidates[best_index]["name"],
            "ber": current,
            "subset": list(selected),
        })

    hards = select_energies(cases, candidates, selected)
    return {
        "indices": selected,
        "names": [candidates[index]["name"] for index in selected],
        "ber": ber_of(cases, hards),
        "syndrome_ones": syndrome_ones(cases, hards),
        "history": history,
    }


def tune_energy_coordinates(
    cases: list[dict],
    candidates: list[dict],
    subset: list[int],
    values: tuple[float, ...],
    rounds: int,
) -> dict:
    best_ber = ber_of(cases, select_energies(cases, candidates, subset))
    for _ in range(rounds):
        improved = False
        for index in subset:
            original = float(candidates[index]["energy_lambda"])
            best_value = original
            for value in values:
                if value == original:
                    continue
                candidates[index]["energy_lambda"] = value
                ber = ber_of(cases, select_energies(cases, candidates, subset))
                if ber < best_ber - 1e-12:
                    best_ber = ber
                    best_value = value
            candidates[index]["energy_lambda"] = best_value
            if best_value != original:
                improved = True
        if not improved:
            break
    return {
        "ber": best_ber,
        "energy_lambdas": {
            candidates[index]["name"]: float(candidates[index]["energy_lambda"])
            for index in subset
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--val-codes", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument("--local-rounds", type=int, default=2)
    parser.add_argument("--top-grid", type=int, default=8)
    parser.add_argument("--grid-tau", default="1.4,1.6,1.8,2,2.5")
    parser.add_argument("--grid-xi", default="0.45,0.5,0.55")
    parser.add_argument("--grid-lam", default="5,6,7,8")
    parser.add_argument("--grid-delta", default="2.25,2.5,2.75")
    parser.add_argument(
        "--energy-lambdas", default="1,2,3,4,5,6,7.5,8,9,10.5,12,16,24"
    )
    parser.add_argument("--energy-rounds", type=int, default=2)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    args.manifest = args.manifest.resolve()
    args.input = args.input.resolve()
    args.answer = args.answer.resolve()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer),
        instance.n, instance.b,
    )
    margins = empirical_margins(instance, answers)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = select_manifest_case(manifest, args.n, args.rate, args.snr)
    val_cases = make_cases(
        args.n, args.rate, args.snr, args.val_codes, args.val_samples,
        margins, instance.snr, args.seed + 1000,
    )
    print(
        f"Target case: N={args.n} R={args.rate:.6f} SNR={args.snr} "
        f"({args.val_codes} codes x {args.val_samples} samples)"
    )

    candidates, params_path = build_accepted_candidates(case, args.manifest, val_cases)
    accepted_indices = list(range(len(candidates)))
    accepted_ber = ber_of(
        val_cases, select_energies(val_cases, candidates, accepted_indices)
    )
    print(f"Accepted portfolio BER: {accepted_ber:.9f}")

    grid_results = grid_search(
        val_cases,
        taus=parse_list(args.grid_tau),
        xis=parse_list(args.grid_xi),
        lambdas=parse_list(args.grid_lam),
        deltas=parse_list(args.grid_delta),
    )
    add_grid_candidates(candidates, grid_results, args.top_grid)

    local_history = []
    if args.local_rounds > 0:
        print("Local parameter search")
        config, hards, local_history = local_search(
            val_cases, grid_results[0]["config"], args.local_rounds
        )
        candidates.append({
            "name": config_name("grid100", config),
            "lambd": float(config[2]),
            "energy_lambda": float(config[2]),
            "hards": hards,
        })

    print("Seeded portfolio search")
    portfolio = seeded_portfolio(val_cases, candidates, accepted_indices)
    print(f"  portfolio names: {portfolio['names']}")
    print(f"  portfolio BER: {portfolio['ber']:.9f}")

    tuned = tune_energy_coordinates(
        val_cases,
        candidates,
        portfolio["indices"],
        parse_list(args.energy_lambdas),
        args.energy_rounds,
    )
    final_hards = select_energies(val_cases, candidates, portfolio["indices"])
    portfolio["ber"] = ber_of(val_cases, final_hards)
    portfolio["syndrome_ones"] = syndrome_ones(val_cases, final_hards)
    portfolio["energy_lambdas"] = tuned["energy_lambdas"]
    portfolio["energy_tuning"] = {
        "rounds": args.energy_rounds,
        "lambdas": list(parse_list(args.energy_lambdas)),
    }
    print(f"After energy tuning: {portfolio['ber']:.9f}")

    label = (
        f"N{args.n}_R{str(round(args.rate, 6)).replace('.', 'p')}"
        f"_SNR{args.snr:g}".replace(".", "p")
    )
    summary_path = args.outdir / f"phase7_{label}_summary_{args.tag}.json"
    params_out = args.outdir / f"phase7_{label}_params_{args.tag}.json"
    shutil.copyfile(params_path, params_out)

    fixed_config = [float(value) for value in case["fixed_candidates"][0][:4]]
    summary = {
        "case": {"n": args.n, "rate": args.rate, "snr": args.snr},
        "data": {
            "manifest": str(args.manifest.relative_to(ROOT)),
            "train_codes": 0,
            "train_samples_per_code": 0,
            "val_codes": args.val_codes,
            "val_samples_per_code": args.val_samples,
            "seed": args.seed,
        },
        "best_fixed_config": fixed_config,
        "local_search_history": local_history,
        "accepted_portfolio_ber": accepted_ber,
        "optimized_portfolio": portfolio,
        "gate": {
            "portfolio_beats_accepted_portfolio": portfolio["ber"] < accepted_ber,
            "guess_count_at_most_six": len(portfolio["indices"]) <= 6,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"Accepted ratio: {accepted_ber / portfolio['ber']:.4f}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {params_out}")


if __name__ == "__main__":
    main()
