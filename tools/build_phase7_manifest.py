#!/usr/bin/env python3
"""Build the C++ v2 manifest from the best Phase 7 summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "artifacts"
DEFAULT_OUTPUT = DEFAULT_OUTDIR / "phase7_case_manifest.json"
DEFAULT_GLOBAL_PARAMS = DEFAULT_OUTDIR / "phase4_unfolded_dsb_params.json"
DEFAULT_GLOBAL_CONFIG = DEFAULT_OUTDIR / "phase6_multi_guess_config.json"
DEFAULT_PROBE_PARAMS = DEFAULT_OUTDIR / "probe_unfolded_params.json"


def parse_config(name: str) -> list[float] | None:
    parts = name.split("_")
    if parts[0] == "fixed":
        values = parts[1:]
    elif parts[0].startswith("grid"):
        values = parts[1:]
    else:
        return None
    if len(values) != 4:
        return None
    return [float(value) for value in values]


def best_summaries(outdir: Path) -> dict[tuple[int, float, float], tuple[float, Path, dict]]:
    grouped: dict[tuple[int, float, float], tuple[float, Path, dict]] = {}
    for path in outdir.glob("phase7_N*_summary*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        case = data["case"]
        key = (int(case["n"]), float(case["rate"]), float(case["snr"]))
        ber = float(data["optimized_portfolio"]["ber"])
        if key not in grouped or ber < grouped[key][0]:
            grouped[key] = (ber, path, data)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--global-params", type=Path, default=DEFAULT_GLOBAL_PARAMS)
    parser.add_argument("--global-config", type=Path, default=DEFAULT_GLOBAL_CONFIG)
    parser.add_argument("--exclude-rates", default="")
    parser.add_argument("--probe-rates", default="")
    parser.add_argument("--probe-params", type=Path, default=DEFAULT_PROBE_PARAMS)
    args = parser.parse_args()
    parse_rates = lambda text: {
        float(value) for value in text.split(",") if value.strip()
    }
    exclude_rates = parse_rates(args.exclude_rates)
    probe_rates = parse_rates(args.probe_rates)
    args.outdir = args.outdir.resolve()
    args.output = args.output.resolve()
    args.global_params = args.global_params.resolve()
    args.global_config = args.global_config.resolve()
    args.probe_params = args.probe_params.resolve()

    global_config = json.loads(args.global_config.read_text(encoding="utf-8"))
    fallback = {
        "n": 0,
        "k": 0,
        "snr": 0.0,
        "use_learned": True,
        "params": str(args.global_params.relative_to(ROOT)),
        "learned_lambda": float(global_config["learned_energy_lambda"]),
        "fixed_candidates": [list(map(float, item)) for item in global_config["fixed_candidates"]],
    }

    cases = []
    for (n, rate, snr), (_, summary_path, data) in sorted(best_summaries(args.outdir).items()):
        if rate in exclude_rates:
            continue
        k = round(n * rate)
        if rate in probe_rates:
            cases.append({
                "n": n,
                "k": k,
                "snr": snr,
                "use_learned": True,
                "params": str(args.probe_params.relative_to(ROOT)),
                "learned_lambda": 0.0,
                "fixed_candidates": [],
                "summary": "artifacts/submission_probe.cpp",
                "portfolio_ber": None,
            })
            continue
        names = data["optimized_portfolio"]["names"]
        energy_lambdas = data["optimized_portfolio"].get("energy_lambdas", {})
        best_fixed = data["best_fixed_config"]
        learned_lambda = float(best_fixed[2])
        use_global = "global_learned" in names
        use_learned = any(
            name in ("global_learned", "per_case_learned") for name in names
        )

        params_name = summary_path.name.replace("summary", "params")
        params_path = args.outdir / params_name
        if not params_path.exists() or use_global or not use_learned:
            params_rel = str(args.global_params.relative_to(ROOT))
        else:
            params_rel = str(params_path.relative_to(ROOT))
        if use_global:
            learned_lambda = float(global_config["learned_energy_lambda"])
        learned_name = "global_learned" if use_global else "per_case_learned"
        if use_learned and learned_name in energy_lambdas:
            learned_lambda = float(energy_lambdas[learned_name])

        fixed = []
        for name in names:
            config = parse_config(name)
            if config is not None:
                energy_lambda = energy_lambdas.get(name, config[2])
                fixed.append(config + [float(energy_lambda)])
        if not fixed:
            fixed = [list(map(float, item)) for item in global_config["fixed_candidates"]]
        # Respect the six-initial-guess limit: six fixed candidates when the
        # learned trajectory is not selected, otherwise at most five fixed.
        fixed = fixed[: 5 if use_learned else 6]

        cases.append({
            "n": n,
            "k": k,
            "snr": snr,
            "use_learned": use_learned,
            "params": params_rel,
            "learned_lambda": learned_lambda,
            "fixed_candidates": fixed,
            "summary": str(summary_path.relative_to(ROOT)),
            "portfolio_ber": float(data["optimized_portfolio"]["ber"]),
        })

    manifest = {"fallback": fallback, "cases": cases}
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(cases)} cases")


if __name__ == "__main__":
    main()
