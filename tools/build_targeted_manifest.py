#!/usr/bin/env python3
"""Replace one case in an accepted manifest with a tuned Phase 7 summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOBAL_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"
DEFAULT_GLOBAL_CONFIG = ROOT / "artifacts" / "phase6_multi_guess_config.json"


def parse_config(name: str) -> list[float] | None:
    parts = name.split("_")
    if not parts or parts[0] not in ("fixed",) and not parts[0].startswith("grid"):
        return None
    values = parts[1:]
    if len(values) != 4:
        return None
    return [float(value) for value in values]


def replacement_case(
    summary_path: Path,
    global_params: Path,
    global_config: dict,
) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    case = data["case"]
    names = data["optimized_portfolio"]["names"]
    energy_lambdas = data["optimized_portfolio"].get("energy_lambdas", {})
    best_fixed = data["best_fixed_config"]
    learned_lambda = float(best_fixed[2])
    use_global = "global_learned" in names
    use_learned = any(
        name in ("global_learned", "per_case_learned") for name in names
    )

    params_path = summary_path.with_name(summary_path.name.replace("summary", "params"))
    if not params_path.exists() or use_global or not use_learned:
        params_rel = str(global_params.relative_to(ROOT))
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
            fixed.append(config + [float(energy_lambdas.get(name, config[2]))])
    if not fixed:
        fixed = [list(map(float, item)) for item in global_config["fixed_candidates"]]
    fixed = fixed[: 5 if use_learned else 6]

    return {
        "n": int(case["n"]),
        "k": round(int(case["n"]) * float(case["rate"])),
        "snr": float(case["snr"]),
        "use_learned": use_learned,
        "params": params_rel,
        "learned_lambda": learned_lambda,
        "fixed_candidates": fixed,
        "summary": str(summary_path.relative_to(ROOT)),
        "portfolio_ber": float(data["optimized_portfolio"]["ber"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-params", type=Path, default=DEFAULT_GLOBAL_PARAMS)
    parser.add_argument("--global-config", type=Path, default=DEFAULT_GLOBAL_CONFIG)
    args = parser.parse_args()

    args.base = args.base.resolve()
    args.summary = args.summary.resolve()
    args.output = args.output.resolve()
    args.global_params = args.global_params.resolve()
    args.global_config = args.global_config.resolve()

    manifest = json.loads(args.base.read_text(encoding="utf-8"))
    global_config = json.loads(args.global_config.read_text(encoding="utf-8"))
    new_case = replacement_case(args.summary, args.global_params, global_config)

    replaced = None
    for index, case in enumerate(manifest["cases"]):
        if (
            int(case["n"]) == new_case["n"]
            and int(case["k"]) == new_case["k"]
            and float(case["snr"]) == new_case["snr"]
        ):
            manifest["cases"][index] = new_case
            replaced = index
            break
    if replaced is None:
        raise ValueError(
            f"Base manifest has no case for n={new_case['n']} k={new_case['k']} "
            f"snr={new_case['snr']}"
        )

    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Replaced case {replaced} with {args.summary} "
        f"(BER {new_case['portfolio_ber']:.9f})"
    )


if __name__ == "__main__":
    main()
