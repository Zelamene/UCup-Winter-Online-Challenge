#!/usr/bin/env python3
"""Collect Phase 7 sweep and validation results into one summary."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    grouped: dict[tuple[int, float, float], list[dict]] = defaultdict(list)
    for path in ARTIFACTS.glob("phase7_N*_summary_sweep.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        case = data["case"]
        grouped[(int(case["n"]), float(case["rate"]), float(case["snr"]))].append(data)

    rows = []
    for key in sorted(grouped):
        best = min(grouped[key], key=lambda item: item["optimized_portfolio"]["ber"])
        accepted = float(best["accepted_portfolio_ber"])
        optimized = float(best["optimized_portfolio"]["ber"])
        rows.append({
            "n": key[0],
            "rate": key[1],
            "snr": key[2],
            "accepted_portfolio_ber": accepted,
            "optimized_portfolio_ber": optimized,
            "improvement_ratio": accepted / optimized,
            "best_fixed_config": best["best_fixed_config"],
            "portfolio": best["optimized_portfolio"]["names"],
        })

    validation = json.loads((ARTIFACTS / "phase7_v2_validation.json").read_text())
    summary = {
        "framework": "per_case_QAIA_dSB_six_guess_energy_selection",
        "submission": "submission_v2.cpp",
        "manifest": "artifacts/phase7_case_manifest.json",
        "per_case": rows,
        "public_validation": validation,
        "all_passed": validation["all_passed"] and all(
            row["improvement_ratio"] > 1.0 for row in rows
        ),
    }
    output = ARTIFACTS / "phase7_final_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
