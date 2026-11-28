#!/usr/bin/env python3
"""Build a v19 manifest by replacing selected v18b cases with K-spin cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent


def trajectory_defaults(case: dict, config: dict) -> dict:
    """Normalise one K-spin trajectory for the v3 exporter and validator."""
    trajectory = dict(config)
    trajectory.setdefault("tau", 2.976)
    trajectory.setdefault("xi", 1.0)
    trajectory.setdefault("delta", 3.441)
    trajectory.setdefault("dt", float(case.get("dt", 0.47)))
    trajectory.setdefault("beta", 0.317)
    trajectory.setdefault("joint_pairs", 1)
    trajectory.setdefault("pair_gamma", 0.0)
    trajectory.setdefault("pair_scale", 1.0)
    trajectory.setdefault("g", 0.98)
    trajectory.setdefault("clip", 2.8)
    trajectory.setdefault("select_unclipped", 0)
    trajectory.setdefault("lambd", 1.0)
    trajectory.setdefault("energy_lambda", 1.0)
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "submissions" / "v18b_greedy_portfolio" / "manifest.json",
    )
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.base.read_text(encoding="utf-8"))
    replacements = json.loads(args.configs.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("cases"), list):
        raise ValueError("base manifest has no cases list")
    if not isinstance(replacements, dict):
        raise ValueError("configs must map case numbers to trajectory configurations")

    changed: list[int] = []
    for raw_index, raw_config in replacements.items():
        index = int(raw_index)
        if not 1 <= index <= len(manifest["cases"]):
            raise ValueError(f"case index {index} out of range 1..{len(manifest['cases'])}")
        base_case = manifest["cases"][index - 1]
        if isinstance(raw_config, dict) and "trajectories" in raw_config:
            raw_trajectories = raw_config["trajectories"]
        else:
            raw_trajectories = [raw_config]
        if not 1 <= len(raw_trajectories) <= 6:
            raise ValueError(f"case {index} needs 1..6 trajectories")
        trajectories = [trajectory_defaults(base_case, item) for item in raw_trajectories]
        case = dict(base_case)
        case.update({
            "mode": "kspin",
            "combine": "energy",
            "dt": trajectories[0]["dt"],
            "steps": 50,
            "trajectories": trajectories,
        })
        note = f"v19 K-spin structural decoder for case {index}"
        if isinstance(raw_config, dict) and raw_config.get("note"):
            note = str(raw_config["note"])
        case["note"] = note
        manifest["cases"][index - 1] = case
        changed.append(index)

    manifest["base"] = str(args.base.resolve())
    manifest["changed_cases"] = sorted(changed)
    manifest["version"] = "v19_kspin"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}: changed cases {sorted(changed)}")


if __name__ == "__main__":
    main()
