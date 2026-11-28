#!/usr/bin/env python3
"""Build a v20/v21 manifest by replacing selected v19 cases with gated winners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as pc  # noqa: E402

ROOT = TOOLS_DIR.parent
KSPIN_DEFAULTS = {
    "tau": 2.976,
    "xi": 1.0,
    "lambd": 1.0,
    "delta": 3.441,
    "dt": 0.47,
    "beta": 0.317,
    "joint_pairs": 1,
    "pair_gamma": 0.0,
    "pair_scale": 1.0,
    "g": 0.98,
    "clip": 2.8,
    "select_unclipped": 0,
    "energy_lambda": 1.0,
}
BSB_DEFAULTS = {
    "tau": 5.184,
    "xi": 0.6,
    "lambd": 9.375,
    "delta": 3.0,
    "dt": 0.4,
    "beta": 0.16,
    "parity_scale": 0.6375,
    "energy_lambda": 9.375,
}


def normalise_trajectory(base_traj: dict, raw: dict, mode: str) -> dict:
    """Fill missing trajectory keys without discarding learned arrays."""
    if not isinstance(raw, dict):
        raise ValueError(f"trajectory must be an object, got {type(raw).__name__}")
    if mode == "kspin":
        result = dict(raw)
        defaults = KSPIN_DEFAULTS
        inherited = None
    else:
        result = {key: value for key, value in raw.items() if key in pc.CONFIG_KEYS}
        defaults = BSB_DEFAULTS
        inherited = pc.CONFIG_KEYS
    for key, value in (base_traj or {}).items():
        if key not in result and (inherited is None or key in inherited) and value is not None:
            result[key] = value
    for key, value in defaults.items():
        result.setdefault(key, value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "artifacts" / "phase16_case_manifest_v19.json",
    )
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v20")
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
            raw_case = dict(raw_config)
            raw_trajectories = raw_case.pop("trajectories")
        else:
            raw_case = {}
            raw_trajectories = [raw_config]
        if not isinstance(raw_trajectories, list) or not 1 <= len(raw_trajectories) <= 6:
            raise ValueError(f"case {index} needs a list of 1..6 trajectories")
        mode = str(raw_case.get("mode", base_case.get("mode", "kspin")))
        if mode not in ("bsb", "dsb", "kspin"):
            raise ValueError(f"case {index}: unsupported mode {mode!r}")
        base_trajectory = base_case.get("trajectories", [{}])[0]
        trajectories = [
            normalise_trajectory(base_trajectory, item, mode) for item in raw_trajectories
        ]
        case = dict(base_case)
        case.update({
            "mode": mode,
            "combine": str(raw_case.get("combine", base_case.get("combine", "energy"))),
            "dt": float(raw_case.get("dt", trajectories[0].get("dt", base_case.get("dt", 0.47)))),
            "steps": int(raw_case.get("steps", base_case.get("steps", 50))),
            "trajectories": trajectories,
        })
        note = str(raw_case.get("note", f"{args.version} selected decoder for case {index}"))
        case["note"] = note
        for key in ("tau_cls", "xi_cls", "beta_cls", "beta_values", "g_values"):
            if key in raw_case:
                case[key] = raw_case[key]
        manifest["cases"][index - 1] = case
        changed.append(index)

    manifest["base"] = str(args.base.resolve())
    manifest["changed_cases"] = sorted(changed)
    manifest["version"] = args.version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}: changed cases {sorted(changed)}")


if __name__ == "__main__":
    main()
