#!/usr/bin/env python3
"""Convert a v2 dSB manifest into an approximate v3 structural manifest.

The v2 learned trajectory uses LLR/H-dependent reliability modifiers that the
v3 manifest intentionally does not model.  With ``--include-learned`` the
learned trajectory is included using only its base per-step schedules; this is
an approximation, not a bit-for-bit v2 replica.  Fixed scalar candidates are
converted exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_TRAJECTORIES = 6
STEPS = 50


def _resolve(path: str, manifest_path: Path) -> Path:
    candidate = Path(path.replace("\\", "/")).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return manifest_path.resolve().parent.parent / candidate


def _load_params(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("steps", -1)) != STEPS:
        raise ValueError(f"{path}: steps must be {STEPS}")
    values = data.get("values")
    if not isinstance(values, dict):
        raise ValueError(f"{path}: missing values")
    return values


def _learned_trajectory(values: dict, energy_lambda: float) -> dict:
    tau = values.get("tau")
    if isinstance(tau, list):
        if len(tau) != 1:
            raise ValueError("learned tau must contain one value")
        tau = tau[0]
    return {
        "tau": float(tau),
        "xi": list(values["xi"]),
        "lambd": list(values["lam"]),
        "delta": list(values["delta"]),
        "dt": list(values["dt"]),
        "energy_lambda": float(energy_lambda),
    }


def _fixed_trajectory(row: list[float]) -> dict:
    if len(row) == 4:
        tau, xi, lambd, delta = row
        energy_lambda = lambd
    elif len(row) == 5:
        tau, xi, lambd, delta, energy_lambda = row
    else:
        raise ValueError(f"fixed candidate must have 4 or 5 values, got {len(row)}")
    return {
        "tau": float(tau),
        "xi": float(xi),
        "lambd": float(lambd),
        "delta": float(delta),
        "energy_lambda": float(energy_lambda),
    }


def _convert_case(raw: dict, manifest_path: Path, include_learned: bool) -> dict:
    trajectories: list[dict] = []
    if include_learned and raw.get("use_learned", True) and raw.get("params"):
        values = _load_params(_resolve(str(raw["params"]), manifest_path))
        trajectories.append(
            _learned_trajectory(values, float(raw.get("learned_lambda", 6.0)))
        )

    for row in raw.get("fixed_candidates", []):
        if len(trajectories) >= MAX_TRAJECTORIES:
            break
        trajectories.append(_fixed_trajectory(row))
    if not trajectories:
        trajectories.append(_fixed_trajectory([4.0, 0.5, 6.0, 2.5]))

    return {
        "n": int(raw.get("n", 0)),
        "k": int(raw.get("k", 0)),
        "snr": float(raw.get("snr", 0.0)),
        "mode": "dsb",
        "combine": "energy",
        "trajectories": trajectories,
        "dt": 0.2,
        "steps": STEPS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-learned",
        action="store_true",
        help="omit the approximate learned trajectory and keep fixed candidates only",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {
        "fallback": _convert_case(
            manifest["fallback"], args.manifest, not args.no_learned
        ),
        "cases": [
            _convert_case(case, args.manifest, not args.no_learned)
            for case in manifest.get("cases", [])
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        "trajectory counts: "
        + ", ".join(
            str(len(case["trajectories"]))
            for case in [result["fallback"]] + result["cases"]
        )
    )


if __name__ == "__main__":
    main()
