#!/usr/bin/env python3
"""Extract the accepted single-trajectory probe parameters into JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "artifacts" / "submission_probe.cpp"
OUTPUT = ROOT / "artifacts" / "probe_unfolded_params.json"


def vector(name: str, source: str) -> list[float]:
    match = re.search(rf"static const double {name}\[STEPS\] = \{{(.*?)\}};", source, re.S)
    if match is None:
        raise ValueError(f"Missing vector {name}")
    return [float(value) for value in match.group(1).split(",")]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tau_match = re.search(r"static constexpr double TAU = ([^;]+);", source)
    if tau_match is None:
        raise ValueError("Missing TAU")
    values = {
        "tau": [float(tau_match.group(1))],
        "dt": vector("DT", source),
        "xi": vector("XI", source),
        "lam": vector("LAM", source),
        "delta": vector("DELTA", source),
        "xi_mod": vector("XI_MOD", source),
        "lam_mod": vector("LAM_MOD", source),
        "delta_mod": vector("DELTA_MOD", source),
    }
    for key, array in values.items():
        expected = 1 if key == "tau" else 50
        if len(array) != expected:
            raise ValueError(f"{key}: expected {expected} values, got {len(array)}")
    payload = {"steps": 50, "values": values}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
