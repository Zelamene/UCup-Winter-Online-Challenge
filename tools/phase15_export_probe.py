#!/usr/bin/env python3
"""Probe that a 33-parameter schedule config is exportable and bit-identical at zero multipliers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402
import tune_cma_schedule as schedules  # noqa: E402

ROOT = TOOLS_DIR.parent


def main() -> None:
    manifest_path = p.V17_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = p.load_v17_config(1296, 648, manifest_path)
    config = schedules.config_from_log_params(base, np.zeros(schedules.DIM))
    case = next(case for case in manifest["cases"] if case["n"] == 1296 and case["k"] == 648)
    traj = case["trajectories"][0]
    for key in ("tau", "xi", "lambd", "delta", "dt", "beta"):
        value = config[key]
        traj[key] = float(value) if isinstance(value, float) else [float(item) for item in value]
    traj["energy_lambda"] = float(config["energy_lambda"])
    out = p.PHASE15_ARTIFACTS / "export_probe_manifest.json"
    p.write_json(out, manifest)
    print(out)


if __name__ == "__main__":
    main()
