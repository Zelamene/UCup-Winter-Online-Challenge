#!/usr/bin/env python3
"""NaN/Inf guard for Phase 17 candidates at learned parameter extremes.

The Phase 17 unfolding bounds keep every log-parameter within +/-log(2) of its
v19 value, i.e. each learned multiplier lies in [0.5, 2]. This tool decodes
every changed trajectory at both endpoints with finite-state assertions enabled
and verifies that the resulting hard decisions remain valid bits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as pc  # noqa: E402

ROOT = TOOLS_DIR.parent
SCALED_KEYS = (
    "tau", "xi", "lambd", "delta", "dt", "beta", "parity_scale",
    "pair_gamma", "pair_scale", "g", "clip", "energy_lambda",
)


def scale_config(config: dict, factor: float) -> dict:
    """Scale every numeric decoder parameter by factor, preserving arrays."""
    result = dict(config)
    for key in SCALED_KEYS:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            result[key] = np.asarray(value, dtype=float) * factor
        else:
            result[key] = float(value) * factor
    for key in ("tau_cls", "xi_cls", "beta_cls"):
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            result[key] = np.asarray(value, dtype=float) * factor
        else:
            result[key] = float(value) * factor
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results: list[dict] = []
    for index in args.changed_cases:
        case = manifest["cases"][index - 1]
        base_config = dict(case)
        base_config.setdefault("mode", "kspin")
        base_config.setdefault("combine", "energy")
        workspace = pc.CaseWorkspace(
            n=int(case["n"]), k=int(case["k"]), seed=911, frames=300,
            snr=float(case["snr"]), cache_dir=pc.V19_CACHE, frame_slice=args.frames,
        )
        for trajectory_index, trajectory in enumerate(case["trajectories"]):
            config = dict(base_config)
            config.update(trajectory)
            for label, factor in (("low", 0.5), ("high", 2.0)):
                scaled = scale_config(config, factor)
                hard = pc.decode_hard(
                    workspace, scaled, debug_finite=True, batch_size=args.batch_size
                )
                valid = bool(np.isin(hard, (0, 1)).all())
                results.append({
                    "case": index,
                    "trajectory": trajectory_index,
                    "extreme": label,
                    "factor": factor,
                    "frames": int(hard.shape[0]),
                    "bits_valid": valid,
                    "pass": valid,
                })
                print(
                    f"case {index:>2d} traj {trajectory_index} {label:>4s}: "
                    f"bits_valid={valid}", flush=True,
                )

    payload = {
        "manifest": str(args.manifest.resolve()),
        "frames": args.frames,
        "factors": [0.5, 2.0],
        "results": results,
        "all_passed": all(item["pass"] for item in results),
    }
    pc.write_json(args.output, payload)
    print(f"Saved {args.output}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
