#!/usr/bin/env python3
"""Build the v13 manifests: v12 non-R=1/2 cases (v3 conversion) + fixed bSB for R=1/2.

By default the four R=1/2 cases use a single bSB trajectory tuned by Stage A
(`phase9_case_manifest_v13.json`). With `--stage-b` they use the piecewise schedules
from the Stage B search instead (`phase9_case_manifest_v13c.json`).
All other cases inherit from the v3 conversion of the accepted v12 manifest, so the
v12 case-5 salvage is preserved (the v9 approximation would silently revert it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Source manifests
V3_V12_SAFE = ROOT / "artifacts" / "phase9_v3_from_v12_safe.json"

# Stage A bSB results for R=1/2
STAGE_A = {
    (648,  324):  ROOT / "artifacts" / "stage_a_N648_R0.5.json",
    (1296, 648):  ROOT / "artifacts" / "stage_a_N1296_R0.5.json",
    (1944, 972):  ROOT / "artifacts" / "stage_a_N1944_R0.5.json",
    (3888, 1944): ROOT / "artifacts" / "stage_a_N3888_R0.5.json",
}

# Stage B piecewise-schedule results for the same cases
STAGE_B = {
    (648,  324):  ROOT / "artifacts" / "stage_b_N648_R0.5.json",
    (1296, 648):  ROOT / "artifacts" / "stage_b_N1296_R0.5.json",
    (1944, 972):  ROOT / "artifacts" / "stage_b_N1944_R0.5.json",
    (3888, 1944): ROOT / "artifacts" / "stage_b_N3888_R0.5.json",
}

OUTPUT_A = ROOT / "artifacts" / "phase9_case_manifest_v13.json"
OUTPUT_B = ROOT / "artifacts" / "phase9_case_manifest_v13c.json"


def bsb_trajectory(cfg: dict) -> dict:
    """Convert a Stage A best_config into a v3 trajectory dict."""
    lambd = float(cfg["lambd"])
    return {
        "tau":          float(cfg["tau"]),
        "xi":           float(cfg["xi"]),
        "lambd":        lambd,
        "delta":        float(cfg["delta"]),
        "dt":           float(cfg["dt"]),
        "energy_lambda": lambd,  # for energy selection; identical to lambd for single traj
        "beta":         float(cfg.get("beta", 0.0)),
        "parity_scale": float(cfg.get("parity_scale", 1.0)),
    }


def bsb_scheduled_trajectory(result: dict) -> dict:
    """Convert a Stage B result into a v3 trajectory dict with per-step schedules."""
    schedules = result["schedules"]
    steps = len(schedules["dt"])
    for name in ("xi", "lambd", "delta", "dt"):
        if len(schedules[name]) != steps:
            raise ValueError(f"schedule {name} has {len(schedules[name])} steps, expected {steps}")
    lambd = [float(v) for v in schedules["lambd"]]
    base_cfg = result.get("stage_a_config", {})
    traj = {
        "tau":           float(base_cfg.get("tau", result["stage_a_config"]["tau"])),
        "xi":            [float(v) for v in schedules["xi"]],
        "lambd":         lambd,
        "delta":         [float(v) for v in schedules["delta"]],
        "dt":            [float(v) for v in schedules["dt"]],
        # Energy selection needs one scale per trajectory; the schedule average keeps it
        # consistent with the lambda the dynamics actually ran at.
        "energy_lambda": sum(lambd) / len(lambd),
        "beta":          float(base_cfg.get("beta", 0.0)),
        "parity_scale":  float(base_cfg.get("parity_scale", 1.0)),
    }
    if "beta" in schedules:
        beta_values = [float(v) for v in schedules["beta"]]
        if len(beta_values) != len(lambd):
            raise ValueError(f"beta schedule has {len(beta_values)} steps, expected {len(lambd)}")
        traj["beta"] = beta_values
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b", action="store_true",
                        help="use Stage B piecewise schedules instead of flat Stage A constants")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (OUTPUT_B if args.stage_b else OUTPUT_A)

    base = json.loads(V3_V12_SAFE.read_text(encoding="utf-8"))

    r12_params: dict[tuple[int, int], dict] = {}
    for (n, k), path in (STAGE_B if args.stage_b else STAGE_A).items():
        result = json.loads(path.read_text(encoding="utf-8"))
        r12_params[(n, k)] = result if args.stage_b else result["best_config"]
        if args.stage_b:
            sched = result["schedules"]
            gain = min(v["gain"] for v in result["validation"].values())
            print(f"N={n} K={k}: scheduled tau={result['stage_a_config']['tau']:.4f} "
                  f"xi=[{min(sched['xi']):.4f},{max(sched['xi']):.4f}] "
                  f"lam=[{min(sched['lambd']):.4f},{max(sched['lambd']):.4f}] "
                  f"delta=[{min(sched['delta']):.4f},{max(sched['delta']):.4f}] "
                  f"dt=[{min(sched['dt']):.4f},{max(sched['dt']):.4f}]  min gain={gain:+.1f}")
        else:
            cfg = result["best_config"]
            print(f"N={n} K={k}: best config tau={cfg['tau']:.4f} "
                  f"xi={cfg['xi']:.4f} lam={cfg['lambd']:.4f} "
                  f"delta={cfg['delta']:.4f} dt={cfg['dt']:.4f}  "
                  f"val_score={result['val_score']:.1f} gain={result['gain']:+.1f}")

    new_cases = []
    for case in base["cases"]:
        n, k = case["n"], case["k"]
        if (n, k) in r12_params:
            # Replace with a single bSB trajectory: flat Stage A, or scheduled Stage B
            traj = (bsb_scheduled_trajectory(r12_params[(n, k)]) if args.stage_b
                    else bsb_trajectory(r12_params[(n, k)]))
            case_dt = traj["dt"]
            if isinstance(case_dt, list):
                case_dt = sum(case_dt) / len(case_dt)
            new_case = {
                "n": n,
                "k": k,
                "snr": case["snr"],
                "mode": "bsb",
                "combine": "energy",
                "trajectories": [traj],
                "dt": case_dt,
                "steps": 50,
            }
            kind = "scheduled" if args.stage_b else "flat"
            print(f"  -> Case N={n} K={k}: replaced with bSB ({kind})")
        else:
            new_case = case
        new_cases.append(new_case)

    manifest = {
        "fallback": base["fallback"],
        "cases": new_cases,
    }

    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output} ({output.stat().st_size} bytes)")
    print(f"Cases: {len(new_cases)} total, 4 bSB R=1/2, {len(new_cases) - 4} inherited dSB")


if __name__ == "__main__":
    main()
