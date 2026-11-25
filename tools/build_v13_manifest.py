#!/usr/bin/env python3
"""Build v13 manifest: v12 non-R=1/2 cases (v3 conversion) + fixed bSB for the four R=1/2 cases.

The R=1/2 cases use a single bSB trajectory tuned by Stage A.
All other cases inherit from the v3 conversion of the accepted v12 manifest, so the
v12 case-5 salvage is preserved (the v9 approximation would silently revert it).
"""

from __future__ import annotations

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

OUTPUT = ROOT / "artifacts" / "phase9_case_manifest_v13.json"


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
    }


def main() -> None:
    base = json.loads(V3_V12_SAFE.read_text(encoding="utf-8"))

    r12_params: dict[tuple[int, int], dict] = {}
    for (n, k), path in STAGE_A.items():
        sa = json.loads(path.read_text(encoding="utf-8"))
        r12_params[(n, k)] = sa["best_config"]
        print(f"N={n} K={k}: best config tau={sa['best_config']['tau']:.4f} "
              f"xi={sa['best_config']['xi']:.4f} lam={sa['best_config']['lambd']:.4f} "
              f"delta={sa['best_config']['delta']:.4f} dt={sa['best_config']['dt']:.4f}  "
              f"val_score={sa['val_score']:.1f} gain={sa['gain']:+.1f}")

    new_cases = []
    for case in base["cases"]:
        n, k = case["n"], case["k"]
        if (n, k) in r12_params:
            # Replace with single bSB trajectory
            traj = bsb_trajectory(r12_params[(n, k)])
            new_case = {
                "n": n,
                "k": k,
                "snr": case["snr"],
                "mode": "bsb",
                "combine": "energy",
                "trajectories": [traj],
                "dt": traj["dt"],
                "steps": 50,
            }
            print(f"  -> Case N={n} K={k}: replaced with bSB")
        else:
            new_case = case
        new_cases.append(new_case)

    manifest = {
        "fallback": base["fallback"],
        "cases": new_cases,
    }

    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"Cases: {len(new_cases)} total, 4 bSB R=1/2, {len(new_cases) - 4} inherited dSB")


if __name__ == "__main__":
    main()
