#!/usr/bin/env python3
"""Phase 15 workstream A1: screen diverse compliant QAIA trajectories.

Every trajectory is a scalar bSB/TAP configuration with joint Gray-pair reaction
terms where v17 already uses them. No per-edge state, no post-processing, and at
most six trajectories may be deployed. The screen reports oracle and
energy-selected gains against cached v17 on identical frames.

A1 gate (N=3888, seed 911):
  oracle >= +8k and realised >= +4k -> continue to A2
  oracle < +5k                      -> stop workstream A
  otherwise                         -> continue with best family, no A2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402

DEFAULT_OUT = p.PHASE15_ARTIFACTS / "a1_portfolio_screen.json"
AXES = ("tau_xi", "lambd_delta", "beta_parity", "pair_gamma", "dt")


def config_key(config: dict) -> str:
    return json.dumps(
        {key: float(config.get(key, 0.0)) for key in p.CONFIG_KEYS},
        sort_keys=True,
        separators=(",", ":"),
    )


def perturb(base: dict, **multipliers: float) -> dict:
    result = dict(base)
    for key, factor in multipliers.items():
        result[key] = float(base[key]) * float(factor)
    return result


def build_families(base: dict) -> dict[str, list[dict]]:
    """Deterministic perturbation families around the accepted v17 trajectory."""
    families: dict[str, list[dict]] = {}
    families["tau_xi"] = [
        perturb(base, tau=0.85, xi=0.9),
        perturb(base, tau=0.925, xi=1.05),
        perturb(base, tau=1.075, xi=0.95),
        perturb(base, tau=1.15, xi=1.1),
    ]
    families["lambd_delta"] = [
        perturb(base, lambd=0.85, delta=0.9),
        perturb(base, lambd=0.925, delta=1.05),
        perturb(base, lambd=1.075, delta=0.95),
        perturb(base, lambd=1.15, delta=1.1),
    ]
    families["beta_parity"] = [
        perturb(base, beta=0.5, parity_scale=0.8),
        perturb(base, beta=0.75, parity_scale=1.1),
        perturb(base, beta=1.25, parity_scale=0.9),
        perturb(base, beta=1.5, parity_scale=1.2),
    ]
    families["pair_gamma"] = [
        perturb(base, pair_gamma=0.5, pair_scale=0.85),
        perturb(base, pair_gamma=0.75, pair_scale=1.075),
        perturb(base, pair_gamma=1.25, pair_scale=0.925),
        perturb(base, pair_gamma=1.5, pair_scale=1.15),
    ]
    families["dt"] = [
        perturb(base, dt=0.75),
        perturb(base, dt=0.875),
        perturb(base, dt=1.125),
        perturb(base, dt=1.25),
    ]
    families["mixed6"] = [
        base,
        families["tau_xi"][1],
        families["lambd_delta"][2],
        families["beta_parity"][1],
        families["pair_gamma"][2],
        families["dt"][1],
    ]
    families["baseline"] = [base]
    return families


def diversity(hards: Sequence[np.ndarray], answers: np.ndarray, k: int) -> dict:
    """Pairwise decision disagreement and error-set Jaccard over trajectories."""
    hards = [np.asarray(hard, dtype=np.uint8)[:, :k] for hard in hards]
    truth = np.asarray(answers, dtype=np.uint8)[:, :k]
    disagreement: list[float] = []
    jaccard: list[float] = []
    for left_index in range(len(hards)):
        for right_index in range(left_index + 1, len(hards)):
            left, right = hards[left_index], hards[right_index]
            disagreement.append(float(np.mean(left != right)))
            left_bad = left != truth
            right_bad = right != truth
            union = np.logical_or(left_bad, right_bad).sum()
            jaccard.append(
                0.0 if union == 0 else float(np.logical_and(left_bad, right_bad).sum() / union)
            )
    return {
        "pair_disagreement_mean": float(np.mean(disagreement)) if disagreement else 0.0,
        "pair_disagreement_max": float(max(disagreement)) if disagreement else 0.0,
        "error_jaccard_mean": float(np.mean(jaccard)) if jaccard else 1.0,
        "error_jaccard_min": float(min(jaccard)) if jaccard else 1.0,
    }


def screen_case(workspace: p.CaseWorkspace, families: dict[str, list[dict]]) -> dict:
    base = p.load_v17_config(workspace.n, workspace.k)
    unique: dict[str, dict] = {}
    for configs in families.values():
        for config in configs:
            unique.setdefault(config_key(config), config)

    hard_cache: dict[str, np.ndarray] = {}
    started = time.time()
    for key, config in unique.items():
        hard_cache[key] = p.decode_hard(workspace, config)
    decode_seconds = time.time() - started

    results: dict[str, dict] = {}
    for name, configs in families.items():
        hards = [hard_cache[config_key(config)] for config in configs]
        outcome = p.portfolio_from_hards(workspace, hards, configs)
        selection = p.gain_report(workspace, outcome["selected_errors"])
        oracle = p.gain_report(workspace, outcome["oracle_errors"])
        results[name] = {
            "trajectories": len(configs),
            "config_indices": [list(unique).index(config_key(c)) for c in configs],
            "selection": selection,
            "oracle": oracle,
            "selection_accuracy": outcome["selection_accuracy"],
            "diversity": diversity(hards, workspace.case["answers"], workspace.k),
            "per_trajectory_errors": np.asarray(outcome["per_trajectory_errors"]).sum(axis=1).astype(int).tolist(),
        }

    return {
        "workspace": workspace.summary(),
        "unique_configs": len(unique),
        "configs": [unique[key] for key in unique],
        "config_keys": list(unique),
        "decode_seconds": decode_seconds,
        "families": results,
    }


def apply_gate(case_10: dict, case_7: dict) -> dict:
    candidates = []
    for name, result in case_10["families"].items():
        if name == "baseline":
            continue
        oracle = result["oracle"]["gain"]
        realised = result["selection"]["gain"]
        candidates.append((name, oracle, realised))
    best = max(candidates, key=lambda item: item[1]) if candidates else ("none", 0.0, 0.0)
    oracle_best, realised_best = best[1], best[2]
    if oracle_best < 5000.0:
        decision = "stop_A"
        reason = "best N=3888 oracle gain below +5k"
    elif oracle_best >= 8000.0 and realised_best >= 4000.0:
        decision = "continue_A2"
        reason = "N=3888 oracle >= +8k and realised >= +4k"
    else:
        decision = "continue_best_family_no_A2"
        reason = "oracle in [5k, 8k) or realised below +4k"
    return {
        "decision": decision,
        "reason": reason,
        "best_family": best[0],
        "best_oracle_gain": oracle_best,
        "best_realised_gain": realised_best,
        "case7_best_oracle_gain": max(
            result["oracle"]["gain"]
            for name, result in case_7["families"].items()
            if name != "baseline"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=int, default=[10, 7])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cache_frames = 300 if args.seed == 911 else 500
    if args.frames > cache_frames:
        raise ValueError(f"frames must be <= {cache_frames} for seed {args.seed}")
    results: dict[int, dict] = {}
    for case_index in args.cases:
        spec = p.CASE_SPECS[case_index - 1]
        _, n, k, _, snr = spec
        workspace = p.CaseWorkspace(
            n, k, args.seed, cache_frames, snr=snr, frame_slice=args.frames
        )
        base = p.load_v17_config(n, k)
        families = build_families(base)
        print(f"case {case_index}: N={n}, frames={args.frames}, configs={len(set(config_key(c) for group in families.values() for c in group))}", flush=True)
        results[case_index] = screen_case(workspace, families)

    gate = None
    if 10 in results and 7 in results:
        gate = apply_gate(results[10], results[7])
    payload = {
        "gate": gate,
        "seed": args.seed,
        "frames": args.frames,
        "noise_scale": p.NOISE_SCALE,
        "llr_sigma": "nominal",
        "bootstrap_resamples": 1000,
        "cases": results,
    }
    p.write_json(args.out, payload)
    print(json.dumps({
        "out": str(args.out),
        "gate": gate,
        "case10": {
            name: {
                "oracle_gain": result["oracle"]["gain"],
                "oracle_ci": result["oracle"]["gain_ci_95"],
                "realised_gain": result["selection"]["gain"],
                "realised_ci": result["selection"]["gain_ci_95"],
                "accuracy": result["selection_accuracy"],
            }
            for name, result in results.get(10, {}).get("families", {}).items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
