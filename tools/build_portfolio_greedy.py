#!/usr/bin/env python3
"""Greedy build an energy-selected QAIA portfolio of at most six trajectories.

Starting from the best single trajectory, each step adds the candidate that
maximises the realised energy-selection gain against cached v17 on identical
training frames. A single ``energy_lambda`` multiplier per case may be tuned on
the same frames; the chosen value must be revalidated on other seeds.
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

import phase15_common as p  # noqa: E402

DEFAULT_OUT = p.PHASE15_ARTIFACTS / "greedy_portfolio.json"
LAMBDA_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="JSON object with a 'cases' map or a flat list of configs")
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--max-trajectories", type=int, default=6)
    parser.add_argument("--tune-lambda", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        candidates = payload.get("cases", {}).get(str(args.case), [])
    else:
        candidates = payload
    if not candidates:
        raise ValueError(f"no candidates for case {args.case}")

    _, n, k, rate, snr = p.CASE_SPECS[args.case - 1]
    cache_frames = 300 if args.seed == 911 else 500
    if args.frames > cache_frames:
        raise ValueError(f"frames must be <= {cache_frames} for seed {args.seed}")
    workspace = p.CaseWorkspace(n, k, args.seed, cache_frames, snr=snr, frame_slice=args.frames)

    hards = [p.decode_hard(workspace, config) for config in candidates]
    error_counts = [int(p.per_frame_errors(hard, workspace.case).sum()) for hard in hards]
    remaining = list(range(len(candidates)))
    best_index = min(remaining, key=lambda index: error_counts[index])
    selected = [best_index]
    best_gain = p.gain_report(workspace, p.per_frame_errors(hards[best_index], workspace.case))["gain"]
    remaining.remove(best_index)
    history = [{
        "step": 0,
        "selected": selected[:],
        "gain": best_gain,
        "candidate_errors": error_counts[best_index],
    }]

    max_trajectories = min(int(args.max_trajectories), 6)
    while remaining and len(selected) < max_trajectories:
        scored = []
        for candidate_index in remaining:
            trial = selected + [candidate_index]
            outcome = p.portfolio_from_hards(
                workspace, [hards[index] for index in trial], [candidates[index] for index in trial]
            )
            gain = p.gain_report(workspace, outcome["selected_errors"])["gain"]
            scored.append((gain, candidate_index, outcome))
        gain, candidate_index, outcome = max(scored, key=lambda item: item[0])
        if gain <= best_gain + 1.0:
            break
        best_gain = gain
        selected.append(candidate_index)
        remaining.remove(candidate_index)
        history.append({
            "step": len(selected) - 1,
            "selected": selected[:],
            "gain": best_gain,
            "oracle_gain": p.gain_report(workspace, outcome["oracle_errors"])["gain"],
            "selection_accuracy": outcome["selection_accuracy"],
        })

    chosen = [candidates[index] for index in selected]
    lambda_multiplier = 1.0
    if args.tune_lambda:
        scored_lambdas = []
        for multiplier in LAMBDA_MULTIPLIERS:
            tuned = [{**config, "energy_lambda": float(config.get("energy_lambda", config["lambd"])) * multiplier}
                     for config in chosen]
            outcome = p.portfolio_from_hards(
                workspace, [hards[index] for index in selected], tuned
            ) if len(chosen) > 1 else {
                "selected_errors": p.per_frame_errors(hards[selected[0]], workspace.case)
            }
            gain = p.gain_report(workspace, outcome["selected_errors"])["gain"]
            scored_lambdas.append((gain, multiplier))
        best_gain, lambda_multiplier = max(scored_lambdas, key=lambda item: item[0])
        if lambda_multiplier != 1.0:
            chosen = [{**config, "energy_lambda": float(config.get("energy_lambda", config["lambd"])) * lambda_multiplier}
                      for config in chosen]

    result = {
        "case": args.case, "n": n, "k": k, "rate": rate,
        "seed": args.seed, "frames": args.frames,
        "candidate_count": len(candidates),
        "selected_indices": selected,
        "energy_lambda_multiplier": lambda_multiplier,
        "training_gain": best_gain,
        "history": history,
        "trajectories": chosen,
    }
    p.write_json(args.out, result)
    print(json.dumps({
        "out": str(args.out), "case": args.case,
        "trajectories": len(chosen), "training_gain": best_gain,
        "energy_lambda_multiplier": lambda_multiplier,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
