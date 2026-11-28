#!/usr/bin/env python3
"""Offline Phase 16 screen: implied-check landscape terms on one case.

No BP, OSD, list decoding, or message state is used. Redundant checks are
row-space combinations of H; their Ising monomials are satisfied by every
codeword, so adding them to the QAIA landscape does not change the solution.
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

DEFAULT_SELECTION = p.ROOT / "artifacts" / "phase15" / "selection_greedy_both3.json"
DEFAULT_OUT = p.PHASE15_ARTIFACTS / "phase16_case10_redundant_screen.json"


def row_ints(h: np.ndarray) -> list[int]:
    n = int(h.shape[1])
    return [
        sum(int(bit) << index for index, bit in enumerate(row))
        for row in np.asarray(h, dtype=np.int8)
    ]


def support_of(value: int, n: int) -> list[int]:
    return [index for index in range(n) if (value >> index) & 1]


def find_redundant_checks(
    h: np.ndarray,
    count: int,
    max_weight: int,
    max_rows: int,
    starts: int,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    """Find low-weight implied checks by greedy row-space XOR reduction."""
    rows = row_ints(h)
    m, n = int(h.shape[0]), int(h.shape[1])
    rng = np.random.default_rng(int(seed))

    pairs: dict[int, tuple[int, int, int]] = {}
    for i in range(m):
        value_i = rows[i]
        for j in range(i + 1, m):
            value = value_i ^ rows[j]
            weight = int(value.bit_count())
            if 3 <= weight <= max_weight:
                previous = pairs.get(value)
                if previous is None or weight < previous[0]:
                    pairs[value] = (weight, i, j)

    ranked = sorted(pairs.items(), key=lambda item: (item[1][0], item[1][1], item[1][2]))
    if not ranked:
        raise RuntimeError(f"no implied check up to weight {max_weight}")

    chosen: dict[int, int] = {}
    for value, (weight, first, second) in ranked[: max(starts, count * 4)]:
        current = value
        current_weight = weight
        used = {first, second}
        for _ in range(max(0, max_rows - 2)):
            best_value = None
            best_weight = current_weight
            best_row = None
            for row_index in rng.permutation(m):
                row_index = int(row_index)
                if row_index in used:
                    continue
                candidate = current ^ rows[row_index]
                candidate_weight = int(candidate.bit_count())
                if 3 <= candidate_weight < best_weight:
                    best_value = candidate
                    best_weight = candidate_weight
                    best_row = row_index
            if best_value is None or best_row is None:
                break
            current = best_value
            current_weight = best_weight
            used.add(best_row)
        chosen[current] = current_weight
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        for value, (weight, _, _) in ranked:
            chosen[value] = weight
            if len(chosen) >= count:
                break

    ordered = sorted(chosen.items(), key=lambda item: (item[1], item[0]))[:count]
    supports = [support_of(value, n) for value, _ in ordered]
    return supports, [weight for _, weight in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=10)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-weight", type=int, default=14)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--search-seed", type=int, default=20261127)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--tap2s", type=float, nargs="+", default=[0.0])
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    _, n, k, rate, snr = p.CASE_SPECS[args.case - 1]
    cache_frames = 300 if args.seed == 911 else 500
    if args.frames > cache_frames:
        raise ValueError(f"frames must be <= {cache_frames} for seed {args.seed}")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    base_configs = selection["cases"][str(args.case)]["trajectories"]
    base_configs = [dict(config) for config in base_configs]
    if not 1 <= len(base_configs) <= 6:
        raise ValueError("selection must contain 1..6 trajectories")

    workspace = p.CaseWorkspace(n, k, args.seed, cache_frames, snr=snr, frame_slice=args.frames)
    supports, weights = find_redundant_checks(
        workspace.case["h"], args.count, args.max_weight, args.max_rows, args.starts, args.search_seed
    )
    print(f"redundant checks: {len(supports)}, weights={weights}", flush=True)

    base_outcome = p.portfolio_errors(workspace, base_configs)
    base_errors = base_outcome["selected_errors"]
    print(
        f"v18b baseline: score {p.gain_report(workspace, base_errors)['candidate_score']:.1f}, "
        f"errors {int(base_errors.sum())}",
        flush=True,
    )

    results = []
    for gamma in args.gammas:
        for tap2 in args.tap2s:
            configs = [dict(config) for config in base_configs]
            if gamma != 0.0:
                for config in configs:
                    config["redundant_supports"] = supports
                    config["redundant_gamma"] = float(gamma)
            if tap2 != 0.0:
                for config in configs:
                    config["tap2"] = float(tap2)
            outcome = p.portfolio_errors(workspace, configs)
            errors = outcome["selected_errors"]
            low, high = p.paired_bootstrap(base_errors, errors, k, samples=1000, seed=5150)
            candidate_score = p.score_from_errors(errors, k)
            base_score = p.score_from_errors(base_errors, k)
            oracle_errors = outcome["oracle_errors"]
            oracle_gain = p.score_from_errors(oracle_errors, k) - base_score
            row = {
                "gamma": float(gamma),
                "tap2": float(tap2),
                "score": candidate_score,
                "gain_vs_v18b": candidate_score - base_score,
                "gain_ci_95_vs_v18b": [low, high],
                "oracle_gain_vs_v18b": oracle_gain,
                "errors": int(errors.sum()),
                "oracle_errors": int(oracle_errors.sum()),
                "selection_accuracy": outcome["selection_accuracy"],
            }
            results.append(row)
            print(
                f"gamma {gamma:g} tap2 {tap2:g}: gain {row['gain_vs_v18b']:+.1f} "
                f"CI [{low:.1f}, {high:.1f}], oracle {oracle_gain:+.1f}",
                flush=True,
            )

    payload = {
        "case": args.case,
        "n": n,
        "k": k,
        "seed": args.seed,
        "frames": args.frames,
        "noise_scale": p.NOISE_SCALE,
        "selection": str(args.selection.resolve()),
        "baseline_trajectories": len(base_configs),
        "redundant_support_count": len(supports),
        "redundant_weights": weights,
        "redundant_supports": supports,
        "search": {
            "max_weight": args.max_weight,
            "max_rows": args.max_rows,
            "starts": args.starts,
            "seed": args.search_seed,
        },
        "bootstrap_resamples": 1000,
        "tap2s": args.tap2s,
        "results": results,
    }
    p.write_json(args.out, payload)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
