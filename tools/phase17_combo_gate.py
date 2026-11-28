#!/usr/bin/env python3
"""Gate a combined Phase 17 manifest against the better single-case manifest.

The integration rule is that a combination must gain at least +1,000 total
score points over the better of the two single-track manifests on both
1,000-frame gate seeds, with a positive paired-bootstrap CI lower bound.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as pc  # noqa: E402
from baseline_cache import cache_path  # noqa: E402
from phase17_case_gate import decode_case  # noqa: E402

ROOT = TOOLS_DIR.parent


def total_bootstrap_delta(
    per_case: list[tuple[np.ndarray, np.ndarray, int]],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Paired bootstrap CI for the summed score gain across differing cases."""
    rng = np.random.default_rng(int(seed))
    deltas = np.zeros(int(samples), dtype=float)
    for base_errors, cand_errors, k in per_case:
        frames = len(base_errors)
        indices = rng.integers(0, frames, size=(int(samples), frames))
        base_scores = np.empty(int(samples), dtype=float)
        cand_scores = np.empty(int(samples), dtype=float)
        total = frames * int(k)
        for row in range(int(samples)):
            draw = indices[row]
            base_scores[row] = pc.score_of(int(base_errors[draw].sum()), total)
            cand_scores[row] = pc.score_of(int(cand_errors[draw].sum()), total)
        deltas += cand_scores - base_scores
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 31337])
    parser.add_argument("--cache-dir", type=Path, default=pc.V19_CACHE)
    parser.add_argument("--min-gain", type=float, default=1000.0)
    parser.add_argument("--bootstrap-seed", type=int, default=5150)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    ref_cases = reference.get("cases", [])
    cand_cases = candidate.get("cases", [])
    if len(ref_cases) != len(cand_cases) or not ref_cases:
        raise ValueError("reference and candidate must have the same non-empty case list")
    differing = [
        index + 1 for index, (ref, cand) in enumerate(zip(ref_cases, cand_cases))
        if ref != cand
    ]
    if not differing:
        raise ValueError("reference and candidate manifests are identical")

    results: list[dict] = []
    workspaces: dict[tuple[int, int, int], pc.CaseWorkspace] = {}
    for seed in args.seeds:
        started = time.perf_counter()
        per_case: list[tuple[np.ndarray, np.ndarray, int]] = []
        case_rows: list[dict] = []
        for index in differing:
            case = cand_cases[index - 1]
            n, k = int(case["n"]), int(case["k"])
            key = (int(seed), n, k)
            workspace = workspaces.get(key)
            if workspace is None:
                cache = cache_path(args.cache_dir, n, k, seed)
                if not cache.exists():
                    raise FileNotFoundError(f"missing baseline cache {cache}")
                frames = int(np.load(cache).size)
                workspace = pc.CaseWorkspace(
                    n=n, k=k, seed=int(seed), frames=frames,
                    snr=float(case["snr"]), cache_dir=args.cache_dir,
                    frame_slice=frames,
                )
                workspaces[key] = workspace
            ref_hard = decode_case(workspace, ref_cases[index - 1], args.batch_size)
            cand_hard = decode_case(workspace, case, args.batch_size)
            ref_errors = pc.per_frame_errors(ref_hard, workspace.case)
            cand_errors = pc.per_frame_errors(cand_hard, workspace.case)
            per_case.append((ref_errors, cand_errors, k))
            gain = pc.score_from_errors(cand_errors, k) - pc.score_from_errors(ref_errors, k)
            case_rows.append({
                "case": index,
                "seed": int(seed),
                "reference_errors": int(ref_errors.sum()),
                "candidate_errors": int(cand_errors.sum()),
                "gain": round(float(gain), 3),
            })
        low, high = total_bootstrap_delta(
            per_case, samples=1000, seed=args.bootstrap_seed + int(seed)
        )
        total_gain = sum(float(row["gain"]) for row in case_rows)
        passed = bool(total_gain >= args.min_gain and low > 0.0)
        results.append({
            "seed": int(seed),
            "frames": len(per_case[0][0]) if per_case else 0,
            "total_gain": round(total_gain, 3),
            "gain_ci_95": [round(low, 3), round(high, 3)],
            "min_gain": args.min_gain,
            "pass": passed,
            "cases": case_rows,
            "elapsed_s": round(time.perf_counter() - started, 3),
        })
        print(
            f"seed {seed}: total {total_gain:+.3f} "
            f"CI=[{low:+.3f},{high:+.3f}] ({'PASS' if passed else 'FAIL'})",
            flush=True,
        )

    payload = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "differing_cases": differing,
        "bootstrap_seed": args.bootstrap_seed,
        "min_gain": args.min_gain,
        "results": results,
        "all_passed": all(item["pass"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
