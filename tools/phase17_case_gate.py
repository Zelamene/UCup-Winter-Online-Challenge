#!/usr/bin/env python3
"""Frame-matched Phase 17 gate of a candidate manifest against accepted v19.

Unlike Phase 16's 500-frame v18b gate, this gate reads the v19 cache, whose
required seeds 2024 and 31337 contain 1,000 frames. It keeps the same paired
bootstrap analysis and requires the CI lower bound to be positive.
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

ROOT = TOOLS_DIR.parent


def decode_case(workspace: pc.CaseWorkspace, case: dict, batch_size: int) -> np.ndarray:
    """Decode one manifest case and return the energy-selected hard decisions."""
    configs = []
    for trajectory in case["trajectories"]:
        config = dict(trajectory)
        config["n"] = int(case["n"])
        config["k"] = int(case["k"])
        config["snr"] = float(case["snr"])
        config["mode"] = str(case.get("mode", "kspin"))
        config["combine"] = str(case.get("combine", "energy"))
        config.setdefault("dt", float(case.get("dt", 0.47)))
        configs.append(config)
    hards = [pc.decode_hard(workspace, config, batch_size=batch_size) for config in configs]
    portfolio = pc.portfolio_from_hards(workspace, hards, configs)
    return np.stack(hards)[portfolio["selected_index"], np.arange(workspace.frames)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 31337])
    parser.add_argument("--min-gain", type=float, default=1500.0)
    parser.add_argument("--frames", type=int, default=0, help="0 means the full cached seed")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=5150)
    parser.add_argument("--cache-dir", type=Path, default=pc.V19_CACHE)
    parser.add_argument("--baseline-label", default="v19")
    parser.add_argument("--changed-cases", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    changed = args.changed_cases
    if changed is None:
        changed = [int(value) for value in manifest.get("changed_cases", [])]
    if not changed:
        raise ValueError("manifest has no changed cases")

    results = []
    for seed in args.seeds:
        for index in changed:
            case = cases[index - 1]
            n, k = int(case["n"]), int(case["k"])
            cache = cache_path(args.cache_dir, n, k, seed)
            if not cache.exists():
                raise FileNotFoundError(f"missing {args.baseline_label} baseline cache {cache}")
            cached_frames = int(np.load(cache).size)
            frames = cached_frames if args.frames <= 0 else min(args.frames, cached_frames)
            workspace = pc.CaseWorkspace(
                n=n, k=k, seed=seed, frames=cached_frames,
                snr=float(case["snr"]), cache_dir=args.cache_dir, frame_slice=frames,
            )
            started = time.perf_counter()
            hard = decode_case(workspace, case, args.batch_size)
            elapsed = time.perf_counter() - started
            errors = pc.per_frame_errors(hard, workspace.case)
            report = pc.gain_report(workspace, errors, args.bootstrap_seed)
            gain = report["gain"]
            low, high = report["gain_ci_95"]
            entry = {
                "seed": seed,
                "case": index,
                "n": n,
                "k": k,
                "frames": frames,
                "trajectory_count": len(case["trajectories"]),
                "elapsed_s": round(elapsed, 3),
                **report,
                "gate_passed": bool(gain >= args.min_gain and low > 0.0),
            }
            results.append(entry)
            print(
                f"seed {seed} case {index:>2d}: gain {gain:+9.1f} "
                f"CI=[{low:+.1f},{high:+.1f}] "
                f"errors {report['baseline_errors']}->{report['candidate_errors']} "
                f"({'PASS' if entry['gate_passed'] else 'FAIL'})",
                flush=True,
            )

    payload = {
        "manifest": str(args.manifest.resolve()),
        "baseline": args.baseline_label,
        "baseline_cache_dir": str(args.cache_dir.resolve()),
        "seeds": args.seeds,
        "min_gain": args.min_gain,
        "results": results,
        "all_passed": all(item["gate_passed"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pc.write_json(args.output, payload)
    print(f"Saved {args.output}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
