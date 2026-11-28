#!/usr/bin/env python3
"""Frame-matched K-spin screen against the accepted v18b baseline.

Each configuration is decoded on exactly the same fixed-rule frames as the
compiled v18b binary, and score gains use the same 1,000-resample paired
bootstrap as the final gate.
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

CASE_DIMS = {1: (648, 324), 4: (1296, 648), 7: (1944, 972), 10: (3888, 1944)}


def load_configs(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("configs", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty list of configs")
    configs = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict) or "name" not in raw:
            raise ValueError(f"{path}: config {index} needs a name")
        config = dict(raw)
        config.setdefault("mode", "kspin")
        config.setdefault("combine", "energy")
        config.setdefault("lambd", 1.0)
        config.setdefault("parity_scale", 1.0)
        config.setdefault("select_unclipped", 0)
        configs.append(config)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=5150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configs = load_configs(args.configs)
    results = []
    for case_index in args.cases:
        if case_index not in CASE_DIMS:
            raise ValueError(f"case {case_index} is not a rate-1/2 K-spin candidate")
        n, k = CASE_DIMS[case_index]
        cache = cache_path(pc.V18_CACHE, n, k, args.seed)
        if not cache.exists():
            raise FileNotFoundError(f"missing v18 baseline cache {cache}")
        cached_frames = int(np.load(cache).size)
        workspace = pc.CaseWorkspace(
            n=n,
            k=k,
            seed=args.seed,
            frames=cached_frames,
            snr=pc.configured_snr(k / n),
            cache_dir=pc.V18_CACHE,
            frame_slice=args.frames,
        )
        print(
            f"case {case_index} N={n} K={k} seed={args.seed} "
            f"frames={workspace.frames} baseline_errors={int(workspace.baseline_errors.sum())}",
            flush=True,
        )
        for config in configs:
            started = time.perf_counter()
            hard = pc.decode_hard(
                workspace, config, batch_size=args.batch_size
            )
            elapsed = time.perf_counter() - started
            errors = pc.per_frame_errors(hard, workspace.case)
            report = pc.gain_report(workspace, errors, args.bootstrap_seed)
            row = {
                "case": case_index,
                "n": n,
                "k": k,
                "seed": args.seed,
                "frames": workspace.frames,
                "name": config["name"],
                "config": config,
                "elapsed_s": elapsed,
                **report,
            }
            results.append(row)
            print(
                f"  {config['name']:<24} errors={int(errors.sum()):>7} "
                f"gain={report['gain']:+9.1f} "
                f"CI=[{report['gain_ci_95'][0]:+.1f},{report['gain_ci_95'][1]:+.1f}] "
                f"{elapsed:.1f}s",
                flush=True,
            )

    payload = {
        "base": "artifacts/baseline_cache_v18",
        "cases": args.cases,
        "seed": args.seed,
        "frames": args.frames,
        "batch_size": args.batch_size,
        "results": results,
    }
    pc.write_json(args.output, payload)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
