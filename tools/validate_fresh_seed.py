#!/usr/bin/env python3
"""Fresh-seed v16 gate: compiled candidate vs compiled v15 on seed 6060.

Only cases whose manifests differ from the accepted v15 manifest are gated.
Each changed case must gain at least ``--min-gain`` points on the same 300
frames with a paired 95% bootstrap CI lower bound above zero. Unchanged cases
must be byte-identical as manifest objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import read_hard, run_binary, write_input  # noqa: E402
from real_bench import build_case  # noqa: E402
from tune_tap_local import paired_bootstrap, score_of  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_V15_MANIFEST = ROOT / "submissions" / "v15_tap_bsb_candidate" / "manifest.json"
DEFAULT_V15_BINARY = ROOT / "artifacts" / "build" / "v15_ref.exe"


def run_binary_errors(
    binary: Path, case: dict, workdir: Path, tag: str
) -> tuple[np.ndarray, float]:
    input_path = workdir / f"{tag}.in"
    output_path = workdir / f"{tag}.out"
    write_input(case, input_path)
    elapsed = run_binary(binary, input_path, output_path)
    hard = read_hard(output_path, case["b"], case["n"])
    errors = np.count_nonzero(
        hard[:, :case["k"]] != case["answers"][:, :case["k"]], axis=1
    ).astype(np.int64)
    return errors, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_V15_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_V15_BINARY)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--min-gain", type=float, default=1500.0)
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "build" / "fresh6060")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "phase12_fresh_seed.json"
    )
    args = parser.parse_args()

    for path in (args.manifest, args.base_manifest, args.candidate, args.baseline):
        if not path.exists():
            raise SystemExit(f"missing file: {path}")

    candidate_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    base_cases = base_manifest.get("cases", [])
    cand_cases = candidate_manifest.get("cases", [])
    if len(base_cases) != len(cand_cases):
        raise SystemExit(f"case count mismatch: {len(base_cases)} vs {len(cand_cases)}")

    unchanged = [
        index + 1 for index, (old, new) in enumerate(zip(base_cases, cand_cases))
        if old == new
    ]
    changed = [
        index + 1 for index, (old, new) in enumerate(zip(base_cases, cand_cases))
        if old != new
    ]
    if not changed:
        raise SystemExit("no changed cases; fresh-seed gate has nothing to test")

    args.workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = []
    print(f"Fresh-seed gate: seed {args.seed}, {args.frames} frames, min gain {args.min_gain:+,.0f}")
    print(f"changed cases: {changed}; unchanged cases: {unchanged}", flush=True)
    for index in changed:
        case_cfg = cand_cases[index - 1]
        n, k, snr = int(case_cfg["n"]), int(case_cfg["k"]), float(case_cfg["snr"])
        noise_scale = float(case_cfg.get("noise_scale", 1.001120))
        case = build_case(n, k, args.frames, snr, args.seed, noise_scale=noise_scale)
        base_errs, base_time = run_binary_errors(
            args.baseline, case, args.workdir, f"case{index}_v15"
        )
        cand_errs, cand_time = run_binary_errors(
            args.candidate, case, args.workdir, f"case{index}_candidate"
        )
        base_total, cand_total = int(base_errs.sum()), int(cand_errs.sum())
        base_score = score_of(base_total, args.frames * k)
        cand_score = score_of(cand_total, args.frames * k)
        gain = cand_score - base_score
        low, high = paired_bootstrap(base_errs, cand_errs, k, 1000, rng)
        passed = gain >= args.min_gain and low > 0.0
        entry = {
            "case": index,
            "n": n,
            "k": k,
            "snr": snr,
            "frames": args.frames,
            "baseline_score": base_score,
            "candidate_score": cand_score,
            "gain": gain,
            "gain_ci_95": [low, high],
            "baseline_errors": base_total,
            "candidate_errors": cand_total,
            "baseline_seconds": round(base_time, 2),
            "candidate_seconds": round(cand_time, 2),
            "passed": passed,
        }
        results.append(entry)
        print(
            f"  case {index:2d} N={n} K={k}: {base_score:,.1f} -> {cand_score:,.1f} "
            f"({gain:+,.1f}, CI [{low:+,.1f}, {high:+,.1f}]) "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    summary = {
        "manifest": str(args.manifest),
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "seed": args.seed,
        "frames": args.frames,
        "required_gain": args.min_gain,
        "changed_cases": changed,
        "unchanged_cases": unchanged,
        "results": results,
        "all_passed": all(item["passed"] for item in results),
        "min_observed_gain": min(item["gain"] for item in results),
        "min_ci_lower": min(item["gain_ci_95"][0] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"all fresh-seed gates passed: {summary['all_passed']}")


if __name__ == "__main__":
    main()
