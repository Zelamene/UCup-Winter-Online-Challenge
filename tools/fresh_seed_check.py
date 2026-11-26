#!/usr/bin/env python3
"""Fresh-seed compiled-binary gain check for a candidate manifest.

The tool compares a base binary (the accepted v13c build) with a candidate
binary on independently generated frames for every case whose v3 manifest
entry changed. Ground truth is used only offline to count errors; the submitted
binary never sees it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import read_hard, run_binary, write_input  # noqa: E402
from real_bench import build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
DEFAULT_BASE_BINARY = ROOT / "artifacts" / "build" / "v13c.exe"
DEFAULT_BASE_MANIFEST = ROOT / "artifacts" / "phase9_case_manifest_v13c.json"


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def changed_cases(base_manifest: dict, candidate_manifest: dict) -> list[dict]:
    base_cases = copy.deepcopy(base_manifest.get("cases", []))
    candidate_cases = copy.deepcopy(candidate_manifest.get("cases", []))
    if len(base_cases) != len(candidate_cases):
        raise ValueError("base and candidate manifests have different case counts")
    changed = []
    for index, (before, after) in enumerate(zip(base_cases, candidate_cases), start=1):
        if before != after:
            changed.append({
                "index": index,
                "n": int(after["n"]),
                "k": int(after["k"]),
                "snr": float(after["snr"]),
            })
    return changed


def evaluate_binary(binary: Path, case: dict, workdir: Path, label: str) -> tuple[int, float, float]:
    input_path = workdir / f"{label}.in"
    output_path = workdir / f"{label}.out"
    answers_path = workdir / f"{label}.answers"
    write_input(case, input_path)
    from real_bench import write_answers
    write_answers(case, answers_path)
    elapsed = run_binary(binary, input_path, output_path)
    hard = read_hard(output_path, case["b"], case["n"])
    errors = int(np_count_nonzero(hard[:, : case["k"]] != case["answers"][:, : case["k"]]))
    return errors, score_of(errors, case["b"] * case["k"]), elapsed


def np_count_nonzero(array) -> int:
    import numpy as np
    return int(np.count_nonzero(array))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-binary", type=Path, default=DEFAULT_BASE_BINARY)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--noise-scale", type=float, default=1.001120)
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "fresh_seed_work")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    cases = changed_cases(base_manifest, candidate_manifest)
    if not cases:
        raise SystemExit("candidate manifest changes no cases; nothing to check")

    args.workdir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in cases:
        n, k, snr = item["n"], item["k"], item["snr"]
        case = build_case(
            n, k, args.samples, snr, args.seed, noise_scale=args.noise_scale
        )
        with tempfile.TemporaryDirectory(prefix=f"fresh_{n}_{k}_", dir=args.workdir) as tmp_name:
            workdir = Path(tmp_name)
            base_errors, base_score, base_seconds = evaluate_binary(
                args.base_binary, case, workdir, "base"
            )
            cand_errors, cand_score, cand_seconds = evaluate_binary(
                args.candidate_binary, case, workdir, "candidate"
            )
        entry = {
            **item,
            "seed": args.seed,
            "frames": case["b"],
            "base_errors": base_errors,
            "base_score": base_score,
            "candidate_errors": cand_errors,
            "candidate_score": cand_score,
            "gain": cand_score - base_score,
            "base_seconds": round(base_seconds, 3),
            "candidate_seconds": round(cand_seconds, 3),
        }
        results.append(entry)
        print(
            f"N={n} K={k} seed={args.seed}: {base_score:,.1f} -> {cand_score:,.1f} "
            f"({cand_score - base_score:+,.1f}); "
            f"errors {base_errors} -> {cand_errors}; "
            f"candidate {cand_seconds:.1f}s",
            flush=True,
        )

    payload = {
        "base_binary": str(args.base_binary),
        "candidate_binary": str(args.candidate_binary),
        "seed": args.seed,
        "samples": args.samples,
        "noise_scale": args.noise_scale,
        "cases": results,
        "min_gain": min(item["gain"] for item in results),
        "all_positive": all(item["gain"] > 0 for item in results),
        "all_runtime_under_120s": all(
            item["candidate_seconds"] < 120.0 for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: min gain {payload['min_gain']:+,.1f}, "
        f"all positive={payload['all_positive']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
