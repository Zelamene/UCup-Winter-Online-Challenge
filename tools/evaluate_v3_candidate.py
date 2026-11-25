#!/usr/bin/env python3
"""Compare a v3 QAIA candidate against a baseline on the realistic bench."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from decode_v3_python import decode_frame, select_case  # noqa: E402
from real_bench import build_case  # noqa: E402

SCORE_SCALE = 100_000.0
CASE_SPECS = (
    (648, 324, 6.0),
    (648, 432, 9.0),
    (648, 486, 10.5),
    (1296, 648, 6.0),
    (1296, 864, 9.0),
    (1296, 972, 10.5),
    (1944, 972, 6.0),
    (1944, 1296, 9.0),
    (1944, 1458, 10.5),
    (3888, 1944, 6.0),
    (3888, 2592, 9.0),
    (3888, 2916, 10.5),
)


def _score_manifest(manifest: dict, cases: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        config = select_case(manifest, case["n"], case["k"], case["snr_db"])
        errors = 0
        for llr, answer in zip(case["llr"], case["answers"]):
            hard = np.asarray(decode_frame(llr, case["h"], config), dtype=np.int8)
            errors += int(np.count_nonzero(hard[: case["k"]] != answer[: case["k"]]))
        bits = case["b"] * case["k"]
        ber = errors / bits
        score = -SCORE_SCALE * math.log10(max(ber, 1.0 / bits))
        rows.append(
            {
                "index": index,
                "n": case["n"],
                "k": case["k"],
                "snr": case["snr_db"],
                "errors": errors,
                "bits": bits,
                "ber": ber,
                "score": score,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--min-total-gain", type=float, default=1000.0)
    parser.add_argument("--max-case-loss", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    cases = [
        build_case(
            n,
            k,
            args.samples,
            snr,
            args.seed + index,
            noise_scale=args.noise_scale,
        )
        for index, (n, k, snr) in enumerate(CASE_SPECS)
    ]
    candidate_rows = _score_manifest(candidate, cases)
    baseline_rows = _score_manifest(baseline, cases)

    comparisons = []
    for base, cand in zip(baseline_rows, candidate_rows):
        delta = cand["score"] - base["score"]
        comparisons.append(
            {
                "index": cand["index"],
                "n": cand["n"],
                "k": cand["k"],
                "snr": cand["snr"],
                "baseline_score": base["score"],
                "candidate_score": cand["score"],
                "delta": delta,
                "passed": delta >= -args.max_case_loss,
            }
        )

    total_gain = sum(item["delta"] for item in comparisons)
    gate = {
        "total_gain": total_gain,
        "min_total_gain": args.min_total_gain,
        "max_case_loss": args.max_case_loss,
        "no_case_regression": all(item["passed"] for item in comparisons),
        "passed": total_gain >= args.min_total_gain
        and all(item["passed"] for item in comparisons),
    }
    payload = {
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "samples": args.samples,
        "seed": args.seed,
        "noise_scale": args.noise_scale,
        "score_scale": SCORE_SCALE,
        "comparisons": comparisons,
        "gate": gate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("case  N     K     SNR   base      cand      delta")
    for item in comparisons:
        print(
            f"{item['index']:>4d}  {item['n']:>4d}  {item['k']:>4d}  "
            f"{item['snr']:>4.1f}  {item['baseline_score']:>8.0f}  "
            f"{item['candidate_score']:>8.0f}  {item['delta']:>+8.0f}"
        )
    print(f"total gain: {total_gain:+,.0f}")
    print(f"gate passed: {gate['passed']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
