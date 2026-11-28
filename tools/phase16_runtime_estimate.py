#!/usr/bin/env python3
"""Estimate judge runtime from accepted v18b judge times and local ratios."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from binary_io import run_binary, write_input  # noqa: E402
from real_bench import N_VALUES, RATE_SNR_DB, build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
NOISE_SCALE = 1.001120
V18_JUDGE_RESULT = ROOT / "submissions" / "v18b_greedy_portfolio" / "judge_result.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-judge-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(V18_JUDGE_RESULT.read_text(encoding="utf-8"))
    judge_times = {
        int(item["case"]): float(item["time_ms"]) / 1000.0
        for item in payload["per_case"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix="runtime16_", dir=args.output.parent) as tmp_name:
        workdir = Path(tmp_name)
        index = 0
        for n in N_VALUES:
            for rate, snr in RATE_SNR_DB:
                index += 1
                k = int(round(n * rate))
                case = build_case(
                    n, k, args.samples, snr, args.seed,
                    noise_scale=NOISE_SCALE, llr_sigma="nominal",
                )
                input_path = workdir / f"case{index}.in"
                write_input(case, input_path)
                base_seconds = run_binary(args.baseline_binary, input_path, workdir / f"case{index}.base.out")
                cand_seconds = run_binary(args.candidate_binary, input_path, workdir / f"case{index}.cand.out")
                ratio = cand_seconds / max(base_seconds, 1e-9)
                estimate = judge_times[index] * ratio
                entry = {
                    "case": index,
                    "n": n,
                    "k": k,
                    "rate": rate,
                    "v18_judge_seconds": round(judge_times[index], 3),
                    "v18_local_seconds": round(base_seconds, 3),
                    "candidate_local_seconds": round(cand_seconds, 3),
                    "local_ratio": round(ratio, 4),
                    "judge_estimate_seconds": round(estimate, 3),
                    "pass": estimate <= args.max_judge_seconds,
                }
                cases.append(entry)
                print(
                    f"case {index:>2d}: v18 judge {judge_times[index]:.3f}s, "
                    f"local ratio {ratio:.3f}x, estimate {estimate:.3f}s "
                    f"({'ok' if entry['pass'] else 'FAIL'})",
                    flush=True,
                )

    worst = max(cases, key=lambda item: item["judge_estimate_seconds"])
    result = {
        "samples": args.samples,
        "seed": args.seed,
        "max_judge_seconds": args.max_judge_seconds,
        "worst": worst,
        "cases": cases,
        "all_passed": all(item["pass"] for item in cases),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}")
    print(f"all_passed={result['all_passed']}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
