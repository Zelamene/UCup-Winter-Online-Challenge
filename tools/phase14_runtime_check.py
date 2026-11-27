#!/usr/bin/env python3
"""Phase 14 runtime guard: v17 must stay below 3x v16 on every case."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import run_binary, write_input  # noqa: E402
from real_bench import N_VALUES, RATE_SNR_DB, build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
NOISE_SCALE = 1.001120
MAX_RATIO = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-binary", type=Path, required=True)
    parser.add_argument("--v17-binary", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "phase14" / "runtime_check.json",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="runtime14_", dir=args.output.parent) as tmp_name:
        workdir = Path(tmp_name)
        case_index = 0
        for n in N_VALUES:
            for rate, snr in RATE_SNR_DB:
                case_index += 1
                k = int(round(n * rate))
                case = build_case(
                    n, k, args.samples, snr, args.seed,
                    noise_scale=NOISE_SCALE, llr_sigma="nominal",
                )
                input_path = workdir / "runtime.in"
                write_input(case, input_path)
                v16_time = run_binary(args.v16_binary, input_path, workdir / "v16.out")
                v17_time = run_binary(args.v17_binary, input_path, workdir / "v17.out")
                ratio = v17_time / max(v16_time, 1e-9)
                entry = {
                    "case_index": case_index,
                    "n": n,
                    "k": k,
                    "rate": rate,
                    "snr": snr,
                    "v16_seconds": round(v16_time, 3),
                    "v17_seconds": round(v17_time, 3),
                    "ratio": round(ratio, 3),
                    "pass": ratio < MAX_RATIO,
                }
                results.append(entry)
                print(
                    f"case {case_index:>2d}: v16 {v16_time:.2f}s, "
                    f"v17 {v17_time:.2f}s, ratio {ratio:.2f}x "
                    f"({'ok' if ratio < MAX_RATIO else 'FAIL'})",
                    flush=True,
                )

    worst = max(results, key=lambda item: item["ratio"])
    payload = {
        "samples": args.samples,
        "seed": args.seed,
        "max_ratio": MAX_RATIO,
        "worst": worst,
        "all_passed": all(item["pass"] for item in results),
        "cases": results,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
