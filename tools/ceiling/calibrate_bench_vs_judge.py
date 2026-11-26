# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Calibrate fixed-rule bench scores against accepted v15 judge scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CEILING_DIR = Path(__file__).resolve().parent
if str(CEILING_DIR) not in sys.path:
    sys.path.insert(0, str(CEILING_DIR))

from common import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CASE_SPECS,
    FRAMES,
    NOISE_SCALE,
    OFFLINE_HEADER,
    ROOT,
    SEEDS,
    check_offline_header,
    fit_linear,
    load_ceiling_cache,
    resolve_path,
    score_bootstrap_ci,
    score_from_errors,
    write_json,
)

DEFAULT_CACHE_DIR = ROOT / "artifacts" / "baseline_cache_v15"
DEFAULT_JUDGE = ROOT / "submissions" / "v15_tap_bsb_candidate" / "judge_result.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase13" / "bench_vs_judge_v15.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    check_offline_header(Path(__file__))
    judge = json.loads(resolve_path(args.judge).read_text(encoding="utf-8"))
    judge_scores = judge.get("per_case")
    if not isinstance(judge_scores, list) or len(judge_scores) != len(CASE_SPECS):
        raise SystemExit("judge_result.json must contain 12 per_case scores")

    rows = []
    bench_scores = []
    judge_values = []
    binary_hashes = set()
    for index, ((n, k, snr), judge_score) in enumerate(
        zip(CASE_SPECS, judge_scores), start=1
    ):
        frame_errors = []
        for seed in SEEDS:
            errors, metadata = load_ceiling_cache(args.cache_dir, n, k, seed, FRAMES)
            frame_errors.append(errors)
            binary_hashes.add(metadata.get("binary_sha256"))
        combined = np.concatenate(frame_errors)
        bits_per_frame = k
        total_errors = int(combined.sum())
        bench_score = score_from_errors(combined, bits_per_frame)
        bootstrap = score_bootstrap_ci(
            combined,
            bits_per_frame,
            replicates=args.replicates,
            seed=args.bootstrap_seed + index,
        )
        bench_scores.append(bench_score)
        judge_values.append(float(judge_score))
        rows.append(
            {
                "index": index,
                "n": n,
                "k": k,
                "snr": snr,
                "rate": round(k / n, 6),
                "frames_per_seed": FRAMES,
                "seeds": list(SEEDS),
                "total_frames": len(combined),
                "total_errors": total_errors,
                "bench_score": bench_score,
                "bench_ci_low": bootstrap["ci_low"],
                "bench_ci_high": bootstrap["ci_high"],
                "judge_score": int(judge_score),
                "judge_minus_bench": float(judge_score) - bench_score,
                "judge_over_bench": float(judge_score) / bench_score,
                "judge_delta_vs_v14": judge.get("per_case_delta_vs_v14", [None] * 12)[
                    index - 1
                ],
            }
        )

    x = np.asarray(bench_scores, dtype=np.float64)
    y = np.asarray(judge_values, dtype=np.float64)
    linear = fit_linear(x, y)
    ratios = np.asarray([row["judge_over_bench"] for row in rows], dtype=np.float64)
    offsets = np.asarray([row["judge_minus_bench"] for row in rows], dtype=np.float64)

    payload = {
        "version": "v15",
        "rules": {
            "frames_per_case_seed": FRAMES,
            "noise_scale": NOISE_SCALE,
            "seeds": list(SEEDS),
            "source": "real_bench.build_case",
        },
        "score_scale": 100_000.0,
        "score_formula": "-100000*log10(BER)",
        "cache_binary_hashes": sorted(str(value) for value in binary_hashes),
        "bench_total": float(x.sum()),
        "judge_total": int(judge["score"]),
        "judge_total_minus_bench_total": float(judge["score"]) - float(x.sum()),
        "calibration": {
            "judge_equals_bench_linear": linear,
            "ratio_mean": float(ratios.mean()),
            "ratio_median": float(np.median(ratios)),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "offset_mean": float(offsets.mean()),
            "offset_median": float(np.median(offsets)),
            "offset_rmse": float(np.sqrt(np.mean(offsets**2))),
        },
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.bootstrap_seed,
            "method": "paired frame bootstrap per case, pooled over fixed seeds",
        },
        "offline_header": OFFLINE_HEADER,
        "cases": rows,
    }
    output = write_json(args.out, payload)

    print("Phase 13 bench-vs-judge calibration (v15)")
    print(f"{'case':>4} {'N':>5} {'rate':>5} {'bench':>10} {'CI low':>10} {'CI high':>10} {'judge':>9} {'diff':>9}")
    for row in rows:
        print(
            f"{row['index']:>4} {row['n']:>5} {row['rate']:>5.3f} "
            f"{row['bench_score']:>10,.0f} {row['bench_ci_low']:>10,.0f} "
            f"{row['bench_ci_high']:>10,.0f} {row['judge_score']:>9,} "
            f"{row['judge_minus_bench']:>9,.0f}"
        )
    print(
        f"TOTAL bench={payload['bench_total']:,.0f} judge={payload['judge_total']:,} "
        f"delta={payload['judge_total_minus_bench_total']:,.0f}"
    )
    print(
        "linear judge=slope*bench+intercept: "
        f"slope={linear['slope']:.6f} intercept={linear['intercept']:,.1f} "
        f"rmse={linear['rmse']:,.1f}"
    )
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
