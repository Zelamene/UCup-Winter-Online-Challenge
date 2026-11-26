# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Compare two offline-analysis caches with paired frame bootstrap CIs."""

from __future__ import annotations

import argparse
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
    load_ceiling_cache,
    paired_bootstrap_ci,
    paired_bootstrap_total_ci,
    resolve_path,
    score_from_errors,
    write_json,
)

DEFAULT_BASELINE = ROOT / "artifacts" / "baseline_cache_v15"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase13" / "cache_comparison.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--require-different-binary",
        action="store_true",
        help="refuse to compare caches produced by the same binary hash",
    )
    args = parser.parse_args()

    check_offline_header(Path(__file__))
    baseline_dir = resolve_path(args.baseline)
    candidate_dir = resolve_path(args.candidate)

    per_seed_rows = []
    case_rows = []
    bootstrap_groups: list[tuple[np.ndarray, np.ndarray, int]] = []
    baseline_hashes: set[str] = set()
    candidate_hashes: set[str] = set()

    for case_index, (n, k, snr) in enumerate(CASE_SPECS, start=1):
        baseline_frames: list[np.ndarray] = []
        candidate_frames: list[np.ndarray] = []
        for seed_index, seed in enumerate(SEEDS):
            base_errors, base_meta = load_ceiling_cache(baseline_dir, n, k, seed, FRAMES)
            cand_errors, cand_meta = load_ceiling_cache(candidate_dir, n, k, seed, FRAMES)
            baseline_hashes.add(str(base_meta.get("binary_sha256")))
            candidate_hashes.add(str(cand_meta.get("binary_sha256")))

            base_score = score_from_errors(base_errors, k)
            cand_score = score_from_errors(cand_errors, k)
            seed_ci = paired_bootstrap_ci(
                base_errors,
                cand_errors,
                k,
                replicates=args.replicates,
                seed=args.bootstrap_seed + case_index * 100 + seed_index,
            )
            per_seed_rows.append(
                {
                    "case_index": case_index,
                    "n": n,
                    "k": k,
                    "snr": snr,
                    "seed": seed,
                    "baseline_errors": int(base_errors.sum()),
                    "candidate_errors": int(cand_errors.sum()),
                    "baseline_score": base_score,
                    "candidate_score": cand_score,
                    "delta": seed_ci["point"],
                    "ci_low": seed_ci["ci_low"],
                    "ci_high": seed_ci["ci_high"],
                }
            )
            baseline_frames.append(base_errors)
            candidate_frames.append(cand_errors)
            bootstrap_groups.append((base_errors, cand_errors, k))

        combined_base = np.concatenate(baseline_frames)
        combined_cand = np.concatenate(candidate_frames)
        case_ci = paired_bootstrap_ci(
            combined_base,
            combined_cand,
            k,
            replicates=args.replicates,
            seed=args.bootstrap_seed + case_index,
        )
        case_rows.append(
            {
                "case_index": case_index,
                "n": n,
                "k": k,
                "snr": snr,
                "rate": round(k / n, 6),
                "frames": len(combined_base),
                "baseline_errors": int(combined_base.sum()),
                "candidate_errors": int(combined_cand.sum()),
                "baseline_score": score_from_errors(combined_base, k),
                "candidate_score": score_from_errors(combined_cand, k),
                "delta": case_ci["point"],
                "ci_low": case_ci["ci_low"],
                "ci_high": case_ci["ci_high"],
                "ci_above_zero": case_ci["ci_low"] > 0.0,
            }
        )

    if len(baseline_hashes) != 1 or len(candidate_hashes) != 1:
        raise SystemExit(
            "each cache directory must contain exactly one binary hash: "
            f"baseline={sorted(baseline_hashes)}, candidate={sorted(candidate_hashes)}"
        )
    baseline_hash = next(iter(baseline_hashes))
    candidate_hash = next(iter(candidate_hashes))
    if args.require_different_binary and baseline_hash == candidate_hash:
        raise SystemExit("candidate binary hash matches baseline; nothing to compare")

    total_ci = paired_bootstrap_total_ci(
        bootstrap_groups,
        replicates=args.replicates,
        seed=args.bootstrap_seed,
    )
    payload = {
        "labels": {"baseline": args.baseline_label, "candidate": args.candidate_label},
        "caches": {
            "baseline": str(baseline_dir),
            "candidate": str(candidate_dir),
            "baseline_binary_sha256": baseline_hash,
            "candidate_binary_sha256": candidate_hash,
        },
        "rules": {
            "frames_per_case_seed": FRAMES,
            "noise_scale": NOISE_SCALE,
            "seeds": list(SEEDS),
            "source": "real_bench.build_case",
        },
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.bootstrap_seed,
            "method": "paired frame bootstrap within each case-seed; pooled CIs resample every group",
        },
        "total": total_ci,
        "cases": case_rows,
        "per_seed": per_seed_rows,
        "offline_header": OFFLINE_HEADER,
    }
    output = write_json(args.out, payload)

    print(
        f"{args.baseline_label} -> {args.candidate_label} "
        f"({candidate_hash[:12]} vs {baseline_hash[:12]})"
    )
    print(f"{'case':>4} {'N':>5} {'rate':>5} {'errors':>10} {'delta':>10} {'CI low':>10} {'CI high':>10}")
    for row in case_rows:
        print(
            f"{row['case_index']:>4} {row['n']:>5} {row['rate']:>5.3f} "
            f"{row['candidate_errors'] - row['baseline_errors']:>10,} "
            f"{row['delta']:>10,.0f} {row['ci_low']:>10,.0f} {row['ci_high']:>10,.0f}"
        )
    print(
        f"TOTAL delta={total_ci['point']:,.0f} "
        f"95% CI=[{total_ci['ci_low']:,.0f}, {total_ci['ci_high']:,.0f}]"
    )
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
