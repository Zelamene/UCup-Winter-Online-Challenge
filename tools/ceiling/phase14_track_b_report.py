# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Build the Phase 14 Track B joint-likelihood ceiling report (offline only).

Loads fixed-rule per-frame error caches and compares per-bit BP, exact joint
Gray-pair BP, optional BP-guided OSD refinements, and the accepted v16 cache
with paired frame bootstraps.  No submission or judge process is touched.
"""

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
    load_ceiling_cache,
    paired_bootstrap_ci,
    paired_bootstrap_total_ci,
    resolve_path,
    score_from_errors,
    write_json,
)

ACCEPTED_V16_JUDGE = 2_023_997
LIVE_TOP_SCORE = 2_517_863
DEFAULT_SLOPE = 0.9470002758290964
TRIGGER_N, TRIGGER_K = 3888, 2592


def load_case_errors(
    cache_dir: Path | None,
    n: int,
    k: int,
    frames: int,
    seeds: tuple[int, ...] | list[int] = SEEDS,
) -> tuple[np.ndarray | None, list[dict]]:
    """Load and concatenate the requested fixed seeds from one cache directory."""
    if cache_dir is None or not cache_dir.exists():
        return None, []
    arrays: list[np.ndarray] = []
    metadata: list[dict] = []
    try:
        for seed in seeds:
            errors, meta = load_ceiling_cache(cache_dir, n, k, seed, frames)
            arrays.append(errors)
            metadata.append(meta)
    except (FileNotFoundError, ValueError):
        return None, []
    return np.concatenate(arrays), metadata


def ci(
    baseline: np.ndarray | None,
    candidate: np.ndarray | None,
    k: int,
    replicates: int,
    seed: int,
) -> dict | None:
    """Paired bootstrap delta with a deterministic per-case seed."""
    if baseline is None or candidate is None:
        return None
    return paired_bootstrap_ci(
        baseline, candidate, k, replicates=replicates, seed=seed
    )


def manifest_scores(path: Path) -> dict[tuple[int, int, str], float]:
    """Read per-case scores from a bp_variants manifest, if it exists."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[int, int, str], float] = {}
    for row in payload.get("rows", []):
        key = (int(row["n"]), int(row["k"]), str(row.get("variant", "")))
        result[key] = float(row["score"])
    return result


def calibration_slope() -> tuple[float, str]:
    """Use the Phase 13 measured bench-to-judge delta slope when available."""
    path = ROOT / "artifacts" / "phase13" / "bench_vs_judge_v15.json"
    if not path.exists():
        return DEFAULT_SLOPE, "fallback constant"
    payload = json.loads(path.read_text(encoding="utf-8"))
    slope = float(
        payload.get("calibration", {})
        .get("judge_equals_bench_linear", {})
        .get("slope", DEFAULT_SLOPE)
    )
    return slope, str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument(
        "--perbit-dir", type=Path, default=ROOT / "artifacts" / "ceiling_cache_bp_perbit"
    )
    parser.add_argument(
        "--joint-dir", type=Path, default=ROOT / "artifacts" / "ceiling_cache_bp_joint"
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=ROOT / "artifacts" / "baseline_cache_v16"
    )
    parser.add_argument("--osd-joint-dir", type=Path, default=None)
    parser.add_argument("--osd-perbit-dir", type=Path, default=None)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "artifacts" / "phase14" / "track_b_ceiling_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "artifacts" / "phase14" / "track_b_ceiling_report.md",
    )
    args = parser.parse_args()

    check_offline_header(Path(__file__))
    from bp_variants import self_test

    factor_check = self_test()
    slope, slope_source = calibration_slope()
    screen_scores = manifest_scores(
        ROOT / "artifacts" / "phase14" / "track_b_bp_screen100_manifest.json"
    )
    screen_perbit_osd_scores = manifest_scores(
        ROOT / "artifacts" / "phase14" / "track_b_bp_perbit_osd2_f100_manifest.json"
    )
    screen_joint_osd_dir = ROOT / "artifacts" / "ceiling_cache_bp_joint_f100_osd2_lr64"
    screen_summary = []
    for n, k, _snr in CASE_SPECS:
        perbit_score = screen_scores.get((n, k, "perbit"))
        joint_score = screen_scores.get((n, k, "joint"))
        perbit_osd_score = screen_perbit_osd_scores.get((n, k, "perbit"))
        joint_osd_errors, _ = load_case_errors(
            screen_joint_osd_dir, n, k, 100, seeds=(SEEDS[0],)
        )
        joint_osd_score = (
            score_from_errors(joint_osd_errors, k)
            if joint_osd_errors is not None
            else None
        )
        if perbit_score is None or joint_score is None:
            continue
        screen_summary.append(
            {
                "n": n,
                "k": k,
                "perbit_score": perbit_score,
                "joint_score": joint_score,
                "joint_gain": joint_score - perbit_score,
                "perbit_osd2_score": perbit_osd_score,
                "perbit_osd2_gain": (
                    perbit_osd_score - perbit_score
                    if perbit_osd_score is not None
                    else None
                ),
                "osd2_score": joint_osd_score,
                "osd2_gain": (
                    joint_osd_score - joint_score if joint_osd_score is not None else None
                ),
            }
        )
    osd_screen_complete = bool(screen_summary) and all(
        row["perbit_osd2_score"] is not None and row["osd2_score"] is not None
        for row in screen_summary
    )
    perbit_dir = resolve_path(args.perbit_dir)
    joint_dir = resolve_path(args.joint_dir)
    baseline_dir = resolve_path(args.baseline_dir)
    osd_joint_dir = resolve_path(args.osd_joint_dir) if args.osd_joint_dir else None
    osd_perbit_dir = resolve_path(args.osd_perbit_dir) if args.osd_perbit_dir else None

    case_rows: list[dict] = []
    total_groups_joint_vs_perbit: list[tuple[np.ndarray, np.ndarray, int]] = []
    total_groups_joint_vs_v16: list[tuple[np.ndarray, np.ndarray, int]] = []
    total_groups_best_vs_v16: list[tuple[np.ndarray, np.ndarray, int]] = []
    total_groups_osd_vs_joint: list[tuple[np.ndarray, np.ndarray, int]] = []

    for index, (n, k, snr) in enumerate(CASE_SPECS, start=1):
        base, _ = load_case_errors(baseline_dir, n, k, args.frames)
        if base is None:
            raise SystemExit(f"missing v16 baseline cache for N={n} K={k}")
        perbit, _ = load_case_errors(perbit_dir, n, k, args.frames)
        joint, _ = load_case_errors(joint_dir, n, k, args.frames)
        osd_joint, _ = load_case_errors(osd_joint_dir, n, k, args.frames)
        osd_perbit, _ = load_case_errors(osd_perbit_dir, n, k, args.frames)

        base_score = score_from_errors(base, k)
        perbit_score = score_from_errors(perbit, k) if perbit is not None else None
        joint_score = score_from_errors(joint, k) if joint is not None else None
        candidates = {"v16": base_score}
        if perbit_score is not None:
            candidates["perbit_bp"] = perbit_score
        if joint_score is not None:
            candidates["joint_bp"] = joint_score
        if osd_joint is not None:
            candidates["joint_bp_osd"] = score_from_errors(osd_joint, k)
        if osd_perbit is not None:
            candidates["perbit_bp_osd"] = score_from_errors(osd_perbit, k)
        best_source = max(candidates, key=candidates.get)
        best_score = candidates[best_source]

        seed_offset = index * 1000
        joint_vs_perbit = ci(
            perbit, joint, k, args.replicates, args.bootstrap_seed + seed_offset
        )
        joint_vs_base = ci(
            base, joint, k, args.replicates, args.bootstrap_seed + seed_offset + 100
        )
        perbit_vs_base = ci(
            base, perbit, k, args.replicates, args.bootstrap_seed + seed_offset + 200
        )
        osd_vs_joint = ci(
            joint, osd_joint, k, args.replicates, args.bootstrap_seed + seed_offset + 300
        )

        if joint is not None and perbit is not None:
            total_groups_joint_vs_perbit.append((perbit, joint, k))
        if joint is not None:
            total_groups_joint_vs_v16.append((base, joint, k))
            best_candidate = joint if best_source == "joint_bp" else (
                osd_joint if best_source == "joint_bp_osd" else (
                    perbit if best_source == "perbit_bp" else (
                        osd_perbit if best_source == "perbit_bp_osd" else base
                    )
                )
            )
            total_groups_best_vs_v16.append((base, best_candidate, k))
        if osd_joint is not None and joint is not None:
            total_groups_osd_vs_joint.append((joint, osd_joint, k))

        row = {
            "case_index": index,
            "n": n,
            "k": k,
            "rate": round(k / n, 6),
            "snr": snr,
            "measured_bp": joint is not None and perbit is not None,
            "frames_per_seed": args.frames,
            "total_frames": len(base),
            "v16_errors": int(base.sum()),
            "v16_score": float(base_score),
            "perbit_bp_score": float(perbit_score) if perbit_score is not None else None,
            "joint_bp_score": float(joint_score) if joint_score is not None else None,
            "joint_bp_osd_score": (
                float(score_from_errors(osd_joint, k)) if osd_joint is not None else None
            ),
            "perbit_bp_osd_score": (
                float(score_from_errors(osd_perbit, k)) if osd_perbit is not None else None
            ),
            "joint_vs_perbit": joint_vs_perbit,
            "joint_vs_v16": joint_vs_base,
            "perbit_vs_v16": perbit_vs_base,
            "osd_joint_vs_joint": osd_vs_joint,
            "best_score": float(best_score),
            "best_source": best_source,
        }
        case_rows.append(row)

    total_v16 = sum(row["v16_score"] for row in case_rows)
    total_best = sum(row["best_score"] for row in case_rows)
    bench_delta = total_best - total_v16
    judge_estimate = ACCEPTED_V16_JUDGE + slope * bench_delta
    gap_to_live = LIVE_TOP_SCORE - judge_estimate

    total_jvp = (
        paired_bootstrap_total_ci(
            total_groups_joint_vs_perbit,
            replicates=args.replicates,
            seed=args.bootstrap_seed,
        )
        if total_groups_joint_vs_perbit
        else None
    )
    total_jvv = (
        paired_bootstrap_total_ci(
            total_groups_joint_vs_v16,
            replicates=args.replicates,
            seed=args.bootstrap_seed + 1,
        )
        if total_groups_joint_vs_v16
        else None
    )
    total_best_ci = (
        paired_bootstrap_total_ci(
            total_groups_best_vs_v16,
            replicates=args.replicates,
            seed=args.bootstrap_seed + 2,
        )
        if total_groups_best_vs_v16
        else None
    )
    total_osd = (
        paired_bootstrap_total_ci(
            total_groups_osd_vs_joint,
            replicates=args.replicates,
            seed=args.bootstrap_seed + 3,
        )
        if total_groups_osd_vs_joint
        else None
    )

    trigger_row = next(
        row for row in case_rows if row["n"] == TRIGGER_N and row["k"] == TRIGGER_K
    )
    gain_vs_perbit = trigger_row["joint_vs_perbit"]
    gain_vs_v16 = trigger_row["joint_vs_v16"]
    measured = trigger_row["measured_bp"]
    trigger_pass = bool(
        measured
        and gain_vs_perbit is not None
        and gain_vs_perbit["point"] > 5000.0
        and gain_vs_perbit["ci_low"] > 0.0
        and gain_vs_v16 is not None
        and gain_vs_v16["point"] > 0.0
    )
    reasons: list[str] = []
    if not measured:
        reasons.append("N3888 R2/3 BP caches are incomplete")
    else:
        reasons.append(
            "joint-vs-perbit point "
            f"{gain_vs_perbit['point']:,.0f} (95% CI low {gain_vs_perbit['ci_low']:,.0f})"
        )
        reasons.append(
            "joint-vs-v16 point "
            f"{gain_vs_v16['point']:,.0f} (95% CI low {gain_vs_v16['ci_low']:,.0f})"
        )
    if not trigger_pass:
        reasons.append("required >5k joint gain with positive CI and v16 superiority not met")

    if judge_estimate >= LIVE_TOP_SCORE:
        verdict = (
            "Measured offline decoders reach or exceed the live top score, so the gap "
            "is largely explained by strong BP/OSD-style decoding."
        )
    else:
        verdict = (
            f"Measured offline decoders reach ~{judge_estimate:,.0f}, leaving "
            f"~{gap_to_live:,.0f} points to the live top score. These are achievable "
            "lower bounds, not a formal ceiling; the remaining gap is unmodelled "
            "by the BP/OSD variants tested here."
        )

    payload = {
        "rules": {
            "frames_per_case_seed": args.frames,
            "noise_scale": NOISE_SCALE,
            "seeds": list(SEEDS),
            "source": "real_bench.build_case via common.build_ceiling_case",
            "bootstrap_replicates": args.replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "caches": {
            "baseline_v16": str(baseline_dir),
            "perbit_bp": str(perbit_dir),
            "joint_bp": str(joint_dir),
            "osd_joint": str(osd_joint_dir) if osd_joint_dir else None,
            "osd_perbit": str(osd_perbit_dir) if osd_perbit_dir else None,
        },
        "pair_factor_self_test": factor_check,
        "cache_fingerprint_note": (
            "Cache metadata stores the source hash at generation time. CLI-only fixes "
            "were applied after long runs launched; decode behavior is unchanged, but "
            "--skip-existing may miss."
        ),
        "osd_screen_status": (
            "both_perbit_and_joint_osd2_screens_completed"
            if osd_screen_complete
            else "perbit_osd2_screen_unavailable_no_comparison"
        ),
        "screen_100f_seed2024": screen_summary,
        "calibration": {
            "baseline_note": "v16 is the accepted base; v17 remains unsubmitted and is not mixed in",
            "accepted_v16_judge_score": ACCEPTED_V16_JUDGE,
            "bench_to_judge_delta_slope": slope,
            "slope_source": slope_source,
            "live_top_score": LIVE_TOP_SCORE,
        },
        "totals": {
            "bench_v16_total": float(total_v16),
            "bench_best_total": float(total_best),
            "bench_best_delta": float(bench_delta),
            "judge_ceiling_estimate": float(judge_estimate),
            "gap_to_live_top": float(gap_to_live),
            "joint_vs_perbit": total_jvp,
            "joint_vs_v16": total_jvv,
            "best_vs_v16": total_best_ci,
            "osd_joint_vs_joint": total_osd,
        },
        "v18_trigger": {
            "case": {"n": TRIGGER_N, "k": TRIGGER_K, "rate": 2.0 / 3.0},
            "threshold_joint_gain": 5000,
            "joint_gain_vs_perbit": gain_vs_perbit,
            "joint_gain_vs_v16": gain_vs_v16,
            "promote_v18": trigger_pass,
            "reasons": reasons,
        },
        "ceiling_assessment": verdict,
        "cases": case_rows,
        "offline_header": OFFLINE_HEADER,
    }
    json_path = write_json(args.out_json, payload)

    lines = [
        "# Phase 14 Track B — joint-likelihood ceiling ladder",
        "",
        "**Offline measuring stick only.** No exporter, submission, judge or gate is touched.",
        "",
        f"Rules: {args.frames} frames/seed, seeds {list(SEEDS)}, noise_scale {NOISE_SCALE}, "
        f"paired frame bootstrap ({args.replicates} replicates).",
        "",
        "## Quick screen (100 frames, seed 2024)",
        "",
        "| N | K | per-bit | joint | joint−perbit | per-bit OSD-2 | OSD-2−per-bit | joint OSD-2 | OSD-2−joint |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in screen_summary:
        perbit_osd_text = (
            f"{row['perbit_osd2_score']:,.0f}"
            if row["perbit_osd2_score"] is not None
            else "n/a"
        )
        perbit_osd_gain = (
            f"{row['perbit_osd2_gain']:+,.0f}"
            if row["perbit_osd2_gain"] is not None
            else "n/a"
        )
        joint_osd_text = (
            f"{row['osd2_score']:,.0f}" if row["osd2_score"] is not None else "n/a"
        )
        joint_osd_gain = (
            f"{row['osd2_gain']:+,.0f}" if row["osd2_gain"] is not None else "n/a"
        )
        lines.append(
            f"| {row['n']} | {row['k']} | {row['perbit_score']:,.0f} | "
            f"{row['joint_score']:,.0f} | {row['joint_gain']:+,.0f} | "
            f"{perbit_osd_text} | {perbit_osd_gain} | {joint_osd_text} | {joint_osd_gain} |"
        )
    if not osd_screen_complete:
        lines.append(
            "Per-bit OSD-2 screen unavailable; no per-bit-vs-joint OSD comparison is claimed."
        )
    lines.extend(
        [
            "",
            "## Per-case scores and joint gains",
            "",
            "| Case | N | K | Rate | v16 bench | per-bit BP | joint BP | joint−perbit (95% CI) | joint−v16 (95% CI) | best |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in case_rows:
        jvp = row["joint_vs_perbit"]
        jvv = row["joint_vs_v16"]
        gain_text = (
            f"{jvp['point']:+,.0f} [{jvp['ci_low']:+,.0f}, {jvp['ci_high']:+,.0f}]"
            if jvp
            else "n/a"
        )
        v16_text = (
            f"{jvv['point']:+,.0f} [{jvv['ci_low']:+,.0f}, {jvv['ci_high']:+,.0f}]"
            if jvv
            else "n/a"
        )
        perbit_text = f"{row['perbit_bp_score']:,.0f}" if row["perbit_bp_score"] else "n/a"
        joint_text = f"{row['joint_bp_score']:,.0f}" if row["joint_bp_score"] else "n/a"
        lines.append(
            f"| {row['case_index']} | {row['n']} | {row['k']} | {row['rate']:.3f} | "
            f"{row['v16_score']:,.0f} | {perbit_text} | {joint_text} | {gain_text} | "
            f"{v16_text} | {row['best_source']} {row['best_score']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Totals and ceiling estimate",
            "",
            f"- Accepted v16 judge score: {ACCEPTED_V16_JUDGE:,} (v17 is held unsubmitted "
            "and is not mixed into this baseline).",
            f"- Fixed-bench v16 total: {total_v16:,.0f}; best measured total: {total_best:,.0f}.",
            f"- Bench delta: {bench_delta:+,.0f}; judge-scaled delta (slope {slope:.6f}): "
            f"{slope * bench_delta:+,.0f}.",
            f"- Judge-scaled ceiling estimate: **{judge_estimate:,.0f}**.",
            f"- Live top {LIVE_TOP_SCORE:,}: remaining gap **{gap_to_live:,.0f}**.",
        ]
    )
    if total_jvp:
        lines.append(
            f"- Joint vs per-bit total delta over measured cases: "
            f"{total_jvp['point']:+,.0f} [{total_jvp['ci_low']:+,.0f}, {total_jvp['ci_high']:+,.0f}]."
        )
    if total_jvv:
        lines.append(
            f"- Joint vs v16 total delta over measured cases: "
            f"{total_jvv['point']:+,.0f} [{total_jvv['ci_low']:+,.0f}, {total_jvv['ci_high']:+,.0f}]."
        )
    if total_osd:
        lines.append(
            f"- Joint OSD vs plain joint total delta: "
            f"{total_osd['point']:+,.0f} [{total_osd['ci_low']:+,.0f}, {total_osd['ci_high']:+,.0f}]."
        )

    lines.extend(
        [
            "",
            "## v18 trigger decision (N=3888, R=2/3)",
            "",
            f"**promote_v18 = {trigger_pass}**.",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## Search methods attempted",
            "",
            "- Per-bit BP: original independent per-bit LLRs, sum-product check updates.",
            "- Joint BP: exact `tools/qam_pairs.py` pair tables with the `d*s0*s1` factor "
            "retained as a pairwise channel factor. Self-check: "
            f"`{factor_check['status']}` at SNR 6/9/10.5 dB.",
            "- BP-guided OSD: order 2 over the 64 least reliable information bits. "
            "Per-bit OSD scores candidates with per-bit LLRs; joint OSD uses the exact "
            "pair metric. Only 100-frame seed-2024 screens were run (side-by-side "
            "scores in the screen table, no OSD bootstrap CI); no 500-frame OSD ladder "
            "was attempted. Order 3 and list-MAP L=64–256 were not run, so no such "
            "ceiling is claimed.",
            "",
            "## Interpretation",
            "",
            verdict,
            "",
            "The exact pair factor uses `tools/qam_pairs.py` `(a, b, d)` including the "
            "`d*s0*s1` coupling; the per-bit variant uses the original independent LLRs.",
            "BP/OSD are achievable decoders, so these numbers are lower bounds on the "
            "true ML ceiling, not upper bounds.",
            "",
            f"Machine-readable report: `{json_path.relative_to(ROOT)}`.",
            "",
        ]
    )
    md_path = resolve_path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Bench best total {total_best:,.0f} vs v16 {total_v16:,.0f} ({bench_delta:+,.0f})")
    print(f"Judge-scaled ceiling estimate {judge_estimate:,.0f}; gap {gap_to_live:,.0f}")
    print(f"v18 promote={trigger_pass}: {'; '.join(reasons)}")
    print(f"Wrote {json_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
