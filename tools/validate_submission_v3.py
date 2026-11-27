#!/usr/bin/env python3
"""Validate a v3 QAIA manifest and generated source."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODES = {"dsb", "bsb"}
COMBINES = {"energy", "vote", "soft_vote"}

# Phase 13 ceiling analysis is offline-only.  Reject generated sources that
# mention any of the isolation markers or helper symbols from tools/ceiling.
OFFLINE_CEILING_SYMBOLS = (
    "OFFLINE ANALYSIS ONLY",
    "tools.ceiling",
    "tools/ceiling",
    "ceiling.common",
    "ceiling.cache_binary",
    "ceiling.calibrate_bench_vs_judge",
    "ceiling.compare_ceiling_caches",
    "phase13_ceiling",
    "paired_bootstrap_ci",
    "paired_bootstrap_total_ci",
    "build_ceiling_case",
    "load_ceiling_cache",
    "calibrate_bench_vs_judge",
    "compare_ceiling_caches",
)


def validate_case(case: dict, label: str) -> tuple[dict, list[str]]:
    errors: list[str] = []
    trajectories = case.get("trajectories", [])
    if not isinstance(trajectories, list):
        errors.append(f"{label}: trajectories must be a list")
        trajectories = []
    if not 1 <= len(trajectories) <= 6:
        errors.append(f"{label}: expected 1..6 trajectories, got {len(trajectories)}")
    if case.get("steps", 50) != 50:
        errors.append(f"{label}: steps must be 50")
    if case.get("mode") not in MODES:
        errors.append(f"{label}: invalid mode {case.get('mode')!r}")
    if case.get("combine") not in COMBINES:
        errors.append(f"{label}: invalid combine {case.get('combine')!r}")

    dt_value = case.get("dt", 0.2)
    if isinstance(dt_value, (list, tuple)):
        if len(dt_value) != 50:
            errors.append(f"{label}: dt must have 50 values")
        elif any(
            not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0
            for item in dt_value
        ):
            errors.append(f"{label}: dt has non-positive values")
    elif not isinstance(dt_value, (int, float)) or not math.isfinite(dt_value) or dt_value <= 0:
        errors.append(f"{label}: invalid dt={dt_value!r}")

    for index, traj in enumerate(trajectories):
        for key in ("tau", "xi", "lambd", "delta"):
            value = traj.get(key)
            if isinstance(value, (list, tuple)):
                if len(value) != 50:
                    errors.append(f"{label}: trajectory {index} {key} must have 50 values")
                elif any(
                    not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0
                    for item in value
                ):
                    errors.append(f"{label}: trajectory {index} {key} has non-positive values")
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"{label}: trajectory {index} has invalid {key}={value!r}")
        if "beta" in traj:
            value = traj["beta"]
            if isinstance(value, (list, tuple)):
                if len(value) != 50:
                    errors.append(f"{label}: trajectory {index} beta must have 50 values")
                elif any(
                    not isinstance(item, (int, float)) or not math.isfinite(item)
                    for item in value
                ):
                    errors.append(f"{label}: trajectory {index} beta has non-finite values")
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{label}: trajectory {index} invalid beta={value!r}")
        if "dt" in traj:
            value = traj["dt"]
            if isinstance(value, (list, tuple)):
                if len(value) != 50:
                    errors.append(f"{label}: trajectory {index} dt must have 50 values")
                elif any(
                    not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0
                    for item in value
                ):
                    errors.append(f"{label}: trajectory {index} dt has non-positive values")
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"{label}: trajectory {index} invalid dt={value!r}")
        if "joint_pairs" in traj and traj["joint_pairs"] not in (0, 1):
            errors.append(f"{label}: trajectory {index} joint_pairs must be 0 or 1")
        if int(traj.get("joint_pairs", 0)):
            if case.get("mode") != "bsb":
                errors.append(f"{label}: trajectory {index} joint_pairs requires bsb mode")
            if int(case.get("n", 0)) % 4 != 0:
                errors.append(f"{label}: trajectory {index} joint_pairs requires n divisible by 4")
        for pair_key in ("pair_gamma", "pair_scale"):
            if pair_key in traj:
                value = traj[pair_key]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"{label}: trajectory {index} invalid {pair_key}={value!r}")
        energy_lambda = traj.get("energy_lambda")
        if case.get("combine") == "energy":
            if not isinstance(energy_lambda, (int, float)) or energy_lambda < 0:
                errors.append(
                    f"{label}: trajectory {index} needs positive energy_lambda for energy combine"
                )

    return {
        "n": case.get("n"),
        "k": case.get("k"),
        "snr": case.get("snr"),
        "mode": case.get("mode"),
        "combine": case.get("combine"),
        "trajectory_count": len(trajectories),
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = args.submission.read_bytes()
    cases = manifest.get("cases", [])
    fallback = manifest.get("fallback")

    errors: list[str] = []
    case_summaries: list[dict] = []
    if fallback is not None:
        summary, errs = validate_case(fallback, "fallback")
        case_summaries.append(summary)
        errors.extend(errs)
    if not 1 <= len(cases) <= 12:
        errors.append(f"expected 1..12 cases, got {len(cases)}")
    for index, case in enumerate(cases, start=1):
        summary, errs = validate_case(case, f"case {index}")
        case_summaries.append(summary)
        errors.extend(errs)

    forbidden = ("belief propagation", "min-sum", "minsum", "bit-flip", "bitflip")
    lowered = source.decode("utf-8", errors="ignore").lower()
    found_forbidden = [token for token in forbidden if token in lowered]
    found_offline = [
        token for token in OFFLINE_CEILING_SYMBOLS if token.lower() in lowered
    ]

    public_summary = None
    if args.public_output is not None:
        text = args.public_output.read_text(encoding="utf-8")
        bad_bits = sorted({token for line in text.splitlines() for token in line.split()} - {"0", "1"})
        if bad_bits:
            errors.append(f"public output contains non-binary tokens: {bad_bits[:5]}")
        public_summary = {
            "lines": len(text.splitlines()),
            "all_binary": not bad_bits,
        }

    max_trajectories = max((item["trajectory_count"] for item in case_summaries), default=0)
    gate = {
        "source_under_512k": len(source) < 512 * 1024,
        "at_most_twelve_cases": len(cases) <= 12,
        "at_most_six_trajectories": max_trajectories <= 6,
        "fifty_steps": all(case.get("steps", 50) == 50 for case in [fallback] + cases if case),
        "valid_modes": all(case.get("mode") in MODES for case in [fallback] + cases if case),
        "valid_combines": all(case.get("combine") in COMBINES for case in [fallback] + cases if case),
        "no_forbidden_tokens": not found_forbidden,
        "no_offline_ceiling_symbols": not found_offline,
    }
    if not all(gate.values()):
        errors.append("one or more gates failed")

    result = {
        "source_bytes": len(source),
        "case_count": len(cases),
        "max_trajectory_count": max_trajectories,
        "cases": case_summaries,
        "public_output": public_summary,
        "forbidden_tokens": found_forbidden,
        "offline_ceiling_symbols": found_offline,
        "gate": gate,
        "all_passed": not errors,
        "errors": errors,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
