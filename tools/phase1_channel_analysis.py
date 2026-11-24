#!/usr/bin/env python3
"""Phase 1: characterize the LLR channel and error/parity-check structure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_answers, parse_instance  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"

ABS_LLR_EDGES = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, math.inf]
P1_EDGES = [
    0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40,
    0.60, 0.80, 0.90, 0.95, 0.97, 0.99, 1.0,
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    quantiles = np.quantile(values, probabilities)
    return {f"p{int(p * 100)}": float(q) for p, q in zip(probabilities, quantiles)}


def conditional_llr_stats(llr: np.ndarray, truth: np.ndarray) -> dict:
    r0 = llr[truth == 0]
    r1 = llr[truth == 1]
    margin = (1 - 2 * truth) * llr

    return {
        "samples": int(truth.size),
        "bit_zero_fraction": float(np.mean(truth == 0)),
        "bit_zero_mean": float(r0.mean()),
        "bit_zero_std": float(r0.std()),
        "bit_zero_quantiles": _quantiles(r0),
        "bit_one_mean": float(r1.mean()),
        "bit_one_std": float(r1.std()),
        "bit_one_quantiles": _quantiles(r1),
        "signed_margin_mean": float(margin.mean()),
        "signed_margin_std": float(margin.std()),
        "signed_margin_quantiles": _quantiles(margin),
        "sign_correct_fraction": float(np.mean(margin > 0)),
    }


def error_by_abs_llr(llr: np.ndarray, truth: np.ndarray) -> list[dict]:
    hard_error = ((llr < 0).astype(np.int8) != truth).astype(np.int8)
    abs_llr = np.abs(llr)
    rows = []

    for low, high in zip(ABS_LLR_EDGES[:-1], ABS_LLR_EDGES[1:]):
        mask = (abs_llr >= low) & (abs_llr < high)
        count = int(np.count_nonzero(mask))
        errors = int(hard_error[mask].sum())
        rows.append({
            "low": low,
            "high": high,
            "count": count,
            "error_count": errors,
            "error_rate": errors / count if count else float("nan"),
            "mean_abs_llr": float(abs_llr[mask].mean()) if count else float("nan"),
            "fraction_of_all_errors": errors / int(hard_error.sum()) if hard_error.sum() else float("nan"),
        })
    return rows


def cumulative_error_by_threshold(llr: np.ndarray, truth: np.ndarray) -> list[dict]:
    hard_error = ((llr < 0).astype(np.int8) != truth)
    abs_llr = np.abs(llr)
    total_errors = int(hard_error.sum())
    rows = []

    for threshold in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]:
        low_confidence = abs_llr < threshold
        errors = int(np.count_nonzero(hard_error & low_confidence))
        rows.append({
            "threshold": threshold,
            "low_confidence_fraction": float(low_confidence.mean()),
            "errors_in_region": errors,
            "fraction_of_errors_captured": errors / total_errors if total_errors else float("nan"),
        })
    return rows


def cross_entropy(alpha: float, llr: np.ndarray, truth: np.ndarray) -> float:
    # p(bit=1 | r) = sigmoid(-alpha*r) because positive r favors bit 0.
    return float(np.mean(np.logaddexp(0.0, -alpha * llr) + truth * alpha * llr))


def fit_llr_scale(llr: np.ndarray, truth: np.ndarray) -> dict:
    coarse = np.arange(0.05, 3.001, 0.05)
    coarse_scores = np.array([cross_entropy(a, llr, truth) for a in coarse])
    best_coarse = float(coarse[int(np.argmin(coarse_scores))])

    fine = np.arange(max(0.01, best_coarse - 0.05), best_coarse + 0.0501, 0.005)
    fine_scores = np.array([cross_entropy(a, llr, truth) for a in fine])
    best_alpha = float(fine[int(np.argmin(fine_scores))])

    return {
        "baseline_alpha": 1.0,
        "baseline_cross_entropy": cross_entropy(1.0, llr, truth),
        "best_alpha": best_alpha,
        "best_cross_entropy": cross_entropy(best_alpha, llr, truth),
        "interpretation": (
            "p(bit=1|r)=sigmoid(-alpha*r); alpha>1 means raw LLRs are under-confident, "
            "alpha<1 means raw LLRs are over-confident"
        ),
    }


def reliability_table(llr: np.ndarray, truth: np.ndarray, alpha: float) -> list[dict]:
    p_one = sigmoid(-alpha * llr)
    rows = []

    for low, high in zip(P1_EDGES[:-1], P1_EDGES[1:]):
        mask = (p_one >= low) & (p_one < high)
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        rows.append({
            "low": low,
            "high": high,
            "count": count,
            "mean_predicted_p_one": float(p_one[mask].mean()),
            "observed_p_one": float(truth[mask].mean()),
            "gap": float(truth[mask].mean() - p_one[mask].mean()),
        })
    return rows


def initialization_stats(llr: np.ndarray, truth: np.ndarray) -> dict:
    sigma_true = (1 - 2 * truth).astype(np.float64)
    x_tanh = np.tanh(llr / 2.0)

    stats = {
        "tanh": [],
        "clip": [],
    }

    for scale in [1.0, 2.0, 4.0, 8.0]:
        x = np.tanh(llr / scale)
        stats["tanh"].append({
            "scale": scale,
            "alignment": float(np.mean(x * sigma_true)),
            "mean_for_sigma_plus_one": float(x[sigma_true == 1].mean()),
            "mean_for_sigma_minus_one": float(x[sigma_true == -1].mean()),
            "mean_absolute": float(np.abs(x).mean()),
            "fraction_near_saturation": float(np.mean(np.abs(x) > 0.99)),
        })

    scales = [5.0, 10.0, 15.0, 20.0, 30.0, float(np.median(np.abs(llr)))]
    seen = set()
    for scale in scales:
        key = round(scale, 6)
        if key in seen or scale <= 0:
            continue
        seen.add(key)
        x_clip = np.clip(llr / scale, -1.0, 1.0)
        stats["clip"].append({
            "scale": float(scale),
            "alignment": float(np.mean(x_clip * sigma_true)),
            "fraction_at_minus_one": float(np.mean(x_clip == -1.0)),
            "fraction_at_plus_one": float(np.mean(x_clip == 1.0)),
            "mean_absolute": float(np.abs(x_clip).mean()),
        })
    return stats


def parity_error_structure(h: np.ndarray, hard_bits: np.ndarray, answers: np.ndarray) -> dict:
    errors = (hard_bits != answers).astype(np.int8)
    errors_per_check = errors.astype(np.int16) @ h.T.astype(np.int16)
    syndrome = errors_per_check % 2

    # A hard-decision error is "covered" if at least one incident check has odd parity.
    covered = np.zeros(errors.shape, dtype=bool)
    for check in range(h.shape[0]):
        involved_bits = np.flatnonzero(h[check])
        covered[:, involved_bits] |= syndrome[:, check, None].astype(bool)

    info_errors = errors[:, : answers.shape[1] - h.shape[0]]
    parity_errors = errors[:, answers.shape[1] - h.shape[0]:]
    info_covered = covered[:, :info_errors.shape[1]]
    parity_covered = covered[:, info_errors.shape[1]:]

    return {
        "errors_per_check_counts": {
            int(k): int(v) for k, v in zip(*np.unique(errors_per_check, return_counts=True))
        },
        "errors_per_check_odd_counts": {
            int(k): int(v) for k, v in zip(*np.unique(errors_per_check[syndrome == 1], return_counts=True))
        },
        "errors_per_check_even_counts": {
            int(k): int(v) for k, v in zip(*np.unique(errors_per_check[syndrome == 0], return_counts=True))
        },
        "violations_per_codeword_min": int(syndrome.sum(axis=1).min()),
        "violations_per_codeword_max": int(syndrome.sum(axis=1).max()),
        "violations_per_codeword_mean": float(syndrome.sum(axis=1).mean()),
        "total_info_errors": int(info_errors.sum()),
        "info_errors_covered_by_violated_check": int((info_errors.astype(bool) & info_covered).sum()),
        "total_parity_errors": int(parity_errors.sum()),
        "parity_errors_covered_by_violated_check": int((parity_errors.astype(bool) & parity_covered).sum()),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    widths = {
        column: max(len(column), *(len(format_cell(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.rjust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join(format_cell(row[column]).rjust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )

    info_llr = instance.llr[:, : instance.k]
    info_truth = answers[:, : instance.k]
    hard_bits = (instance.llr < 0).astype(np.int8)

    conditional = conditional_llr_stats(info_llr, info_truth)
    error_rows = error_by_abs_llr(info_llr, info_truth)
    cumulative_rows = cumulative_error_by_threshold(info_llr, info_truth)
    scale_fit = fit_llr_scale(info_llr, info_truth)
    reliability = reliability_table(info_llr, info_truth, scale_fit["best_alpha"])
    init_stats = initialization_stats(info_llr, info_truth)
    parity_stats = parity_error_structure(instance.h, hard_bits, answers)

    print_table(
        "HARD-DECISION ERROR RATE BY |LLR|",
        error_rows,
        ["low", "high", "count", "error_count", "error_rate", "fraction_of_all_errors"],
    )
    print_table(
        "LOW-CONFIDENCE ERROR CAPTURE",
        cumulative_rows,
        ["threshold", "low_confidence_fraction", "errors_in_region", "fraction_of_errors_captured"],
    )
    print_table(
        "LLR RELIABILITY AFTER SCALE FIT",
        reliability,
        ["low", "high", "count", "mean_predicted_p_one", "observed_p_one", "gap"],
    )
    print("\nCONDITIONAL LLR STATISTICS")
    print(json.dumps(conditional, indent=2, sort_keys=True))
    print("\nLLR SCALE FIT")
    print(json.dumps(scale_fit, indent=2, sort_keys=True))
    print("\nINITIALIZATION STATISTICS")
    print(json.dumps(init_stats, indent=2, sort_keys=True))
    print("\nPARITY/ERROR STRUCTURE FOR HARD DECISION")
    print(json.dumps(parity_stats, indent=2, sort_keys=True))

    error_rates = [row["error_rate"] for row in error_rows if row["count"] > 0]
    monotonic_decrease = all(
        low >= high - 0.005 for low, high in zip(error_rates[:-1], error_rates[1:])
    )

    gate = {
        "more_than_90_percent_llr_signs_correct": conditional["sign_correct_fraction"] > 0.90,
        "error_rate_decreases_with_confidence": monotonic_decrease,
        "tanh_initialization_aligned_with_truth": max(
            row["alignment"] for row in init_stats["tanh"]
        ) > 0.5,
        "most_hard_errors_covered_by_violated_checks": (
            parity_stats["info_errors_covered_by_violated_check"]
            / max(1, parity_stats["total_info_errors"])
            > 0.5
        ),
    }
    print("\nPHASE 1 GATE")
    print(json.dumps(gate, indent=2, sort_keys=True))

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "conditional_llr": conditional,
        "error_by_abs_llr": error_rows,
        "cumulative_error_by_threshold": cumulative_rows,
        "llr_scale_fit": scale_fit,
        "reliability": reliability,
        "initialization": init_stats,
        "parity_error_structure": parity_stats,
        "gate": gate,
    }
    (args.outdir / "phase1_channel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        args.outdir / "phase1_error_by_abs_llr.csv",
        error_rows,
        ["low", "high", "count", "error_count", "error_rate", "mean_abs_llr", "fraction_of_all_errors"],
    )
    write_csv(
        args.outdir / "phase1_llr_reliability.csv",
        reliability,
        ["low", "high", "count", "mean_predicted_p_one", "observed_p_one", "gap"],
    )
    print(f"\nArtifacts written to {args.outdir}")

    if not all(gate.values()):
        raise SystemExit("Phase 1 gate failed; inspect the channel summary.")


if __name__ == "__main__":
    main()
