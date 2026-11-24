#!/usr/bin/env python3
"""Phase 0 harness: validate LDPC input/answer structure and establish baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_INPUT = """\
14 7 1 6.0
0 1 1 1 1 1 1 1 0 0 0 0 0 0
1 0 1 1 1 1 1 0 1 0 0 0 0 0
1 1 0 1 1 1 1 0 0 1 0 0 0 0
1 1 1 0 1 1 1 0 0 0 1 0 0 0
1 1 1 1 0 1 1 0 0 0 0 1 0 0
1 1 1 1 1 0 1 0 0 0 0 0 1 0
1 1 1 1 1 1 0 0 0 0 0 0 0 1
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
1 0 0 0 0 0 0 0 1 1 1 1 1 1
0 1 0 0 0 0 0 1 0 1 1 1 1 1
0 0 1 0 0 0 0 1 1 0 1 1 1 1
0 0 0 1 0 0 0 1 1 1 0 1 1 1
0 0 0 0 1 0 0 1 1 1 1 0 1 1
0 0 0 0 0 1 0 1 1 1 1 1 0 1
0 0 0 0 0 0 1 1 1 1 1 1 1 0
"""

SAMPLE_OUTPUT = "1 1 1 1 1 1 1 0 0 0 0 0 0 0\n"


@dataclass(frozen=True)
class Instance:
    n: int
    k: int
    b: int
    snr: float
    h: np.ndarray
    llr: np.ndarray
    g: np.ndarray
    trailing_nonempty_lines: int

    @property
    def m(self) -> int:
        return self.n - self.k


def _binary_row(line: str, expected: int, label: str) -> np.ndarray:
    values = np.fromstring(line, sep=" ", dtype=np.int16)
    if values.size != expected:
        raise ValueError(f"{label}: expected {expected} values, got {values.size}")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{label}: contains a value other than 0 or 1")
    return values.astype(np.int8)


def _float_row(line: str, expected: int, label: str) -> np.ndarray:
    values = np.fromstring(line, sep=" ", dtype=np.float64)
    if values.size != expected:
        raise ValueError(f"{label}: expected {expected} values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: contains a non-finite value")
    return values


def parse_instance(text: str, source: str) -> Instance:
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{source}: empty input")

    header = lines[0].split()
    if len(header) != 4:
        raise ValueError(f"{source}: header must contain 4 values, got {len(header)}")

    n, k, b = (int(header[i]) for i in range(3))
    snr = float(header[3])
    m = n - k
    if n <= 0 or k <= 0 or b <= 0 or m <= 0 or k > n:
        raise ValueError(f"{source}: invalid N/K/B combination: {n=}, {k=}, {b=}")

    expected_lines = 1 + m + b + k
    if len(lines) < expected_lines:
        raise ValueError(
            f"{source}: expected at least {expected_lines} lines, got {len(lines)}"
        )

    pos = 1
    h = np.empty((m, n), dtype=np.int8)
    for row in range(m):
        h[row] = _binary_row(lines[pos], n, f"{source} H row {row + 1}")
        pos += 1

    llr = np.empty((b, n), dtype=np.float64)
    for row in range(b):
        llr[row] = _float_row(lines[pos], n, f"{source} LLR row {row + 1}")
        pos += 1

    g = np.empty((k, n), dtype=np.int8)
    for row in range(k):
        g[row] = _binary_row(lines[pos], n, f"{source} G row {row + 1}")
        pos += 1

    trailing = sum(bool(line.strip()) for line in lines[pos:])
    return Instance(n, k, b, snr, h, llr, g, trailing)


def parse_answers(text: str, source: str, n: int, b: int) -> np.ndarray:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != b:
        raise ValueError(f"{source}: expected {b} answer lines, got {len(lines)}")

    answers = np.empty((b, n), dtype=np.int8)
    for row, line in enumerate(lines):
        answers[row] = _binary_row(line, n, f"{source} answer row {row + 1}")
    return answers


def _value_counts(values: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {int(value): int(count) for value, count in zip(unique, counts)}


def analyze_structure(instance: Instance, label: str) -> dict:
    h, g = instance.h, instance.g
    m, k = instance.m, instance.k

    h_is_parity_identity = np.array_equal(h[:, k:], np.eye(m, dtype=np.int8))
    g_is_systematic_identity = np.array_equal(g[:, :k], np.eye(k, dtype=np.int8))
    g_matches_h = np.array_equal(g[:, k:], h[:, :k].T)

    a_row_weights = h[:, :k].sum(axis=1)
    a_col_weights = h[:, :k].sum(axis=0)
    expected_a_col_weight = float(a_row_weights.mean() * m / k)

    return {
        "label": label,
        "n": instance.n,
        "k": instance.k,
        "m": m,
        "b": instance.b,
        "snr_db": instance.snr,
        "rate": instance.k / instance.n,
        "trailing_nonempty_input_lines": instance.trailing_nonempty_lines,
        "h_row_weight_counts": _value_counts(h.sum(axis=1)),
        "a_row_weight_counts": _value_counts(a_row_weights),
        "a_column_weight_counts": _value_counts(a_col_weights),
        "expected_a_column_weight": expected_a_col_weight,
        "h_equals_concat_a_identity": bool(h_is_parity_identity),
        "g_equals_concat_identity_a_transpose": bool(g_is_systematic_identity),
        "g_parity_matches_h_information": bool(g_matches_h),
    }


def analyze_answers(instance: Instance, answers: np.ndarray) -> dict:
    h, g, llr = instance.h, instance.g, instance.llr
    k = instance.k

    if answers.shape != llr.shape:
        raise ValueError(
            f"answer shape {answers.shape} does not match LLR shape {llr.shape}"
        )

    info = answers[:, :k]
    syndrome = (answers.astype(np.int16) @ h.T.astype(np.int16)) % 2
    regenerated = (info.astype(np.int16) @ g.astype(np.int16)) % 2

    hard_bits = (llr < 0).astype(np.int8)
    hard_errors = int(np.count_nonzero(hard_bits[:, :k] != info))
    hard_syndrome = (hard_bits.astype(np.int16) @ h.T.astype(np.int16)) % 2

    llr_zeros = int(np.count_nonzero(llr == 0))

    return {
        "answer_rows_satisfying_all_checks": int(np.count_nonzero(syndrome.sum(axis=1) == 0)),
        "answer_rows_failing_any_check": int(np.count_nonzero(syndrome.sum(axis=1) > 0)),
        "answer_syndrome_ones": int(syndrome.sum()),
        "answer_mismatches_against_uG": int(np.count_nonzero(regenerated != answers)),
        "hard_decision_errors_info_bits": hard_errors,
        "hard_decision_info_bits_total": int(info.size),
        "hard_decision_info_ber": hard_errors / info.size,
        "hard_decision_rows_with_any_check_failure": int(
            np.count_nonzero(hard_syndrome.sum(axis=1) > 0)
        ),
        "hard_decision_syndrome_ones": int(hard_syndrome.sum()),
        "llr_zero_count": llr_zeros,
        "llr_min": float(llr.min()),
        "llr_max": float(llr.max()),
        "llr_mean_absolute": float(np.mean(np.abs(llr))),
        "negative_llr_matches_bit_one_rate": float(
            np.mean(answers[llr < 0] == 1) if np.any(llr < 0) else float("nan")
        ),
        "positive_llr_matches_bit_zero_rate": float(
            np.mean(answers[llr > 0] == 0) if np.any(llr > 0) else float("nan")
        ),
    }


def analyze_sample() -> dict:
    instance = parse_instance(SAMPLE_INPUT, "PDF sample")
    expected = parse_answers(SAMPLE_OUTPUT, "PDF sample answer", instance.n, instance.b)
    hard = (instance.llr < 0).astype(np.int8)

    return {
        "parsed_header": {
            "n": instance.n,
            "k": instance.k,
            "b": instance.b,
            "snr_db": instance.snr,
        },
        "matches_pdf_sample_output": bool(np.array_equal(hard, expected)),
        "answer_satisfies_all_checks": bool(
            np.all((expected.astype(np.int16) @ instance.h.T.astype(np.int16)) % 2 == 0)
        ),
        "structure": analyze_structure(instance, "PDF sample"),
    }


def print_dict(title: str, values: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(json.dumps(values, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parent.parent / "Problem statements"
    parser.add_argument("--input", type=Path, default=default_dir / "publish.in")
    parser.add_argument("--answer", type=Path, default=default_dir / "publish.ans")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "phase0_summary.json",
    )
    args = parser.parse_args()

    sample = analyze_sample()
    print_dict("PDF SAMPLE", sample)

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )

    structure = analyze_structure(instance, args.input.name)
    metrics = analyze_answers(instance, answers)
    print_dict("PUBLIC DATA STRUCTURE", structure)
    print_dict("PUBLIC DATA BASELINES", metrics)

    summary = {
        "sample": sample,
        "public_structure": structure,
        "public_baselines": metrics,
        "checks_passed": {
            "sample_matches": sample["matches_pdf_sample_output"],
            "h_systematic_form": structure["h_equals_concat_a_identity"],
            "g_systematic_form": structure["g_equals_concat_identity_a_transpose"],
            "g_matches_h": structure["g_parity_matches_h_information"],
            "answers_valid": metrics["answer_rows_failing_any_check"] == 0
            and metrics["answer_mismatches_against_uG"] == 0,
        },
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSummary written to {args.summary_json}")

    if not all(summary["checks_passed"].values()):
        raise SystemExit("Phase 0 validation failed; inspect the summary above.")


if __name__ == "__main__":
    main()
