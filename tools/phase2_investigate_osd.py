#!/usr/bin/env python3
"""Diagnostic-only ordered-statistics decoding reference.

This is not submission code. It estimates how much lower BER a near-ML
valid-codeword search can achieve on the public channel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_answers, parse_instance  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"


def row_reduce_columns(h: np.ndarray, column_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R and pivot column positions with R[:, pivots] in identity form."""
    m, n = h.shape
    r = h[:, column_order].astype(np.uint8).copy()
    u = np.eye(m, dtype=np.uint8)
    pivot_positions: list[int] = []
    used_columns = np.zeros(n, dtype=bool)

    for pivot_row in range(m):
        pivot_col = -1
        for candidate_col in range(n):
            if used_columns[candidate_col]:
                continue
            candidates = np.flatnonzero(r[pivot_row:, candidate_col])
            if candidates.size > 0:
                pivot_col = candidate_col
                source = pivot_row + int(candidates[0])
                break
        if pivot_col < 0:
            raise RuntimeError("Could not find a full-rank pivot set")

        if source != pivot_row:
            r[[pivot_row, source]] = r[[source, pivot_row]]
            u[[pivot_row, source]] = u[[source, pivot_row]]

        for row in range(m):
            if row != pivot_row and r[row, pivot_col]:
                r[row] ^= r[pivot_row]
                u[row] ^= u[pivot_row]
        pivot_positions.append(pivot_col)
        used_columns[pivot_col] = True

    return r, np.asarray(pivot_positions, dtype=np.int64)


def score_codeword(code: np.ndarray, llr: np.ndarray) -> float:
    return float(-np.dot(code, llr))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--osd2", action="store_true")
    parser.add_argument("--summary-name", default="phase2_osd_summary.json")
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )

    limit = instance.b if args.max_samples <= 0 else min(args.max_samples, instance.b)
    n, k, m = instance.n, instance.k, instance.m
    errors0 = errors1 = errors2 = 0
    scores0: list[float] = []
    scores1: list[float] = []
    scores2: list[float] = []
    start = time.perf_counter()

    for sample in range(limit):
        llr = instance.llr[sample]
        truth = answers[sample, :k]
        hard = (llr < 0).astype(np.uint8)

        # Least reliable columns become pivots; the most reliable K bits become
        # the information set.
        order = np.argsort(np.abs(llr), kind="stable")
        r_reduced, pivot_positions = row_reduce_columns(instance.h, order)
        free_positions = np.setdiff1d(np.arange(n), pivot_positions)
        pivot_original = order[pivot_positions]
        free_original = order[free_positions]

        free_bits = hard[free_original]
        pivot_bits = (r_reduced[:, free_positions] @ free_bits) % 2

        code0 = np.zeros(n, dtype=np.uint8)
        code0[free_original] = free_bits
        code0[pivot_original] = pivot_bits
        score0 = score_codeword(code0, llr)
        scores0.append(score0)
        errors0 += int(np.count_nonzero(code0[:k] != truth))

        # OSD-1: flip each information-set bit and recompute the score change.
        free_weights = (2 * free_bits.astype(np.float64) - 1) * llr[free_original]
        pivot_weights = (2 * pivot_bits.astype(np.float64) - 1) * llr[pivot_original]
        pivot_deltas = pivot_weights @ r_reduced[:, free_positions]
        candidate_scores = score0 + free_weights + pivot_deltas
        best_free = int(np.argmax(candidate_scores))
        scores1.append(float(candidate_scores[best_free]))

        bits1 = free_bits.copy()
        bits1[best_free] ^= 1
        pivots1 = pivot_bits ^ r_reduced[:, free_positions[best_free]]
        code1 = np.zeros(n, dtype=np.uint8)
        code1[free_original] = bits1
        code1[pivot_original] = pivots1
        errors1 += int(np.count_nonzero(code1[:k] != truth))

        if args.osd2:
            deltas = free_weights + pivot_deltas
            top_two = np.argpartition(deltas, -2)[-2:]
            single_score = float(np.max(candidate_scores))
            pair_score = score0 + float(np.sum(deltas[top_two]))

            if pair_score >= single_score and pair_score >= score0:
                bits2 = free_bits.copy()
                bits2[top_two] ^= 1
                pivots2 = (
                    pivot_bits
                    ^ r_reduced[:, free_positions[top_two[0]]]
                    ^ r_reduced[:, free_positions[top_two[1]]]
                )
                best_score = pair_score
            elif single_score >= score0:
                bits2 = free_bits.copy()
                bits2[best_free] ^= 1
                pivots2 = pivot_bits ^ r_reduced[:, free_positions[best_free]]
                best_score = single_score
            else:
                bits2 = free_bits
                pivots2 = pivot_bits
                best_score = score0

            code2 = np.zeros(n, dtype=np.uint8)
            code2[free_original] = bits2
            code2[pivot_original] = pivots2
            scores2.append(best_score)
            errors2 += int(np.count_nonzero(code2[:k] != truth))

    hard_ber = float(np.mean(
        (instance.llr[:limit, :k] < 0) != answers[:limit, :k]
    ))
    result = {
        "diagnostic_only": True,
        "samples": limit,
        "hard_ber": hard_ber,
        "osd0_ber": errors0 / (limit * k),
        "osd1_ber": errors1 / (limit * k),
        "osd2_ber": errors2 / (limit * k) if args.osd2 else None,
        "osd0_mean_score": float(np.mean(scores0)),
        "osd1_mean_score": float(np.mean(scores1)),
        "osd2_mean_score": float(np.mean(scores2)) if args.osd2 else None,
        "truth_mean_score": float(np.mean([
            score_codeword(answers[i], instance.llr[i]) for i in range(limit)
        ])),
        "runtime_seconds": time.perf_counter() - start,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / args.summary_name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Summary written to {args.outdir / args.summary_name}")


if __name__ == "__main__":
    main()
