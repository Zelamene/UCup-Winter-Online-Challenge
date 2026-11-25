#!/usr/bin/env python3
"""Diagnostic-only BP-guided OSD reference.

Uses BP soft posteriors only for ordering/hard decisions, then searches valid
codewords by ordered statistics. Not submission code.
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
from phase2_investigate_osd import row_reduce_columns  # noqa: E402
from phase2_reference_decoder import (  # noqa: E402
    build_graph,
    compute_posterior,
    sum_product_check_update,
    update_v2c,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"


def bp_posteriors(h: np.ndarray, llr: np.ndarray, max_iter: int) -> np.ndarray:
    graph = build_graph(h)
    batch = llr.shape[0]
    v2c = llr[:, graph.check_neighbors].copy()
    posterior = llr.copy()
    for _ in range(max_iter):
        c2v = sum_product_check_update(v2c)
        flat = c2v.reshape(batch, graph.edge_count)
        posterior = compute_posterior(llr, flat, graph)
        v2c = update_v2c(llr, flat, graph).reshape(
            batch, graph.m, graph.check_neighbors.shape[1]
        )
    return posterior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--summary-name", default="phase2_bp_osd_summary.json")
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )
    limit = instance.b if args.max_samples <= 0 else min(args.max_samples, instance.b)
    n, k = instance.n, instance.k

    start = time.perf_counter()
    posteriors = bp_posteriors(instance.h, instance.llr[:limit], args.max_iter)

    errors_bp = errors_osd0 = errors_osd1 = 0
    score_bp = score_osd0 = score_osd1 = 0.0

    for sample in range(limit):
        llr = instance.llr[sample]
        truth = answers[sample, :k]
        posterior = posteriors[sample]
        hard_bp = (posterior < 0).astype(np.uint8)
        errors_bp += int(np.count_nonzero(hard_bp[:k] != truth))
        score_bp += float(-np.dot(hard_bp, llr))

        order = np.argsort(np.abs(posterior), kind="stable")
        reduced, pivots = row_reduce_columns(instance.h, order)
        free_positions = np.setdiff1d(np.arange(n), pivots)
        pivot_original = order[pivots]
        free_original = order[free_positions]

        free_bits = hard_bp[free_original]
        pivot_bits = (reduced[:, free_positions] @ free_bits) % 2

        code0 = np.zeros(n, dtype=np.uint8)
        code0[free_original] = free_bits
        code0[pivot_original] = pivot_bits
        errors_osd0 += int(np.count_nonzero(code0[:k] != truth))
        score_osd0 += float(-np.dot(code0, llr))

        free_weights = (2 * free_bits.astype(float) - 1) * llr[free_original]
        pivot_weights = (2 * pivot_bits.astype(float) - 1) * llr[pivot_original]
        pivot_deltas = pivot_weights @ reduced[:, free_positions]
        candidate_scores = (-np.dot(code0, llr)) + free_weights + pivot_deltas
        best_free = int(np.argmax(candidate_scores))

        bits1 = free_bits.copy()
        bits1[best_free] ^= 1
        pivots1 = pivot_bits ^ reduced[:, free_positions[best_free]]
        code1 = np.zeros(n, dtype=np.uint8)
        code1[free_original] = bits1
        code1[pivot_original] = pivots1
        errors_osd1 += int(np.count_nonzero(code1[:k] != truth))
        score_osd1 += float(-np.dot(code1, llr))

    result = {
        "diagnostic_only": True,
        "samples": limit,
        "bp_info_ber": errors_bp / (limit * k),
        "bp_osd0_info_ber": errors_osd0 / (limit * k),
        "bp_osd1_info_ber": errors_osd1 / (limit * k),
        "bp_mean_score": score_bp / limit,
        "bp_osd0_mean_score": score_osd0 / limit,
        "bp_osd1_mean_score": score_osd1 / limit,
        "truth_mean_score": float(np.mean([
            -np.dot(answers[i], instance.llr[i]) for i in range(limit)
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
