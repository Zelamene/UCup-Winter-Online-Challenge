#!/usr/bin/env python3
"""Diagnostic-only investigation of stronger offline BP schedules.

This file is not submission code. It tests whether the Phase 2 target was
unrealistic or whether the flooding implementation was simply weak.
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
from phase2_reference_decoder import (  # noqa: E402
    build_graph,
    compute_posterior,
    min_sum_check_update,
    sum_product_check_update,
    update_v2c,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"


def evaluate(hard: np.ndarray, h: np.ndarray, llr: np.ndarray, answers: np.ndarray) -> dict:
    info_truth = answers[:, : h.shape[1] - h.shape[0]]
    hard_info = hard[:, :info_truth.shape[1]]
    info_errors = hard_info != info_truth
    syndrome = (hard.astype(np.int16) @ h.T.astype(np.int16)) % 2
    valid = np.all(syndrome == 0, axis=1)

    return {
        "info_ber": float(info_errors.sum() / info_truth.size),
        "frame_error_rate": float(np.any(info_errors, axis=1).mean()),
        "info_errors": int(info_errors.sum()),
        "syndrome_failures": int(np.count_nonzero(~valid)),
        "syndrome_ones": int(syndrome.sum()),
        "false_valid_codewords": int(np.count_nonzero(valid & np.any(info_errors, axis=1))),
        "mean_llr_score": float(np.mean(-np.sum(hard * llr, axis=1))),
    }


def decode_damped_flooding(
    h: np.ndarray,
    llr: np.ndarray,
    answers: np.ndarray,
    damping: float,
    max_iter: int = 50,
) -> dict:
    graph = build_graph(h)
    batch = llr.shape[0]
    v2c = llr[:, graph.check_neighbors].copy()
    c2v = np.zeros_like(v2c)
    start = time.perf_counter()
    iterations_run = 0

    for iteration in range(1, max_iter + 1):
        new_c2v = sum_product_check_update(v2c)
        c2v = new_c2v if iteration == 1 else (1.0 - damping) * c2v + damping * new_c2v
        c2v_flat = c2v.reshape(batch, graph.edge_count)
        posterior = compute_posterior(llr, c2v_flat, graph)
        hard = (posterior < 0).astype(np.int8)
        iterations_run = iteration
        v2c = update_v2c(llr, c2v_flat, graph).reshape(
            batch, graph.m, graph.check_neighbors.shape[1]
        )

    result = {
        "method": "damped_flooding_sum_product",
        "damping": damping,
        "iterations": iterations_run,
        "runtime_seconds": time.perf_counter() - start,
    }
    result.update(evaluate(hard, h, llr, answers))
    return result


def decode_layered(
    h: np.ndarray,
    llr: np.ndarray,
    answers: np.ndarray,
    method: str,
    damping: float = 0.0,
    alpha: float = 0.8,
    max_iter: int = 50,
) -> dict:
    graph = build_graph(h)
    posterior = llr.copy()
    c2v = np.zeros((llr.shape[0], graph.m, graph.check_neighbors.shape[1]), dtype=np.float64)
    start = time.perf_counter()
    iterations_run = 0

    for iteration in range(1, max_iter + 1):
        for check in range(graph.m):
            neighbors = graph.check_neighbors[check]
            extrinsic = posterior[:, neighbors] - c2v[:, check, :]

            if method == "sum_product":
                message = sum_product_check_update(extrinsic[:, None, :])[:, 0, :]
            elif method == "min_sum":
                message = min_sum_check_update(extrinsic[:, None, :], alpha)[:, 0, :]
            else:
                raise ValueError(f"Unknown layered method: {method}")

            if iteration > 1 and damping:
                message = (1.0 - damping) * c2v[:, check, :] + damping * message

            c2v[:, check, :] = message
            posterior[:, neighbors] = extrinsic + message

        iterations_run = iteration

    hard = (posterior < 0).astype(np.int8)
    result = {
        "method": f"layered_{method}",
        "damping": damping,
        "alpha": alpha if method == "min_sum" else None,
        "iterations": iterations_run,
        "runtime_seconds": time.perf_counter() - start,
    }
    result.update(evaluate(hard, h, llr, answers))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-iter", type=int, default=50)
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )

    baseline_hard = (instance.llr < 0).astype(np.int8)
    baseline = evaluate(baseline_hard, instance.h, instance.llr, answers)
    truth_score = float(np.mean(-np.sum(answers * instance.llr, axis=1)))
    print(f"Hard baseline BER: {baseline['info_ber']:.9f}")
    print(f"Truth mean LLR score: {truth_score:.3f}\n")

    runs = []
    for damping in [0.25, 0.50, 0.75]:
        runs.append(decode_damped_flooding(
            instance.h, instance.llr, answers, damping, args.max_iter
        ))

    for alpha in [0.75, 0.90, 1.00]:
        runs.append(decode_layered(
            instance.h, instance.llr, answers, "min_sum",
            damping=0.0, alpha=alpha, max_iter=args.max_iter
        ))

    runs.append(decode_layered(
        instance.h, instance.llr, answers, "sum_product",
        damping=0.0, max_iter=args.max_iter
    ))

    header = (
        f"{'Method':<30} {'Param':>6} {'BER':>12} {'FER':>8} "
        f"{'Syn':>7} {'FalseOK':>8} {'Score':>12} {'Sec':>8}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        param = run.get("damping", run.get("alpha"))
        param_text = "-" if param is None else f"{param:.2f}"
        print(
            f"{run['method']:<30} {param_text:>6} {run['info_ber']:>12.9f} "
            f"{run['frame_error_rate']:>8.4f} {run['syndrome_failures']:>7d} "
            f"{run['false_valid_codewords']:>8d} {run['mean_llr_score']:>12.3f} "
            f"{run['runtime_seconds']:>8.2f}"
        )

    best = min(runs, key=lambda run: run["info_ber"])
    print(f"\nBest BER: {best['info_ber']:.9f} using {best['method']}")
    print(f"Improvement over hard decision: {baseline['info_ber'] / best['info_ber']:.3f}x")
    print(f"Best score vs truth score: {best['mean_llr_score']:.3f} vs {truth_score:.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "diagnostic_only": True,
        "hard_baseline": baseline,
        "truth_mean_llr_score": truth_score,
        "runs": runs,
        "best": {
            "method": best["method"],
            "info_ber": best["info_ber"],
            "improvement_over_hard": baseline["info_ber"] / best["info_ber"],
        },
    }
    (args.outdir / "phase2_investigation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Summary written to {args.outdir / 'phase2_investigation_summary.json'}")


if __name__ == "__main__":
    main()
