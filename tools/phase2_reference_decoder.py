#!/usr/bin/env python3
"""Phase 2: offline BP/min-sum reference decoders.

DIAGNOSTIC ONLY. These decoders are forbidden in the submitted solution.
They establish an achievable BER target for the public channel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import (  # noqa: E402
    SAMPLE_INPUT,
    SAMPLE_OUTPUT,
    parse_answers,
    parse_instance,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"

EPS = 1e-12
CHECKPOINTS = {1, 2, 3, 5, 10, 20, 50}


@dataclass(frozen=True)
class LDPCGraph:
    n: int
    m: int
    check_neighbors: np.ndarray  # (M, row_weight)
    edge_col: np.ndarray  # (E,)
    degree: np.ndarray  # (N,)
    var_offsets: np.ndarray  # (N + 1,), CSR-style boundaries into var_edges
    var_edges: np.ndarray  # (E,), edges grouped by variable

    @property
    def edge_count(self) -> int:
        return int(self.edge_col.size)


def build_graph(h: np.ndarray) -> LDPCGraph:
    m, n = h.shape
    row_weights = h.sum(axis=1)
    unique_weights = np.unique(row_weights)
    if unique_weights.size != 1:
        raise ValueError("This reference decoder assumes a row-regular LDPC matrix")
    row_weight = int(unique_weights[0])

    check_neighbors = np.empty((m, row_weight), dtype=np.int32)
    for row in range(m):
        neighbors = np.flatnonzero(h[row])
        if neighbors.size != row_weight:
            raise RuntimeError("Unexpected row weight while building graph")
        check_neighbors[row] = neighbors

    edge_col = check_neighbors.reshape(-1).astype(np.int32)
    degree = np.bincount(edge_col, minlength=n).astype(np.int16)
    var_edges = np.argsort(edge_col, kind="stable").astype(np.int32)
    var_offsets = np.zeros(n + 1, dtype=np.int64)
    var_offsets[1:] = np.cumsum(degree)

    return LDPCGraph(n, m, check_neighbors, edge_col, degree, var_offsets, var_edges)


def sum_product_check_update(v2c: np.ndarray) -> np.ndarray:
    """Check-node update using the tanh rule, shaped (batch, M, row_weight)."""
    q = np.tanh(v2c * 0.5)
    q = np.clip(q, -1.0 + EPS, 1.0 - EPS)

    signs = np.sign(q)
    signs[signs == 0] = 1.0
    magnitudes = np.maximum(np.abs(q), EPS)
    log_magnitudes = np.log(magnitudes)
    log_excluding = log_magnitudes.sum(axis=2, keepdims=True) - log_magnitudes

    product_signs = np.prod(signs, axis=2, keepdims=True)
    # Since sign^-1 == sign, multiplying removes the current edge's sign.
    excluded_signs = product_signs * signs
    excluded_product = np.clip(
        excluded_signs * np.exp(log_excluding), -1.0 + EPS, 1.0 - EPS
    )
    return 2.0 * np.arctanh(excluded_product)


def min_sum_check_update(v2c: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized min-sum check-node update."""
    signs = np.sign(v2c)
    signs[signs == 0] = 1.0
    magnitudes = np.abs(v2c)

    product_signs = np.prod(signs, axis=2, keepdims=True)
    excluded_signs = product_signs * signs

    ordered = np.partition(magnitudes, [0, 1], axis=2)
    smallest = ordered[:, :, 0:1]
    second = ordered[:, :, 1:2]
    excluded_minimum = np.where(magnitudes == smallest, second, smallest)
    return np.clip(alpha * excluded_signs * excluded_minimum, -50.0, 50.0)


def compute_posterior(
    llr: np.ndarray, c2v_flat: np.ndarray, graph: LDPCGraph
) -> np.ndarray:
    posterior = llr.copy()
    for variable in range(graph.n):
        start = graph.var_offsets[variable]
        end = graph.var_offsets[variable + 1]
        if start < end:
            posterior[:, variable] += c2v_flat[:, graph.var_edges[start:end]].sum(axis=1)
    return posterior


def update_v2c(
    llr: np.ndarray, c2v_flat: np.ndarray, graph: LDPCGraph
) -> np.ndarray:
    v2c = llr[:, graph.edge_col].copy()
    for variable in range(graph.n):
        start = graph.var_offsets[variable]
        end = graph.var_offsets[variable + 1]
        edges = graph.var_edges[start:end]
        if edges.size <= 1:
            continue
        incoming_total = c2v_flat[:, edges].sum(axis=1)
        v2c[:, edges] = (
            llr[:, variable, None] + incoming_total[:, None] - c2v_flat[:, edges]
        )
    return v2c


def decode(
    h: np.ndarray,
    llr: np.ndarray,
    answers: np.ndarray | None,
    decoder: str,
    alpha: float = 0.8,
    max_iter: int = 50,
) -> dict:
    graph = build_graph(h)
    batch_size = llr.shape[0]
    v2c = llr[:, graph.check_neighbors].copy()
    converged = np.zeros(batch_size, dtype=bool)
    convergence_iter = np.zeros(batch_size, dtype=np.int16)
    curve: list[dict] = []

    start = time.perf_counter()

    def evaluate(hard: np.ndarray, iteration: int, c2v_for_update: np.ndarray | None) -> None:
        syndrome = (hard.astype(np.int16) @ h.T.astype(np.int16)) % 2
        row = {
            "iteration": iteration,
            "violated_check_rows": int(np.count_nonzero(syndrome.sum(axis=1) > 0)),
            "syndrome_ones": int(syndrome.sum()),
        }
        if answers is not None:
            info_truth = answers[:, : h.shape[1] - h.shape[0]]
            info_errors = int(np.count_nonzero(hard[:, :info_truth.shape[1]] != info_truth))
            row["info_errors"] = info_errors
            row["info_ber"] = info_errors / info_truth.size
            row["frame_errors"] = int(np.count_nonzero(
                np.any(hard[:, :info_truth.shape[1]] != info_truth, axis=1)
            ))
        curve.append(row)

    initial_hard = (llr < 0).astype(np.int8)
    initial_syndrome = (initial_hard.astype(np.int16) @ h.T.astype(np.int16)) % 2
    already_converged = np.all(initial_syndrome == 0, axis=1)
    convergence_iter[already_converged] = 0
    converged |= already_converged
    evaluate(initial_hard, 0, None)

    iterations_run = 0
    final_hard = initial_hard

    for iteration in range(1, max_iter + 1):
        c2v = (
            sum_product_check_update(v2c)
            if decoder == "sum_product"
            else min_sum_check_update(v2c, alpha)
        )
        c2v_flat = c2v.reshape(batch_size, graph.edge_count)

        posterior = compute_posterior(llr, c2v_flat, graph)
        final_hard = (posterior < 0).astype(np.int8)

        syndrome = (final_hard.astype(np.int16) @ h.T.astype(np.int16)) % 2
        now_converged = np.all(syndrome == 0, axis=1)
        newly_converged = now_converged & ~converged
        convergence_iter[newly_converged] = iteration
        converged |= now_converged
        iterations_run = iteration

        if iteration in CHECKPOINTS or now_converged.all():
            evaluate(final_hard, iteration, c2v_flat)

        if converged.all():
            break

        if iteration < max_iter:
            v2c = update_v2c(llr, c2v_flat, graph).reshape(
                batch_size, graph.m, graph.check_neighbors.shape[1]
            )

    runtime = time.perf_counter() - start

    result = {
        "decoder": decoder,
        "alpha": alpha if decoder == "min_sum" else None,
        "iterations_run": iterations_run,
        "converged_codewords": int(converged.sum()),
        "codewords": batch_size,
        "convergence_iter_mean": float(convergence_iter.mean()),
        "convergence_iter_max": int(convergence_iter.max()),
        "convergence_iteration_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(convergence_iter, return_counts=True))
        },
        "runtime_seconds": runtime,
        "curve": curve,
    }

    if answers is not None:
        info_truth = answers[:, : h.shape[1] - h.shape[0]]
        hard_info = final_hard[:, :info_truth.shape[1]]
        info_errors = hard_info != info_truth
        result.update({
            "final_info_errors": int(info_errors.sum()),
            "final_info_ber": float(info_errors.sum() / info_truth.size),
            "final_frame_errors": int(np.any(info_errors, axis=1).sum()),
            "final_frame_error_rate": float(np.any(info_errors, axis=1).mean()),
            "false_valid_codewords": int(np.count_nonzero(converged & np.any(info_errors, axis=1))),
            "improvement_over_hard_decision": None,
        })

    if result["curve"] and result["curve"][-1]["iteration"] != iterations_run:
        evaluate(final_hard, iterations_run, None)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-iter", type=int, default=50)
    args = parser.parse_args()

    print("Phase 2 runs conventional decoders for diagnosis only.")
    print("None of these decoders may appear in the submitted solution.")

    sample = parse_instance(SAMPLE_INPUT, "PDF sample")
    sample_answers = parse_answers(SAMPLE_OUTPUT, "PDF sample answer", sample.n, sample.b)
    # Directly verify that BP does not alter the already-valid sample hard decision.
    sample_graph = build_graph(sample.h)
    sample_v2c = sample.llr[:, sample_graph.check_neighbors].copy()
    sample_c2v = sum_product_check_update(sample_v2c).reshape(sample.b, sample_graph.edge_count)
    sample_posterior = compute_posterior(sample.llr, sample_c2v, sample_graph)
    sample_ok = bool(
        np.array_equal((sample_posterior < 0).astype(np.int8), sample_answers)
    )

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )

    hard_truth = answers[:, : instance.k]
    hard_errors = int(np.count_nonzero((instance.llr[:, : instance.k] < 0) != hard_truth))
    hard_ber = hard_errors / hard_truth.size

    runs: list[dict] = []
    runs.append(decode(instance.h, instance.llr, answers, "sum_product", max_iter=args.max_iter))
    for alpha in [0.5, 0.7, 0.8, 0.9, 1.0]:
        runs.append(decode(
            instance.h, instance.llr, answers, "min_sum", alpha, max_iter=args.max_iter
        ))

    for run in runs:
        run["improvement_over_hard_decision"] = hard_ber / max(run["final_info_ber"], 1e-12)

    best = min(runs, key=lambda run: run["final_info_ber"])

    print(f"\nPDF sample unchanged by one BP pass: {sample_ok}")
    print(f"Hard-decision baseline BER: {hard_ber:.9f}\n")

    header = f"{'Decoder':<14} {'Alpha':>6} {'Iter':>5} {'BER':>12} {'FER':>10} {'Conv':>8} {'Seconds':>9}"
    print(header)
    print("-" * len(header))
    for run in runs:
        alpha_text = "-" if run["alpha"] is None else f"{run['alpha']:.2f}"
        print(
            f"{run['decoder']:<14} {alpha_text:>6} {run['iterations_run']:>5} "
            f"{run['final_info_ber']:>12.9f} {run['final_frame_error_rate']:>10.6f} "
            f"{run['converged_codewords']:>4}/{run['codewords']:<3} "
            f"{run['runtime_seconds']:>9.3f}"
        )

    initial_syndrome_ones = best["curve"][0]["syndrome_ones"]
    final_syndrome_ones = best["curve"][-1]["syndrome_ones"]
    syndrome_reduction = (
        (initial_syndrome_ones - final_syndrome_ones) / initial_syndrome_ones
        if initial_syndrome_ones
        else 1.0
    )
    ber_improvement = hard_ber / max(best["final_info_ber"], 1e-12)

    gate = {
        "pdf_sample_unchanged": sample_ok,
        "reference_ber_at_least_3x_better": ber_improvement >= 3.0,
        "syndrome_violations_reduced_by_at_least_90_percent": syndrome_reduction >= 0.90,
    }

    print("\nPHASE 2 GATE")
    for key, value in gate.items():
        print(f"{key}: {value}")
    print(f"ber_improvement: {ber_improvement:.6f}")
    print(f"syndrome_reduction: {syndrome_reduction:.6f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "diagnostic_only": True,
        "hard_decision_info_ber": hard_ber,
        "sample_check_passed": sample_ok,
        "runs": runs,
        "best_decoder": {
            "decoder": best["decoder"],
            "alpha": best["alpha"],
            "final_info_ber": best["final_info_ber"],
        },
        "ber_improvement": ber_improvement,
        "initial_syndrome_ones": initial_syndrome_ones,
        "final_syndrome_ones": final_syndrome_ones,
        "syndrome_reduction": syndrome_reduction,
        "gate": gate,
    }
    (args.outdir / "phase2_reference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSummary written to {args.outdir / 'phase2_reference_summary.json'}")

    if not all(gate.values()):
        raise SystemExit("Phase 2 gate failed; inspect the reference summaries.")


if __name__ == "__main__":
    main()
