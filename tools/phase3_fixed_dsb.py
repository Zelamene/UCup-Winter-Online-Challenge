#!/usr/bin/env python3
"""Phase 3: fixed-parameter dSB decoder baseline.

The decoder implements the discrete simulated bifurcation dynamics only. It
does not use belief propagation, min-sum, bit flipping, or ground-truth bits.
This phase selects one fixed configuration on a subset, then evaluates it on
the public dataset as the baseline for later deep unfolding.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import SAMPLE_INPUT, SAMPLE_OUTPUT, parse_answers, parse_instance  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"

MAX_ALLOWED_STEPS = 50


@dataclass(frozen=True)
class DSBConfig:
    """Fixed scalar parameters of one dSB evolution."""

    tau: float
    xi: float
    lambd: float
    delta: float
    dt: float
    steps: int
    gradient_sign: int

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.xi <= 0 or self.lambd <= 0 or self.dt <= 0:
            raise ValueError("dSB scale parameters must be positive")
        if not 1 <= self.steps <= MAX_ALLOWED_STEPS:
            raise ValueError(f"steps must be in [1, {MAX_ALLOWED_STEPS}]")
        if self.gradient_sign not in (-1, 1):
            raise ValueError("gradient_sign must be +1 (minimize L) or -1 (literal Eq. 6)")


class CheckTable:
    """Padded sparse access to parity-check neighbors."""

    def __init__(self, h: np.ndarray) -> None:
        rows = [np.flatnonzero(h[row]) for row in range(h.shape[0])]
        width = max((row.size for row in rows), default=1)
        self.indices = np.zeros((len(rows), width), dtype=np.int64)
        self.mask = np.zeros((len(rows), width), dtype=bool)
        for row, columns in enumerate(rows):
            self.indices[row, : columns.size] = columns
            self.mask[row, : columns.size] = True

    def parity_products(self, spins: np.ndarray) -> np.ndarray:
        gathered = spins[:, self.indices]
        neutral = np.where(self.mask[None, :, :], gathered, 1)
        return np.prod(neutral, axis=2)


def decode_dsb(config: DSBConfig, h: np.ndarray, llr: np.ndarray) -> np.ndarray:
    """Decode LLR rows with fixed-parameter dSB.

    Spins use sigma=+1 for bit 0 and sigma=-1 for bit 1. The minimize-L
    gradient adds r_i + lambda/2 * sum_m lambda_m * product_excluding_i.
    gradient_sign=-1 applies the literal plus-gradient written in Eq. (6).
    """

    batch, n = llr.shape
    checks = CheckTable(h)

    x = np.tanh(llr / config.tau)
    y = np.zeros_like(x)

    if config.steps == 1:
        schedule = np.zeros(1)
    else:
        schedule = np.linspace(0.0, 1.0, config.steps)

    for step in range(config.steps):
        spins = np.where(x >= 0, 1.0, -1.0)
        products = checks.parity_products(spins)

        # dSB force: grad_i = r_i + (lambda/2) sum_{m contains i} P_m s_i.
        force = llr.copy()
        gathered = spins[:, checks.indices]
        scatter = (
            0.5
            * config.lambd
            * products[:, :, None]
            * gathered
            * checks.mask[None, :, :]
        )
        force[:, checks.indices] += scatter
        force *= config.gradient_sign

        pump = -(config.delta - schedule[step]) * x
        y += config.dt * (pump + config.xi * force)
        x += config.dt * y

        clipped = np.abs(x) > 1.0
        if np.any(clipped):
            x[clipped] = np.where(x[clipped] >= 0, 1.0, -1.0)
            y[clipped] = 0.0

    return (x < 0).astype(np.int8)


def evaluate(
    h: np.ndarray, llr: np.ndarray, answers: np.ndarray, config: DSBConfig
) -> dict:
    hard = decode_dsb(config, h, llr)
    info_truth = answers[:, : h.shape[1] - h.shape[0]]
    info_errors = hard[:, :info_truth.shape[1]] != info_truth

    spins = 1.0 - 2.0 * hard.astype(np.float64)
    products = CheckTable(h).parity_products(spins)
    syndrome_failures = int(np.count_nonzero(np.any(products < 0, axis=1)))
    syndrome_ones = int(np.count_nonzero(products < 0))
    energy = float(np.mean(
        -np.sum(llr * spins, axis=1)
        + 0.5 * config.lambd * np.sum(1.0 - products, axis=1)
    ))

    return {
        "config": asdict(config),
        "samples": int(llr.shape[0]),
        "info_errors": int(info_errors.sum()),
        "info_ber": float(info_errors.sum() / info_truth.size),
        "frame_errors": int(np.any(info_errors, axis=1).sum()),
        "frame_error_rate": float(np.any(info_errors, axis=1).mean()),
        "rows_with_failed_checks": syndrome_failures,
        "syndrome_ones": syndrome_ones,
        "mean_energy": energy,
    }


def config_grid() -> list[DSBConfig]:
    return [
        DSBConfig(
            tau=tau,
            xi=xi,
            lambd=lambd,
            delta=delta,
            dt=dt,
            steps=MAX_ALLOWED_STEPS,
            gradient_sign=sign,
        )
        for sign, tau, xi, lambd, delta, dt in itertools.product(
            (1, -1),
            (4.0, 8.0, 16.0),
            (0.01, 0.03, 0.06),
            (0.5, 1.0, 2.0),
            (0.25, 0.5),
            (0.1, 0.2),
        )
    ]


def rank_key(result: dict) -> tuple[float, int, float]:
    return (result["info_ber"], result["syndrome_ones"], result["mean_energy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sweep-samples", type=int, default=40)
    parser.add_argument("--summary-name", default="phase3_fixed_dsb_summary.json")
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )
    sweep_count = min(instance.b, max(1, args.sweep_samples))

    hard_errors = int(np.count_nonzero(
        (instance.llr[:, : instance.k] < 0) != answers[:, : instance.k]
    ))
    hard_ber = hard_errors / (instance.b * instance.k)
    hard_products = CheckTable(instance.h).parity_products(
        np.where(instance.llr < 0, -1.0, 1.0)
    )
    hard_syndrome_ones = int(np.count_nonzero(hard_products < 0))

    print(f"Hard-decision baseline BER: {hard_ber:.9f}")
    print(f"Sweeping fixed dSB configurations on {sweep_count} samples\n")

    sweep_llr = instance.llr[:sweep_count]
    sweep_answers = answers[:sweep_count]
    sweep_results: list[dict] = []
    start = time.perf_counter()

    for config in config_grid():
        result = evaluate(instance.h, sweep_llr, sweep_answers, config)
        sweep_results.append(result)

    sweep_seconds = time.perf_counter() - start
    sweep_results.sort(key=rank_key)
    print(f"Evaluated {len(sweep_results)} configurations in {sweep_seconds:.2f}s")
    print("Top sweep configurations:")
    header = f"{'BER':>12} {'Synd':>7} {'Tau':>5} {'Xi':>6} {'Lam':>5} {'Delta':>6} {'dt':>5} {'Sign':>5}"
    print(header)
    print("-" * len(header))
    for result in sweep_results[:10]:
        cfg = result["config"]
        print(
            f"{result['info_ber']:>12.9f} {result['syndrome_ones']:>7d} "
            f"{cfg['tau']:>5.1f} {cfg['xi']:>6.3f} {cfg['lambd']:>5.2f} "
            f"{cfg['delta']:>6.3f} {cfg['dt']:>5.2f} {cfg['gradient_sign']:>5d}"
        )

    best_config = DSBConfig(**sweep_results[0]["config"])

    sample = parse_instance(SAMPLE_INPUT, "PDF sample")
    sample_expected = parse_answers(
        SAMPLE_OUTPUT, "PDF sample answer", sample.n, sample.b
    )
    sample_decoded = decode_dsb(best_config, sample.h, sample.llr)
    sample_ok = bool(np.array_equal(sample_decoded, sample_expected))

    full_result = evaluate(instance.h, instance.llr, answers, best_config)
    syndrome_reduction = (
        (hard_syndrome_ones - full_result["syndrome_ones"]) / hard_syndrome_ones
        if hard_syndrome_ones
        else 1.0
    )

    gate = {
        "pdf_sample_matches": sample_ok,
        "steps_within_limit": best_config.steps <= MAX_ALLOWED_STEPS,
        "ber_below_hard_decision": full_result["info_ber"] < hard_ber,
        "syndrome_ones_reduced_by_at_least_50_percent": syndrome_reduction >= 0.50,
    }

    print("\nBest full-dataset dSB result:")
    print(json.dumps(full_result, indent=2, sort_keys=True))
    print("\nPHASE 3 GATE")
    for key, value in gate.items():
        print(f"{key}: {value}")

    summary = {
        "framework": "fixed_parameter_dsb",
        "diagnostic_ground_truth_used_only_for_evaluation": True,
        "hard_decision_info_ber": hard_ber,
        "hard_decision_syndrome_ones": hard_syndrome_ones,
        "sweep_samples": sweep_count,
        "sweep_seconds": sweep_seconds,
        "sweep_results": sweep_results,
        "best": full_result,
        "syndrome_reduction": syndrome_reduction,
        "pdf_sample_matches": sample_ok,
        "gate": gate,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / args.summary_name
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSummary written to {summary_path}")

    if not all(gate.values()):
        raise SystemExit("Phase 3 gate failed; inspect the dSB sweep summary.")


if __name__ == "__main__":
    main()
