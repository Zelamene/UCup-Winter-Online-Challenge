#!/usr/bin/env python3
"""Phase 6: six parallel dSB trajectories with energy-based selection.

The learned unfolded decoder is one candidate; five fixed dSB schedules are the
others. Selection uses only the received LLRs and parity-check energy, never
ground-truth bits. Six candidates is the maximum allowed by the rules.
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

from phase0_harness import SAMPLE_INPUT, SAMPLE_OUTPUT, parse_answers, parse_instance  # noqa: E402
from phase3_fixed_dsb import DSBConfig, CheckTable, decode_dsb  # noqa: E402
from phase4_unfolded_dsb import (  # noqa: E402
    BASE_KEYS,
    MOD_KEYS,
    RawParams,
    build_features,
    decode_hard,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"
DEFAULT_OUTDIR = ROOT / "artifacts"
MAX_GUESSES = 6
LEARNED_ENERGY_LAMBDA = 6.0
FIXED_CANDIDATES = (
    (4.0, 0.5, 6.0, 2.5),
    (4.0, 0.5, 5.0, 2.5),
    (4.0, 0.4, 5.0, 2.0),
    (4.0, 0.4, 6.0, 2.0),
    (8.0, 0.5, 6.0, 2.5),
)


def load_params(path: Path) -> RawParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["values"]
    log_values = {key: np.log(np.asarray(values[key], dtype=np.float64)) for key in BASE_KEYS}
    log_values.update({key: np.asarray(values[key], dtype=np.float64) for key in MOD_KEYS})
    return RawParams(log_values)


def energy(hard: np.ndarray, llr: np.ndarray, checks: CheckTable, lambd: float) -> np.ndarray:
    spins = 1.0 - 2.0 * hard.astype(np.float64)
    products = checks.parity_products(spins)
    return -np.sum(llr * spins, axis=1) + 0.5 * lambd * np.sum(1.0 - products, axis=1)


def decode_candidates(
    params: RawParams, h: np.ndarray, llr: np.ndarray, features: dict[str, np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[float]]:
    checks = CheckTable(h)
    if features is None:
        features = build_features(llr, checks)
    learned = decode_hard(params, h.astype(np.float64), llr, checks, features)
    bits = [learned]
    lambdas = [LEARNED_ENERGY_LAMBDA]
    for tau, xi, lambd, delta in FIXED_CANDIDATES:
        config = DSBConfig(
            tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50, gradient_sign=1
        )
        bits.append(decode_dsb(config, h, llr))
        lambdas.append(lambd)
    return bits, lambdas


def select_by_energy(
    bits: list[np.ndarray], lambdas: list[float], llr: np.ndarray, checks: CheckTable
) -> tuple[np.ndarray, np.ndarray]:
    energies = np.stack([energy(item, llr, checks, lambd) for item, lambd in zip(bits, lambdas)])
    pick = np.argmin(energies, axis=0)
    stacked = np.stack(bits)
    selected = stacked[
        pick[:, None], np.arange(llr.shape[0])[:, None], np.arange(llr.shape[1])[None, :]
    ]
    return selected, pick


def evaluate_bits(hard: np.ndarray, llr: np.ndarray, answers: np.ndarray, k: int, checks: CheckTable) -> dict:
    errors = hard[:, :k] != answers[:, :k]
    spin = 1.0 - 2.0 * hard.astype(np.float64)
    products = checks.parity_products(spin)
    return {
        "info_ber": float(errors.sum() / errors.size),
        "frame_error_rate": float(np.any(errors, axis=1).mean()),
        "syndrome_ones": int(np.count_nonzero(products < 0)),
        "rows_with_failed_checks": int(np.count_nonzero(np.any(products < 0, axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--summary-name", default="phase6_multi_guess_summary.json")
    parser.add_argument("--config-name", default="phase6_multi_guess_config.json")
    args = parser.parse_args()

    params = load_params(args.params)
    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )
    checks = CheckTable(instance.h)
    features = build_features(instance.llr, checks)

    start = time.perf_counter()
    bits, lambdas = decode_candidates(params, instance.h, instance.llr, features)
    selected, pick = select_by_energy(bits, lambdas, instance.llr, checks)
    seconds = time.perf_counter() - start

    permutation = np.random.default_rng(20261125).permutation(instance.b)
    train_idx, val_idx = permutation[:400], permutation[400:]
    val = {
        "learned": evaluate_bits(bits[0][val_idx], instance.llr[val_idx], answers[val_idx], instance.k, checks),
        "selected": evaluate_bits(selected[val_idx], instance.llr[val_idx], answers[val_idx], instance.k, checks),
    }
    full = {
        "learned": evaluate_bits(bits[0], instance.llr, answers, instance.k, checks),
        "selected": evaluate_bits(selected, instance.llr, answers, instance.k, checks),
    }

    sample = parse_instance(SAMPLE_INPUT, "PDF sample")
    sample_answer = parse_answers(SAMPLE_OUTPUT, "PDF sample answer", sample.n, sample.b)
    sample_bits, sample_lambdas = decode_candidates(params, sample.h, sample.llr)
    sample_checks = CheckTable(sample.h)
    sample_selected, _ = select_by_energy(sample_bits, sample_lambdas, sample.llr, sample_checks)
    sample_ok = bool(np.array_equal(sample_selected, sample_answer))

    gate = {
        "guess_count_at_most_six": len(bits) <= MAX_GUESSES,
        "pdf_sample_matches": sample_ok,
        "validation_ber_below_learned_single": val["selected"]["info_ber"] < val["learned"]["info_ber"],
        "full_ber_below_learned_single": full["selected"]["info_ber"] < full["learned"]["info_ber"],
        "runtime_under_120s_per_frame": seconds / instance.b < 120.0,
    }

    print(f"Learned single validation BER: {val['learned']['info_ber']:.9f}")
    print(f"Selected validation BER:       {val['selected']['info_ber']:.9f}")
    print(f"Learned single full BER:       {full['learned']['info_ber']:.9f}")
    print(f"Selected full BER:             {full['selected']['info_ber']:.9f}")
    print(f"Seconds per frame:             {seconds / instance.b:.4f}")
    print("\nPHASE 6 GATE")
    for key, value in gate.items():
        print(f"{key}: {value}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / args.summary_name
    summary_path.write_text(
        json.dumps(
            {
                "framework": "multi_trajectory_dsb_energy_selection",
                "guess_count": len(bits),
                "learned_energy_lambda": LEARNED_ENERGY_LAMBDA,
                "fixed_candidates": FIXED_CANDIDATES,
                "selection_counts": np.bincount(pick, minlength=len(bits)).tolist(),
                "validation": val,
                "full_public": full,
                "pdf_sample_matches": sample_ok,
                "seconds_per_frame": seconds / instance.b,
                "gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = args.outdir / args.config_name
    config_path.write_text(
        json.dumps(
            {
                "learned_energy_lambda": LEARNED_ENERGY_LAMBDA,
                "fixed_candidates": FIXED_CANDIDATES,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSummary written to {summary_path}")
    print(f"Config written to {config_path}")

    if not all(gate.values()):
        raise SystemExit("Phase 6 gate failed; inspect the summary.")


if __name__ == "__main__":
    main()
