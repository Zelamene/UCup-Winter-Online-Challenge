#!/usr/bin/env python3
"""Phase 5: generalization checks for the unfolded dSB decoder.

The learned Phase 4 parameters are code-independent. This harness applies them
to freshly generated systematic LDPC codes with the public ensemble structure
and evaluates the required N/R/SNR combinations. Ground truth is generated and
used only offline to score BER and frame errors; the decoder never sees it.
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
from phase3_fixed_dsb import CheckTable  # noqa: E402
from phase4_unfolded_dsb import (
    BASE_KEYS,
    MOD_KEYS,
    RawParams,
    build_features,
    decode_hard,
)
from phase6_multi_guess import decode_candidates, select_by_energy  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"
DEFAULT_OUTDIR = ROOT / "artifacts"
N_VALUES = (648, 1296, 1944, 3888)
RATE_SNR_DB = {0.5: 6.0, 2.0 / 3.0: 9.0, 0.75: 10.5}
RATE_LABELS = {0.5: "1/2", 2.0 / 3.0: "2/3", 0.75: "3/4"}


def load_params(path: Path) -> RawParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["values"]
    log_values = {
        key: np.asarray(values[key], dtype=np.float64) for key in BASE_KEYS
    }
    for key in BASE_KEYS:
        log_values[key] = np.log(log_values[key])
    for key in MOD_KEYS:
        log_values[key] = np.asarray(values[key], dtype=np.float64)
    params = RawParams(log_values)
    sizes = {value.size for value in params.values.values()}
    if payload.get("steps") != 50 or sizes != {50, 1}:
        raise ValueError("Unexpected Phase 4 parameter shape")
    return params


def empirical_margins(instance, answers: np.ndarray) -> np.ndarray:
    signed = (1.0 - 2.0 * answers.astype(np.float64)) * instance.llr
    return signed.ravel()


def make_a(k: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Create a k x m A matrix with column weight 2 and uniform row weights."""
    if k % m:
        raise ValueError(f"k={k} must be divisible by m={m} for this ensemble")
    repeats = k // m
    first_blocks = []
    second_blocks = []
    for _ in range(repeats):
        block = rng.permutation(m)
        shift = int(rng.integers(1, m))
        first_blocks.append(block)
        second_blocks.append(np.roll(block, shift))
    first = np.concatenate(first_blocks)
    second = np.concatenate(second_blocks)
    a = np.zeros((m, k), dtype=np.int8)
    a[first, np.arange(k)] = 1
    a[second, np.arange(k)] = 1
    return a


def make_case(
    n: int,
    rate: float,
    count: int,
    snr_db: float,
    margins: np.ndarray,
    public_snr_db: float,
    rng: np.random.Generator,
) -> dict:
    k = int(round(n * rate))
    m = n - k
    a = make_a(k, m, rng)
    h = np.concatenate((a, np.eye(m, dtype=np.int8)), axis=1)
    g = np.concatenate((np.eye(k, dtype=np.int8), a.T), axis=1)

    info = rng.integers(0, 2, size=(count, k), dtype=np.int8)
    parity = (info.astype(np.int16) @ a.T.astype(np.int16) % 2).astype(np.int8)
    words = np.concatenate((info, parity), axis=1)

    # Phase 1 observed LLR scaling is proportional to sqrt(SNR). Public SNR is
    # used as the calibration anchor; this only synthesizes channel inputs.
    scale = 10.0 ** ((snr_db - public_snr_db) / 20.0)
    drawn = rng.choice(margins, size=(count, n), replace=True) * scale
    llr = (1.0 - 2.0 * words.astype(np.float64)) * drawn

    valid_codewords = bool(np.all((g.astype(np.int16) @ h.T.astype(np.int16) % 2) == 0))
    return {
        "n": n,
        "k": k,
        "m": m,
        "rate": rate,
        "snr_db": snr_db,
        "h": h,
        "g": g,
        "llr": llr,
        "answers": words,
        "valid_codewords": valid_codewords,
    }


def score_case(params: RawParams, case: dict, multi_guess: bool = False) -> dict:
    h = case["h"]
    llr = case["llr"]
    answers = case["answers"]
    k = case["k"]
    checks = CheckTable(h)
    features = build_features(llr, checks)

    start = time.perf_counter()
    if multi_guess:
        bits, lambdas = decode_candidates(params, h, llr, features)
        hard, _ = select_by_energy(bits, lambdas, llr, checks)
        learned = bits[0]
    else:
        hard = decode_hard(params, h.astype(np.float64), llr, checks, features)
        learned = None
    decode_seconds = time.perf_counter() - start

    errors = hard[:, :k] != answers[:, :k]
    final_spins = 1.0 - 2.0 * hard.astype(np.float64)
    products = checks.parity_products(final_spins)
    hard_errors = (llr[:, :k] < 0) != answers[:, :k]

    return {
        "n": case["n"],
        "rate_label": RATE_LABELS[case["rate"]],
        "rate": case["rate"],
        "snr_db": case["snr_db"],
        "samples": llr.shape[0],
        "info_ber": float(errors.sum() / errors.size),
        "learned_single_ber": (
            float(np.mean(learned[:, :k] != answers[:, :k])) if learned is not None else None
        ),
        "hard_decision_ber": float(hard_errors.sum() / hard_errors.size),
        "frame_error_rate": float(np.any(errors, axis=1).mean()),
        "syndrome_ones": int(np.count_nonzero(products < 0)),
        "decode_seconds": decode_seconds,
        "seconds_per_frame": decode_seconds / llr.shape[0],
        "valid_codewords": case["valid_codewords"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--summary-name", default="phase5_generalization_summary.json")
    parser.add_argument("--multi-guess", action="store_true")
    args = parser.parse_args()

    params = load_params(args.params)
    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )
    margins = empirical_margins(instance, answers)
    rng = np.random.default_rng(args.seed)

    results = []
    for n in N_VALUES:
        for rate, snr_db in RATE_SNR_DB.items():
            case = make_case(
                n, rate, args.samples, snr_db, margins, instance.snr, rng
            )
            result = score_case(params, case, multi_guess=args.multi_guess)
            results.append(result)
            extra = ""
            if result["learned_single_ber"] is not None:
                extra = f" learned={result['learned_single_ber']:.6f}"
            print(
                f"N={n:>4d} R={result['rate_label']:>3s} SNR={snr_db:>4.1f} dB "
                f"BER={result['info_ber']:.6f} hard={result['hard_decision_ber']:.6f} "
                f"FER={result['frame_error_rate']:.3f}{extra} "
                f"{result['seconds_per_frame']:.4f}s/frame"
            )

    gate = {
        "all_codewords_valid": all(row["valid_codewords"] for row in results),
        "max_frame_runtime_under_120s": max(
            row["seconds_per_frame"] for row in results
        ) < 120.0,
        "decoder_beats_hard_decision_in_every_case": all(
            row["info_ber"] < row["hard_decision_ber"] for row in results
        ),
    }

    print("\nPHASE 5 GATE")
    for key, value in gate.items():
        print(f"{key}: {value}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / args.summary_name
    summary_path.write_text(
        json.dumps(
            {
                "framework": (
                    "six_candidate_dsb_energy_selection"
                    if args.multi_guess
                    else "deep_unfolded_dsb"
                ),
                "params_file": str(args.params),
                "channel_model": "resampled_public_signed_margins_scaled_by_sqrt_snr",
                "public_snr_db": instance.snr,
                "samples_per_case": args.samples,
                "results": results,
                "gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary written to {summary_path}")

    if not all(gate.values()):
        raise SystemExit("Phase 5 gate failed; inspect the summary.")


if __name__ == "__main__":
    main()
