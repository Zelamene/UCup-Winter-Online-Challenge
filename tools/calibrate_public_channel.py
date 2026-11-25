#!/usr/bin/env python3
"""Calibrate the real 16-QAM channel against the published input/answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import (  # noqa: E402
    analyze_answers,
    analyze_structure,
    parse_answers,
    parse_instance,
)
from real_h_generator import validate_h  # noqa: E402
from real_qam_channel import (  # noqa: E402
    calibrate_noise_scale,
    generate_llrs,
    raw_ber,
    theoretical_raw_ber,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase9_public_calibration.json"
TARGET_RAW_BER = 0.0535336


def _magnitude_stats(llr: np.ndarray) -> dict:
    absolute = np.abs(llr)
    quantiles = np.quantile(absolute, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    labels = ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return {
        "mean": float(absolute.mean()),
        "std": float(absolute.std()),
        "quantiles": {label: float(value) for label, value in zip(labels, quantiles)},
        "per_position_mean": absolute.mean(axis=0).tolist(),
        "zero_count": int(np.count_nonzero(llr == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )
    if answers.shape != instance.llr.shape:
        raise ValueError("public answers and LLRs have different shapes")

    hard = (instance.llr < 0).astype(np.int8)
    errors = hard != answers
    k = instance.k
    per_position_errors = errors.sum(axis=0)
    raw_values = raw_ber(instance.llr, answers)

    h_stats = validate_h(instance.h, instance.k / instance.n)
    structure = analyze_structure(instance, str(args.input))
    answer_stats = analyze_answers(instance, answers)

    scale = calibrate_noise_scale(
        target_raw_ber=TARGET_RAW_BER,
        snr_db=instance.snr,
        samples=args.calibration_samples,
        seed=args.seed,
    )

    simulated_rng = np.random.default_rng(args.seed + 1)
    simulated_bits = simulated_rng.integers(0, 2, size=(instance.b, instance.n), dtype=np.int8)
    simulated_llr = generate_llrs(
        simulated_bits,
        instance.snr,
        np.random.default_rng(args.seed + 2),
        noise_scale=scale,
    )
    simulated_errors = (simulated_llr < 0).astype(np.int8) != simulated_bits

    public_per_position_ber = (per_position_errors / instance.b).tolist()
    simulated_per_position_ber = simulated_errors.mean(axis=0).tolist()
    low_group = np.arange(instance.n) % 2 == 0
    high_group = ~low_group

    payload = {
        "source": {
            "input": str(args.input),
            "answer": str(args.answer),
            "header": {
                "n": instance.n,
                "k": instance.k,
                "m": instance.m,
                "b": instance.b,
                "snr_db": instance.snr,
            },
        },
        "raw_ber": {
            "all_bits": raw_values,
            "info_bits": float(errors[:, :k].mean()),
            "parity_bits": float(errors[:, k:].mean()),
            "per_position": public_per_position_ber,
            "per_position_error_counts": per_position_errors.tolist(),
            "even_position_ber": float(errors[:, low_group].mean()),
            "odd_position_ber": float(errors[:, high_group].mean()),
            "target_info_ber": TARGET_RAW_BER,
            "target_gap": float(errors[:, :k].mean() - TARGET_RAW_BER),
        },
        "h_stats": {
            "validation": h_stats,
            "structure": structure,
            "answer_checks": answer_stats,
        },
        "llr_magnitude_stats": _magnitude_stats(instance.llr),
        "calibration": {
            "target_raw_ber": TARGET_RAW_BER,
            "snr_db": instance.snr,
            "noise_scale": scale,
            "theoretical_ber_at_scale": theoretical_raw_ber(scale, instance.snr),
            "simulation_samples": args.calibration_samples,
            "seed": args.seed,
            "simulated_raw_ber": float(simulated_errors.mean()),
            "simulated_per_position_ber": simulated_per_position_ber,
            "simulated_mean_absolute_llr": float(np.mean(np.abs(simulated_llr))),
            "public_mean_absolute_llr": float(np.mean(np.abs(instance.llr))),
        },
        "checks": {
            "header_matches_public_facts": bool(
                instance.n == 1944
                and instance.k == 1458
                and instance.b == 500
                and abs(instance.snr - 10.3) < 1e-9
            ),
            "h_row_weight_7": bool(np.all(instance.h.sum(axis=1) == 7)),
            "h_info_column_weight_2": bool(np.all(instance.h[:, :k].sum(axis=0) == 2)),
            "info_ber_near_target": bool(
                abs(errors[:, :k].mean() - TARGET_RAW_BER) < 1e-4
            ),
            "noise_scale_near_one": bool(abs(scale - 1.0) < 0.05),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    raw = payload["raw_ber"]
    print(
        f"public raw BER={raw['all_bits']:.7f} info={raw['info_bits']:.7f} "
        f"parity={raw['parity_bits']:.7f}"
    )
    print(
        f"per-position even={raw['even_position_ber']:.6f} "
        f"odd={raw['odd_position_ber']:.6f}"
    )
    print(
        f"H row weights={h_stats['row_weight_counts']} "
        f"info col weights={h_stats['info_column_weight_counts']}"
    )
    print(
        f"|LLR| mean={payload['llr_magnitude_stats']['mean']:.6f} "
        f"median={payload['llr_magnitude_stats']['quantiles']['p50']:.6f}"
    )
    print(f"calibrated noise_scale={scale:.6f} (simulated BER={payload['calibration']['simulated_raw_ber']:.6f})")
    print(f"checks={json.dumps(payload['checks'], sort_keys=True)}")
    print(f"Wrote {args.output}")

    if not all(payload["checks"].values()):
        raise SystemExit("Public calibration checks failed; inspect the JSON output.")


if __name__ == "__main__":
    main()
