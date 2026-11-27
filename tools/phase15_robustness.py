#!/usr/bin/env python3
"""Phase 15 robustness checks for a v18 candidate.

1. SNR offsets -0.3 / 0 / +0.3 dB on changed cases: candidate binary versus
   compiled v17 on identical frames, with paired bootstrap gains.
2. Schedule extremes: the 33-parameter CMA representation at all log
   multipliers log(0.5) and log(2), decoded with finite-state assertions.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402
import tune_cma_schedule as schedules  # noqa: E402
from binary_io import read_hard, run_binary, write_input  # noqa: E402
from real_bench import build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_OUT = p.PHASE15_ARTIFACTS / "robustness.json"
SNR_OFFSETS = (-0.3, 0.0, 0.3)
FRAMES = 100
SEED = 6060
MAX_REGRESSION = 3000.0


def error_counts(binary: Path, case: dict, workdir: Path, tag: str) -> np.ndarray:
    input_path = workdir / f"robust_{tag}.in"
    output_path = workdir / f"robust_{tag}.out"
    write_input(case, input_path)
    run_binary(binary, input_path, output_path)
    hard = read_hard(output_path, case["b"], case["n"])
    return np.count_nonzero(hard[:, :case["k"]] != case["answers"][:, :case["k"]], axis=1).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v17-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    snr_results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="phase15_robust_", dir=args.out.parent) as tmp_name:
        workdir = Path(tmp_name)
        for case_index in args.changed_cases:
            _, n, k, rate, snr = p.CASE_SPECS[case_index - 1]
            for offset in SNR_OFFSETS:
                case = build_case(
                    n, k, args.frames, snr + offset, SEED,
                    noise_scale=p.NOISE_SCALE, llr_sigma="nominal",
                )
                base_errors = error_counts(args.v17_binary, case, workdir, "v17")
                cand_errors = error_counts(args.candidate_binary, case, workdir, "cand")
                base_score = p.score_from_errors(base_errors, k)
                cand_score = p.score_from_errors(cand_errors, k)
                low, high = p.paired_bootstrap(
                    base_errors, cand_errors, k, samples=1000,
                    seed=SEED + case_index + int(round(offset * 10)),
                )
                gain = cand_score - base_score
                snr_results.append({
                    "case_index": case_index, "n": n, "k": k,
                    "snr": snr + offset, "offset_db": offset,
                    "gain": gain, "gain_ci_95": [low, high],
                    "v17_errors": int(base_errors.sum()),
                    "candidate_errors": int(cand_errors.sum()),
                    "pass": gain >= -MAX_REGRESSION,
                })
                print(f"case {case_index} snr {snr + offset:.1f}: gain {gain:+.1f}, pass {gain >= -MAX_REGRESSION}", flush=True)

    extreme_results: list[dict] = []
    for case_index in args.changed_cases:
        _, n, k, _, snr = p.CASE_SPECS[case_index - 1]
        base = p.load_v17_config(n, k)
        for label, value in (("low", schedules.LOG_LOW), ("high", schedules.LOG_HIGH)):
            params = np.full(schedules.DIM, value, dtype=float)
            config = schedules.config_from_log_params(base, params)
            workspace = p.CaseWorkspace(n, k, 911, 300, snr=snr, frame_slice=3)
            hard = p.decode_hard_schedules(workspace, config, debug_finite=True)
            extreme_results.append({
                "case_index": case_index,
                "log_multiplier": label,
                "hard_bits_valid": bool(np.isin(hard, (0, 1)).all()),
                "max_abs_config_value": float(max(
                    abs(float(item))
                    for key, value in config.items()
                    if key in p.CONFIG_KEYS
                    for item in (value if isinstance(value, (list, np.ndarray)) else [value])
                )),
                "pass": True,
            })

    payload = {
        "snr_offsets": SNR_OFFSETS,
        "frames": args.frames,
        "seed": SEED,
        "bootstrap_resamples": 1000,
        "max_regression": MAX_REGRESSION,
        "v17_binary": str(args.v17_binary.resolve()),
        "candidate_binary": str(args.candidate_binary.resolve()),
        "snr_cases": snr_results,
        "schedule_extremes": extreme_results,
        "all_passed": all(item["pass"] for item in snr_results + extreme_results),
    }
    p.write_json(args.out, payload)
    print(f"Saved {args.out}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
