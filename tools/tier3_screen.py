#!/usr/bin/env python3
"""Screen Tier 3 dynamics against accepted v14 on synthetic R=1/2 cases."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
import sys  # noqa: E402

sys.path.insert(0, str(TOOLS_DIR))

from binary_io import read_hard, run_binary, write_input  # noqa: E402
from decode_v3_python import _build_sparse  # noqa: E402
from real_bench import build_case  # noqa: E402
from tier3_dynamics import decode_qsb, decode_simcim  # noqa: E402


def score(errors: int, bits: int) -> float:
    return -100_000.0 * math.log10(max(errors / bits, 1.0 / bits))


def bootstrap_gain(
    baseline_errors: np.ndarray, candidate_errors: np.ndarray, bits_per_frame: int, draws: int
) -> tuple[float, float]:
    frames = baseline_errors.size
    rng = np.random.default_rng(20261126)
    gains = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        pick = rng.integers(0, frames, size=frames)
        base = int(baseline_errors[pick].sum())
        cand = int(candidate_errors[pick].sum())
        gains[index] = score(cand, frames * bits_per_frame) - score(
            base, frames * bits_per_frame
        )
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts" / "build" / "v14.exe")
    parser.add_argument("--modes", nargs="+", choices=("qsb", "simcim"), default=["qsb", "simcim"])
    parser.add_argument("--ns", nargs="+", type=int, default=[648, 1296, 1944, 3888])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--noise-scale", type=float, default=1.001120)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    presets = json.loads(args.presets.read_text(encoding="utf-8"))
    selected = {
        name: params
        for name, params in presets.items()
        if params.get("mode") in args.modes
    }

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="tier3_screen_", dir=ROOT / "artifacts") as tmp_name:
        workdir = Path(tmp_name)
        for seed in args.seeds:
            for n in args.ns:
                k = n // 2
                case = build_case(
                    n,
                    k,
                    args.frames,
                    args.snr,
                    seed,
                    noise_scale=args.noise_scale,
                )
                input_path = workdir / f"case_{seed}_{n}.in"
                output_path = workdir / f"case_{seed}_{n}.out"
                write_input(case, input_path)
                run_binary(args.baseline, input_path, output_path)
                baseline = read_hard(output_path, args.frames, n)
                truth = case["answers"]
                baseline_errors = np.count_nonzero(
                    baseline[:, :k] != truth[:, :k], axis=1
                ).astype(np.int64)
                sparse = _build_sparse(case["h"])
                base_total = int(baseline_errors.sum())
                base_score = score(base_total, args.frames * k)

                for name, params in selected.items():
                    candidate_errors = np.zeros(args.frames, dtype=np.int64)
                    for frame in range(args.frames):
                        if params["mode"] == "qsb":
                            hard, _soft = decode_qsb(
                                case["llr"][frame], sparse, params, args.steps
                            )
                        else:
                            hard, _soft = decode_simcim(
                                case["llr"][frame], sparse, params, args.steps
                            )
                        candidate_errors[frame] = np.count_nonzero(
                            np.asarray(hard[:k], dtype=np.int8) != truth[frame, :k]
                        )
                    cand_total = int(candidate_errors.sum())
                    cand_score = score(cand_total, args.frames * k)
                    low, high = bootstrap_gain(
                        baseline_errors,
                        candidate_errors,
                        k,
                        args.bootstrap,
                    )
                    entry = {
                        "seed": seed,
                        "n": n,
                        "k": k,
                        "preset": name,
                        "mode": params["mode"],
                        "baseline_errors": base_total,
                        "candidate_errors": cand_total,
                        "baseline_score": base_score,
                        "candidate_score": cand_score,
                        "gain": cand_score - base_score,
                        "ci_low": low,
                        "ci_high": high,
                        "screen_pass": bool(cand_score - base_score >= 5_000.0 and low > 0.0),
                    }
                    rows.append(entry)
                    print(
                        f"seed={seed} n={n} {name}: "
                        f"{base_score:,.0f} -> {cand_score:,.0f} "
                        f"({entry['gain']:+,.0f}, CI [{low:+,.0f},{high:+,.0f}])"
                        f"{' PASS' if entry['screen_pass'] else ''}",
                        flush=True,
                    )

    payload = {
        "presets": str(args.presets),
        "baseline": str(args.baseline),
        "frames": args.frames,
        "snr": args.snr,
        "noise_scale": args.noise_scale,
        "steps": args.steps,
        "bootstrap_draws": args.bootstrap,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
