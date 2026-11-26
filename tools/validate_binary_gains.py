#!/usr/bin/env python3
"""Score the compiled submission on the exact Stage B validation frames.

`tune_bsb_stage_b.py` measures its gains with the Python batched simulator. This
tool rebuilds the identical validation frames (`real_bench.build_case` with the
same seed and noise scale the tuner used), writes them in judge format, runs both
compiled binaries on them, and reports what the C++ code actually scores.

Two things are checked per case and seed:

* the candidate binary reproduces the Python `candidate_score` from the Stage B
  artifact (that is what parity on synthetic inputs only implied), and
* the v12 binary reproduces the Python `base_score` on the same frames, so the
  gain is measured between two compiled decoders rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from baseline_cache import DEFAULT_CACHE_DIR, load_baseline_errors  # noqa: E402
from binary_io import read_hard, run_binary, write_input  # noqa: E402
from real_bench import build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def evaluate(binary: Path, case: dict, workdir: Path, tag: str) -> tuple[int, float, float]:
    input_path = workdir / f"{tag}.in"
    output_path = workdir / f"{tag}.out"
    write_input(case, input_path)
    elapsed = run_binary(binary, input_path, output_path)
    hard = read_hard(output_path, case["b"], case["n"])
    k = case["k"]
    errors = int(np.count_nonzero(hard[:, :k] != case["answers"][:, :k]))
    return errors, score_of(errors, case["b"] * k), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts" / "build" / "v13c.exe")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts" / "build" / "v12_ref.exe")
    parser.add_argument("--stage-b", type=Path, nargs="+", default=None)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "build" / "valbin")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "binary_validation.json")
    args = parser.parse_args()

    stage_b_paths = args.stage_b or [
        ROOT / "artifacts" / f"stage_b_N{n}_R0.5.json" for n in (648, 1296, 1944, 3888)
    ]
    for path in (args.candidate, args.baseline):
        if not path.exists():
            raise SystemExit(f"missing binary: {path}")

    results = []
    for path in stage_b_paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        n, k, snr = artifact["n"], artifact["k"], artifact["snr"]
        noise_scale = artifact.get("noise_scale", 1.001120)
        frames = int(artifact.get("val_samples", args.val_samples))
        use_cache = k / n > 0.5
        baseline_source = "v13c_binary_cache" if use_cache else "v12_binary"
        print(f"=== N={n} K={k} R={artifact['rate']:.3f} SNR={snr:.1f} ===", flush=True)
        per_seed = {}
        for seed_value in artifact["val_seeds"]:
            case = build_case(n, k, frames, snr, seed_value, noise_scale=noise_scale)
            if use_cache:
                cached = load_baseline_errors(
                    args.cache_dir, n, k, seed_value, expected_frames=frames
                )
                base_errors = int(cached.sum())
                base_score = score_of(base_errors, frames * k)
                base_time = 0.0
            else:
                base_errors, base_score, base_time = evaluate(
                    args.baseline, case, args.workdir, f"n{n}_s{seed_value}_v12"
                )
            cand_errors, cand_score, cand_time = evaluate(
                args.candidate, case, args.workdir, f"n{n}_s{seed_value}_candidate"
            )
            recorded = artifact["validation"][str(seed_value)]
            entry = {
                "seed": seed_value,
                "frames": case["b"],
                "baseline_source": baseline_source,
                "baseline_errors": base_errors,
                "baseline_score": base_score,
                "candidate_errors": cand_errors,
                "candidate_score": cand_score,
                "gain": cand_score - base_score,
                "recorded_base_errors": recorded["base_errors"],
                "recorded_candidate_errors": recorded["candidate_errors"],
                "recorded_gain": recorded["gain"],
                "gain_delta_vs_python": (cand_score - base_score) - recorded["gain"],
                "baseline_seconds": round(base_time, 2),
                "candidate_seconds": round(cand_time, 2),
            }
            per_seed[str(seed_value)] = entry
            print(
                f"  seed {seed_value}: {baseline_source} {base_score:.1f} ({base_errors} bit errors) -> "
                f"candidate {cand_score:.1f} ({cand_errors} bit errors), gain {cand_score - base_score:+.1f}  "
                f"[python said {recorded['gain']:+.1f}, recorded errors {recorded['base_errors']}/"
                f"{recorded['candidate_errors']}]  "
                f"{base_time:.1f}s / {cand_time:.1f}s",
                flush=True,
            )
        results.append({
            "n": n,
            "k": k,
            "rate": artifact["rate"],
            "snr": snr,
            "artifact": path.name,
            "seeds": per_seed,
            "min_gain": min(item["gain"] for item in per_seed.values()),
        })

    summary = {
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "cases": results,
        "min_gain": min(item["min_gain"] for item in results),
        "max_abs_delta_vs_python": max(
            abs(entry["gain_delta_vs_python"])
            for item in results
            for entry in item["seeds"].values()
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    print(f"min compiled gain: {summary['min_gain']:+.1f}")
    print(f"max |compiled - python| gain delta: {summary['max_abs_delta_vs_python']:.3f}")


if __name__ == "__main__":
    main()
