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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from real_bench import build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def write_input(case: dict, path: Path) -> None:
    """Write a case in judge format: header, H, LLR rows, G."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{case['n']} {case['k']} {case['b']} {case['snr_db']:g}"]
    lines.extend(" ".join(map(str, row.tolist())) for row in case["h"])
    lines.extend(" ".join(f"{value:.17g}" for value in row) for row in case["llr"])
    lines.extend(" ".join(map(str, row.tolist())) for row in case["g"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_binary(binary: Path, input_path: Path, output_path: Path) -> float:
    start = time.perf_counter()
    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        proc = subprocess.run([str(binary)], stdin=fin, stdout=fout)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} exited with {proc.returncode} on {input_path.name}")
    return elapsed


def read_hard(output_path: Path, frames: int, n: int) -> np.ndarray:
    """Read one frame per line; bits may be space separated or contiguous."""
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != frames:
        raise ValueError(f"{output_path.name}: expected {frames} frames, found {len(lines)}")
    hard = np.empty((frames, n), dtype=np.int8)
    for index, line in enumerate(lines):
        digits = line.split() if len(line.split()) == n else list(line.strip())
        if len(digits) != n:
            raise ValueError(f"{output_path.name}: frame {index} has {len(digits)} bits, expected {n}")
        hard[index] = np.asarray([int(d) for d in digits], dtype=np.int8)
    return hard


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
        print(f"=== N={n} K={k} R={artifact['rate']:.3f} SNR={snr:.1f} ===", flush=True)
        per_seed = {}
        for seed_value in artifact["val_seeds"]:
            case = build_case(n, k, 500, snr, seed_value, noise_scale=noise_scale)
            base_errors, base_score, base_time = evaluate(
                args.baseline, case, args.workdir, f"n{n}_s{seed_value}_v12"
            )
            cand_errors, cand_score, cand_time = evaluate(
                args.candidate, case, args.workdir, f"n{n}_s{seed_value}_v13c"
            )
            recorded = artifact["validation"][str(seed_value)]
            entry = {
                "seed": seed_value,
                "frames": case["b"],
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
                f"  seed {seed_value}: v12 {base_score:.1f} ({base_errors} bit errors) -> "
                f"v13c {cand_score:.1f} ({cand_errors} bit errors), gain {cand_score - base_score:+.1f}  "
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
