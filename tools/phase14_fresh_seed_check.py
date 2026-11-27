#!/usr/bin/env python3
"""Phase 14 A4 fresh-seed gate: compare v17 with v16 on seed 6060.

Changed cases must gain at least +1,500 points with a positive 95% paired
bootstrap lower bound. Unchanged cases must produce byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import read_hard, run_binary, write_input  # noqa: E402
from real_bench import N_VALUES, RATE_SNR_DB, build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
SEED = 6060
SAMPLES = 300
NOISE_SCALE = 1.001120
MIN_GAIN = 1500.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def paired_bootstrap_ci(
    base_errors: np.ndarray, cand_errors: np.ndarray, k: int, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    frames = len(base_errors)
    total = frames * k
    indices = rng.integers(0, frames, size=(samples, frames))
    base_scores = np.array(
        [score_of(int(base_errors[row].sum()), total) for row in indices]
    )
    cand_scores = np.array(
        [score_of(int(cand_errors[row].sum()), total) for row in indices]
    )
    deltas = cand_scores - base_scores
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_case(
    binary: Path, case: dict, workdir: Path, tag: str
) -> tuple[np.ndarray, bytes]:
    input_path = workdir / f"fresh_{tag}.in"
    output_path = workdir / f"fresh_{tag}.out"
    write_input(case, input_path)
    run_binary(binary, input_path, output_path)
    hard = read_hard(output_path, case["b"], case["n"])
    return hard, output_path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16-binary", type=Path, required=True)
    parser.add_argument("--v17-binary", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "phase14" / "fresh_seed_6060.json",
    )
    args = parser.parse_args()

    changed = set(args.changed_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="fresh6060_", dir=args.output.parent) as tmp_name:
        workdir = Path(tmp_name)
        case_index = 0
        for n in N_VALUES:
            for rate, snr in RATE_SNR_DB:
                case_index += 1
                k = int(round(n * rate))
                case = build_case(
                    n, k, SAMPLES, snr, SEED,
                    noise_scale=NOISE_SCALE, llr_sigma="nominal",
                )
                base_hard, base_bytes = run_case(args.v16_binary, case, workdir, "v16")
                cand_hard, cand_bytes = run_case(args.v17_binary, case, workdir, "v17")
                base_errors = np.count_nonzero(
                    base_hard[:, :k] != case["answers"][:, :k], axis=1
                ).astype(np.int64)
                cand_errors = np.count_nonzero(
                    cand_hard[:, :k] != case["answers"][:, :k], axis=1
                ).astype(np.int64)
                base_score = score_of(int(base_errors.sum()), SAMPLES * k)
                cand_score = score_of(int(cand_errors.sum()), SAMPLES * k)
                gain = cand_score - base_score
                byte_identical = base_bytes == cand_bytes
                is_changed = case_index in changed

                entry = {
                    "case_index": case_index,
                    "n": n,
                    "k": k,
                    "rate": rate,
                    "snr": snr,
                    "changed": is_changed,
                    "byte_identical": byte_identical,
                    "baseline_score": base_score,
                    "candidate_score": cand_score,
                    "gain": gain,
                    "baseline_errors": int(base_errors.sum()),
                    "candidate_errors": int(cand_errors.sum()),
                }
                if is_changed:
                    low, high = paired_bootstrap_ci(
                        base_errors, cand_errors, k, 2000, SEED + case_index
                    )
                    entry["gain_ci_95"] = [low, high]
                    entry["gate_passed"] = gain >= MIN_GAIN and low > 0.0
                else:
                    entry["gain_ci_95"] = None
                    entry["gate_passed"] = byte_identical
                results.append(entry)
                print(
                    f"case {case_index:>2d} {'changed' if is_changed else 'unchanged'}: "
                    f"gain {gain:+8.1f}, identical {byte_identical}, "
                    f"gate {entry['gate_passed']}",
                    flush=True,
                )

    payload = {
        "seed": SEED,
        "samples": SAMPLES,
        "noise_scale": NOISE_SCALE,
        "min_changed_gain": MIN_GAIN,
        "v16_binary": str(args.v16_binary.resolve()),
        "v16_sha256": sha256(args.v16_binary),
        "v17_binary": str(args.v17_binary.resolve()),
        "v17_sha256": sha256(args.v17_binary),
        "changed_cases": sorted(changed),
        "cases": results,
        "all_passed": all(item["gate_passed"] for item in results),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
