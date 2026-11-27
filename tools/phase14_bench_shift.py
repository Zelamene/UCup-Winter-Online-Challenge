#!/usr/bin/env python3
"""Phase 14 A0 regression: score v16 with old and new LLR sigma conventions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from baseline_cache import sha256  # noqa: E402
from binary_io import read_hard, run_binary, write_input  # noqa: E402
from real_bench import N_VALUES, RATE_SNR_DB, build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0


def score_case(binary: Path, case: dict, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / "case.in"
    output_path = workdir / "case.out"
    write_input(case, input_path)
    elapsed = run_binary(binary, input_path, output_path)
    hard = read_hard(output_path, case["b"], case["n"])
    errors = np.count_nonzero(
        hard[:, : case["k"]] != case["answers"][:, : case["k"]], axis=1
    ).astype(np.int64)
    total = int(errors.sum())
    ber = max(total / (case["b"] * case["k"]), 1.0 / (case["b"] * case["k"]))
    return {
        "errors": total,
        "score": -SCORE_SCALE * math.log10(ber),
        "seconds": round(elapsed, 3),
    }


def compute_entry(
    binary: Path,
    case_index: int,
    n: int,
    k: int,
    snr: float,
    seed: int,
    samples: int,
    noise_scale: float,
    output_dir: Path,
) -> dict:
    """Score one case with old and new LLR conventions."""
    old_case = build_case(
        n, k, samples, snr, seed,
        noise_scale=noise_scale, llr_sigma="noise_scale",
    )
    new_case = build_case(
        n, k, samples, snr, seed,
        noise_scale=noise_scale, llr_sigma="nominal",
    )
    with tempfile.TemporaryDirectory(prefix="bench_shift_", dir=output_dir) as tmp_name:
        workdir = Path(tmp_name)
        old = score_case(binary, old_case, workdir / "old")
        new = score_case(binary, new_case, workdir / "new")
    shift = new["score"] - old["score"]
    return {
        "case_index": case_index,
        "n": n,
        "k": k,
        "rate": k / n,
        "snr": snr,
        "seed": seed,
        "old_score": old["score"],
        "new_score": new["score"],
        "shift": shift,
        "old_errors": old["errors"],
        "new_errors": new["errors"],
        "old_seconds": old["seconds"],
        "new_seconds": new["seconds"],
        "raw_ber_identical": bool(
            np.array_equal(old_case["llr"] < 0, new_case["llr"] < 0)
        ),
        "within_1k": abs(shift) < 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 31337, 6060])
    parser.add_argument("--noise-scale", type=float, default=1.001120)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "phase14" / "bench_v2_shift.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    binary = args.binary.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    tasks: list[tuple[int, int, float, int]] = []
    for seed in args.seeds:
        case_index = 0
        for n in N_VALUES:
            for rate, snr in RATE_SNR_DB:
                tasks.append((case_index, n, int(round(n * rate)), snr, seed))
                case_index += 1

    entries: list[dict | None] = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                compute_entry,
                binary, case_index, n, k, snr, seed, args.samples,
                args.noise_scale, args.output.parent,
            ): position
            for position, (case_index, n, k, snr, seed) in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(futures):
            position = futures[future]
            entry = future.result()
            entries[position] = entry
            print(
                f"case {entry['case_index']:>2d} seed {entry['seed']}: "
                f"shift {entry['shift']:+8.1f} "
                f"({'ok' if entry['within_1k'] else 'FAIL'})",
                flush=True,
            )
    entries = [entry for entry in entries if entry is not None]

    worst = max(entries, key=lambda item: abs(item["shift"]))
    payload = {
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "samples": args.samples,
        "seeds": args.seeds,
        "noise_scale": args.noise_scale,
        "old_llr_sigma": "noise_scale",
        "new_llr_sigma": "nominal",
        "gate": {
            "all_within_1k": all(item["within_1k"] for item in entries),
            "all_raw_ber_identical": all(item["raw_ber_identical"] for item in entries),
            "worst_abs_shift": abs(worst["shift"]),
            "worst_case": {
                "case_index": worst["case_index"],
                "seed": worst["seed"],
                "shift": worst["shift"],
            },
        },
        "cases": entries,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))
    print(f"Saved {args.output}")
    if not payload["gate"]["all_within_1k"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
