#!/usr/bin/env python3
"""Per-frame v13c baseline errors on the exact real-bench validation frames.

The high-rate cases inherit v12's six-trajectory dSB portfolios in v13c. Decoding
those portfolios frame by frame in Python is the bottleneck during bSB tuning, so
this tool runs the compiled v13c binary once per (N, K, seed) and stores one
information-bit error count per frame. Stage A, Stage B, and binary validation
reuse the resulting ``.npy`` files.
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
from real_bench import N_VALUES, RATE_SNR_DB, build_case, write_answers  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "baseline_cache"
DEFAULT_BINARY = ROOT / "artifacts" / "build" / "v13c.exe"
SCORE_SCALE = 100_000.0


def cache_path(cache_dir: Path, n: int, k: int, seed: int) -> Path:
    """Return the cache filename for one case and seed."""
    return Path(cache_dir) / f"N{n}_K{k}_seed{seed}.npy"


def load_baseline_errors(
    cache_dir: Path,
    n: int,
    k: int,
    seed: int,
    expected_frames: int | None = None,
) -> np.ndarray:
    """Load cached per-frame errors, optionally checking the frame count."""
    path = cache_path(cache_dir, n, k, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"missing baseline cache {path}; run tools/baseline_cache.py first"
        )
    errors = np.load(path)
    if errors.ndim != 1 or errors.dtype.kind not in "iu":
        raise ValueError(f"{path}: expected a 1-D integer array, got {errors.shape} {errors.dtype}")
    if expected_frames is not None and len(errors) != expected_frames:
        raise ValueError(
            f"{path}: expected {expected_frames} frames, found {len(errors)}"
        )
    return errors.astype(np.int64, copy=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_one(
    n: int,
    k: int,
    snr: float,
    seed: int,
    samples: int,
    binary: Path,
    noise_scale: float,
    cache_dir: Path,
    llr_sigma: str | float = "nominal",
) -> dict:
    """Build one case, run the baseline binary, and store per-frame errors."""
    binary = Path(binary)
    cache_dir = Path(cache_dir)
    if not binary.exists():
        raise FileNotFoundError(f"missing baseline binary: {binary}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(
        n, k, samples, snr, seed,
        noise_scale=noise_scale, llr_sigma=llr_sigma,
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="baseline_", dir=cache_dir) as tmp_name:
        workdir = Path(tmp_name)
        input_path = workdir / "case.in"
        output_path = workdir / "case.out"
        answers_path = workdir / "case.answers"
        write_input(case, input_path)
        write_answers(case, answers_path)
        elapsed = run_binary(binary, input_path, output_path)
        hard = read_hard(output_path, case["b"], case["n"])

    errors = np.count_nonzero(
        hard[:, :k] != case["answers"][:, :k], axis=1
    ).astype(np.int64)
    total = int(errors.sum())
    ber = max(total / (samples * k), 1.0 / (samples * k))
    score = -SCORE_SCALE * math.log10(ber)

    path = cache_path(cache_dir, n, k, seed)
    np.save(path, errors)
    metadata = {
        "n": n,
        "k": k,
        "snr": snr,
        "seed": seed,
        "samples": samples,
        "noise_scale": noise_scale,
        "llr_sigma": llr_sigma,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "total_info_errors": total,
        "score": score,
        "seconds": round(elapsed, 3),
        "array": path.name,
        "total_wall_seconds": round(time.perf_counter() - started, 3),
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"N={n} K={k} seed={seed}: {total} errors, score {score:,.1f}, "
        f"{elapsed:.1f}s -> {path}",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1296)
    parser.add_argument("--rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--noise-scale", type=float, default=1.001120)
    parser.add_argument(
        "--llr-sigma", default="nominal",
        help="'nominal' (default), 'noise_scale', or an explicit positive sigma",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--all-high-rate",
        action="store_true",
        help="generate every R=2/3 and R=3/4 case for all requested seeds",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="generate all 12 (N, rate) cases for all requested seeds",
    )
    args = parser.parse_args()

    seeds = args.seeds or [args.seed]
    rates = RATE_SNR_DB if args.all_cases else (RATE_SNR_DB[1:] if args.all_high_rate else None)
    if rates is not None:
        for n in N_VALUES:
            for rate, snr in rates:
                k = int(round(n * rate))
                for seed in seeds:
                    generate_one(
                        n, k, snr, seed, args.samples, args.binary,
                        args.noise_scale, args.cache_dir, args.llr_sigma,
                    )
        return

    snr = args.snr
    if snr is None:
        candidates = [configured for rate, configured in RATE_SNR_DB if abs(rate - args.rate) < 1e-6]
        if not candidates:
            raise SystemExit(f"no configured SNR for rate {args.rate}; pass --snr explicitly")
        snr = candidates[0]
    k = int(round(args.n * args.rate))
    for seed in seeds:
        generate_one(
            args.n, k, snr, seed, args.samples, args.binary,
            args.noise_scale, args.cache_dir, args.llr_sigma,
        )


if __name__ == "__main__":
    main()
