# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Cache per-frame information-bit errors for an offline analysis binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CEILING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CEILING_DIR))

from common import (  # noqa: E402
    CASE_SPECS,
    FRAMES,
    NOISE_SCALE,
    OFFLINE_HEADER,
    ROOT,
    SEEDS,
    build_ceiling_case,
    case_key,
    check_offline_header,
    resolve_path,
    score_from_errors,
    snr_for,
    validate_metadata,
    write_json,
)
from binary_io import read_hard, run_binary, write_input  # noqa: E402


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_one(
    binary: Path,
    n: int,
    k: int,
    seed: int,
    frames: int,
    cache_dir: Path,
    skip_existing: bool,
) -> dict:
    """Run one binary on one fixed-rule case and cache its per-frame errors."""
    binary = resolve_path(binary)
    cache_dir = resolve_path(cache_dir)
    if not binary.exists():
        raise FileNotFoundError(f"missing analysis binary: {binary}")
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not one of {SEEDS}")

    binary_hash = sha256(binary)
    stem = f"{case_key(n, k)}_seed{seed}"
    metadata_path = cache_dir / f"{stem}.json"
    if skip_existing and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            validate_metadata(existing, n, k, seed, frames)
            if existing.get("binary_sha256") == binary_hash:
                print(f"skip {stem}: cache already matches binary {binary_hash[:12]}", flush=True)
                return existing
        except (TypeError, ValueError):
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    snr = snr_for(n, k)
    case = build_ceiling_case(n, k, seed, frames)
    if not np.isclose(case["snr_db"], snr, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"SNR mismatch for N={n} K={k}: {case['snr_db']} != {snr}")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ceiling_cache_", dir=cache_dir) as tmp_name:
        workdir = Path(tmp_name)
        input_path = workdir / "case.in"
        output_path = workdir / "case.out"
        write_input(case, input_path)
        elapsed = run_binary(binary, input_path, output_path)
        hard = read_hard(output_path, case["b"], case["n"])

    errors = np.count_nonzero(
        hard[:, :k] != case["answers"][:, :k], axis=1
    ).astype(np.int64)
    total = int(errors.sum())
    array_path = cache_dir / f"{stem}.npy"
    np.save(array_path, errors)

    metadata = {
        "array": array_path.name,
        "binary": str(binary),
        "binary_sha256": binary_hash,
        "frames": frames,
        "k": k,
        "n": n,
        "noise_scale": NOISE_SCALE,
        "offline_header": OFFLINE_HEADER,
        "samples": frames,
        "score": score_from_errors(errors, k),
        "seconds": round(elapsed, 3),
        "seed": seed,
        "snr": snr,
        "tool": "tools/ceiling/cache_binary.py",
        "total_info_errors": total,
        "total_wall_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(metadata_path, metadata)
    print(
        f"{stem}: {total} errors, score {metadata['score']:,.1f}, "
        f"{elapsed:.1f}s -> {array_path}",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "artifacts" / "ceiling_cache",
    )
    parser.add_argument("--samples", type=int, default=FRAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "artifacts" / "phase13" / "ceiling_cache_manifest.json",
    )
    args = parser.parse_args()

    check_offline_header(Path(__file__))
    if args.samples != FRAMES:
        raise SystemExit(f"Phase 13 requires --samples {FRAMES}; got {args.samples}")
    bad_seeds = sorted(set(args.seeds) - set(SEEDS))
    if bad_seeds:
        raise SystemExit(f"Phase 13 seeds must be within {SEEDS}; got {bad_seeds}")

    if args.all_cases:
        specs = list(CASE_SPECS)
    else:
        if args.n is None or args.rate is None:
            raise SystemExit("pass --n and --rate, or use --all-cases")
        k = int(round(args.n * args.rate))
        snr = args.snr if args.snr is not None else snr_for(args.n, k)
        specs = [(args.n, k, snr)]

    rows = []
    for n, k, snr in specs:
        for seed in args.seeds:
            rows.append(
                cache_one(
                    args.binary, n, k, seed, args.samples, args.cache_dir, args.skip_existing
                )
            )

    manifest = {
        "binary": str(resolve_path(args.binary)),
        "binary_sha256": sha256(resolve_path(args.binary)),
        "cache_dir": str(resolve_path(args.cache_dir)),
        "cases": rows,
        "frames": args.samples,
        "noise_scale": NOISE_SCALE,
        "offline_header": OFFLINE_HEADER,
        "seeds": list(args.seeds),
        "tool": "tools/ceiling/cache_binary.py",
    }
    path = write_json(args.manifest_out, manifest)
    print(f"Cache manifest written to {path}", flush=True)


if __name__ == "__main__":
    main()
