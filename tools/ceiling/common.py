# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Shared helpers for Phase 13 ceiling analysis.

This package measures what stronger, non-QAIA decoders could achieve.  It is
never imported by the exporter or by any submitted source.  All decoders here
are measuring sticks only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CEILING_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CEILING_DIR.parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from real_bench import N_VALUES, RATE_SNR_DB, build_case as build_real_bench_case  # noqa: E402

OFFLINE_HEADER = "# OFFLINE ANALYSIS ONLY — never imported by the exporter"
SCORE_SCALE = 100_000.0
NOISE_SCALE = 1.00112
SEEDS = (2024, 31337, 6060)
FRAMES = 500
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261126

CASE_SPECS = tuple(
    (n, int(round(n * rate)), snr)
    for n in N_VALUES
    for rate, snr in RATE_SNR_DB
)

# Keep this list in sync with validate_submission_v3.py's isolation gate.
OFFLINE_SYMBOLS = (
    "OFFLINE ANALYSIS ONLY",
    "tools.ceiling",
    "tools/ceiling",
    "ceiling.common",
    "ceiling.cache_binary",
    "ceiling.calibrate_bench_vs_judge",
    "ceiling.compare_ceiling_caches",
    "phase13_ceiling",
    "paired_bootstrap_ci",
    "paired_bootstrap_total_ci",
    "build_ceiling_case",
    "load_ceiling_cache",
    "calibrate_bench_vs_judge",
    "compare_ceiling_caches",
)


def resolve_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(str(value).replace("\\", "/")).expanduser()
    return path if path.is_absolute() else ROOT / path


def case_key(n: int, k: int) -> str:
    """Return the stable cache key for one (N, K) case."""
    return f"N{n}_K{k}"


def snr_for(n: int, k: int) -> float:
    """Return the configured SNR for a Phase 13 case."""
    for spec_n, spec_k, spec_snr in CASE_SPECS:
        if spec_n == n and spec_k == k:
            return spec_snr
    raise KeyError(f"unknown Phase 13 case N={n} K={k}")


def build_ceiling_case(n: int, k: int, seed: int, frames: int = FRAMES) -> dict:
    """Build one fixed-rule bench case from real_bench.build_case."""
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not one of {SEEDS}")
    return build_real_bench_case(
        n, k, frames, snr_for(n, k), seed, noise_scale=NOISE_SCALE
    )


def score_from_total(errors: int | np.integer, bits: int) -> float:
    """Score one error total using the judge-style log formula."""
    ber = max(float(errors) / bits, 1.0 / bits)
    return -SCORE_SCALE * math.log10(ber)


def score_from_errors(errors: np.ndarray, bits_per_frame: int) -> float:
    """Score a per-frame error array."""
    return score_from_total(int(np.asarray(errors).sum()), len(errors) * bits_per_frame)


def check_offline_header(path: Path) -> None:
    """Fail if a Phase 13 tool no longer carries the isolation header."""
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if first_line != OFFLINE_HEADER:
        raise ValueError(f"{path}: first line must be {OFFLINE_HEADER!r}")


def validate_metadata(metadata: dict, n: int, k: int, seed: int, frames: int) -> None:
    """Verify the fixed Phase 13 frame-generation rules in cache metadata."""
    if int(metadata.get("n", -1)) != n or int(metadata.get("k", -1)) != k:
        raise ValueError(
            f"metadata case mismatch: expected N={n} K={k}, got "
            f"N={metadata.get('n')} K={metadata.get('k')}"
        )
    if int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"metadata seed mismatch: expected {seed}, got {metadata.get('seed')}")
    if int(metadata.get("samples", -1)) != frames:
        raise ValueError(
            f"metadata frame mismatch: expected {frames}, got {metadata.get('samples')}"
        )
    actual_noise = float(metadata.get("noise_scale", float("nan")))
    if not math.isclose(actual_noise, NOISE_SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"metadata noise_scale mismatch: expected {NOISE_SCALE}, got {actual_noise}"
        )


def load_ceiling_cache(
    cache_dir: str | Path,
    n: int,
    k: int,
    seed: int,
    frames: int = FRAMES,
    expected_binary_sha256: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Load and validate one per-frame information-bit error cache."""
    directory = Path(cache_dir)
    stem = f"{case_key(n, k)}_seed{seed}"
    array_path = directory / f"{stem}.npy"
    metadata_path = directory / f"{stem}.json"
    if not array_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing cache pair for {stem} in {directory}")

    errors = np.load(array_path)
    if errors.ndim != 1 or errors.dtype.kind not in "iu":
        raise ValueError(f"{array_path}: expected a 1-D integer array, got {errors.shape} {errors.dtype}")
    if len(errors) != frames:
        raise ValueError(f"{array_path}: expected {frames} frames, got {len(errors)}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, n, k, seed, frames)
    if expected_binary_sha256 is not None:
        actual = metadata.get("binary_sha256")
        if actual != expected_binary_sha256:
            raise ValueError(
                f"{metadata_path}: binary hash {actual} != expected {expected_binary_sha256}"
            )
    return errors.astype(np.int64, copy=False), metadata


def score_bootstrap_ci(
    errors: np.ndarray,
    bits_per_frame: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap the score of one decoder on one paired frame set."""
    errors = np.asarray(errors, dtype=np.int64)
    frames = len(errors)
    bits = frames * bits_per_frame
    rng = np.random.default_rng(seed)
    scores = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, frames, size=frames)
        scores[index] = score_from_total(errors[sampled].sum(), bits)
    low, high = np.percentile(scores, (2.5, 97.5))
    return {
        "point": score_from_errors(errors, bits_per_frame),
        "mean": float(scores.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": replicates,
        "bootstrap_seed": seed,
    }


def paired_bootstrap_ci(
    baseline_errors: np.ndarray,
    candidate_errors: np.ndarray,
    bits_per_frame: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired frame bootstrap CI for candidate_score - baseline_score."""
    baseline = np.asarray(baseline_errors, dtype=np.int64)
    candidate = np.asarray(candidate_errors, dtype=np.int64)
    if baseline.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {baseline.shape} vs {candidate.shape}")
    frames = len(baseline)
    if frames == 0:
        raise ValueError("cannot bootstrap an empty frame set")
    bits = frames * bits_per_frame
    point = score_from_errors(candidate, bits_per_frame) - score_from_errors(
        baseline, bits_per_frame
    )

    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, frames, size=frames)
        candidate_score = score_from_total(candidate[sampled].sum(), bits)
        baseline_score = score_from_total(baseline[sampled].sum(), bits)
        deltas[index] = candidate_score - baseline_score
    low, high = np.percentile(deltas, (2.5, 97.5))
    return {
        "point": point,
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=1)) if replicates > 1 else 0.0,
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": replicates,
        "bootstrap_seed": seed,
    }


def paired_bootstrap_total_ci(
    groups: list[tuple[np.ndarray, np.ndarray, int]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap CI for the summed score delta over independent groups.

    Each group is ``(baseline_errors, candidate_errors, k)``.  Frames are paired
    within each group and resampled independently across groups.
    """
    if not groups:
        raise ValueError("no bootstrap groups supplied")
    prepared: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    point = 0.0
    for baseline, candidate, bits_per_frame in groups:
        baseline = np.asarray(baseline, dtype=np.int64)
        candidate = np.asarray(candidate, dtype=np.int64)
        if baseline.shape != candidate.shape or len(baseline) == 0:
            raise ValueError("invalid bootstrap group")
        frames = len(baseline)
        point += score_from_errors(candidate, bits_per_frame) - score_from_errors(
            baseline, bits_per_frame
        )
        prepared.append((baseline, candidate, frames * bits_per_frame, frames))

    rng = np.random.default_rng(seed)
    totals = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        total = 0.0
        for baseline, candidate, bits, frames in prepared:
            sampled = rng.integers(0, frames, size=frames)
            total += score_from_total(candidate[sampled].sum(), bits)
            total -= score_from_total(baseline[sampled].sum(), bits)
        totals[index] = total
    low, high = np.percentile(totals, (2.5, 97.5))
    return {
        "point": point,
        "mean": float(totals.mean()),
        "std": float(totals.std(ddof=1)) if replicates > 1 else 0.0,
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": replicates,
        "bootstrap_seed": seed,
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = slope*x + intercept and report simple fit diagnostics."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two paired points for a linear fit")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def write_json(path: str | Path, payload: dict) -> Path:
    """Write a deterministic JSON artifact."""
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
