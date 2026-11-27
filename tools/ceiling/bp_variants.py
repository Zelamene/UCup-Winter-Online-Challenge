# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Offline per-bit and exact joint Gray-pair BP ceiling ladder.

This tool is a measuring stick only.  It never writes submission code and is
never imported by an exporter.  The joint variant runs loopy BP on the LDPC
check graph plus one exact pairwise channel factor per Gray pair; the pair
factor uses the ``a, b, d`` tables from ``tools/qam_pairs.py`` and keeps the
``d*s0*s1`` coupling rather than flattening it into two independent LLRs.

Caches use the ``tools/ceiling/cache_binary.py`` metadata conventions so they
can be loaded with ``common.load_ceiling_cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

CEILING_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CEILING_DIR.parent
for _entry in (str(CEILING_DIR), str(TOOLS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

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
    write_json,
)
import qam_pairs  # noqa: E402

EPS = 1e-12
DEFAULT_MAX_ITER = 30
DEFAULT_LR_BITS = 64
DEFAULT_CHUNK = 256

# Track B screen/ladder: the two R=2/3 cases that decide the v18 trigger plus
# the matching R=1/2 cases that bound the already-promoted v17 joint work.
TARGET_CASES = ((1944, 1296), (3888, 2592), (1944, 972), (3888, 1944))

# Building a fixed-rule case is much slower than decoding it.  Reuse the exact
# same generated frames for the per-bit and joint variants within one run.
_CASE_CACHE: dict[tuple[int, int, int, int], dict] = {}


def get_case(n: int, k: int, seed: int, frames: int) -> dict:
    """Return a generated fixed-rule case, building it only once per run."""
    key = (n, k, seed, frames)
    if key not in _CASE_CACHE:
        _CASE_CACHE[key] = build_ceiling_case(n, k, seed, frames)
    return _CASE_CACHE[key]


@dataclass(frozen=True)
class CheckGraph:
    """Row-regular check graph with a sparse variable-to-edge incidence."""

    neighbors: np.ndarray  # (m, row_weight)
    edge_col: np.ndarray  # (E,)
    incidence: csr_matrix  # (n, E), maps edge messages to variable posteriors

    @property
    def edge_count(self) -> int:
        return int(self.edge_col.size)


def build_check_graph(h: np.ndarray) -> CheckGraph:
    """Build the check-neighbour table and the variable incidence matrix."""
    h = np.asarray(h)
    m, n = h.shape
    row_weights = np.unique(h.sum(axis=1))
    if row_weights.size != 1:
        raise ValueError("bp_variants requires a row-regular H")
    weight = int(row_weights[0])
    neighbors = np.empty((m, weight), dtype=np.int32)
    for row in range(m):
        found = np.flatnonzero(h[row])
        if found.size != weight:
            raise ValueError(f"check row {row} has weight {found.size}, expected {weight}")
        neighbors[row] = found
    edge_col = neighbors.reshape(-1).astype(np.int32, copy=False)
    incidence = csr_matrix(
        (np.ones(edge_col.size, dtype=np.float64), (edge_col, np.arange(edge_col.size))),
        shape=(n, edge_col.size),
    )
    return CheckGraph(neighbors=neighbors, edge_col=edge_col, incidence=incidence)


def check_update(v2c: np.ndarray) -> np.ndarray:
    """Sum-product check-node update on LLRs shaped (batch, m, row_weight)."""
    q = np.clip(np.tanh(v2c * 0.5), -1.0 + EPS, 1.0 - EPS)
    signs = np.sign(q)
    signs[signs == 0] = 1.0
    magnitudes = np.maximum(np.abs(q), EPS)
    log_magnitudes = np.log(magnitudes)
    log_excluding = log_magnitudes.sum(axis=2, keepdims=True) - log_magnitudes
    product_signs = np.prod(signs, axis=2, keepdims=True)
    excluded_signs = product_signs * signs
    excluded = np.clip(
        excluded_signs * np.exp(log_excluding), -1.0 + EPS, 1.0 - EPS
    )
    return 2.0 * np.arctanh(excluded)


def logcosh(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(cosh(x)) for pair-factor messages."""
    absolute = np.abs(np.asarray(x, dtype=np.float64))
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - math.log(2.0)


def interleave(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Put per-pair LLRs back into consecutive bit order."""
    result = np.empty((first.shape[0], first.shape[1] * 2), dtype=np.float64)
    result[:, 0::2] = first
    result[:, 1::2] = second
    return result


def pair_marginal_llrs(a: np.ndarray, b: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-bit marginals of the joint pair factor with no check messages."""
    first = 2.0 * a + logcosh(b + d) - logcosh(b - d)
    second = 2.0 * b + logcosh(a + d) - logcosh(a - d)
    return first, second


def pair_factor_messages(
    a: np.ndarray, b: np.ndarray, d: np.ndarray, v2p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Message from each exact pair factor to its two variable nodes.

    ``v2p`` holds extrinsic check-side LLRs; the channel fields ``a`` and ``b``
    stay inside the factor, and ``d`` couples the two spins.
    """
    incoming_second = b + 0.5 * v2p[:, :, 1]
    incoming_first = a + 0.5 * v2p[:, :, 0]
    to_first = 2.0 * a + logcosh(incoming_second + d) - logcosh(incoming_second - d)
    to_second = 2.0 * b + logcosh(incoming_first + d) - logcosh(incoming_first - d)
    return to_first, to_second


def channel_metric(code: np.ndarray, llr: np.ndarray) -> np.ndarray:
    """Per-bit log-likelihood score (common constant dropped).

    ``code`` has shape (candidates, n) with bits in {0, 1}; positive LLRs
    favour bit 0, so the spin is ``1 - 2*bit``.
    """
    spins = 1.0 - 2.0 * np.asarray(code, dtype=np.float64)
    return spins @ (0.5 * np.asarray(llr, dtype=np.float64))


def joint_metric(
    code: np.ndarray, a: np.ndarray, b: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Exact Gray-pair log-likelihood score (common constant dropped)."""
    spins = 1.0 - 2.0 * np.asarray(code, dtype=np.float64)
    first = spins[:, 0::2]
    second = spins[:, 1::2]
    return np.sum(a * first + b * second + d * first * second, axis=1)


def encode_info(info: np.ndarray, a_columns: np.ndarray) -> np.ndarray:
    """Encode information bits with the systematic H=[A|I] parity equations."""
    info = np.asarray(info, dtype=np.uint8)
    parity = np.zeros(a_columns.shape[1], dtype=np.uint8)
    for index in np.flatnonzero(info):
        parity ^= a_columns[index]
    return parity


def candidate_scores(
    code: np.ndarray,
    llr: np.ndarray,
    a: np.ndarray | None,
    b: np.ndarray | None,
    d: np.ndarray | None,
) -> np.ndarray:
    """Rank candidate codewords under the per-bit or exact joint channel model."""
    if a is None:
        return channel_metric(code, llr)
    return joint_metric(code, a, b, d)


def osd_refine(
    info: np.ndarray,
    posterior: np.ndarray,
    a_columns: np.ndarray,
    llr: np.ndarray,
    pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    order: int,
    lr_bits: int,
    chunk: int,
) -> np.ndarray:
    """BP-guided ordered-search refinement over the least reliable info bits.

    Candidates are systematic codewords obtained by flipping 1, 2 or 3 of the
    ``lr_bits`` least reliable information bits.  The selected candidate is the
    one with the best exact channel metric.  This is deliberately a bounded
    offline search, not a claim of true ML decoding.
    """
    if order <= 0:
        return info
    frames, k = info.shape
    count = min(int(lr_bits), k)
    has_pair_tables = pair_tables is not None
    a_t, b_t, d_t = pair_tables if has_pair_tables else (None, None, None)
    refined = info.copy()

    for frame in range(frames):
        base_info = info[frame]
        base_parity = encode_info(base_info, a_columns)
        base_code = np.concatenate((base_info, base_parity))

        def score_candidates(codes: np.ndarray) -> np.ndarray:
            """Use exact pair scoring only when pair tables are available."""
            if not has_pair_tables:
                return channel_metric(codes, llr[frame])
            return joint_metric(codes, a_t[frame], b_t[frame], d_t[frame])

        base_metric = float(score_candidates(base_code[None, :])[0])
        reliable = np.argsort(np.abs(posterior[frame, :k]), kind="stable")[:count]

        # OSD-r includes every lower-order candidate, not only flips of rank r.
        blocks: list[np.ndarray] = [reliable[:, None]]
        if order >= 2:
            first, second = np.triu_indices(count, k=1)
            blocks.append(np.column_stack((reliable[first], reliable[second])))
        if order >= 3:
            first, second, third = np.triu_indices(count, k=3)
            blocks.append(
                np.column_stack((reliable[first], reliable[second], reliable[third]))
            )

        best_metric = base_metric
        best_info = base_info.copy()
        for flips in blocks:
            for start in range(0, flips.shape[0], chunk):
                block = flips[start : start + chunk]
                size = block.shape[0]
                candidates = np.repeat(base_info[None, :], size, axis=0)
                parity = np.repeat(base_parity[None, :], size, axis=0)
                for column in range(block.shape[1]):
                    positions = block[:, column]
                    candidates[np.arange(size), positions] ^= 1
                    parity ^= a_columns[positions]
                codes = np.concatenate((candidates, parity), axis=1)
                metrics = score_candidates(codes)
                index = int(np.argmax(metrics))
                if metrics[index] > best_metric:
                    best_metric = float(metrics[index])
                    best_info = candidates[index].copy()
        refined[frame] = best_info

    return refined


def decode_case(
    case: dict,
    joint: bool,
    max_iter: int,
    osd_order: int = 0,
    osd_lr_bits: int = DEFAULT_LR_BITS,
    osd_chunk: int = DEFAULT_CHUNK,
) -> tuple[np.ndarray, dict]:
    """Decode one fixed-rule case and return per-frame info-bit errors."""
    llr = np.asarray(case["llr"], dtype=np.float64)
    frames, n = llr.shape
    k = int(case["k"])
    graph = build_check_graph(case["h"])
    a_columns = np.asarray(case["h"][:, :k], dtype=np.uint8).T.copy()

    pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if joint:
        a, b, d = qam_pairs.pair_tables(llr, float(case["snr_db"]))
        pair_tables = (a, b, d)
        first, second = pair_marginal_llrs(a, b, d)
        channel = interleave(first, second)
        v2p = np.zeros((frames, n // 2, 2), dtype=np.float64)
        max_pair_error = float(
            np.max(np.abs(qam_pairs.second_bit_llr_y(
                qam_pairs.recover_amplitude(llr[:, 0::2], qam_pairs.nominal_sigma(case["snr_db"])),
                qam_pairs.nominal_sigma(case["snr_db"]),
            ) - llr[:, 1::2]))
        )
    else:
        channel = llr
        v2p = None
        max_pair_error = None

    v2c = channel[:, graph.neighbors].copy()
    converged = np.zeros(frames, dtype=bool)
    posterior = channel.copy()
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        c2v = check_update(v2c).reshape(frames, graph.edge_count)
        contributions = (graph.incidence @ c2v.T).T

        if joint:
            to_first, to_second = pair_factor_messages(a, b, d, v2p)
            channel = interleave(to_first, to_second)
            posterior = channel + contributions
            v2p[:, :, 0] = posterior[:, 0::2] - to_first
            v2p[:, :, 1] = posterior[:, 1::2] - to_second
        else:
            posterior = channel + contributions

        hard = (posterior < 0).astype(np.uint8)
        syndrome_sum = np.mod(hard[:, graph.neighbors].sum(axis=2), 2).sum(axis=1)
        converged |= syndrome_sum == 0

        if iteration == max_iter or converged.all():
            break
        v2c = posterior[:, graph.neighbors] - c2v.reshape(
            frames, graph.neighbors.shape[0], graph.neighbors.shape[1]
        )

    hard_info = (posterior[:, :k] < 0).astype(np.uint8)
    if osd_order > 0:
        hard_info = osd_refine(
            hard_info, posterior, a_columns, llr, pair_tables, osd_order, osd_lr_bits, osd_chunk
        )

    truth = np.asarray(case["answers"], dtype=np.uint8)[:, :k]
    errors = np.count_nonzero(hard_info != truth, axis=1).astype(np.int64)
    stats = {
        "iterations": iterations,
        "converged_frames": int(converged.sum()),
        "frames": frames,
        "syndrome_valid_fraction": float(converged.mean()),
        "pair_consistency_max_abs_error": max_pair_error,
        "osd_order": int(osd_order),
        "osd_lr_bits": int(osd_lr_bits) if osd_order > 0 else 0,
    }
    return errors, stats


def self_test() -> dict:
    """Verify the joint pair factor reproduces exact per-bit marginal LLRs."""
    rng = np.random.default_rng(14014)
    for snr in (6.0, 9.0, 10.5):
        sigma = qam_pairs.nominal_sigma(snr)
        y = rng.uniform(-2.0, 2.0, size=4096)
        first_llr = qam_pairs.first_bit_llr_y(y, sigma)
        second_llr = qam_pairs.second_bit_llr_y(y, sigma)
        llr = np.column_stack((first_llr, second_llr, first_llr, second_llr))
        tables = qam_pairs.pair_tables(llr, snr)
        a, b, d = (table[:, 0] for table in tables)
        reconstructed_first, reconstructed_second = pair_marginal_llrs(a, b, d)
        if np.max(np.abs(reconstructed_first - first_llr)) > 1e-8:
            raise AssertionError(f"first-bit marginal mismatch at SNR {snr}")
        if np.max(np.abs(reconstructed_second - second_llr)) > 1e-8:
            raise AssertionError(f"second-bit marginal mismatch at SNR {snr}")
        # With zero extrinsic messages, the factor-to-variable messages must be
        # exactly the original independent channel LLRs.
        to_first, to_second = pair_factor_messages(
            a[:, None], b[:, None], d[:, None], np.zeros((llr.shape[0], 1, 2))
        )
        if np.max(np.abs(to_first[:, 0] - first_llr)) > 1e-8:
            raise AssertionError(f"joint BP channel message mismatch at SNR {snr}")
        if np.max(np.abs(to_second[:, 0] - second_llr)) > 1e-8:
            raise AssertionError(f"joint BP channel message mismatch at SNR {snr}")
    return {
        "checked_snr_values": [6.0, 9.0, 10.5],
        "max_marginal_error_tolerance": 1e-8,
        "status": "pass",
    }


def tool_hash(variant: str, max_iter: int, osd_order: int, osd_lr_bits: int) -> str:
    """Stable fingerprint of this decoder configuration for cache metadata."""
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    config = json.dumps(
        {
            "max_iter": max_iter,
            "osd_lr_bits": osd_lr_bits,
            "osd_order": osd_order,
            "variant": variant,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest.update(config)
    return digest.hexdigest()


def parse_cases(specs: list[str] | None) -> list[tuple[int, int, float]]:
    """Resolve CLI case specs to (n, k, snr); default is the Track B ladder."""
    available = {(n, k): snr for n, k, snr in CASE_SPECS}
    if not specs:
        wanted = list(TARGET_CASES)
    else:
        wanted = []
        for token in specs:
            token = token.strip()
            if token.lower() == "all":
                wanted = [(n, k) for n, k, _ in CASE_SPECS]
                break
            if token.isdigit():
                index = int(token)
                if not 1 <= index <= len(CASE_SPECS):
                    raise ValueError(f"case index {index} outside 1..{len(CASE_SPECS)}")
                n, k, _ = CASE_SPECS[index - 1]
                wanted.append((n, k))
                continue
            match_k = re.fullmatch(r"N(\d+)_[Kk](\d+)", token, re.IGNORECASE)
            match_r = re.fullmatch(
                r"N(\d+)_[Rr](\d+)[-_.](\d+)", token, re.IGNORECASE
            )
            match_r_decimal = re.fullmatch(
                r"N(\d+)_[Rr](0?\.\d+|\d+\.\d+)", token, re.IGNORECASE
            )
            match_pair = re.fullmatch(r"(\d+)[x,](\d+)", token, re.IGNORECASE)
            if match_k:
                n, k = int(match_k.group(1)), int(match_k.group(2))
            elif match_r:
                n = int(match_r.group(1))
                numerator, denominator = int(match_r.group(2)), int(match_r.group(3))
                if denominator == 0:
                    raise ValueError(f"invalid rate in case {token!r}")
                rate = numerator / denominator if numerator else float(f"0.{denominator}")
                k = int(round(n * rate))
            elif match_r_decimal:
                n = int(match_r_decimal.group(1))
                k = int(round(n * float(match_r_decimal.group(2))))
            elif match_pair:
                n, k = int(match_pair.group(1)), int(match_pair.group(2))
            else:
                raise ValueError(f"cannot parse case {token!r}")
            if (n, k) not in available:
                raise ValueError(f"unknown case {token!r} -> N={n} K={k}")
            wanted.append((n, k))

    unique: list[tuple[int, int]] = []
    for item in wanted:
        if item not in unique:
            unique.append(item)
    return [(n, k, available[(n, k)]) for n, k in unique]


def cache_one(
    n: int,
    k: int,
    snr: float,
    seed: int,
    frames: int,
    joint: bool,
    cache_dir: Path,
    max_iter: int,
    osd_order: int,
    osd_lr_bits: int,
    osd_chunk: int,
    skip_existing: bool,
) -> dict:
    """Decode and cache one case-seed pair with fixed bench rules."""
    variant = "joint" if joint else "perbit"
    fingerprint = tool_hash(variant, max_iter, osd_order, osd_lr_bits)
    stem = f"{case_key(n, k)}_seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = cache_dir / f"{stem}.npy"
    metadata_path = cache_dir / f"{stem}.json"

    if skip_existing and array_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("binary_sha256") == fingerprint
            and int(existing.get("samples", -1)) == frames
            and int(existing.get("n", -1)) == n
            and int(existing.get("k", -1)) == k
            and int(existing.get("seed", -1)) == seed
        ):
            print(f"skip {variant} {stem}: matching cache", flush=True)
            return existing

    started = time.perf_counter()
    case = get_case(n, k, seed, frames)
    if not math.isclose(case["snr_db"], snr, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"SNR mismatch for N={n} K={k}")
    errors, stats = decode_case(
        case, joint, max_iter, osd_order, osd_lr_bits, osd_chunk
    )
    np.save(array_path, errors)
    metadata = {
        "array": array_path.name,
        "binary": "tools/ceiling/bp_variants.py",
        "binary_sha256": fingerprint,
        "frames": frames,
        "k": k,
        "n": n,
        "noise_scale": NOISE_SCALE,
        "offline_header": OFFLINE_HEADER,
        "samples": frames,
        "score": score_from_errors(errors, k),
        "seconds": round(time.perf_counter() - started, 3),
        "seed": seed,
        "snr": snr,
        "tool": "tools/ceiling/bp_variants.py",
        "total_info_errors": int(errors.sum()),
        "total_wall_seconds": round(time.perf_counter() - started, 3),
        "variant": variant,
        **stats,
    }
    write_json(metadata_path, metadata)
    print(
        f"{variant} {stem}: {int(errors.sum())} errors, score {metadata['score']:,.1f}, "
        f"{metadata['seconds']:.1f}s -> {array_path}",
        flush=True,
    )
    return metadata


def default_cache_dir(joint: bool, frames: int, osd_order: int, osd_lr_bits: int) -> Path:
    """Fixed-ladder dirs are stable; screens and OSD runs get suffixed dirs."""
    name = "artifacts/ceiling_cache_bp_joint" if joint else "artifacts/ceiling_cache_bp_perbit"
    if frames != FRAMES:
        name += f"_f{frames}"
    if osd_order > 0:
        name += f"_osd{osd_order}_lr{osd_lr_bits}"
    return ROOT / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", action="store_true", help="run exact Gray-pair joint BP")
    parser.add_argument("--per-bit", action="store_true", help="run original per-bit LLR BP")
    parser.add_argument("--both", action="store_true", help="run both variants")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--osd-order", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--osd-lr-bits", type=int, default=DEFAULT_LR_BITS)
    parser.add_argument("--osd-chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    check_offline_header(Path(__file__))
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True), flush=True)
        return
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    bad_seeds = sorted(set(args.seeds) - set(SEEDS))
    if bad_seeds:
        raise SystemExit(f"fixed bench seeds must be within {SEEDS}; got {bad_seeds}")
    if args.max_iter <= 0 or args.osd_lr_bits <= 0 or args.osd_chunk <= 0:
        raise SystemExit("iteration, LR-bit and chunk counts must be positive")
    if args.both:
        variants = [False, True]
    elif args.joint and args.per_bit:
        variants = [True, False]
    elif args.joint:
        variants = [True]
    elif args.per_bit:
        variants = [False]
    else:
        variants = [False]

    cases = parse_cases(args.cases)
    all_rows: list[dict] = []
    for joint in variants:
        variant = "joint" if joint else "perbit"
        cache_dir = resolve_path(args.cache_dir) if args.cache_dir else default_cache_dir(
            joint, args.frames, args.osd_order, args.osd_lr_bits
        )
        print(
            f"{variant}: {len(cases)} cases, {len(args.seeds)} seeds, {args.frames} frames, "
            f"max_iter={args.max_iter}, osd_order={args.osd_order} -> {cache_dir}",
            flush=True,
        )
        for n, k, snr in cases:
            for seed in args.seeds:
                row = cache_one(
                    n,
                    k,
                    snr,
                    seed,
                    args.frames,
                    joint,
                    cache_dir,
                    args.max_iter,
                    args.osd_order,
                    args.osd_lr_bits,
                    args.osd_chunk,
                    args.skip_existing,
                )
                row["variant"] = variant
                all_rows.append(row)

    suffix = f"f{args.frames}"
    if args.osd_order:
        suffix += f"_osd{args.osd_order}_lr{args.osd_lr_bits}"
    manifest_path = resolve_path(args.manifest_out) if args.manifest_out else (
        ROOT / "artifacts" / "phase14" / f"track_b_bp_{suffix}.json"
    )
    manifest = {
        "cases": [{"n": n, "k": k, "snr": snr} for n, k, snr in cases],
        "frames": args.frames,
        "max_iter": args.max_iter,
        "noise_scale": NOISE_SCALE,
        "offline_header": OFFLINE_HEADER,
        "osd_chunk": args.osd_chunk,
        "osd_lr_bits": args.osd_lr_bits,
        "osd_order": args.osd_order,
        "rows": all_rows,
        "seeds": list(args.seeds),
        "tool": "tools/ceiling/bp_variants.py",
    }
    write_json(manifest_path, manifest)
    print(f"Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
