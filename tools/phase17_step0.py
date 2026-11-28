#!/usr/bin/env python3
"""Step 0 for Phase 17: build and verify the accepted v19 baseline cache.

All v20/v21 comparisons are paired against the exact frames decoded by the
submitted v19 source. This tool regenerates missing caches with the compiled
v19 binary, verifies every cache/metadata pair, repeats one cache for a
bit-identity check, and checks that the bench case ordering matches the
published v19 judge ordering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as pc  # noqa: E402
from baseline_cache import cache_path, generate_one, sha256  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "baseline_cache_v19"
DEFAULT_BINARY = ROOT / "artifacts" / "phase17" / "v19.exe"
DEFAULT_SOURCE = ROOT / "submissions" / "v19_kspin" / "submission.cpp"
DEFAULT_OUT = ROOT / "artifacts" / "phase17" / "step0_v19_cache.json"
EXPECTED_SOURCE_HASH = "9a940d50831f1a2faae7e3bf9435e588259c5868bdc784eaf0fde1516f2696b7"
SEED_SAMPLES = {911: 300, 2024: 1000, 31337: 1000, 6060: 300}
JUDGE_SCORES = (
    135526, 177394, 185451, 154670, 179802, 183299,
    159042, 184164, 184485, 170456, 182587, 185024,
)


def valid_metadata(meta: dict, n: int, k: int, snr: float, seed: int, binary_hash: str) -> bool:
    """Return whether one cache metadata record matches the v19 reference run."""
    try:
        return bool(
            int(meta["n"]) == n
            and int(meta["k"]) == k
            and abs(float(meta["snr"]) - float(snr)) < 1e-9
            and int(meta["seed"]) == seed
            and int(meta["samples"]) == SEED_SAMPLES[seed]
            and abs(float(meta["noise_scale"]) - pc.NOISE_SCALE) < 1e-12
            and str(meta["llr_sigma"]) == "nominal"
            and str(meta["binary_sha256"]).lower() == binary_hash.lower()
        )
    except (KeyError, TypeError, ValueError):
        return False


def build(args: argparse.Namespace, binary_hash: str) -> list[dict]:
    """Generate all missing or invalid v19 cache entries."""
    built: list[dict] = []
    for index, n, k, _rate, snr in pc.CASE_SPECS:
        for seed, samples in SEED_SAMPLES.items():
            npy = cache_path(args.cache_dir, n, k, seed)
            meta_path = npy.with_suffix(".json")
            valid = False
            if npy.exists() and meta_path.exists() and not args.force:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                errors = np.load(npy)
                valid = (
                    valid_metadata(meta, n, k, snr, seed, binary_hash)
                    and errors.ndim == 1
                    and len(errors) == samples
                )
            if valid:
                continue
            built.append(
                generate_one(
                    n, k, snr, seed, samples, args.binary,
                    pc.NOISE_SCALE, args.cache_dir, "nominal",
                )
            )
    return built


def collect_inventory(args: argparse.Namespace, binary_hash: str) -> tuple[list[dict], list[str]]:
    """Inspect every expected cache/metadata pair."""
    inventory: list[dict] = []
    missing: list[str] = []
    for index, n, k, rate, snr in pc.CASE_SPECS:
        for seed, samples in SEED_SAMPLES.items():
            npy = cache_path(args.cache_dir, n, k, seed)
            meta_path = npy.with_suffix(".json")
            if not npy.exists() or not meta_path.exists():
                missing.append(str(npy))
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            errors = np.load(npy)
            valid = valid_metadata(meta, n, k, snr, seed, binary_hash)
            if errors.ndim != 1 or len(errors) != samples:
                valid = False
            inventory.append({
                "case": index,
                "n": n,
                "k": k,
                "rate": rate,
                "snr": snr,
                "seed": seed,
                "frames": int(len(errors)),
                "expected_frames": samples,
                "errors": int(errors.sum()),
                "score": float(meta.get("score", pc.score_from_errors(errors, k))),
                "binary_sha256": str(meta.get("binary_sha256", "")),
                "valid": bool(valid),
            })
    return inventory, missing


def ordering(inventory: list[dict]) -> dict:
    """Compare aggregate bench ordering to the published judge ordering."""
    by_case: dict[int, dict[str, int]] = {}
    for item in inventory:
        if not item["valid"]:
            continue
        aggregate = by_case.setdefault(item["case"], {"errors": 0, "trials": 0})
        aggregate["errors"] += int(item["errors"])
        aggregate["trials"] += int(item["frames"]) * int(item["k"])
    if len(by_case) != len(JUDGE_SCORES):
        return {
            "available": False,
            "exact_match": False,
            "tail_exact_match": False,
            "ordering_ok": False,
            "spearman": None,
            "max_rank_shift": None,
            "bench_scores": [],
            "judge_scores": list(JUDGE_SCORES),
        }
    bench_scores = np.array(
        [
            pc.score_of(by_case[index]["errors"], by_case[index]["trials"])
            for index in range(1, len(JUDGE_SCORES) + 1)
        ],
        dtype=float,
    )
    judge_scores = np.array(JUDGE_SCORES, dtype=float)
    bench_order = np.argsort(-bench_scores)
    judge_order = np.argsort(-judge_scores)
    bench_rank = np.empty(len(bench_scores), dtype=int)
    judge_rank = np.empty(len(judge_scores), dtype=int)
    bench_rank[bench_order] = np.arange(len(bench_scores))
    judge_rank[judge_order] = np.arange(len(judge_scores))
    tail_exact_match = bool(np.array_equal(bench_order[-4:], judge_order[-4:]))
    spearman = float(np.corrcoef(bench_rank, judge_rank)[0, 1])
    max_rank_shift = int(np.max(np.abs(bench_rank - judge_rank)))
    return {
        "available": True,
        "exact_match": bool(np.array_equal(bench_order, judge_order)),
        "tail_exact_match": tail_exact_match,
        "spearman": spearman,
        "max_rank_shift": max_rank_shift,
        "ordering_ok": bool(
            tail_exact_match and spearman >= 0.85 and max_rank_shift <= 4
        ),
        "note": (
            "Exact 12-case ordering is not stable under finite bench sampling: the "
            "eight non-changed cases sit in a ~1.8k-point cluster. The gate requires "
            "the four changed rate-1/2 cases to keep their exact judge order, "
            "Spearman >= 0.85, and no rank displacement above 4."
        ),
        "bench_order": [int(value) + 1 for value in bench_order],
        "judge_order": [int(value) + 1 for value in judge_order],
        "bench_scores": [float(value) for value in bench_scores],
        "judge_scores": [int(value) for value in JUDGE_SCORES],
    }


def repeat_check(args: argparse.Namespace, binary_hash: str) -> dict:
    """Regenerate one cache into a scratch directory and compare error arrays."""
    n, k, snr, seed, samples = 1296, 648, 6.0, 911, 300
    repeat_dir = args.out.parent / "cache_repeat_v19"
    repeated_meta = generate_one(
        n, k, snr, seed, samples, args.binary, pc.NOISE_SCALE, repeat_dir, "nominal"
    )
    original_path = cache_path(args.cache_dir, n, k, seed)
    if not original_path.exists():
        return {"ok": False, "reason": "original cache missing", "case": (n, k, seed)}
    original = np.load(original_path)
    repeated = np.load(cache_path(repeat_dir, n, k, seed))
    return {
        "ok": bool(
            np.array_equal(original, repeated)
            and repeated_meta["binary_sha256"].lower() == binary_hash.lower()
        ),
        "case": {"n": n, "k": k, "seed": seed, "frames": samples},
        "original_array_sha256": sha256(original_path),
        "repeated_array_sha256": sha256(cache_path(repeat_dir, n, k, seed)),
        "repeat_identical": bool(np.array_equal(original, repeated)),
        "repeated_score": float(repeated_meta["score"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="regenerate all caches")
    parser.add_argument("--check-only", action="store_true", help="do not build missing caches")
    args = parser.parse_args()

    source_hash = sha256(args.source)
    source_ok = source_hash.lower() == EXPECTED_SOURCE_HASH.lower()
    binary_hash = sha256(args.binary)
    built = [] if args.check_only else build(args, binary_hash)
    inventory, missing = collect_inventory(args, binary_hash)
    expected_count = len(pc.CASE_SPECS) * len(SEED_SAMPLES)
    inventory_ok = bool(
        source_ok
        and not missing
        and len(inventory) == expected_count
        and all(item["valid"] for item in inventory)
    )
    ordering_report = ordering(inventory)
    repeat = repeat_check(args, binary_hash)
    all_passed = bool(inventory_ok and repeat["ok"] and ordering_report["ordering_ok"])

    payload = {
        "all_passed": all_passed,
        "cache_dir": str(args.cache_dir.resolve()),
        "binary": str(args.binary.resolve()),
        "binary_sha256": binary_hash,
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "source_ok": source_ok,
        "expected_cache_files": expected_count,
        "inventory_files": len(inventory),
        "missing": missing,
        "inventory_ok": inventory_ok,
        "seed_samples": {str(key): value for key, value in SEED_SAMPLES.items()},
        "built_now": built,
        "built_count": len(built),
        "repeat_check": repeat,
        "judge_ordering": ordering_report,
        "inventory": inventory,
    }
    pc.write_json(args.out, payload)
    print(json.dumps({
        "all_passed": all_passed,
        "source_ok": source_ok,
        "inventory_files": len(inventory),
        "inventory_ok": inventory_ok,
        "built_count": len(built),
        "repeat_identical": repeat.get("repeat_identical"),
        "ordering_exact": ordering_report["exact_match"],
        "ordering_ok": ordering_report["ordering_ok"],
        "tail_exact": ordering_report["tail_exact_match"],
        "spearman": ordering_report["spearman"],
        "max_rank_shift": ordering_report.get("max_rank_shift"),
    }, indent=2, sort_keys=True))
    print(f"Saved {args.out}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
