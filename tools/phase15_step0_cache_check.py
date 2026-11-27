#!/usr/bin/env python3
"""Step 0 proof that the Phase 15 v17 baseline cache is repeatable.

Checks all 32 expected cache/metadata pairs, records the exact reference binary
hash, regenerates one cache into a temporary directory, and verifies that the
per-frame error array is bit-identical to the stored one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(TOOLS_DIR))

from baseline_cache import cache_path, generate_one, sha256  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_CACHE = ROOT / "artifacts" / "baseline_cache_v17"
DEFAULT_BINARY = ROOT / "artifacts" / "build" / "v17_ref.exe"
DEFAULT_SOURCE = ROOT / "submissions" / "v17_joint_pairs" / "submission.cpp"
DEFAULT_OUT = ROOT / "artifacts" / "phase15" / "step0_cache_check.json"
CASES = (
    (648, 324, 0.5, 6.0),
    (648, 432, 2.0 / 3.0, 9.0),
    (1296, 648, 0.5, 6.0),
    (1296, 864, 2.0 / 3.0, 9.0),
    (1944, 972, 0.5, 6.0),
    (1944, 1296, 2.0 / 3.0, 9.0),
    (3888, 1944, 0.5, 6.0),
    (3888, 2592, 2.0 / 3.0, 9.0),
)
SEEDS = (911, 2024, 31337, 6060)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    binary_hash = sha256(args.binary)
    source_hash = sha256(args.source)
    expected_source_hash = "a80aef906b31a97cdc061667af6a13cdb18f136572c85b4b09242c8be6bb76d6"

    inventory: list[dict] = []
    missing: list[str] = []
    for n, k, rate, snr in CASES:
        for seed in SEEDS:
            npy = cache_path(args.cache_dir, n, k, seed)
            meta_path = npy.with_suffix(".json")
            if not npy.exists() or not meta_path.exists():
                missing.append(str(npy))
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            errors = np.load(npy)
            expected_frames = 300 if seed == 911 else 500
            inventory.append({
                "n": n,
                "k": k,
                "rate": rate,
                "snr": snr,
                "seed": seed,
                "frames": int(len(errors)),
                "expected_frames": expected_frames,
                "errors": int(errors.sum()),
                "score": float(meta["score"]),
                "noise_scale": float(meta["noise_scale"]),
                "llr_sigma": meta["llr_sigma"],
                "binary_sha256": meta["binary_sha256"],
            })

    source_ok = bool(source_hash.lower() == expected_source_hash)
    inventory_ok = (
        source_ok
        and not missing
        and len(inventory) == 32
        and all(item["frames"] == item["expected_frames"] for item in inventory)
        and all(abs(item["noise_scale"] - 1.00112) < 1e-12 for item in inventory)
        and all(item["llr_sigma"] == "nominal" for item in inventory)
        and all(item["binary_sha256"].lower() == binary_hash.lower() for item in inventory)
    )

    repeat_dir = args.out.parent / "cache_repeat"
    repeated = generate_one(
        1296, 648, 6.0, 911, 300, args.binary, 1.00112, repeat_dir, "nominal"
    )
    original_errors = np.load(cache_path(args.cache_dir, 1296, 648, 911))
    repeated_errors = np.load(cache_path(repeat_dir, 1296, 648, 911))
    repeat_identical = bool(np.array_equal(original_errors, repeated_errors))

    original_hash = hashlib.sha256(original_errors.tobytes()).hexdigest()
    repeated_hash = hashlib.sha256(repeated_errors.tobytes()).hexdigest()
    payload = {
        "all_passed": bool(inventory_ok and repeat_identical),
        "cache_dir": str(args.cache_dir),
        "binary": str(args.binary),
        "binary_sha256": binary_hash,
        "source": str(args.source),
        "source_sha256": source_hash,
        "source_ok": source_ok,
        "expected_cache_files": 32,
        "inventory_files": len(inventory),
        "missing": missing,
        "inventory_ok": bool(inventory_ok),
        "repeat_case": {"n": 1296, "k": 648, "seed": 911, "frames": 300},
        "repeat_identical": repeat_identical,
        "original_array_sha256": original_hash,
        "repeated_array_sha256": repeated_hash,
        "repeated_score": repeated["score"],
        "inventory": inventory,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "all_passed", "cache_dir", "binary_sha256", "source_sha256", "source_ok",
        "inventory_files", "inventory_ok", "repeat_identical",
        "original_array_sha256", "repeated_array_sha256",
    )}, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
