#!/usr/bin/env python3
"""Evaluate historical decoder manifests on the realistic bench.

The script builds one shared set of cases for every (N, K, SNR) operating
point, decodes them with each historical manifest, and checks whether the
resulting score deltas agree in sign with the judge deltas recorded during the
submission history.

Note: the exact per-case judge score array for v3 was never recorded.  The
v3 -> v2 and v4 -> v3 comparisons therefore use the known aggregate signs from
the history instead of invented magnitudes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

try:
    from decode_v2_python import decode_frame, select_case  # noqa: E402
    from phase0_harness import Instance  # noqa: E402
    from phase3_fixed_dsb import CheckTable  # noqa: E402
    from real_bench import build_case, write_answers, write_input  # noqa: E402,F401
except ModuleNotFoundError as exc:
    if exc.name == "real_bench":
        raise SystemExit(
            "tools/real_bench.py is required to run this evaluator; "
            "it must provide build_case(), write_input(), and write_answers()."
        ) from exc
    raise


SCORE_SCALE = 100_000.0
SIGN_EPS = 1e-6

CASE_SPECS: tuple[tuple[int, int, float], ...] = (
    (648, 324, 6.0),
    (648, 432, 9.0),
    (648, 486, 10.5),
    (1296, 648, 6.0),
    (1296, 864, 9.0),
    (1296, 972, 10.5),
    (1944, 972, 6.0),
    (1944, 1296, 9.0),
    (1944, 1458, 10.5),
    (3888, 1944, 6.0),
    (3888, 2592, 9.0),
    (3888, 2916, 10.5),
)

VERSION_ORDER: tuple[str, ...] = (
    "v1",
    "v2",
    "v3",
    "v4",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
)

DEFAULT_VERSIONS = ",".join(VERSION_ORDER)

DEFAULT_MANIFESTS: dict[str, str] = {
    "v1": "artifacts/phase7_case_manifest_v1.json",
    "v2": "submissions/v2_per_case_all/manifest.json",
    "v3": "submissions/v3_hybrid_fallback/manifest.json",
    "v4": "submissions/v4_hybrid_probe/manifest.json",
    "v6": "artifacts/phase7_case_manifest_v6.json",
    "v7": "artifacts/phase7_case_manifest_v7.json",
    "v8": "artifacts/phase7_case_manifest_v8.json",
    "v9": "artifacts/phase7_case_manifest_v9.json",
    "v10": "artifacts/phase7_case_manifest_v10.json",
    "v11": "artifacts/phase7_case_manifest_v11_rebased.json",
}

# Judge per-case scores in historical case order.  None means the exact array
# was not preserved; the available sign is recorded in JUDGE_DELTA_SIGNS.
REFERENCE: dict[str, list[int] | None] = {
    "v1": [
        90167, 135404, 154193, 90957, 134979, 152854,
        89867, 134823, 153758, 89806, 135239, 153825,
    ],
    "v2": [
        86433, 142849, 168188, 88223, 141894, 165603,
        85629, 142582, 166013, 86150, 139539, 166455,
    ],
    "v3": None,
    "v4": [
        90167, 142849, 168188, 90957, 141894, 165603,
        89867, 142582, 166013, 89806, 139539, 166455,
    ],
    "v6": [
        90167, 142849, 168188, 90957, 141894, 165603,
        89867, 142582, 166013, 89806, 142852, 166455,
    ],
    "v7": [
        90167, 144269, 168188, 90957, 143207, 165603,
        89867, 142847, 166013, 89806, 142852, 166455,
    ],
    "v8": [
        90167, 144269, 168188, 90957, 143207, 165603,
        89867, 142847, 167085, 89806, 142852, 166455,
    ],
    "v9": [
        90167, 144269, 171480, 90957, 143207, 167094,
        89867, 142847, 167085, 89806, 142852, 167210,
    ],
    "v10": [
        90167, 144269, 171480, 90957, 143578, 167094,
        89867, 142847, 167085, 89806, 141113, 167210,
    ],
    "v11": [
        90167, 144269, 171480, 90957, 143578, 166457,
        89867, 142847, 167085, 89806, 142852, 167210,
    ],
}

REFERENCE_TOTALS: dict[str, int] = {
    "v1": 1_515_872,
    "v2": 1_579_558,
    "v3": 1_588_472,
    "v4": 1_593_920,
    "v6": 1_597_233,
    "v7": 1_600_231,
    "v8": 1_601_303,
    "v9": 1_606_841,
    "v10": 1_605_473,
    "v11": 1_606_575,
}

COMPARISON_VERSION: dict[str, str | None] = {
    "v1": None,
    "v2": "v1",
    "v3": "v2",
    "v4": "v3",
    "v6": "v4",
    "v7": "v6",
    "v8": "v7",
    "v9": "v8",
    "v10": "v9",
    "v11": "v9",
}

# Only used when exact per-case judge scores are unavailable.  Values are
# delta signs: +1 gain, -1 regression, 0 unchanged.
JUDGE_DELTA_SIGNS: dict[tuple[str, str], tuple[int | None, ...]] = {
    # v3 restored most of the accepted R=1/2 decoder but used the wrong local
    # fallback, so only the four R=1/2 cases moved relative to v2.
    ("v3", "v2"): (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
    # v4 used the exact accepted probe fallback, gaining the remaining R=1/2
    # points; all high-rate cases were unchanged.
    ("v4", "v3"): (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
}


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/")).expanduser()
    return path if path.is_absolute() else ROOT / path


def _normalize_version(value: str) -> str:
    version = value.strip().lower()
    if not version:
        raise ValueError("empty version in --versions")
    return version if version.startswith("v") else f"v{version}"


def _parse_versions(value: str) -> list[str]:
    versions = [_normalize_version(item) for item in value.split(",") if item.strip()]
    if not versions:
        raise ValueError("--versions must contain at least one version")
    return list(dict.fromkeys(versions))


def _parse_manifest_overrides(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("--manifests must be a JSON object mapping versions to paths")
    return {_normalize_version(key): str(path) for key, path in data.items()}


def _sign(value: float | int | None) -> int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    if value > SIGN_EPS:
        return 1
    if value < -SIGN_EPS:
        return -1
    return 0


def _load_manifests(
    versions: list[str], overrides: dict[str, str]
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for version in versions:
        raw_path = overrides.get(version, DEFAULT_MANIFESTS.get(version))
        if raw_path is None:
            _warn(f"no known manifest path for {version}; skipping")
            continue
        path = _resolve_path(raw_path)
        if not path.is_file():
            _warn(f"skipping {version}: manifest not found at {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"could not read manifest {path}: {exc}") from exc
        if not isinstance(data, dict) or "cases" not in data or "fallback" not in data:
            raise SystemExit(f"manifest {path} must contain 'cases' and 'fallback'")
        resolved = path.resolve()
        manifests[version] = {
            "data": data,
            "path": resolved,
            # All historical manifests store artifact paths relative to the
            # repository root, regardless of where the manifest itself lives.
            "root": ROOT,
        }
    if not manifests:
        raise SystemExit("no historical manifests were available; nothing to evaluate")
    return manifests


def _prepare_manifest_case(case: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(case)
    for key in ("params", "summary"):
        value = prepared.get(key)
        if isinstance(value, str):
            prepared[key] = value.replace("\\", "/")
    return prepared


def _build_shared_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    shared: list[dict[str, Any]] = []
    required_keys = {"n", "k", "b", "snr_db", "h", "g", "llr", "answers", "rate", "seed"}

    for index, (expected_n, expected_k, snr) in enumerate(CASE_SPECS, start=1):
        raw = build_case(
            expected_n,
            expected_k,
            args.samples,
            snr,
            args.seed,
            irregular=args.irregular,
            noise_scale=args.noise_scale,
        )
        if not isinstance(raw, dict):
            raise SystemExit(f"case {index}: build_case() did not return a dict")
        missing = sorted(required_keys - raw.keys())
        if missing:
            raise SystemExit(f"case {index}: build_case() missing keys: {', '.join(missing)}")

        n = int(raw["n"])
        k = int(raw["k"])
        b = int(raw["b"])
        snr_db = float(raw["snr_db"])
        h = np.asarray(raw["h"], dtype=np.int8)
        g = np.asarray(raw["g"], dtype=np.int8)
        llr = np.asarray(raw["llr"], dtype=np.float64)
        answers = np.asarray(raw["answers"], dtype=np.int8)

        if (n, k, snr_db) != (expected_n, expected_k, snr):
            raise SystemExit(
                f"case {index}: build_case() returned {(n, k, snr_db)}, "
                f"expected {(expected_n, expected_k, snr)}"
            )
        if b != args.samples:
            raise SystemExit(
                f"case {index}: build_case() returned {b} frames, expected {args.samples}"
            )
        if h.shape != (n - k, n):
            raise SystemExit(f"case {index}: unexpected H shape {h.shape}, expected {(n - k, n)}")
        if g.shape != (k, n):
            raise SystemExit(f"case {index}: unexpected G shape {g.shape}, expected {(k, n)}")
        if llr.shape != (b, n):
            raise SystemExit(f"case {index}: unexpected LLR shape {llr.shape}, expected {(b, n)}")
        if answers.ndim == 1 and b == 1:
            answers = answers.reshape(1, -1)
        if answers.ndim != 2 or answers.shape[0] != b:
            raise SystemExit(
                f"case {index}: unexpected answers shape {answers.shape}, expected first dim {b}"
            )
        if answers.shape[1] == n:
            info_answers = answers[:, :k]
        elif answers.shape[1] == k:
            info_answers = answers
        else:
            raise SystemExit(
                f"case {index}: answers width {answers.shape[1]} is neither n={n} nor k={k}"
            )

        instance = Instance(n, k, b, snr_db, h, llr, g, 0)
        shared.append(
            {
                "index": index,
                "n": n,
                "k": k,
                "snr": snr_db,
                "rate": float(raw["rate"]),
                "seed": int(raw["seed"]),
                "instance": instance,
                "answers": answers,
                "info_answers": info_answers,
                "checks": CheckTable(h),
            }
        )
        print(
            f"built case {index:2d}/12: N={n} K={k} SNR={snr_db:g} frames={b}",
            flush=True,
        )
    return shared


def _evaluate_manifest(
    version: str, entry: dict[str, Any], shared: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest = entry["data"]
    root = entry["root"]
    cases: list[dict[str, Any]] = []

    for bench_case in shared:
        instance: Instance = bench_case["instance"]
        n = instance.n
        k = instance.k
        case_config = _prepare_manifest_case(
            select_case(manifest, n, k, bench_case["snr"])
        )

        errors = 0
        for row, llr in enumerate(instance.llr):
            hard = np.asarray(
                decode_frame(llr, instance.h, bench_case["checks"], case_config, root),
                dtype=np.int8,
            ).reshape(-1)
            if hard.size != n:
                raise SystemExit(
                    f"{version} case {bench_case['index']}: decoder returned "
                    f"{hard.size} bits, expected {n}"
                )
            errors += int(
                np.count_nonzero(hard[:k] != bench_case["info_answers"][row])
            )

        bits = instance.b * k
        ber = errors / bits
        floor = 1.0 / bits
        score = -SCORE_SCALE * math.log10(max(ber, floor))
        cases.append(
            {
                "index": bench_case["index"],
                "n": n,
                "k": k,
                "snr": bench_case["snr"],
                "rate": bench_case["rate"],
                "errors": errors,
                "bits": bits,
                "ber": ber,
                "score": score,
            }
        )

    return {
        "version": version,
        "manifest": str(entry["path"]),
        "cases": cases,
        "total_score": sum(case["score"] for case in cases),
        "total_errors": sum(case["errors"] for case in cases),
        "total_bits": sum(case["bits"] for case in cases),
    }


def _judge_delta(
    version: str, comparison: str, index: int
) -> tuple[float | int | None, int | None, str | None]:
    reference = REFERENCE.get(version)
    comparison_reference = REFERENCE.get(comparison)
    if reference is not None and comparison_reference is not None:
        delta = reference[index] - comparison_reference[index]
        return delta, _sign(delta), "score"
    signs = JUDGE_DELTA_SIGNS.get((version, comparison))
    if signs is not None and index < len(signs) and signs[index] is not None:
        value = signs[index]
        return value, value, "sign"
    return None, None, None


def _attach_comparisons(results: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals = {"sign_matches": 0, "sign_mismatches": 0, "sign_skipped": 0}

    for version, result in results.items():
        comparison = COMPARISON_VERSION.get(version)
        result["comparison"] = comparison

        if comparison is not None and comparison not in results:
            _warn(
                f"{version}: comparison version {comparison} was not evaluated; "
                "sign checks skipped"
            )
        base_cases = results[comparison]["cases"] if comparison in results else None
        result["comparison_available"] = base_cases is not None

        matches = mismatches = skipped = 0
        for index, case in enumerate(result["cases"]):
            bench_delta: float | None = None
            bench_sign: int | None = None
            judge_delta: float | int | None = None
            judge_sign: int | None = None
            judge_kind: str | None = None
            sign_match: bool | None = None

            if base_cases is not None:
                bench_delta = case["score"] - base_cases[index]["score"]
                bench_sign = _sign(bench_delta)
                if comparison is not None:
                    judge_delta, judge_sign, judge_kind = _judge_delta(
                        version, comparison, index
                    )
                if judge_sign is not None:
                    sign_match = bench_sign == judge_sign

            case["bench_delta"] = bench_delta
            case["bench_delta_sign"] = bench_sign
            case["judge_delta"] = judge_delta
            case["judge_delta_sign"] = judge_sign
            case["judge_delta_kind"] = judge_kind
            case["sign_match"] = sign_match

            if sign_match is True:
                matches += 1
            elif sign_match is False:
                mismatches += 1
            else:
                skipped += 1

        result["summary"] = {
            "sign_matches": matches,
            "sign_mismatches": mismatches,
            "sign_skipped": skipped,
        }
        totals["sign_matches"] += matches
        totals["sign_mismatches"] += mismatches
        totals["sign_skipped"] += skipped

    return totals


def _print_table(results: dict[str, dict[str, Any]]) -> None:
    headers = ("version", "vs", "bench_total", "match", "mismatch", "skip")
    rows: list[tuple[str, ...]] = []
    for version, result in results.items():
        comparison = result.get("comparison") or "-"
        if result["comparison"] is not None and not result["comparison_available"]:
            comparison += "*"
        summary = result["summary"]
        rows.append(
            (
                version,
                comparison,
                f"{result['total_score']:,.1f}",
                str(summary["sign_matches"]),
                str(summary["sign_mismatches"]),
                str(summary["sign_skipped"]),
            )
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    print("* comparison version was not evaluated")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--irregular", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--versions", type=str, default=DEFAULT_VERSIONS)
    parser.add_argument(
        "--manifests",
        type=str,
        help='JSON object mapping versions to manifest paths, e.g. \'{"v9":"path/to.json"}\'',
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/phase9_history_bench.json")
    )
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    try:
        args.version_list = _parse_versions(args.versions)
        args.manifest_overrides = _parse_manifest_overrides(args.manifests)
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifests = _load_manifests(args.version_list, args.manifest_overrides)
    shared = _build_shared_cases(args)

    results: dict[str, dict[str, Any]] = {}
    for version in args.version_list:
        if version not in manifests:
            continue
        print(f"evaluating {version} from {manifests[version]['path']}", flush=True)
        results[version] = _evaluate_manifest(version, manifests[version], shared)

    totals = _attach_comparisons(results)
    payload = {
        "samples": args.samples,
        "seed": args.seed,
        "irregular": args.irregular,
        "noise_scale": args.noise_scale,
        "score_scale": SCORE_SCALE,
        "score_formula": "-100000*log10(BER)",
        "case_specs": [
            {"n": n, "k": k, "snr": snr} for n, k, snr in CASE_SPECS
        ],
        "judge_reference_totals": REFERENCE_TOTALS,
        "versions": results,
        "summary": totals,
    }

    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    _print_table(results)
    print()
    print(
        "sign agreement: "
        f"{totals['sign_matches']} matches, "
        f"{totals['sign_mismatches']} mismatches, "
        f"{totals['sign_skipped']} skipped"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
