#!/usr/bin/env python3
"""Build realistic LDPC cases: systematic H, coded bits, Gray 16-QAM LLRs.

Public input format after the header line ``n k b snr``:
  m rows of H, b rows of LLR floats, k rows of generator matrix G.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from real_h_generator import generate_h, validate_h  # noqa: E402
from real_qam_channel import generate_llrs, raw_ber  # noqa: E402

ROOT = TOOLS_DIR.parent
N_VALUES = (648, 1296, 1944, 3888)
RATE_SNR_DB = ((0.5, 6.0), (2.0 / 3.0, 9.0), (0.75, 10.5))


def build_case(
    n: int,
    k: int,
    b: int,
    snr_db: float,
    seed: int,
    irregular: float = 0.0,
    noise_scale: float = 1.0,
) -> dict:
    """Return one complete judge-format case with ground-truth answers."""
    n, k, b = int(n), int(k), int(b)
    if not 0 < k < n or b <= 0:
        raise ValueError(f"invalid case dimensions: n={n}, k={k}, b={b}")
    rate = k / n
    rng = np.random.default_rng(seed)

    h = generate_h(n, rate, rng, irregular=float(irregular))
    m = n - k
    a = h[:, :k]
    g = np.concatenate((np.eye(k, dtype=np.int8), a.astype(np.int8).T), axis=1)

    info = rng.integers(0, 2, size=(b, k), dtype=np.int8)
    parity = (info.astype(np.int16) @ a.T.astype(np.int16) % 2).astype(np.int8)
    answers = np.concatenate((info, parity), axis=1)
    llr = generate_llrs(answers, snr_db, rng, noise_scale=noise_scale)

    case = {
        "n": n,
        "k": k,
        "b": b,
        "snr_db": float(snr_db),
        "h": h,
        "g": g,
        "llr": llr,
        "answers": answers,
        "rate": rate,
        "seed": int(seed),
    }

    g_product = (g.astype(np.int16) @ h.T.astype(np.int16)) % 2
    syndrome = (answers.astype(np.int16) @ h.T.astype(np.int16)) % 2
    assert g.shape == (k, n) and h.shape == (m, n)
    assert llr.shape == answers.shape == (b, n)
    assert np.all(g_product == 0) and np.all(syndrome == 0)
    assert np.all(np.isfinite(llr))
    return case


def write_input(case: dict, path: Path | str) -> None:
    """Write H, LLRs and G in the judge input format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{case['n']} {case['k']} {case['b']} {case['snr_db']:g}"]
    lines.extend(" ".join(str(int(value)) for value in row) for row in case["h"])
    lines.extend(" ".join(f"{float(value):.17g}" for value in row) for row in case["llr"])
    lines.extend(" ".join(str(int(value)) for value in row) for row in case["g"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_answers(case: dict, path: Path | str) -> None:
    """Write the ground-truth codewords, one space-separated row per frame."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(int(value)) for value in row) for row in case["answers"]]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def case_stats(case: dict) -> dict:
    """Summarize channel and H structure without changing the case."""
    errors = (case["llr"] < 0).astype(np.int8) != case["answers"]
    k = case["k"]
    h_stats = validate_h(case["h"], case["rate"])
    return {
        "n": case["n"],
        "k": k,
        "b": case["b"],
        "rate": case["rate"],
        "snr_db": case["snr_db"],
        "seed": case["seed"],
        "raw_ber": float(errors.mean()),
        "info_raw_ber": float(errors[:, :k].mean()),
        "parity_raw_ber": float(errors[:, k:].mean()),
        "per_position_ber": errors.mean(axis=0).tolist(),
        "llr_mean_absolute": float(np.mean(np.abs(case["llr"]))),
        "llr_median_absolute": float(np.median(np.abs(case["llr"]))),
        "h_validation": h_stats,
    }


def _write_stats(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Stats written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=648)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--irregular", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--input-out", type=Path, default=None)
    parser.add_argument("--answers-out", type=Path, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="emit statistics for all 12 (N, rate) cases instead of one case",
    )
    args = parser.parse_args()

    if args.all_cases:
        entries: list[dict] = []
        for n in N_VALUES:
            for rate, snr_db in RATE_SNR_DB:
                k = int(round(n * rate))
                index = len(entries)
                case = build_case(
                    n,
                    k,
                    args.samples,
                    snr_db,
                    args.seed + index,
                    irregular=args.irregular,
                    noise_scale=args.noise_scale,
                )
                entry = case_stats(case)
                entry["case_index"] = index
                entries.append(entry)
                print(
                    f"case {index:>2d}: N={n:>4d} K={k:>4d} SNR={snr_db:>4.1f} "
                    f"raw_ber={entry['raw_ber']:.6f}"
                )
        stats_path = args.stats_out or ROOT / "artifacts" / "real_bench_all_cases_stats.json"
        _write_stats({"all_cases": entries}, stats_path)
        return

    k = int(round(args.n * args.rate))
    case = build_case(
        args.n,
        k,
        args.samples,
        args.snr,
        args.seed,
        irregular=args.irregular,
        noise_scale=args.noise_scale,
    )
    if args.input_out is not None:
        write_input(case, args.input_out)
        print(f"Input written to {args.input_out}")
    if args.answers_out is not None:
        write_answers(case, args.answers_out)
        print(f"Answers written to {args.answers_out}")

    stats = case_stats(case)
    stats_path = args.stats_out or ROOT / "artifacts" / "real_bench_stats.json"
    _write_stats(stats, stats_path)
    print(
        f"N={case['n']} K={case['k']} B={case['b']} SNR={case['snr_db']:g} "
        f"raw_ber={stats['raw_ber']:.6f}"
    )


if __name__ == "__main__":
    main()
