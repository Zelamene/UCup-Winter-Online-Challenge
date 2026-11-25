#!/usr/bin/env python3
"""Reference Python decoder for the Phase 7 per-case C++ submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_instance  # noqa: E402
from phase3_fixed_dsb import CheckTable, DSBConfig  # noqa: E402
from phase4_unfolded_dsb import MOD_KEYS, PARAMETER_KEYS, RawParams, build_features  # noqa: E402
from phase5_generalization import load_params  # noqa: E402
from phase7_sparse_dsb import (  # noqa: E402
    check_products,
    decode_fixed_sparse,
    decode_hard_sparse,
    sparse_access,
)

ROOT = TOOLS_DIR.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase7_case_manifest.json"
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase7_python_public.out"


def select_case(manifest: dict, n: int, k: int, snr: float) -> dict:
    for case in manifest["cases"]:
        if (
            case["n"] == n
            and case["k"] == k
            and abs(float(case["snr"]) - snr) < 0.25
        ):
            return case
    return manifest["fallback"]


def decode_frame(
    llr: np.ndarray,
    h: np.ndarray,
    checks: CheckTable,
    case: dict,
    root: Path,
) -> np.ndarray:
    use_learned = bool(case.get("use_learned", True))
    candidates: list[np.ndarray] = []
    lambdas: list[float] = []
    if use_learned:
        params_path = Path(case["params"])
        if not params_path.is_absolute():
            params_path = root / params_path
        params = load_params(params_path)
        features = build_features(llr[None, :], checks)
        learned_params = RawParams({
            key: np.asarray(params.values[key], dtype=np.float64) for key in PARAMETER_KEYS
        })
        candidates.append(decode_hard_sparse(
            learned_params,
            h.astype(np.float64),
            llr[None, :],
            checks,
            features,
        )[0])
        lambdas.append(float(case["learned_lambda"]))

    for values in case["fixed_candidates"]:
        tau, xi, lambd, delta = (float(value) for value in values[:4])
        config = DSBConfig(
            tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50,
            gradient_sign=1,
        )
        candidates.append(decode_fixed_sparse(config, h, llr[None, :])[0])
        lambdas.append(float(values[4]) if len(values) >= 5 else lambd)

    safe, mask = sparse_access(checks)
    energies = []
    for hard, lambd in zip(candidates, lambdas):
        spins = np.where(hard == 0, 1.0, -1.0)[None, :]
        products = check_products(spins, safe, mask)[0]
        energies.append(
            -float(np.sum(llr * spins[0]))
            + 0.5 * lambd * float(np.sum(1.0 - products))
        )
    return candidates[int(np.argmin(energies))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.resolve().parent.parent
    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    case = select_case(manifest, instance.n, instance.k, instance.snr)
    print(f"Selected case: n={case['n']} k={case['k']} snr={case['snr']}")
    print(f"Fixed candidates: {len(case['fixed_candidates'])}")

    checks = CheckTable(instance.h)
    count = instance.b if args.frames <= 0 else min(args.frames, instance.b)
    lines = []
    for index in range(count):
        hard = decode_frame(instance.llr[index], instance.h, checks, case, root)
        lines.append(" ".join(str(int(value)) for value in hard))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({count} frames)")


if __name__ == "__main__":
    main()
