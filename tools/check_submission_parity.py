#!/usr/bin/env python3
"""Check that generated C++ constants and decoder logic match Python Phase 6."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import SAMPLE_INPUT, parse_instance  # noqa: E402
from phase3_fixed_dsb import CheckTable  # noqa: E402
from phase6_multi_guess import (  # noqa: E402
    FIXED_CANDIDATES,
    LEARNED_ENERGY_LAMBDA,
    decode_candidates,
    load_params,
    select_by_energy,
)

ROOT = TOOLS_DIR.parent
DEFAULT_SUBMISSION = ROOT / "submission.cpp"
DEFAULT_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"


class CppConstants:
    def __init__(self, source: str) -> None:
        self.tau = float(self.scalar("TAU", source))
        self.learned_lambda = float(self.scalar("LEARNED_ENERGY_LAMBDA", source))
        self.dt = self.vector("DT", source)
        self.xi = self.vector("XI", source)
        self.lam = self.vector("LAM", source)
        self.delta = self.vector("DELTA", source)
        self.xi_mod = self.vector("XI_MOD", source)
        self.lam_mod = self.vector("LAM_MOD", source)
        self.delta_mod = self.vector("DELTA_MOD", source)
        fixed = re.search(
            r"static const FixedConfig FIXED_CANDIDATES\[5\] = \{(.*?)\};", source, re.S
        )
        if fixed is None:
            raise ValueError("Fixed candidates missing from C++ source")
        numbers = [
            float(value)
            for value in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", fixed.group(1), re.I)
        ]
        self.fixed = np.asarray(numbers, dtype=np.float64).reshape(5, 4)

    @staticmethod
    def scalar(name: str, source: str) -> str:
        match = re.search(rf"static constexpr double {name} = ([^;]+);", source)
        if match is None:
            raise ValueError(f"Scalar {name} missing from C++ source")
        return match.group(1)

    @staticmethod
    def vector(name: str, source: str) -> np.ndarray:
        match = re.search(rf"static const double {name}\[STEPS\] = \{{(.*?)\}};", source, re.S)
        if match is None:
            raise ValueError(f"Vector {name} missing from C++ source")
        return np.asarray([float(value) for value in match.group(1).split(",")], dtype=np.float64)


def decode_learned_cpp(c: CppConstants, r: np.ndarray, checks: list[np.ndarray]) -> np.ndarray:
    n = len(r)
    var = np.tanh(np.abs(r) / 8.0)
    check = np.asarray([np.tanh(min(abs(r[j]) for j in row) / 8.0) for row in checks])
    x = np.tanh(r / c.tau).copy()
    y = np.zeros(n)
    for t in range(50):
        spins = np.where(x >= 0, 1.0, -1.0)
        weighted = np.zeros(n)
        for row_index, row in enumerate(checks):
            product = 1.0
            for j in row:
                product *= spins[j]
            gain = c.lam[t] * np.exp(c.lam_mod[t] * check[row_index])
            for j in row:
                weighted[j] += product * gain
        schedule = t / 49.0
        for j in range(n):
            force = r[j] + 0.5 * spins[j] * weighted[j]
            delta_gain = c.delta[t] * np.exp(c.delta_mod[t] * var[j])
            xi_gain = c.xi[t] * np.exp(c.xi_mod[t] * var[j])
            ydot = -(delta_gain - schedule) * x[j] + xi_gain * force
            y_new = y[j] + c.dt[t] * ydot
            x_new = x[j] + c.dt[t] * y_new
            if abs(x_new) > 1.0:
                x[j] = 1.0 if x_new >= 0 else -1.0
                y[j] = 0.0
            else:
                x[j] = x_new
                y[j] = y_new
    return (x < 0).astype(np.int8)


def decode_fixed_cpp(config: np.ndarray, r: np.ndarray, checks: list[np.ndarray]) -> np.ndarray:
    tau, xi, lambd, delta = (float(value) for value in config)
    n = len(r)
    x = np.tanh(r / tau).copy()
    y = np.zeros(n)
    for t in range(50):
        spins = np.where(x >= 0, 1.0, -1.0)
        weighted = np.zeros(n)
        for row in checks:
            product = 1.0
            for j in row:
                product *= spins[j]
            for j in row:
                weighted[j] += product
        schedule = t / 49.0
        for j in range(n):
            force = r[j] + 0.5 * lambd * spins[j] * weighted[j]
            pump = -(delta - schedule) * x[j]
            y_new = y[j] + 0.2 * (pump + xi * force)
            x_new = x[j] + 0.2 * y_new
            if abs(x_new) > 1.0:
                x[j] = 1.0 if x_new >= 0 else -1.0
                y[j] = 0.0
            else:
                x[j] = x_new
                y[j] = y_new
    return (x < 0).astype(np.int8)


def energy_cpp(hard: np.ndarray, r: np.ndarray, checks: list[np.ndarray], lambd: float) -> float:
    spins = np.where(hard == 0, 1.0, -1.0)
    channel = -np.sum(r * spins)
    penalty = 0.0
    for row in checks:
        product = 1.0
        for j in row:
            product *= spins[j]
        penalty += 1.0 - product
    return float(channel + 0.5 * lambd * penalty)


def candidates_and_selection(
    c: CppConstants, r: np.ndarray, checks: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, int]:
    candidates = [decode_learned_cpp(c, r, checks)]
    candidates.extend(decode_fixed_cpp(config, r, checks) for config in c.fixed)
    lambdas = [c.learned_lambda] + c.fixed[:, 2].tolist()
    energies = [energy_cpp(h, r, checks, lambd) for h, lambd in zip(candidates, lambdas)]
    pick = int(np.argmin(energies))
    return candidates, np.asarray(candidates[pick]), pick


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    constants = CppConstants(args.submission.read_text(encoding="utf-8"))
    constants_ok = (
        np.allclose([constants.tau], [8.0803580992176549])
        and np.allclose(constants.fixed, np.asarray(FIXED_CANDIDATES))
        and constants.learned_lambda == LEARNED_ENERGY_LAMBDA
        and len(constants.dt) == 50
    )
    print(f"C++ constants match Phase 6 config: {constants_ok}")

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    checks = [np.flatnonzero(instance.h[row]) for row in range(instance.m)]
    params = load_params(args.params)
    count = min(args.frames, instance.b)
    llr = instance.llr[:count]

    py_bits, py_lambdas = decode_candidates(params, instance.h, llr)
    py_selected, py_pick = select_by_energy(
        py_bits, py_lambdas, llr, CheckTable(instance.h)
    )

    candidate_mismatches = np.zeros(6, dtype=np.int64)
    selected_mismatches = 0
    pick_mismatches = 0
    for i in range(count):
        cpp_candidates, cpp_selected, cpp_pick = candidates_and_selection(
            constants, llr[i], checks
        )
        for j in range(6):
            candidate_mismatches[j] += int(np.count_nonzero(cpp_candidates[j] != py_bits[j][i]))
        selected_mismatches += int(np.count_nonzero(cpp_selected != py_selected[i]))
        pick_mismatches += int(cpp_pick != py_pick[i])

    print(f"Public frames checked: {count}")
    print(f"Candidate bit mismatches by trajectory: {candidate_mismatches.tolist()}")
    print(f"Selected bit mismatches: {selected_mismatches}")
    print(f"Selection index mismatches: {pick_mismatches}")

    sample = parse_instance(SAMPLE_INPUT, "PDF sample")
    sample_checks = [np.flatnonzero(sample.h[row]) for row in range(sample.m)]
    sample_py_bits, sample_py_lambdas = decode_candidates(params, sample.h, sample.llr)
    sample_py_selected, _ = select_by_energy(
        sample_py_bits, sample_py_lambdas, sample.llr, CheckTable(sample.h)
    )
    _, sample_cpp_selected, _ = candidates_and_selection(
        constants, sample.llr[0], sample_checks
    )
    sample_ok = bool(np.array_equal(sample_cpp_selected, sample_py_selected[0]))
    print(f"Sample selection matches Python: {sample_ok}")

    gate = constants_ok and not candidate_mismatches.any() and not selected_mismatches
    if not gate:
        raise SystemExit("C++ parity check failed")


if __name__ == "__main__":
    main()
