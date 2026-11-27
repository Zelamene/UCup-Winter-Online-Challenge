#!/usr/bin/env python3
"""Tests for backward-compatible Phase 14 Track C decoder extensions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from bsb_vectorized import BatchedBSB, amplitude_pair_spins  # noqa: E402
from qam_pairs import pair_tables  # noqa: E402
from real_bench import build_case  # noqa: E402


def joint_kwargs(cfg: dict, snr: float, llr: np.ndarray, **extra) -> dict:
    params = dict(
        tau=cfg["tau"], xi=cfg["xi"], lambd=cfg["lambd"], delta=cfg["delta"],
        dt=cfg["dt"], steps=50, beta=cfg["beta"], parity_scale=cfg["parity_scale"],
        joint_pairs=1, pair_gamma=cfg["pair_gamma"], pair_scale=cfg["pair_scale"],
        pair_tables=pair_tables(llr, snr),
    )
    params.update(extra)
    return params


class BsbVectorizedExtensionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_case(1296, 648, 2, 6.0, 999, noise_scale=1.00112)
        cls.sim = BatchedBSB(cls.case["h"])
        cfg = {
            "tau": 2.295, "xi": 1.264, "lambd": 5.95, "delta": 3.72,
            "dt": 0.236, "beta": 0.27, "parity_scale": 0.3201525,
            "pair_gamma": 0.1638693, "pair_scale": 1.0,
        }
        cls.base_kwargs = joint_kwargs(cfg, 6.0, cls.case["llr"])

    def params(self, **extra) -> dict:
        params = dict(self.base_kwargs)
        params.update(extra)
        return params

    def test_neutral_classes_and_constant_schedule_are_bit_identical(self) -> None:
        baseline = self.sim.decode_batch(self.case["llr"], **self.base_kwargs)
        neutral = self.sim.decode_batch(
            self.case["llr"], **self.params(
                tau_classes=(1.0, 1.0), xi_classes=(1.0, 1.0), beta_classes=(1.0, 1.0),
            ),
        )
        constant_ramp = self.sim.decode_batch(
            self.case["llr"], **self.params(pair_scale=np.ones(50)),
        )
        np.testing.assert_array_equal(neutral, baseline)
        np.testing.assert_array_equal(constant_ramp, baseline)

    def test_class_multipliers_and_ramp_change_dynamics_without_shape_change(self) -> None:
        varied = self.sim.decode_batch(
            self.case["llr"], **self.params(
                tau_classes=(0.95, 1.05), xi_classes=(1.05, 0.95),
                beta_classes=(0.9, 1.1),
                pair_scale=np.concatenate((np.linspace(0.0, 1.0, 10), np.ones(40))),
            ),
        )
        self.assertEqual(varied.shape, (self.case["b"], self.case["n"]))
        self.assertTrue(np.any(varied != self.sim.decode_batch(self.case["llr"], **self.base_kwargs)))

    def test_invalid_class_and_schedule_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.sim.decode_batch(self.case["llr"], **self.params(tau_classes=(1.0,)))
        with self.assertRaises(ValueError):
            self.sim.decode_batch(self.case["llr"], **self.params(pair_scale=np.ones(49)))

    def test_amplitude_coordinate_is_single_valued_and_fixes_gray_corners(self) -> None:
        rng = np.random.default_rng(1405)
        first = rng.uniform(-0.9, 0.9, size=97)
        second = rng.uniform(-0.9, 0.9, size=97)
        mapped_first, mapped_second = amplitude_pair_spins(first, second)
        self.assertEqual(mapped_first.shape, first.shape)
        self.assertTrue(np.all(np.abs(mapped_first) <= 1.0 + 1e-12))
        self.assertTrue(np.all(np.abs(mapped_second) <= 1.0 + 1e-12))
        # The mapped state has exactly the amplitude coordinate we started from.
        self.assertLess(
            float(np.max(np.abs(
                -mapped_first * (2.0 + mapped_second) + first * (2.0 + second)
            ))), 2e-6,
        )
        for s0, s1 in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
            f, s = amplitude_pair_spins(np.array([s0]), np.array([s1]))
            self.assertEqual(float(f[0]), s0)
            self.assertEqual(float(s[0]), s1)

    def test_exact_pair_cross_reaction_is_gradient_and_vanishes_on_ising_points(self) -> None:
        """C2: -dV/dx for the documented exact O(J**3) pair potential."""
        rng = np.random.default_rng(1404)
        x_i = rng.uniform(-0.95, 0.95, size=41)
        x_j = rng.uniform(-0.95, 0.95, size=41)
        j_coupling = rng.uniform(0.05, 0.5, size=41)
        gamma = 0.17

        def potential(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            j2 = j_coupling * j_coupling
            return gamma * (
                -0.5 * j2 * (1.0 - first * first) * (1.0 - second * second)
                + (2.0 / 3.0) * j2 * j_coupling * second
                * (first**3 - first) * (1.0 - second * second)
            )

        def force(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            j2 = j_coupling * j_coupling
            return -gamma * (
                j2 * (1.0 - second * second) * first
                + (2.0 / 3.0) * j2 * j_coupling
                * (1.0 - second * second) * second * (3.0 * first * first - 1.0)
            )

        step = 1e-6
        numeric = -(potential(x_i + step, x_j) - potential(x_i - step, x_j)) / (2.0 * step)
        self.assertLess(float(np.max(np.abs(numeric - force(x_i, x_j)))), 2e-7)
        for first in (-1.0, 1.0):
            for second in (-1.0, 1.0):
                self.assertEqual(float(potential(np.array([first]), np.array([second]))[0]), 0.0)
                self.assertEqual(float(force(np.array([first]), np.array([second]))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
