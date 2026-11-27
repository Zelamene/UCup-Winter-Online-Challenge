#!/usr/bin/env python3
"""Tests for exact Gray-pair likelihood recovery (Phase 14)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import qam_pairs  # noqa: E402
from phase0_harness import parse_instance  # noqa: E402
from real_qam_channel import generate_llrs  # noqa: E402


class QamPairsTests(unittest.TestCase):
    def test_public_file_partner_prediction(self) -> None:
        path = ROOT / "Problem statements" / "publish.in"
        instance = parse_instance(path.read_text(encoding="utf-8"), str(path))
        stats = qam_pairs.consistency(instance.llr, instance.snr)
        self.assertLess(stats["p99_abs_error"], 1e-8)
        self.assertEqual(stats["bad_fraction"], 0.0)

    def test_first_bit_llr_is_monotone(self) -> None:
        grid = np.linspace(-qam_pairs.BRACKET, qam_pairs.BRACKET, 2001)
        for snr in (6.0, 9.0, 10.3, 10.5):
            llr = qam_pairs.first_bit_llr_y(grid, qam_pairs.nominal_sigma(snr))
            self.assertTrue(np.all(np.diff(llr) < 0.0), f"non-monotone at SNR {snr}")

    def test_amplitude_round_trip(self) -> None:
        rng = np.random.default_rng(1401)
        y_true = rng.uniform(-1.5, 1.5, size=5000)
        for snr in (6.0, 9.0, 10.3, 10.5):
            sigma = qam_pairs.nominal_sigma(snr)
            llr = qam_pairs.first_bit_llr_y(y_true, sigma)
            recovered = qam_pairs.recover_amplitude(llr, sigma)
            self.assertLess(float(np.max(np.abs(recovered - y_true))), 1e-9)

    def test_saturated_inputs_are_finite(self) -> None:
        for snr in (6.0, 10.3):
            sigma = qam_pairs.nominal_sigma(snr)
            recovered = qam_pairs.recover_amplitude(np.array([-1e6, 1e6]), sigma)
            self.assertTrue(np.all(np.isfinite(recovered)))
            self.assertTrue(np.all(np.abs(recovered) <= qam_pairs.BRACKET + 1e-12))

    def test_generated_llrs_are_consistent(self) -> None:
        rng = np.random.default_rng(1402)
        bits = rng.integers(0, 2, size=(32, 40), dtype=np.int8)
        llr = generate_llrs(bits, 6.0, rng)
        stats = qam_pairs.consistency(llr, 6.0)
        self.assertLess(stats["max_abs_error"], 1e-8)
        self.assertEqual(stats["bad_fraction"], 0.0)

    def test_pair_reaction_polynomial_is_gradient_and_vanishes_on_ising_points(self) -> None:
        """Check the (2d)^2 (1-x_j^2) x_i pair reaction term numerically.

        The reaction is the negative gradient of
        ``0.5 * (2d)^2 * (1 - x_j^2) * x_i^2``, which is zero at ``x_i, x_j = +-1``.
        It therefore reshapes only the continuous bSB trajectory, never the Ising
        objective that the ±1 hard spins represent.
        """
        rng = np.random.default_rng(1403)
        d = rng.uniform(0.02, 0.4, size=37)
        xi = rng.uniform(-0.95, 0.95, size=37)
        xj = rng.uniform(-0.95, 0.95, size=37)
        coupling_sq = (2.0 * d) ** 2

        def energy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            return 0.5 * coupling_sq * (1.0 - second * second) * first * first

        exact_force = -coupling_sq * (1.0 - xj * xj) * xi
        step = 1e-6
        numeric_force = -(energy(xi + step, xj) - energy(xi - step, xj)) / (2.0 * step)
        self.assertLess(float(np.max(np.abs(numeric_force - exact_force))), 2e-8)

        corners = np.array([-1.0, 1.0])
        for first in corners:
            for second in corners:
                self.assertEqual(float(energy(np.array([first]), np.array([second]))[0]), 0.0)
                self.assertEqual(
                    float((-coupling_sq * (1.0 - second * second) * first)[0]), 0.0
                )


if __name__ == "__main__":
    unittest.main()
