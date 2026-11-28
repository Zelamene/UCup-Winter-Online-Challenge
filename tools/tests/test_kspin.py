#!/usr/bin/env python3
"""Tests for the K-spin QAIA decoder and its batched implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from bsb_vectorized import BatchedBSB, EPS  # noqa: E402
from decode_v3_python import decode_trajectory  # noqa: E402
from kspin_reference import (  # noqa: E402
    DEFAULT_PARAMS,
    _onsager_edge,
    decode as scalar_decode,
    energy as scalar_energy,
    info_topology,
    pair_channel,
    parity_weights,
)
from qam_pairs import pair_tables  # noqa: E402
from real_bench import build_case  # noqa: E402


PARAMS = dict(DEFAULT_PARAMS)


class KSpinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_case(
            648, 324, 2, 6.0, 4242, noise_scale=1.00112, llr_sigma="nominal"
        )
        cls.sim = BatchedBSB(cls.case["h"])
        cls.tables = pair_tables(cls.case["llr"], 6.0)

    def test_topology_rejects_non_systematic_h(self) -> None:
        matrix = np.eye(3, dtype=np.int8)
        with self.assertRaises(ValueError):
            info_topology(matrix, 3)

    def test_every_decoded_frame_is_a_codeword(self) -> None:
        for joint in (0, 1):
            params = dict(PARAMS, joint_pairs=joint)
            hard = np.stack(
                [
                    scalar_decode(
                        self.case["llr"][index],
                        self.case["h"],
                        324,
                        6.0,
                        params,
                    )
                    for index in range(self.case["b"])
                ]
            )
            syndrome = hard.astype(np.int16) @ self.case["h"].T.astype(np.int16) % 2
            self.assertEqual(int(np.count_nonzero(syndrome)), 0)

    def test_batched_matches_scalar_reference_per_bit_and_joint(self) -> None:
        for joint in (0, 1):
            params = dict(PARAMS, joint_pairs=joint)
            batched = self.sim.decode_batch(
                self.case["llr"],
                params["tau"],
                params["xi"],
                1.0,
                params["delta"],
                dt=params["dt"],
                steps=50,
                beta=params["beta"],
                joint_pairs=joint,
                pair_tables=self.tables if joint else None,
                mode="kspin",
                g=params["g"],
                clip=params["clip"],
            )
            scalar = np.stack(
                [
                    scalar_decode(
                        self.case["llr"][index],
                        self.case["h"],
                        324,
                        6.0,
                        params,
                    )
                    for index in range(self.case["b"])
                ]
            )
            np.testing.assert_array_equal(batched, scalar)

    def test_decode_trajectory_matches_scalar_kspin_reference(self) -> None:
        settings = ((0, 0.0), (1, 0.0), (1, 0.1))
        for joint, pair_gamma in settings:
            params = dict(PARAMS, joint_pairs=joint, pair_gamma=pair_gamma)
            actual, _soft = decode_trajectory(
                self.case["llr"][0],
                self.case["h"],
                params,
                mode="kspin",
                dt=params["dt"],
                steps=50,
                snr=6.0,
            )
            expected = scalar_decode(
                self.case["llr"][0], self.case["h"], 324, 6.0, params, steps=50
            )
            np.testing.assert_array_equal(actual, expected)

    def test_batched_matches_scalar_reference_with_pair_gamma(self) -> None:
        params = dict(PARAMS, joint_pairs=1, pair_gamma=0.1)
        batched = self.sim.decode_batch(
            self.case["llr"],
            params["tau"],
            params["xi"],
            1.0,
            params["delta"],
            dt=params["dt"],
            steps=50,
            beta=params["beta"],
            joint_pairs=1,
            pair_tables=self.tables,
            mode="kspin",
            g=params["g"],
            clip=params["clip"],
            pair_gamma=params["pair_gamma"],
        )
        scalar = np.stack(
            [
                scalar_decode(
                    self.case["llr"][index],
                    self.case["h"],
                    324,
                    6.0,
                    params,
                )
                for index in range(self.case["b"])
            ]
        )
        np.testing.assert_array_equal(batched, scalar)

    def test_energy_matches_eliminated_kspin_energy(self) -> None:
        k = 324
        info = np.array([1, 0] * (k // 2), dtype=np.int64)
        hard = np.empty((1, self.case["n"]), dtype=np.int64)
        hard[0, :k] = info
        _m, supports, _degree = info_topology(self.case["h"], k)
        for row, cols in enumerate(supports):
            parity = 0
            for col in cols:
                parity ^= int(info[col])
            hard[0, k + row] = parity

        llr = self.case["llr"][:1]
        full_spins = 1.0 - 2.0 * hard.astype(np.float64)
        a, b, d = self.tables
        pair_linear = np.empty((1, k), dtype=np.float64)
        pair_linear[0, 0::2] = 2.0 * a[0, : k // 2]
        pair_linear[0, 1::2] = 2.0 * b[0, : k // 2]
        pair_d = np.empty((1, k), dtype=np.float64)
        pair_d[0, 0::2] = 2.0 * d[0, : k // 2]
        pair_d[0, 1::2] = 2.0 * d[0, : k // 2]
        channel = pair_linear + pair_d * full_spins[:, np.arange(k) ^ 1]
        raw_weights = llr[:, k:]
        direct = -np.sum(channel * full_spins[:, :k], axis=1)
        for row, cols in enumerate(supports):
            direct -= raw_weights[0, row] * np.prod(full_spins[0, cols])

        for select_unclipped in (0, 1):
            weights = (
                raw_weights[0]
                if select_unclipped
                else parity_weights(llr[0], k, PARAMS["g"], PARAMS["clip"])
            )
            manual = direct + np.sum(
                (raw_weights[0] - weights)
                * np.array([np.prod(full_spins[0, cols]) for cols in supports])
            )
            actual = self.sim.compute_kspin_energies(
                hard,
                llr,
                6.0,
                joint_pairs=1,
                pair_tables=self.tables,
                g=PARAMS["g"],
                clip=PARAMS["clip"],
                select_unclipped=select_unclipped,
            )
            np.testing.assert_allclose(actual, manual, rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(
                scalar_energy(
                    info,
                    llr[0],
                    self.case["h"],
                    k,
                    6.0,
                    dict(PARAMS, joint_pairs=1),
                    select_unclipped=select_unclipped,
                ),
                manual[0],
                rtol=0.0,
                atol=1e-9,
            )

    def test_mean_field_gradient_matches_numerical_derivative(self) -> None:
        k = 324
        rng = np.random.default_rng(106)
        x = rng.uniform(-0.8, 0.8, size=k)
        soft = np.where(np.abs(x) < EPS, np.where(x >= 0, EPS, -EPS), x)
        rates = self.case["llr"][0]
        weights = parity_weights(rates, k, PARAMS["g"], PARAMS["clip"])
        _m, supports, degree = info_topology(self.case["h"], k)

        def mean_field_energy(values: np.ndarray) -> float:
            value = -float(np.sum(values * rates[:k]))
            for row, cols in enumerate(supports):
                value -= weights[row] * float(np.prod(values[cols]))
            return value

        exact = rates[:k].copy()
        for row, cols in enumerate(supports):
            values = soft[cols]
            prefix = np.ones(degree + 1)
            suffix = np.ones(degree + 1)
            for index in range(degree):
                prefix[index + 1] = prefix[index] * values[index]
            for index in range(degree - 1, -1, -1):
                suffix[index] = suffix[index + 1] * values[index]
            exact[cols] += weights[row] * prefix[:degree] * suffix[1:]

        epsilon = 1e-6
        numerical = np.empty(k)
        for index in range(k):
            plus = soft.copy()
            minus = soft.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = -(
                mean_field_energy(plus) - mean_field_energy(minus)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(exact, numerical, rtol=1e-5, atol=1e-5)

    def test_onsager_recurrence_matches_cavity_sum(self) -> None:
        rng = np.random.default_rng(220)
        for degree in (2, 3, 6):
            spins = rng.uniform(-0.95, 0.95, size=degree)
            actual = _onsager_edge(spins)
            expected = np.empty(degree)
            for index in range(degree):
                total = 0.0
                for other in range(degree):
                    if other == index:
                        continue
                    product = 1.0 - spins[other] ** 2
                    for position in range(degree):
                        if position not in (index, other):
                            product *= spins[position] ** 2
                    total += product
                expected[index] = total
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_select_unclipped_requires_binary_flag(self) -> None:
        with self.assertRaises(ValueError):
            self.sim.compute_kspin_energies(
                np.zeros((1, self.case["n"]), dtype=np.int8),
                self.case["llr"][:1],
                6.0,
                select_unclipped=2,
            )


if __name__ == "__main__":
    unittest.main()
