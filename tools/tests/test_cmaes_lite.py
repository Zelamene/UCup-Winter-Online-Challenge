#!/usr/bin/env python3
"""Tests for the dependency-free CMA-ES used by Phase 15."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from cmaes_lite import CMAESLite  # noqa: E402


class CmaesLiteTests(unittest.TestCase):
    def test_rotated_ellipsoid_reaches_tolerance_within_3000_evaluations(self) -> None:
        rng = np.random.default_rng(1501)
        dim = 8
        rotation, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        eigenvalues = np.exp(np.linspace(0.0, math.log(128.0), dim))
        hessian = rotation.T @ np.diag(eigenvalues) @ rotation
        target = np.ones(dim)
        start = np.full(dim, 0.6)
        best = math.inf

        optimizer = CMAESLite(
            dim,
            mean=start,
            sigma=0.15,
            population_size=12,
            mu=6,
            seed=1501,
            bounds=(0.5, 2.0),
        )
        while optimizer.evaluations < 3000:
            candidates = optimizer.ask()
            fitness = [
                float((candidate - target) @ hessian @ (candidate - target))
                for candidate in candidates
            ]
            optimizer.tell(candidates, fitness)
            best = min(best, min(fitness))
            if best < 1e-6:
                break
        self.assertLess(best, 1e-6, f"best={best:g}, evals={optimizer.evaluations}")

    def test_box_bounds_and_checkpoint_round_trip(self) -> None:
        optimizer = CMAESLite(
            4, mean=[0.1, 1.5, 3.0, -2.0], sigma=0.2,
            population_size=10, mu=5, seed=77, bounds=(0.5, 2.0),
        )
        candidates = optimizer.ask()
        self.assertTrue(all(np.all(c >= 0.5) and np.all(c <= 2.0) for c in candidates))
        optimizer.tell(candidates, [float(np.sum((c - 1.0) ** 2)) for c in candidates])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cma.json"
            optimizer.save(path)
            restored = CMAESLite.load(path)
            self.assertEqual(restored.state(), optimizer.state())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["generation"], 1)

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CMAESLite(0)
        with self.assertRaises(ValueError):
            CMAESLite(2, sigma=0.0)
        with self.assertRaises(ValueError):
            CMAESLite(2, bounds=(2.0, 0.5))


if __name__ == "__main__":
    unittest.main()
