#!/usr/bin/env python3
"""Small dependency-free CMA-ES for low-dimensional offline tuning.

This is deliberately local: it has no ``cma`` dependency, uses a fixed random
seed, supports box constraints by clipping samples, and can checkpoint all
adaptive state to JSON so long searches can be resumed.

The objective is minimised. ``ask`` returns candidates and ``tell`` consumes
the same list plus one fitness per candidate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


class CMAESLite:
    """(mu/mu, lambda)-CMA-ES with rank-one and rank-mu covariance updates."""

    def __init__(
        self,
        dimension: int,
        mean: Sequence[float] | np.ndarray | None = None,
        sigma: float = 0.15,
        population_size: int = 12,
        mu: int | None = None,
        seed: int = 5150,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        dim = int(dimension)
        if dim <= 0:
            raise ValueError("dimension must be positive")
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if mu is None:
            mu = max(2, population_size // 2)
        if not 2 <= mu <= population_size:
            raise ValueError("mu must be in 2..population_size")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")

        self.dim = dim
        self.lam = int(population_size)
        self.mu = int(mu)
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError("bounds must satisfy low < high")

        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1, dtype=float))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = float(1.0 / np.sum(self.weights**2))

        self.cc = (4.0 + self.mu_eff / dim) / (
            dim + 4.0 + 2.0 * self.mu_eff / dim
        )
        self.cs = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((dim + 2.0) ** 2 + self.mu_eff),
        )
        self.cmu = max(0.0, self.cmu)
        self.ds = 1.0 + 2.0 * max(
            0.0, math.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0
        ) + self.cs
        self.chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        initial = np.ones(dim, dtype=float) if mean is None else np.asarray(mean, dtype=float)
        if initial.shape != (dim,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"mean must have shape ({dim},) and be finite")
        self.mean = self._clip(initial.copy())
        self.sigma = float(sigma)
        self.cov = np.eye(dim, dtype=float)
        self.ps = np.zeros(dim, dtype=float)
        self.pc = np.zeros(dim, dtype=float)
        self.generation = 0
        self.evaluations = 0
        self.best_fitness = math.inf
        self.best_solution: np.ndarray | None = None
        self._rng = np.random.default_rng(int(seed))
        self._pending: list[np.ndarray] | None = None

    def _clip(self, values: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return values
        return np.clip(values, self.bounds[0], self.bounds[1])

    def ask(self) -> list[np.ndarray]:
        """Return ``lambda`` candidate vectors sampled from the current model."""
        if self._pending is not None:
            raise RuntimeError("ask() called twice before tell()")
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-12)
        transform = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
        candidates: list[np.ndarray] = []
        for _ in range(self.lam):
            z = self._rng.standard_normal(self.dim)
            candidates.append(self._clip(self.mean + self.sigma * (transform @ z)))
        self._pending = candidates
        return [candidate.copy() for candidate in candidates]

    def tell(self, solutions: Sequence[np.ndarray], fitness: Sequence[float]) -> None:
        """Update the model from one generation of evaluated candidates."""
        if self._pending is None:
            raise RuntimeError("tell() called before ask()")
        if len(solutions) != self.lam or len(fitness) != self.lam:
            raise ValueError(f"tell() expects {self.lam} solutions and fitness values")
        values = np.asarray(fitness, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("all fitness values must be finite")

        for solution, value in zip(solutions, values):
            if value < self.best_fitness:
                self.best_fitness = float(value)
                self.best_solution = self._clip(np.asarray(solution, dtype=float).copy())

        order = np.argsort(values, kind="stable")[: self.mu]
        selected = np.asarray([self._pending[int(index)] for index in order], dtype=float)
        old_mean = self.mean.copy()
        self.mean = self._clip(np.sum(self.weights[:, None] * selected, axis=0))
        self.evaluations += self.lam

        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-12)
        inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
        y_w = (self.mean - old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(
            self.cs * (2.0 - self.cs) * self.mu_eff
        ) * (inv_sqrt @ y_w)

        generation = self.generation + 1
        hsig = (
            np.linalg.norm(self.ps)
            / math.sqrt(1.0 - (1.0 - self.cs) ** (2 * generation))
            / self.chi_n
            < 1.4 + 2.0 / (self.dim + 1.0)
        )
        self.pc = (1.0 - self.cc) * self.pc
        if hsig:
            self.pc += math.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * y_w

        ys = (selected - old_mean) / self.sigma
        rank_mu = (ys * self.weights[:, None]).T @ ys
        rank_one = np.outer(self.pc, self.pc)
        if not hsig:
            rank_one += self.cc * (2.0 - self.cc) * self.cov
        self.cov = (
            (1.0 - self.c1 - self.cmu) * self.cov
            + self.c1 * rank_one
            + self.cmu * rank_mu
        )
        self.cov = 0.5 * (self.cov + self.cov.T)

        self.sigma *= math.exp(
            (self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1.0)
        )
        self.sigma = float(np.clip(self.sigma, 1e-6, 10.0))
        self.generation = generation
        self._pending = None

    def state(self) -> dict:
        """Return a JSON-serialisable snapshot of all adaptive state."""
        return {
            "version": 1,
            "dimension": self.dim,
            "population_size": self.lam,
            "mu": self.mu,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "cov": self.cov.tolist(),
            "ps": self.ps.tolist(),
            "pc": self.pc.tolist(),
            "generation": self.generation,
            "evaluations": self.evaluations,
            "best_fitness": self.best_fitness,
            "best_solution": (
                None if self.best_solution is None else self.best_solution.tolist()
            ),
        }

    def save(self, path: Path | str) -> None:
        """Write a checkpoint; atomic replacement avoids truncated resumes."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(self.state(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(target)

    @classmethod
    def load(cls, path: Path | str) -> "CMAESLite":
        """Restore a checkpoint written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported checkpoint version: {payload.get('version')!r}")
        optimizer = cls(
            dimension=int(payload["dimension"]),
            mean=payload["mean"],
            sigma=float(payload["sigma"]),
            population_size=int(payload["population_size"]),
            mu=int(payload["mu"]),
            seed=0,
            bounds=payload.get("bounds"),
        )
        optimizer.cov = np.asarray(payload["cov"], dtype=float)
        optimizer.ps = np.asarray(payload["ps"], dtype=float)
        optimizer.pc = np.asarray(payload["pc"], dtype=float)
        optimizer.generation = int(payload["generation"])
        optimizer.evaluations = int(payload["evaluations"])
        optimizer.best_fitness = float(payload["best_fitness"])
        best = payload.get("best_solution")
        optimizer.best_solution = None if best is None else np.asarray(best, dtype=float)
        return optimizer
