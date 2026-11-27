#!/usr/bin/env python3
"""Vectorized batched bSB (ballistic Simulated Bifurcation) simulation and tuning harness."""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
import scipy.sparse as sp

from qam_pairs import pair_tables as build_pair_tables  # noqa: E402

EPS = 1e-12
SCORE_SCALE = 100_000.0


def pair_channel_arrays(
    llr: np.ndarray,
    tables: tuple[np.ndarray, np.ndarray, np.ndarray],
    pair_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the batched linear pair field and per-site pair coupling."""
    a, b, d = tables
    channel = np.empty_like(llr, dtype=np.float64)
    channel[:, 0::2] = 2.0 * a
    channel[:, 1::2] = 2.0 * b
    pair_d = np.empty_like(llr, dtype=np.float64)
    pair_d[:, 0::2] = d
    pair_d[:, 1::2] = d
    return channel, pair_d


class BatchedBSB:
    """Pre-computes check topology for fast batched bSB simulation across B frames."""

    def __init__(self, h: np.ndarray) -> None:
        self.h = np.asarray(h, dtype=np.int8)
        self.m, self.n = self.h.shape
        # Extract row non-zero columns.
        rows = [np.flatnonzero(self.h[i]) for i in range(self.m)]
        self.deg = max((len(r) for r in rows), default=0)
        self.cols = np.zeros((self.m, self.deg), dtype=np.int64)
        for i, r in enumerate(rows):
            self.cols[i, : len(r)] = r

        # Build CSR scatter matrix: (m * deg, n)
        row_idx = np.arange(self.m * self.deg)
        col_idx = self.cols.ravel()
        data = np.ones(self.m * self.deg, dtype=np.float64)
        self.scatter_matrix = sp.csr_matrix((data, (row_idx, col_idx)), shape=(self.m * self.deg, self.n))

    def decode_batch(
        self,
        llr: np.ndarray,
        tau: float,
        xi: Union[float, Sequence[float], np.ndarray],
        lambd: Union[float, Sequence[float], np.ndarray],
        delta: Union[float, Sequence[float], np.ndarray],
        dt: Union[float, Sequence[float], np.ndarray] = 0.2,
        steps: int = 50,
        beta: Union[float, Sequence[float], np.ndarray] = 0.0,
        parity_scale: float = 1.0,
        dsb_finish: int = 0,
        joint_pairs: int = 0,
        pair_gamma: float = 0.0,
        pair_scale: float = 1.0,
        pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Run batched bSB on B frames of LLRs and return hard bits of shape (B, N).

        ``dsb_finish`` switches the last N steps to hard dSB check products; it
        is still QAIA-only and is used solely by the structural screen.
        ``joint_pairs`` replaces the per-bit channel field with the exact joint
        Gray-pair field; ``pair_tables`` must then be supplied from qam_pairs.
        """
        llr = np.asarray(llr, dtype=np.float64)
        b = llr.shape[0]

        # Convert schedules to float or 1D arrays
        xi_is_seq = isinstance(xi, (list, tuple, np.ndarray))
        lambd_is_seq = isinstance(lambd, (list, tuple, np.ndarray))
        delta_is_seq = isinstance(delta, (list, tuple, np.ndarray))
        dt_is_seq = isinstance(dt, (list, tuple, np.ndarray))
        beta_is_seq = isinstance(beta, (list, tuple, np.ndarray))

        if xi_is_seq:
            xi = np.asarray(xi, dtype=np.float64)
        if lambd_is_seq:
            lambd = np.asarray(lambd, dtype=np.float64)
        if delta_is_seq:
            delta = np.asarray(delta, dtype=np.float64)
        if dt_is_seq:
            dt = np.asarray(dt, dtype=np.float64)
        if beta_is_seq:
            beta = np.asarray(beta, dtype=np.float64)
            if beta.ndim != 1 or len(beta) != steps:
                raise ValueError(f"beta schedule has {beta.size} values, expected {steps}")

        if dsb_finish < 0 or dsb_finish > steps:
            raise ValueError(f"dsb_finish must be in 0..{steps}, got {dsb_finish}")
        dsb_start = steps - int(dsb_finish) if dsb_finish else steps + 1

        joint = int(joint_pairs)
        if joint not in (0, 1):
            raise ValueError(f"joint_pairs must be 0 or 1, got {joint_pairs}")
        if joint:
            if llr.shape[1] % 4 != 0:
                raise ValueError(f"joint_pairs requires n divisible by 4, got {llr.shape[1]}")
            if pair_tables is None:
                raise ValueError("joint_pairs requires pair_tables from qam_pairs.pair_tables")
            a, bp, d = pair_tables
            if a.shape != (b, llr.shape[1] // 2) or bp.shape != a.shape or d.shape != a.shape:
                raise ValueError(f"pair tables have shape {a.shape}, expected {(b, llr.shape[1] // 2)}")
            channel_linear = np.empty_like(llr)
            channel_linear[:, 0::2] = 2.0 * a
            channel_linear[:, 1::2] = 2.0 * bp
            pair_d = np.empty_like(llr)
            pair_d[:, 0::2] = d
            pair_d[:, 1::2] = d
            partner_index = np.arange(llr.shape[1], dtype=np.intp) ^ 1
            pair_coupling = (2.0 * float(pair_scale)) * pair_d
            pair_reaction = (float(pair_gamma) * 4.0) * (pair_d * pair_d)
        else:
            channel_linear = llr
            pair_d = None

        x = np.tanh(llr / float(tau)).astype(np.float64, copy=True)
        y = np.zeros_like(x)

        pref = np.ones((b, self.m, self.deg + 1), dtype=np.float64)
        suff = np.ones((b, self.m, self.deg + 1), dtype=np.float64)
        check_scale = np.ones(self.n, dtype=np.float64)
        check_scale[self.n - self.m:] = parity_scale  # H=[A|I]: last m columns are parity bits
        if beta_is_seq or beta != 0.0:
            pp = np.ones((b, self.m, self.deg + 1)); qp = np.zeros((b, self.m, self.deg + 1))
            ps = np.ones((b, self.m, self.deg + 1)); qs = np.zeros((b, self.m, self.deg + 1))

        for step in range(steps):
            # Soft spin clipping [-1, 1] with near-zero guard
            soft = np.clip(x, -1.0, 1.0)
            spins = np.where(np.abs(soft) < EPS, np.where(soft >= 0.0, EPS, -EPS), soft)
            if step >= dsb_start:
                spins = np.where(x >= 0.0, 1.0, -1.0)

            # Gather along check edges
            gathered = spins[:, self.cols]  # (B, M, deg)
            pref[:, :, 1:] = np.cumprod(gathered, axis=2)
            suff[:, :, :-1] = np.cumprod(gathered[:, :, ::-1], axis=2)[:, :, ::-1]

            # Excluded products: product of all check bits except the target bit
            prod_excl = pref[:, :, : self.deg] * suff[:, :, 1:]
            weighted_sum = prod_excl.reshape(b, -1) @ self.scatter_matrix

            l_val = float(lambd[step]) if lambd_is_seq else float(lambd)
            xi_val = float(xi[step]) if xi_is_seq else float(xi)
            d_val = float(delta[step]) if delta_is_seq else float(delta)
            dt_val = float(dt[step]) if dt_is_seq else float(dt)

            coupling = 0.5 * l_val
            if parity_scale != 1.0:
                force = channel_linear + (coupling * check_scale) * weighted_sum
            else:
                force = channel_linear + coupling * weighted_sum
            if pair_d is not None:
                partner = spins[:, partner_index]
                force += pair_coupling * partner
                if float(pair_gamma) != 0.0:
                    force -= pair_reaction * (1.0 - partner * partner) * spins
            beta_val = float(beta[step]) if beta_is_seq else float(beta)
            if beta_val != 0.0:
                # Onsager/TAP reaction: gradient of a polynomial that is constant on +-1 spins
                q = gathered * gathered
                u = 1.0 - q
                for t in range(self.deg):
                    pp[:, :, t + 1] = pp[:, :, t] * q[:, :, t]
                    qp[:, :, t + 1] = qp[:, :, t] * q[:, :, t] + pp[:, :, t] * u[:, :, t]
                for t in range(self.deg - 1, -1, -1):
                    ps[:, :, t] = ps[:, :, t + 1] * q[:, :, t]
                    qs[:, :, t] = qs[:, :, t + 1] * q[:, :, t] + ps[:, :, t + 1] * u[:, :, t]
                per_edge = qp[:, :, : self.deg] * ps[:, :, 1:] + pp[:, :, : self.deg] * qs[:, :, 1:]
                reaction = (beta_val * (coupling * coupling)) * spins * (per_edge.reshape(b, -1) @ self.scatter_matrix)
                if parity_scale != 1.0:
                    reaction = reaction * check_scale
                force = force - reaction
            schedule = step / (steps - 1) if steps > 1 else 0.0
            pump = -(d_val - schedule) * x
            y = y + dt_val * (pump + xi_val * force)
            x_new = x + dt_val * y

            clipped = np.abs(x_new) > 1.0
            if np.any(clipped):
                x_new[clipped] = np.where(x_new[clipped] >= 0.0, 1.0, -1.0)
                y[clipped] = 0.0
            x = x_new

        return (x < 0.0).astype(np.int8)

    def evaluate(
        self,
        llr: np.ndarray,
        answers: np.ndarray,
        k: int,
        tau: float,
        xi: Union[float, Sequence[float], np.ndarray],
        lambd: Union[float, Sequence[float], np.ndarray],
        delta: Union[float, Sequence[float], np.ndarray],
        dt: Union[float, Sequence[float], np.ndarray] = 0.2,
        steps: int = 50,
        beta: Union[float, Sequence[float], np.ndarray] = 0.0,
        parity_scale: float = 1.0,
        joint_pairs: int = 0,
        pair_gamma: float = 0.0,
        pair_scale: float = 1.0,
        pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[float, float, int]:
        """Compute score, BER, and bit error count on systematic information bits."""
        hard = self.decode_batch(
            llr, tau, xi, lambd, delta, dt=dt, steps=steps, beta=beta,
            parity_scale=parity_scale, joint_pairs=joint_pairs,
            pair_gamma=pair_gamma, pair_scale=pair_scale, pair_tables=pair_tables,
        )
        b = llr.shape[0]
        total_info_bits = b * k
        errors = int(np.count_nonzero(hard[:, :k] != answers[:, :k]))
        ber = max(errors / total_info_bits, 1.0 / total_info_bits)
        score = -SCORE_SCALE * math.log10(ber)
        return score, ber, errors

    def compute_energy(
        self,
        hard: np.ndarray,
        channel_field: np.ndarray,
        energy_lambda: float,
        pair_d: np.ndarray | None = None,
        pair_scale: float = 1.0,
    ) -> np.ndarray:
        """Compute Ising energy for each frame in hard (B, N). Returns array of shape (B,)."""
        spins = np.where(hard == 0, 1.0, -1.0)
        channel = -np.sum(channel_field * spins, axis=1)
        if pair_d is not None:
            channel -= 2.0 * float(pair_scale) * np.sum(
                pair_d[:, 0::2] * spins[:, 0::2] * spins[:, 1::2], axis=1
            )
        gathered = spins[:, self.cols]
        products = np.prod(gathered, axis=2)
        penalty = np.sum(1.0 - products, axis=1)
        return channel + 0.5 * float(energy_lambda) * penalty

    def evaluate_ensemble(
        self,
        llr: np.ndarray,
        answers: np.ndarray,
        k: int,
        trajectories: list[dict],
        combine: str = "energy",
        dt: float = 0.2,
        steps: int = 50,
        snr: float | None = None,
    ) -> tuple[float, float, int]:
        """Evaluate an ensemble of trajectories with energy or vote combine."""
        b = llr.shape[0]
        if not trajectories:
            raise ValueError("No trajectories provided")
        tables = None
        if any(int(traj.get("joint_pairs", 0)) for traj in trajectories):
            if snr is None:
                raise ValueError("joint_pairs requires snr")
            tables = build_pair_tables(llr, snr)

        candidates = []
        energies = []
        for traj in trajectories:
            tau = traj["tau"]
            xi = traj["xi"]
            lambd = traj["lambd"]
            delta = traj["delta"]
            traj_dt = traj.get("dt", dt)
            hard = self.decode_batch(
                llr, tau, xi, lambd, delta, dt=traj_dt, steps=steps,
                beta=traj.get("beta", 0.0),
                parity_scale=traj.get("parity_scale", 1.0),
                joint_pairs=traj.get("joint_pairs", 0),
                pair_gamma=traj.get("pair_gamma", 0.0),
                pair_scale=traj.get("pair_scale", 1.0),
                pair_tables=tables,
            )
            candidates.append(hard)
            if combine == "energy":
                e_lam = traj.get("energy_lambda", lambd)
                if isinstance(e_lam, (list, tuple, np.ndarray)):
                    e_lam = float(e_lam[-1])
                energy_field = llr
                energy_pair_d = None
                if tables is not None and int(traj.get("joint_pairs", 0)):
                    energy_field, energy_pair_d = pair_channel_arrays(
                        llr, tables, float(traj.get("pair_scale", 1.0))
                    )
                energies.append(self.compute_energy(
                    hard, energy_field, float(e_lam), energy_pair_d,
                    float(traj.get("pair_scale", 1.0)),
                ))

        candidates = np.stack(candidates, axis=0)  # (T, B, N)
        if len(trajectories) == 1:
            selected = candidates[0]
        elif combine == "energy":
            energies = np.stack(energies, axis=0)  # (T, B)
            best_idx = np.argmin(energies, axis=0)  # (B,)
            selected = candidates[best_idx, np.arange(b)]
        elif combine == "vote":
            votes = np.sum(candidates, axis=0)
            selected = (2 * votes >= len(trajectories)).astype(np.int8)
        else:
            raise ValueError(f"Unsupported combine mode: {combine}")

        total_info_bits = b * k
        errors = int(np.count_nonzero(selected[:, :k] != answers[:, :k]))
        ber = max(errors / total_info_bits, 1.0 / total_info_bits)
        score = -SCORE_SCALE * math.log10(ber)
        return score, ber, errors
