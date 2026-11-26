#!/usr/bin/env python3
"""Tier 3 diagnostic dynamics: quadratized SB and noisy SimCIM.

These are offline screening tools only. They are not wired into the accepted
submission exporter and must never be copied into a submitted source file.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from decode_v3_python import (
    MAX_TRAJECTORIES,
    MODES,
    _build_sparse,
    _check_contribution,
    _energy,
    _finite,
    _schedule_values,
)

SCHEDULE_KEYS = ("tau", "xi", "lambd", "delta", "dt")
EPS = 1e-12


def _schedules(params: dict[str, Any], steps: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    defaults = {"xi": 1.2, "lambd": 3.2, "delta": 0.001, "dt": 0.25}
    for key in ("xi", "lambd", "delta", "dt"):
        out[key] = _schedule_values(params.get(key, defaults[key]), f"{key}", steps)
    out["tau"] = np.full(steps, _finite(params.get("tau", 2.2), "tau"))
    return out


def _clipped(x: np.ndarray) -> np.ndarray:
    out = np.clip(x, -1.0, 1.0)
    return np.where(np.abs(out) < EPS, np.where(out >= 0.0, EPS, -EPS), out)


def _get_spin(z: np.ndarray, is_aux: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    out = z[idx].astype(np.float64, copy=True)
    mask = np.asarray(is_aux, dtype=bool)
    if np.any(mask):
        out[mask] = z[n + idx[mask]]
    return out


def build_qsb(sparse: tuple) -> dict[str, np.ndarray]:
    """Build auxiliary-spin arrays for pairwise quadratization of checks."""
    safe, mask, _flat_cols, counts, m, n = sparse
    left_is_aux: list[bool] = []
    left_idx: list[int] = []
    right_is_aux: list[bool] = []
    right_idx: list[int] = []
    energy_left_is_aux: list[bool] = []
    energy_left_idx: list[int] = []
    energy_right_idx: list[int] = []

    for row in range(m):
        cols = safe[row, : counts[row]]
        degree = int(cols.size)
        if degree < 3:
            raise ValueError(f"check {row} has degree {degree}; expected at least 3")
        chain_len = degree - 3

        prev_left_is_aux = False
        prev_left_idx = int(cols[0])
        for i in range(chain_len):
            if i == 0:
                li, laux = False, int(cols[0])
                ri, raux = False, int(cols[1])
            else:
                li, laux = True, len(left_is_aux) - 1
                ri, raux = False, int(cols[i + 1])
            left_is_aux.append(laux)
            left_idx.append(laux)
            right_is_aux.append(raux)
            right_idx.append(raux)
            prev_left_is_aux, prev_left_idx = True, len(left_is_aux) - 1

        aux_b = len(left_is_aux)
        left_is_aux.append(False)
        left_idx.append(int(cols[degree - 2]))
        right_is_aux.append(False)
        right_idx.append(int(cols[degree - 1]))

        energy_left_is_aux.append(prev_left_is_aux)
        energy_left_idx.append(prev_left_idx)
        energy_right_idx.append(aux_b)

    aux_count = len(left_is_aux)
    return {
        "left_is_aux": np.asarray(left_is_aux, dtype=bool),
        "left_idx": np.asarray(left_idx, dtype=np.intp),
        "right_is_aux": np.asarray(right_is_aux, dtype=bool),
        "right_idx": np.asarray(right_idx, dtype=np.intp),
        "energy_left_is_aux": np.asarray(energy_left_is_aux, dtype=bool),
        "energy_left_idx": np.asarray(energy_left_idx, dtype=np.intp),
        "energy_right_idx": np.asarray(energy_right_idx, dtype=np.intp),
        "aux_count": np.asarray([aux_count], dtype=np.intp),
    }


def decode_qsb(
    llr: np.ndarray,
    sparse: tuple,
    params: dict[str, Any],
    steps: int = 50,
) -> tuple[list[int], np.ndarray]:
    """Ballistic SB on a quadratized Ising Hamiltonian."""
    n = sparse[5]
    rate = np.asarray(llr, dtype=np.float64)
    if rate.ndim != 1 or rate.size != n:
        raise ValueError(f"llr must have shape ({n},), got {rate.shape}")
    q = build_qsb(sparse)
    aux_count = int(q["aux_count"][0])
    total = n + aux_count
    sched = _schedules(params, steps)
    lambda_q = _finite(params.get("lambda_q", 3.0), "lambda_q")
    lambda_c = _finite(params.get("lambda_c", float(np.mean(sched["lambd"]))), "lambda_c")
    hybrid_lambda = _finite(params.get("hybrid_lambda", 0.0), "hybrid_lambda")
    aux_pump = bool(params.get("aux_pump", True))

    z = np.empty(total, dtype=np.float64)
    z[:n] = np.tanh(rate / sched["tau"][0])
    for index in range(aux_count):
        left = _get_spin(z, q["left_is_aux"], q["left_idx"], n)[index]
        right = _get_spin(z, q["right_is_aux"], q["right_idx"], n)[index]
        z[n + index] = np.clip(left * right, -1.0, 1.0)

    velocity = np.zeros(total, dtype=np.float64)
    for step in range(steps):
        soft = _clipped(z)
        left = _get_spin(soft, q["left_is_aux"], q["left_idx"], n)
        right = _get_spin(soft, q["right_is_aux"], q["right_idx"], n)
        residual = soft[n:] - left * right
        coef = 2.0 * lambda_q * residual

        force = np.zeros(total, dtype=np.float64)
        force[:n] += rate
        if hybrid_lambda > 0.0:
            _products, weighted_sum = _check_contribution(soft, sparse)
            force[:n] += 0.5 * hybrid_lambda * weighted_sum

        force[n:] += -coef
        left_target = np.where(q["left_is_aux"], n + q["left_idx"], q["left_idx"])
        right_target = np.where(q["right_is_aux"], n + q["right_idx"], q["right_idx"])
        np.add.at(force, left_target, coef * right)
        np.add.at(force, right_target, coef * left)

        energy_left = _get_spin(
            soft, q["energy_left_is_aux"], q["energy_left_idx"], n
        )
        energy_right_target = n + q["energy_right_idx"]
        energy_right = soft[energy_right_target]
        energy_left_target = np.where(
            q["energy_left_is_aux"], n + q["energy_left_idx"], q["energy_left_idx"]
        )
        np.add.at(force, energy_left_target, lambda_c * energy_right)
        np.add.at(force, energy_right_target, lambda_c * energy_left)

        schedule = step / (steps - 1) if steps > 1 else 0.0
        pump = -(float(sched["delta"][step]) - schedule) * z
        if not aux_pump:
            pump[n:] = 0.0
        dt = float(sched["dt"][step])
        velocity = velocity + dt * (pump + float(sched["xi"][step]) * force)
        z_new = z + dt * velocity
        clipped = np.abs(z_new) > 1.0
        if np.any(clipped):
            z_new[clipped] = np.where(z_new[clipped] >= 0.0, 1.0, -1.0)
            velocity[clipped] = 0.0
        z = z_new

    data = z[:n]
    return (data < 0.0).astype(np.int64).tolist(), data


def decode_simcim(
    llr: np.ndarray,
    sparse: tuple,
    params: dict[str, Any],
    steps: int = 50,
) -> tuple[list[int], np.ndarray]:
    """Noisy SimCIM-style continuous-spin dynamics."""
    n = sparse[5]
    rate = np.asarray(llr, dtype=np.float64)
    if rate.ndim != 1 or rate.size != n:
        raise ValueError(f"llr must have shape ({n},), got {rate.shape}")
    sched = _schedules(params, steps)
    saturation = _finite(params.get("saturation", 1.5), "saturation")
    noise0 = _finite(params.get("noise0", 0.1), "noise0")
    noise_decay = _finite(params.get("noise_decay", 1.0), "noise_decay")
    if noise0 < 0.0:
        raise ValueError("noise0 must be non-negative")
    seed = int(params.get("seed", 0))
    pump_delta = params.get("pump_delta")
    rng = np.random.default_rng(seed)

    x = np.tanh(rate / sched["tau"][0])
    for step in range(steps):
        soft = _clipped(x)
        _products, weighted_sum = _check_contribution(soft, sparse)
        force = rate + 0.5 * float(sched["lambd"][step]) * weighted_sum
        schedule = step / (steps - 1) if steps > 1 else 0.0
        noise_amp = noise0 * math.pow(max(1.0 - schedule, 0.0), noise_decay)
        dt = float(sched["dt"][step])
        pump = 0.0
        if pump_delta is not None:
            pump = -(float(_finite(pump_delta, "pump_delta")) - schedule) * x
        update = (
            -x
            + saturation * np.tanh(x)
            + float(sched["xi"][step]) * force
            + pump
            + noise_amp * rng.standard_normal(n)
        )
        x_new = x + dt * update
        clipped = np.abs(x_new) > 1.0
        if np.any(clipped):
            x_new[clipped] = np.where(x_new[clipped] >= 0.0, 1.0, -1.0)
        x = x_new

    return (x < 0.0).astype(np.int64).tolist(), x


def decode_candidate(
    llr: np.ndarray,
    h: np.ndarray,
    mode: str,
    params: dict[str, Any],
    steps: int = 50,
) -> list[int]:
    """Decode one frame with a Tier 3 diagnostic mode."""
    sparse = _build_sparse(h)
    if mode == "qsb":
        hard, _soft = decode_qsb(llr, sparse, params, steps)
        return hard
    if mode == "simcim":
        hard, _soft = decode_simcim(llr, sparse, params, steps)
        return hard
    raise ValueError(f"invalid Tier 3 mode {mode!r}; expected 'qsb' or 'simcim'")
