#!/usr/bin/env python3
"""Exact joint likelihood for one Gray 16-QAM I/Q component.

The judge input carries both per-bit LLRs of every Gray pair. The first bit of
each pair is strictly monotone in the received real amplitude, so the amplitude
can be recovered from that LLR and used to rebuild the joint pair likelihood

    log p(y | b0, b1) = const + a*s0 + b*s1 + d*s0*s1,   s = 1 - 2b.

This module is shared by the Python reference decoder, the batched tuner, the
C++ exporter (which mirrors the same formulas), and the consistency tests. It
is part of the QAIA channel model; it is not an offline ceiling helper.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

LEVELS = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) / math.sqrt(10.0)
BRACKET = 3.0 / math.sqrt(10.0) + 12.0
BISECTION_STEPS = 64
NEWTON_STEPS = 2
CONSISTENCY_TOL = 1e-6
ArrayLike = Union[float, np.ndarray]


def nominal_sigma(snr_db: float) -> float:
    """Return the real-dimension AWGN sigma implied by the stated SNR."""
    return 1.0 / math.sqrt(2.0 * 10.0 ** (float(snr_db) / 10.0))


def _logsumexp2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Stable log(exp(first) + exp(second)), elementwise."""
    peak = np.maximum(first, second)
    return peak + np.log1p(np.exp(-np.abs(first - second)))


def _component_loglikes(y: np.ndarray, sigma: float) -> np.ndarray:
    """Log p(y | level) without the common normalization, shape (..., 4)."""
    shifted = np.asarray(y, dtype=np.float64)[..., None] - LEVELS
    return -0.5 * (shifted * shifted) / (sigma * sigma)


def first_bit_llr_y(y: ArrayLike, sigma: float) -> np.ndarray:
    """Exact first-bit LLR at received amplitude ``y`` (decreasing in y)."""
    loglikes = _component_loglikes(y, sigma)
    return _logsumexp2(loglikes[..., 0], loglikes[..., 1]) - _logsumexp2(
        loglikes[..., 2], loglikes[..., 3]
    )


def second_bit_llr_y(y: ArrayLike, sigma: float) -> np.ndarray:
    """Exact second-bit LLR at received amplitude ``y``."""
    loglikes = _component_loglikes(y, sigma)
    return _logsumexp2(loglikes[..., 0], loglikes[..., 3]) - _logsumexp2(
        loglikes[..., 1], loglikes[..., 2]
    )


def first_bit_llr_derivative(y: ArrayLike, sigma: float) -> np.ndarray:
    """Analytic derivative d(first-bit LLR)/dy."""
    loglikes = _component_loglikes(y, sigma)
    first = loglikes[..., 0]
    second = loglikes[..., 1]
    peak = np.maximum(first, second)
    weight_first = np.exp(first - peak)
    weight_second = np.exp(second - peak)
    total = weight_first + weight_second
    mu_negative = (weight_first * LEVELS[0] + weight_second * LEVELS[1]) / total

    first = loglikes[..., 2]
    second = loglikes[..., 3]
    peak = np.maximum(first, second)
    weight_first = np.exp(first - peak)
    weight_second = np.exp(second - peak)
    total = weight_first + weight_second
    mu_positive = (weight_first * LEVELS[2] + weight_second * LEVELS[3]) / total

    return (mu_negative - mu_positive) / (sigma * sigma)


def recover_amplitude(L_first: ArrayLike, sigma: float) -> np.ndarray:
    """Invert the strictly monotone first-bit LLR to a received amplitude.

    Uses a bracketed bisection over the full plausible amplitude range, followed
    by two safeguarded Newton steps. Saturated LLRs return a finite amplitude at
    the bracket edge rather than NaN or infinity.
    """
    value = np.asarray(L_first, dtype=np.float64)
    scalar = value.ndim == 0
    flat = np.atleast_1d(value)
    low = np.full(flat.shape, -BRACKET, dtype=np.float64)
    high = np.full(flat.shape, BRACKET, dtype=np.float64)

    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        llr = first_bit_llr_y(middle, sigma)
        # The first-bit LLR decreases with y: L(middle) > target means move right.
        move_right = llr > flat
        low = np.where(move_right, middle, low)
        high = np.where(move_right, high, middle)

    y = 0.5 * (low + high)
    for _ in range(NEWTON_STEPS):
        residual = first_bit_llr_y(y, sigma) - flat
        derivative = first_bit_llr_derivative(y, sigma)
        step = np.where(np.abs(derivative) > 1e-300, residual / derivative, 0.0)
        y = np.clip(y - step, -BRACKET, BRACKET)

    return float(y[0]) if scalar else y


def pair_tables(L: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a, b, d)`` for every consecutive two-bit Gray pair.

    ``L`` has shape ``(..., n)`` with ``n % 4 == 0``. Each output has shape
    ``(..., n // 2)`` and uses the convention from the module docstring.
    """
    llr = np.asarray(L, dtype=np.float64)
    if llr.ndim < 1 or llr.shape[-1] % 4 != 0:
        raise ValueError(f"LLR last axis must be divisible by 4, got {llr.shape}")
    sigma = nominal_sigma(snr_db)
    pairs = llr.reshape(llr.shape[:-1] + (-1, 2))
    y = recover_amplitude(pairs[..., 0], sigma)
    loglikes = _component_loglikes(y, sigma)

    # Gray order is 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
    ell_00 = loglikes[..., 0]
    ell_01 = loglikes[..., 1]
    ell_11 = loglikes[..., 2]
    ell_10 = loglikes[..., 3]
    a = (ell_00 + ell_01 - ell_10 - ell_11) / 4.0
    b = (ell_00 - ell_01 + ell_10 - ell_11) / 4.0
    d = (ell_00 - ell_01 - ell_10 + ell_11) / 4.0
    return a, b, d


def consistency(L: np.ndarray, snr_db: float) -> dict:
    """Check that each partner LLR agrees with the recovered amplitude."""
    llr = np.asarray(L, dtype=np.float64)
    if llr.ndim < 1 or llr.shape[-1] % 4 != 0:
        raise ValueError(f"LLR last axis must be divisible by 4, got {llr.shape}")
    sigma = nominal_sigma(snr_db)
    pairs = llr.reshape(llr.shape[:-1] + (-1, 2))
    y = recover_amplitude(pairs[..., 0], sigma)
    predicted = second_bit_llr_y(y, sigma)
    actual = pairs[..., 1]
    errors = np.abs(predicted - actual)
    tolerance = CONSISTENCY_TOL * np.maximum(1.0, np.abs(actual))
    return {
        "pairs": int(errors.size),
        "max_abs_error": float(np.max(errors)) if errors.size else 0.0,
        "p99_abs_error": float(np.percentile(errors, 99)) if errors.size else 0.0,
        "bad_fraction": float(np.mean(errors > tolerance)) if errors.size else 0.0,
        "tolerance_multiplier": CONSISTENCY_TOL,
    }
