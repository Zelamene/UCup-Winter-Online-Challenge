#!/usr/bin/env python3
"""Sparse dSB kernels for per-case optimization.

These implement the same QAIA/dSB dynamics as the accepted dense decoder, but
accumulate check contributions through parity-check neighbor lists instead of a
dense H multiplication. This makes large synthetic searches practical.
"""

from __future__ import annotations

import numpy as np

from phase3_fixed_dsb import CheckTable, DSBConfig
from phase4_unfolded_dsb import (
    BASE_KEYS,
    MOD_KEYS,
    PARAMETER_KEYS,
    RawParams,
    sigmoid,
)


def sparse_access(checks: CheckTable) -> tuple[np.ndarray, np.ndarray]:
    """Return padded neighbor indices safe for scatter and the padding mask."""
    safe_indices = np.where(checks.mask, checks.indices, 0)
    return safe_indices, checks.mask


def check_products(spins: np.ndarray, safe: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gathered = spins[:, safe]
    neutral = np.where(mask[None, :, :], gathered, 1.0)
    return np.prod(neutral, axis=2)


def scatter_check_values(
    values: np.ndarray, safe: np.ndarray, mask: np.ndarray, n: int
) -> np.ndarray:
    batch = values.shape[0]
    out = np.zeros((batch, n), dtype=np.float64)
    rows = np.arange(batch)[:, None, None]
    cols = safe[None, :, :]
    np.add.at(out, (rows, cols), values[:, :, None] * mask[None, :, :])
    return out


def forward_sparse(
    params: RawParams,
    h_float: np.ndarray,
    llr: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
    need_cache: bool = False,
) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    del h_float
    vals = params.exp_values()
    steps = vals["dt"].size
    schedule = np.linspace(0.0, 1.0, steps)
    tau = float(vals["tau"][0])
    safe, mask = sparse_access(checks)

    x = np.tanh(llr / tau)
    y = np.zeros_like(x)
    cache: list[dict] = []

    for step in range(steps):
        spins = np.where(x >= 0, 1.0, -1.0)
        products = check_products(spins, safe, mask)

        lam_factor = np.exp(vals["lam_mod"][step] * features["check"])
        gains = vals["lam"][step] * lam_factor
        weighted_sum = scatter_check_values(products * gains, safe, mask, x.shape[1])
        force = llr + 0.5 * spins * weighted_sum

        delta_factor = np.exp(vals["delta_mod"][step] * features["var"])
        xi_factor = np.exp(vals["xi_mod"][step] * features["var"])
        ydot = -(vals["delta"][step] * delta_factor - schedule[step]) * x
        ydot += vals["xi"][step] * xi_factor * force
        y_raw = y + vals["dt"][step] * ydot
        x_raw = x + vals["dt"][step] * y_raw
        clipped = np.abs(x_raw) > 1.0
        x_next = np.where(clipped, np.where(x_raw >= 0, 1.0, -1.0), x_raw)
        y_next = np.where(clipped, 0.0, y_raw)

        if need_cache:
            cache.append({
                "x": x,
                "y": y,
                "spins": spins,
                "products": products,
                "weighted_sum": weighted_sum,
                "force": force,
                "ydot": ydot,
                "y_raw": y_raw,
                "x_raw": x_raw,
                "clipped": clipped,
                "x_next": x_next,
                "y_next": y_next,
                "xi_factor": xi_factor,
                "lam_factor": lam_factor,
                "delta_factor": delta_factor,
            })

        x, y = x_next, y_next

    return x, cache, vals


def loss_and_grad_sparse(
    params: RawParams,
    llr: np.ndarray,
    targets: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
    syn_weight: float,
    temp: float,
) -> tuple[float, dict[str, np.ndarray]]:
    x_final, cache, vals = forward_sparse(params, None, llr, checks, features, need_cache=True)
    steps = vals["dt"].size
    k = targets.shape[1]
    batch, n = llr.shape
    safe, mask = sparse_access(checks)

    z = -x_final[:, :k] / temp
    prob = sigmoid(z)
    bce = float(np.mean(np.logaddexp(0.0, z) - targets * z))

    soft_spins = np.tanh(x_final / temp)
    soft_products = check_products(soft_spins, safe, mask)
    parity_loss = 0.5 * np.mean((1.0 - soft_products) ** 2)
    loss = bce + syn_weight * parity_loss

    grad_x = np.zeros_like(x_final)
    grad_x[:, :k] = (targets - prob) / (temp * batch * k)

    grad_p = -syn_weight * (1.0 - soft_products) / soft_products.size
    safe_spins = np.where(
        np.abs(soft_spins) > 1e-6, soft_spins, np.sign(soft_spins + 1e-12) * 1e-6
    )
    grad_soft = scatter_check_values(grad_p * soft_products, safe, mask, n) / safe_spins
    grad_x += grad_soft * (1.0 - soft_spins**2) / temp

    grad_y = np.zeros_like(grad_x)
    grads = {key: np.zeros_like(params.values[key]) for key in PARAMETER_KEYS}
    schedule = np.linspace(0.0, 1.0, steps)

    for step in range(steps - 1, -1, -1):
        item = cache[step]
        clipped = item["clipped"]
        grad_x_raw = np.where(clipped, 0.0, grad_x)
        grad_y_raw = np.where(clipped, 0.0, grad_y + vals["dt"][step] * grad_x)

        grads["dt"][step] += np.sum(grad_y_raw * item["ydot"] + grad_x_raw * item["y_raw"])

        grad_x_next = grad_x_raw.copy()
        grad_y_next = grad_y_raw
        grad_ydot = vals["dt"][step] * grad_y_raw

        grad_delta_vec = -item["x"] * grad_ydot
        grads["delta"][step] += np.sum(grad_delta_vec * item["delta_factor"])
        grads["delta_mod"][step] += np.sum(
            grad_delta_vec * vals["delta"][step] * item["delta_factor"] * features["var"]
        )
        grad_x_next += grad_delta_vec

        grad_force = item["xi_factor"] * grad_ydot * vals["xi"][step]
        grads["xi"][step] += np.sum(item["force"] * grad_ydot * item["xi_factor"])
        grads["xi_mod"][step] += np.sum(
            item["force"] * grad_ydot * vals["xi"][step] * item["xi_factor"] * features["var"]
        )

        grad_x_next += 0.5 * item["weighted_sum"] * grad_force

        grad_weighted_sum = 0.5 * item["spins"] * grad_force
        grad_sum_check = np.sum(
            grad_weighted_sum[:, safe] * mask[None, :, :], axis=2
        )
        grad_lam_m = item["products"] * grad_sum_check
        grads["lam"][step] += np.sum(grad_lam_m * item["lam_factor"])
        grads["lam_mod"][step] += np.sum(
            grad_lam_m * vals["lam"][step] * item["lam_factor"] * features["check"]
        )

        check_contrib = grad_lam_m * vals["lam"][step] * item["lam_factor"]
        grad_from_products = scatter_check_values(check_contrib, safe, mask, n)
        grad_x_next += item["spins"] * grad_from_products

        grad_x, grad_y = grad_x_next, grad_y_next

    tau = float(vals["tau"][0])
    d_tanh = 1.0 - np.tanh(llr / tau) ** 2
    grads["tau"][0] += np.sum(grad_x * (-(llr / tau**2) * d_tanh))

    return loss, {
        key: grads[key] * (vals[key] if key in BASE_KEYS else 1.0)
        for key in PARAMETER_KEYS
    }


def decode_hard_sparse(
    params: RawParams,
    h_float: np.ndarray,
    llr: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
) -> np.ndarray:
    x, _, _ = forward_sparse(params, h_float, llr, checks, features)
    return (x < 0).astype(np.int8)


def decode_fixed_sparse(config: DSBConfig, h: np.ndarray, llr: np.ndarray) -> np.ndarray:
    checks = CheckTable(h)
    safe, mask = sparse_access(checks)
    schedule = np.zeros(1) if config.steps == 1 else np.linspace(0.0, 1.0, config.steps)

    x = np.tanh(llr / config.tau)
    y = np.zeros_like(x)
    for step in range(config.steps):
        spins = np.where(x >= 0, 1.0, -1.0)
        products = check_products(spins, safe, mask)
        weighted_sum = scatter_check_values(products, safe, mask, x.shape[1])
        force = llr + 0.5 * config.lambd * spins * weighted_sum
        force *= config.gradient_sign
        pump = -(config.delta - schedule[step]) * x
        y += config.dt * (pump + config.xi * force)
        x += config.dt * y
        clipped = np.abs(x) > 1.0
        if np.any(clipped):
            x[clipped] = np.where(x[clipped] >= 0, 1.0, -1.0)
            y[clipped] = 0.0
    return (x < 0).astype(np.int8)
