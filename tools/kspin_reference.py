#!/usr/bin/env python3
"""Frozen scalar reference for the K-spin QAIA decoder.

The systematic generator has ``H = [A | I]``.  Substituting every parity spin
by the product of its row's information spins leaves a degree-six spin problem
whose every state is a codeword.  The continuous bSB dynamics run on the K
information spins only; parity output bits are re-encoded at the end.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TOOLS_DIR))

from qam_pairs import pair_tables as build_pair_tables  # noqa: E402

EPS = 1e-12
STEPS = 50

DEFAULT_PARAMS: dict[str, Any] = {
    "tau": 2.976,
    "xi": 1.0,
    "delta": 3.441,
    "dt": 0.47,
    "g": 0.98,
    "clip": 2.8,
    "beta": 0.317,
    "pair_gamma": 0.0,
    "joint_pairs": 1,
}


def _finite(value: object, label: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return out


def normalise_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return validated scalar K-spin parameters with defaults filled in."""
    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    result = {
        "tau": _finite(merged["tau"], "tau"),
        "xi": _finite(merged["xi"], "xi"),
        "delta": _finite(merged["delta"], "delta"),
        "dt": _finite(merged["dt"], "dt"),
        "g": _finite(merged["g"], "g"),
        "clip": _finite(merged["clip"], "clip"),
        "beta": _finite(merged["beta"], "beta"),
        "pair_gamma": _finite(merged["pair_gamma"], "pair_gamma"),
        "joint_pairs": int(merged["joint_pairs"]),
        "select_unclipped": int(merged.get("select_unclipped", 0)),
    }
    if result["tau"] <= 0.0 or result["dt"] <= 0.0 or result["clip"] <= 0.0:
        raise ValueError("tau, dt and clip must be positive")
    if result["joint_pairs"] not in (0, 1):
        raise ValueError("joint_pairs must be 0 or 1")
    if result["select_unclipped"] not in (0, 1):
        raise ValueError("select_unclipped must be 0 or 1")
    return result


def info_topology(h: np.ndarray, k: int) -> tuple[int, list[list[int]], int]:
    """Validate ``H=[A|I]`` and return (m, info columns per check, info degree)."""
    matrix = np.asarray(h, dtype=np.int8)
    if matrix.ndim != 2:
        raise ValueError(f"h must be 2-D, got shape {matrix.shape}")
    n, rows = matrix.shape[1], matrix.shape[0]
    m = n - int(k)
    if m != rows or int(k) <= 0 or int(k) % 2 != 0:
        raise ValueError(f"invalid systematic dimensions: n={n} k={k} rows={rows}")
    identity = matrix[:, int(k):]
    if not np.array_equal(identity, np.eye(m, dtype=np.int8)):
        raise ValueError("K-spin requires parity columns to be the identity")
    supports: list[list[int]] = []
    degree = 0
    for row in range(m):
        cols = np.flatnonzero(matrix[row, : int(k)]).astype(int).tolist()
        if not cols:
            raise ValueError(f"check {row} has no information-bit support")
        degree = max(degree, len(cols))
        supports.append(cols)
    return m, supports, degree


def parity_weights(llr: np.ndarray, k: int, g: float, clip: float) -> np.ndarray:
    """Soft-clipped parity weights ``g * clip * tanh(L_par / clip)``."""
    rates = np.asarray(llr, dtype=np.float64)
    return float(g) * float(clip) * np.tanh(rates[int(k):] / float(clip))


def pair_channel(rates: np.ndarray, tables: tuple[np.ndarray, np.ndarray, np.ndarray], k: int):
    """Return linear and D=2d arrays aligned with the information positions."""
    a, b, d = tables
    linear = np.empty(k, dtype=np.float64)
    d_values = np.empty(k, dtype=np.float64)
    linear[0::2] = 2.0 * a[: k // 2]
    linear[1::2] = 2.0 * b[: k // 2]
    d_values[0::2] = 2.0 * d[: k // 2]
    d_values[1::2] = 2.0 * d[: k // 2]
    return linear, d_values


def _onsager_edge(spins: np.ndarray) -> np.ndarray:
    """e_{c,i} for one check, in ascending support order."""
    q = spins * spins
    u = 1.0 - q
    deg = len(spins)
    pp = np.ones(deg + 1, dtype=np.float64)
    qp = np.zeros(deg + 1, dtype=np.float64)
    for index in range(deg):
        pp[index + 1] = pp[index] * q[index]
        qp[index + 1] = qp[index] * q[index] + pp[index] * u[index]
    ps = np.ones(deg + 1, dtype=np.float64)
    qs = np.zeros(deg + 1, dtype=np.float64)
    for index in range(deg - 1, -1, -1):
        ps[index] = ps[index + 1] * q[index]
        qs[index] = qs[index + 1] * q[index] + ps[index + 1] * u[index]
    return qp[:deg] * ps[1:] + pp[:deg] * qs[1:]


def decode(
    llr: np.ndarray,
    h: np.ndarray,
    k: int,
    snr: float,
    params: dict[str, Any],
    steps: int = STEPS,
    return_soft: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode one frame and return the full N-bit hard codeword.

    With ``return_soft=True`` also return the final K information spins so the
    scalar reference can reproduce soft-vote portfolios.
    """
    cfg = normalise_params(params)
    rates = np.asarray(llr, dtype=np.float64)
    k = int(k)
    n = int(rates.size)
    if rates.ndim != 1 or rates.size != n:
        raise ValueError("llr must be a 1-D array")
    m, supports, degree = info_topology(h, k)
    if degree != 0 and any(len(support) != degree for support in supports):
        raise ValueError("K-spin reference expects a regular information check degree")
    if steps < 1:
        raise ValueError("steps must be positive")
    if cfg["joint_pairs"] and k % 4 != 0:
        raise ValueError("joint_pairs requires K divisible by 4")

    if cfg["joint_pairs"]:
        tables = build_pair_tables(rates, float(snr))
        channel_linear, pair_d = pair_channel(rates, tables, k)
    else:
        channel_linear, pair_d = rates[:k].copy(), None
    weights = parity_weights(rates, k, cfg["g"], cfg["clip"])

    x = np.tanh(rates[:k] / cfg["tau"]).astype(np.float64, copy=True)
    y = np.zeros(k, dtype=np.float64)
    for step in range(int(steps)):
        soft = np.clip(x, -1.0, 1.0)
        soft = np.where(np.abs(soft) < EPS, np.where(soft >= 0.0, EPS, -EPS), soft)
        if pair_d is not None:
            partner = soft[np.arange(k, dtype=np.intp) ^ 1]
            field = channel_linear + pair_d * partner
            if cfg["pair_gamma"] != 0.0:
                # pair_d is D_P = 2*d_P, matching the batched and C++ forms
                # gamma * D_P^2 * (1 - x_partner^2) * x_i.
                field = field - (
                    cfg["pair_gamma"] * pair_d * pair_d
                    * (1.0 - partner * partner) * soft
                )
        else:
            field = channel_linear

        weighted = np.zeros(k, dtype=np.float64)
        reaction = np.zeros(k, dtype=np.float64)
        for row, cols in enumerate(supports):
            values = soft[cols]
            pref = np.ones(degree + 1, dtype=np.float64)
            for index in range(degree):
                pref[index + 1] = pref[index] * values[index]
            suff = np.ones(degree + 1, dtype=np.float64)
            for index in range(degree - 1, -1, -1):
                suff[index] = suff[index + 1] * values[index]
            excluded = pref[:degree] * suff[1:]
            weighted[cols] += weights[row] * excluded
            if cfg["beta"] != 0.0:
                edge = _onsager_edge(values)
                reaction[cols] += cfg["beta"] * weights[row] * weights[row] * values * edge

        force = field + weighted - reaction
        schedule = step / (steps - 1) if steps > 1 else 0.0
        pump = -(cfg["delta"] - schedule) * x
        y = y + cfg["dt"] * (pump + cfg["xi"] * force)
        x_new = x + cfg["dt"] * y
        clipped = np.abs(x_new) > 1.0
        if np.any(clipped):
            x_new[clipped] = np.where(x_new[clipped] >= 0.0, 1.0, -1.0)
            y[clipped] = 0.0
        x = x_new
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(force))):
            raise FloatingPointError(f"non-finite K-spin state at step {step}")

    info_bits = (x < 0.0).astype(np.int8)
    hard = np.empty(n, dtype=np.int8)
    hard[:k] = info_bits
    for row, cols in enumerate(supports):
        parity = 0
        for col in cols:
            parity ^= int(info_bits[col])
        hard[k + row] = parity
    if return_soft:
        return hard, x
    return hard


def energy(
    info_bits: np.ndarray,
    llr: np.ndarray,
    h: np.ndarray,
    k: int,
    snr: float,
    params: dict[str, Any],
    select_unclipped: int | None = None,
) -> float:
    """Exact K-spin energy on a hard information-bit codeword."""
    cfg = normalise_params(params)
    if select_unclipped is not None:
        cfg["select_unclipped"] = int(select_unclipped)
    rates = np.asarray(llr, dtype=np.float64)
    bits = np.asarray(info_bits, dtype=np.int8)
    m, supports, _degree = info_topology(h, k)
    if bits.size != int(k):
        raise ValueError("info_bits must have K entries")
    spins = 1.0 - 2.0 * bits.astype(np.float64)
    if cfg["joint_pairs"]:
        tables = build_pair_tables(rates, float(snr))
        linear, pair_d = pair_channel(rates, tables, k)
        partner = spins[np.arange(k, dtype=np.intp) ^ 1]
        field = linear + pair_d * partner
    else:
        field = rates[:k]
    value = -float(np.sum(field * spins))
    weights = rates[k:k + m].copy() if cfg["select_unclipped"] else parity_weights(
        rates, k, cfg["g"], cfg["clip"]
    )
    for row, cols in enumerate(supports):
        product = 1.0
        for col in cols:
            product *= spins[col]
        value -= weights[row] * product
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, help="JSON parameter file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    from phase0_harness import parse_instance  # noqa: E402

    text = args.input.read_text(encoding="utf-8")
    instance = parse_instance(text, str(args.input))
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
    count = instance.b if args.frames <= 0 else min(args.frames, instance.b)
    lines = []
    for index in range(count):
        hard = decode(instance.llr[index], instance.h, instance.k, instance.snr, params)
        lines.append(" ".join(str(int(value)) for value in hard))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({count} frames)")


if __name__ == "__main__":
    main()
