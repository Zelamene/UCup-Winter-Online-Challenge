#!/usr/bin/env python3
"""Reference Python decoder for the structural QAIA v3 manifest.

Only dSB/bSB check-product dynamics are implemented. There is no message
passing, reliability propagation, or post-processing stage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import kspin_reference as kspin_ref  # noqa: E402
from phase0_harness import parse_instance  # noqa: E402
from qam_pairs import pair_tables as build_pair_tables  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase8_case_manifest_v3.json"
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase8_python_v3_public.out"

MODES = ("dsb", "bsb", "kspin")
COMBINES = ("energy", "vote", "soft_vote")
MAX_TRAJECTORIES = 6
MANIFEST_STEPS = 50
SCHEDULE_KEYS = ("tau", "xi", "lambd", "delta", "dt")
EPS = 1e-12


def _finite(value: object, label: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return out


def _schedule_values(value: object, label: str, steps: int) -> np.ndarray:
    """Return a positive per-step value array from a scalar or 50-value list."""
    if isinstance(value, (list, tuple)):
        if len(value) != steps:
            raise ValueError(f"{label} must have {steps} values, got {len(value)}")
        out = np.asarray([_finite(item, f"{label}[{i}]") for i, item in enumerate(value)], dtype=np.float64)
    else:
        out = np.full(steps, _finite(value, label), dtype=np.float64)
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError(f"{label} must contain only positive finite values")
    return out


def _beta_values(value: object, label: str, steps: int) -> np.ndarray:
    """Return a per-step beta array from a finite scalar or ``steps``-value list.

    Beta is an added-energy coefficient rather than a bSB schedule scale, so zero
    and negative values are arithmetically valid; the tuner only searches positive
    values.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != steps:
            raise ValueError(f"{label} must have {steps} values, got {len(value)}")
        out = np.asarray([_finite(item, f"{label}[{i}]") for i, item in enumerate(value)], dtype=np.float64)
    else:
        out = np.full(steps, _finite(value, label), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} must contain finite values")
    return out


def _build_sparse(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Padded check-neighbor lists in ascending column order."""
    matrix = np.asarray(h)
    if matrix.ndim != 2:
        raise ValueError(f"h must be a 2-D array, got shape {matrix.shape}")
    m, n = (int(value) for value in matrix.shape)
    rows = [np.flatnonzero(matrix[row]) for row in range(m)]
    max_degree = max((len(row) for row in rows), default=0)
    safe = np.zeros((m, max_degree), dtype=np.intp)
    mask = np.zeros((m, max_degree), dtype=bool)
    for row_index, cols in enumerate(rows):
        if cols.size:
            safe[row_index, : cols.size] = cols
            mask[row_index, : cols.size] = True
    flat_cols = safe[mask]
    counts = mask.sum(axis=1)
    return safe, mask, flat_cols, counts, m, n


def _pair_channel_field(
    rates: np.ndarray, tables: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the linear pair field and per-site coupling for joint Gray pairs."""
    a, _b, d = tables
    if rates.size % 2 != 0 or 2 * a.size != rates.size or 2 * d.size != rates.size:
        raise ValueError(f"pair tables do not match {rates.size} LLRs")
    linear = np.empty_like(rates, dtype=np.float64)
    pair_d = np.empty_like(rates, dtype=np.float64)
    linear[0::2] = 2.0 * a
    linear[1::2] = 2.0 * _b
    pair_d[0::2] = d
    pair_d[1::2] = d
    return linear, pair_d


def _check_contribution(
    spins: np.ndarray,
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one check product per row and excluded products summed into variable sites."""
    safe, mask, flat_cols, counts, m, n = sparse
    if m == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(n, dtype=np.float64)
    gathered = spins[safe]
    neutral = np.where(mask, gathered, 1.0)
    deg = neutral.shape[1]
    if deg == 0:
        return np.ones(m, dtype=np.float64), np.zeros(n, dtype=np.float64)

    pref_excl = np.ones((m, deg + 1), dtype=np.float64)
    pref_excl[:, 1:] = np.cumprod(neutral, axis=1)
    suff_excl = np.ones((m, deg + 1), dtype=np.float64)
    suff_excl[:, :-1] = np.cumprod(neutral[:, ::-1], axis=1)[:, ::-1]

    products = suff_excl[:, 0]
    prod_excl = pref_excl[:, :deg] * suff_excl[:, 1:]

    weighted_sum = np.zeros(n, dtype=np.float64)
    if flat_cols.size:
        np.add.at(weighted_sum, flat_cols, prod_excl[mask])
    return products, weighted_sum


def _onsager_contribution(
    spins: np.ndarray,
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
) -> np.ndarray:
    """Per-variable sum over checks of sum_{j in c\\i} (1-x_j^2) prod_{l in c\\{i,j}} x_l^2.

    x_i times this sum is half the gradient of the polynomial
    G(x) = sum_c [sum_j prod_{l!=j} x_l^2 - (d_c-1) prod_l x_l^2], which equals 1 for every
    check on +-1 spins. Adding beta*J^2/2*G to the energy reshapes the continuous bSB
    landscape (an Onsager/TAP-style reaction term) without changing the Ising problem.
    No per-edge state is kept between steps.
    """
    safe, mask, flat_cols, counts, m, n = sparse
    total = np.zeros(n, dtype=np.float64)
    if m == 0 or not flat_cols.size:
        return total
    q = np.where(mask, spins[safe] * spins[safe], 1.0)
    u = np.where(mask, 1.0 - q, 0.0)
    deg = q.shape[1]
    pp = np.ones((m, deg + 1), dtype=np.float64)
    qp = np.zeros((m, deg + 1), dtype=np.float64)
    for t in range(deg):
        pp[:, t + 1] = pp[:, t] * q[:, t]
        qp[:, t + 1] = qp[:, t] * q[:, t] + pp[:, t] * u[:, t]
    ps = np.ones((m, deg + 1), dtype=np.float64)
    qs = np.zeros((m, deg + 1), dtype=np.float64)
    for t in range(deg - 1, -1, -1):
        ps[:, t] = ps[:, t + 1] * q[:, t]
        qs[:, t] = qs[:, t + 1] * q[:, t] + ps[:, t + 1] * u[:, t]
    per_edge = qp[:, :deg] * ps[:, 1:] + pp[:, :deg] * qs[:, 1:]
    np.add.at(total, flat_cols, per_edge[mask])
    return total


def _decode_trajectory_sparse(
    llr: np.ndarray,
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
    traj: dict,
    mode: str,
    dt: float,
    steps: int,
    pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[int], np.ndarray]:
    if mode not in MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {MODES}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if mode == "kspin":
        raise ValueError("kspin trajectories must use decode_trajectory or decode_frame")

    schedules = {
        key: _schedule_values(traj.get(key), f"trajectory {key}", steps)
        for key in ("tau", "xi", "lambd", "delta")
    }
    dt_values = _schedule_values(traj.get("dt", dt), "trajectory dt", steps)
    tau = float(schedules["tau"][0])

    rates = np.asarray(llr, dtype=np.float64)
    if rates.ndim != 1:
        raise ValueError(f"llr must be a 1-D array, got shape {rates.shape}")
    if rates.size != sparse[5]:
        raise ValueError(f"llr has {rates.size} entries, expected {sparse[5]}")

    joint_pairs = int(traj.get("joint_pairs", 0))
    if joint_pairs not in (0, 1):
        raise ValueError(f"joint_pairs must be 0 or 1, got {traj.get('joint_pairs')!r}")
    pair_gamma = _finite(traj.get("pair_gamma", 0.0), "trajectory pair_gamma")
    pair_scale = _finite(traj.get("pair_scale", 1.0), "trajectory pair_scale")
    if joint_pairs:
        if mode != "bsb":
            raise ValueError("joint_pairs requires bsb mode")
        if rates.size % 4 != 0:
            raise ValueError(f"joint_pairs requires n divisible by 4, got {rates.size}")
        if pair_tables is None:
            raise ValueError("joint_pairs requires pair tables from qam_pairs.pair_tables")
        channel_linear, pair_d = _pair_channel_field(rates, pair_tables)
        partner_index = np.arange(rates.size, dtype=np.intp) ^ 1
        pair_coupling = (2.0 * pair_scale) * pair_d
        pair_reaction = (pair_gamma * 4.0) * (pair_d * pair_d)
    else:
        channel_linear = rates
        pair_d = None

    beta_values = _beta_values(traj.get("beta", 0.0), "trajectory beta", steps)
    use_reaction = bool(np.any(beta_values != 0.0))
    parity_scale = _finite(traj.get("parity_scale", 1.0), "trajectory parity_scale")
    if (use_reaction or parity_scale != 1.0) and mode != "bsb":
        raise ValueError("beta/parity_scale require bsb mode")
    check_scale = None
    if parity_scale != 1.0:
        check_scale = np.ones(sparse[5], dtype=np.float64)
        check_scale[sparse[5] - sparse[4]:] = parity_scale  # H=[A|I]: last m columns are parity bits

    x = np.tanh(rates / tau).astype(np.float64, copy=True)
    y = np.zeros_like(x)
    for step in range(steps):
        if mode == "dsb":
            spins = np.where(x >= 0.0, 1.0, -1.0)
        else:
            soft = np.clip(x, -1.0, 1.0)
            spins = np.where(
                np.abs(soft) < EPS,
                np.where(soft >= 0.0, EPS, -EPS),
                soft,
            )

        _products, weighted_sum = _check_contribution(spins, sparse)
        lambd = float(schedules["lambd"][step])
        xi = float(schedules["xi"][step])
        delta = float(schedules["delta"][step])
        dt_step = float(dt_values[step])
        coupling = 0.5 * lambd
        if check_scale is None:
            force = channel_linear + coupling * weighted_sum
        else:
            force = channel_linear + (coupling * check_scale) * weighted_sum
        if pair_d is not None:
            partner = spins[partner_index]
            force += pair_coupling * partner
            if pair_gamma != 0.0:
                force -= pair_reaction * (1.0 - partner * partner) * spins
        beta = float(beta_values[step])
        if beta != 0.0:
            reaction = (beta * (coupling * coupling)) * spins * _onsager_contribution(spins, sparse)
            if check_scale is not None:
                reaction = reaction * check_scale
            force = force - reaction
        schedule = step / (steps - 1) if steps > 1 else 0.0
        pump = -(delta - schedule) * x
        y = y + dt_step * (pump + xi * force)
        x_new = x + dt_step * y
        clipped = np.abs(x_new) > 1.0
        if np.any(clipped):
            x_new[clipped] = np.where(x_new[clipped] >= 0.0, 1.0, -1.0)
            y[clipped] = 0.0
        x = x_new

    return (x < 0.0).astype(np.int64).tolist(), x


def decode_trajectory(
    llr: np.ndarray,
    h: np.ndarray,
    traj: dict,
    mode: str = "dsb",
    dt: float = 0.2,
    steps: int = 50,
    snr: float | None = None,
) -> tuple[list[int], np.ndarray]:
    """Run one QAIA trajectory and return (hard bits, final soft spins)."""
    _schedule_values(dt, "dt", int(steps))
    rates = np.asarray(llr, dtype=np.float64)
    if mode == "kspin":
        if snr is None:
            raise ValueError("kspin requires the case SNR")
        matrix = np.asarray(h, dtype=np.int8)
        if matrix.ndim != 2:
            raise ValueError(f"h must be 2-D, got shape {matrix.shape}")
        m, n = matrix.shape
        hard, soft = kspin_ref.decode(
            rates, matrix, int(n) - int(m), snr, traj,
            steps=int(steps), return_soft=True,
        )
        return hard.astype(np.int64).tolist(), soft
    sparse = _build_sparse(h)
    tables = None
    if int(traj.get("joint_pairs", 0)):
        if snr is None:
            raise ValueError("joint_pairs requires the case SNR")
        tables = build_pair_tables(rates, snr)
    return _decode_trajectory_sparse(
        rates, sparse, traj, mode, dt, int(steps), tables
    )


def _validate_case(case: dict) -> tuple[str, str, list[dict], float, int]:
    if not isinstance(case, dict):
        raise ValueError(f"case must be a dict, got {type(case).__name__}")
    mode = case.get("mode")
    combine = case.get("combine")
    if mode not in MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {MODES}")
    if combine not in COMBINES:
        raise ValueError(f"invalid combine {combine!r}; expected one of {COMBINES}")

    trajectories = case.get("trajectories")
    if not isinstance(trajectories, (list, tuple)):
        raise ValueError("trajectories must be a list")
    if not 1 <= len(trajectories) <= MAX_TRAJECTORIES:
        raise ValueError(f"expected 1..{MAX_TRAJECTORIES} trajectories, got {len(trajectories)}")
    if int(case.get("steps", -1)) != MANIFEST_STEPS:
        raise ValueError(f"steps must be {MANIFEST_STEPS}, got {case.get('steps')!r}")

    dt = case.get("dt", 0.2)
    _schedule_values(dt, "dt", MANIFEST_STEPS)

    for index, traj in enumerate(trajectories):
        if not isinstance(traj, dict):
            raise ValueError(f"trajectory {index} must be a dict")
        for key in ("tau", "xi", "lambd", "delta"):
            _schedule_values(traj.get(key), f"trajectory {index} {key}", MANIFEST_STEPS)
        if "dt" in traj:
            _schedule_values(traj["dt"], f"trajectory {index} dt", MANIFEST_STEPS)
        joint = int(traj.get("joint_pairs", 0))
        if joint not in (0, 1):
            raise ValueError(f"trajectory {index}: joint_pairs must be 0 or 1")
        if joint:
            if mode not in ("bsb", "kspin"):
                raise ValueError(f"trajectory {index}: joint_pairs requires bsb or kspin mode")
            if int(case.get("n", 0)) % 4 != 0:
                raise ValueError(f"trajectory {index}: joint_pairs requires n divisible by 4")
            if "snr" not in case:
                raise ValueError("joint_pairs requires the case SNR")
            _finite(traj.get("pair_gamma", 0.0), f"trajectory {index} pair_gamma")
            _finite(traj.get("pair_scale", 1.0), f"trajectory {index} pair_scale")
        if combine == "energy":
            if "energy_lambda" not in traj:
                raise ValueError(f"trajectory {index} needs energy_lambda for energy combine")
            energy_lambda = _finite(
                traj["energy_lambda"], f"trajectory {index} energy_lambda"
            )
            if energy_lambda < 0.0:
                raise ValueError(
                    f"trajectory {index} energy_lambda must be non-negative, got {energy_lambda}"
                )
    return mode, combine, list(trajectories), dt, MANIFEST_STEPS


def _energy(
    hard: list[int],
    rates: np.ndarray,
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
    lambd: float,
    pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    pair_scale: float = 1.0,
) -> float:
    spins = np.where(np.asarray(hard, dtype=np.int8) == 0, 1.0, -1.0)
    products, _weighted_sum = _check_contribution(spins, sparse)
    channel = 0.0
    for col in range(rates.size):
        channel -= float(rates[col]) * float(spins[col])
    if pair_tables is not None:
        _a, _b, d = pair_tables
        for pair in range(d.size):
            channel -= (2.0 * float(pair_scale) * float(d[pair])) * (
                float(spins[2 * pair]) * float(spins[2 * pair + 1])
            )
    penalty = 0.0
    for product in products:
        penalty += 1.0 - float(product)
    return channel + 0.5 * lambd * penalty


def _encode_kspin_parity(h: np.ndarray, k: int, info_bits: np.ndarray) -> np.ndarray:
    """Re-encode parity positions from decoded information bits."""
    _m, supports, _degree = kspin_ref.info_topology(h, k)
    n = int(np.asarray(h).shape[1])
    hard = np.empty(n, dtype=np.int64)
    hard[:k] = info_bits
    for row, cols in enumerate(supports):
        parity = 0
        for col in cols:
            parity ^= int(info_bits[col])
        hard[k + row] = parity
    return hard


def _decode_kspin_frame(
    llr: np.ndarray,
    h: np.ndarray,
    case: dict,
    trajectories: list[dict],
    steps: int,
) -> list[int]:
    """Scalar K-spin frame decode, mirroring the exported C++ operation order."""
    rates = np.asarray(llr, dtype=np.float64)
    matrix = np.asarray(h, dtype=np.int8)
    m, n = matrix.shape
    k = n - m
    if k % 2 != 0:
        raise ValueError("kspin requires even K")
    for traj in trajectories:
        if _finite(traj.get("parity_scale", 1.0), "parity_scale") != 1.0:
            raise ValueError("kspin does not use parity_scale; it must be 1")
    results = []
    for traj in trajectories:
        hard, soft = kspin_ref.decode(
            rates, h, k, float(case["snr"]), traj, steps=steps, return_soft=True
        )
        results.append((hard[:k].copy(), soft.copy()))

    if len(results) == 1:
        info_bits = results[0][0]
    elif case["combine"] == "energy":
        energies = [
            kspin_ref.energy(
                info, rates, h, k, float(case["snr"]), traj,
                select_unclipped=int(traj.get("select_unclipped", 0)),
            )
            for (info, _soft), traj in zip(results, trajectories)
        ]
        info_bits = results[int(np.argmin(energies))][0]
    elif case["combine"] == "vote":
        ones = np.sum(np.stack([info for info, _ in results]), axis=0)
        info_bits = (2 * ones >= len(results)).astype(np.int64)
    else:
        total = np.sum(np.stack([soft for _, soft in results]), axis=0)
        info_bits = (total < 0.0).astype(np.int64)
    return _encode_kspin_parity(h, k, info_bits).tolist()


def decode_frame(llr: np.ndarray, h: np.ndarray, case: dict) -> list[int]:
    """Decode one frame using the combine rule declared by ``case``."""
    mode, combine, trajectories, dt, steps = _validate_case(case)
    if mode == "kspin":
        return _decode_kspin_frame(llr, h, case, trajectories, steps)

    sparse = _build_sparse(h)
    rates = np.asarray(llr, dtype=np.float64)
    tables = (
        build_pair_tables(rates, float(case["snr"]))
        if any(int(traj.get("joint_pairs", 0)) for traj in trajectories)
        else None
    )
    results = [
        _decode_trajectory_sparse(rates, sparse, traj, mode, dt, steps, tables)
        for traj in trajectories
    ]

    if combine == "energy":
        energies = [
            _energy(
                hard,
                _pair_channel_field(rates, tables)[0] if tables is not None and int(traj.get("joint_pairs", 0)) else rates,
                sparse,
                _finite(traj["energy_lambda"], f"trajectory {index} energy_lambda"),
                tables if int(traj.get("joint_pairs", 0)) else None,
                _finite(traj.get("pair_scale", 1.0), f"trajectory {index} pair_scale"),
            )
            for index, (hard, _x) in enumerate(results)
            for traj in (trajectories[index],)
        ]
        return results[int(np.argmin(energies))][0]

    if combine == "vote":
        hards = np.asarray([hard for hard, _x in results], dtype=np.int64)
        ones = hards.sum(axis=0)
        return (2 * ones >= len(results)).astype(np.int64).tolist()

    total = np.zeros(rates.size, dtype=np.float64)
    for _hard, x_final in results:
        total += x_final
    return (total < 0.0).astype(np.int64).tolist()


def select_case(manifest: dict, n: int, k: int, snr: float) -> dict:
    """Return the nearest SNR case with an exact (n, k) match, or fallback."""
    nearest: dict | None = None
    best_diff = math.inf
    for case in manifest.get("cases", []):
        if case.get("n") != n or case.get("k") != k:
            continue
        diff = abs(_finite(case.get("snr"), "case snr") - _finite(snr, "snr"))
        if diff < best_diff:
            nearest = case
            best_diff = diff
    return nearest if nearest is not None else manifest["fallback"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    case = select_case(manifest, instance.n, instance.k, instance.snr)
    print(f"Selected case: n={case['n']} k={case['k']} snr={case['snr']}")
    print(
        f"Mode: {case['mode']}, combine: {case['combine']}, "
        f"trajectories: {len(case['trajectories'])}"
    )

    count = instance.b if args.frames <= 0 else min(args.frames, instance.b)
    lines = []
    for index in range(count):
        hard = decode_frame(instance.llr[index], instance.h, case)
        lines.append(" ".join(str(int(value)) for value in hard))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({count} frames)")


if __name__ == "__main__":
    main()
