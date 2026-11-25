#!/usr/bin/env python3
"""Generate systematic LDPC parity-check matrices with the public ensemble shape.

Public facts modelled here:
  * every H row has weight 7;
  * the last m = n - k columns form an identity matrix;
  * the info part A therefore has row weight 6 and total weight 6*m;
  * info column weights are near 6*m/k (6, 3, 2 for rates 1/2, 2/3, 3/4).

The generator uses ``numpy.random.Generator`` only and is reproducible for a
fixed seed.
"""

from __future__ import annotations

import numpy as np

ROW_WEIGHT = 7
INFO_ROW_WEIGHT = ROW_WEIGHT - 1


def _column_weights(k: int, m: int, irregular: float, rng: np.random.Generator) -> np.ndarray:
    """Balanced info-column weight vector with total 6*m.

    ``irregular`` moves a bounded number of unit weights between columns while
    preserving the total.  It is a perturbation strength, not a new total.
    """
    if m <= 0 or k <= 0:
        raise ValueError(f"invalid dimensions: k={k}, m={m}")
    if irregular < 0:
        raise ValueError(f"irregular must be non-negative, got {irregular}")

    total = INFO_ROW_WEIGHT * m
    base, remainder = divmod(total, k)
    weights = np.full(k, base, dtype=np.int64)
    if remainder:
        weights[rng.permutation(k)[:remainder]] += 1

    if irregular > 0.0:
        max_deviation = max(1, int(round(irregular * weights.mean())))
        low = max(1, base - max_deviation)
        high = min(m, base + max_deviation)
        transfers = max(1, int(round(irregular * k / 2.0)))
        for _ in range(transfers):
            donors = np.flatnonzero(weights > low)
            if donors.size == 0:
                break
            donor = int(rng.choice(donors))
            recipients = np.flatnonzero((weights < high) & (np.arange(k) != donor))
            if recipients.size == 0:
                break
            recipient = int(rng.choice(recipients))
            weights[donor] -= 1
            weights[recipient] += 1

    assert weights.shape == (k,)
    assert np.all(weights > 0) and np.all(weights <= m)
    assert int(weights.sum()) == total
    return weights


def _build_info_matrix(weights: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Build an m x k 0/1 matrix with row sums 6 and prescribed column sums.

    Rows are filled one at a time.  Columns whose remaining demand equals the
    number of unfilled rows are forced into the current row; this guarantees
    that all demands can still be met.  Remaining slots are sampled without
    replacement with probability proportional to the outstanding demand.
    """
    k = int(weights.size)
    demand = weights.astype(np.int64, copy=True)
    a = np.zeros((m, k), dtype=np.int8)

    for row in range(m):
        rows_left = m - row
        urgent = np.flatnonzero(demand == rows_left)
        if urgent.size > INFO_ROW_WEIGHT:
            raise RuntimeError("infeasible column demands in generate_h")
        chosen = [int(value) for value in urgent]
        slots = INFO_ROW_WEIGHT - len(chosen)

        if slots > 0:
            pool = np.flatnonzero(demand > 0)
            if urgent.size:
                pool = pool[~np.isin(pool, urgent)]
            probabilities = demand[pool].astype(np.float64)
            probabilities /= probabilities.sum()
            picks = rng.choice(pool, size=slots, replace=False, p=probabilities)
            chosen.extend(int(value) for value in picks)

        if len(chosen) != INFO_ROW_WEIGHT or len(set(chosen)) != INFO_ROW_WEIGHT:
            raise RuntimeError("could not choose distinct info columns for a row")
        a[row, chosen] = 1
        demand[chosen] -= 1
        if np.any(demand < 0):
            raise RuntimeError("negative column demand in generate_h")

    assert np.all(demand == 0)
    assert np.array_equal(a.sum(axis=1), np.full(m, INFO_ROW_WEIGHT, dtype=np.int64))
    assert np.array_equal(a.sum(axis=0), weights)
    return a


def generate_h(
    n: int,
    rate: float,
    rng: np.random.Generator,
    irregular: float = 0.0,
) -> np.ndarray:
    """Return an int8 H of shape (n - k, n) with systematic [A | I] form.

    ``k`` is ``round(n * rate)``.  Row sums are exactly 7, the final m columns
    are the identity, and info-column sums are balanced around 6*m/k.  Each
    row uses distinct info columns because A is a binary matrix.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    k = int(round(float(n) * float(rate)))
    m = int(n) - k
    if not 0 < k < int(n):
        raise ValueError(f"rate {rate} gives invalid k={k} for n={n}")

    weights = _column_weights(k, m, float(irregular), rng)
    a = _build_info_matrix(weights, m, rng)
    h = np.concatenate((a, np.eye(m, dtype=np.int8)), axis=1).astype(np.int8, copy=False)

    checks = validate_h(h, rate)
    assert checks["all_checks_passed"]
    return h


def validate_h(h: np.ndarray, rate: float) -> dict:
    """Check public H structure and return descriptive statistics.

    Raises ``ValueError`` for broken shape, non-binary entries, wrong row
    weight, or a non-identity parity block.  The generator matrix implied by
    the systematic form is also checked against ``H * G^T = 0 (mod 2)``.
    """
    array = np.asarray(h)
    if array.ndim != 2 or array.shape[0] >= array.shape[1]:
        raise ValueError(f"H must be 2-D with more columns than rows, got {array.shape}")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("H contains values other than 0 and 1")

    m, n = array.shape
    k = n - m
    expected_k = int(round(float(n) * float(rate)))
    if k != expected_k:
        raise ValueError(f"rate {rate} implies k={expected_k}, but H has k={k}")

    row_sums = array.astype(np.int64).sum(axis=1)
    if not np.all(row_sums == ROW_WEIGHT):
        counts = dict(zip(*np.unique(row_sums, return_counts=True)))
        raise ValueError(f"row weights must all be {ROW_WEIGHT}, got {counts}")

    identity = np.eye(m, dtype=array.dtype)
    parity_identity = bool(np.array_equal(array[:, k:], identity))
    if not parity_identity:
        raise ValueError("the last m columns of H are not an identity matrix")

    a = array[:, :k]
    info_row_sums = a.astype(np.int64).sum(axis=1)
    info_col_sums = a.astype(np.int64).sum(axis=0)
    target = INFO_ROW_WEIGHT * m / k

    g = np.concatenate((np.eye(k, dtype=np.int8), a.astype(np.int8).T), axis=1)
    syndrome = (g.astype(np.int16) @ array.T.astype(np.int16)) % 2
    g_valid = bool(np.all(syndrome == 0))

    stats = {
        "n": n,
        "k": k,
        "m": m,
        "rate": k / n,
        "row_weight": ROW_WEIGHT,
        "row_weight_counts": {
            int(value): int(count) for value, count in zip(*np.unique(row_sums, return_counts=True))
        },
        "info_row_weight": float(info_row_sums.mean()),
        "info_column_weight_target": float(target),
        "info_column_weight_min": int(info_col_sums.min()),
        "info_column_weight_max": int(info_col_sums.max()),
        "info_column_weight_mean": float(info_col_sums.mean()),
        "info_column_weight_counts": {
            int(value): int(count)
            for value, count in zip(*np.unique(info_col_sums, return_counts=True))
        },
        "parity_block_identity": parity_identity,
        "generator_matches_h": g_valid,
        "all_checks_passed": bool(parity_identity and g_valid and np.all(row_sums == ROW_WEIGHT)),
    }
    assert stats["all_checks_passed"]
    return stats


if __name__ == "__main__":
    demo_rng = np.random.default_rng(0)
    for demo_rate in (0.5, 2.0 / 3.0, 0.75):
        demo_h = generate_h(648, demo_rate, demo_rng)
        print(f"rate={demo_rate:.6f} {validate_h(demo_h, demo_rate)}")
