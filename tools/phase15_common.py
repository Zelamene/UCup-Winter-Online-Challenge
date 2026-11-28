#!/usr/bin/env python3
"""Shared frame-matched evaluation helpers for Phase 15.

All Phase 15 workstreams use this module so that candidate trajectories are
compared with the compiled v17 baseline on identical fixed-rule frames, with
the same Python/batched decoder, energy combine, and paired bootstrap.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from baseline_cache import load_baseline_errors  # noqa: E402
from bsb_vectorized import BatchedBSB  # noqa: E402
from qam_pairs import pair_tables as build_pair_tables  # noqa: E402
from real_bench import RATE_SNR_DB, build_case  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
STEPS = 50
NOISE_SCALE = 1.001120
V17_MANIFEST = ROOT / "submissions" / "v17_joint_pairs" / "manifest.json"
V17_CACHE = ROOT / "artifacts" / "baseline_cache_v17"
V18_CACHE = ROOT / "artifacts" / "baseline_cache_v18"
PHASE15_ARTIFACTS = ROOT / "artifacts" / "phase15"
CONFIG_KEYS = (
    "tau", "xi", "lambd", "delta", "dt", "beta", "parity_scale",
    "joint_pairs", "pair_gamma", "pair_scale", "energy_lambda",
    "g", "clip", "select_unclipped",
)
CASE_SPECS = tuple(
    (index + 1, n, int(round(n * rate)), rate, snr)
    for index, (n, rate, snr) in enumerate(
        (n, rate, snr)
        for n in (648, 1296, 1944, 3888)
        for rate, snr in RATE_SNR_DB
    )
)
CASE_BY_DIM = {(n, k): (index, rate, snr) for index, n, k, rate, snr in CASE_SPECS}


def configured_snr(rate: float) -> float:
    for configured_rate, snr in RATE_SNR_DB:
        if abs(float(configured_rate) - float(rate)) < 1e-4:
            return float(snr)
    raise ValueError(f"no configured SNR for rate {rate}")


def score_of(errors: int | np.integer, total: int) -> float:
    ber = max(float(errors) / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def score_from_errors(errors: np.ndarray, k: int) -> float:
    total = int(errors.sum())
    return score_of(total, len(errors) * k)


def per_frame_errors(hard: np.ndarray, case: dict) -> np.ndarray:
    k = int(case["k"])
    return np.count_nonzero(hard[:, :k] != case["answers"][:, :k], axis=1)


def paired_bootstrap(
    base_errors: np.ndarray,
    candidate_errors: np.ndarray,
    k: int,
    samples: int = 1000,
    seed: int = 5150,
) -> tuple[float, float]:
    """95% paired-bootstrap CI for candidate score minus baseline score."""
    base = np.asarray(base_errors, dtype=np.int64)
    candidate = np.asarray(candidate_errors, dtype=np.int64)
    if base.shape != candidate.shape:
        raise ValueError(f"paired bootstrap shape mismatch: {base.shape} vs {candidate.shape}")
    frames = len(base)
    total = frames * int(k)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, frames, size=(int(samples), frames))
    base_scores = np.empty(int(samples), dtype=float)
    candidate_scores = np.empty(int(samples), dtype=float)
    for row in range(int(samples)):
        draw = indices[row]
        base_scores[row] = score_of(base[draw].sum(), total)
        candidate_scores[row] = score_of(candidate[draw].sum(), total)
    deltas = candidate_scores - base_scores
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def load_v17_config(n: int, k: int, manifest: Path = V17_MANIFEST) -> dict:
    """Return the single accepted v17 trajectory config for (n, k)."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    cases = [
        case for case in payload.get("cases", [])
        if int(case.get("n", -1)) == int(n) and int(case.get("k", -1)) == int(k)
    ]
    if len(cases) != 1:
        raise ValueError(f"{manifest}: expected one case for {(n, k)}, got {len(cases)}")
    trajectories = cases[0].get("trajectories", [])
    if len(trajectories) != 1:
        raise ValueError(f"{manifest}: expected one trajectory for {(n, k)}")
    traj = trajectories[0]
    snr = float(cases[0]["snr"])
    cfg = {key: float(traj.get(key, 1.0 if key in ("pair_scale", "energy_lambda") else 0.0))
           for key in CONFIG_KEYS}
    cfg["joint_pairs"] = int(traj.get("joint_pairs", 0))
    cfg["n"] = int(n)
    cfg["k"] = int(k)
    cfg["snr"] = snr
    cfg["mode"] = str(cases[0].get("mode", "bsb"))
    cfg["combine"] = str(cases[0].get("combine", "energy"))
    cfg["dt"] = float(traj.get("dt", cases[0].get("dt", 0.2)))
    return cfg


def with_overrides(config: dict, **overrides) -> dict:
    result = dict(config)
    for key, value in overrides.items():
        if key not in CONFIG_KEYS and key not in ("n", "k", "snr", "mode", "combine"):
            raise KeyError(f"unknown config key {key!r}")
        result[key] = value
    return result


@dataclass
class CaseWorkspace:
    """Generated frames, topology, pair tables and cached v17 errors for one case."""

    n: int
    k: int
    seed: int
    frames: int
    snr: float | None = None
    cache_dir: Path = V17_CACHE
    frame_slice: int | None = None
    _case: dict = field(init=False, repr=False)
    _sim: BatchedBSB = field(init=False, repr=False)
    _tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(init=False, repr=False)
    _baseline: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n = int(self.n)
        self.k = int(self.k)
        self.seed = int(self.seed)
        self.frames = int(self.frames)
        if self.snr is None:
            self.snr = configured_snr(self.k / self.n)
        self._case = build_case(
            self.n, self.k, self.frames, float(self.snr), self.seed,
            noise_scale=NOISE_SCALE, llr_sigma="nominal",
        )
        self._sim = BatchedBSB(self._case["h"])
        self._baseline = load_baseline_errors(
            Path(self.cache_dir), self.n, self.k, self.seed, expected_frames=self.frames
        )
        if self.frame_slice is not None:
            limit = int(self.frame_slice)
            if not 1 <= limit <= self.frames:
                raise ValueError(f"frame_slice must be in 1..{self.frames}, got {limit}")
            self._case["llr"] = self._case["llr"][:limit]
            self._case["answers"] = self._case["answers"][:limit]
            self._baseline = self._baseline[:limit]
            self.frames = limit
        self._tables = build_pair_tables(self._case["llr"], float(self.snr))

    @property
    def case(self) -> dict:
        return self._case

    @property
    def sim(self) -> BatchedBSB:
        return self._sim

    @property
    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._tables is None:
            raise RuntimeError("pair tables are unavailable")
        return self._tables

    @property
    def baseline_errors(self) -> np.ndarray:
        return self._baseline

    @property
    def baseline_score(self) -> float:
        return score_from_errors(self._baseline, self.k)

    def summary(self) -> dict:
        return {
            "n": self.n, "k": self.k, "seed": self.seed, "frames": self.frames,
            "snr": float(self.snr), "noise_scale": NOISE_SCALE,
            "baseline_errors": int(self._baseline.sum()),
            "baseline_score": self.baseline_score,
        }


def decode_hard(
    workspace: CaseWorkspace,
    config: dict,
    debug_finite: bool = False,
    batch_size: int | None = None,
) -> np.ndarray:
    """Decode one configuration and return hard bits of shape (frames, n)."""
    if str(config.get("mode", "bsb")) == "kspin":
        return decode_kspin(workspace, config, debug_finite, batch_size)
    if any(isinstance(config.get(key), (list, tuple, np.ndarray))
           for key in ("xi", "lambd", "delta", "dt", "beta")):
        return decode_hard_schedules(workspace, config, debug_finite)
    joint = int(config.get("joint_pairs", 0))
    if joint and workspace._tables is None:
        raise ValueError("joint_pairs requires pair tables")
    return workspace.sim.decode_batch(
        workspace.case["llr"],
        tau=config["tau"],
        xi=config["xi"],
        lambd=config["lambd"],
        delta=config["delta"],
        dt=config["dt"],
        steps=STEPS,
        beta=config.get("beta", 0.0),
        parity_scale=config.get("parity_scale", 1.0),
        joint_pairs=joint,
        pair_gamma=float(config.get("pair_gamma", 0.0)),
        pair_scale=float(config.get("pair_scale", 1.0)),
        pair_tables=workspace.tables if joint else None,
        redundant_supports=config.get("redundant_supports"),
        redundant_gamma=float(config.get("redundant_gamma", 0.0)),
        tap2=float(config.get("tap2", 0.0)),
        debug_finite=debug_finite,
    )


def decode_kspin(
    workspace: CaseWorkspace,
    config: dict,
    debug_finite: bool = False,
    batch_size: int | None = None,
) -> np.ndarray:
    """Decode one K-spin configuration and return full N-bit codewords."""
    joint = int(config.get("joint_pairs", 0))
    llr = workspace.case["llr"]
    kwargs = dict(
        tau=config["tau"],
        xi=config["xi"],
        lambd=config.get("lambd", 1.0),
        delta=config["delta"],
        dt=config["dt"],
        steps=STEPS,
        beta=config.get("beta", 0.0),
        joint_pairs=joint,
        pair_gamma=float(config.get("pair_gamma", 0.0)),
        pair_tables=workspace.tables if joint else None,
        mode="kspin",
        g=float(config.get("g", 1.0)),
        clip=float(config.get("clip", 1.0e30)),
        select_unclipped=int(config.get("select_unclipped", 0)),
        debug_finite=debug_finite,
    )
    if batch_size is None or int(batch_size) >= len(llr):
        return workspace.sim.decode_batch(llr, **kwargs)
    size = int(batch_size)
    if size <= 0:
        raise ValueError("batch_size must be positive")
    chunks = []
    for start in range(0, len(llr), size):
        chunk_kwargs = dict(kwargs)
        if joint:
            chunk_kwargs["pair_tables"] = tuple(
                table[start:start + size] for table in workspace.tables
            )
        chunks.append(workspace.sim.decode_batch(llr[start:start + size], **chunk_kwargs))
    return np.concatenate(chunks, axis=0)


def decode_hard_schedules(workspace: CaseWorkspace, config: dict, debug_finite: bool = False) -> np.ndarray:
    """Decode one configuration with per-step schedules for the five tunable groups."""
    if str(config.get("mode", "bsb")) == "kspin":
        return decode_kspin(workspace, config, debug_finite)
    joint = int(config.get("joint_pairs", 0))
    return workspace.sim.decode_batch(
        workspace.case["llr"],
        tau=config["tau"],
        xi=config["xi"],
        lambd=config["lambd"],
        delta=config["delta"],
        dt=config["dt"],
        steps=STEPS,
        beta=config.get("beta", 0.0),
        parity_scale=config.get("parity_scale", 1.0),
        joint_pairs=joint,
        pair_gamma=float(config.get("pair_gamma", 0.0)),
        pair_scale=float(config.get("pair_scale", 1.0)),
        pair_tables=workspace.tables if joint else None,
        redundant_supports=config.get("redundant_supports"),
        redundant_gamma=float(config.get("redundant_gamma", 0.0)),
        tap2=float(config.get("tap2", 0.0)),
        debug_finite=debug_finite,
    )


def pair_channel(tables: tuple[np.ndarray, np.ndarray, np.ndarray], frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the C++-equivalent linear pair channel and per-bit d array."""
    a, b, d = tables
    channel = np.empty((frames, a.shape[1] * 2), dtype=np.float64)
    channel[:, 0::2] = 2.0 * a
    channel[:, 1::2] = 2.0 * b
    pair_d = np.empty_like(channel, dtype=np.float64)
    pair_d[:, 0::2] = d
    pair_d[:, 1::2] = d
    return channel, pair_d


def check_penalty(workspace: CaseWorkspace, spins: np.ndarray) -> np.ndarray:
    """Sum of 1-product(check row) for all frames, matching the C++ energy."""
    cols = workspace.sim.cols
    degrees = np.asarray(workspace.case["h"].sum(axis=1), dtype=np.int64)
    mask = np.arange(cols.shape[1])[None, :] < degrees[:, None]
    gathered = spins[:, cols]
    products = np.prod(np.where(mask[None, :, :], gathered, 1.0), axis=2)
    return np.sum(1.0 - products, axis=1)


def trajectory_energies(
    workspace: CaseWorkspace,
    hard: np.ndarray,
    config: dict,
    energy_lambda: float | None = None,
) -> np.ndarray:
    """Exact C++ energy-combine score for one trajectory's hard decisions."""
    if str(config.get("mode", "bsb")) == "kspin":
        joint = int(config.get("joint_pairs", 0))
        return workspace.sim.compute_kspin_energies(
            hard,
            workspace.case["llr"],
            float(workspace.snr),
            joint_pairs=joint,
            pair_tables=workspace.tables if joint else None,
            g=float(config.get("g", 1.0)),
            clip=float(config.get("clip", 1.0e30)),
            select_unclipped=int(config.get("select_unclipped", 0)),
        )
    spins = 1.0 - 2.0 * np.asarray(hard, dtype=np.float64)
    joint = int(config.get("joint_pairs", 0))
    if joint:
        channel, pair_d = pair_channel(workspace.tables, workspace.frames)
    else:
        channel, pair_d = workspace.case["llr"], None
    value = -np.sum(channel * spins, axis=1)
    if joint and pair_d is not None:
        value -= np.sum(
            2.0 * float(config.get("pair_scale", 1.0)) * pair_d[:, 0::2]
            * spins[:, 0::2] * spins[:, 1::2],
            axis=1,
        )
    value += 0.5 * float(
        config.get("energy_lambda", config["lambd"]) if energy_lambda is None else energy_lambda
    ) * check_penalty(workspace, spins)
    return value


def portfolio_from_hards(
    workspace: CaseWorkspace,
    hards: Sequence[np.ndarray],
    configs: Sequence[dict],
    energy_lambda: float | None = None,
) -> dict:
    """Energy-select already-decoded trajectories; return selected/oracle errors."""
    if not 1 <= len(configs) <= 6 or len(hards) != len(configs):
        raise ValueError(f"portfolio must contain 1..6 trajectories, got {len(configs)}")
    per_trajectory = np.stack([per_frame_errors(hard, workspace.case) for hard in hards])
    energies = np.stack([
        trajectory_energies(workspace, hard, config, energy_lambda)
        for hard, config in zip(hards, configs)
    ], axis=1)
    selected_index = np.argmin(energies, axis=1)
    selected = np.take_along_axis(per_trajectory, selected_index[None, :], axis=0)[0]
    oracle_index = np.argmin(per_trajectory, axis=0)
    accuracy = float(np.mean(selected_index == oracle_index))
    return {
        "selected_errors": selected,
        "oracle_errors": per_trajectory[oracle_index, np.arange(len(selected_index))],
        "per_trajectory_errors": per_trajectory,
        "selected_index": selected_index,
        "oracle_index": oracle_index,
        "selection_accuracy": accuracy,
        "selected_score": score_from_errors(selected, workspace.k),
        "oracle_score": score_from_errors(
            per_trajectory[oracle_index, np.arange(len(selected_index))], workspace.k
        ),
    }


def portfolio_errors(
    workspace: CaseWorkspace,
    configs: Sequence[dict],
    energy_lambda: float | None = None,
) -> dict:
    """Decode and energy-select a portfolio; return selected and oracle errors."""
    hards = [decode_hard(workspace, config) for config in configs]
    return portfolio_from_hards(workspace, hards, configs, energy_lambda)


def gain_report(
    workspace: CaseWorkspace,
    candidate_errors: np.ndarray,
    bootstrap_seed: int = 5150,
) -> dict:
    """Paired score gain versus cached v17 on identical frames."""
    low, high = paired_bootstrap(
        workspace.baseline_errors, candidate_errors, workspace.k,
        samples=1000, seed=bootstrap_seed,
    )
    candidate_score = score_from_errors(candidate_errors, workspace.k)
    return {
        "baseline_score": workspace.baseline_score,
        "candidate_score": candidate_score,
        "gain": candidate_score - workspace.baseline_score,
        "gain_ci_95": [low, high],
        "baseline_errors": int(workspace.baseline_errors.sum()),
        "candidate_errors": int(candidate_errors.sum()),
    }


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot serialise {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )
