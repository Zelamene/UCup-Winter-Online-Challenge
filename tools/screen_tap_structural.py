#!/usr/bin/env python3
"""Small 100-frame screen for QAIA-compliant structural tweaks to tuned TAP-bSB.

The baseline is a Stage A' flat TAP/bSB configuration. Three safe variants are
screened against it on identical frames:

* ``beta_ramp`` — a per-step beta schedule, still the same added-energy gradient;
* ``portfolio6`` — six TAP trajectories with energy selection, still QAIA only;
* ``dsb_finish`` — switch the last five or ten steps to hard dSB check products.

Promotion bars are ``+4,000`` at N=1296 and ``+8,000`` at N=3888, on every
screen seed with a positive paired-bootstrap CI lower bound.
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

from bsb_vectorized import BatchedBSB  # noqa: E402
from real_bench import build_case  # noqa: E402
from tune_tap_local import paired_bootstrap  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
STEPS = 50
TAP_KEYS = ("tau", "xi", "lambd", "delta", "dt", "beta", "parity_scale")


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def per_frame_errors(hard: np.ndarray, answers: np.ndarray, k: int) -> np.ndarray:
    return np.count_nonzero(hard[:, :k] != answers[:, :k], axis=1)


def beta_ramp(beta: float) -> np.ndarray:
    """Mild early emphasis: 1.15x at step 0 decaying to 0.85x at the end."""
    t = np.linspace(0.0, 1.0, STEPS)
    return beta * (1.15 - 0.30 * t)


def decode_flat(case: dict, sim: BatchedBSB, cfg: dict, **kwargs) -> np.ndarray:
    return sim.decode_batch(
        case["llr"],
        tau=cfg["tau"], xi=cfg["xi"], lambd=cfg["lambd"],
        delta=cfg["delta"], dt=cfg["dt"], steps=STEPS,
        beta=cfg["beta"], parity_scale=cfg["parity_scale"],
        **kwargs,
    )


def decode_portfolio(case: dict, sim: BatchedBSB, cfg: dict) -> np.ndarray:
    """Six TAP trajectories with the existing v14-style energy selection."""
    changes = (
        {},
        {"beta": 0.8},
        {"beta": 1.2},
        {"parity_scale": 0.85},
        {"parity_scale": 1.15},
        {"delta": 0.9},
    )
    candidates = []
    energies = []
    for scale in changes:
        traj = dict(cfg)
        for key, factor in scale.items():
            traj[key] *= factor
        hard = decode_flat(case, sim, traj)
        candidates.append(hard)
        energies.append(sim.compute_energy(hard, case["llr"], 8.0))
    chosen = np.argmin(np.stack(energies), axis=0)
    return np.stack(candidates)[chosen, np.arange(case["b"])]


def variants_for(cfg: dict) -> dict[str, dict]:
    variants: dict[str, dict] = {}
    ramp_cfg = dict(cfg)
    ramp_cfg["beta"] = beta_ramp(float(cfg["beta"]))
    variants["beta_ramp"] = {"config": ramp_cfg, "kwargs": {}}
    variants["portfolio6"] = {"config": cfg, "kwargs": {}, "portfolio": True}
    for finish in (5, 10):
        variants[f"dsb_finish_{finish}"] = {
            "config": cfg, "kwargs": {"dsb_finish": finish}
        }
    return variants


def evaluate_hard(hard: np.ndarray, case: dict) -> tuple[np.ndarray, int, float]:
    errs = per_frame_errors(hard, case["answers"], case["k"])
    total = int(errs.sum())
    return errs, total, score_of(total, case["b"] * case["k"])


def screen(
    stage_a_path: Path,
    frames: int,
    seeds: list[int],
    output: Path,
) -> dict:
    artifact = json.loads(stage_a_path.read_text(encoding="utf-8"))
    cfg = dict(artifact.get("best_config", artifact))
    missing = [key for key in TAP_KEYS if key not in cfg]
    if missing:
        raise SystemExit(f"{stage_a_path}: missing TAP fields {missing}")
    n, k = int(artifact["n"]), int(artifact["k"])
    snr = float(artifact["snr"])
    noise_scale = float(artifact.get("noise_scale", 1.001120))
    bar = 8000.0 if n >= 3888 else 4000.0

    cases = {
        seed: build_case(n, k, frames, snr, seed, noise_scale=noise_scale)
        for seed in seeds
    }
    sims = {seed: BatchedBSB(case["h"]) for seed, case in cases.items()}
    variants = variants_for(cfg)
    baseline = {seed: decode_flat(cases[seed], sims[seed], cfg) for seed in seeds}
    base_eval = {
        seed: evaluate_hard(baseline[seed], cases[seed]) for seed in seeds
    }

    print(f"Structural screen: N={n} K={k} R={artifact['rate']:.4f}, {frames} frames")
    print(f"promotion bar: {bar:+,.0f} on every seed")
    print(
        f"baseline scores: " + " ".join(
            f"s{seed}={base_eval[seed][2]:,.1f}" for seed in seeds
        ),
        flush=True,
    )

    rng = np.random.default_rng(5150)
    results = {}
    for name, spec in variants.items():
        per_seed = {}
        for seed in seeds:
            case, sim = cases[seed], sims[seed]
            if spec.get("portfolio"):
                hard = decode_portfolio(case, sim, spec["config"])
            else:
                hard = decode_flat(case, sim, spec["config"], **spec["kwargs"])
            errs, total, score = evaluate_hard(hard, case)
            base_errs = base_eval[seed][0]
            low, high = paired_bootstrap(base_errs, errs, k, 1000, rng)
            gain = score - base_eval[seed][2]
            per_seed[str(seed)] = {
                "baseline_score": base_eval[seed][2],
                "variant_score": score,
                "gain": gain,
                "gain_ci_95": [low, high],
                "variant_errors": total,
                "baseline_errors": int(base_errs.sum()),
            }
            print(
                f"  {name:16s} seed {seed}: {score:,.1f} "
                f"({gain:+,.1f}, 95% CI [{low:+,.1f}, {high:+,.1f}])",
                flush=True,
            )
        all_pass = all(
            item["gain"] >= bar and item["gain_ci_95"][0] > 0.0
            for item in per_seed.values()
        )
        results[name] = {
            "seeds": per_seed,
            "min_gain": min(item["gain"] for item in per_seed.values()),
            "promoted": all_pass,
        }

    summary = {
        "stage_a": str(stage_a_path),
        "n": n,
        "k": k,
        "rate": artifact["rate"],
        "snr": snr,
        "frames": frames,
        "seeds": seeds,
        "noise_scale": noise_scale,
        "promotion_bar": bar,
        "baseline_scores": {str(seed): base_eval[seed][2] for seed in seeds},
        "variants": results,
        "promoted_variants": sorted(
            name for name, item in results.items() if item["promoted"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(f"promoted: {summary['promoted_variants']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 6060])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts" / "structural_screen.json"
    )
    args = parser.parse_args()
    screen(args.stage_a, args.frames, args.seeds, args.out)


if __name__ == "__main__":
    main()
