#!/usr/bin/env python3
"""Phase 14 Track C: screen QAIA-compliant structural variants of v17 joint pairs.

Baseline is the v17 ``best_config`` for each R=1/2 case. Variants:

* C1 — sign-bit / magnitude-bit class multipliers for tau, xi and beta;
* C2 — exact two-spin Gray-pair TAP reaction through O(J**3), including the
  cross term, as the gradient of an Ising-invariant polynomial;
* C3 — one soft PAM amplitude coordinate per I/Q component mapped back to the
  pair's (sign, magnitude) soft bits in the pair channel/reaction terms;
* C4 — linear pair_scale ramp from 0 to the exact 1 over the first 10/20 steps.

Each variant is retuned on seed 2024 only over a small grid of its *new*
parameters, then scored on both 2024 and 6060 against the same v17 baseline.
Promotion requires +4,000 (N=1296) or +8,000 (N=3888) on every seed and a
positive 95% paired-frame-bootstrap CI lower bound on every seed.

No variant here is integrated into v17; this tool only writes screen artifacts.
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
from qam_pairs import pair_tables  # noqa: E402
from real_bench import build_case  # noqa: E402
from tune_tap_local import paired_bootstrap  # noqa: E402

ROOT = TOOLS_DIR.parent
SCORE_SCALE = 100_000.0
STEPS = 50
VARIANTS = ("C1", "C2", "C3", "C4")


def score_of(errors: int, total: int) -> float:
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber)


def per_frame_errors(hard: np.ndarray, case: dict) -> np.ndarray:
    return np.count_nonzero(hard[:, : case["k"]] != case["answers"][:, : case["k"]], axis=1)


def evaluate(hard: np.ndarray, case: dict) -> tuple[np.ndarray, int, float]:
    errs = per_frame_errors(hard, case)
    total = int(errs.sum())
    return errs, total, score_of(total, case["b"] * case["k"])


def decode(case: dict, sim: BatchedBSB, cfg: dict, tables: tuple, **extra) -> np.ndarray:
    params = dict(
        tau=cfg["tau"], xi=cfg["xi"], lambd=cfg["lambd"], delta=cfg["delta"],
        dt=cfg["dt"], steps=STEPS, beta=cfg["beta"],
        parity_scale=cfg["parity_scale"], joint_pairs=int(cfg["joint_pairs"]),
        pair_gamma=cfg["pair_gamma"], pair_scale=cfg["pair_scale"],
        pair_tables=tables,
    )
    params.update(extra)
    return sim.decode_batch(case["llr"], **params)


def ramp(run_in: int) -> np.ndarray:
    if not 1 <= run_in < STEPS:
        raise ValueError(f"ramp must be in 1..{STEPS - 1}, got {run_in}")
    return np.concatenate((np.linspace(0.0, 1.0, run_in), np.ones(STEPS - run_in)))


def candidates(variant: str, cfg: dict) -> list[tuple[str, dict, dict]]:
    """Return (name, decode extra kwargs, artifact-visible parameters) triples."""
    base_gamma = float(cfg["pair_gamma"])
    out: list[tuple[str, dict, dict]] = []
    if variant == "C1":
        moves: dict[str, tuple[float, float]] = {
            "tau_sign_0.90": (0.90, 1.0), "tau_sign_1.10": (1.10, 1.0),
            "tau_mag_0.90": (1.0, 0.90), "tau_mag_1.10": (1.0, 1.10),
            "xi_sign_0.90": (0.90, 1.0), "xi_sign_1.10": (1.10, 1.0),
            "xi_mag_0.90": (1.0, 0.90), "xi_mag_1.10": (1.0, 1.10),
            "beta_sign_0.90": (0.90, 1.0), "beta_sign_1.10": (1.10, 1.0),
            "beta_mag_0.90": (1.0, 0.90), "beta_mag_1.10": (1.0, 1.10),
        }
        for name, mult in moves.items():
            key, direction = name.split("_", 1)
            kwargs = {f"{key}_classes": mult}
            out.append((name, kwargs, {f"{key}_classes": list(mult), "move": direction}))
        combined = (
            ("combined_sign_up", (1.05, 0.95), (0.95, 1.05), (1.05, 0.95)),
            ("combined_mag_up", (0.95, 1.05), (1.05, 0.95), (0.95, 1.05)),
        )
        for name, tau, xi, beta in combined:
            kwargs = {
                "tau_classes": tau, "xi_classes": xi, "beta_classes": beta,
            }
            out.append((name, kwargs, {key: list(val) for key, val in kwargs.items()}))
    elif variant == "C2":
        for scale in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
            gamma = base_gamma * scale
            out.append((
                f"exact_pair_gamma_x{scale:.1f}",
                {"pair_exact": 1, "pair_gamma": gamma},
                {"pair_exact": 1, "pair_gamma": gamma, "gamma_scale": scale},
            ))
    elif variant == "C3":
        for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
            name = f"amp_scale_{scale:g}"
            out.append((
                name,
                {"pair_amplitude": 1, "amp_scale": scale, "amp_steps": 8},
                {"pair_amplitude": 1, "amp_scale": scale, "amp_steps": 8},
            ))
    elif variant == "C4":
        for run_in in (10, 20):
            schedule = ramp(run_in)
            out.append((
                f"pair_ramp_{run_in}",
                {"pair_scale": schedule},
                {"pair_scale_ramp_steps": run_in, "final_pair_scale": 1.0},
            ))
    else:
        raise ValueError(f"unknown variant {variant}")
    return out


def screen_one(
    variant: str,
    n: int,
    cfg: dict,
    snr: float,
    noise_scale: float,
    frames: int,
    seeds: list[int],
) -> dict:
    k = int(round(n * 0.5))
    bar = 8000.0 if n >= 3888 else 4000.0
    cases = {seed: build_case(n, k, frames, snr, seed, noise_scale=noise_scale) for seed in seeds}
    sims = {seed: BatchedBSB(case["h"]) for seed, case in cases.items()}
    tables = {seed: pair_tables(case["llr"], snr) for seed, case in cases.items()}
    baseline = {
        seed: evaluate(decode(cases[seed], sims[seed], cfg, tables[seed]), cases[seed])
        for seed in seeds
    }
    print(
        f"{variant} N={n}: baseline " + " ".join(
            f"s{seed}={baseline[seed][2]:,.1f}" for seed in seeds
        ) + f" | bar {bar:+,.0f}",
        flush=True,
    )

    train_seed = seeds[0]
    rng = np.random.default_rng(5150 + n + VARIANTS.index(variant))
    retune = []
    for name, kwargs, params in candidates(variant, cfg):
        hard = decode(cases[train_seed], sims[train_seed], cfg, tables[train_seed], **kwargs)
        errs, total, score = evaluate(hard, cases[train_seed])
        low, high = paired_bootstrap(baseline[train_seed][0], errs, k, 1000, rng)
        gain = score - baseline[train_seed][2]
        retune.append({
            "name": name, "params": params, "score": score, "gain": gain,
            "errors": total, "gain_ci_95": [low, high],
        })
        print(f"  {name:26s} s{train_seed}: {gain:+,.1f} CI[{low:+,.1f},{high:+,.1f}]", flush=True)

    chosen = max(retune, key=lambda item: item["gain"])
    chosen_kwargs = next(kwargs for name, kwargs, _ in candidates(variant, cfg) if name == chosen["name"])
    per_seed = {}
    for seed in seeds:
        hard = decode(cases[seed], sims[seed], cfg, tables[seed], **chosen_kwargs)
        errs, total, score = evaluate(hard, cases[seed])
        low, high = paired_bootstrap(baseline[seed][0], errs, k, 1000, rng)
        gain = score - baseline[seed][2]
        per_seed[str(seed)] = {
            "baseline_score": baseline[seed][2],
            "variant_score": score,
            "gain": gain,
            "gain_ci_95": [low, high],
            "baseline_errors": int(baseline[seed][0].sum()),
            "variant_errors": total,
        }
    promoted = all(
        item["gain"] >= bar and item["gain_ci_95"][0] > 0.0
        for item in per_seed.values()
    )
    return {
        "variant": variant,
        "n": n,
        "k": k,
        "rate": 0.5,
        "snr": snr,
        "frames": frames,
        "seeds": seeds,
        "noise_scale": noise_scale,
        "baseline_config": cfg,
        "promotion_bar": bar,
        "retune_seed": train_seed,
        "retune": retune,
        "chosen": {"name": chosen["name"], "params": chosen["params"]},
        "baseline_scores": {str(seed): baseline[seed][2] for seed in seeds},
        "seeds_result": per_seed,
        "min_gain": min(item["gain"] for item in per_seed.values()),
        "min_ci_lower": min(item["gain_ci_95"][0] for item in per_seed.values()),
        "promoted": promoted,
        "exporter_change_needed_if_promoted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1296, 3888])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 6060])
    parser.add_argument("--variant", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "phase14")
    args = parser.parse_args()

    for n in args.n:
        path = ROOT / "artifacts" / "phase14" / f"joint_N{n}_R05.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        cfg = dict(artifact["best_config"])
        snr = float(artifact["snr"])
        noise_scale = 1.00112
        for variant in args.variant:
            result = screen_one(variant, n, cfg, snr, noise_scale, args.frames, args.seeds)
            out = args.out_dir / f"{variant.lower()}_screen_N{n}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(
                f"  CHOSEN {result['chosen']['name']}: min gain "
                f"{result['min_gain']:+,.1f}, CI low {result['min_ci_lower']:+,.1f}, "
                f"promoted={result['promoted']} -> {out}",
                flush=True,
            )


if __name__ == "__main__":
    main()
