#!/usr/bin/env python3
"""Gradient-free coordinate tuning for v3 QAIA trajectory ensembles."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from bsb_vectorized import BatchedBSB  # noqa: E402
from decode_v3_python import decode_frame  # noqa: E402
from real_bench import build_case  # noqa: E402

KEYS = ("tau", "xi", "lambd", "delta")
SCORE_SCALE = 100000.0


def case_config(mode: str, combine: str, trajectories: list[dict]) -> dict:
    return {
        "n": 0,
        "k": 0,
        "snr": 0.0,
        "mode": mode,
        "combine": combine,
        "trajectories": trajectories,
        "dt": 0.2,
        "steps": 50,
    }


def score_case(case: dict, config: dict) -> tuple[float, float]:
    config = dict(config)
    config["n"] = case["n"]
    config["k"] = case["k"]
    config["snr"] = case.get("snr", case.get("snr_db"))

    if config.get("mode") == "bsb":
        if "_sim" not in case:
            case["_sim"] = BatchedBSB(case["h"])
        sim: BatchedBSB = case["_sim"]
        score, ber, _ = sim.evaluate_ensemble(
            case["llr"],
            case["answers"],
            case["k"],
            config["trajectories"],
            combine=config.get("combine", "vote"),
            dt=config.get("dt", 0.2),
            steps=config.get("steps", 50),
        )
        return score, ber

    n = case["n"]
    k = case["k"]
    errors = 0
    total = 0
    for row, llr in enumerate(case["llr"]):
        hard = decode_frame(llr, case["h"], config)
        errors += int(np.count_nonzero(np.asarray(hard[:k], dtype=np.int8) != case["answers"][row, :k]))
        total += k
    ber = max(errors / total, 1.0 / total)
    return -SCORE_SCALE * math.log10(ber), ber


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--val-seed", type=int, default=2024)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--trajectories", type=int, default=6)
    parser.add_argument("--mode", choices=("dsb", "bsb"), default="bsb")
    parser.add_argument("--combine", choices=("energy", "vote", "soft_vote"), default="vote")
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    k = int(round(args.n * args.rate))
    if args.trajectories > 6:
        raise SystemExit("at most 6 trajectories are allowed")

    if args.configs is not None:
        trajectories = json.loads(args.configs.read_text(encoding="utf-8"))
        if isinstance(trajectories, dict):
            trajectories = trajectories["trajectories"]
        if len(trajectories) != args.trajectories:
            raise SystemExit("--configs length must match --trajectories")
    else:
        base = {"tau": 2.0, "xi": 0.5, "lambd": 6.0, "delta": 2.5}
        if args.mode == "bsb":
            base = {"tau": 2.0, "xi": 0.4, "lambd": 4.0, "delta": 2.0}
        trajectories = []
        for _ in range(args.trajectories):
            traj = dict(base)
            for key in KEYS:
                traj[key] *= float(math.exp(rng.normal(0.0, 0.15)))
            if args.combine == "energy":
                traj["energy_lambda"] = traj["lambd"]
            trajectories.append(traj)

    train = build_case(args.n, k, args.samples, args.snr, args.seed)
    config = case_config(args.mode, args.combine, trajectories)
    best_score, best_ber = score_case(train, config)
    print(f"initial train score={best_score:.0f} ber={best_ber:.6f}")

    for trial in range(args.trials):
        candidate = copy.deepcopy(trajectories)
        index = int(rng.integers(0, len(candidate)))
        key = str(rng.choice(KEYS))
        candidate[index][key] = float(
            candidate[index][key] * math.exp(rng.normal(0.0, args.sigma))
        )
        if args.combine == "energy":
            candidate[index]["energy_lambda"] = candidate[index]["lambd"]
        trial_config = case_config(args.mode, args.combine, candidate)
        score, ber = score_case(train, trial_config)
        if score > best_score:
            trajectories = candidate
            best_score = score
            best_ber = ber
        if (trial + 1) % 20 == 0:
            print(f"{trial + 1}/{args.trials} best train score={best_score:.0f} ber={best_ber:.6f}")

    val = build_case(args.n, k, args.val_samples, args.snr, args.val_seed)
    val_config = case_config(args.mode, args.combine, trajectories)
    val_score, val_ber = score_case(val, val_config)

    result = {
        "n": args.n,
        "k": k,
        "rate": args.rate,
        "snr": args.snr,
        "mode": args.mode,
        "combine": args.combine,
        "train_seed": args.seed,
        "validation_seed": args.val_seed,
        "train_samples": args.samples,
        "validation_samples": args.val_samples,
        "trials": args.trials,
        "train_score": best_score,
        "train_ber": best_ber,
        "validation_score": val_score,
        "validation_ber": val_ber,
        "trajectories": trajectories,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"validation score={val_score:.0f} ber={val_ber:.6f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
