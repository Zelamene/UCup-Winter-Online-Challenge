#!/usr/bin/env python3
"""Tune the N=1944 R=3/4 case directly on the public input and answers.

The first half of the public frames is used for selection and the second half
for validation. Ground truth is used only offline for scoring candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from decode_v2_python import decode_frame  # noqa: E402
from phase0_harness import parse_answers, parse_instance  # noqa: E402
from phase3_fixed_dsb import CheckTable, DSBConfig  # noqa: E402
from phase4_unfolded_dsb import PARAMETER_KEYS, RawParams, build_features  # noqa: E402
from phase5_generalization import load_params  # noqa: E402
from phase7_sparse_dsb import (  # noqa: E402
    check_products,
    decode_fixed_sparse,
    decode_hard_sparse,
    sparse_access,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase7_case_manifest_v7.json"
DEFAULT_BASE_OUTPUT = ROOT / "artifacts" / "phase7_v7_public.out"


def parse_list(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def select_manifest_case(manifest: dict, n: int, rate: float, snr: float) -> dict:
    for case in manifest["cases"]:
        if (
            int(case["n"]) == n
            and abs(int(case["k"]) / int(case["n"]) - rate) < 1e-9
            and abs(float(case["snr"]) - snr) < 0.25
        ):
            return case
    raise ValueError(f"No manifest case for n={n}, rate={rate}, snr={snr}")


def decode_learned_batch(
    llr: np.ndarray,
    h: np.ndarray,
    checks: CheckTable,
    params_path: Path,
) -> np.ndarray:
    payload = load_params(params_path)
    params = RawParams({
        key: np.asarray(payload.values[key], dtype=np.float64) for key in PARAMETER_KEYS
    })
    features = build_features(llr, checks)
    return decode_hard_sparse(params, h.astype(np.float64), llr, checks, features)


def decode_fixed_batch(
    llr: np.ndarray,
    h: np.ndarray,
    config: tuple[float, float, float, float],
) -> np.ndarray:
    tau, xi, lambd, delta = config
    cfg = DSBConfig(
        tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50,
        gradient_sign=1,
    )
    return decode_fixed_sparse(cfg, h, llr)


class Objective:
    def __init__(self, llr: np.ndarray, answers: np.ndarray, h: np.ndarray, k: int):
        self.llr = llr
        self.answers = answers
        self.k = int(k)
        self.n = h.shape[1]
        self.checks = CheckTable(h)
        self.safe, self.mask = sparse_access(self.checks)
        self.half = llr.shape[0] // 2

    def ber(self, hards: np.ndarray, part: str = "all") -> float:
        if part == "train":
            expected = self.half
            answers = self.answers[: self.half]
            if hards.shape[0] > expected:
                hards = hards[: self.half]
        elif part == "val":
            expected = self.llr.shape[0] - self.half
            answers = self.answers[self.half :]
            if hards.shape[0] > expected:
                hards = hards[self.half :]
        else:
            answers = self.answers
        errors = hards[:, : self.k] != answers[:, : self.k]
        return float(errors.sum() / errors.size)

    def energies(self, hards: np.ndarray, lambd: float, part: str) -> np.ndarray:
        if part == "train":
            llr = self.llr[: self.half]
            hards = hards[: self.half]
        elif part == "val":
            llr = self.llr[self.half :]
            hards = hards[self.half :]
        else:
            llr = self.llr
        spins = np.where(hards == 0, 1.0, -1.0)
        products = check_products(spins, self.safe, self.mask)
        return (
            -np.sum(llr * spins, axis=1)
            + 0.5 * lambd * np.sum(1.0 - products, axis=1)
        )

    def select(self, candidates: list[dict], indices: list[int], part: str) -> np.ndarray:
        sliced = [candidates[index]["hards"] for index in indices]
        stacked = np.stack(sliced)
        scores = np.stack([
            self.energies(candidates[index]["hards"],
                          candidates[index].get("energy_lambda", candidates[index]["lambd"]),
                          part)
            for index in indices
        ])
        pick = np.argmin(scores, axis=0)
        count = scores.shape[1]
        offset = self.half if part == "val" else 0
        frame_index = np.arange(offset, offset + count)
        return stacked[pick, frame_index][:, : self.n]


def local_search(
    objective: Objective,
    llr: np.ndarray,
    h: np.ndarray,
    best: tuple[float, float, float, float],
    rounds: int,
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    config = best
    hards = decode_fixed_batch(llr, h, config)
    current = objective.ber(hards, "train")
    deltas = np.array([0.2, 0.05, 1.0, 0.25])
    for round_index in range(rounds):
        improved = False
        for axis in range(4):
            for direction in (-1.0, 1.0):
                trial = list(config)
                trial[axis] = max(0.05, trial[axis] + direction * deltas[axis])
                trial_hards = decode_fixed_batch(llr, h, tuple(trial))
                trial_ber = objective.ber(trial_hards, "train")
                if trial_ber < current:
                    config = tuple(trial)
                    hards = trial_hards
                    current = trial_ber
                    improved = True
        print(f"Local round {round_index + 1}: train BER {current:.9f} config={config}")
        deltas *= 0.5
        if not improved:
            break
    return config, hards


def greedy_portfolio(
    objective: Objective,
    candidates: list[dict],
    limit: int = 6,
    seed: list[int] | None = None,
) -> list[int]:
    bers = [objective.ber(candidate["hards"], "train") for candidate in candidates]
    if seed is not None:
        selected = list(seed)
    else:
        selected = [int(np.argmin(bers))]
    current = bers[selected[0]]
    while len(selected) < limit:
        best_index = None
        for index in range(len(candidates)):
            if index in selected:
                continue
            trial = selected + [index]
            value = objective.ber(objective.select(candidates, trial, "train"), "train")
            if value < current:
                current = value
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
    return selected


def tune_energy(
    objective: Objective,
    candidates: list[dict],
    indices: list[int],
    values: tuple[float, ...],
    rounds: int,
) -> float:
    for round_index in range(rounds):
        improved = False
        for index in indices:
            original = candidates[index].get("energy_lambda", candidates[index]["lambd"])
            best_value = original
            best_ber = objective.ber(objective.select(candidates, indices, "train"), "train")
            for value in values:
                if value == original:
                    continue
                candidates[index]["energy_lambda"] = value
                ber = objective.ber(objective.select(candidates, indices, "train"), "train")
                if ber < best_ber:
                    best_ber = ber
                    best_value = value
            candidates[index]["energy_lambda"] = best_value
            if best_value != original:
                improved = True
        if not improved:
            break
    return objective.ber(objective.select(candidates, indices, "train"), "train")


def name_of(config: tuple[float, float, float, float], rank: int | None = None) -> str:
    tau, xi, lambd, delta = config
    prefix = "grid" if rank is None else f"grid{rank}"
    return f"{prefix}_{tau:g}_{xi:g}_{lambd:g}_{delta:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-output", type=Path, default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--tag", default="public")
    parser.add_argument("--local-rounds", type=int, default=3)
    parser.add_argument(
        "--grid-tau", default="1.6,2,2.5,4,6,8"
    )
    parser.add_argument("--grid-xi", default="0.4,0.5")
    parser.add_argument("--grid-lam", default="5,6,7,8")
    parser.add_argument("--grid-delta", default="1.5,2,2.5")
    parser.add_argument("--top-grid", type=int, default=8)
    parser.add_argument(
        "--energy-lambdas", default="1,2,3,4,5,6,8,10,12,16"
    )
    args = parser.parse_args()

    args.input = args.input.resolve()
    args.answer = args.answer.resolve()
    args.manifest = args.manifest.resolve()
    args.base_output = args.base_output.resolve()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"),
        str(args.answer),
        instance.n,
        instance.b,
    )
    llr = np.asarray(instance.llr, dtype=np.float64)
    h = np.asarray(instance.h)
    objective = Objective(llr, answers, h, instance.k)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = select_manifest_case(manifest, instance.n, 0.75, instance.snr)

    params_path = Path(case["params"])
    if not params_path.is_absolute():
        params_path = args.manifest.resolve().parent.parent / params_path
    candidates: list[dict] = []
    learned_hards = decode_learned_batch(llr, h, objective.checks, params_path)
    candidates.append({
        "name": "per_case_learned",
        "lambd": float(case["learned_lambda"]),
        "energy_lambda": float(case["learned_lambda"]),
        "hards": learned_hards,
    })
    for values in case["fixed_candidates"]:
        config = tuple(float(value) for value in values[:4])
        tau, xi, lambd, delta = config
        candidates.append({
            "name": f"fixed_{tau:g}_{xi:g}_{lambd:g}_{delta:g}",
            "lambd": config[2],
            "energy_lambda": float(values[4]) if len(values) >= 5 else config[2],
            "hards": decode_fixed_batch(llr, h, config),
        })

    initial_names = [candidate["name"] for candidate in candidates]
    initial_indices = list(range(len(candidates)))
    print(f"Initial train BER: {objective.ber(objective.select(candidates, initial_indices, 'train'), 'train'):.9f}")
    print(f"Initial val BER: {objective.ber(objective.select(candidates, initial_indices, 'val'), 'val'):.9f}")

    grid_results = []
    for tau in parse_list(args.grid_tau):
        for xi in parse_list(args.grid_xi):
            for lambd in parse_list(args.grid_lam):
                for delta in parse_list(args.grid_delta):
                    config = (tau, xi, lambd, delta)
                    hards = decode_fixed_batch(llr, h, config)
                    train_ber = objective.ber(hards, "train")
                    val_ber = objective.ber(hards, "val")
                    grid_results.append({
                        "config": config,
                        "train_ber": train_ber,
                        "val_ber": val_ber,
                        "hards": hards,
                    })
    grid_results.sort(key=lambda item: item["train_ber"])
    print("Top fixed configurations by train BER:")
    for item in grid_results[: args.top_grid]:
        print(
            f"  train={item['train_ber']:.9f} val={item['val_ber']:.9f} "
            f"config={item['config']}"
        )

    best_config, best_hards = local_search(
        objective, llr, h, grid_results[0]["config"], args.local_rounds
    )
    local_result = {
        "config": best_config,
        "train_ber": objective.ber(best_hards, "train"),
        "val_ber": objective.ber(best_hards, "val"),
        "hards": best_hards,
    }
    print(f"Best local config: {local_result['config']}")
    print(f"  train={local_result['train_ber']:.9f} val={local_result['val_ber']:.9f}")

    ranked = [item for item in grid_results if item["config"] != best_config]
    ranked.insert(0, local_result)
    for rank, item in enumerate(ranked):
        candidates.append({
            "name": name_of(item["config"], rank),
            "lambd": item["config"][2],
            "energy_lambda": item["config"][2],
            "hards": item["hards"],
        })

    indices = greedy_portfolio(objective, candidates, limit=6, seed=initial_indices)
    train_ber = objective.ber(objective.select(candidates, indices, "train"), "train")
    val_ber = objective.ber(objective.select(candidates, indices, "val"), "val")
    print(f"Greedy portfolio names: {[candidates[i]['name'] for i in indices]}")
    print(f"  train={train_ber:.9f} val={val_ber:.9f}")

    tune_energy(
        objective, candidates, indices, parse_list(args.energy_lambdas), rounds=2
    )
    train_ber = objective.ber(objective.select(candidates, indices, "train"), "train")
    val_ber = objective.ber(objective.select(candidates, indices, "val"), "val")
    all_hards = objective.select(candidates, indices, "all")
    all_ber = objective.ber(all_hards, "all")
    print(f"After energy tuning: train={train_ber:.9f} val={val_ber:.9f} all={all_ber:.9f}")

    baseline = np.array([
        [int(value) for value in line.split()]
        for line in args.base_output.read_text(encoding="utf-8").splitlines()
    ])
    baseline_ber = objective.ber(baseline, "all")
    print(f"Baseline all BER: {baseline_ber:.9f}")

    energy_lambdas = {
        candidates[index]["name"]: float(
            candidates[index].get("energy_lambda", candidates[index]["lambd"])
        )
        for index in indices
    }
    summary = {
        "case": {"n": instance.n, "rate": 0.75, "snr": 10.5},
        "data": {
            "input": str(args.input.relative_to(ROOT)),
            "train_frames": objective.half,
            "val_frames": llr.shape[0] - objective.half,
            "grid_configs": len(grid_results),
            "base_initial_names": initial_names,
        },
        "best_fixed_config": list(best_config),
        "local_result": {
            "config": list(best_config),
            "train_ber": local_result["train_ber"],
            "val_ber": local_result["val_ber"],
        },
        "accepted_portfolio_ber": baseline_ber,
        "optimized_portfolio": {
            "names": [candidates[index]["name"] for index in indices],
            "ber": all_ber,
            "train_ber": train_ber,
            "val_ber": val_ber,
            "energy_lambdas": energy_lambdas,
        },
        "gate": {
            "val_beats_baseline": val_ber < objective.ber(baseline, "val"),
            "all_beats_baseline": all_ber < baseline_ber,
            "guess_count_at_most_six": len(indices) <= 6,
        },
    }
    summary_path = args.outdir / f"phase7_N1944_R3_4_SNR10p5_summary_{args.tag}.json"
    params_out = args.outdir / f"phase7_N1944_R3_4_SNR10p5_params_{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    params_out.write_text(params_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {params_out}")


if __name__ == "__main__":
    main()
