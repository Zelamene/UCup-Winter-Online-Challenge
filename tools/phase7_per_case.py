#!/usr/bin/env python3
"""Phase 7: per-case dSB optimization for one N/R/SNR operating point.

All decoding remains QAIA/dSB only. Ground truth is used offline for grid
search, training, and portfolio selection, never by the decoder.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import parse_instance, parse_answers  # noqa: E402
from phase3_fixed_dsb import CheckTable, DSBConfig  # noqa: E402
from phase4_unfolded_dsb import (  # noqa: E402
    MOD_KEYS,
    PARAMETER_KEYS,
    Adam,
    RawParams,
    build_features,
    constant_params,
)
from phase5_generalization import empirical_margins, make_case  # noqa: E402
from phase6_multi_guess import FIXED_CANDIDATES, load_params  # noqa: E402
from phase7_sparse_dsb import (  # noqa: E402
    check_products,
    decode_fixed_sparse,
    decode_hard_sparse,
    loss_and_grad_sparse,
    sparse_access,
)

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_GLOBAL_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"
DEFAULT_OUTDIR = ROOT / "artifacts"

RATE_LABELS = {0.5: "1_2", 2.0 / 3.0: "2_3", 0.75: "3_4"}

PARAM_BOUNDS = {
    "tau": (np.log(1.0), np.log(32.0)),
    "dt": (np.log(0.01), np.log(0.5)),
    "xi": (np.log(0.01), np.log(2.0)),
    "lam": (np.log(1.0), np.log(20.0)),
    "delta": (np.log(0.1), np.log(10.0)),
}


def clone_params(params: RawParams) -> RawParams:
    return RawParams({key: value.copy() for key, value in params.values.items()})


def clip_params(params: RawParams) -> None:
    for key, value in params.values.items():
        if key in PARAM_BOUNDS:
            low, high = PARAM_BOUNDS[key]
        else:
            low, high = -1.0, 1.0
        params.values[key] = np.clip(value, low, high)


def export_params(params: RawParams, path: Path) -> None:
    values = params.exp_values()
    payload = {
        "steps": 50,
        "values": {key: values[key].tolist() for key in PARAMETER_KEYS},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_cases(
    n: int,
    rate: float,
    snr: float,
    codes: int,
    samples: int,
    margins: np.ndarray,
    public_snr: float,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(codes):
        case = make_case(n, rate, samples, snr, margins, public_snr, rng)
        checks = CheckTable(case["h"])
        case["checks"] = checks
        case["features"] = build_features(case["llr"], checks)
        cases.append(case)
    return cases


def decode_config(cases: list[dict], config: DSBConfig) -> list[np.ndarray]:
    return [decode_fixed_sparse(config, case["h"], case["llr"]) for case in cases]


def decode_learned(cases: list[dict], params: RawParams) -> list[np.ndarray]:
    return [
        decode_hard_sparse(
            params,
            case["h"].astype(np.float64),
            case["llr"],
            case["checks"],
            case["features"],
        )
        for case in cases
    ]


def ber_of(cases: list[dict], hards: list[np.ndarray]) -> float:
    errors = 0
    total = 0
    for case, hard in zip(cases, hards):
        k = case["k"]
        wrong = hard[:, :k] != case["answers"][:, :k]
        errors += int(np.count_nonzero(wrong))
        total += wrong.size
    return errors / max(total, 1)


def syndrome_ones(cases: list[dict], hards: list[np.ndarray]) -> int:
    total = 0
    for case, hard in zip(cases, hards):
        spins = np.where(hard == 0, 1.0, -1.0)
        safe, mask = sparse_access(case["checks"])
        products = check_products(spins, safe, mask)
        total += int(np.count_nonzero(products < 0))
    return total


def energy(hard: np.ndarray, llr: np.ndarray, checks: CheckTable, lambd: float) -> np.ndarray:
    spins = np.where(hard == 0, 1.0, -1.0)
    safe, mask = sparse_access(checks)
    products = check_products(spins, safe, mask)
    return -np.sum(llr * spins, axis=1) + 0.5 * lambd * np.sum(1.0 - products, axis=1)


def select_energies(
    cases: list[dict], candidates: list[dict], subset: list[int]
) -> list[np.ndarray]:
    selected = []
    for case_index, case in enumerate(cases):
        energies = np.stack([
            energy(
                candidates[index]["hards"][case_index],
                case["llr"],
                case["checks"],
                candidates[index].get("energy_lambda", candidates[index]["lambd"]),
            )
            for index in subset
        ])
        pick = np.argmin(energies, axis=0)
        stacked = np.stack([candidates[index]["hards"][case_index] for index in subset])
        batch = case["llr"].shape[0]
        n = case["n"]
        selected.append(
            stacked[pick[:, None], np.arange(batch)[:, None], np.arange(n)[None, :]]
        )
    return selected


def grid_search(
    cases: list[dict],
    taus: tuple[float, ...],
    xis: tuple[float, ...],
    lambdas: tuple[float, ...],
    deltas: tuple[float, ...],
) -> list[dict]:
    results = []
    configs = list(itertools.product(taus, xis, lambdas, deltas))
    print(f"Grid search: {len(configs)} configurations")
    start = time.perf_counter()
    for number, (tau, xi, lambd, delta) in enumerate(configs, start=1):
        config = DSBConfig(
            tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50, gradient_sign=1
        )
        hards = decode_config(cases, config)
        result = {
            "config": (tau, xi, lambd, delta),
            "lambd": lambd,
            "ber": ber_of(cases, hards),
            "syndrome_ones": syndrome_ones(cases, hards),
            "hards": hards,
        }
        results.append(result)
        if number % 10 == 0:
            print(f"  {number}/{len(configs)} elapsed={time.perf_counter() - start:.1f}s")
    results.sort(key=lambda item: (item["ber"], item["syndrome_ones"]))
    print("Top grid configurations:")
    for item in results[:8]:
        tau, xi, lambd, delta = item["config"]
        print(
            f"  ber={item['ber']:.6f} synd={item['syndrome_ones']:>6d} "
            f"tau={tau:.1f} xi={xi:.2f} lam={lambd:.1f} delta={delta:.2f}"
        )
    return results


def local_search(
    cases: list[dict],
    config: tuple[float, float, float, float],
    rounds: int,
) -> tuple[tuple[float, float, float, float], list[np.ndarray], list[dict]]:
    current = list(map(float, config))
    best_hards = decode_config(cases, DSBConfig(
        tau=current[0], xi=current[1], lambd=current[2], delta=current[3],
        dt=0.2, steps=50, gradient_sign=1,
    ))
    best_ber = ber_of(cases, best_hards)
    history = [{"round": 0, "config": list(current), "ber": best_ber}]
    factors = (0.8, 0.9, 1.0, 1.1, 1.25)

    for round_number in range(1, rounds + 1):
        improved = False
        for axis in range(4):
            for factor in factors:
                if factor == 1.0:
                    continue
                trial = current.copy()
                trial[axis] *= factor
                candidate = DSBConfig(
                    tau=trial[0], xi=trial[1], lambd=trial[2], delta=trial[3],
                    dt=0.2, steps=50, gradient_sign=1,
                )
                hards = decode_config(cases, candidate)
                value = ber_of(cases, hards)
                if value < best_ber:
                    best_ber = value
                    best_hards = hards
                    current = trial
                    improved = True
        history.append({"round": round_number, "config": list(current), "ber": best_ber})
        print(f"local search round {round_number}: ber={best_ber:.9f} config={current}")
        if not improved:
            break

    return tuple(current), best_hards, history


def train(
    params: RawParams,
    train_cases: list[dict],
    val_cases: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    syn_weight: float,
    temp: float,
    mod_scale: float,
    eval_every: int,
    seed: int,
) -> tuple[RawParams, list[dict]]:
    optimizer = Adam(params, lr)
    rng = np.random.default_rng(seed)
    best = clone_params(params)
    best_ber = ber_of(val_cases, decode_learned(val_cases, params))
    history = [{"epoch": 0, "val_ber": best_ber, "best_ber": best_ber}]
    print(f"Initial validation BER: {best_ber:.9f}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        for case in train_cases:
            order = rng.permutation(case["llr"].shape[0])
            for start in range(0, order.size, batch_size):
                idx = order[start : start + batch_size]
                llr = case["llr"][idx]
                targets = case["answers"][idx, : case["k"]].astype(np.float64)
                features = {
                    key: value[idx] for key, value in case["features"].items()
                }
                _, grads = loss_and_grad_sparse(
                    params, llr, targets, case["checks"], features, syn_weight, temp
                )
                for key in MOD_KEYS:
                    grads[key] *= mod_scale
                optimizer.step(params, grads)
                clip_params(params)

        if epoch % eval_every == 0 or epoch == epochs:
            val_hards = decode_learned(val_cases, params)
            val_ber = ber_of(val_cases, val_hards)
            if val_ber < best_ber:
                best_ber = val_ber
                best = clone_params(params)
            history.append({
                "epoch": epoch,
                "val_ber": val_ber,
                "best_ber": best_ber,
                "seconds": time.perf_counter() - epoch_start,
            })
            print(
                f"epoch {epoch:>3d} val_ber={val_ber:.9f} best={best_ber:.9f} "
                f"secs={time.perf_counter() - epoch_start:.1f}"
            )
    return best, history


def greedy_portfolio(cases: list[dict], candidates: list[dict], limit: int = 6) -> dict:
    bers = [ber_of(cases, [candidate["hards"][i] for i in range(len(cases))]) for candidate in candidates]
    first = int(np.argmin(bers))
    selected = [first]
    history = [{"added": candidates[first]["name"], "ber": bers[first], "subset": [first]}]

    while len(selected) < limit:
        best_index = None
        best_ber = ber_of(cases, select_energies(cases, candidates, selected))
        for index in range(len(candidates)):
            if index in selected:
                continue
            trial = selected + [index]
            trial_ber = ber_of(cases, select_energies(cases, candidates, trial))
            if trial_ber < best_ber:
                best_ber = trial_ber
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        history.append({
            "added": candidates[best_index]["name"],
            "ber": best_ber,
            "subset": list(selected),
        })
        print(f"  portfolio {len(selected)}: {candidates[best_index]['name']} ber={best_ber:.9f}")

    final_hards = select_energies(cases, candidates, selected)
    return {
        "indices": selected,
        "names": [candidates[index]["name"] for index in selected],
        "ber": ber_of(cases, final_hards),
        "syndrome_ones": syndrome_ones(cases, final_hards),
        "history": history,
    }


def tune_energy_scale(
    cases: list[dict],
    candidates: list[dict],
    subset: list[int],
    scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
) -> tuple[float, float, list[np.ndarray], dict[int, float]]:
    """Tune a common scale on energy penalties without changing decoding."""
    original = {
        index: float(candidates[index].get("energy_lambda", candidates[index]["lambd"]))
        for index in subset
    }
    best_scale = 1.0
    best_ber = ber_of(cases, select_energies(cases, candidates, subset))
    best_hards = select_energies(cases, candidates, subset)

    for scale in scales:
        if scale == 1.0:
            continue
        for index in subset:
            candidates[index]["energy_lambda"] = original[index] * scale
        hards = select_energies(cases, candidates, subset)
        value = ber_of(cases, hards)
        if value < best_ber:
            best_scale = scale
            best_ber = value
            best_hards = hards

    for index in subset:
        candidates[index]["energy_lambda"] = original[index] * best_scale
    print(f"Energy scale tuning: scale={best_scale:.3f} ber={best_ber:.9f}")
    return best_scale, best_ber, best_hards, {
        index: original[index] * best_scale for index in subset
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3888)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--global-params", type=Path, default=DEFAULT_GLOBAL_PARAMS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--train-codes", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-codes", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--syn-weight", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--mod-scale", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument("--top-grid", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--grid-tau", default="2,4,8")
    parser.add_argument("--grid-xi", default="0.3,0.4,0.5")
    parser.add_argument("--grid-lam", default="4,6,8")
    parser.add_argument("--grid-delta", default="1.5,2.5")
    parser.add_argument("--local-rounds", type=int, default=0)
    args = parser.parse_args()

    if args.rate not in RATE_LABELS:
        raise ValueError(f"Unsupported rate {args.rate}")

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"),
        str(args.answer),
        instance.n,
        instance.b,
    )
    margins = empirical_margins(instance, answers)
    label = f"N{args.n}_R{RATE_LABELS[args.rate]}_SNR{args.snr:g}".replace(".", "p")
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Target case: N={args.n} R={args.rate} SNR={args.snr}")
    train_cases = make_cases(
        args.n, args.rate, args.snr, args.train_codes, args.train_samples,
        margins, instance.snr, args.seed
    )
    val_cases = make_cases(
        args.n, args.rate, args.snr, args.val_codes, args.val_samples,
        margins, instance.snr, args.seed + 1000
    )

    parse_list = lambda text: tuple(float(value) for value in text.split(","))
    grid = grid_search(
        val_cases,
        taus=parse_list(args.grid_tau),
        xis=parse_list(args.grid_xi),
        lambdas=parse_list(args.grid_lam),
        deltas=parse_list(args.grid_delta),
    )
    best_config = grid[0]["config"]
    local_history: list[dict] = []
    if args.local_rounds > 0:
        print("Local parameter search")
        best_config, local_hards, local_history = local_search(
            val_cases, best_config, args.local_rounds
        )
        local_ber = ber_of(val_cases, local_hards)
        if local_ber <= grid[0]["ber"]:
            grid.insert(0, {
                "config": best_config,
                "lambd": best_config[2],
                "ber": local_ber,
                "syndrome_ones": syndrome_ones(val_cases, local_hards),
                "hards": local_hards,
            })
        print(f"Best after local search: ber={local_ber:.9f} config={best_config}")

    global_params = load_params(args.global_params)
    global_hards = decode_learned(val_cases, global_params)
    baseline = {
        "ber": ber_of(val_cases, global_hards),
        "syndrome_ones": syndrome_ones(val_cases, global_hards),
    }
    print(f"Global learned single BER: {baseline['ber']:.9f}")

    initial = constant_params(
        50,
        dt=0.2,
        xi=best_config[1],
        lam=best_config[2],
        delta=best_config[3],
        tau=best_config[0],
    )
    if args.no_train:
        learned = initial
        history = []
    else:
        learned, history = train(
            initial,
            train_cases,
            val_cases,
            args.epochs,
            args.batch_size,
            args.lr,
            args.syn_weight,
            args.temp,
            args.mod_scale,
            args.eval_every,
            args.seed + 2000,
        )
    learned_hards = decode_learned(val_cases, learned)
    learned_result = {
        "ber": ber_of(val_cases, learned_hards),
        "syndrome_ones": syndrome_ones(val_cases, learned_hards),
    }
    print(f"Per-case learned BER: {learned_result['ber']:.9f}")

    candidates: list[dict] = []
    candidates.append({
        "name": "global_learned",
        "lambd": 6.0,
        "energy_lambda": 6.0,
        "hards": global_hards,
    })
    for tau, xi, lambd, delta in FIXED_CANDIDATES:
        config = DSBConfig(
            tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50, gradient_sign=1
        )
        candidates.append({
            "name": f"fixed_{tau:g}_{xi:g}_{lambd:g}_{delta:g}",
            "lambd": lambd,
            "energy_lambda": lambd,
            "hards": decode_config(val_cases, config),
        })
    candidates.append({
        "name": "per_case_learned",
        "lambd": float(best_config[2]),
        "energy_lambda": float(best_config[2]),
        "hards": learned_hards,
    })
    for rank, item in enumerate(grid[: args.top_grid]):
        tau, xi, lambd, delta = item["config"]
        config = DSBConfig(
            tau=tau, xi=xi, lambd=lambd, delta=delta, dt=0.2, steps=50, gradient_sign=1
        )
        candidates.append({
            "name": f"grid{rank}_{tau:g}_{xi:g}_{lambd:g}_{delta:g}",
            "lambd": lambd,
            "energy_lambda": lambd,
            "hards": item["hards"] if item["config"] == best_config else decode_config(val_cases, config),
        })

    print("Greedy six-candidate portfolio search")
    portfolio = greedy_portfolio(val_cases, candidates, limit=6)
    baseline_subset = list(range(6))
    baseline_selected = select_energies(val_cases, candidates, baseline_subset)
    baseline_portfolio_ber = ber_of(val_cases, baseline_selected)

    energy_scale, energy_ber, energy_hards, energy_lambdas = tune_energy_scale(
        val_cases, candidates, portfolio["indices"]
    )
    if energy_ber < portfolio["ber"]:
        portfolio = dict(portfolio)
        portfolio["ber"] = energy_ber
        portfolio["syndrome_ones"] = syndrome_ones(val_cases, energy_hards)
        portfolio["energy_scale"] = energy_scale
        portfolio["energy_lambdas"] = {
            candidates[index]["name"]: energy_lambdas[index]
            for index in portfolio["indices"]
        }
        print(f"  tuned portfolio ber={energy_ber:.9f}")

    summary = {
        "case": {"n": args.n, "rate": args.rate, "snr": args.snr},
        "data": {
            "train_codes": args.train_codes,
            "train_samples_per_code": args.train_samples,
            "val_codes": args.val_codes,
            "val_samples_per_code": args.val_samples,
            "seed": args.seed,
        },
        "best_fixed_config": best_config,
        "local_search_history": local_history,
        "global_learned_single": baseline,
        "per_case_learned_single": learned_result,
        "accepted_portfolio_ber": baseline_portfolio_ber,
        "optimized_portfolio": portfolio,
        "training_history": history,
        "gate": {
            "per_case_beats_global_single": learned_result["ber"] < baseline["ber"],
            "portfolio_beats_accepted_portfolio": portfolio["ber"] < baseline_portfolio_ber,
            "guess_count_at_most_six": len(portfolio["indices"]) <= 6,
        },
    }
    print("\nPHASE 7 GATE")
    for key, value in summary["gate"].items():
        print(f"{key}: {value}")

    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = args.outdir / f"phase7_{label}_summary{suffix}.json"
    params_path = args.outdir / f"phase7_{label}_params{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    export_params(learned, params_path)
    print(f"\nSummary written to {summary_path}")
    print(f"Parameters written to {params_path}")


if __name__ == "__main__":
    main()
