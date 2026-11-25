#!/usr/bin/env python3
"""Phase 4: deep-unfolded dSB with NumPy-only custom backpropagation.

The decoder remains dSB only. Learnable parameters are shared across variables
or checks at each unfolding step, with small reliability-dependent modulations,
keeping the exported parameter set tiny. Ground truth is used only by the
offline training loss and for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase0_harness import SAMPLE_INPUT, SAMPLE_OUTPUT, parse_answers, parse_instance  # noqa: E402
from phase3_fixed_dsb import CheckTable  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_INPUT = ROOT / "Problem statements" / "publish.in"
DEFAULT_ANSWER = ROOT / "Problem statements" / "publish.ans"
DEFAULT_OUTDIR = ROOT / "artifacts"
MAX_ALLOWED_STEPS = 50
BASE_KEYS = ("dt", "xi", "lam", "delta", "tau")
MOD_KEYS = ("xi_mod", "lam_mod", "delta_mod")
PARAMETER_KEYS = BASE_KEYS + MOD_KEYS


@dataclass
class RawParams:
    """Log-parameters; exponentiation enforces positive scales."""

    values: dict[str, np.ndarray]

    def exp_values(self) -> dict[str, np.ndarray]:
        values = {key: np.exp(self.values[key]) for key in BASE_KEYS}
        values.update({key: self.values[key].copy() for key in MOD_KEYS})
        return values


def constant_params(steps: int, dt: float, xi: float, lam: float, delta: float, tau: float) -> RawParams:
    return RawParams({
        "dt": np.full(steps, np.log(dt)),
        "xi": np.full(steps, np.log(xi)),
        "lam": np.full(steps, np.log(lam)),
        "delta": np.full(steps, np.log(delta)),
        "tau": np.log(np.array([tau])),
        "xi_mod": np.zeros(steps),
        "lam_mod": np.zeros(steps),
        "delta_mod": np.zeros(steps),
    })


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def build_features(llr: np.ndarray, checks: CheckTable) -> dict[str, np.ndarray]:
    var_reliability = np.tanh(np.abs(llr) / 8.0)
    gathered = np.abs(llr)[:, checks.gather_indices]
    safe = np.where(checks.mask[None, :, :], gathered, 1e9)
    check_reliability = np.tanh(np.min(safe, axis=2) / 8.0)
    return {
        "var": var_reliability,
        "check": check_reliability,
    }


def forward(
    params: RawParams,
    h_float: np.ndarray,
    llr: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
    need_cache: bool = False,
) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    vals = params.exp_values()
    steps = vals["dt"].size
    schedule = np.linspace(0.0, 1.0, steps)
    tau = float(vals["tau"][0])

    x = np.tanh(llr / tau)
    y = np.zeros_like(x)
    cache: list[dict] = []

    for step in range(steps):
        spins = np.where(x >= 0, 1.0, -1.0)
        products = checks.parity_products(spins)

        xi_factor = np.exp(vals["xi_mod"][step] * features["var"])
        lam_factor = np.exp(vals["lam_mod"][step] * features["check"])
        delta_factor = np.exp(vals["delta_mod"][step] * features["var"])
        xi_vec = vals["xi"][step] * xi_factor
        lam_vec = vals["lam"][step] * lam_factor
        delta_vec = vals["delta"][step] * delta_factor

        weighted_sum = (products * lam_vec) @ h_float
        force = llr + 0.5 * spins * weighted_sum

        ydot = -(delta_vec - schedule[step]) * x + xi_vec * force
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


def loss_and_grad(
    params: RawParams,
    h_float: np.ndarray,
    llr: np.ndarray,
    targets: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
    syn_weight: float,
    temp: float,
) -> tuple[float, dict[str, np.ndarray]]:
    x_final, cache, vals = forward(params, h_float, llr, checks, features, need_cache=True)
    steps = vals["dt"].size
    k = targets.shape[1]
    batch, n = llr.shape

    z = -x_final[:, :k] / temp
    prob = sigmoid(z)
    bce = float(np.mean(np.logaddexp(0.0, z) - targets * z))

    soft_spins = np.tanh(x_final / temp)
    soft_products = checks.parity_products(soft_spins)
    parity_loss = 0.5 * np.mean((1.0 - soft_products) ** 2)
    loss = bce + syn_weight * parity_loss

    grad_x = np.zeros_like(x_final)
    grad_x[:, :k] = (targets - prob) / (temp * batch * k)

    grad_p = -syn_weight * (1.0 - soft_products) / soft_products.size
    safe_spins = np.where(
        np.abs(soft_spins) > 1e-6, soft_spins, np.sign(soft_spins + 1e-12) * 1e-6
    )
    grad_soft = ((grad_p * soft_products) @ h_float) / safe_spins
    grad_x += grad_soft * (1.0 - soft_spins**2) / temp

    grad_y = np.zeros_like(grad_x)
    grads = {key: np.zeros_like(params.values[key]) for key in PARAMETER_KEYS}
    schedule = np.linspace(0.0, 1.0, steps)
    var_feature = features["var"]
    check_feature = features["check"]

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
        grads["delta_mod"][step] += np.sum(grad_delta_vec * vals["delta"][step] * item["delta_factor"] * var_feature)
        grad_x_next += grad_delta_vec

        grad_force = item["xi_factor"] * grad_ydot * vals["xi"][step]
        grads["xi"][step] += np.sum(item["force"] * grad_ydot * item["xi_factor"])
        grads["xi_mod"][step] += np.sum(item["force"] * grad_ydot * vals["xi"][step] * item["xi_factor"] * var_feature)

        grad_weighted_sum = 0.5 * item["spins"] * grad_force
        grad_x_next += 0.5 * item["weighted_sum"] * grad_force

        grad_lam_m = item["products"] * (grad_weighted_sum @ h_float.T)
        grads["lam"][step] += np.sum(grad_lam_m * item["lam_factor"])
        grads["lam_mod"][step] += np.sum(
            grad_lam_m * vals["lam"][step] * item["lam_factor"] * check_feature
        )

        grad_check_gain = (grad_weighted_sum @ h_float.T) * item["products"]
        grad_spins = item["spins"] * ((grad_check_gain * vals["lam"][step] * item["lam_factor"]) @ h_float)
        grad_x_next += grad_spins

        grad_x, grad_y = grad_x_next, grad_y_next

    tau = float(vals["tau"][0])
    d_tanh = 1.0 - np.tanh(llr / tau) ** 2
    grads["tau"][0] += np.sum(grad_x * (-(llr / tau**2) * d_tanh))

    grad_raw = {
        key: grads[key] * (vals[key] if key in BASE_KEYS else 1.0)
        for key in PARAMETER_KEYS
    }
    return loss, grad_raw


class Adam:
    def __init__(self, params: RawParams, lr: float) -> None:
        self.lr = lr
        self.t = 0
        self.m = {key: np.zeros_like(value) for key, value in params.values.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.values.items()}

    def step(self, params: RawParams, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key in PARAMETER_KEYS:
            self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * grads[key]
            self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * grads[key] ** 2
            mhat = self.m[key] / (1.0 - beta1**self.t)
            vhat = self.v[key] / (1.0 - beta2**self.t)
            params.values[key] -= self.lr * mhat / (np.sqrt(vhat) + eps)


def clip_grads(grads: dict[str, np.ndarray], max_norm: float) -> float:
    norm = np.sqrt(sum(np.sum(value**2) for value in grads.values()))
    scale = min(1.0, max_norm / max(norm, 1e-12))
    for key in grads:
        grads[key] *= scale
    return float(norm)


def decode_hard(
    params: RawParams,
    h_float: np.ndarray,
    llr: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
) -> np.ndarray:
    x, _, _ = forward(params, h_float, llr, checks, features)
    return (x < 0).astype(np.int8)


def evaluate(
    params: RawParams,
    h: np.ndarray,
    h_float: np.ndarray,
    llr: np.ndarray,
    answers: np.ndarray,
    checks: CheckTable,
    features: dict[str, np.ndarray],
) -> dict:
    hard = decode_hard(params, h_float, llr, checks, features)
    info_truth = answers[:, : h.shape[1] - h.shape[0]]
    info_errors = hard[:, :info_truth.shape[1]] != info_truth
    final_spins = 1.0 - 2.0 * hard.astype(np.float64)
    products = checks.parity_products(final_spins)
    return {
        "info_errors": int(info_errors.sum()),
        "info_ber": float(info_errors.sum() / info_truth.size),
        "frame_error_rate": float(np.any(info_errors, axis=1).mean()),
        "rows_with_failed_checks": int(np.count_nonzero(np.any(products < 0, axis=1))),
        "syndrome_ones": int(np.count_nonzero(products < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--syn-weight", type=float, default=0.10)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=400)
    parser.add_argument("--summary-name", default="phase4_unfolded_dsb_summary.json")
    parser.add_argument("--params-name", default="phase4_unfolded_dsb_params.json")
    args = parser.parse_args()

    instance = parse_instance(args.input.read_text(encoding="utf-8"), str(args.input))
    answers = parse_answers(
        args.answer.read_text(encoding="utf-8"), str(args.answer), instance.n, instance.b
    )
    k = instance.k
    h_float = instance.h.astype(np.float64)
    checks = CheckTable(instance.h)
    targets = answers[:, :k].astype(np.float64)
    features = build_features(instance.llr, checks)

    train_count = min(instance.b - 1, max(1, args.train_samples))
    permutation = np.random.default_rng(20261125).permutation(instance.b)
    train_idx, val_idx = permutation[:train_count], permutation[train_count:]

    baseline = constant_params(
        MAX_ALLOWED_STEPS, dt=0.2, xi=0.4, lam=6.0, delta=2.0, tau=8.0
    )
    baseline_val = evaluate(
        baseline, instance.h, h_float, instance.llr[val_idx], answers[val_idx],
        checks, {key: value[val_idx] for key, value in features.items()}
    )
    hard_val_ber = float(np.mean(
        (instance.llr[val_idx, :k] < 0) != answers[val_idx, :k]
    ))

    print(f"Train samples: {len(train_idx)}, validation samples: {len(val_idx)}")
    print(f"Fixed dSB validation BER: {baseline_val['info_ber']:.9f}")
    print(f"Hard decision validation BER: {hard_val_ber:.9f}\n")

    params = constant_params(
        MAX_ALLOWED_STEPS, dt=0.2, xi=0.4, lam=6.0, delta=2.0, tau=8.0
    )
    optimizer = Adam(params, args.lr)
    best_val_ber = float("inf")
    best_params = None
    start = time.perf_counter()
    train_features = {key: value[train_idx] for key, value in features.items()}
    val_features = {key: value[val_idx] for key, value in features.items()}

    for epoch in range(1, args.epochs + 1):
        loss, grads = loss_and_grad(
            params,
            h_float,
            instance.llr[train_idx],
            targets[train_idx],
            checks,
            train_features,
            args.syn_weight,
            args.temp,
        )
        raw_norm = clip_grads(grads, 5.0)
        optimizer.step(params, grads)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val = evaluate(
                params, instance.h, h_float, instance.llr[val_idx], answers[val_idx],
                checks, val_features
            )
            if val["info_ber"] < best_val_ber:
                best_val_ber = val["info_ber"]
                best_params = RawParams({key: value.copy() for key, value in params.values.items()})
            print(
                f"epoch {epoch:>3d} loss={loss:.6f} grad_norm={raw_norm:>8.2f} "
                f"val_ber={val['info_ber']:.9f} val_synd={val['syndrome_ones']}"
            )

    if best_params is None:
        raise RuntimeError("No validation evaluation was performed")
    params = best_params
    training_seconds = time.perf_counter() - start

    sample = parse_instance(SAMPLE_INPUT, "PDF sample")
    sample_expected = parse_answers(SAMPLE_OUTPUT, "PDF sample answer", sample.n, sample.b)
    sample_checks = CheckTable(sample.h)
    sample_features = build_features(sample.llr, sample_checks)
    sample_decoded = decode_hard(
        params, sample.h.astype(np.float64), sample.llr, sample_checks, sample_features
    )
    sample_ok = bool(np.array_equal(sample_decoded, sample_expected))

    final_val = evaluate(
        params, instance.h, h_float, instance.llr[val_idx], answers[val_idx],
        checks, val_features
    )
    full = evaluate(
        params, instance.h, h_float, instance.llr, answers, checks, features
    )
    values = params.exp_values()
    parameter_count = sum(value.size for value in params.values.values())

    gate = {
        "pdf_sample_matches": sample_ok,
        "steps_within_limit": MAX_ALLOWED_STEPS <= 50,
        "val_ber_below_fixed_dsb": final_val["info_ber"] < baseline_val["info_ber"],
        "val_ber_below_hard_decision": final_val["info_ber"] < hard_val_ber,
        "parameters_under_512k": parameter_count < 512 * 1024,
    }

    print("\nBest validation result:")
    print(json.dumps(final_val, indent=2, sort_keys=True))
    print("\nPHASE 4 GATE")
    for key, value in gate.items():
        print(f"{key}: {value}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "framework": "deep_unfolded_dsb_input_dependent",
        "training_uses_ground_truth_only_offline": True,
        "trainable_parameters": parameter_count,
        "training_seconds": training_seconds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "synergy_weight": args.syn_weight,
        "temperature": args.temp,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "hard_decision_validation_ber": hard_val_ber,
        "fixed_dsb_validation": baseline_val,
        "best_validation": final_val,
        "full_public": full,
        "pdf_sample_matches": sample_ok,
        "learned_values": {key: value.tolist() for key, value in values.items()},
        "gate": gate,
    }
    summary_path = args.outdir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    params_path = args.outdir / args.params_name
    params_path.write_text(
        json.dumps(
            {
                "steps": MAX_ALLOWED_STEPS,
                "values": {key: value.tolist() for key, value in values.items()},
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"\nSummary written to {summary_path}")
    print(f"Parameters written to {params_path}")

    if not all(gate.values()):
        raise SystemExit("Phase 4 gate failed; inspect the training summary.")


if __name__ == "__main__":
    main()
