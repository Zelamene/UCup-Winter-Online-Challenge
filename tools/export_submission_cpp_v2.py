#!/usr/bin/env python3
"""Export per-case Phase 7 parameters into a single-file C++17 dSB decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase7_case_manifest.json"
DEFAULT_OUTPUT = ROOT / "submission_v2.cpp"

CPP_TEMPLATE = r'''// Single-file QAIA/dSB decoder with per-case unfolding parameters.
// Six dSB trajectories at most; lowest-energy result is selected.
// No BP, min-sum, bit-flip, or ground-truth information is used.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using std::size_t;

static constexpr int STEPS = 50;
static constexpr double FIXED_DT = 0.2;

struct FixedConfig {
    double tau;
    double xi;
    double lambd;
    double delta;
    double energy_lambda;
};

struct LearnedParams {
    double tau;
    const double* dt;
    const double* xi;
    const double* lam;
    const double* delta;
    const double* xi_mod;
    const double* lam_mod;
    const double* delta_mod;
};

struct CaseConfig {
    int n;
    int k;
    double snr;
    bool use_learned;
    LearnedParams learned;
    double learned_lambda;
    int fixed_count;
    const FixedConfig* fixed;
};

__ARRAYS__

static std::vector<int> decode_learned(
    const std::vector<double>& r,
    const std::vector<std::vector<int>>& checks,
    const LearnedParams& p
) {
    const int n = static_cast<int>(r.size());
    const int m = static_cast<int>(checks.size());

    std::vector<double> var_reliability(n);
    std::vector<double> check_reliability(m);
    std::vector<double> spins(n);
    std::vector<double> weighted_sum(n);
    std::vector<double> force(n);
    std::vector<double> ydot(n);
    std::vector<double> y_raw(n);
    std::vector<double> x_raw(n);
    std::vector<double> x(n);
    std::vector<double> y(n);

    for (int col = 0; col < n; ++col) {
        var_reliability[col] = std::tanh(std::fabs(r[col]) / 8.0);
    }
    for (int row = 0; row < m; ++row) {
        double smallest = 1e100;
        for (int col : checks[row]) {
            smallest = std::min(smallest, std::fabs(r[col]));
        }
        check_reliability[row] = std::tanh(smallest / 8.0);
    }
    for (int col = 0; col < n; ++col) {
        x[col] = std::tanh(r[col] / p.tau);
        y[col] = 0.0;
    }

    for (int step = 0; step < STEPS; ++step) {
        for (int col = 0; col < n; ++col) {
            spins[col] = (x[col] >= 0.0) ? 1.0 : -1.0;
        }
        std::fill(weighted_sum.begin(), weighted_sum.end(), 0.0);
        for (int row = 0; row < m; ++row) {
            double product = 1.0;
            for (int col : checks[row]) {
                product *= spins[col];
            }
            const double gain = p.lam[step] * std::exp(
                p.lam_mod[step] * check_reliability[row]
            );
            for (int col : checks[row]) {
                weighted_sum[col] += product * gain;
            }
        }

        const double schedule = static_cast<double>(step) / (STEPS - 1);
        for (int col = 0; col < n; ++col) {
            force[col] = r[col] + 0.5 * spins[col] * weighted_sum[col];
            const double delta_gain = p.delta[step] * std::exp(
                p.delta_mod[step] * var_reliability[col]
            );
            const double xi_gain = p.xi[step] * std::exp(
                p.xi_mod[step] * var_reliability[col]
            );
            ydot[col] = -(delta_gain - schedule) * x[col] + xi_gain * force[col];
            y_raw[col] = y[col] + p.dt[step] * ydot[col];
            x_raw[col] = x[col] + p.dt[step] * y_raw[col];
            if (std::fabs(x_raw[col]) > 1.0) {
                x[col] = (x_raw[col] >= 0.0) ? 1.0 : -1.0;
                y[col] = 0.0;
            } else {
                x[col] = x_raw[col];
                y[col] = y_raw[col];
            }
        }
    }

    std::vector<int> hard(n);
    for (int col = 0; col < n; ++col) {
        hard[col] = (x[col] < 0.0) ? 1 : 0;
    }
    return hard;
}

static std::vector<int> decode_fixed(
    const std::vector<double>& r,
    const std::vector<std::vector<int>>& checks,
    const FixedConfig& config
) {
    const int n = static_cast<int>(r.size());
    const int m = static_cast<int>(checks.size());

    std::vector<double> spins(n);
    std::vector<double> weighted_sum(n);
    std::vector<double> force(n);
    std::vector<double> x(n);
    std::vector<double> y(n);

    for (int col = 0; col < n; ++col) {
        x[col] = std::tanh(r[col] / config.tau);
        y[col] = 0.0;
    }

    for (int step = 0; step < STEPS; ++step) {
        for (int col = 0; col < n; ++col) {
            spins[col] = (x[col] >= 0.0) ? 1.0 : -1.0;
        }
        std::fill(weighted_sum.begin(), weighted_sum.end(), 0.0);
        for (int row = 0; row < m; ++row) {
            double product = 1.0;
            for (int col : checks[row]) {
                product *= spins[col];
            }
            for (int col : checks[row]) {
                weighted_sum[col] += product;
            }
        }

        const double schedule = static_cast<double>(step) / (STEPS - 1);
        for (int col = 0; col < n; ++col) {
            force[col] = r[col] + 0.5 * config.lambd * spins[col] * weighted_sum[col];
            const double pump = -(config.delta - schedule) * x[col];
            const double y_new = y[col] + FIXED_DT * (pump + config.xi * force[col]);
            const double x_new = x[col] + FIXED_DT * y_new;
            if (std::fabs(x_new) > 1.0) {
                x[col] = (x_new >= 0.0) ? 1.0 : -1.0;
                y[col] = 0.0;
            } else {
                x[col] = x_new;
                y[col] = y_new;
            }
        }
    }

    std::vector<int> hard(n);
    for (int col = 0; col < n; ++col) {
        hard[col] = (x[col] < 0.0) ? 1 : 0;
    }
    return hard;
}

static double energy(
    const std::vector<int>& hard,
    const std::vector<double>& r,
    const std::vector<std::vector<int>>& checks,
    double lambd
) {
    const int n = static_cast<int>(r.size());
    std::vector<double> spins(n);
    for (int col = 0; col < n; ++col) {
        spins[col] = (hard[col] == 0) ? 1.0 : -1.0;
    }

    double channel = 0.0;
    for (int col = 0; col < n; ++col) {
        channel -= r[col] * spins[col];
    }
    double penalty = 0.0;
    for (const auto& row : checks) {
        double product = 1.0;
        for (int col : row) {
            product *= spins[col];
        }
        penalty += 1.0 - product;
    }
    return channel + 0.5 * lambd * penalty;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n = 0;
    int k = 0;
    int b = 0;
    double snr = 0.0;
    if (!(std::cin >> n >> k >> b >> snr)) {
        return 0;
    }

    std::vector<std::vector<int>> checks(n - k);
    for (int row = 0; row < n - k; ++row) {
        checks[row].reserve(8);
        for (int col = 0; col < n; ++col) {
            int value = 0;
            std::cin >> value;
            if (value != 0) {
                checks[row].push_back(col);
            }
        }
    }

    std::vector<std::vector<double>> llr(b, std::vector<double>(n));
    for (int sample = 0; sample < b; ++sample) {
        for (int col = 0; col < n; ++col) {
            std::cin >> llr[sample][col];
        }
    }

    int unused = 0;
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cin >> unused;
        }
    }

    const CaseConfig* active = &FALLBACK;
    for (const CaseConfig& candidate : CASES) {
        if (candidate.n == n && candidate.k == k &&
            std::fabs(candidate.snr - snr) < 0.25) {
            active = &candidate;
            break;
        }
    }

    for (int sample = 0; sample < b; ++sample) {
        std::vector<std::vector<int>> candidates;
        candidates.reserve(6);
        std::vector<double> lambdas;
        lambdas.reserve(6);

        if (active->use_learned) {
            candidates.push_back(decode_learned(llr[sample], checks, active->learned));
            lambdas.push_back(active->learned_lambda);
        }
        for (int i = 0; i < active->fixed_count; ++i) {
            candidates.push_back(decode_fixed(llr[sample], checks, active->fixed[i]));
            lambdas.push_back(active->fixed[i].energy_lambda);
        }

        std::vector<int> best = candidates[0];
        double best_energy = energy(best, llr[sample], checks, lambdas[0]);
        for (size_t i = 1; i < candidates.size(); ++i) {
            const double value = energy(candidates[i], llr[sample], checks, lambdas[i]);
            if (value < best_energy) {
                best = candidates[i];
                best_energy = value;
            }
        }

        for (int col = 0; col < n; ++col) {
            std::cout << best[col];
            std::cout << ((col + 1 == n) ? '\n' : ' ');
        }
    }
    return 0;
}
'''


def fmt_array(values: list[float]) -> str:
    return ", ".join(f"{float(value):.17g}" for value in values)


def fmt_fixed(configs: list[list[float]]) -> str:
    rows = []
    for config in configs:
        row = [float(value) for value in config]
        if len(row) == 4:
            row.append(row[2])
        if len(row) != 5:
            raise ValueError(f"fixed config must have 4 or 5 values, got {len(row)}")
        rows.append("{" + fmt_array(row) + "}")
    return ", ".join(rows)


def make_model(prefix: str, payload: dict) -> str:
    values = payload["values"]
    keys = ("dt", "xi", "lam", "delta", "xi_mod", "lam_mod", "delta_mod")
    if payload.get("steps") != 50:
        raise ValueError(f"{prefix}: expected 50 steps")
    for key in keys:
        if len(values[key]) != 50:
            raise ValueError(f"{prefix}: {key} must have 50 values")
    if len(values["tau"]) != 1:
        raise ValueError(f"{prefix}: tau must be scalar")

    blocks = []
    names = []
    for key in keys:
        name = f"{prefix}_{key.upper()}"
        names.append(name)
        blocks.append(f"static const double {name}[STEPS] = {{{fmt_array(values[key])}}};")
    learned_name = f"{prefix}_LEARNED"
    pointers = ", ".join(names)
    blocks.append(
        f"static const LearnedParams {learned_name} = "
        f"{{{float(values['tau'][0]):.17g}, {pointers}}};"
    )
    return "\n".join(blocks), learned_name


def make_case_block(
    index: int,
    model: dict,
    params: dict,
    width: int = 0,
) -> tuple[str, str]:
    prefix = f"MODEL_{index}"
    arrays, learned_name = make_model(prefix, params)
    configs = model["fixed_candidates"]
    fixed_name = f"FIXED_{index}"
    if configs:
        fixed = (
            f"static const FixedConfig {fixed_name}[{len(configs)}] = "
            f"{{{fmt_fixed(configs)}}};"
        )
    else:
        fixed = f"static const FixedConfig* {fixed_name} = nullptr;"
    n = int(model["n"])
    k = int(model["k"])
    snr = float(model["snr"])
    learned_lambda = float(model.get("learned_lambda", 6.0))
    use_learned = "true" if bool(model.get("use_learned", True)) else "false"
    case = (
        f"{{{n}, {k}, {snr:.17g}, {use_learned}, {learned_name}, "
        f"{learned_lambda:.17g}, {len(configs)}, {fixed_name}}}"
    )
    del width
    return arrays + "\n" + fixed, case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.resolve().parent.parent
    fallback = manifest["fallback"]
    cases = manifest["cases"]

    pieces: list[str] = []
    case_entries: list[str] = []
    for index, model in enumerate([fallback] + cases):
        params_path = Path(model["params"])
        if not params_path.is_absolute():
            params_path = root / params_path
        blocks, entry = make_case_block(index, model, json.loads(params_path.read_text()))
        pieces.append(blocks)
        if index == 0:
            pieces.append(f"static const CaseConfig FALLBACK = {entry};")
        else:
            case_entries.append(entry)

    pieces.append(
        f"static const CaseConfig CASES[{len(case_entries)}] = {{"
        + ", ".join(case_entries)
        + "};"
    )
    source = CPP_TEMPLATE.replace("__ARRAYS__", "\n\n".join(pieces))
    if "__" in source:
        raise ValueError("Unreplaced template token remains")

    args.output.write_text(source, encoding="utf-8", newline="\n")
    size = args.output.stat().st_size
    print(f"Wrote {args.output} ({size} bytes)")
    if size >= 512 * 1024:
        raise SystemExit("Generated submission exceeds 512K")


if __name__ == "__main__":
    main()
