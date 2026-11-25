#!/usr/bin/env python3
"""Export learned Phase 4 parameters and Phase 6 candidates into C++17."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from phase6_multi_guess import FIXED_CANDIDATES, LEARNED_ENERGY_LAMBDA  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_PARAMS = ROOT / "artifacts" / "phase4_unfolded_dsb_params.json"
DEFAULT_OUTPUT = ROOT / "submission.cpp"


def fmt_array(values: list[float]) -> str:
    return ", ".join(f"{float(value):.17g}" for value in values)


def fmt_configs(configs: tuple[tuple[float, float, float, float], ...]) -> str:
    return ", ".join("{" + fmt_array(list(config)) + "}" for config in configs)


CPP_TEMPLATE = r'''// Single-file QAIA/dSB decoder with compact learned unfolding parameters.
// Six dSB trajectories are generated and the lowest-energy result is selected.
// No BP, min-sum, bit-flip, or ground-truth information is used.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using std::size_t;

static constexpr int STEPS = 50;
static constexpr double TAU = __TAU__;
static constexpr double LEARNED_ENERGY_LAMBDA = __LEARNED_LAMBDA__;
static constexpr double FIXED_DT = 0.2;
static const double DT[STEPS] = {__DT__};
static const double XI[STEPS] = {__XI__};
static const double LAM[STEPS] = {__LAM__};
static const double DELTA[STEPS] = {__DELTA__};
static const double XI_MOD[STEPS] = {__XI_MOD__};
static const double LAM_MOD[STEPS] = {__LAM_MOD__};
static const double DELTA_MOD[STEPS] = {__DELTA_MOD__};

struct FixedConfig {
    double tau;
    double xi;
    double lambd;
    double delta;
};

static const FixedConfig FIXED_CANDIDATES[5] = {__FIXED__};

static std::vector<int> decode_learned(
    const std::vector<double>& r, const std::vector<std::vector<int>>& checks
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
        x[col] = std::tanh(r[col] / TAU);
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
            const double check_gain = LAM[step] * std::exp(
                LAM_MOD[step] * check_reliability[row]
            );
            for (int col : checks[row]) {
                weighted_sum[col] += product * check_gain;
            }
        }

        const double schedule = static_cast<double>(step) / (STEPS - 1);
        for (int col = 0; col < n; ++col) {
            force[col] = r[col] + 0.5 * spins[col] * weighted_sum[col];
            const double delta_gain = DELTA[step] * std::exp(
                DELTA_MOD[step] * var_reliability[col]
            );
            const double xi_gain = XI[step] * std::exp(
                XI_MOD[step] * var_reliability[col]
            );
            ydot[col] = -(delta_gain - schedule) * x[col] + xi_gain * force[col];
            y_raw[col] = y[col] + DT[step] * ydot[col];
            x_raw[col] = x[col] + DT[step] * y_raw[col];
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
    (void)snr;
    const int m = n - k;

    std::vector<std::vector<int>> checks(m);
    for (int row = 0; row < m; ++row) {
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

    // G is not needed by the decoder; consume it.
    int unused = 0;
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cin >> unused;
        }
    }

    for (int sample = 0; sample < b; ++sample) {
        std::vector<std::vector<int>> candidates;
        candidates.reserve(6);
        candidates.push_back(decode_learned(llr[sample], checks));
        for (int i = 0; i < 5; ++i) {
            candidates.push_back(decode_fixed(llr[sample], checks, FIXED_CANDIDATES[i]));
        }

        std::vector<int> best = candidates[0];
        double best_energy = energy(best, llr[sample], checks, LEARNED_ENERGY_LAMBDA);
        for (int i = 0; i < 5; ++i) {
            const double candidate_energy = energy(
                candidates[i + 1], llr[sample], checks, FIXED_CANDIDATES[i].lambd
            );
            if (candidate_energy < best_energy) {
                best = candidates[i + 1];
                best_energy = candidate_energy;
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if len(FIXED_CANDIDATES) != 5:
        raise ValueError("Expected exactly five fixed dSB candidates")

    payload = json.loads(args.params.read_text(encoding="utf-8"))
    values = payload["values"]
    keys = ("dt", "xi", "lam", "delta", "xi_mod", "lam_mod", "delta_mod")
    if payload.get("steps") != 50:
        raise ValueError("Expected 50 unfolding steps")
    for key in keys:
        if len(values[key]) != 50:
            raise ValueError(f"{key} must contain 50 values")
    if len(values["tau"]) != 1:
        raise ValueError("tau must contain one value")

    source = CPP_TEMPLATE
    replacements = {
        "__TAU__": f"{float(values['tau'][0]):.17g}",
        "__LEARNED_LAMBDA__": f"{float(LEARNED_ENERGY_LAMBDA):.17g}",
        "__DT__": fmt_array(values["dt"]),
        "__XI__": fmt_array(values["xi"]),
        "__LAM__": fmt_array(values["lam"]),
        "__DELTA__": fmt_array(values["delta"]),
        "__XI_MOD__": fmt_array(values["xi_mod"]),
        "__LAM_MOD__": fmt_array(values["lam_mod"]),
        "__DELTA_MOD__": fmt_array(values["delta_mod"]),
        "__FIXED__": fmt_configs(FIXED_CANDIDATES),
    }
    for token, value in replacements.items():
        source = source.replace(token, value)

    if "__" in source:
        raise ValueError("Unreplaced template token remains in generated C++")

    args.output.write_text(source, encoding="utf-8", newline="\n")
    size = args.output.stat().st_size
    print(f"Wrote {args.output} ({size} bytes)")
    if size >= 512 * 1024:
        raise SystemExit("Generated submission exceeds 512K")


if __name__ == "__main__":
    main()
