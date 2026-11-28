#!/usr/bin/env python3
"""Export a structural QAIA v3 manifest into one C++17 source file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "phase8_case_manifest_v3.json"
DEFAULT_OUTPUT = ROOT / "submission_v3.cpp"

MODES = {"dsb": "Mode::Dsb", "bsb": "Mode::Bsb", "kspin": "Mode::KSpin"}
COMBINES = {
    "energy": "Combine::Energy",
    "vote": "Combine::Vote",
    "soft_vote": "Combine::SoftVote",
}
MAX_TRAJECTORIES = 6
STEPS = 50
PARAM_KEYS = ("tau", "xi", "lambd", "delta")
SCHEDULE_KEYS = ("tau", "xi", "lambd", "delta", "dt")

CPP_TEMPLATE = r'''// Single-file structural QAIA decoder for the v3 manifest.
// dSB/bSB check-product dynamics, at most six trajectories, fifty steps.
// Only QAIA check products are used; no external decoding stages.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using std::size_t;

static constexpr int STEPS = 50;
static constexpr double EPS = 1e-12;

enum class Mode { Dsb, Bsb, KSpin };
enum class Combine { Energy, Vote, SoftVote };

struct TrajectoryConfig {
    double tau;
    double xi;
    double lambd;
    double delta;
    double energy_lambda;
    double dt;
    const double* tau_values;
    const double* xi_values;
    const double* lambd_values;
    const double* delta_values;
    const double* dt_values;
    Mode mode;
    double beta;          // Onsager/TAP reaction strength (bSB only); 0 disables
    double parity_scale;  // check-force scale on parity columns (last m); 1 disables
    const double* beta_values;
    int joint_pairs;      // 1 = exact Gray-pair channel field (bSB only)
    double pair_gamma;    // pair reaction strength
    double pair_scale;    // diagnostic scale on the pair coupling; 1 is exact
    double g;             // KSpin parity-coupling scale
    double clip;          // KSpin soft-clip radius for parity LLRs
    int select_unclipped; // KSpin energy selection uses raw parity LLRs
};

struct CaseConfig {
    int n;
    int k;
    double snr;
    Combine combine;
    Mode mode;
    int count;
    const TrajectoryConfig* trajectories;
};

__ARRAYS__

// Exact joint Gray-pair likelihood recovery. Both the Python reference and
// this exporter use the same bracket, bisection count and Newton steps.
static const double QAM_LEVELS[4] = {
    -3.0 / std::sqrt(10.0), -1.0 / std::sqrt(10.0),
     1.0 / std::sqrt(10.0),  3.0 / std::sqrt(10.0)
};
static const double QAM_BRACKET = 3.0 / std::sqrt(10.0) + 12.0;

static double log_sum_exp2(double first, double second) {
    const double peak = std::max(first, second);
    return peak + std::log1p(std::exp(-std::fabs(first - second)));
}

// Bisection needs millions of softplus values per frame. A cubic-interpolated
// table reproduces log1p(exp(-x)) to about 1e-10, far below the decoder's LLR
// resolution; the final consistency check still uses the exact log_sum_exp2.
static const int SOFTPLUS_STEPS = 1600;
static const double SOFTPLUS_MAX = 40.0;
static double softplus_table[SOFTPLUS_STEPS + 4];
static bool softplus_ready = false;

static void ensure_softplus_table() {
    if (softplus_ready) return;
    const double step = SOFTPLUS_MAX / SOFTPLUS_STEPS;
    for (int index = 0; index <= SOFTPLUS_STEPS + 3; ++index) {
        softplus_table[index] = std::log1p(std::exp(-step * index));
    }
    softplus_ready = true;
}

static double softplus_neg(double x) {
    if (x >= SOFTPLUS_MAX) return 0.0;
    ensure_softplus_table();
    const double scaled = x * SOFTPLUS_STEPS / SOFTPLUS_MAX;
    int index = static_cast<int>(scaled);
    if (index > SOFTPLUS_STEPS - 1) index = SOFTPLUS_STEPS - 1;
    const double t = scaled - index;
    const double p0 = softplus_table[index];
    const double p1 = softplus_table[index + 1];
    const double p2 = softplus_table[index + 2];
    const double p3 = softplus_table[index + 3];
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double l0 = (-t3 + 6.0 * t2 - 11.0 * t + 6.0) / 6.0;
    const double l1 = (t3 - 5.0 * t2 + 6.0 * t) / 2.0;
    const double l2 = (-t3 + 4.0 * t2 - 3.0 * t) / 2.0;
    const double l3 = (t3 - 3.0 * t2 + 2.0 * t) / 6.0;
    return p0 * l0 + p1 * l1 + p2 * l2 + p3 * l3;
}

static double fast_log_sum_exp2(double first, double second) {
    const double peak = std::max(first, second);
    return peak + softplus_neg(std::fabs(first - second));
}

// The common -y^2/(2 sigma^2) term cancels from every log-likelihood
// difference below, so precompute only the y-linear part of each component.
struct PairChannel {
    double slope[4];
    double intercept[4];
    double inv_sigma_sq;
};

static PairChannel make_pair_channel(double sigma) {
    PairChannel channel;
    channel.inv_sigma_sq = 1.0 / (sigma * sigma);
    for (int i = 0; i < 4; ++i) {
        const double level = QAM_LEVELS[i];
        channel.slope[i] = level * channel.inv_sigma_sq;
        channel.intercept[i] = -0.5 * level * level * channel.inv_sigma_sq;
    }
    return channel;
}

static void component_loglikes(double y, const PairChannel& channel, double out[4]) {
    for (int i = 0; i < 4; ++i) {
        out[i] = channel.slope[i] * y + channel.intercept[i];
    }
}

static double first_bit_llr_y(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    return fast_log_sum_exp2(ell[0], ell[1]) - fast_log_sum_exp2(ell[2], ell[3]);
}

// K-spin bit identity requires the same exact log1p(exp(-x)) used by
// qam_pairs.py.  Non-K-spin cases keep the faster interpolated table.
static double first_bit_llr_y_exact(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    return log_sum_exp2(ell[0], ell[1]) - log_sum_exp2(ell[2], ell[3]);
}

static double second_bit_llr_y(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    return log_sum_exp2(ell[0], ell[3]) - log_sum_exp2(ell[1], ell[2]);
}

static double first_bit_llr_derivative(double y, const PairChannel& channel) {
    double ell[4];
    component_loglikes(y, channel, ell);
    double peak = std::max(ell[0], ell[1]);
    double w0 = std::exp(ell[0] - peak);
    double w1 = std::exp(ell[1] - peak);
    const double mu_negative = (w0 * QAM_LEVELS[0] + w1 * QAM_LEVELS[1]) / (w0 + w1);
    peak = std::max(ell[2], ell[3]);
    w0 = std::exp(ell[2] - peak);
    w1 = std::exp(ell[3] - peak);
    const double mu_positive = (w0 * QAM_LEVELS[2] + w1 * QAM_LEVELS[3]) / (w0 + w1);
    return (mu_negative - mu_positive) * channel.inv_sigma_sq;
}

static double recover_amplitude(
    double target, const PairChannel& channel, bool exact = false
) {
    double low = -QAM_BRACKET;
    double high = QAM_BRACKET;
    for (int i = 0; i < 64; ++i) {
        const double middle = 0.5 * (low + high);
        const double value = exact
            ? first_bit_llr_y_exact(middle, channel)
            : first_bit_llr_y(middle, channel);
        if (value > target) {
            low = middle;
        } else {
            high = middle;
        }
    }
    double y = 0.5 * (low + high);
    for (int i = 0; i < 2; ++i) {
        const double derivative = first_bit_llr_derivative(y, channel);
        if (std::fabs(derivative) < 1e-300) break;
        const double value = exact
            ? first_bit_llr_y_exact(y, channel)
            : first_bit_llr_y(y, channel);
        const double step = (value - target) / derivative;
        y = std::min(QAM_BRACKET, std::max(-QAM_BRACKET, y - step));
    }
    return y;
}

struct PairTables {
    std::vector<double> linear;
    std::vector<double> d;
};

static PairTables build_pair_tables(
    const std::vector<double>& llr, double snr, double& bad_fraction,
    bool exact = false
) {
    const int n = static_cast<int>(llr.size());
    PairTables tables;
    tables.linear.assign(n, 0.0);
    tables.d.assign(n, 0.0);
    const double sigma = 1.0 / std::sqrt(2.0 * std::pow(10.0, snr / 10.0));
    const PairChannel channel = make_pair_channel(sigma);
    int bad = 0;
    for (int pair = 0; pair < n / 2; ++pair) {
        const int first = 2 * pair;
        const int second = first + 1;
        const double y = recover_amplitude(llr[first], channel, exact);
        double ell[4];
        component_loglikes(y, channel, ell);
        // Gray order: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
        const double a = (ell[0] + ell[1] - ell[3] - ell[2]) / 4.0;
        const double b = (ell[0] - ell[1] + ell[3] - ell[2]) / 4.0;
        const double d = (ell[0] - ell[1] - ell[3] + ell[2]) / 4.0;
        tables.linear[first] = 2.0 * a;
        tables.linear[second] = 2.0 * b;
        tables.d[first] = d;
        tables.d[second] = d;
        const double tolerance = 1e-6 * std::max(1.0, std::fabs(llr[second]));
        if (std::fabs(second_bit_llr_y(y, channel) - llr[second]) > tolerance) {
            ++bad;
        }
    }
    bad_fraction = (n > 0) ? static_cast<double>(bad) / (n / 2) : 0.0;
    return tables;
}

static std::vector<int> run_trajectory(
    const std::vector<double>& llr,
    const std::vector<int>& starts,
    const std::vector<int>& cols,
    const TrajectoryConfig& cfg,
    std::vector<double>& x_out,
    const std::vector<double>* pair_linear,
    const std::vector<double>* pair_d
) {
    const int n = static_cast<int>(llr.size());
    const bool joint = (cfg.joint_pairs == 1 && pair_linear != nullptr && pair_d != nullptr);
    const std::vector<double>& channel = joint ? *pair_linear : llr;
    const int m = static_cast<int>(starts.size()) - 1;
    std::vector<double> spins(n);
    std::vector<double> weighted(n, 0.0);
    std::vector<double> force(n);
    std::vector<double> x(n);
    std::vector<double> y(n);
    int max_deg = 0;
    for (int row = 0; row < m; ++row) {
        max_deg = std::max(max_deg, starts[row + 1] - starts[row]);
    }
    std::vector<double> pref(max_deg + 1, 1.0);
    std::vector<double> suff(max_deg + 1, 1.0);
    const bool use_reaction = (cfg.beta != 0.0 || cfg.beta_values != nullptr);
    const bool use_scale = (cfg.parity_scale != 1.0);
    const int parity_start = n - m;  // H=[A|I]
    std::vector<double> reaction(use_reaction ? n : 0, 0.0);
    std::vector<double> qq(max_deg + 1), uu(max_deg + 1);
    std::vector<double> pp(max_deg + 1), qp(max_deg + 1), ps(max_deg + 1), qs(max_deg + 1);

    const double tau = cfg.tau_values ? cfg.tau_values[0] : cfg.tau;
    for (int col = 0; col < n; ++col) {
        x[col] = std::tanh(llr[col] / tau);
        y[col] = 0.0;
    }

    for (int step = 0; step < STEPS; ++step) {
        const double xi = cfg.xi_values ? cfg.xi_values[step] : cfg.xi;
        const double lambd = cfg.lambd_values ? cfg.lambd_values[step] : cfg.lambd;
        const double delta = cfg.delta_values ? cfg.delta_values[step] : cfg.delta;
        const double dt = cfg.dt_values ? cfg.dt_values[step] : cfg.dt;
        const double beta = cfg.beta_values ? cfg.beta_values[step] : cfg.beta;
        for (int col = 0; col < n; ++col) {
            if (cfg.mode == Mode::Dsb) {
                spins[col] = (x[col] >= 0.0) ? 1.0 : -1.0;
            } else {
                double value = std::min(1.0, std::max(-1.0, x[col]));
                if (std::fabs(value) < EPS) {
                    value = (value >= 0.0) ? EPS : -EPS;
                }
                spins[col] = value;
            }
        }
        std::fill(weighted.begin(), weighted.end(), 0.0);
        for (int row = 0; row < m; ++row) {
            const int start = starts[row];
            const int deg = starts[row + 1] - start;
            if (deg <= 0) continue;
            pref[0] = 1.0;
            for (int i = 0; i < deg; ++i) {
                pref[i + 1] = pref[i] * spins[cols[start + i]];
            }
            suff[deg] = 1.0;
            for (int i = deg - 1; i >= 0; --i) {
                suff[i] = suff[i + 1] * spins[cols[start + i]];
            }
            for (int i = 0; i < deg; ++i) {
                weighted[cols[start + i]] += pref[i] * suff[i + 1];
            }
        }
        if (use_reaction) {
            // sum_c sum_{j in c\i} (1-x_j^2) prod_{l in c\{i,j}} x_l^2 ; gradient of a term constant on +-1 spins
            std::fill(reaction.begin(), reaction.end(), 0.0);
            for (int row = 0; row < m; ++row) {
                const int start = starts[row];
                const int deg = starts[row + 1] - start;
                if (deg <= 0) continue;
                for (int t = 0; t < deg; ++t) {
                    const double s = spins[cols[start + t]];
                    qq[t] = s * s;
                    uu[t] = 1.0 - qq[t];
                }
                pp[0] = 1.0; qp[0] = 0.0;
                for (int t = 0; t < deg; ++t) {
                    pp[t + 1] = pp[t] * qq[t];
                    qp[t + 1] = qp[t] * qq[t] + pp[t] * uu[t];
                }
                ps[deg] = 1.0; qs[deg] = 0.0;
                for (int t = deg - 1; t >= 0; --t) {
                    ps[t] = ps[t + 1] * qq[t];
                    qs[t] = qs[t + 1] * qq[t] + ps[t + 1] * uu[t];
                }
                for (int t = 0; t < deg; ++t) {
                    reaction[cols[start + t]] += qp[t] * ps[t + 1] + pp[t] * qs[t + 1];
                }
            }
        }
        const double schedule = static_cast<double>(step) / (STEPS - 1);
        const double coupling = 0.5 * lambd;
        for (int col = 0; col < n; ++col) {
            const double scale = (use_scale && col >= parity_start) ? cfg.parity_scale : 1.0;
            if (use_scale) {
                force[col] = channel[col] + (coupling * scale) * weighted[col];
            } else {
                force[col] = channel[col] + coupling * weighted[col];
            }
            if (joint) {
                const int partner = col ^ 1;
                const double other = spins[partner];
                const double strength = (*pair_d)[col];
                force[col] += 2.0 * cfg.pair_scale * strength * other;
                if (cfg.pair_gamma != 0.0) {
                    force[col] -= 4.0 * cfg.pair_gamma * strength * strength
                                  * (1.0 - other * other) * spins[col];
                }
            }
            if (beta != 0.0) {
                double r = ((beta * (coupling * coupling)) * spins[col]) * reaction[col];
                if (use_scale) r = r * scale;
                force[col] = force[col] - r;
            }
            const double pump = -(delta - schedule) * x[col];
            const double y_new = y[col] + dt * (pump + xi * force[col]);
            const double x_new = x[col] + dt * y_new;
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
    x_out = x;
    return hard;
}

static std::vector<double> kspin_parity_weights(
    const std::vector<double>& llr, int k, int m, double g, double clip
) {
    std::vector<double> weights(static_cast<size_t>(m));
    for (int row = 0; row < m; ++row) {
        weights[static_cast<size_t>(row)] =
            g * clip * std::tanh(llr[static_cast<size_t>(k + row)] / clip);
    }
    return weights;
}

static std::vector<int> run_kspin_trajectory(
    const std::vector<double>& llr,
    int k,
    const std::vector<int>& info_starts,
    const std::vector<int>& info_cols,
    const TrajectoryConfig& cfg,
    std::vector<double>& x_out,
    const std::vector<double>* pair_linear,
    const std::vector<double>* pair_d
) {
    const int n = static_cast<int>(llr.size());
    const int m = n - k;
    const bool joint = (cfg.joint_pairs == 1 && pair_linear != nullptr && pair_d != nullptr);
    std::vector<double> x(static_cast<size_t>(k));
    std::vector<double> y(static_cast<size_t>(k), 0.0);
    std::vector<double> spins(static_cast<size_t>(k));
    std::vector<double> weighted(static_cast<size_t>(k), 0.0);
    std::vector<double> reaction(static_cast<size_t>(k), 0.0);
    std::vector<double> field(static_cast<size_t>(k), 0.0);
    std::vector<double> force(static_cast<size_t>(k), 0.0);
    std::vector<double> weights = kspin_parity_weights(llr, k, m, cfg.g, cfg.clip);

    int max_deg = 0;
    for (int row = 0; row < m; ++row) {
        max_deg = std::max(max_deg, info_starts[static_cast<size_t>(row + 1)]
                                    - info_starts[static_cast<size_t>(row)]);
    }
    std::vector<double> pref(static_cast<size_t>(max_deg + 1), 1.0);
    std::vector<double> suff(static_cast<size_t>(max_deg + 1), 1.0);
    std::vector<double> qq(static_cast<size_t>(max_deg + 1));
    std::vector<double> uu(static_cast<size_t>(max_deg + 1));
    std::vector<double> pp(static_cast<size_t>(max_deg + 1));
    std::vector<double> qp(static_cast<size_t>(max_deg + 1));
    std::vector<double> ps(static_cast<size_t>(max_deg + 1));
    std::vector<double> qs(static_cast<size_t>(max_deg + 1));

    const double tau = cfg.tau_values ? cfg.tau_values[0] : cfg.tau;
    for (int col = 0; col < k; ++col) {
        x[static_cast<size_t>(col)] = std::tanh(llr[static_cast<size_t>(col)] / tau);
    }

    for (int step = 0; step < STEPS; ++step) {
        for (int col = 0; col < k; ++col) {
            double value = std::min(1.0, std::max(-1.0, x[static_cast<size_t>(col)]));
            if (std::fabs(value) < EPS) {
                value = (value >= 0.0) ? EPS : -EPS;
            }
            spins[static_cast<size_t>(col)] = value;
        }
        for (int col = 0; col < k; ++col) {
            if (joint) {
                const int partner = col ^ 1;
                const double value = spins[static_cast<size_t>(col)];
                const double other = spins[static_cast<size_t>(partner)];
                const double d = (*pair_d)[static_cast<size_t>(col)];
                double value_field = (*pair_linear)[static_cast<size_t>(col)] + 2.0 * d * other;
                if (cfg.pair_gamma != 0.0) {
                    value_field -= 4.0 * cfg.pair_gamma * d * d
                                   * (1.0 - other * other) * value;
                }
                field[static_cast<size_t>(col)] = value_field;
            } else {
                field[static_cast<size_t>(col)] = llr[static_cast<size_t>(col)];
            }
        }

        std::fill(weighted.begin(), weighted.end(), 0.0);
        std::fill(reaction.begin(), reaction.end(), 0.0);
        const bool use_reaction = (cfg.beta != 0.0);
        for (int row = 0; row < m; ++row) {
            const int start = info_starts[static_cast<size_t>(row)];
            const int deg = info_starts[static_cast<size_t>(row + 1)] - start;
            if (deg <= 0) continue;
            pref[0] = 1.0;
            for (int index = 0; index < deg; ++index) {
                pref[static_cast<size_t>(index + 1)] =
                    pref[static_cast<size_t>(index)]
                    * spins[static_cast<size_t>(info_cols[static_cast<size_t>(start + index)])];
            }
            suff[static_cast<size_t>(deg)] = 1.0;
            for (int index = deg - 1; index >= 0; --index) {
                suff[static_cast<size_t>(index)] =
                    suff[static_cast<size_t>(index + 1)]
                    * spins[static_cast<size_t>(info_cols[static_cast<size_t>(start + index)])];
            }
            for (int index = 0; index < deg; ++index) {
                const int col = info_cols[static_cast<size_t>(start + index)];
                weighted[static_cast<size_t>(col)] +=
                    weights[static_cast<size_t>(row)] * pref[static_cast<size_t>(index)]
                    * suff[static_cast<size_t>(index + 1)];
            }
            if (!use_reaction) continue;
            for (int index = 0; index < deg; ++index) {
                const double value =
                    spins[static_cast<size_t>(info_cols[static_cast<size_t>(start + index)])];
                qq[static_cast<size_t>(index)] = value * value;
                uu[static_cast<size_t>(index)] = 1.0 - qq[static_cast<size_t>(index)];
            }
            pp[0] = 1.0; qp[0] = 0.0;
            for (int index = 0; index < deg; ++index) {
                pp[static_cast<size_t>(index + 1)] =
                    pp[static_cast<size_t>(index)] * qq[static_cast<size_t>(index)];
                qp[static_cast<size_t>(index + 1)] =
                    qp[static_cast<size_t>(index)] * qq[static_cast<size_t>(index)]
                    + pp[static_cast<size_t>(index)] * uu[static_cast<size_t>(index)];
            }
            ps[static_cast<size_t>(deg)] = 1.0; qs[static_cast<size_t>(deg)] = 0.0;
            for (int index = deg - 1; index >= 0; --index) {
                ps[static_cast<size_t>(index)] =
                    ps[static_cast<size_t>(index + 1)] * qq[static_cast<size_t>(index)];
                qs[static_cast<size_t>(index)] =
                    qs[static_cast<size_t>(index + 1)] * qq[static_cast<size_t>(index)]
                    + ps[static_cast<size_t>(index + 1)] * uu[static_cast<size_t>(index)];
            }
            const double w = weights[static_cast<size_t>(row)];
            for (int index = 0; index < deg; ++index) {
                const int col = info_cols[static_cast<size_t>(start + index)];
                const double value = spins[static_cast<size_t>(col)];
                reaction[static_cast<size_t>(col)] += cfg.beta * w * w * value
                    * (qp[static_cast<size_t>(index)] * ps[static_cast<size_t>(index + 1)]
                       + pp[static_cast<size_t>(index)] * qs[static_cast<size_t>(index + 1)]);
            }
        }

        const double xi = cfg.xi_values ? cfg.xi_values[step] : cfg.xi;
        const double delta = cfg.delta_values ? cfg.delta_values[step] : cfg.delta;
        const double dt = cfg.dt_values ? cfg.dt_values[step] : cfg.dt;
        const double schedule = static_cast<double>(step) / (STEPS - 1);
        for (int col = 0; col < k; ++col) {
            const double f = field[static_cast<size_t>(col)]
                             + weighted[static_cast<size_t>(col)]
                             - reaction[static_cast<size_t>(col)];
            const double pump = -(delta - schedule) * x[static_cast<size_t>(col)];
            const double y_new = y[static_cast<size_t>(col)] + dt * (pump + xi * f);
            const double x_new = x[static_cast<size_t>(col)] + dt * y_new;
            if (std::fabs(x_new) > 1.0) {
                x[static_cast<size_t>(col)] = (x_new >= 0.0) ? 1.0 : -1.0;
                y[static_cast<size_t>(col)] = 0.0;
            } else {
                x[static_cast<size_t>(col)] = x_new;
                y[static_cast<size_t>(col)] = y_new;
            }
        }
    }

    std::vector<int> hard(static_cast<size_t>(k));
    for (int col = 0; col < k; ++col) {
        hard[static_cast<size_t>(col)] = (x[static_cast<size_t>(col)] < 0.0) ? 1 : 0;
    }
    x_out = x;
    return hard;
}

static double kspin_energy_score(
    const std::vector<int>& info_hard,
    const std::vector<double>& llr,
    int k,
    const std::vector<int>& info_starts,
    const std::vector<int>& info_cols,
    const std::vector<double>& clipped_weights,
    const std::vector<double>* pair_linear,
    const std::vector<double>* pair_d,
    int select_unclipped
) {
    const int m = static_cast<int>(clipped_weights.size());
    std::vector<double> spins(static_cast<size_t>(k));
    for (int col = 0; col < k; ++col) {
        spins[static_cast<size_t>(col)] =
            (info_hard[static_cast<size_t>(col)] == 0) ? 1.0 : -1.0;
    }
    double value = 0.0;
    for (int col = 0; col < k; ++col) {
        double field = llr[static_cast<size_t>(col)];
        if (pair_linear != nullptr && pair_d != nullptr) {
            field = (*pair_linear)[static_cast<size_t>(col)]
                    + 2.0 * (*pair_d)[static_cast<size_t>(col)]
                    * spins[static_cast<size_t>(col ^ 1)];
        }
        value -= field * spins[static_cast<size_t>(col)];
    }
    for (int row = 0; row < m; ++row) {
        double product = 1.0;
        for (int pos = info_starts[static_cast<size_t>(row)];
             pos < info_starts[static_cast<size_t>(row + 1)]; ++pos) {
            product *= spins[static_cast<size_t>(info_cols[static_cast<size_t>(pos)])];
        }
        const double weight = select_unclipped
            ? llr[static_cast<size_t>(k + row)]
            : clipped_weights[static_cast<size_t>(row)];
        value -= weight * product;
    }
    return value;
}

static std::vector<int> encode_kspin_parity(
    const std::vector<int>& info_hard,
    int k,
    const std::vector<int>& info_starts,
    const std::vector<int>& info_cols
) {
    const int m = static_cast<int>(info_starts.size()) - 1;
    const int n = k + m;
    std::vector<int> hard(static_cast<size_t>(n));
    for (int col = 0; col < k; ++col) {
        hard[static_cast<size_t>(col)] = info_hard[static_cast<size_t>(col)];
    }
    for (int row = 0; row < m; ++row) {
        int parity = 0;
        for (int pos = info_starts[static_cast<size_t>(row)];
             pos < info_starts[static_cast<size_t>(row + 1)]; ++pos) {
            parity ^= info_hard[static_cast<size_t>(info_cols[static_cast<size_t>(pos)])];
        }
        hard[static_cast<size_t>(k + row)] = parity;
    }
    return hard;
}

static double energy_score(
    const std::vector<int>& hard,
    const std::vector<double>& channel,
    const std::vector<int>& starts,
    const std::vector<int>& cols,
    double lambd,
    const std::vector<double>* pair_d,
    double pair_scale
) {
    const int n = static_cast<int>(channel.size());
    const int m = static_cast<int>(starts.size()) - 1;
    std::vector<double> spins(n);
    for (int col = 0; col < n; ++col) {
        spins[col] = (hard[col] == 0) ? 1.0 : -1.0;
    }
    double channel_energy = 0.0;
    for (int col = 0; col < n; ++col) {
        channel_energy -= channel[col] * spins[col];
    }
    if (pair_d != nullptr) {
        for (int pair = 0; pair < n / 2; ++pair) {
            channel_energy -= (2.0 * pair_scale * (*pair_d)[2 * pair])
                              * spins[2 * pair] * spins[2 * pair + 1];
        }
    }
    double penalty = 0.0;
    for (int row = 0; row < m; ++row) {
        double product = 1.0;
        for (int pos = starts[row]; pos < starts[row + 1]; ++pos) {
            product *= spins[cols[pos]];
        }
        penalty += 1.0 - product;
    }
    return channel_energy + 0.5 * lambd * penalty;
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
    const int m = n - k;
    if (m < 0) {
        return 0;
    }

    std::vector<int> starts(static_cast<size_t>(m) + 1, 0);
    std::vector<int> cols;
    cols.reserve(static_cast<size_t>(m) * 8);
    std::vector<int> info_starts(static_cast<size_t>(m) + 1, 0);
    std::vector<int> info_cols;
    info_cols.reserve(static_cast<size_t>(m) * 8);
    for (int row = 0; row < m; ++row) {
        starts[row] = static_cast<int>(cols.size());
        info_starts[row] = static_cast<int>(info_cols.size());
        for (int col = 0; col < n; ++col) {
            int value = 0;
            std::cin >> value;
            if (value != 0) {
                cols.push_back(col);
                if (col < k) info_cols.push_back(col);
            }
        }
    }
    starts[m] = static_cast<int>(cols.size());
    info_starts[m] = static_cast<int>(info_cols.size());

    std::vector<std::vector<double>> llr(
        static_cast<size_t>(b), std::vector<double>(static_cast<size_t>(n))
    );
    for (int sample = 0; sample < b; ++sample) {
        for (int col = 0; col < n; ++col) {
            std::cin >> llr[static_cast<size_t>(sample)][col];
        }
    }
    int unused = 0;
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cin >> unused;
        }
    }
    static_cast<void>(unused);

    const CaseConfig* active = &FALLBACK;
    double best_diff = std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < static_cast<size_t>(__COUNT__); ++index) {
        const CaseConfig& candidate = CASES[index];
        if (candidate.n != n || candidate.k != k) {
            continue;
        }
        const double diff = std::fabs(candidate.snr - snr);
        if (diff < best_diff) {
            best_diff = diff;
            active = &candidate;
        }
    }

    const bool case_kspin = (active->mode == Mode::KSpin);
    bool case_joint = false;
    for (int index = 0; index < active->count; ++index) {
        if (active->trajectories[index].joint_pairs == 1) case_joint = true;
    }

    for (int sample = 0; sample < b; ++sample) {
        const std::vector<double>& row = llr[static_cast<size_t>(sample)];
        const std::vector<double>* row_channel = &row;
        std::vector<PairTables> frame_tables;
        bool frame_joint = case_joint;
        if (case_joint) {
            double bad_fraction = 0.0;
            frame_tables.push_back(
                build_pair_tables(row, active->snr, bad_fraction, case_kspin)
            );
            if (bad_fraction > 0.001) {
                frame_joint = false;
            } else {
                row_channel = &frame_tables[0].linear;
            }
        }
        const std::vector<double>* frame_d =
            frame_joint ? &frame_tables[0].d : nullptr;
        std::vector<int> decided;

        if (case_kspin) {
            const int k = active->k;
            std::vector<std::vector<int>> info_hards(
                static_cast<size_t>(active->count), std::vector<int>(static_cast<size_t>(k))
            );
            std::vector<std::vector<double>> softs(
                static_cast<size_t>(active->count), std::vector<double>(static_cast<size_t>(k))
            );
            std::vector<std::vector<double>> kweights(
                static_cast<size_t>(active->count), std::vector<double>(static_cast<size_t>(m))
            );
            if (active->count > 1) {
                std::vector<std::thread> workers;
                workers.reserve(static_cast<size_t>(active->count));
                for (int index = 0; index < active->count; ++index) {
                    workers.emplace_back([&, index]() {
                        const TrajectoryConfig& cfg = active->trajectories[static_cast<size_t>(index)];
                        kweights[static_cast<size_t>(index)] = kspin_parity_weights(
                            row, k, m, cfg.g, cfg.clip
                        );
                        const bool traj_joint = (cfg.joint_pairs == 1 && frame_joint);
                        info_hards[static_cast<size_t>(index)] = run_kspin_trajectory(
                            row, k, info_starts, info_cols, cfg,
                            softs[static_cast<size_t>(index)],
                            traj_joint ? &frame_tables[0].linear : nullptr,
                            traj_joint ? &frame_tables[0].d : nullptr
                        );
                    });
                }
                for (std::thread& worker : workers) {
                    worker.join();
                }
            } else {
                const TrajectoryConfig& cfg = active->trajectories[0];
                kweights[0] = kspin_parity_weights(row, k, m, cfg.g, cfg.clip);
                const bool traj_joint = (cfg.joint_pairs == 1 && frame_joint);
                info_hards[0] = run_kspin_trajectory(
                    row, k, info_starts, info_cols, cfg, softs[0],
                    traj_joint ? &frame_tables[0].linear : nullptr,
                    traj_joint ? &frame_tables[0].d : nullptr
                );
            }

            std::vector<int> selected_info = info_hards[0];
            if (active->combine == Combine::Energy) {
                double best_energy = kspin_energy_score(
                    selected_info, row, k, info_starts, info_cols, kweights[0],
                    active->trajectories[0].joint_pairs == 1 && frame_joint
                        ? &frame_tables[0].linear : nullptr,
                    active->trajectories[0].joint_pairs == 1 && frame_joint
                        ? &frame_tables[0].d : nullptr,
                    active->trajectories[0].select_unclipped
                );
                for (int index = 1; index < active->count; ++index) {
                    const TrajectoryConfig& cfg = active->trajectories[static_cast<size_t>(index)];
                    const bool traj_joint = (cfg.joint_pairs == 1 && frame_joint);
                    const double value = kspin_energy_score(
                        info_hards[static_cast<size_t>(index)], row, k,
                        info_starts, info_cols, kweights[static_cast<size_t>(index)],
                        traj_joint ? &frame_tables[0].linear : nullptr,
                        traj_joint ? &frame_tables[0].d : nullptr,
                        cfg.select_unclipped
                    );
                    if (value < best_energy) {
                        best_energy = value;
                        selected_info = info_hards[static_cast<size_t>(index)];
                    }
                }
            } else if (active->combine == Combine::Vote) {
                for (int col = 0; col < k; ++col) {
                    int ones = 0;
                    for (int index = 0; index < active->count; ++index) {
                        ones += info_hards[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    selected_info[static_cast<size_t>(col)] = (2 * ones >= active->count) ? 1 : 0;
                }
            } else {
                for (int col = 0; col < k; ++col) {
                    double total = 0.0;
                    for (int index = 0; index < active->count; ++index) {
                        total += softs[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    selected_info[static_cast<size_t>(col)] = (total < 0.0) ? 1 : 0;
                }
            }
            decided = encode_kspin_parity(selected_info, k, info_starts, info_cols);
        } else {
            std::vector<std::vector<int>> hards(static_cast<size_t>(active->count));
            std::vector<std::vector<double>> softs(static_cast<size_t>(active->count));
            std::vector<double> lambdas(static_cast<size_t>(active->count));
            if (active->count > 1) {
                std::vector<std::thread> workers;
                workers.reserve(static_cast<size_t>(active->count));
                for (int index = 0; index < active->count; ++index) {
                    workers.emplace_back([&, index]() {
                        hards[static_cast<size_t>(index)] = run_trajectory(
                            row, starts, cols, active->trajectories[index],
                            softs[static_cast<size_t>(index)], row_channel, frame_d
                        );
                        lambdas[static_cast<size_t>(index)] =
                            active->trajectories[index].energy_lambda;
                    });
                }
                for (std::thread& worker : workers) {
                    worker.join();
                }
            } else {
                hards[0] = run_trajectory(
                    row, starts, cols, active->trajectories[0], softs[0],
                    row_channel, frame_d
                );
                lambdas[0] = active->trajectories[0].energy_lambda;
            }

            decided = hards[0];
            if (active->combine == Combine::Energy) {
                double best_energy = energy_score(
                    decided, *row_channel, starts, cols, lambdas[0],
                    active->trajectories[0].joint_pairs == 1 ? frame_d : nullptr,
                    active->trajectories[0].pair_scale
                );
                for (int index = 1; index < active->count; ++index) {
                    const double value = energy_score(
                        hards[static_cast<size_t>(index)], *row_channel, starts, cols,
                        lambdas[static_cast<size_t>(index)],
                        active->trajectories[static_cast<size_t>(index)].joint_pairs == 1
                            ? frame_d : nullptr,
                        active->trajectories[static_cast<size_t>(index)].pair_scale
                    );
                    if (value < best_energy) {
                        best_energy = value;
                        decided = hards[static_cast<size_t>(index)];
                    }
                }
            } else if (active->combine == Combine::Vote) {
                for (int col = 0; col < n; ++col) {
                    int ones = 0;
                    for (int index = 0; index < active->count; ++index) {
                        ones += hards[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    decided[static_cast<size_t>(col)] = (2 * ones >= active->count) ? 1 : 0;
                }
            } else {
                for (int col = 0; col < n; ++col) {
                    double total = 0.0;
                    for (int index = 0; index < active->count; ++index) {
                        total += softs[static_cast<size_t>(index)][static_cast<size_t>(col)];
                    }
                    decided[static_cast<size_t>(col)] = (total < 0.0) ? 1 : 0;
                }
            }
        }

        for (int col = 0; col < n; ++col) {
            std::cout << decided[static_cast<size_t>(col)];
            std::cout << ((col + 1 == n) ? '\n' : ' ');
        }
    }
    return 0;
}
'''


def _fmt(value: float) -> str:
    return f"{float(value):.17g}"


def _finite(value: object, label: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return out


def _schedule_values(value: object, label: str) -> list[float]:
    """Validate and return per-step values from a scalar or 50-value list."""
    if isinstance(value, (list, tuple)):
        if len(value) != STEPS:
            raise ValueError(f"{label} must have {STEPS} values, got {len(value)}")
        result = [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]
    else:
        result = [_finite(value, label)] * STEPS
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{label} must contain only positive finite values")
    return result


def _beta_values(value: object, label: str) -> list[float]:
    """Validate and return per-step beta values; zero is allowed."""
    if isinstance(value, (list, tuple)):
        if len(value) != STEPS:
            raise ValueError(f"{label} must have {STEPS} values, got {len(value)}")
        result = [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]
    else:
        result = [_finite(value, label)] * STEPS
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def validate_case(case: object, label: str) -> dict:
    if not isinstance(case, dict):
        raise ValueError(f"{label}: case must be an object")
    mode = case.get("mode")
    combine = case.get("combine")
    if mode not in MODES:
        raise ValueError(f"{label}: invalid mode {mode!r}")
    if combine not in COMBINES:
        raise ValueError(f"{label}: invalid combine {combine!r}")
    if int(case.get("steps", -1)) != STEPS:
        raise ValueError(f"{label}: steps must be {STEPS}, got {case.get('steps')!r}")

    trajectories = case.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError(f"{label}: trajectories must be a list")
    if not 1 <= len(trajectories) <= MAX_TRAJECTORIES:
        raise ValueError(
            f"{label}: expected 1..{MAX_TRAJECTORIES} trajectories, got {len(trajectories)}"
        )

    dt = case.get("dt", 0.2)
    _schedule_values(dt, f"{label}: dt")

    schedules: list[dict[str, list[float]]] = []
    beta_schedules: list[list[float]] = []
    for index, traj in enumerate(trajectories):
        if not isinstance(traj, dict):
            raise ValueError(f"{label}: trajectory {index} must be an object")
        entry: dict[str, list[float]] = {}
        for key in PARAM_KEYS:
            entry[key] = _schedule_values(
                traj.get(key), f"{label}: trajectory {index} {key}"
            )
        entry["dt"] = _schedule_values(
            traj.get("dt", dt), f"{label}: trajectory {index} dt"
        )
        schedules.append(entry)
        beta_values = _beta_values(traj.get("beta", 0.0), f"{label}: trajectory {index} beta")
        beta_schedules.append(beta_values)
        parity_scale = _finite(traj.get("parity_scale", 1.0), f"{label}: trajectory {index} parity_scale")
        if mode == "kspin":
            if parity_scale != 1.0:
                raise ValueError(f"{label}: kspin does not use parity_scale (trajectory {index})")
            if any(value != 1.0 for value in entry["lambd"]):
                raise ValueError(f"{label}: kspin does not use lambd (trajectory {index})")
        elif mode != "bsb" and (any(value != 0.0 for value in beta_values) or parity_scale != 1.0):
            raise ValueError(f"{label}: beta/parity_scale require bsb mode (trajectory {index})")
        joint = int(traj.get("joint_pairs", 0))
        if joint not in (0, 1):
            raise ValueError(f"{label}: trajectory {index} joint_pairs must be 0 or 1")
        if joint:
            if mode not in ("bsb", "kspin"):
                raise ValueError(f"{label}: trajectory {index} joint_pairs requires bsb or kspin mode")
            if int(case.get("n", 0)) % 4 != 0:
                raise ValueError(f"{label}: trajectory {index} joint_pairs requires n divisible by 4")
            _finite(traj.get("pair_gamma", 0.0), f"{label}: trajectory {index} pair_gamma")
            _finite(traj.get("pair_scale", 1.0), f"{label}: trajectory {index} pair_scale")
        g_value = _finite(traj.get("g", 1.0), f"{label}: trajectory {index} g")
        clip_value = _finite(traj.get("clip", 1.0e30), f"{label}: trajectory {index} clip")
        if mode == "kspin" and clip_value <= 0.0:
            raise ValueError(f"{label}: kspin clip must be positive (trajectory {index})")
        select_unclipped = int(traj.get("select_unclipped", 0))
        if select_unclipped not in (0, 1):
            raise ValueError(f"{label}: select_unclipped must be 0 or 1 (trajectory {index})")
        if combine == "energy" and "energy_lambda" not in traj:
            raise ValueError(
                f"{label}: energy combine requires energy_lambda on trajectory {index}"
            )
        default_energy_lambda = entry["lambd"][0]
        energy_lambda = _finite(
            traj.get("energy_lambda", default_energy_lambda),
            f"{label}: trajectory {index} energy_lambda",
        )
        if combine == "energy" and energy_lambda < 0.0:
            raise ValueError(
                f"{label}: energy combine requires non-negative energy_lambda on trajectory {index}"
            )

    n = case.get("n")
    k = case.get("k")
    if not isinstance(n, int) or not isinstance(k, int) or n < 0 or k < 0 or k > n:
        raise ValueError(f"{label}: invalid n={n!r} k={k!r}")
    if mode == "kspin" and (k % 2 != 0 or (n - k) <= 0):
        raise ValueError(f"{label}: kspin requires even K and positive M")
    snr = _finite(case.get("snr"), f"{label}: snr")
    return {
        "n": n,
        "k": k,
        "snr": snr,
        "mode": mode,
        "combine": combine,
        "trajectories": trajectories,
        "dt": dt,
        "schedules": schedules,
        "beta_schedules": beta_schedules,
    }


def make_case_block(index: int, case: dict) -> tuple[str, str]:
    prefix = f"TRAJ_{index}"
    schedule_blocks: list[str] = []
    case_dt_values = _schedule_values(case["dt"], f"{prefix} dt")
    if isinstance(case["dt"], (list, tuple)):
        case_dt_name = f"{prefix}_CASE_DT"
        schedule_blocks.append(
            f"static const double {case_dt_name}[STEPS] = "
            f"{{{', '.join(_fmt(v) for v in case_dt_values)}}};"
        )
        case_dt_ptr = case_dt_name
    else:
        case_dt_ptr = "nullptr"

    rows = []
    for traj_index, (traj, schedule, beta_schedule) in enumerate(
        zip(case["trajectories"], case["schedules"], case["beta_schedules"])
    ):
        names: dict[str, str] = {}
        for key in PARAM_KEYS:
            raw = traj.get(key)
            if isinstance(raw, (list, tuple)):
                name = f"{prefix}_{traj_index}_{key.upper()}"
                schedule_blocks.append(
                    f"static const double {name}[STEPS] = "
                    f"{{{', '.join(_fmt(v) for v in schedule[key])}}};"
                )
                names[key] = name
            else:
                names[key] = "nullptr"

        if "dt" in traj and isinstance(traj["dt"], (list, tuple)):
            dt_name = f"{prefix}_{traj_index}_DT"
            schedule_blocks.append(
                f"static const double {dt_name}[STEPS] = "
                f"{{{', '.join(_fmt(v) for v in schedule['dt'])}}};"
            )
            dt_ptr = dt_name
        elif "dt" in traj:
            dt_ptr = "nullptr"
        else:
            dt_ptr = case_dt_ptr

        beta_raw = traj.get("beta", 0.0)
        if isinstance(beta_raw, (list, tuple)):
            beta_name = f"{prefix}_{traj_index}_BETA"
            schedule_blocks.append(
                f"static const double {beta_name}[STEPS] = "
                f"{{{', '.join(_fmt(v) for v in beta_schedule)}}};"
            )
            beta_ptr = beta_name
        else:
            beta_ptr = "nullptr"

        defaults = [_fmt(schedule[key][0]) for key in PARAM_KEYS]
        pointers = [names[key] for key in PARAM_KEYS]
        energy_lambda = _finite(
            traj.get("energy_lambda", schedule["lambd"][0]),
            f"{prefix} energy_lambda",
        )
        rows.append(
            "{"
            + ", ".join(
                defaults
                + [_fmt(energy_lambda), _fmt(schedule["dt"][0])]
                + pointers
                + [dt_ptr, MODES[case["mode"]]]
                + [_fmt(beta_schedule[0]),
                   _fmt(_finite(traj.get("parity_scale", 1.0), f"{prefix} parity_scale")),
                   beta_ptr,
                   str(int(traj.get("joint_pairs", 0))),
                   _fmt(_finite(traj.get("pair_gamma", 0.0), f"{prefix} pair_gamma")),
                   _fmt(_finite(traj.get("pair_scale", 1.0), f"{prefix} pair_scale")),
                    _fmt(_finite(traj.get("g", 1.0), f"{prefix} g")),
                    _fmt(_finite(traj.get("clip", 1.0e30), f"{prefix} clip")),
                    str(int(traj.get("select_unclipped", 0)))]
            )
            + "}"
        )

    array = (
        f"static const TrajectoryConfig {prefix}[{len(rows)}] = "
        f"{{{', '.join(rows)}}};"
    )
    entry = (
        f"{{{case['n']}, {case['k']}, {_fmt(case['snr'])}, "
        f"{COMBINES[case['combine']]}, {MODES[case['mode']]}, {len(rows)}, {prefix}}}"
    )
    return "\n".join(schedule_blocks + [array]), entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "fallback" not in manifest:
        raise ValueError("manifest must contain a 'fallback' entry")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest 'cases' must be a list")

    pieces: list[str] = []
    case_entries: list[str] = []
    for index, raw in enumerate([manifest["fallback"]] + cases):
        case = validate_case(raw, "fallback" if index == 0 else f"case {index}")
        array, entry = make_case_block(index, case)
        pieces.append(array)
        if index == 0:
            pieces.append(f"static const CaseConfig FALLBACK = {entry};")
        else:
            case_entries.append(entry)

    if case_entries:
        pieces.append(
            f"static const CaseConfig CASES[{len(case_entries)}] = "
            f"{{{', '.join(case_entries)}}};"
        )
        case_count = len(case_entries)
    else:
        pieces.append("static const CaseConfig CASES[1] = {FALLBACK};")
        case_count = 0

    source = (
        CPP_TEMPLATE.replace("__ARRAYS__", "\n\n".join(pieces))
        .replace("__COUNT__", str(case_count))
    )
    if "__" in source:
        raise ValueError("unreplaced template token remains")
    forbidden = ("belief propagation", "min-sum", "minsum", "bit-flip", "bitflip")
    lowered = source.lower()
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(f"generated source contains forbidden tokens: {found}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8", newline="\n")
    size = args.output.stat().st_size
    print(f"Wrote {args.output} ({size} bytes)")
    if size >= 512 * 1024:
        raise SystemExit("Generated submission exceeds 512K")


if __name__ == "__main__":
    main()
