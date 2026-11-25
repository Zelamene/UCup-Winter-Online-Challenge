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

MODES = {"dsb": "Mode::Dsb", "bsb": "Mode::Bsb"}
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
#include <utility>
#include <vector>

using std::size_t;

static constexpr int STEPS = 50;
static constexpr double EPS = 1e-12;

enum class Mode { Dsb, Bsb };
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
};

struct CaseConfig {
    int n;
    int k;
    double snr;
    Combine combine;
    int count;
    const TrajectoryConfig* trajectories;
};

__ARRAYS__

static std::vector<int> run_trajectory(
    const std::vector<double>& llr,
    const std::vector<int>& starts,
    const std::vector<int>& cols,
    const TrajectoryConfig& cfg,
    std::vector<double>& x_out
) {
    const int n = static_cast<int>(llr.size());
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
        const double schedule = static_cast<double>(step) / (STEPS - 1);
        for (int col = 0; col < n; ++col) {
            force[col] = llr[col] + (0.5 * lambd) * weighted[col];
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

static double energy_score(
    const std::vector<int>& hard,
    const std::vector<double>& llr,
    const std::vector<int>& starts,
    const std::vector<int>& cols,
    double lambd
) {
    const int n = static_cast<int>(llr.size());
    const int m = static_cast<int>(starts.size()) - 1;
    std::vector<double> spins(n);
    for (int col = 0; col < n; ++col) {
        spins[col] = (hard[col] == 0) ? 1.0 : -1.0;
    }
    double channel = 0.0;
    for (int col = 0; col < n; ++col) {
        channel -= llr[col] * spins[col];
    }
    double penalty = 0.0;
    for (int row = 0; row < m; ++row) {
        double product = 1.0;
        for (int pos = starts[row]; pos < starts[row + 1]; ++pos) {
            product *= spins[cols[pos]];
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
    const int m = n - k;
    if (m < 0) {
        return 0;
    }

    std::vector<int> starts(static_cast<size_t>(m) + 1, 0);
    std::vector<int> cols;
    cols.reserve(static_cast<size_t>(m) * 8);
    for (int row = 0; row < m; ++row) {
        starts[row] = static_cast<int>(cols.size());
        for (int col = 0; col < n; ++col) {
            int value = 0;
            std::cin >> value;
            if (value != 0) {
                cols.push_back(col);
            }
        }
    }
    starts[m] = static_cast<int>(cols.size());

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
        if (diff < 0.25 && diff < best_diff) {
            best_diff = diff;
            active = &candidate;
        }
    }

    for (int sample = 0; sample < b; ++sample) {
        const std::vector<double>& row = llr[static_cast<size_t>(sample)];
        std::vector<std::vector<int>> hards;
        std::vector<std::vector<double>> softs;
        hards.reserve(static_cast<size_t>(active->count));
        softs.reserve(static_cast<size_t>(active->count));
        std::vector<double> lambdas;
        lambdas.reserve(static_cast<size_t>(active->count));

        for (int index = 0; index < active->count; ++index) {
            std::vector<double> soft;
            hards.push_back(run_trajectory(
                row, starts, cols, active->trajectories[index], soft
            ));
            softs.push_back(std::move(soft));
            lambdas.push_back(active->trajectories[index].energy_lambda);
        }

        std::vector<int> decided = hards[0];
        if (active->combine == Combine::Energy) {
            double best_energy = energy_score(
                decided, row, starts, cols, lambdas[0]
            );
            for (int index = 1; index < active->count; ++index) {
                const double value = energy_score(
                    hards[static_cast<size_t>(index)], row, starts, cols,
                    lambdas[static_cast<size_t>(index)]
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
    for traj_index, (traj, schedule) in enumerate(
        zip(case["trajectories"], case["schedules"])
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
            )
            + "}"
        )

    array = (
        f"static const TrajectoryConfig {prefix}[{len(rows)}] = "
        f"{{{', '.join(rows)}}};"
    )
    entry = (
        f"{{{case['n']}, {case['k']}, {_fmt(case['snr'])}, "
        f"{COMBINES[case['combine']]}, {len(rows)}, {prefix}}}"
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
