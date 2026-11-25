# v10 — second-pass `R=2/3` refinement candidate

Status: rejected. Judge score `1,605,473`, down `1,368` from accepted v9.

Base: accepted v9 (`1,606,841`). This candidate changes only two cases:

| Case | N | R | Tuning seed ratio | Cross-seed ratios |
|---|---:|---|---:|---|
| 5 | 1296 | 2/3 | `1.085` | `1.038`, `1.032` |
| 11 | 3888 | 2/3 | `1.062` | `1.025`, `1.027` |

`N=648` (`1.010`, cross-seed `0.998`) and `N=1944` (`1.035`, cross-seed `1.007`, `0.986`) were rejected as non-robust and are unchanged.

## Method

A new accepted-portfolio refiner compares candidate portfolios directly against the currently accepted manifest portfolio on the same synthetic validation cases. Each target case used:

- 252-configuration grid over `tau 1.3–2.0`, `xi 0.45–0.55`, `lambda 5–7`, `delta 2.0–2.75`
- two local-search rounds
- six-trajectory portfolio search seeded from the accepted portfolio
- coordinate tuning of energy-selection lambdas

Cross-seed checks used independent synthetic cases at seeds `1234` and `2024`.

## Validation

- Public output bit-identical to accepted v9
- Source size: `82,606` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`
- Python/C++ parity on synthetic inputs for both changed cases: `0` bit mismatches

Judge outcome: case 5 improved `143,207 → 143,578` (`+371`), but case 11 regressed `142,852 → 141,113` (`-1,739`). Net change: `-1,368`. The cross-seed-positive `N=3888` change did not transfer; this is now the clearest example of synthetic energy-lambda tuning failing to predict judge behavior.

Manifest diff confirms v10 differs from accepted v9 only on cases 5 and 11.
