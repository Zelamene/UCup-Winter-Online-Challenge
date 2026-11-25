# v7 — refined `R=2/3` cases candidate

Status: accepted. Judge score `1,600,231`.

Base: accepted v6 (`1,597,233`). This candidate changes only the three remaining `R=2/3` cases:

| Case | N | Judge score in v6 |
|---|---:|---:|
| 2 | 648 | 142,849 |
| 5 | 1296 | 141,894 |
| 8 | 1944 | 142,582 |

Case 11 (`N=3888, R=2/3`) was already refined in v6 and is unchanged here.

## Method

For each target case:

- 180-configuration grid over `tau 1.4–2.5`, `xi 0.45–0.55`, `lambda 5–8`, `delta 2.25–2.75`
- three local-search rounds
- greedy six-trajectory portfolio
- separate coordinate tuning of energy-selection lambdas

Synthetic optimized/accepted BER ratios versus the current accepted decoder:

- `N=648`: `1.525`
- `N=1296`: `1.374`
- `N=1944`: `1.385`

## Validation

- Public output bit-identical to accepted v6
- Source size: `82,292` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`
- Python reference decoder matches C++ on synthetic inputs for all three changed cases (`0` bit mismatches)

Judge outcome: cases 2, 5, and 8 improved by `+1,420`, `+1,313`, and `+265`. All other cases were unchanged. Total score rose from `1,597,233` to `1,600,231`.
