# v11 — second-pass `R=3/4` refinement candidate

Status: candidate for judge submission; not yet accepted.

Base: v10 candidate (`submission_v10.cpp`). v11 changes only case 6:

- `N=1296`
- `K=972`
- `SNR` dispatch `10.5`

## Case selection

| Case | Tuning-seed ratio | Cross-seed | Decision |
|---|---:|---:|---|
| 3 `N=648` | `1.069` | `1.004`, `0.996` | rejected |
| 6 `N=1296` | `1.021` | `1.022`, `1.024` | included |
| 9 `N=1944` | `1.027` | `1.005`, `1.000` | rejected |
| 12 `N=3888` | `1.012` | `1.003`, `1.011`, `1.002` | rejected |

Only case 6 remained consistently better on independent synthetic seeds.

## Method

The accepted-portfolio refiner searched a 189-configuration grid over `tau 1.3–4`, `xi 0.4–0.5`, `lambda 5–7`, `delta 2–2.5`, followed by local search, six-trajectory portfolio replacement, and energy-lambda tuning. Candidate portfolios were compared directly against the accepted manifest portfolio.

## Validation

- Public output bit-identical to v10
- Source size: `82,703` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`
- Python/C++ parity on synthetic `N=1296, R=3/4`: `0` bit mismatches

Manifest diff confirms v11 differs from v10 only on case 6. If v10 is not accepted, v11 must be rebuilt on the previous accepted base; the case-6 configuration is identical in v9 and v10.
