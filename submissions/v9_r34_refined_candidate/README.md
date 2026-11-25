# v9 — refined `R=3/4` cases candidate

Status: candidate for judge submission; not yet accepted.

Base: v8 candidate (`submission_v8.cpp`). v9 changes only the three `R=3/4` cases that were not directly tuned on public data:

| Case | N | Synthetic ratio vs accepted |
|---|---:|---:|
| 3 | 648 | `1.201` |
| 6 | 1296 | `1.178` |
| 12 | 3888 | `1.119` |

Case 9 stays at the public-tuned v8 configuration. All `R=1/2` and `R=2/3` cases are unchanged.

## Method

For each target case:

- 216-configuration grid over `tau 1.4–4`, `xi 0.4–0.5`, `lambda 5–8`, `delta 2–2.5`
- three local-search rounds
- greedy six-trajectory portfolio
- separate energy-lambda tuning

## Validation

- Public output bit-identical to v8
- Python/C++ parity on synthetic inputs for all three changed cases: `0` bit mismatches
- Source size: `78,782` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`

If v8 is not accepted, v9 must be rebuilt on the previous accepted base rather than submitted as-is.
