# v8 — public-tuned `R=3/4` case candidate

Status: accepted. Judge score `1,601,303`.

Base: accepted v7 (`1,600,231`). Only case 9 is changed:

- `N=1944`
- `K=1458`
- `SNR` dispatch `10.5`

## Method

Case 9 was tuned directly on the public 500-frame input using the provided answers, split into:

- frames `0–249` for candidate selection
- frames `250–499` for validation

The search covered fixed dSB configurations and a six-candidate energy portfolio, including the existing learned trajectory. Energy-selection lambdas were tuned separately from decoder lambdas.

Public BER:

- v7: `0.025788752`
- v8: `0.024784636`

Both halves improved:

- train: `0.025997257 → 0.024603567`
- val: `0.025580247 → 0.024965706`

## Validation

- C++ public BER matches the Python tuner exactly
- Full 500-frame Python/C++ parity: `0` bit mismatches
- Source size: `82,352` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`

Judge outcome: case 9 rose from `166,013` to `167,085` (`+1,072`). All other cases were unchanged. Total score rose from `1,600,231` to `1,601,303`.
