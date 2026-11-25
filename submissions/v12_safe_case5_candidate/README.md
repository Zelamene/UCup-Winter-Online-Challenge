# v12 — accepted case-5-only improvement

Status: **accepted judge submission and current best**.

Judge score: `1,607,212` (`+371` over v9).

Base: accepted v9 (`1,606,841`). v12 changes only case 5 (`N=1296`, `R=2/3`, `SNR=9.0`), using the exact configuration that produced `+371` in both v10 and v11. The third judge run reproduced the gain exactly.

This candidate does **not** include:

- v10 case 11 (`-1,739`)
- v11 case 6 (`-637`)
- any synthetic-only refinement
- any structural decoder change

## Validation

- Public output bit-identical to accepted v9
- Source size: `78,801` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`
- Python/C++ parity on synthetic `N=1296, R=2/3`: `0` bit mismatches
- Manifest diff vs v9: only case 5 changed

## Judge result

- Total runtime: `26,341ms`
- Peak case runtime: `5,578ms` on case 11
- Peak memory: `19.6 MiB`
- All 12 cases accepted

Full per-case data is in `judge_result.json`.
