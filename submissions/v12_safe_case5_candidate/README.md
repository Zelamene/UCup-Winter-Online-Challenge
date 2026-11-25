# v12 — safe case-5-only candidate

Status: prepared for judge submission, not yet submitted.

Base: accepted v9 (`1,606,841`). v12 changes only case 5 (`N=1296`, `R=2/3`, `SNR=9.0`), using the exact configuration that produced `+371` twice on the judge in v10 and v11.

Expected score if the judge result transfers a third time: `1,607,212`.

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

If submitted, archive the judge result here immediately and mark it accepted or no-gain based on the score table.
