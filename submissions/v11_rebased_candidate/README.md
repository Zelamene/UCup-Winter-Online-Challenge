# v11 — rebased salvage candidate

Status: candidate for judge submission; not yet accepted.

Base: accepted v9 (`1,606,841`), **not** the rejected v10. v11 combines the two individually supported changes from the last refinement round:

| Case | N | R | Evidence | Expected direction |
|---|---:|---|---|---|
| 5 | 1296 | 2/3 | v10 judge delta `+371`; synthetic ratios `1.085 / 1.038 / 1.032` | improve |
| 6 | 1296 | 3/4 | synthetic ratios `1.021 / 1.022 / 1.024` | improve |

The rejected v10 case-11 change (`N=3888, R=2/3`) is **not** included. It regressed by `-1,739` on the judge despite positive synthetic checks.

## Why this base

v10 scored `1,605,473`, down `1,368` from v9. Its case-5 change was positive and case-11 change was negative. Rebuilding on v9 removes the regression while keeping the case-5 gain.

The case-6 configuration is identical in v9 and v10, so the v11 high-rate refinement is unaffected by the rebase.

## Validation

- Public output bit-identical to accepted v9
- Source size: `78,898` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`
- Python/C++ parity: `0` bit mismatches on synthetic inputs for both changed cases
- Manifest diff confirms only cases 5 and 6 differ from v9

If v11 is accepted, archive it immediately and record the per-case judge deltas.
