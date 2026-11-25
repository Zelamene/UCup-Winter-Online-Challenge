# v11 — rebased accepted candidate, no score improvement

Status: accepted by the judge at `1,606,575`, but below the best accepted v9 by `266`. v9 remains the best source.

Base: accepted v9 (`1,606,841`), **not** the below-best v10. v11 combined two individually supported changes:

| Case | N | R | Evidence before submission | Judge delta vs v9 |
|---|---:|---|---|---:|
| 5 | 1296 | 2/3 | v10 judge delta `+371`; synthetic ratios `1.085 / 1.038 / 1.032` | `+371` |
| 6 | 1296 | 3/4 | synthetic ratios `1.021 / 1.022 / 1.024` | `-637` |

The below-best v10 case-11 change (`N=3888, R=2/3`) was **not** included. Removing it restored case 11 from `141,113` in v10 back to v9's `142,852`.

## Key findings

- The v9 rebase and case-5 salvage worked: case 5 gained `+371` for the second time.
- Case 6 did not transfer: `167,094 → 166,457`, a `-637` regression despite consistent synthetic ratios around `1.02`.
- Net change was `-266`; the reliable case-5 gain was too small to absorb the case-6 loss.
- Public output was bit-identical to v9, so public BER could not detect either changed case.
- Runtime was not a limiting factor: `27,613ms` total, `5,784ms` peak.

## Validation before submission

- Public output bit-identical to accepted v9
- Source size: `78,898` bytes
- Max guesses: `6`
- 50 dSB iterations only
- Clean C++17 compile with `-Wall -Wextra -pedantic`
- Python/C++ parity: `0` bit mismatches on synthetic inputs for both changed cases
- Manifest diff confirmed only cases 5 and 6 differed from v9

No v12 was built or submitted from this result. The v11 case-6 configuration should be discarded.
