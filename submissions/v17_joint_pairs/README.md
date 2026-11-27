# v17 — Phase 14 joint Gray-pair candidate

Accepted submission. The full Phase 14 gate passed, and the judge returned `2,038,794` (`+14,797` over v16).

Base: accepted v16 (`submissions/v16_tap_case_tuning/`), score `2,023,997`.

Changed cases:

- case 4: `N=1296, K=648, SNR=6.0` (`R=1/2`)
- case 7: `N=1944, K=972, SNR=6.0` (`R=1/2`)
- case 10: `N=3888, K=1944, SNR=6.0` (`R=1/2`)

All other cases, including public case 9, are byte-identical to v16. The only structural change is exact Gray-pair coupling in the three changed `R=1/2` cases: recovered amplitude, pair channel table, mean-field pair field, TAP/Onsager reaction, and matching energy combine. No per-edge messages are stored and the Ising solution on `±1` spins is unchanged.

A3 joint gains versus v16, 500 frames per validation seed:

| Case | Seed 2024 | Seed 31337 | Gate |
|---:|---:|---:|---|
| 4 | `+2,508` (CI low `+713`) | `+4,973` (CI low `+3,002`) | pass |
| 7 | `+4,913` (CI low `+2,907`) | `+7,677` (CI low `+5,158`) | pass |
| 10 | `+9,493` (CI low `+7,379`) | `+7,099` (CI low `+4,971`) | pass |

Case 1 was searched twice and failed, so it remains v16-identical. Case 10's first two-cycle search timed out; the one-cycle artifact above cleared the gate with no parameter on a bound.

Fresh seed `6060`, 300 frames, versus v16: case 4 `+4,299` (CI `[+2,227, +6,377]`), case 7 `+5,481` (CI `[+2,441, +8,739]`), case 10 `+8,573` (CI `[+5,753, +11,505]`); all nine unchanged cases are byte-identical. Total local fresh-seed gain is `+18,353`, roughly `+17,380` after the Phase 13 judge-slope calibration.

Full A4 gate: `artifacts/phase14/final_gate.json`, `all_passed=true`. Source is `28,829` bytes (SHA-256 `A80AEF906B31A97CDC061667AF6A13CDB18F136572C85B4B09242C8BE6BB76D6`); strict C++17 compile, validator, offline isolation, public case-9 identity, three-way bit identity, consistency guard, fresh-seed gate and `3×` runtime guard all pass. Worst runtime ratio is `1.48×` v16 on case 4.

Judge result: `Accepted` at `2,038,794`, `+14,797` over accepted v16 (`judge_result.json`). This archive is the base for Phase 15; no later candidate should be built from v16.
