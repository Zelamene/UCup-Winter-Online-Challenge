# v19 — Local Phase 16 K-spin candidate

Local candidate only. It has **not** been submitted and has no judge score. Accepted v18b remains the best submission and the fallback.

Base: accepted v18b (`submissions/v18b_greedy_portfolio/`), judge score `2,051,900`.

Changed cases: 1, 4, 7, 10. All are rate `1/2`. Cases 2, 3, 5, 6, 8, 9, 11, 12 are byte-identical to v18b, including public case 9.

## Decoder

For systematic `H=[A|I]`, each parity spin is replaced by the product of its information-support spins. The remaining degree-six spin system has only valid codewords as states, so the 50-step QAIA dynamics run on the K information spins and the parity bits are re-encoded at the output. The field uses the same exact Gray-pair channel tables as v18: linear terms `2a_P` / `2b_P`, pair term `D_P·x_p`, and TAP reaction `−γ·D_P²·(1−x_p²)·x_i`. Parity weights are soft-clipped `g·clip·tanh(L_{k+c}/clip)`; selection energy is `−Σ_i ℓ_i s_i − Σ_c w_c ∏_{j∈N(c)} s_j`. There is no per-edge memory and no forbidden post-processing.

Common settings: `xi=1.0`, `delta=3.441`, `dt=0.47`, `beta=0.317`, `g=0.98`, `clip=2.8`, `joint_pairs=1`, `select_unclipped=0`. Cases 1, 4 and 10 use one trajectory with `tau=2.976`, `pair_gamma=0`. Case 7 uses a three-trajectory energy portfolio with `(tau,pair_gamma)` of `(2.976,0)`, `(3.5,0)`, `(2.976,0.1)`.

## Validation

`final_gate.json` reports `all_passed=true` with a 26-test suite, strict C++17 compile, validator, offline isolation, unchanged public case-9 SHA-256 `1BBF2BBAA56E6AD871BB6FFE7F0AC4D501C31F3CE7A684195CAFD8F1AAA8723A`, and scalar/batched/C++ bit identity on all changed cases.

| Seed | Case 1 | Case 4 | Case 7 | Case 10 | Total |
|---|---:|---:|---:|---:|---:|
| 2024 | `+6,549.2` `[+3,502.8,+9,600.7]` | `+9,663.2` `[+6,906.0,+12,460.0]` | `+3,938.1` `[+966.5,+6,990.9]` | `+8,282.9` `[+5,495.0,+11,241.4]` | `+28,433.4` |
| 31337 | `+6,705.7` `[+4,191.7,+9,630.4]` | `+8,979.3` `[+6,260.7,+11,886.4]` | `+8,736.5` `[+5,799.2,+11,983.6]` | `+10,399.9` `[+7,093.8,+13,623.9]` | `+34,821.4` |

Fresh seed `6060`, 300 frames: case 1 `+6,960.2` `[+3,951.3,+10,384.5]`, case 4 `+7,079.5` `[+3,606.7,+10,654.7]`, case 7 `+10,666.3` `[+6,983.2,+14,654.5]`, case 10 `+9,092.2` `[+5,103.3,+13,521.3]`. All 95% CIs are paired frame bootstrap with 1,000 resamples.

Runtime judge estimate worst case: `13.403 s` on case 10, under the `60 s` local cap and the `120 s` contest limit. Source size: `46,677` bytes.

Source SHA-256: `9A940D50831F1A2FAAE7E3BF9435E588259C5868BDC784EAF0FDE1516F2696B7`.

Status: prepared, gated locally, **pending explicit submission approval**.
