# v15 — TAP-corrected bSB on all 12 cases

Base: v14 (accepted, `1,847,842`). Status: **candidate, not yet submitted.**

## What changed

Every case runs one bSB trajectory with an added **Onsager (TAP) reaction term** and a **parity-column check scale**:

```
force_i = r_i + J*h_i*sum_c prod_{j in c\i} x_j  -  beta*J^2*h_i*x_i*sum_c e_{c,i}
e_{c,i}  = sum_{j in c\i} (1 - x_j^2) * prod_{l in c\{i,j}} x_l^2
J = lambda/2,   h_i = parity_scale for parity columns (last n-k), 1 otherwise
```

The reaction term is exactly `-(beta*J^2/2) * dG/dx_i` (with `h_i = 1`), where

```
G(x) = sum_c [ sum_{j in c} prod_{l in c\j} x_l^2  -  (d_c - 1) * prod_{l in c} x_l^2 ]
```

`G` equals 1 for every check on ±1 spins, so it adds a constant to Eq. (1) on binary states. The Ising problem and its ground states are unchanged; only the continuous landscape that bSB moves through is reshaped. This was checked numerically (finite-difference gradient, and `G = 1` on random ±1 vectors). No per-edge state carries over between steps, so this is not BP, min-sum or bit-flip.

Why it helps: plain soft-spin SB behaves like a naive mean-field decoder, which settles at poor fixed points. At R=1/2, v14's output still violated about 59 of 648 checks, even though the true codeword had lower energy in 94–100% of frames. The reaction term is the standard correction for that.

## Parameters (one setting per rate, tuned only at N=1296, 100 frames, seed 777)

| Rate | tau | xi | lambda | delta | dt | beta | parity_scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/2 | 2.295 | 1.264 | 5.95 | 3.72 | 0.236 | 0.27 | 0.427 |
| 2/3 | 5.184 | 0.60 | 9.375 | 3.0 | 0.40 | 0.16 | 0.6375 |
| 3/4 | 5.0 | 0.40 | 10.0 | 2.25 | 0.20 | 0.12 | 0.872 |

## Validation

- Strict compile (`-std=c++17 -O2 -Wall -Wextra -pedantic`), `19,752` bytes, `STEPS = 50`, one trajectory per case; `validate_submission_v3.py` passes.
- Python reference decoder, batched tuner and C++ agree bit for bit on the new term.
- The patched exporter reproduces the submitted v14 binary exactly on the v14 manifest (public input plus three bench cases).
- **Fresh seed 6060 (300 frames per case), paired against the v14 binary:** total `+174,193`, and every case's 95% bootstrap CI is above zero. See `artifacts/v15_fresh_seed_6060.json`.

| Case | N | K | v14 | v15 | Gain | 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 648 | 324 | 111,278 | 130,175 | +18,897 | [+15,443, +22,696] |
| 2 | 648 | 432 | 166,545 | 181,049 | +14,504 | [+11,736, +17,572] |
| 3 | 648 | 486 | 180,965 | 184,092 | +3,127 | [+1,358, +4,770] |
| 4 | 1296 | 648 | 113,206 | 135,103 | +21,897 | [+18,755, +25,336] |
| 5 | 1296 | 864 | 167,288 | 181,517 | +14,230 | [+11,627, +17,027] |
| 6 | 1296 | 972 | 180,440 | 184,154 | +3,714 | [+2,457, +5,144] |
| 7 | 1944 | 972 | 111,902 | 139,375 | +27,473 | [+23,626, +31,559] |
| 8 | 1944 | 1296 | 167,825 | 182,787 | +14,962 | [+12,719, +17,086] |
| 9 | 1944 | 1458 | 181,112 | 184,968 | +3,855 | [+2,765, +4,888] |
| 10 | 3888 | 1944 | 113,880 | 147,592 | +33,712 | [+30,224, +37,689] |
| 11 | 3888 | 2592 | 170,191 | 184,038 | +13,846 | [+12,406, +15,493] |
| 12 | 3888 | 2916 | 180,965 | 184,940 | +3,975 | [+3,270, +4,738] |

- **Public input (case 9):** BER `0.019217 → 0.017494` (+4,080).
- **Runtime:** at most 2.3 s for 300 frames at N=3888 locally, roughly 1.5× v14.

**Expected judge score: ≈ 2.02M.** The bench put v14 about 2,200 below its judge score, and v15 gains +174k on the bench.

## Next

Retune each case with Stage A/B, adding `beta` and `parity_scale` to the search. `tools/bsb_vectorized.py` now accepts both. The R=1/2 gain grows with N (+19k at 648 to +34k at 3888), so per-case tuning, especially at large N, should add more.
