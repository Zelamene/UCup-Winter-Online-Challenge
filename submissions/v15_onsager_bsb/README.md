# v15 — Onsager-corrected bSB

Accepted by the judge at `2,015,823`, a `+167,981` gain over accepted v14
(`1,847,842`). This is the current best accepted version and the base for all
future candidates.

## Contents

- `submission.cpp` — single-file C++17 decoder, `19,752` bytes.
- `manifest.json` — v3 manifest with two added trajectory fields, `beta` and
  `parity_scale`.
- `validation.json` — source-size, dispatch, trajectory-count, public-output,
  and forbidden-token gates.
- `judge_result.json` — accepted score, per-case scores, deltas vs v14, and
  judge runtime/memory.

## Architecture

All 12 cases keep one v14-style scheduled bSB trajectory. The change is a
reaction term in the continuous spin force:

- `beta = 0.27`, `parity_scale = 0.42687` for `R=1/2`;
- `beta = 0.16`, `parity_scale = 0.6375` for `R=2/3`;
- `beta = 0.12`, `parity_scale = 0.871875` for `R=3/4`.

There are still 50 steps, one trajectory per case, exact `(N, K)` dispatch with
nearest configured SNR, and no external decoding stage.

## Why this is within the QAIA family

The Onsager/TAP term is the gradient of an added polynomial that evaluates to a
constant on `±1` spins. It therefore does not change the Ising problem or its
solutions; it only changes the continuous landscape that ballistic SB moves
through. The identity was checked numerically.

The implementation uses only current soft spins and current check products. It
stores no per-edge message between steps, so it is not BP, min-sum, or
bit-flip. It also uses no ground-truth bits at inference and no forbidden
post-processing.

The term does come from the same statistical-physics tradition as BP, so the
QAIA classification is ultimately a judgement call for the organizers. That
argument is recorded here deliberately: the decoder remains SB/bSB dynamics on
an unchanged Ising Hamiltonian.

## Validation before submission

- Fresh seed `6060`, 300 frames per case, paired against the v14 binary: all
  12 cases improved and all 95% bootstrap intervals had lower bound above zero.
- `R=1/2`: `+18.9k` to `+33.7k`, with larger gains on longer codes.
- `R=2/3`: about `+14k`.
- `R=3/4`: about `+3k` to `+4k`.
- Public case 9 BER improved from `0.019216735` to `0.017493827`.
- Python reference decoder, batched tuner, and C++ binary agreed bit for bit.
- The patched exporter reproduces the submitted v14 binary exactly when `beta`
  and `parity_scale` are left at their defaults.

## Judge result

| Case | `N` | `R` | v15 score | delta vs v14 |
|---:|---:|---:|---:|---:|
| 1 | 648 | 1/2 | 129,979 | +15,969 |
| 2 | 648 | 2/3 | 177,394 | +11,279 |
| 3 | 648 | 3/4 | 185,451 | +3,334 |
| 4 | 1296 | 1/2 | 140,530 | +25,054 |
| 5 | 1296 | 2/3 | 179,802 | +12,666 |
| 6 | 1296 | 3/4 | 183,299 | +3,984 |
| 7 | 1944 | 1/2 | 137,314 | +25,862 |
| 8 | 1944 | 2/3 | 184,164 | +15,274 |
| 9 | 1944 | 3/4 | 184,485 | +4,180 |
| 10 | 3888 | 1/2 | 145,794 | +32,501 |
| 11 | 3888 | 2/3 | 182,587 | +14,142 |
| 12 | 3888 | 3/4 | 185,024 | +3,736 |

Judge runtime was `384–4,279 ms` per case and memory `3.5–19.4 MiB`, both well
inside the limits. Runtime is roughly `1.5×` v14.

Do not modify this folder when preparing a new candidate. Copy its manifest into
a new artifact and build a separate submission folder instead.
