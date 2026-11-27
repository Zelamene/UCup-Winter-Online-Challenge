# LDPC QAIA submission — strategy and knowledge base

Objective: increase the judge score beyond the current accepted best while keeping every submission valid under the contest rules.

## Rules and hard constraints

- Decoder family: QAIA only — dSB / SB / SimCIM / LQA and variants.
- Forbidden as decoder or post-processing: BP, min-sum, bit-flip, derivatives.
- At most `50` QAIA iterations.
- At most `6` initial guesses / trajectories per code.
- No ground-truth bits at inference. Offline training and candidate selection may use labels.
- Source under `512 KiB`; runtime under `2 minutes` per case.
- Cases: `N ∈ {648,1296,1944,3888}`, `R ∈ {1/2,2/3,3/4}`; SNR thresholds `6.0 / 9.0 / 10.5 dB`.
- Score is higher-is-better and proportional to `-10 lg BER` summed over cases.

## Current state

| Item | Value |
|---|---:|
| Best accepted | v17, `2,038,794` |
| Last submitted | v17, `2,038,794` (`+14,797` vs v16) |
| Local candidate | none; Phase 15 screens are compared against cached v17 |
| Active decision | use v17 as the base for all future candidates |
| Public input dispatch | `N=1944, R=3/4, SNR≈10.3` (nearest configured `10.5`) |
| Public BER, v8–v13c | `0.024784636` |
| Public BER, v14 | `0.019216735` |
| Public BER, v15 | `0.017493827` |
| v15 per-case scores | `[129979,177394,185451,140530,179802,183299,137314,184164,184485,145794,182587,185024]` |
| v16 per-case scores | `[129979,177394,185451,140530,179802,183299,140699,184164,184485,150583,182587,185024]` |
| v15 judge runtime | `384–4279ms` per case — well under the 2-minute limit |

The best accepted architecture is:

- one v14-style scheduled bSB trajectory per case, with an added Onsager/TAP reaction term;
- rate-specific `beta` and `parity_scale` fields, tuned at `N=1296`;
- exact `(N, K)` dispatch with the nearest configured SNR;
- decoder lambda and energy-selection lambda are separate parameters;
- 50 steps, one trajectory per case, QAIA/bSB only, except exact Gray-pair coupling on accepted v17 cases 4/7/10.

bSB is compliant ballistic SB: the check gradient uses continuous `x_j` in Eq. (1)
with `sigma -> x`. No per-edge messages carry over between steps, so it is not
BP, min-sum, or bit-flip. The v15 reaction term is the gradient of a polynomial
that is constant on `±1` spins; it changes only the continuous landscape, not
the Ising problem. Its BP-adjacent statistical-physics lineage is documented in
`submissions/v15_tap_bsb_candidate/README.md`.

Judge status `Accepted` means the submission was valid. It does not mean the score improved; v10 and v11 were both accepted but scored below v9.

## Phase 11 — TAP-corrected bSB (v15 accepted)

Tier 3 screened alternative QAIA formulations at `N=1296, R=1/2` (100 frames). The tuned bSB baseline was `111,943`.

| Formulation | Best |
|---|---:|
| noisy bSB, 6 seeds + energy selection | 113,782 |
| SimCIM with noise (300 configs) | 109,175 |
| LQA (300 configs) | 105,991 |
| split checks with auxiliary spins (300 configs) | 96,505 |
| **bSB + Onsager/TAP reaction term** | **134,014** |

The reaction term is the gradient of a polynomial that is constant on ±1 spins, so the Ising problem is unchanged (see `submissions/v15_tap_bsb_candidate/README.md`). With one setting per rate, tuned only at N=1296, v15 gains `+174,193` over the v14 binary on fresh seed 6060, with every case's CI above zero. The public BER drops to `0.017494`. New trajectory fields: `beta` (default 0) and `parity_scale` (default 1), supported by `decode_v3_python.py`, `bsb_vectorized.py` and `export_submission_cpp_v3.py`. With the defaults, output is bit-identical to v14. The judge returned `2,015,823`, up `167,981` from v14, and `submissions/v15_tap_bsb_candidate/` is the accepted archive.

## Phase 12 — per-case TAP tuning

Phase 12 keeps the accepted v15 architecture and retunes the seven scalar TAP/bSB fields per case. The comparator is the compiled v15 binary, cached per frame in `artifacts/baseline_cache_v15/` for seeds 911, 2024, 31337 and 6060. These caches use `real_bench.build_case(..., noise_scale=1.00112)` with 500 frames per case and seed, so both the tuner and the validator compare the same realization frame-for-frame.

### Method

1. **Stage A′ — local parameter search.** `tools/tune_tap_local.py` starts from the accepted v15 setting for a case and runs a parallel log-space coordinate search over `tau`, `xi`, `lambd`, `delta`, `dt`, `beta`, and `parity_scale`. Bounds are `tau 1–8`, `xi 0.2–3`, `lambd 2–20`, `delta 1.5–8`, `dt 0.12–0.5`, `beta 0.05–0.45`, and `parity_scale 0.15–1.2`. Training uses seed 911 with 300 frames; validation uses 500 frames on seeds 2024 and 31337. Six workers evaluate probe moves in parallel. This is deliberately local: wide random grids and global retunes repeatedly failed in earlier phases.
2. **Constant beta by default.** `beta` stays at the accepted per-rate value unless a tiny end-of-run or scheduled variant clears every gate. `--beta-knots` is only an optional Stage B′ probe, not part of the flat Stage A′ search.
3. **Retries for near misses.** Near-gate cases were rerun from their best configuration with train seed 5151. No retry produced a new accepted move, so the original best result stands.
4. **Stage B′ — conditional schedule probe.** `tools/tune_bsb_stage_b.py --tap-from` starts from a Stage A′ winner and may add piecewise-linear parameter schedules. A schedule is kept only if it beats flat Stage A′ on **both** validation seeds. The case-10 search was reproduced twice (`151,620.3 → 151,708.4`, only `+88` train score); the first validation crashed on the old v12/v13c cache and the corrected rerun timed out before validation. `tools/tune_bsb_stage_b.py` now defaults to `artifacts/baseline_cache_v15/` for `--tap-from`, but the schedule is dropped because the training gain is far below bench noise.
5. **Bootstrap gates.** Every gain uses paired frame bootstrap 95% CIs: the candidate must gain `>= +2,000` at `R=1/2` or `>= +1,500` at `R=2/3` on **both** seeds, with both CI lower bounds above zero. Any parameter landing exactly on a search bound must be widened once and rerun.
6. **Binary confirmation.** `tools/validate_binary_gains.py` rebuilds the same frames and compares the compiled candidate against `artifacts/build/v15_ref.exe`, so the reported gains are not only Python-simulator results.
7. **v16 acceptance gate.** A built candidate must gain `>= +1,500` on every changed case with fresh seed 6060, 300 frames, paired against v15, with CI above zero. Unchanged cases must be byte-identical. Public case 9 must remain byte-identical or drop below v15's public BER `0.017493827`.
8. **Structural screen.** `tools/screen_tap_structural.py` screens a beta ramp, a six-trajectory energy portfolio, and a dSB finish against the retuned flat TAP-bSB. Promotion bars are `+4,000` at `N=1296` and `+8,000` at `N=3888`; a miss means the structure is dropped.

### Stage A′ findings (500 frames)

Stage A′ is complete for all 8 tuned cases. Every gain is versus the compiled v15 binary on identical frames; CI values are 95% lower bounds on that seed's gain.

| Case | `N / rate` | Seed 2024 gain / CI low | Seed 31337 gain / CI low | Boundary | Decision |
|---:|---|---:|---:|---|---|
| 1 | 648 / R=1/2 | −1,029 / −2,893 | −2,036 / −4,101 | no | Failed both seeds; keep v15 |
| 2 | 648 / R=2/3 | −714 / −2,405 | −512 / −2,283 | `tau=8` hit | Failed; widening rerun required |
| 3 | 648 / R=3/4 | — | — | — | Not tuned; R=3/4 skipped |
| 4 | 1296 / R=1/2 | +1,327 / +60 | +2,175 / +762 | no | Failed `+2,000` gate; 1000-frame retest required |
| 5 | 1296 / R=2/3 | −69 / −842 | +361 / −373 | no | Failed both seeds; keep v15 |
| 6 | 1296 / R=3/4 | — | — | — | Not tuned; R=3/4 skipped |
| 7 | 1944 / R=1/2 | +3,878 / +2,341 | +1,979 / +307 | no | Missed strict gate by 21.5; 1000-frame retest required |
| 8 | 1944 / R=2/3 | +822 / +4 | +158 / −803 | no | Failed both seeds; keep v15 |
| 9 | 1944 / R=3/4 | — | — | — | Not tuned; public case stays byte-identical |
| 10 | 3888 / R=1/2 | +5,675 / +4,321 | +7,007 / +5,402 | no | **Passed** flat Stage A′ |
| 11 | 3888 / R=2/3 | +1,035 / +251 | +452 / −350 | `dt=0.5` hit | Failed; widening rerun required |
| 12 | 3888 / R=3/4 | — | — | — | Not tuned; R=3/4 skipped |

The flat Stage A′ winner for case 10 is `tau=2.295`, `xi=1.264`, `lambd=5.95`, `delta=3.72`, `dt≈0.34841`, `beta=0.27`, `parity_scale≈0.28814`. It has no boundary hit.

### Boundary reruns

`tools/tune_tap_local.py` gained a targeted `--bounds-override KEY=LOW:HIGH` option so one parameter can be widened without moving every bound via `--bounds-scale`. Each bound case was rerun exactly once.

| Case | Widening | Train gain | Seed 2024 / CI low | Seed 31337 / CI low | Edge after rerun | Result |
|---:|---|---:|---:|---:|---|---|
| 2 | `tau 1–8 → 1–12` | +1,261.5 | −1,567 / −3,111 | +1,507 / −223 | no | Failed; dropped |
| 11 | `dt 0.12–0.5 → 0.12–0.7` | 0.0 | +1,035 / +251 | +452 / −350 | no | Failed; dropped |

Both R=2/3 boundary candidates were dropped, so no R=2/3 case is eligible for v16.

### High-power near-miss re-tests

New v15 per-frame caches were generated in `artifacts/baseline_cache_v15_1000/` for cases 4 and 7 on seeds `2024` and `31337`, 1000 frames each. `load_baseline_errors` requires an exact frame-count match, so the 1000-frame tests cannot reuse the 500-frame caches.

| Case | Frames | Seed 2024 gain / CI low | Seed 31337 gain / CI low | Gate |
|---:|---:|---:|---:|---|
| 4 | 1000 | +1,021.6 / +21.7 | +1,781.6 / +667.1 | Failed `+2,000`; dropped |
| 7 | 1000 | +2,676.9 / +1,527.5 | +3,820.8 / +2,644.1 | **Passed** |

Case 7's flat config is `tau=2.295`, `xi=1.264`, `lambd=5.95`, `delta=3.72`, `dt≈0.34841`, `beta=0.27`, `parity_scale≈0.35537`. It is retained only under the revised 1000-frame gate; under the original strict 500-frame rule it still misses seed 31337 by 21.5 points.

Compiled-binary confirmation on the same 1000 frames:

| Seed | Case 7 gain / CI | Case 10 gain / CI |
|---|---:|---:|
| 2024 | +2,676.9 / [+1,523.0, +3,895.3] | +5,528.2 / [+4,343.4, +6,627.4] |
| 31337 | +3,820.8 / [+2,596.1, +5,036.3] | +5,758.8 / [+4,604.9, +7,018.4] |

Fresh-seed gate on seed `6060`, 300 frames, minimum `+1,500` per changed case:

| Case | Baseline → candidate score | Gain / CI | Result |
|---:|---:|---:|---|
| 7 | 137,874.8 → 141,201.7 | +3,326.8 / [+1,026.7, +5,779.7] | Passed |
| 10 | 140,981.1 → 144,637.8 | +3,656.8 / [+1,873.3, +5,740.1] | Passed |

### Stage B′ finding

The case-10 schedule search was reproduced twice: `151,620.3 → 151,708.4`, a training gain of only `+88.1` over `121` evaluations. The first validation crashed because `tools/tune_bsb_stage_b.py --tap-from` still defaulted to the old v12/v13c cache; the tool now defaults to `artifacts/baseline_cache_v15/` and records `baseline_cache_dir` in its artifact. The corrected rerun timed out after `7,200 s` before validation, so no schedule artifact was produced. Because `+88` is far below the bench's roughly `±1.5k` per-case reliability, Stage B′ is dropped and flat case 10 is retained. Decision: `artifacts/phase12_stage_b_prime_decision.json`.

### Structural screen finding

`tools/screen_tap_structural.py` screened case 10 with 100 frames on seeds `2024` and `6060`, `noise_scale=1.00112`, and a `+8,000` promotion bar. Flat Stage A′ baselines were `141,928.9` and `144,050.7`. Nothing was promoted.

| Variant | Seed 2024 gain / CI low | Seed 6060 gain / CI low | Promoted |
|---|---:|---:|---|
| `beta_ramp` | −2,920.8 / −6,014.5 | −1,537.2 / −4,886.5 | no |
| `portfolio6` | −466.8 / −3,624.5 | −1,232.9 / −4,615.7 | no |
| `dsb_finish_5` | −2,118.1 / −2,855.9 | −2,559.6 / −3,550.4 | no |
| `dsb_finish_10` | −4,451.2 / −5,860.1 | −4,634.8 / −5,981.0 | no |

### Bench calibration finding

The bench is reliable for decisions of several thousand points but not marginal ones. Fixed-rule v15 bench total is `1,996,724.5` versus the accepted judge total `2,015,823`, a `+19,099` judge bias. The linear fit has slope `0.947`, intercept `10,410.4`, RMSE `2,112.7`, MAE `1,535.3`, and max residual `4,674.0` points per case (`artifacts/phase13/bench_vs_judge_v15.json`). Offset mean is `+1,591.5` and ratio range is `0.9931–1.0609`. This supports the case-10 decision but does not make the 21.5-point case-7 miss credible.

### Accepted v16 candidate

`submissions/v16_tap_case_tuning/` is the accepted v16 archive. It is v15 plus flat Stage A′ cases 7 and 10. Cases `1–6, 8, 9, 11, 12` are byte-identical to v15. The judge returned `2,023,997` (`+8,174` over v15).

| Check | Result |
|---|---|
| Manifest / source | `artifacts/phase12_case_manifest_v16.json`; `submission.cpp` = `19,964` bytes |
| Source SHA-256 | `5F304437925F758AFD9A645D418D8E663B30D50048FACF2C4FABDF7995E94046` |
| `validate_submission_v3.py` | All gates true; no forbidden tokens or offline-ceiling symbols |
| Strict C++17 compile | Passed with `w64devkit/bin/g++.exe -std=c++17 -O2 -Wall -Wextra -pedantic` |
| Public case 9 | BER `0.017493827`; output byte-identical to v15 |
| Public output SHA-256 | `1BBF2BBAA56E6AD871BB6FFE7F0AC4D501C31F3CE7A684195CAFD8F1AAA8723A`, same as v15 |
| Fresh seed `6060` | Passed both changed cases |
| Status | accepted by judge at `2,023,997` |

### Finding index

- `artifacts/tap_a_prime_*.json` — Stage A′ artifacts, including the 1000-frame retests.
- `artifacts/tap_a_prime_N648_R2-3_wide.json` and `artifacts/tap_a_prime_N3888_R2-3_wide.json` — boundary reruns.
- `artifacts/phase12_stage_b_prime_decision.json` — Stage B′ drop decision.
- `artifacts/phase12_structural_N3888_R0.5.json` — structural screen.
- `artifacts/phase12_v16_*` — validator, binary-gain, fresh-seed, and public-output artifacts.
- `submissions/v16_tap_case_tuning/` — self-contained local candidate archive.

### Current stop position

- Phase 12 is complete. The only v16 changes are cases 7 and 10, both flat Stage A′.
- The judge score `2,023,997` is at the low edge of the `2.03–2.05M` target; all eight high-rate cases and the other three `R=1/2` cases remain byte-identical to v15.
- R=3/4 is skipped because no changed case has cleared a gate in passing.
- Phase 12 cleanup was committed and pushed as `e6ac605`.
- This Phase 12 stop position was superseded by accepted v17; Phase 15 uses v17 as its base.

## Phase 13 — offline ceiling analysis

Phase 13 is offline analysis only. BP, OSD, list-MAP, and other non-QAIA decoders are measuring sticks here and never enter a manifest or submission. Every Python tool in `tools/ceiling/` starts with `# OFFLINE ANALYSIS ONLY — never imported by the exporter`, and `tools/validate_submission_v3.py` rejects those symbols in generated source; an injected `paired_bootstrap_ci` probe was correctly rejected.

Fixed rules are `real_bench.build_case(..., noise_scale=1.00112)`, seeds `2024`, `31337`, `6060`, 500 frames per case per seed, and paired frame bootstrap 95% CIs. `artifacts/phase13/cache_self_test.json` compared the v15 cache with itself and reported exactly zero delta and zero-width CI on every case. `artifacts/phase13/cache_smoke_manifest.json` confirmed `cache_binary.py` on seed `6060` with the v15 reference binary hash prefix `125a7211740a271b`.

Step 0 recorded `2,517,863` as the live top UCup scoreboard total by `haminh109` on 2026-11-26 (`artifacts/phase13/step0_target_identity.json`). It is a real hidden-judge score, not a local ceiling: rank 1, second place `2,516,559`, gap `304`, contest live until 2026-12-08. The gap over accepted v15 is `+502,040`. The official formula is `-Σ 10·lg(BER_a)` over 12 hidden datasets, scaled by `10,000` by the judge. No source or algorithm is public, so this is not evidence that `2,517,863` is reachable by a QAIA-compliant decoder. The earlier offline full-BP estimate is about `2.27M`, still below the live target.

## Phase 14 — joint Gray-pair coupling

Phase 14 builds a local v17 candidate on accepted v16. The decoder remains QAIA-only: `tools/qam_pairs.py` recovers the exact transmitted 16-QAM amplitude from each pair of Gray bit LLRs and forms a channel table `log p(y|b0,b1)=const+a·s0+b·s1+d·s0·s1`. When `joint_pairs=1`, the bit field adds the mean-field reaction `2·pair_scale·d·x_other` and the TAP/Onsager correction `-pair_gamma·(2d)²·(1−x_other²)·x_self`. This is the negative gradient of a polynomial that is constant on `±1` spins; it stores no per-edge messages and does not change the Ising solution. The energy combine uses the same exact pair interaction. The exporter precomputes the `y`-linear part of the component log-likelihoods and uses a cubic-interpolated softplus only inside the 64-step bisection; the final pair consistency check still uses the exact stable log-sum-exp, and the Python reference remains exact. Defaults (`joint_pairs=0`, `pair_gamma=0`, `pair_scale=1`) leave cases bit-identical to v16.

A0 changed the bench LLR convention to nominal σ by default: noise still uses `σ·noise_scale`, but generated LLRs now match the judge's public file. The v16 baseline cache was rebuilt for all 12 cases on seeds `911`, `2024`, `31337`, `6060` (48 files). `tools/phase14_bench_shift.py` passed the `±1,000`-point regression on all 36 case/seed pairs; the worst shift was `+568.3` on case 6, seed `6060`.

A3 joint tuning passed on cases 4 (`N=1296`, `+2,508 / +4,973`), 7 (`N=1944`, `+4,913 / +7,677`) and 10 (`N=3888`, `+9,493 / +7,099`) across validation seeds `2024 / 31337`, all with positive paired-bootstrap CIs and no parameter on a bound. Case 1 failed twice and remains v16-identical. Case 10's first two-cycle search timed out before writing an artifact; the one-cycle search cleared the gate. The partial v17 first hit a marginal runtime failure on case 4 (`3.05×` v16). The exporter now precomputes the `y`-linear component log-likelihoods and uses a cubic-interpolated softplus inside bisection, which reduced the worst runtime ratio to `1.48×` while preserving bit identity and the exact final consistency check. The full v17 candidate is in `submissions/v17_joint_pairs/`. Its complete A4 gate (`artifacts/phase14/final_gate.json`) reports `all_passed=true`: strict C++17 compile, validator, offline isolation, public case-9 SHA-256 `1BBF2BBA…8723A`, Python/batched/C++ bit identity, swapped-pair consistency guard, fresh seed `6060` (`+4,299 / +5,481 / +8,573` on cases 4/7/10, all other cases byte-identical), and the runtime guard. Total fresh-seed gain is `+18,353`, roughly `+17,380` after the Phase 13 judge-slope calibration. The v17 work was committed to `main` as `33e7f4c` and accepted by the judge at `2,038,794` (`+14,797` over v16). It is now the base for Phase 15.

### Post-v17 research — Tracks B and C

Track B completed the offline joint-likelihood ceiling ladder in `tools/ceiling/bp_variants.py` and `tools/ceiling/phase14_track_b_report.py`. v18 was **not** promoted: at `N=3888, R=2/3`, joint BP beat per-bit BP by only `+780` (95% CI `[+515, +1,047]`), far below the `>5,000` trigger. Joint likelihood helped `R=1/2` substantially (`+12,058` at `N=1944`, `+20,554` at `N=3888`). With the small omitted `R=1/2` cases sampled at only `100 / 60` frames (`±3–5k`), the partial judge-scaled achievable-decoder headline is `≈ 2.25M (partial; 3/4 cases ML-limited)`; the live top `2,517,863` remains unmodelled by the BP/OSD variants tested. This is an achievable lower bound, not a formal ceiling. OSD-2 on joint BP lost `−5k` to `−10k` at `R=1/2`, with maximum-likelihood selection choosing a wrong codeword in `0.34 / 0.70 / 0.58` fractions on the sampled cases. OSD-2 was only a 100-frame side-by-side screen; no order-3 OSD or list-MAP ceiling is claimed. Reports: `artifacts/phase14/track_b_ceiling_report.md` and `.json`.

Track C completed C1–C4 screens in `tools/screen_joint_structural.py` with default-off decoder extensions in `tools/bsb_vectorized.py`. **No variant was promoted.** C1 (sign/magnitude bit classes) came closest but was seed-unstable and below the bar; C2 (exact pair TAP cross term), C3 (single amplitude coordinate), and C4 (`pair_scale` ramp) all lost score. C5 gradient deep unfolding is blocked because torch is not installed. The decoder extensions are default-off; scalar v17 behavior remained bit-identical, verified by 11 unit tests and three-way Python/batched/C++ identity on cases 4/7/10. Reports: `artifacts/phase14/track_c_report.md` and `.json`.

## Phase 15 — v18 research plan

Phase 15 is gated offline research on the accepted v17. Every candidate gain is paired frame-for-frame against a cached v17 binary on identical `real_bench.build_case(..., noise_scale=1.00112)` frames, with a 1,000-resample paired-bootstrap 95% CI. The target is `2.06–2.08M`, or `2.10M+` if the independent workstreams stack; the realistic QAIA/bSB ceiling inferred from Track B is about `2.25M`.

- **A — trajectory portfolio.** Screen diverse compliant QAIA trajectories for cases 10 and 7 first. Continue only if the oracle gain is at least `+8k` and realised energy-selection gain at least `+4k` at `N=3888`; drop A if oracle gain is below `+5k`. A2 must clear `+2k` on both validation seeds.
- **B — learned schedules.** Use a local CMA-ES implementation (33 log-multipliers in `[0.5, 2]`, `σ₀=0.15`, `λ=12`, `μ=6`) on cases `10 → 7 → 4 → 1`. Rate `1/2` needs `+2k`, rate `2/3` needs `+1.5k`, both with positive CI lower bounds. Case 10 must show `+5k` held-out gain by `30 Nov` or B is killed.
- **C — rate-2/3 joint-pair spare bet.** Tune only `pair_gamma`, `beta`, `parity_scale`, and `dt` for case 11; drop C entirely unless case 11 clears `+1.5k` on both validation seeds.
- **Combination.** Keep A+B only if it beats the better single stream by at least `+1k` on both seeds. Otherwise ship the best single passing stream or leave v17 unchanged.
- **Fallback.** Prepare v18a with only the best single validated change. If a combined v18 regresses any changed case by more than `3k`, submit v18a; if that also regresses, resubmit v17 source.

Fixed submission rules remain: at most 50 steps, at most six trajectories, no per-edge state, QAIA-only energy gradients/mean-field reaction terms, public case 9 byte-identical, strict C++17 compile, validator, offline isolation, three-way bit identity, and runtime judge estimate at most `60 s`.

Workstream B is killed at the case-10 checkpoint: four CMA generations (33 log-multipliers, `λ=12`, `μ=6`) produced validation gains `-4,687` (seed 2024, CI `[-6,821, -2,638]`) and `-2,364` (seed 31337, CI `[-4,800, -28]`), both failing the `+2k` rate-1/2 gate and the `+5k` held-out checkpoint (`artifacts/phase15/workstream_b_case10_schedule.json`). No learned-schedule variant will be built for cases 7/4/1.

Workstream C is dropped: four CMA generations on case 11 produced best validation gains `-420` (seed 2024, CI `[-1099, +322]`) and `-179` (seed 31337, CI `[-917, +483]`), both far below the required `+1,500` with positive lower bounds (`artifacts/phase15/workstream_c_case11.json`). No case 8/11 joint-pair variant will be built.

Greedy case 10 also improved training gain from `+5,759` to `+9,491` with six trajectories, and held-out seeds 2024/31337 gave `+6,556` (CI `[4,657, 8,816]`) and `+7,512` (CI `[5,490, 9,708]`), both above the original portfolio. A two-case greedy archive is exported and passes validator, public SHA, and three-way identity (`artifacts/phase15/v18_greedy_both*`). Fresh seed 6060 passes both changed cases: case 7 `+5,803` and case 10 `+3,974`, all unchanged cases byte-identical (`artifacts/phase15/v18_greedy_both_fresh_seed_6060.json`). Its full final gate passed every correctness gate but failed the `60 s` judge-estimate cap on case 10 (`69.5 s` for 500 frames with six trajectories), so the six-trajectory case-10 portfolio is rejected. A five-trajectory case-10 variant also failed the runtime cap (`77.6 s`, despite passing fresh seed at `+3,574`), so it is rejected. Three- and four-trajectory case-10 variants are now being tested for the best gain that still clears `60 s` (`artifacts/phase15/v18_greedy_both3*` and `v18_greedy_both4*`). The three-trajectory case-10 portfolio clears both held-out seeds: seed 2024 `+3,909` (CI `[2,546, 5,322]`) and seed 31337 `+5,807` (CI `[4,055, 7,755]`). The four-trajectory portfolio is stronger on both: seed 2024 `+4,419` (CI `[2,833, 6,306]`) and seed 31337 `+6,251` (CI `[4,284, 8,430]`). Fresh seed 6060 passes for both: three trajectories `+2,747`, four trajectories `+3,385`, with every unchanged case byte-identical. Runtime measurements are distorted by a low-battery power limit (currently charging at `2%`, CPU capped at `2.1 GHz`): unchanged v17 rose from `~34 s` to `~62 s` on 500-frame case 10, so the absolute `60 s` local cap cannot be certified right now. A bit-preserving concurrent-trajectory exporter was added and passes validator/public SHA/three-way identity. The two-trajectory fallback fails fresh seed (`+1,049`, CI low below zero), so the final selection is the three-trajectory case-10 portfolio with the five-trajectory case-7 portfolio, packaged as `submissions/v18b_greedy_portfolio/` (`prepared_not_submitted`). A clean final gate should be rerun after the machine returns to normal power. The case-7-only final gate passed compile, tests, isolation, validator, identity, guard, fresh seed, and runtime, and failed only on a robustness-report formatting bug that tried to parse `mode="bsb"` as a float; `tools/phase15_robustness.py` is fixed. Greedy case 7 fresh-seed 6060 gain is `+5,803`, above the original `+4,437`.

The greedy five-trajectory case-7 portfolio is now the leading v18 candidate: training gain `+8,045`, held-out seed 2024 `+7,431` (CI `[5,661, 9,474]`), seed 31337 `+5,739` (CI `[3,967, 7,656]`), both above the original four-trajectory gains. It exports to `artifacts/phase15/v18_greedy7.cpp` (validator pass, public case-9 SHA unchanged, portfolio three-way identity pass) and is included in the final `v18b` archive.

Fresh seed `6060`, 300 frames, is mixed for the original two-case v18a: case 7 gains `+4,437` (CI `[1,852, 7,674]`) but case 10 gains only `+1,947` with CI low `−46`, so the two-case candidate fails the final checklist (`artifacts/phase15/v18a_fresh_seed_6060.json`). A case-7-only archive was exported and passes validator, public SHA, and three-way identity (`artifacts/phase15/v18a_case7only*`); it is superseded by the stronger five-trajectory case-7 portfolio in `v18b`. Greedy portfolio search improved case 7 training gain from `+4,552` to `+8,045` with five trajectories.

Workstream A passed both gates with the `tau_xi` family on cases 7 and 10. A1 (200 seed-911 frames) found case-10 oracle `+11,918` and realised `+5,759` (CI `[2,465, 9,525]`); A2 reproduced gains on seeds 2024/31337 of `+3,186 / +5,065` for case 10 and `+6,900 / +5,026` for case 7, all with positive 95% CIs (`artifacts/phase15/a1_portfolio_screen.json`, `a2_validation_seed2024.json`, `a2_validation_seed31337.json`). A v18a manifest with only these two cases changed exports to strict C++17, preserves the public case-9 SHA, and passes a portfolio-aware three-way scalar/batched/C++ bit-identity check (`tools/phase15_equivalence.py`).

Step 0 is complete: `submissions/v17_joint_pairs/judge_result.json` records the accepted total, the v17 README and candidate status are marked accepted, and the Track B report now uses the partial `≈ 2.25M (partial; 3/4 cases ML-limited)` headline with the added 100-frame cases 1/2/4 and the joint-OSD rejection. `artifacts/build/v17_ref.exe` was compiled from the exact submitted v17 source with strict C++17 flags. `artifacts/baseline_cache_v17/` contains frame-matched v17 errors for all 12 cases on seeds `911 / 2024 / 31337 / 6060`; `artifacts/phase15/step0_cache_check.json` proves source hash, inventory and repeat-generation bit identity (`all_passed=true`). `tools/cmaes_lite.py` passes its rotated-ellipsoid and checkpoint tests, and a 33-parameter schedule manifest exports to strict C++17 with unchanged public case-9 SHA while zero multipliers reproduce v17 bit-for-bit (`artifacts/phase15/export_probe.cpp`).

## Submission history

| Version | Score | Delta | Outcome |
|---|---:|---:|---|
| v1 | `1,515,872` | — | accepted; single deep-unfolded probe decoder |
| v2 | `1,579,558` | `+63,686` | per-case tuning everywhere; `R=1/2` regressed |
| v3 | `1,588,472` | `+8,914` | hybrid idea right, but wrong fallback baseline |
| v4 | `1,593,920` | `+78,048` vs v1 | accepted hybrid: exact probe for `R=1/2`, tuned elsewhere |
| v5 | not submitted | — | broad global energy tuning; rejected locally after public BER worsened |
| v6 | `1,597,233` | `+3,313` | refined case 11 (`N=3888, R=2/3`) |
| v7 | `1,600,231` | `+2,998` | refined the other three `R=2/3` cases |
| v8 | `1,601,303` | `+1,072` | directly tuned case 9 on public input |
| v9 | `1,606,841` | `+5,538` | refined cases 3, 6, 12 (`R=3/4`) |
| v10 | `1,605,473` | `-1,368` vs v9 | accepted by judge but below best; case 11 regressed |
| v11 | `1,606,575` | `-266` vs v9 | accepted by judge but no improvement; case 6 regressed |
| v12 | `1,607,212` | `+371` vs v9 | accepted new best: v9 + case-5-only salvage |
| v13 | not submitted | — | flat bSB `R=1/2` candidate; superseded by scheduled v13c |
| v13c | `1,700,646` | `+93,434` vs v12 | accepted new best: scheduled bSB for all four `R=1/2` cases |
| v14 | `1,847,842` | `+147,196` vs v13c | accepted new best: all eight high-rate cases moved from dSB portfolios to tuned bSB |
| v15 | `2,015,823` | `+167,981` vs v14 | accepted new best: Onsager/TAP correction added to bSB on all 12 cases |
| v16 | `2,023,997` | `+8,174` vs v15 | accepted new best: v15 + flat Stage A′ cases 7 and 10 |
| v17 | `2,038,794` | `+14,797` vs v16 | accepted new best: joint Gray-pair coupling on cases 4, 7, 10 |

Rate-level totals show where all durable progress came from:

| Version | `R=1/2` | `R=2/3` | `R=3/4` | Total |
|---|---:|---:|---:|---:|
| v1 | `360,797` | `540,445` | `614,630` | `1,515,872` |
| v4 | `360,797` | `566,864` | `666,259` | `1,593,920` |
| v6 | `360,797` | `570,177` | `666,259` | `1,597,233` |
| v7 | `360,797` | `573,175` | `666,259` | `1,600,231` |
| v8 | `360,797` | `573,175` | `667,331` | `1,601,303` |
| v9 | `360,797` | `573,175` | `672,869` | `1,606,841` |
| v10 | `360,797` | `571,807` | `672,869` | `1,605,473` |
| v11 | `360,797` | `573,546` | `672,232` | `1,606,575` |
| v12 | `360,797` | `573,546` | `672,869` | `1,607,212` |
| v13c | `454,231` | `573,546` | `672,869` | `1,700,646` |
| v14 | `454,231` | `670,586` | `723,025` | `1,847,842` |
| v15 | `553,617` | `723,947` | `738,259` | `2,015,823` |

Through v12, `R=1/2` had never improved over v1 and every durable gain came from high-rate per-case tuning. v13c changed that: scheduled bSB lifted `R=1/2` from `360,797` to `454,231` while leaving the eight high-rate cases bit-identical to v12. v14 completed the transition by replacing all eight high-rate dSB portfolios with tuned bSB trajectories, adding `+97,040` at `R=2/3` and `+50,156` at `R=3/4` with no regression in any case.

v15 changed the landscape rather than the schedules. The v14 trajectories were kept, and a reaction term plus parity-column scaling were added with rate-specific strengths. Every case improved: `+99,386` at `R=1/2`, `+53,361` at `R=2/3`, and `+15,234` at `R=3/4`. The largest single gain was `+32,501` at `N=3888, R=1/2`.

## Strategies that worked

1. **Hybrid dispatch.** Keeping the exact accepted probe decoder for `R=1/2` and using tuned portfolios only for high rates produced the largest single jump (v4).
2. **Per-case, not global, configuration.** Each `(N,R,SNR)` operating point has its own learned trajectory and fixed candidates. Broad shared tuning repeatedly underperformed.
3. **Grid + local search around the current tuned region.** v6–v9 all came from narrower grids around known-good dSB parameters, followed by local search and greedy portfolio replacement.
4. **Six-trajectory energy portfolios.** Multiple dSB trajectories with energy-based selection consistently beat single trajectories at high rates.
5. **Separating decoder strength from selection strength.** Decoder `lambda` and energy-selection `lambda` are distinct; v6 established this and later versions rely on it.
6. **Direct public-data tuning for case 9.** With 500 public frames and labels, a 250/250 selection-validation split produced a real `+1,072` judge gain (v8).
7. **One-case-at-a-time changes.** Successful versions changed one or a tight group of cases, making judge deltas attributable.
8. **Rebasing away a bad case works.** v11 rebuilt from v9 removed v10's `-1,739` case-11 regression exactly and preserved the `+371` case-5 gain.
9. **Cross-seed robustness checks are useful only as a weak filter.** They rejected several apparent gains before submission, but they failed to predict both v10 case 11 and v11 case 6.
10. **Stage A → gated Stage B generalizes to high rates.** Rate-specific anchors found large gains at `R=2/3` and `R=3/4`; schedules were kept only when they beat flat Stage A on both validation seeds. The v14 total was predicted at `1,847,584` and judged at `1,847,842`, an error of only `258` points.
11. **Fixing the continuous landscape, not just the schedule.** v15 kept v14's schedules and added an Onsager/TAP reaction term that is the gradient of a polynomial constant on `±1` spins. It improved all 12 cases and added `+167,981`, showing that plain bSB behaves like mean-field SB and that the reaction correction is the dominant remaining QAIA-internal lever.

## Assumptions and directions that proved wrong

1. **Synthetic `6 dB` extrapolation for `R=1/2`.** v2's per-case tuning reversed the judge ranking for all four `R=1/2` cases. The probe decoder remains frozen.
2. **"Fallback" means accepted baseline.** v3 used the wrong local `submission.cpp` instead of the judge-accepted probe, losing `5,448` points versus v1 on `R=1/2`.
3. **Broad energy-lambda tuning.** The v5 global energy retune worsened public BER and was rejected before submission.
4. **Synthetic BER ratios as score predictors.** Ratios do not transfer linearly, and small ratios often reverse sign. v7 ratios around `1.37–1.53` produced gains from `+265` to `+1,420`; v10 case 11 had ratios `1.025–1.027` but lost `1,739`; v11 case 6 had ratios `1.021–1.024` but lost `637`.
5. **Energy-only retuning at `N=3888, R=2/3`.** v10 case 11 changed only energy-selection lambdas and regressed badly. This is the clearest example of synthetic energy selection overfitting channel/model mismatch at large `N`.
6. **Marginal cross-seed gains near `1.02`.** v11 case 6 looked consistently positive on three synthetic comparisons but still cost `637` on the judge. Multi-seed consistency in the `1.00–1.03` band is not evidence of a real judge gain.
7. **Public BER as a global proxy.** The public input dispatches only to case 9. v9, v10, and v11 all produced the same public output, despite materially different judge scores on cases 5, 6, and 11.
8. **Judge `Accepted` status as evidence of improvement.** v10 and v11 were both accepted as valid submissions but were below v9. The score table, not the status label, decides the best version.
9. **Adding independently "supported" case changes.** v10 combined a genuine case-5 gain with a bad case-11 change. v11 removed the bad case-11 change but added a bad case-6 change. The case-5 gain is repeatable, but it was too small to absorb another regression.
10. **Trajectory replacement being automatically safer than energy-only tuning.** v11 case 6 changed the portfolio/trajectory set and still regressed. The previous conclusion that trajectory changes were safer was too broad.
11. **Treating plain soft-spin SB as the full QAIA search space.** Every naive alternative in the Tier 3 screen stalled near mean-field behavior. The Onsager correction produced the first large structural gain over v14. The remaining caveat is classification, not numerics: the term is BP-adjacent in origin, even though it is implemented as an SB landscape term with no stored per-edge messages.

## v10 post-mortem

| Case | Change | Judge delta |
|---|---|---:|
| 5 `N=1296, R=2/3` | second-pass portfolio replacement | `+371` |
| 11 `N=3888, R=2/3` | energy-lambda-only retune | `-1,739` |
| total |  | `-1,368` |

Lesson: salvage case 5, discard case 11, and rebuild on accepted v9. The rebuild hypothesis was confirmed by v11.

## v11 post-mortem and key findings

v11 rebuilt from accepted v9 and changed only cases 5 and 6. It scored `1,606,575`, down `266` from v9.

| Case | Change | Judge delta | Finding |
|---|---|---:|---|
| 5 `N=1296, R=2/3` | v10 salvage | `+371` | Gain reproduced exactly; rebase/salvage logic was correct |
| 6 `N=1296, R=3/4` | second-pass portfolio replacement | `-637` | Consistent synthetic ratios `1.021–1.024` did not transfer |
| total |  | `-266` | No improvement over v9 |

Key findings from this run:

1. **The v9 rebase was successful.** Case 11 returned from `141,113` in v10 to `142,852` in v11, exactly matching v9. The v10 case-11 change is definitively discarded.
2. **Case 5 is the only repeatable judge-validated gain after v9.** It produced `+371` in both v10 and v11. A case-5-only successor would mathematically be `1,607,212`. `submissions/v12_safe_case5_candidate/` now contains a prepared candidate with only that change; it has not been submitted.
3. **Case 6 should be discarded.** Its v11 score fell `167,094 → 166,457`. The synthetic improvement signal was not merely weak on one seed; it was consistent across three comparisons and still wrong on the judge.
4. **Small synthetic gains are noise for this task.** Multi-seed ratios around `1.02` should not be used to justify a submission. Even a ratio near `1.06` failed at large `N` in v10.
5. **The public proxy remains blind.** v11's public output was bit-identical to v9, so public BER offered no evidence about cases 5 and 6.
6. **Runtime and memory were never the limiting factor.** v11 used `27,613ms` total and peaked at `5,784ms`, far below the `2 minute` per-case limit; the loss was entirely BER/scoring.
7. **Do not combine a validated small gain with an unvalidated synthetic gain.** The one reliable `+371` was overwhelmed by the `-637` case-6 change.

## Current knowledge base and tools

All work now lives on `main`: the `optimize-phase8-case11` work was merged at `541dfab`, and the Phase 9 tooling follows in `9f6c6a5`, `b386ee8`, and `3c2d1c0`. Version folders are under `submissions/`, each with its own source, manifest, summaries, params, and judge result where available.

Key tools:

- `tools/phase7_per_case.py` — original synthetic grid/local/portfolio search.
- `tools/tune_public_case.py` — direct tuning on the 500-frame public case.
- `tools/refine_accepted_case.py` — compares new portfolios directly against the current accepted manifest portfolio.
- `tools/compare_accepted_summary.py` — independent-seed robustness checks.
- `tools/build_targeted_manifest.py` — replaces one case in a manifest from a tuned summary.
- `tools/export_submission_cpp_v2.py` — emits C++17 submission from a manifest.
- `tools/validate_submission_v2.py` — checks size, dispatch count, guess count, binary output, and public BER.
- `tools/decode_v2_python.py` — Python reference decoder for C++ parity checks.

New Phase 9 tooling:

- `tools/real_h_generator.py` — generates `H=[A|I]` with row weight `7` and correct info-column weights.
- `tools/real_qam_channel.py` — Gray-coded 16-QAM + AWGN LLR generator.
- `tools/real_bench.py` — shared-frame case builder for realistic `N/R/SNR` evaluation.
- `tools/calibrate_public_channel.py` — public channel/code statistics and noise-scale calibration.
- `tools/evaluate_history_on_real_bench.py` — checks whether the new bench reproduces historical judge delta signs.
- `tools/decode_v3_python.py` — Python dSB/bSB ensemble decoder with energy, vote, and soft-vote combine.
- `tools/export_submission_cpp_v3.py` — emits a C++17 v3 decoder from a manifest.
- `tools/validate_submission_v3.py` — validates v3 manifest/source gates.
- `tools/convert_manifest_v2_to_v3.py` — converts accepted v2 trajectories into approximate v3 manifests (learned reliability modifiers are not preserved).
- `tools/tune_v3_case.py` — paired gradient-free tuning on bench cases with a held-out split.
- `tools/evaluate_v3_candidate.py` — full 12-case candidate-vs-baseline gate on shared bench frames.
- `tools/bsb_vectorized.py` — batched bSB simulation over all frames at once, used by the tuners.
- `tools/tune_bsb_stage_a.py` — Stage A wide-grid search for one bSB trajectory.
- `tools/tune_bsb_stage_b.py` — Stage B piecewise-linear schedule search with multi-seed validation.
- `tools/build_v13_manifest.py` — applies tuned bSB trajectories to the `R=1/2` cases of a v3 manifest.
- `tools/summarize_stage_a.py` — tabulates Stage A results.

Validation gate for every candidate:

- clean `g++ -std=c++17 -O2 -Wall -Wextra -pedantic` compile;
- source under `512 KiB`;
- at most 6 trajectories and 50 iterations;
- public output not worse than accepted;
- Python/C++ bit parity on synthetic cases for every changed case;
- public dispatch cases remain bit-identical unless public tuning is intentional.

## Phase 9 implementation status

Phase 9 replaces the invalid surrogate bench before any structural tuning is trusted.

Current status:

1. **Phase 0 — safe banking:** `submissions/v12_safe_case5_candidate/` was submitted and accepted at `1,607,212`, exactly the predicted `+371` over v9. It now serves as the base for future candidates.
2. **Phase 1 — real bench:** implemented. `H=[A|I]` has row weight `7` and info-column weights `6/3/2`; Gray-coded 16-QAM LLRs match the public raw BER and even/odd bit-position pattern. Calibration wrote `artifacts/phase9_public_calibration.json`; calibrated noise scale is about `1.001`.
3. **Phase 2 — historical calibration:** a 100-frame 12-case run matched `106/108` known judge delta signs. A 500-frame rerun reproduced v10 case 5 (`+372` vs judge `+371`) and v10 case 11 (`-1,759` vs judge `-1,739`) almost exactly. It did not reproduce v11 case 6 (`+1,250` bench vs `-637` judge), so the bench screens large effects but is not an oracle for marginal gains.
4. **Phase 3 — structural decoder:** v3 Python/C++ tools implement dSB, bSB, and six-trajectory energy/vote/soft-vote combining, including per-step schedule arrays for `tau/xi/lambd/delta/dt`. Generated C++ was compiled with `g++ -std=c++17 -O2 -Wall -Wextra -pedantic` and matched Python bit-for-bit on scalar and scheduled trajectories.
5. **Phase 4 — tuning and dispatch (superseded by item 6):** `tools/tune_v3_case.py` provides paired train/validation tuning. The original case-1 structural screen reported bSB/vote `85,936` and bSB/energy `85,869` against a `92,295` baseline and rejected bSB. That screen was measuring a defective gradient, so its verdict is withdrawn rather than confirmed.
6. **bSB gradient defect and fix.** `_check_contribution` multiplied every check product by the target spin, so the soft-spin force became `x_i^2 * prod(j!=i)` instead of Eq. (1)'s `prod(j!=i)`; that shrinks the force on exactly the uncertain bits, which is why bSB lost to voting. `9f6c6a5` computes excluded products with prefix/suffix buffers in both the Python decoder and the C++ exporter. dSB cannot change by construction (the two forms are equal for ±1 spins) and was verified bit-identical to the v12 binary on all eight dSB cases and on the 500-frame public input.
7. **Stage A — wide grid, four `R=1/2` cases.** One bSB trajectory per case, paired against v12 on 500 held-out frames (seed 2024). All four clear the `+3,000` gate: `N=648 +20,405`, `N=1296 +19,990`, `N=1944 +20,571`, `N=3888 +21,148`. Caveat: one seed, and `delta` landed on `5.0` for three of the four cases, which is what Stage B exists to test.
8. **Stage B — schedules, all four `R=1/2` cases.** `tools/tune_bsb_stage_b.py` replaces `xi/lambd/delta/dt` with piecewise-linear schedules on five knots (steps `0, 12, 25, 37, 49`; 20 parameters) and runs a log-space coordinate pattern search on train seed `911`. Every number below is paired frame-for-frame against the v12 baseline on the same frames; seed `2024` is the Stage A validation seed, seed `31337` has never been used for any selection.

| `N` | gain, seed 2024 | gain, seed 31337 | schedule advantage over flat, 2024 / 31337 |
|---|---:|---:|---:|
| 648 | `+21,653` | `+23,139` | `+1,248` / `+1,574` |
| 1296 | `+21,164` | `+23,845` | `+1,175` / `+1,153` |
| 1944 | `+21,080` | `+21,769` | `+814` / `+1,162` |
| 3888 | `+22,683` | `+22,826` | `+1,519` / `+1,798` |

Schedules beat the flat Stage A configuration on every case and every seed (`+814` to `+1,798`), so the effect is neither seed selection nor one lucky case. `delta` left `5.0` in every case (`4.44..5.66`), so Stage A's repeated boundary value was an artifact of searching only constants. `dt` also rose above the Stage A grid cap of `0.3` at `N=1296` (`0.345`) and `N=3888` (`0.321`), which suggests the flat grid was boundary-limited there too. The scheduled trajectory exports and compiles clean, matches the Python decoder bit-for-bit at `N=648`, and leaves the public output unchanged.
9. **v13 candidate.** `artifacts/phase9_case_manifest_v13.json` applies the four Stage A trajectories to `R=1/2` and inherits every other case from the v3 conversion of the accepted v12 manifest (`artifacts/phase9_v3_from_v12_safe.json`). The first draft inherited from the v9 approximation instead, which silently reverted v12's case-5 salvage; it showed up as an `N=1296, R=2/3` output difference against the v12 binary and is fixed. `submissions/v13_bsb_r12_candidate/submission.cpp` is `41,919` bytes, compiles clean under the strict flags, passes `validate_submission_v3.py`, and is bit-identical to v12 on the public input and on all eight dSB cases. The Stage B schedules live in `artifacts/phase9_case_manifest_v13b.json` and are not part of the v13 submission yet.

10. **Compiled-binary validation (v13c).** `tools/validate_binary_gains.py` rebuilds the exact Stage B validation frames (`real_bench.build_case`, seed 2024 and 31337, 500 frames, `noise_scale 1.001120`), writes them in judge format, and scores both compiled decoders on them, so the gain is measured between two `g++ -O2` binaries rather than taken from the Python simulator. All eight case/seed pairs reproduce the Python bit-error counts exactly (v13c 20,619/12,524 at `N=648` seed 2024 and so on; max gain delta `0.000`), with a minimum gain of `+21,080`. Runtime is far below the limit and well under v12 at the same size: `N=3888` 500 frames takes `19.7 s` against v12's `56.6 s`, `N=1944` takes `8.4 s` against `27.1 s`. Artifact: `artifacts/binary_validation.json`.

11. **v13c accepted.** `artifacts/phase9_case_manifest_v13c.json` applies the Stage B schedules to all four `R=1/2` cases. The judge scored it `1,700,646`, up `93,434` from v12. The eight high-rate cases remained bit-identical to v12, so the entire gain came from the four bSB trajectories. `submissions/v13c_scheduled_r12_candidate/` is the accepted, self-contained archive and is now the base for every candidate.

At the time, v13c was the best accepted source; it has since been superseded by v14 and v15. Do not reuse v11 case 6, and do not treat v10/v11 synthetic evidence as proof of expected judge gains.

## Phase 10 tooling

The high-rate cases still use v12 dSB portfolios. Phase 10 generalizes the Stage A/B pipeline to `R=2/3` and `R=3/4` before building a v14 candidate:

- `tools/baseline_cache.py` runs the compiled v13c binary once per `(N, K, seed)` and stores per-frame errors under `artifacts/baseline_cache/`, so high-rate tuning does not repeatedly decode six dSB trajectories in Python.
- `tools/binary_io.py` holds the judge-format input/output helpers shared by the cache and binary validator.
- `tools/tune_bsb_stage_a.py` now uses rate-dependent anchors and search ranges. `R=1/2` keeps the historical ranges; higher rates use `xi 0.15–1.5`, `lambd 3–20`, `delta 1.5–6`, `tau 1–6`, and `dt 0.15–0.35`.
- `tools/tune_bsb_stage_b.py` reads the baseline cache for high-rate cases, validates Stage A identity, and reports whether schedules beat flat Stage A on both seeds.
- `tools/build_manifest_from_stages.py` replaces only named Stage A/B cases in the v13c base and refuses to write if any non-target case differs.
- `tools/validate_binary_gains.py` accepts Stage B artifacts of any rate and reads the baseline cache instead of rerunning v12 on high-rate cases.
- v3 Python and C++ dispatch now use an exact `(n, k)` match and the nearest configured SNR, rather than falling back outside `0.25 dB`.

The refactor was checked by rerunning Stage A for `N=648, R=1/2`: gain `+20,543` against the original `+20,405`, a `+138` difference inside the required `±200` tolerance. The selected parameters drifted slightly, which appears to come from the historical tuner version that produced the original artifact, but the held-out gain reproduces.

All 16 high-rate baseline caches (eight cases × seeds `2024` and `31337`, 500 frames each) have been generated from `artifacts/build/v13c.exe`. A two-trial Stage A smoke test at `N=1296, R=2/3` using the crude anchor scored `+19,052` on seed `2024`, consistent with the earlier order-of-magnitude probe.

Full Stage A results on seed `2024`:

| `N` | `R=2/3` | `R=3/4` |
|---:|---:|---:|
| 648 | `+22,950` (dt on upper bound; widened rerun running) | `+9,181` |
| 1296 | `+24,240` | `+12,845` |

All clear the `+3,000` gate. The only boundary hit was `dt=0.35` at `N=648, R=2/3`; the required one-time `1.4×` rerun produced `+22,252` with no edge hit, so that wider result is the candidate for Stage B. Stage A at `N=1944` passed cleanly at `+23,740` (`R=2/3`) and `+13,295` (`R=3/4`). Stage B at `N=1944, R=3/4` failed the both-seed rule (`+13,162` vs flat `+13,295` on seed 2024, `+13,650` vs `+13,515` on seed 31337), so that case keeps flat Stage A. Stage B at `N=1944, R=2/3` passed (`+24,293` / `+25,812` vs flat `+23,740` / `+25,521`). All eight Stage A cases now have passing results. The `N=3888, R=2/3` widened rerun gained `+26,176` with no boundary hit; its original run gained `+25,529` but hit `dt=0.35`. `N=3888, R=3/4` gained `+14,227`. Stage B at `N=3888, R=3/4` failed the both-seed rule (`+14,214` vs flat `+14,227` on seed 2024, `+14,491` vs `+14,390` on seed 31337), so that case keeps flat Stage A. Stage B at `N=3888, R=2/3` passed (`+26,225` / `+26,105` vs flat `+26,176` / `+25,688`).

The final v14 manifest replaced all eight high-rate cases: flat Stage A for `N=648` at both rates, `N=1944, R=3/4`, and `N=3888, R=3/4`; scheduled Stage B for `N=1296` at both rates and `N=1944/R=2/3` and `N=3888/R=2/3`. The four `R=1/2` cases remained bit-identical to accepted v13c. `submissions/v14_high_rate_candidate/` is the self-contained accepted archive.

The full submission checklist is recorded in `artifacts/v14_submission_checklist.json`: strict C++17 compile, `49,009` bytes, 50 steps, at most 6 trajectories, no forbidden tokens, Python/C++ bit parity on all 12 cases, public BER `0.019217` below `0.024785`, and a fresh seed `777` gain of `+8,896` on the weakest changed case with peak candidate runtime `15.7s`. The predicted judge score was `1,847,584`; the judge returned `1,847,842`, an error of `258` points.

One tooling note: the coordinate-polish step clips `dt` to the configured range but multiplies the other parameters by a log-normal factor without clipping. The final widened `N=3888, R=2/3` config therefore has `tau=7.60`, slightly above its nominal `7.0` widened high bound. This is still a valid bSB parameter; the boundary gate checks exact range hits in random search, and no such hit occurred.

Stage B at `N=648, R=3/4` found only `+386` of train gain and did not beat flat Stage A on both seeds (`+9,126` vs `+9,181` on seed 2024). Stage B at `N=648, R=2/3` also failed the both-seed rule (`+22,852` vs flat `+22,252` on seed 2024, but `+23,725` vs `+24,034` on seed 31337). Both cases kept flat Stage A. Stage B at `N=1296` passed at both rates: `R=2/3` gained `+24,250` / `+25,172` vs flat `+24,240` / `+24,959`, and `R=3/4` gained `+13,217` / `+14,214` vs flat `+12,845` / `+13,637`.

v14 was submitted with explicit authorization and accepted at `1,847,842`. It was the base for v15.

## Stop condition

Do not submit a structural candidate until it clears the real-bench gate on the v15 base, including fresh-seed checks and, for a changed case 9, a public BER below the accepted v15 value `0.017493827`. Preserve accepted sources and archive each version separately. Submit only after explicit approval.
