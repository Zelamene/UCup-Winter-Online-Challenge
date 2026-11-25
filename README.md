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
| Best accepted | v9, `1,606,841` |
| Last submitted, rejected | v10, `1,605,473` (`-1,368` vs v9) |
| Pending candidate | v11 rebased on v9, combining v10's case-5 gain and v11's case-6 refinement |
| Public input dispatch | `N=1944, R=3/4, SNR≈10.5` |
| Public BER, v8–v10 | `0.024784636` |
| Accepted per-case scores, v9 | `[90167,144269,171480,90957,143207,167094,89867,142847,167085,89806,142852,167210]` |

The accepted architecture is:

- exact v1 probe decoder for all `R=1/2` cases;
- per-case six-trajectory dSB/QAIA energy portfolios for `R=2/3` and `R=3/4`;
- dispatch by `(N, K, SNR)`;
- decoder lambda and energy-selection lambda are separate parameters;
- 50 steps, at most 6 trajectories, QAIA/dSB only.

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
| v10 | `1,605,473` | `-1,368` | rejected; case 11 regressed despite positive synthetic checks |

Rate-level totals show where all progress came from:

| Version | `R=1/2` | `R=2/3` | `R=3/4` | Total |
|---|---:|---:|---:|---:|
| v1 | `360,797` | `540,445` | `614,630` | `1,515,872` |
| v4 | `360,797` | `566,864` | `666,259` | `1,593,920` |
| v6 | `360,797` | `570,177` | `666,259` | `1,597,233` |
| v7 | `360,797` | `573,175` | `666,259` | `1,600,231` |
| v8 | `360,797` | `573,175` | `667,331` | `1,601,303` |
| v9 | `360,797` | `573,175` | `672,869` | `1,606,841` |
| v10 | `360,797` | `571,807` | `672,869` | `1,605,473` |

`R=1/2` has never improved over v1. Every durable gain came from high-rate per-case tuning.

## Strategies that worked

1. **Hybrid dispatch.** Keeping the exact accepted probe decoder for `R=1/2` and using tuned portfolios only for high rates produced the largest single jump (v4).
2. **Per-case, not global, configuration.** Each `(N,R,SNR)` operating point has its own learned trajectory and fixed candidates. Broad shared tuning repeatedly underperformed.
3. **Grid + local search around the current tuned region.** v6–v9 all came from narrower grids around known-good dSB parameters, followed by local search and greedy portfolio replacement.
4. **Six-trajectory energy portfolios.** Multiple dSB trajectories with energy-based selection consistently beat single trajectories at high rates.
5. **Separating decoder strength from selection strength.** Decoder `lambda` and energy-selection `lambda` are distinct; v6 established this and later versions rely on it.
6. **Direct public-data tuning for case 9.** With 500 public frames and labels, a 250/250 selection-validation split produced a real `+1,072` judge gain (v8).
7. **One-case-at-a-time changes.** Every successful version changed one or a tight group of cases, making judge deltas attributable.
8. **Cross-seed robustness checks.** They filtered out several apparent gains (v10 `N=648`, v11 cases 3, 9, 12) before submission. They are useful but not sufficient.

## Assumptions and directions that proved wrong

1. **Synthetic `6 dB` extrapolation for `R=1/2`.** v2's per-case tuning reversed the judge ranking for all four `R=1/2` cases. The probe decoder remains frozen.
2. **"Fallback" means accepted baseline.** v3 used the wrong local `submission.cpp` instead of the judge-accepted probe, losing `5,448` points versus v1 on `R=1/2`.
3. **Broad energy-lambda tuning.** The v5 global energy retune worsened public BER and was rejected before submission.
4. **Synthetic BER ratios as score predictors.** Large ratios translated into much smaller judge gains, and small ratios sometimes reversed sign. v7 ratios around `1.37–1.53` produced gains from `+265` to `+1,420`; v10's `N=3888` cross-seed-positive retune produced `-1,739`.
5. **Energy-only retuning at `N=3888, R=2/3`.** v10 case 11 changed only energy-selection lambdas and regressed badly. This is the strongest evidence that synthetic energy selection can overfit channel/model mismatch at large `N`.
6. **Marginal cross-seed gains.** Ratios near `1.00–1.02` are not reliable. The rejected v11 cases 3, 9, and 12 and v10 case 8 all looked weakly positive locally but did not justify submission risk. The accepted-quality signal is closer to `>=1.03` consistently across seeds, and even that is not a guarantee.
7. **Public BER as a global proxy.** The public input dispatches only to case 9. Changes to the other 11 cases leave public output bit-identical, so public BER cannot validate them.
8. **Attributing gains to large trajectory changes.** v10 case 5 gained only `+371` despite a `1.085` tuning-seed ratio; v10 case 11 regressed despite multi-seed robustness. Judge transfer remains the final authority.
9. **Combining independently positive case changes blindly.** v10 combined a positive case 5 change with a misleading case 11 change and net regressed. Cases must be judged independently and regressions must be removed before combining.

## v10 post-mortem

| Case | Change | Judge delta |
|---|---|---:|
| 5 `N=1296, R=2/3` | second-pass portfolio replacement | `+371` |
| 11 `N=3888, R=2/3` | energy-lambda-only retune | `-1,739` |
| total |  | `-1,368` |

Lesson: salvage case 5, discard case 11. Any successor to v10 must be rebuilt on accepted v9, not on v10.

## Current knowledge base and tools

All work lives on branch `optimize-phase8-case11`. Version folders are under `submissions/`, each with its own source, manifest, summaries, params, and judge result where available.

Key tools:

- `tools/phase7_per_case.py` — original synthetic grid/local/portfolio search.
- `tools/tune_public_case.py` — direct tuning on the 500-frame public case.
- `tools/refine_accepted_case.py` — compares new portfolios directly against the current accepted manifest portfolio.
- `tools/compare_accepted_summary.py` — independent-seed robustness checks.
- `tools/build_targeted_manifest.py` — replaces one case in a manifest from a tuned summary.
- `tools/export_submission_cpp_v2.py` — emits C++17 submission from a manifest.
- `tools/validate_submission_v2.py` — checks size, dispatch count, guess count, binary output, and public BER.
- `tools/decode_v2_python.py` — Python reference decoder for C++ parity checks.

Validation gate for every candidate:

- clean `g++ -std=c++17 -O2 -Wall -Wextra -pedantic` compile;
- source under `512 KiB`;
- at most 6 trajectories and 50 iterations;
- public output not worse than accepted;
- Python/C++ bit parity on synthetic cases for every changed case;
- public dispatch cases remain bit-identical unless public tuning is intentional.

## Next step

Submit `submission_v11.cpp` from `submissions/v11_rebased_candidate/`. It contains only the two individually supported changes:

- case 5 `N=1296, R=2/3` from v10 (`+371` judge gain);
- case 6 `N=1296, R=3/4` from the second-pass refinement (consistent synthetic cross-seed ratios around `1.02`).

Do not carry v10's case 11 change forward. Future work should prefer trajectory-changing refinements with consistent multi-seed ratios, avoid energy-only micro-tuning at `N=3888`, and continue testing one or two cases at a time so judge feedback remains attributable.
