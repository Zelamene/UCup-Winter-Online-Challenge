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
| Last submitted | v11, `1,606,575` (`-266` vs v9) |
| Active decision | stop after documentation; do **not** proceed to v12 |
| Public input dispatch | `N=1944, R=3/4, SNR≈10.5` |
| Public BER, v8–v11 | `0.024784636` |
| v9 per-case scores | `[90167,144269,171480,90957,143207,167094,89867,142847,167085,89806,142852,167210]` |
| v11 per-case scores | `[90167,144269,171480,90957,143578,166457,89867,142847,167085,89806,142852,167210]` |
| v11 runtime | `27,613ms` total, `5,784ms` peak — well under the limit |

The best accepted architecture remains:

- exact v1 probe decoder for all `R=1/2` cases;
- per-case six-trajectory dSB/QAIA energy portfolios for `R=2/3` and `R=3/4`;
- dispatch by `(N, K, SNR)`;
- decoder lambda and energy-selection lambda are separate parameters;
- 50 steps, at most 6 trajectories, QAIA/dSB only.

Judge status `Accepted` means the submission was valid. It does not mean the score improved; v10 and v11 were both accepted but scored below v9.

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

`R=1/2` has never improved over v1. Every durable gain came from high-rate per-case tuning. v11 recovered the v10 case-11 loss, kept the v10 case-5 gain, and lost the same amount again through case 6.

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
2. **Case 5 is the only repeatable judge-validated gain after v9.** It produced `+371` in both v10 and v11. A case-5-only successor would mathematically be `1,607,212`, but no such v12 was built or submitted.
3. **Case 6 should be discarded.** Its v11 score fell `167,094 → 166,457`. The synthetic improvement signal was not merely weak on one seed; it was consistent across three comparisons and still wrong on the judge.
4. **Small synthetic gains are noise for this task.** Multi-seed ratios around `1.02` should not be used to justify a submission. Even a ratio near `1.06` failed at large `N` in v10.
5. **The public proxy remains blind.** v11's public output was bit-identical to v9, so public BER offered no evidence about cases 5 and 6.
6. **Runtime and memory were never the limiting factor.** v11 used `27,613ms` total and peaked at `5,784ms`, far below the `2 minute` per-case limit; the loss was entirely BER/scoring.
7. **Do not combine a validated small gain with an unvalidated synthetic gain.** The one reliable `+371` was overwhelmed by the `-637` case-6 change.

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

## Stop condition

No v12 was built or submitted. Keep v9 as the best accepted source. Future work should not reuse v11 case 6 and should treat v10/v11 synthetic evidence as historical context only, not as proof of expected judge gains.
