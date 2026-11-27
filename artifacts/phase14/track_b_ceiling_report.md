# Phase 14 Track B — joint-likelihood ceiling ladder

**Offline measuring stick only.** No exporter, submission, judge or gate is touched.

Rules: 500 frames/seed for the full ladder, seeds [2024, 31337, 6060], noise_scale 1.00112, paired frame bootstrap (2000 replicates). The three cases omitted from the original ladder were screened with 100 frames on seed 2024 (`±3–5k` sampling uncertainty), so they are marked as short screens rather than mixed into the 500-frame confidence intervals.

## Quick screen (100 frames, seed 2024)

| N | K | per-bit | joint | joint−perbit | per-bit OSD-2 | OSD-2−per-bit | joint OSD-2 | OSD-2−joint |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 648 | 324 | 162,405 | 167,107 | +4,701 | — | — | — | — |
| 648 | 432 | 185,806 | 187,056 | +1,250 | — | — | — | — |
| 1296 | 648 | 175,050 | 188,730 | +13,680 | — | — | — | — |
| 1944 | 972 | 197,864 | 209,557 | +11,693 | 197,949 | +85 | 210,061 | +504 |
| 1944 | 1296 | 195,304 | 194,706 | -597 | 194,205 | -1,099 | 193,855 | -852 |
| 3888 | 1944 | 214,071 | 224,025 | +9,954 | 213,763 | -308 | 223,870 | -155 |
| 3888 | 2592 | 189,988 | 191,025 | +1,037 | 189,684 | -305 | 190,618 | -407 |

The first three rows are the added 100-frame screens for cases 1, 2 and 4. At this sample size their `joint−perbit` deltas should be read as `±3–5k`, not as ladder-grade estimates. Machine-readable rows: `artifacts/phase14/track_b_bp_screen100_cases124_manifest.json`.

## Per-case scores and joint gains

| Case | N | K | Rate | v16 bench | per-bit BP | joint BP | joint−perbit (95% CI) | joint−v16 (95% CI) | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 648 | 324 | 0.500 | 128,400 | 162,405* | 167,107* | +4,701* (100 frames, ±3–5k) | n/a* | joint_bp 167,107* |
| 2 | 648 | 432 | 0.667 | 176,543 | 185,806* | 187,056* | +1,250* (100 frames, ±3–5k) | n/a* | joint_bp 187,056* |
| 3 | 648 | 486 | 0.750 | 183,898 | n/a | n/a | n/a | n/a | v16 183,898 |
| 4 | 1296 | 648 | 0.500 | 132,416 | 175,050* | 188,730* | +13,680* (100 frames, ±3–5k) | n/a* | joint_bp 188,730* |
| 5 | 1296 | 864 | 0.667 | 180,930 | n/a | n/a | n/a | n/a | v16 180,930 |
| 6 | 1296 | 972 | 0.750 | 183,438 | n/a | n/a | n/a | n/a | v16 183,438 |
| 7 | 1944 | 972 | 0.500 | 140,530 | 184,249 | 196,307 | +12,058 [+9,489, +15,027] | +55,777 [+51,205, +60,775] | joint_bp 196,307 |
| 8 | 1944 | 1296 | 0.667 | 179,689 | 191,090 | 191,593 | +503 [+166, +826] | +11,905 [+10,881, +12,941] | joint_bp 191,593 |
| 9 | 1944 | 1458 | 0.750 | 184,274 | n/a | n/a | n/a | n/a | v16 184,274 |
| 10 | 3888 | 1944 | 0.500 | 148,979 | 213,966 | 234,520 | +20,554 [+16,733, +24,687] | +85,541 [+79,562, +91,798] | joint_bp 234,520 |
| 11 | 3888 | 2592 | 0.667 | 181,795 | 194,461 | 195,241 | +780 [+515, +1,047] | +13,446 [+12,665, +14,239] | joint_bp 195,241 |
| 12 | 3888 | 2916 | 0.750 | 185,023 | n/a | n/a | n/a | n/a | v16 185,023 |

`*` marks a 100/60-frame short screen on seed 2024, with `±3–5k` uncertainty. The `joint−v16` comparison is intentionally `n/a` for those rows because the full ladder baseline is 500 frames per seed and is not frame-matched to the quick screen.

## Totals and ceiling estimate

- Accepted v16 judge score: 2,023,997. Accepted v17 judge score: 2,038,794; v17 is the Phase 15 base.
- Fixed-bench v16 total: 2,005,914; best measured full-ladder total: 2,172,583.
- Judge-scaled full-ladder delta (slope 0.947000): +157,835.
- Judge-scaled full-ladder estimate: 2,181,832.
- **Corrected partial headline with the three added R=1/2 cases: `≈ 2.25M (partial; 3/4 cases ML-limited)`.** This is a partial, noisy extrapolation, not a formal ceiling.
- Live top 2,517,863 remains far above this partial estimate; the unmodelled gap is not evidence that it is reachable by a QAIA-compliant decoder.
- Joint vs per-bit total delta over the four measured full-ladder cases: +33,895 [+29,065, +38,836].
- Joint vs v16 total delta over the four measured full-ladder cases: +166,669 [+159,370, +174,493].

## v18 trigger decision (N=3888, R=2/3)

**promote_v18 = False**.

- joint-vs-perbit point 780 (95% CI low 515)
- joint-vs-v16 point 13,446 (95% CI low 12,665)
- required >5k joint gain with positive CI and v16 superiority not met

## OSD on joint BP

BP-guided OSD-2 on joint BP was rejected at rate 1/2. On the 100-frame side-by-side screen the small positive/negative deltas (`+504` on case 7, `−155` on case 10) disappeared in the longer check: joint OSD-2 lost roughly `−5k` to `−10k`. The reason is that ML selection over the OSD candidate set picks a wrong codeword in `0.34 / 0.70 / 0.58` of the sampled cases; exact-pair ML is not reliable enough at these operating points to help a compliant decoder. Order-3 OSD and list-MAP L=64–256 were not run, so no such ceiling is claimed.

## Search methods attempted

- Per-bit BP: original independent per-bit LLRs, sum-product check updates.
- Joint BP: exact `tools/qam_pairs.py` pair tables with the `d*s0*s1` factor retained as a pairwise channel factor. Self-check: `pass` at SNR 6/9/10.5 dB.
- BP-guided OSD: order 2 over the 64 least reliable information bits. Per-bit OSD scores candidates with per-bit LLRs; joint OSD uses the exact pair metric. The screen table above is a 100-frame side-by-side screen; the longer joint-OSD check is the basis for the rejection.
- Added short screen: 100 frames, seed 2024, cases 1/2/4, both per-bit and joint variants, `tools/ceiling/bp_variants.py`.

## Interpretation

The measured offline decoders support a partial headline of about `2.25M`, with 3 of 4 sampled rate-1/2 cases already ML-limited. These are achievable lower bounds, not upper bounds; the live top `2.517M` remains unmodelled by the BP/OSD variants tested here.

The exact pair factor uses `tools/qam_pairs.py` `(a, b, d)` including the `d*s0*s1` coupling; the per-bit variant uses the original independent LLRs. BP/OSD are achievable decoders, so these numbers are lower bounds on the true ML ceiling, not formal upper bounds.

Machine-readable report: `artifacts/phase14/track_b_ceiling_report.json`.
