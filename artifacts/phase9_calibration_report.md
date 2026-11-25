# Phase 9 bench calibration report

## Public channel calibration

Source: `artifacts/phase9_public_calibration.json`.

- Public info raw BER: `0.0535336`
- Public even/odd bit-position BER: about `3.54% / 7.16%`
- Public `H`: row weight `7`, info column weight `2`, parity block `I`
- Calibrated `noise_scale`: `1.001120`
- Simulated BER at that scale: `0.053729`

The generated bench matches the public channel closely enough to replace the old resampled-public-LLR surrogate.

## Historical sign calibration

Command:

```powershell
python tools/evaluate_history_on_real_bench.py --samples 100 --noise-scale 1.001120 --versions v1,v2,v3,v4,v6,v7,v8,v9,v10,v11 --out artifacts/phase9_history_bench_100.json
```

Result: `106` sign matches, `2` mismatches, `12` skipped baseline cases.

Bench totals:

| Version | Bench total |
|---|---:|
| v1 | 1,511,773 |
| v2 | 1,573,948 |
| v3 | 1,582,589 |
| v4 | 1,587,805 |
| v6 | 1,589,549 |
| v7 | 1,592,748 |
| v8 | 1,594,469 |
| v9 | 1,604,423 |
| v10 | 1,603,997 |
| v11 | 1,603,760 |

Critical checks:

- v2 `R=1/2` regression vs v1: reproduced.
- v10 case 11 regression vs v9: reproduced.
- v11 case 6 regression vs v9: reproduced.
- Only mismatch: case 5, where the judge gave `+371` but the 100-frame bench gave `-191`.

A 500-frame v9/v10/v11 rerun resolved most of the ambiguity:

| Version | Bench total | Judge total |
|---|---:|---:|
| v9 | 1,603,236 | 1,606,841 |
| v10 | 1,601,850 | 1,605,473 |
| v11 | 1,604,858 | 1,606,575 |

Per-case deltas versus v9:

| Case | Judge v10 | Bench v10 | Judge v11 | Bench v11 |
|---|---:|---:|---:|---:|
| 5 | `+371` | `+372` | `+371` | `+372` |
| 11 | `-1,739` | `-1,759` | `0` | `0` |
| 6 | `0` | `0` | `-637` | `+1,250` |

So the 100-frame case-5 mismatch was sampling noise. The bench reproduces the v10 case-5 and case-11 deltas almost exactly. It does **not** reproduce the v11 case-6 regression, so marginal gains around `~1,000` points still require judge validation.

The subsequently submitted v12 scored `1,607,212`, exactly the predicted `+371` over v9 from three independent judge runs. This further confirms that case 5 is a real, repeatable gain and that the bench is useful for this class of portfolio swap.

## Interpretation

The bench is now historically credible for large effects and for the v10 deltas, but it is not a perfect oracle for every marginal case. It should be used for structural ranking and regression screening, not for claiming exact judge-point magnitudes.
