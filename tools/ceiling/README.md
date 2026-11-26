# Phase 13 ceiling analysis (offline only)

Every file in this directory is an **offline measuring stick**. Nothing here may be
imported by an exporter, included in a manifest, or compiled into a submission.
BP, OSD, list-MAP, and other non-QAIA decoders are allowed only here.

Every Python tool starts with:

```python
# OFFLINE ANALYSIS ONLY — never imported by the exporter
```

`tools/validate_submission_v3.py` rejects generated source containing these
symbols so an offline tool cannot leak into a submission.

## Fixed bench rules

- Frames: `real_bench.build_case(..., noise_scale=1.00112)`
- Seeds: `2024`, `31337`, `6060`
- Frames per case per seed: `500`
- Deltas and totals: paired frame bootstrap, 95% percentile CIs

## Tools

- `cache_binary.py`: run one offline binary on all fixed-rule cases and store
  per-frame information-bit errors.
- `compare_ceiling_caches.py`: compare two caches with paired bootstrap CIs.
- `calibrate_bench_vs_judge.py`: fit fixed-rule v15 bench scores to the accepted
  v15 judge per-case scores.
- `check_offline_isolation.py`: verify headers and exporter isolation.

## Typical use

```bash
python tools/ceiling/cache_binary.py --binary <offline-binary> --cache-dir artifacts/ceiling_cache_bp
python tools/ceiling/compare_ceiling_caches.py \
  --baseline artifacts/baseline_cache_v15 \
  --candidate artifacts/ceiling_cache_bp \
  --out artifacts/phase13/bp_vs_v15.json
python tools/ceiling/calibrate_bench_vs_judge.py
python tools/ceiling/check_offline_isolation.py
```

Step 0 result: `2,517,863` is the live top score on the 2026 UCup Winter
Challenge scoreboard (`haminh109`), not a value derivable from the local bench.
The official score is `-Σ_a 10·lg(BER_a)`, scaled by `10^4` by the judge, over
12 hidden datasets. The scoring dataset is not public.
