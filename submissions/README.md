# Submission versions

Each version is self-contained in its own folder.

| Version | Folder | Judge score | Outcome |
|---|---|---:|---|
| v1 | `v1_accepted_probe` | `1,515,872` | Original accepted single-trajectory dSB decoder |
| v2 | `v2_per_case_all` | `1,579,558` | Per-case tuning on all 12 cases; high rates improved, `R=1/2` regressed |
| v3 | `v3_hybrid_fallback` | `1,588,472` | Restored old fallback for `R=1/2`, but used the wrong local baseline |
| v4 | `v4_hybrid_probe` | `1,593,920` | Accepted hybrid: probe decoder for `R=1/2`, per-case tuning elsewhere |
| v6 | `v6_case11_refined` | `1,597,233` | v4 with refined `N=3888, R=2/3` case only |
| v7 | `v7_r23_refined_accepted` | `1,600,231` | v6 with all three remaining `R=2/3` cases refined |
| v8 | `v8_public_tuned_accepted` | `1,601,303` | v7 with case 9 tuned directly on public input |
| v9 | `v9_r34_refined_candidate` | pending | v8 with the other three `R=3/4` cases refined |

## Rebuild

The v2/v3/v4 manifests reference parameter files under `artifacts/` relative to the repository root. Generate a submission with:

```bash
python tools/export_submission_cpp_v2.py --manifest artifacts/phase7_case_manifest_v4.json --output submissions/v4_hybrid_probe/submission.cpp
```

All versions use QAIA/dSB only, at most 50 iterations and at most six initial guesses.
