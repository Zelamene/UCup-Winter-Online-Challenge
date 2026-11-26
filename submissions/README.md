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
| v9 | `v9_r34_refined_accepted` | `1,606,841` | v8 with the other three `R=3/4` cases refined |
| v10 | `v10_pass2_r23_regressed` | `1,605,473` | accepted but below best: case 11 regressed despite positive synthetic cross-seed checks |
| v11 | `v11_rebased_accepted_no_gain` | `1,606,575` | accepted but no improvement: case 5 gained 371, case 6 lost 637 |
| v12 | `v12_safe_case5_candidate` | `1,607,212` | accepted new best: v9 + case-5-only salvage, gain reproduced a third time |
| v13c | `v13c_scheduled_r12_candidate` | `1,700,646` | accepted new best: scheduled bSB for all four `R=1/2` cases |
| v14 | `v14_high_rate_candidate` | `1,847,842` | accepted new best: all eight high-rate cases tuned with bSB; `+147,196` vs v13c |
| v15 | `v15_onsager_bsb` | `2,015,823` | accepted new best: Onsager/TAP correction on all 12 bSB cases; `+167,981` vs v14 |

## Rebuild

The v2/v3/v4 manifests reference parameter files under `artifacts/` relative to the repository root. Generate a submission with:

```bash
python tools/export_submission_cpp_v2.py --manifest artifacts/phase7_case_manifest_v4.json --output submissions/v4_hybrid_probe/submission.cpp
```

All versions use QAIA dSB/bSB only, at most 50 iterations and at most six initial guesses. bSB is ballistic SB with continuous spin values in the check gradient; no message passing or post-processing is used. v15 adds an Onsager/TAP reaction term that is the gradient of an Ising-invariant polynomial; its BP-adjacent lineage and QAIA argument are documented in `v15_onsager_bsb/README.md`.
