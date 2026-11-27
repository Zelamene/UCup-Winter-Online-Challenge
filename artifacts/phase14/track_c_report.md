# Phase 14 Track C — structural variants on v17 joint pairs

Baseline: v17 `best_config` from `joint_N1296_R05.json`, `joint_N1944_R05.json`,
`joint_N3888_R05.json` (held unsubmitted). Screening: 100 frames per seed,
seeds 2024 and 6060, `noise_scale=1.00112`, R=1/2 at N=1296 and N=3888.
Each variant's new parameters were retuned on seed 2024 only; the chosen
candidate was then scored on both seeds against the identical v17 decode.
Gains use 1,000 paired whole-frame bootstrap resamples.

Promotion bars: **+4,000** at N=1296 and **+8,000** at N=3888 on every seed,
with a positive 95% paired-bootstrap CI lower bound on every seed.
**No variant was promoted.** `promoted_variants = []`.

## Results

| Variant | N | Chosen retune | Gain s2024 (95% CI) | Gain s6060 (95% CI) | Promoted |
|---|---:|---|---:|---:|:---:|
| C1 per-bit classes | 1296 | `xi_classes=[1.0, 0.9]` | +2,650 [−1,084, +6,618] | −1,094 [−4,391, +1,898] | no |
| C1 per-bit classes | 3888 | `xi_classes=[0.9, 1.0]` | +3,201 [−394, +7,177] | +487 [−3,512, +4,825] | no |
| C2 exact pair TAP | 1296 | `pair_exact=1`, gamma ×0.4 | −399 [−3,416, +2,523] | −3,448 [−6,983, −428] | no |
| C2 exact pair TAP | 3888 | `pair_exact=1`, gamma ×0.2 | −1,529 [−5,581, +1,922] | −2,868 [−6,466, +612] | no |
| C3 amplitude coordinate | 1296 | `amp_scale=1.2` | −29,717 [−37,522, −23,316] | −27,264 [−33,945, −21,721] | no |
| C3 amplitude coordinate | 3888 | `amp_scale=1.2` | −44,679 [−52,994, −38,054] | −47,705 [−54,686, −41,203] | no |
| C4 pair_scale ramp | 1296 | 10-step ramp | −8,065 [−13,453, −3,252] | −9,576 [−14,789, −4,380] | no |
| C4 pair_scale ramp | 3888 | 10-step ramp | −10,303 [−16,361, −4,919] | −11,846 [−18,272, −6,463] | no |

Full per-candidate retune tables, baseline scores, error counts and CIs are in
`c1_screen_N1296.json`, `c1_screen_N3888.json`, and the corresponding
`c2`/`c3`/`c4` artifacts. `track_c_report.json` is the machine-readable summary.

## Variant notes and compliance

**C1 — per-bit-position classes.** Sign-bit (positions 0, 2, …) and magnitude-bit
(positions 1, 3, …) multipliers were screened over ±10% on tau, xi and beta,
plus two small combined moves. Best gains were below +3,300 and no CI lower
bound was positive on both seeds. This is QAIA-only: the multipliers only
rescale the existing tau initialisation, xi force gain and beta Onsager
gradient; no new state, messages, hard decisions or post-processing appear, and
the run stays at 50 steps.

**C2 — exact pair Onsager form.** For the coupled Gray pair with
`J = 2*pair_scale*d`, the two-spin Plefka/TAP expansion through O(J³) gives

```
R_i = -gamma * J^2 (1-x_j^2) x_i
      - gamma * (2/3) J^3 (1-x_j^2) x_j (3 x_i^2 - 1) + O(J^4)
```

which is `-dV/dx_i` for

```
V = -0.5*gamma*J^2*(1-x_i^2)*(1-x_j^2)
    + (2/3)*gamma*J^3*x_j*(x_i^3-x_i)*(1-x_j^2)
```

Both `V` and `R_i` vanish at `x_i, x_j = ±1`, so the Ising objective and its
solutions are unchanged. The O(J³) term is the cross term derived from the
exact two-spin marginal; a unit test checks the gradient numerically and the
corner values. The exact form was unstable at the tuned gamma (large negative
gains); retuning gamma to 20–40% of baseline removed the instability but did
not clear the bar. No per-edge state is stored.

**C3 — pair-level amplitude coordinate.** Implemented as a screening variant:
one continuous soft PAM amplitude per I/Q Gray pair, `a = -x_sign(2+x_mag)` in
sqrt(10) units, mapped back to the pair's soft spins through a maximum-entropy
four-level family whose inverse is solved by Newton steps. The map is used only
in the pair channel/reaction terms, fixes all four hard Gray corners, carries
no per-edge state, and takes no hard decision before the final step. It is
QAIA-only in structure but empirically destructive (−27k to −48k). The underlying
obstacle is that one amplitude coordinate cannot preserve both Gray marginals:
near levels ±1 the mapped sign-bit magnetization is only about 0.47 instead of
1, so the balanced pair/check field of v17 is lost. A faithful amplitude
coordinate would need a second degree of freedom, which would no longer be
"one coordinate per component"; no promoted candidate came from this route.

**C4 — joint field schedule.** Linear `pair_scale` ramp from 0 to the exact
value 1 over the first 10 or 20 of 50 steps, screening both lengths. Both ramps
lost ground on both seeds; the coupling needs to be present from step 0. This
remains a time-dependent coupling in the same Hamiltonian gradient, with no new
state or post-processing.

**C5 — gradient deep unfolding.** Blocked: `python -c 'import torch'` raises
`ModuleNotFoundError: No module named 'torch'`. No packages were installed, and
no deep-unfolding experiment was run.

## Verification and integration status

- `python -m unittest discover -s tools/tests -p "test_*.py"` → **11 tests OK**,
  including neutral-class/constant-schedule bit identity, C2 gradient/corner
  test, and C3 amplitude-map consistency.
- `python -m py_compile` passed on all changed tools.
- Scalar v17 joint decode stayed bit-identical after the decoder extension:
  3-frame SHA-256 for N=1296 `618dbcf99837fc65f4c13781c35d6500b816628b0ff74b3bafaf0453924dbd43`
  and N=3888 `2f132e30a48a4d5b6b14f642774287e1eb6510d427a4b534331cb00bdc10676e`
  matched the pre-extension decoder, with identical error counts (31, 270).
- No exporter change is needed because nothing was promoted. If a future
  candidate were promoted, C1 would need manifest class multipliers, C2 a
  `pair_exact` flag plus the O(J³) cross term, C3 an amplitude-map/inverse, and
  C4 a per-step `pair_scale` schedule in the C++ exporter. None were integrated
  into v17; no full gate, judge run, commit, or push was performed.

## Files changed

- `tools/bsb_vectorized.py` — default-off extensions for C1–C4 and the C3
  amplitude helper.
- `tools/screen_joint_structural.py` — new screening tool.
- `tools/tests/test_bsb_vectorized.py` — new focused tests.
- `artifacts/phase14/c{1,2,3,4}_screen_N{1296,3888}.json` — eight screens.
- `artifacts/phase14/track_c_report.md`, `track_c_report.json` — this report.
