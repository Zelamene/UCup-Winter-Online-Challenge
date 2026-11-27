# v18b — Phase 15 energy-selected tau/xi portfolio

Prepared candidate; **not submitted**. It remains v17 except for the changed cases listed below.

Base: accepted v17 (`submissions/v17_joint_pairs/`), judge score `2,038,794`.

Changed cases: 7, 10.

Each changed case uses up to six QAIA-compliant bSB trajectories with tuned `tau`/`xi`/`delta`/`lambd`
settings and exact C++/Python energy selection. All trajectories keep 50 steps, at most six
trajectories, no per-edge state, and unchanged `±1` Ising semantics. Public case 9 and every
unchanged case remain byte-identical to v17.

Validation from `greedy_case7_seed2024.json` (seed 2024, 500 frames):
- case 7: gain `+7430.9`, CI `[5661.1, 9474.4]`

Validation from `greedy_case7_seed31337.json` (seed 31337, 500 frames):
- case 7: gain `+5739.3`, CI `[3966.9, 7655.6]`

Validation from `greedy_case10_3_seed2024.json` (seed 2024, 500 frames):
- case 10: gain `+3908.9`, CI `[2545.8, 5322.2]`

Validation from `greedy_case10_3_seed31337.json` (seed 31337, 500 frames):
- case 10: gain `+5807.0`, CI `[4055.0, 7754.6]`

Validation from `v18_greedy_both3_fresh_seed_6060.json` (seed 6060, 300 frames):
- case 1: gain `+0.0`
- case 2: gain `+0.0`
- case 3: gain `+0.0`
- case 4: gain `+0.0`
- case 5: gain `+0.0`
- case 6: gain `+0.0`
- case 7: gain `+5802.9`, CI `[3515.9, 8366.9]`
- case 8: gain `+0.0`
- case 9: gain `+0.0`
- case 10: gain `+2747.3`, CI `[1267.9, 4383.9]`
- case 11: gain `+0.0`
- case 12: gain `+0.0`

SNR/schedule-extreme robustness: `all_passed=true` (`robustness.json`).

Runtime note: the source runs independent trajectories concurrently without changing decode
results; validator, public case-9 SHA, and scalar/batched/C++ bit identity all pass. The local
absolute `60 s` judge-estimate gate could not be certified while this machine was charging at
2% with the CPU capped to `2.1 GHz`; unchanged v17 itself measured `~62 s` instead of its normal
`~34 s` on 500-frame case 10. Under normal power this candidate is projected at roughly
`50–60 s`, but a clean final gate must be rerun before submission.

Source SHA-256: `18655192D6CF415A911E9486B1E1D94E69FD40A8DFF35FBEAA40CB76ADC88100`.
Final gate: `not completed; runtime cap unverified under low-battery power limit`.

Do not submit without explicit user approval.
