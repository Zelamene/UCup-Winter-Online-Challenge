# v18b — Accepted Phase 15 energy-selected tau/xi portfolio

Accepted submission. The judge returned `2,051,900` (`+13,106` over accepted v17). This archive
is labelled `v18b`; the submission was reported as `v18`.

Base: accepted v17 (`submissions/v17_joint_pairs/`), judge score `2,038,794`.

Changed cases: 7, 10.

Each changed case uses only QAIA-compliant bSB trajectories with tuned `tau`/`xi`/`delta`/`lambd`
settings and exact C++/Python energy selection: five trajectories on case 7 and three on case 10.
All trajectories keep 50 steps, stay within the six-trajectory cap, store no per-edge state, and
preserve the unchanged `±1` Ising semantics. Public case 9 and every other unchanged case remain
byte-identical to v17.

## Judge result

`judge_result.json` records the full accepted result.

| # | Status | Score | Time Cost | Memory Cost |
|---:|---|---:|---:|---:|
| 1 | Accepted | 129979 | 589ms | 3.5 MiB |
| 2 | Accepted | 177394 | 452ms | 3.5 MiB |
| 3 | Accepted | 185451 | 389ms | 3.5 MiB |
| 4 | Accepted | 145151 | 2142ms | 6.7 MiB |
| 5 | Accepted | 179802 | 935ms | 6.8 MiB |
| 6 | Accepted | 183299 | 817ms | 6.7 MiB |
| 7 | Accepted | 150030 | 9880ms | 10.6 MiB |
| 8 | Accepted | 184164 | 1461ms | 10 MiB |
| 9 | Accepted | 184485 | 1268ms | 10 MiB |
| 10 | Accepted | 164534 | 13801ms | 20.1 MiB |
| 11 | Accepted | 182587 | 3219ms | 19.4 MiB |
| 12 | Accepted | 185024 | 2837ms | 19.4 MiB |

Judge runtime is `389–13,801ms` per case (`37,790ms` total), comfortably below the 120-second
per-case limit; peak judge memory is `20.1 MiB`. This accepted run supersedes the earlier local
`60 s` certification caveat, which was caused by a low-battery CPU cap.

## Local validation

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

The source runs independent trajectories concurrently without changing decode results; validator,
public case-9 SHA, and scalar/batched/C++ bit identity all pass.

Source SHA-256: `18655192D6CF415A911E9486B1E1D94E69FD40A8DFF35FBEAA40CB76ADC88100`.

Status: submitted and accepted. No further submission is pending.
