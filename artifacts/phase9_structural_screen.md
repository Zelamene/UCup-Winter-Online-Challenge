# Phase 9 structural screen — case 1 (`N=648`, `R=1/2`)

All runs use the realistic bench with `noise_scale=1.001120`, 80 training frames, 100 held-out frames, seeds `911`/`2024`, and five trajectories unless noted.

| Variant | Combine | Train score | Held-out score | Result |
|---|---|---:|---:|---|
| v9 approximate baseline (learned schedules) | energy | 89,394 | 92,295 | reference |
| bSB | vote | 84,193 | 85,936 | rejected |
| dSB | vote | 89,381 | 91,461 | rejected |
| dSB | soft_vote | 89,394 | 91,693 | rejected |
| bSB | energy | 83,995 | 85,869 | rejected |

The v9-style energy-selected dSB portfolio remains the strongest of the tested structures. Majority/soft voting loses the per-frame adaptivity of energy selection. Continuous bSB check products are clearly worse under this setup, with or without energy selection.

No tested structural replacement clears the `+1,000` held-out gate. Further work should focus on better energy selection or trajectory diversity rather than replacing dSB check products or energy selection with voting.
