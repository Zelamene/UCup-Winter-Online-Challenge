# v16 — Phase 12 per-case TAP tuning candidate

This archive was submitted and accepted by the judge at `2,023,997` (`+8,174` over v15).

Base: accepted v15 (`submissions/v15_tap_bsb_candidate/`).

Changed cases:

- case 7: `N=1944, K=972, SNR=6.0` (R=1/2)
- case 10: `N=3888, K=1944, SNR=6.0` (R=1/2)

All other cases are byte-identical to v15. The exported source is `19,964` bytes and passes `tools/validate_submission_v3.py`.

Local gates:

- Compiled-binary gain, seed 2024, 1000 frames: case 7 `+2,677`, case 10 `+5,528`
- Compiled-binary gain, seed 31337, 1000 frames: case 7 `+3,821`, case 10 `+5,759`
- Fresh seed 6060, 300 frames: case 7 `+3,327`, case 10 `+3,657`
- Public case 9 BER unchanged at `0.017493827`

Stage B′ on case 10 was dropped: its schedule search gained only `+88` on the training score, well below bench noise, so flat Stage A′ is retained.

Case 7 misses the original strict 500-frame gate on seed 31337 by `21.5` points (`+1,978`); it passes the 1000-frame re-test, which is the revised cleanup gate. If the strict 500-frame rule is required, case 7 must be dropped and v16 rebuilt with case 10 only.
