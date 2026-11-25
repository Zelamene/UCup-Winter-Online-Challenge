# v6 — case-11 refinement candidate

Status: accepted. Judge score `1,597,233`.

Base: v4 (`1,593,920`). Only the `N=3888, R=2/3, SNR=9.0` case is changed.

What changed:

- Finer grid around the current tuned dSB region: `tau 1.4–2.5`, `xi 0.45–0.55`, `lambda 5–8`, `delta 2.25–2.75`.
- Three local-search rounds.
- Separate energy-lambda tuning, so energy selection can differ from the decoder's check-penalty strength.
- Still six trajectories at most, 50 dSB iterations, QAIA/dSB only.

Validation:

- Public output is bit-identical to accepted v4 (public input dispatches to `N=1944, R=3/4`).
- Python reference decoder matches C++ on synthetic `N=3888, R=2/3` input: `0` bit mismatches.
- Source size: `83,889` bytes.
- Max guesses: `6`.
- Clean C++17 compile with `-Wall -Wextra -pedantic`.

Synthetic validation on this case improved the optimized/accepted BER ratio from about `1.263` to `1.285`.

Judge outcome: case 11 rose from `139,539` to `142,852` (`+3,313`). All other cases were unchanged. Total score rose from `1,593,920` to `1,597,233`.
