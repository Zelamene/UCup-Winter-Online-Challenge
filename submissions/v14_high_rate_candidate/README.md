# v14 — tuned bSB for every case

Accepted by the judge at `1,847,842`, a `+147,196` gain over accepted v13c
(`1,700,646`). This is the current best accepted version and the base for all
future candidates.

## Contents

- `submission.cpp` — single-file C++17 decoder exported from
  `artifacts/phase10_case_manifest_v14.json`.
- `manifest.json` — the v3 manifest used for the export.
- `validation.json` — source-size, dispatch, trajectory-count, public-output,
  and public-BER gates.
- `judge_result.json` — accepted score, per-case scores, and deltas vs v13c.

## Architecture

All 12 cases now use one tuned ballistic-SB (bSB) trajectory each. The four
`R=1/2` cases are unchanged from v13c. The eight high-rate cases replace v12
six-trajectory dSB portfolios with Stage A configurations, four of which were
improved further by gated Stage B schedules. Dispatch uses exact `(n, k)` and
the nearest configured SNR.

bSB is compliant ballistic SB: the check gradient uses the continuous soft spin
`x_j` in Eq. (1) with `sigma -> x`. No per-edge messages persist between QAIA
steps, so this is not BP, min-sum, or bit-flip.

Do not modify this folder when preparing a new candidate. Copy its manifest into
a new artifact and build a separate submission folder instead.
