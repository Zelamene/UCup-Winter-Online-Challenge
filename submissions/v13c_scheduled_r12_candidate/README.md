# v13c — scheduled bSB for R=1/2

Accepted by the judge at `1,700,646`, a `+93,434` gain over accepted v12
(`1,607,212`). This is the current base for all further candidates.

## Contents

- `submission.cpp` — single-file C++17 decoder exported from
  `artifacts/phase9_case_manifest_v13c.json`.
- `manifest.json` — the v3 manifest used for the export.
- `validation.json` — source-size, dispatch, trajectory-count, and public-output gates.
- `judge_result.json` — accepted score and delta.

## Architecture

The four `R=1/2` cases use one scheduled bSB trajectory each, with five-knot
piecewise-linear `xi/lambd/delta/dt` schedules from the Stage B searches. The
eight `R=2/3` and `R=3/4` cases are copied unchanged from accepted v12 dSB
portfolios. Dispatch uses an exact `(n, k)` match and the nearest configured SNR.

bSB is compliant ballistic SB: the check gradient uses the continuous soft spin
`x_j` in Eq. (1) with `sigma -> x`. No per-edge messages persist between QAIA
steps, so this is not BP, min-sum, or bit-flip.

Do not modify this folder when preparing a new candidate. Copy its manifest into
a new artifact and build a separate submission folder instead.
