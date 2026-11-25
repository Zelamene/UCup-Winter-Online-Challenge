# v4 — accepted hybrid

Judge score: `1,593,920` (`+78,048` vs v1)

Final architecture:

- `R=1/2`: exact v1 probe decoder
- `R=2/3`, `R=3/4`: v2 per-case tuned dSB portfolios
- dispatch on `(N, K, SNR)`
- at most six trajectories, 50 iterations, QAIA/dSB only

Verification:

- judge score matched the prediction exactly
- `R=1/2` output matched the probe decoder bit-for-bit on a synthetic test case
- public BER `0.025789` vs v1 `0.027281`
- clean C++17 compile, `82,020` bytes
