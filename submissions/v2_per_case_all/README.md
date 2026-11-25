# v2 — per-case tuning on all 12 cases

Judge score: `1,579,558` (`+63,686` vs v1)

Per-case grid search, local parameter search and greedy six-candidate energy portfolios were applied to every `(N, R, SNR)` case.

Result by rate:

- `R=1/2`: all four cases regressed (`-14,362` total)
- `R=2/3`: all four cases improved
- `R=3/4`: all four cases improved

Lesson: the synthetic `6 dB` channel extrapolation overfitted and reversed the judge ranking for `R=1/2`.
