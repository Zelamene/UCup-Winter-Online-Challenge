# v3 — hybrid with wrong fallback

Judge score: `1,588,472` (`+8,914` vs v2)

`R=1/2` was removed from the per-case dispatch so those cases would use an accepted fallback. However, the local `submission.cpp` was `15.5 KiB`, while the judge-accepted baseline was the `12.1 KiB` probe decoder.

Consequence: `R=1/2` still lost `5,448` points versus v1, while the other eight cases kept their gains.
