# Draft reply to Timon's round-12 email

Drafted 2026-09-18. NOT YET SENT -- Omar's call, per the project's own
standing convention (never send a reply without his review first).

---

Dear Professor Rabczuk,

All three points are done, with real GPU results throughout.

1. Cauchy stress, fixed region: implemented (region-weighted average,
99th percentile, and true max, region fixed in physical space across
resolutions), verified on CPU before spending any GPU time, then run
for all six cases against the final checkpoints, FEM and the operator
both scored against the same fine reference on the same input field.
One clear finding: the true max converges far more slowly than the
region average -- e.g. FEM's own max is still 26.6-64.5% across
N=3-49 while its displacement error is already under 0.1% there --
exactly the effect your own caution about the pointwise maximum
predicted. A second finding worth flagging: the operator's own accuracy
is not always monotonic in resolution (two cases get worse again past
N=29), unlike FEM, which converges smoothly everywhere.

2. Training-cost table added (all seven runs: resolutions, samples,
epochs, wall-clock, cost/sample). Peak memory during training was never
instrumented for these runs, reported honestly as not measured rather
than guessed. Per your own note, no controlled re-run was attempted.

3. Break-even now uses the compile+TF32 number throughout; the two
tables differ only in which FEM baseline is used (accuracy-matched
N=11 vs. resolution-matched N=1401), not in how the operator itself is
run.

Attached: the full Report and the shorter Summary, both updated with
all of the above.

Best regards,
Omar
