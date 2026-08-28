# Point 2 results — accuracy/cost Pareto, operator vs FEM

One case so far: **B1 × Neo-Hookean**, run twice. The other five cases have not
been run.

* **run3**, 2026-08-28, 1 h 54 m
* **run4**, 2026-08-28, 6 h 24 m — the same configuration on a slower Colab
  runtime, not a changed one

## What was measured

Both sides on the same 20 problem instances, against the same N=101 fine-mesh
reference, on the same device, at batch size 1.

| N | nodes | FEM err | operator err | accuracy gap | FEM cost run3 / run4 | operator cost run3 / run4 | speed-up run3 / run4 |
|---|---|---|---|---|---|---|---|
| 13 | 169 | 0.608% | 6.76% | 11.1× | 3.1 s / 8.9 s | 1.620 / 5.491 ms | 1,914× / 1,630× |
| 17 | 289 | 0.383% | 5.93% | 15.5× | 5.5 s / 16.0 s | 1.612 / 5.517 ms | 3,406× / 2,898× |
| 21 | 441 | 0.270% | 5.12% | 18.9× | 8.6 s / 24.9 s | 1.610 / 4.584 ms | 5,327× / 5,432× |
| 25 | 625 | 0.203% | 4.43% | 21.8× | 12.4 s / 36.1 s | 1.610 / 5.542 ms | 7,676× / 6,513× |
| 29 | 841 | 0.156% | 3.93% | 25.2× | 16.8 s / 49.1 s | 1.663 / 5.508 ms | 10,097× / 8,906× |
| 33 | 1089 | 0.124% | 3.70% | 29.8× | 21.9 s / 63.6 s | 1.625 / 5.530 ms | 13,491× / 11,502× |
| 37 | 1369 | 0.101% | 3.69% | 36.5× | 27.7 s / 81.0 s | 1.658 / 5.563 ms | 16,705× / 14,562× |
| 41 | 1681 | 0.083% | 3.84% | 46.1× | 34.1 s / 99.3 s | 1.640 / 5.550 ms | 20,812× / 17,895× |
| 49 | 2401 | 0.059% | 4.44% | 75.2× | 49.1 s / 142.6 s | 4.613 / 5.555 ms | 10,654× / 25,676× |

## What the two runs together establish

**The errors are identical — every printed digit, both sides, all nine
resolutions.** Two independent runs hours apart on different hardware. The
accuracy half of this study is fully reproducible and can be quoted without
qualification.

**The wall-clock half is not, and the failure is systematic rather than
random.** run4's FEM is 2.887–2.925× slower than run3's at every single
resolution — a spread of 1.3%, which is a different machine, not noise. The
operator moved with it: 1.610–1.663 ms in run3 against 4.584–5.563 ms in run4.
Absolute per-sample timings from either run describe the Colab instance that
produced them, not the method.

**The ratio survives what the absolute numbers do not.** Excluding N=49 the
two runs' speed-ups agree within 17% at every resolution, because both sides
slowed by roughly the same factor. The speed-up is the quotable quantity; the
milliseconds are not.

**run4's timings, not run3's, are the ones consistent with the rest of the
report.** Table 10a measures B1 × Neo-Hookean at batch size 1 as 4.582 ms
(median of 50 repeats). run4 at N=21 — the same 441-node mesh — gives 4.584 ms.
run3 gives 1.610 ms, a third of it. Whatever run3's machine was, it is not the
one the report's inference-latency table was measured on, so the run3
speed-up column would overstate the operator against the report's own numbers.

## The N=49 anomaly: did not reproduce

run3's operator time at N=49 was 4.613 ms against 1.610–1.663 ms everywhere
else, and was flagged as unexplained. In run4 the operator at N=49 is
5.555 ms, squarely inside that run's own 4.584–5.563 ms band. **The outlier is
a property of that one run, not of N=49.** No cause was isolated and none is
needed — the measurement simply does not repeat.

One residual: run4's own N=21 point (4.584 ms) is 17% faster than its other
eight resolutions with no pattern to it, so roughly 20% jitter should be
assumed on any single batch-size-1 latency here.

## The findings themselves

**The two methods never compete on accuracy.** FEM at its coarsest setting —
N=13, 169 nodes — is 0.608%, already 6.1× more accurate than the operator at
its best (3.69% at N=37). There is no mesh at which the operator matches even
the cheapest FEM solve measured.

**The Pareto front is two branches with an empty middle.** Below a few seconds
per problem the operator is the only thing on the plot; above it, FEM, at
0.06–0.6%. Nothing occupies the range between, so this is not a curve to pick
a point on — it is a choice between two regimes.

**Cost scaling is where the operator's argument lives.** Its inference cost is
flat in mesh size in both runs while the FEM solve grows superlinearly, so the
speed-up climbs by an order of magnitude across the sweep (1,630× → 17,895× on
run4's numbers, excluding the N=49 point run3 disagrees about). That trend is a
stronger statement than any single speed-up figure.

**Operator accuracy is not monotone in N**: best at N=33–41 (3.69–3.84%), worse
at both ends. Qualitatively the zero-shot finding — a minimum near the training
resolutions of 21 and 33 — though the minimum sits at a different N than
Table 12 puts it.

## Still not comparable to Table 12

Neither run changes this, because both used the same configuration. Two
independent reasons:

* *Different metric.* This script uses the combined relative L2, `‖e‖/‖u‖` over
  both displacement components at once — the convergence-study convention of
  Section 4.4, and the right one here because the FEM side is a convergence
  curve and both sides must be scored identically. Tables 5, 11 and 12 use the
  per-component average `0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v))`. On B1 the
  loaded component v dominates u, so the combined norm is the smaller measure.
* *Different problems.* `pareto_analysis.py` draws seeds `900_000 + i`; the
  zero-shot eval draws `20_000_000 + i`.

Both push the same direction, so no conversion factor between the two tables
can be quoted from these runs — it would conflate them.

## Also worth remembering

The FEM side is the **CPU reference solver**, which is what generating a new
solution costs today. The GPU-native solver of Section 8.5 is 71.7–171.5×
faster than it, so a GPU-to-GPU version of this plot would close roughly two
orders of magnitude of the speed-up column while leaving the accuracy columns
untouched.
