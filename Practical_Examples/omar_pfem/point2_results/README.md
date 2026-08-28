# Point 2 results — accuracy/cost Pareto, operator vs FEM

One case so far: **B1 × Neo-Hookean**, from the `pareto_analysis.py` run that
finished 2026-08-28 after 1 h 54 m. The other five cases have not been run.

## What was measured

Both sides on the same 20 problem instances, against the same N=101 fine-mesh
reference, on the same device, at batch size 1.

| N | nodes | FEM err | FEM cost | operator err | operator cost | speed-up | accuracy gap |
|---|---|---|---|---|---|---|---|
| 13 | 169 | 0.608% | 3.1 s | 6.76% | 1.620 ms | 1,914× | 11.1× |
| 17 | 289 | 0.383% | 5.5 s | 5.93% | 1.612 ms | 3,406× | 15.5× |
| 21 | 441 | 0.270% | 8.6 s | 5.12% | 1.610 ms | 5,327× | 18.9× |
| 25 | 625 | 0.203% | 12.4 s | 4.43% | 1.610 ms | 7,676× | 21.8× |
| 29 | 841 | 0.156% | 16.8 s | 3.93% | 1.663 ms | 10,097× | 25.2× |
| 33 | 1089 | 0.124% | 21.9 s | 3.70% | 1.625 ms | 13,491× | 29.8× |
| 37 | 1369 | 0.101% | 27.7 s | 3.69% | 1.658 ms | 16,705× | 36.5× |
| 41 | 1681 | 0.083% | 34.1 s | 3.84% | 1.640 ms | 20,812× | 46.1× |
| 49 | 2401 | 0.059% | 49.1 s | 4.44% | 4.613 ms | 10,654× | 75.2× |

## What it says

**The two never compete on accuracy.** The FEM at its coarsest setting — N=13,
169 nodes, 3.1 s — is 0.608%, already 6.1× more accurate than the operator at
its best (3.69% at N=37). There is no mesh at which the operator matches even
the cheapest FEM solve measured.

**The Pareto front is two branches with an empty middle.** Below about three
seconds per problem the operator is the only thing on the plot, at 1.6 ms and
3.7–6.8%. Above three seconds it is FEM, at 0.06–0.6%. Nothing occupies the
range between, so this is not a curve to pick a point on — it is a choice
between two regimes.

**Cost is where the operator's argument actually lives.** Its inference cost is
flat in mesh size (1.610–1.663 ms from 169 to 1,681 nodes) while the FEM solve
grows superlinearly (3.1 s to 34.1 s for a 10× node count). The speed-up
therefore climbs from 1,914× to 20,812× with resolution. That trend is a
stronger statement than any single speed-up number.

**Operator accuracy is not monotone in N**: best at N=33–41 (3.69–3.84%), worse
at both ends. This is qualitatively the zero-shot finding — a minimum near the
training resolutions of 21 and 33 — though the minimum sits at a different N
than Table 12 puts it.

## Two things that need stating, not smoothing over

**1. These numbers cannot be laid alongside Table 12, for two independent
reasons.**

  * *Different metric.* This script uses the combined relative L2,
    `‖e‖/‖u‖` over both displacement components at once — the convergence-study
    convention of Section 4.4, and the right one here because the FEM side is a
    convergence curve and both sides must be scored identically. Tables 5, 11
    and 12 use the per-component average `0.5*(rms(e_u)/rms(u) +
    rms(e_v)/rms(v))`. On B1 the loaded component v dominates u, so the
    combined norm is the smaller of the two.
  * *Different problems.* `pareto_analysis.py` draws seeds `900_000 + i`; the
    zero-shot eval draws `20_000_000 + i`. These are not the same physical
    samples scored two ways.

  Both effects push the same direction, so no conversion factor between the
  two tables can be quoted from this run — it would conflate them.

**2. The N=49 operator timing is out of line and is not explained.** Every
resolution from 13 to 41 lands in 1.610–1.663 ms; N=49 reports 4.613 ms, a
2.8× jump for a 1.43× increase in nodes. It is a median of 20 repeats after 5
warm-ups, so it is not a single stray sample, but it is also the last
measurement in a two-hour run. Re-timing the operator alone at N=49 costs
seconds and would settle it. The number is recorded here as measured.

## Also worth remembering

The FEM side is the **CPU reference solver**, which is what generating a new
solution costs today. The GPU-native solver of Section 8.5 is 71.7–171.5×
faster than it, so a GPU-to-GPU version of this plot would close roughly two
orders of magnitude of the speed-up column while leaving the accuracy columns
untouched.
