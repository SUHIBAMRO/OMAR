# Reply to Timon's round-8, point 1 — drafted 2026-09-06, revised same day

Omar's decision: answer point 1's benchmark-scale question using his own
suggested method ("checking some QoIs and requesting stricter error
tolerances versus your high fidelity ground truth"), rather than scaling
up to a new problem (e.g. the tire example) — the latter would open a
large new work cycle that is out of scope right now.

**This draft was revised once already**, after a closer read of the
report text surrounding Table 6a turned up two things the first pass
overstated:
- The N=1001/1401 rows have the CG solver hitting its iteration cap
  without reaching cg_tol on every Newton iteration, so their numbers
  carry extra, unquantified error and aren't clean data points.
- The report's own text says H1/energy still don't reach the advisor's
  own previously-stated 1e-4 target even at N=1401 — a stronger point in
  the same direction, but not the same claim as "0.2% costs 3 hours,"
  which the first draft had implied.

The revised table below only uses the clean N=51-401 range, and states
the 1e-4/N=1401 story separately with its caveats rather than folding it
into a clean number.

**Omar's verdict after reviewing this revision (2026-09-06): the core
argument is sound, but NOT ready to treat as the paper's final answer
without two more things**:
1. Re-measure the same table once the preconditioner (work-queue item
   #4) is improved — item #4 is now a hard prerequisite for closing this
   point, not just "nice to do before #13."
2. Add an engineering QoI like peak stress, since Table 6a's L2/H1/
   energy are all norms of the error FIELD, not the literal example
   ("maximum stresses or similar") Timon's email named.

**(2) is now done, including the real sweep** (`high_dof_convergence_
study.py`'s `compute_peak_stress_error()`, peak Frobenius-norm PK1
stress plus a stress-field L2 norm; verified on tiny CPU cases for both
B1 and B2 first, catching and fixing a real bug along the way — B2's
analytic material field expects polar (theta, r) coordinates, not
Cartesian). **The real GPU sweep finished 2026-09-07**
(`highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean.json`),
and it does NOT tell the same clean story L2/H1/energy do — this needs
to go to Timon honestly, not smoothed over:

| N | H1_semi_rel | peak_stress_rel | CG failures |
|---|---|---|---|
| 51 | 1.76% | 1.97% | 0 |
| 101 | 1.10% | 3.17% | 0 |
| 201 | 0.63% | 0.22% | 0 |
| 401 | 0.36% | 5.19% | 20 |
| 701 | 0.29% | 7.58% | 30 |
| 1001 | 0.18% | 1.95% | 40 |
| 1401 | 0.16% | 3.72% | 80 |

H1 (a global field norm) decreases smoothly and monotonically with
refinement, as always. **Peak stress does not** — it is non-monotonic at
every step, and the overall least-squares convergence rate across all 7
points is actually *negative* (-0.29, meaning the fitted trend is
slightly worsening, not improving, with refinement). This is not a bug:
a pointwise maximum is a local, non-averaged functional, and it is
well known in FEM that such quantities converge far less smoothly than
global norms, especially without a stress-recovery/smoothing step,
which nothing in this pipeline currently does. It is also not fully
separable from the CG issue Table 6a already has: N=401 onward all have
CG failures (hitting the 2000-iteration cap without reaching cg_tol),
and those four rows are also the four largest and most erratic
peak-stress errors (5.2%, 7.6%, 2.0%, 3.7%) — some of this noise is
very plausibly leftover algebraic error from unconverged CG, on top of
the genuine discretization-driven roughness. **These N=1001/1401 rows
still used the OLD checkpoints (pre-item-#4 preconditioner) — the
block-Jacobi improvement has not yet been applied to this table**, so
disentangling "genuinely noisy QoI" from "noisy because CG didn't
converge" needs the item #4 re-run before it can be said cleanly.

**Reading this honestly, it now argues in Timon's favor more than mine**:
a naive tolerance-vs-cost table works for L2/H1/energy, but the one QoI
he actually named by example (max stress) does NOT reduce to "run it
longer, tolerance improves" — which is closer to the kind of problem
goal-oriented error estimation (which he separately proposed, R7) exists
to solve, rather than something a plain convergence study can paper over.
I'd rather send this finding as-is than round it into looking cleaner
than it is.

**So: do not send this draft yet.** It should wait until item #4's
actual re-run (block2x2 preconditioner) on N=401-1401 is done, so we
know whether the CG failures were driving the peak-stress noise — that
re-run is the direct next step, not a nice-to-have.

---

Subject: Round-8, point 1 — benchmark scale

Dear Professor Rabczuk,

On whether B1/B2 are demanding enough to justify a neural operator: I
used your suggested check rather than scaling up to a new problem for
now (that would be a substantial separate effort I'd rather scope
deliberately later, possibly around the tire example you mentioned).

Table 6a in the current report (B1 x Neo-Hookean's error against a
~10M-DOF reference, six resolutions with wall-clock cost) gives a direct
tolerance-vs-cost answer for the accuracy range that is cleanly measured
(N=51 to N=701, no solver issues):

- Reaching <=1% relative error (H1 semi-norm and tangent energy norm)
  needs N=201 (80,802 DOF): about 15 minutes.
- Reaching <=0.5% needs N=401 (321,602 DOF): about 30 minutes.

So even at a modest accuracy target, this problem costs tens of minutes
of FEM time, not milliseconds.

I want to be upfront about where the evidence gets weaker, rather than
stretch it further than it holds. The table also has an N=1401 point
(~198 minutes) at 0.16% H1 error, but I'd flag it rather than lean on
it: the CG solver hit its iteration cap without reaching cg_tol on every
Newton iteration at that mesh, so that number carries some unquantified
error beyond discretization and isn't a clean data point. More
importantly, this is the same section where you had earlier asked for a
1e-4 relative-error target, and it's honest to say directly: even at
N=1401, with that reference and that solver, we do not reach it in H1 or
the energy norm (they land at 1.6x10^-3 and 7.8x10^-4 respectively) —
closing that gap would need a substantially finer reference than the
~10M-DOF one we have, beyond what's computationally tractable here right
now. I read that as arguably a stronger point in the same direction (the
problem doesn't get "easy" even at large compute), but I didn't want to
fold it into a clean "0.2% costs 3 hours" number when the data behind it
isn't clean.

One more caveat: these times come from the same solver/preconditioner
you flagged as suboptimal (points 1 and 5). I'd expect a better
preconditioner to only make these numbers faster, and I don't think any
realistic speedup brings multi-million-DOF CG solves into the
millisecond range, so I think the core conclusion (1%/0.5% tolerances
cost tens of minutes) survives regardless. I plan to re-measure this
table once the preconditioner work is done.

I also ran the peak-stress QoI you named as an example ("maximum
stresses or similar") at the same resolutions, and it's worth reporting
honestly rather than folding into the same story: it does NOT converge
as cleanly as H1/energy. Where H1 error drops smoothly from 1.8% at
N=51 to 0.16% at N=1401, the peak-stress relative error is non-monotonic
at every step (1.97% -> 3.17% -> 0.22% -> 5.19% -> 7.58% -> 1.95% ->
3.72%), and its fitted trend across all seven points is actually
slightly negative rather than improving. Part of this is likely
unavoidable -- a pointwise maximum is a local, non-smoothed functional
and is known to converge far less regularly than a global norm -- but
part of it may be the same solver issue: N=401 onward are exactly where
CG starts missing its convergence tolerance, and those rows also carry
the largest, most erratic peak-stress errors. I want to re-run this
once the preconditioner fix (point 1/5) is in before drawing a firm
conclusion about how much of the noise is genuine versus solver
artifact. Either way, I think this is a real finding worth naming to
you directly: the QoI you asked about specifically does not reduce to
"run it longer and the tolerance improves" the way the field norms do,
which looks like exactly the kind of case your goal-oriented
error-estimation suggestion (round 7) was aimed at, rather than
something a convergence table alone resolves.

Last limitation: this is currently B1 x Neo-Hookean only, the one case
this high-DOF sweep was run for -- not yet checked on the other five
combinations. Happy to extend it if you'd like broader evidence before
this goes into the paper, or if you'd still prefer the larger
industrial-scale problem instead.

Best regards,

Omar
