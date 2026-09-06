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

**(2) is now done at the code level** (same day): `high_dof_convergence_
study.py` has a new `compute_peak_stress_error()` function (peak
Frobenius-norm PK1 stress, plus a stress-field L2 norm), wired into the
main sweep and the JSON report. Verified end-to-end on tiny CPU test
cases for BOTH B1 and B2 — catching and fixing a real bug along the way
(B2's analytic material field expects polar (theta, r) coordinates, not
Cartesian, which the first version of the new code got wrong). **Not yet
run at the real N=51-1401 sweep scale** (needs GPU, like the other
Colab-run studies in this project) — so there are no real peak-stress-
vs-N numbers yet, only a verified-correct implementation.

**So: do not send this draft yet.** It should wait until at least (1) is
further along, or be sent explicitly framed as a progress update with
both open items named, not as a final answer.

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

Last limitation: this is currently B1 x Neo-Hookean only, the one case
this high-DOF sweep was run for -- not yet checked on the other five
combinations. Happy to extend it if you'd like broader evidence before
this goes into the paper, or if you'd still prefer the larger
industrial-scale problem instead.

Best regards,

Omar
