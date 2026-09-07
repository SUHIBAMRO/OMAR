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

**(2) is now done at the code level, TWICE** — the first version had a
real measurement bug, caught by Omar asking the right question ("هل في
طريقه صحيحه لعمله ولا المشكله باشي تاني؟") after the first real sweep's
numbers looked wrong, rather than accepting a "this QoI is just noisy"
explanation at face value.

**Run 1 (2026-09-06/07, `high_dof_convergence_study.py`'s original
`compute_peak_stress_error()`)**: peak stress predicted vs. reference
were both computed as `max(|P|)` over the COARSE mesh's OWN Gauss
points. This is a real bug, not an inherent QoI property: that point set
gets denser as the coarse resolution N under test increases, so the
"reference" peak value silently drifted UPWARD across rows purely from
sampling more points closer to wherever the true maximum actually is —
confirmed directly in the data, `peak_stress_ref` itself climbed from
13.9 at N=51 to 39.3 at N=1401, a ~2.8x range, despite "reference" being
supposed to mean one fixed target. The resulting "peak stress error"
column (1.97% / 3.17% / 0.22% / 5.19% / 7.58% / 1.95% / 3.72%, overall
fitted rate *negative*) was measuring how the sampling density changed
between rows, not how accurate each coarse solution's stress prediction
was. This was NOT simply "pointwise QoIs are noisier than norms" (true
in general, but not the dominant effect here) — it was comparing against
a moving target, which will produce exactly this kind of directionless,
even backwards-trending noise regardless of solver quality.

**Fixed 2026-09-07**: `find_fine_peak_stress()` now locates the fine
reference's own peak-stress point ONCE per order (from the fine mesh's
own dense Gauss points), and every coarse resolution's stress is
evaluated at that SAME fixed physical point (via the fine mesh's own
exact point-location FE evaluation, already validated, applied to the
coarse mesh too). Verified on CPU: `peak_stress_ref` is now byte-for-byte
identical across different coarse N in the same sweep (e.g. 12.4648592
at both N=6 and N=11, B1; confirmed also on B2's polar-coordinate path),
and the error at those two points now moves the expected direction
(30.8% -> 20.2% as N increases) instead of drifting incoherently. This
now makes peak-stress error a genuine, fixed-target pointwise QoI —
still expected to converge less smoothly than a global norm like L2/H1
in general (that part of the original reasoning was correct even though
the specific numbers weren't trustworthy), but no longer inflated by a
moving reference on top of that.

**The Run-1 numbers above are RETRACTED — do not use them anywhere.**
The real, corrected sweep has not been re-run yet (needs the same GPU
Colab pipeline as before); this section of the reply will be rewritten
once it has, with the item #4 (block-Jacobi preconditioner) re-run
folded in at the same time so N=401-1401's CG-failure question is
addressed in the same pass rather than needing a third run.

**So: do not send this draft yet.** Two GPU runs are still needed before
this point closes: (1) the corrected peak-stress sweep with the fixed
`compute_peak_stress_error`, and (2) item #4's `--precond_kind block2x2`
re-run at N=401-1401 to see whether CG failures there go away. Both
should ideally happen in one combined run rather than two.

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

I also implemented the peak-stress QoI you named as an example ("maximum
stresses or similar"). The first real run's numbers turned out to be
measuring an artifact -- the "reference" peak value I was comparing
against was silently redefined at every mesh resolution instead of being
one fixed target, which produced meaningless, directionless noise that
I nearly reported as a genuine finding about the QoI. Caught and fixed
before sending anything based on it. I'll include the corrected
peak-stress-vs-N numbers together with the preconditioner re-run once
both are done -- I'd rather send this a few days later with numbers I
trust than send something now and have to retract it.

Last limitation: this is currently B1 x Neo-Hookean only, the one case
this high-DOF sweep was run for -- not yet checked on the other five
combinations. Happy to extend it if you'd like broader evidence before
this goes into the paper, or if you'd still prefer the larger
industrial-scale problem instead.

Best regards,

Omar
