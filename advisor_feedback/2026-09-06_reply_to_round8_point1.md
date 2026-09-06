# Reply to Timon's round-8, point 1 — drafted 2026-09-06

Omar's decision: answer point 1's benchmark-scale question using his own
suggested method ("checking some QoIs and requesting stricter error
tolerances versus your high fidelity ground truth"), rather than scaling
up to a new problem (e.g. the tire example) — the latter would open a
large new work cycle that is out of scope right now.

No new experiment was needed. Table 6a (already in report v56, from
`high_dof_convergence_study.py`, B1 x Neo-Hookean vs. a ~10M-DOF
reference) already carries exactly this: relative error in H1 semi-norm
and the tangent energy norm, together with wall-clock cost, at 6
resolutions from N=51 (5,202 DOF) to N=1401 (3,925,602 DOF). Reading it
as a tolerance-vs-cost table:

| target relative error | resolution needed | wall-clock |
|---|---|---|
| <=1% (H1 and energy) | N=201 | ~15 min |
| <=0.5% | N=401 | ~30 min |
| <=0.2% (strict, verification-grade) | N=701-1401 | ~100-200 min |

None of these are "milliseconds" at any accuracy level a paper would
plausibly claim as verified.

**Honesty caveat, included in the reply rather than omitted**: these
wall-clock numbers are measured with the same solver/preconditioner
Timon separately flagged as suboptimal (points 1 and 5, the CG-hidden
element cost). That is a reason these numbers are a conservative LOWER
BOUND, not an inflated one — improving the preconditioner (work-queue
item #4) will only make the numbers faster, and even a large constant-
factor speedup will not bring multi-million-DOF CG solves into the
millisecond range. So the argument survives, and arguably strengthens
once #4 is done and the same table can be re-measured.

**Scope limit, stated plainly**: Table 6a is B1 x Neo-Hookean only, the
one case this high-DOF sweep was run for. It is not yet checked whether
the other five geometry x material combinations show the same pattern.
Offered as a follow-up if Timon wants broader evidence before this
argument goes in the paper, not hidden as if already covered.

---

Subject: Round-8, point 1 — benchmark scale

Dear Professor Rabczuk,

On whether B1/B2 are demanding enough to justify a neural operator: I
used your suggested check rather than scaling up to a new problem for
now (that would be a substantial separate effort I'd rather scope
deliberately later, possibly around the tire example you mentioned).

Table 6a in the current report already gives me what I need: B1 x
Neo-Hookean's relative error against a ~10M-DOF reference solution, at
six resolutions, together with the wall-clock cost of each solve.
Reading it as a required-accuracy-vs-cost table:

- Reaching <=1% relative error (H1 semi-norm and tangent energy norm)
  needs N=201 (80,802 DOF): about 15 minutes.
- Reaching <=0.5% needs N=401 (321,602 DOF): about 30 minutes.
- A strict <=0.2% verification-grade tolerance needs N=701-1401
  (roughly 1-4 million DOF): 100-200 minutes.

So at any accuracy level I'd be comfortable calling "verified" for the
paper, this problem costs minutes to hours of FEM time, not
milliseconds.

One caveat I want to flag rather than gloss over: these times come from
the same solver/preconditioner you noted is still suboptimal (points 1
and 5). I read that as making this a conservative lower bound, not an
inflated one — a better preconditioner will only make these numbers
faster, and I don't expect any realistic speedup to bring multi-million-
DOF CG solves down to milliseconds, so I think the conclusion holds
either way. I plan to re-measure this same table once the preconditioner
work is done, so we have the honest post-fix numbers too.

The one limitation I'd flag: this is currently only measured for B1 x
Neo-Hookean, since that is the one case the high-DOF sweep was run for.
Happy to extend it to the other five cases if you'd like broader
evidence before this goes into the paper, or if you'd still prefer the
larger industrial-scale problem instead.

Best regards,

Omar
