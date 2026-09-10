# Reply to Timon's "GPU FEM still too slow" concern — drafted 2026-09-10

Do not send without Omar's own review. This closes round-9 item 5
("My main concern remains that our current GPU FEM implementation is
still too slow to serve as a competitive baseline"), now that every
number it depends on (matched-precision accuracy/convergence/timing,
the timing breakdown, the NO@N1401 measurement) is real and committed.

---

Subject: On "GPU FEM still too slow" — the real tradeoff, with numbers

Dear Professor Rabczuk,

Taking your concern directly rather than around it: yes, at every
resolution where both solvers can actually run, torch-fem is faster
than our matrix-free GPU-native solver — 204–306× faster in
wall-clock, now measured at matched FP64/1e-8 precision (not the
unfair float32/1e-3 comparison from before). That gap is real and I'm
not minimizing it.

But "faster" and "a competitive baseline" are answering two different
questions, and I want to lay out why, with evidence rather than
architecture-talk alone.

**Where torch-fem wins, and why.** torch-fem explicitly assembles a
sparse tangent stiffness matrix once per Newton iteration and solves
it with CG (Jacobi-preconditioned) — a well-optimized, mature code
path. Our own solver never assembles that matrix at all; every
Hessian-vector product is obtained by automatic differentiation
instead. That's slower per iteration, by design — matrix-free exists
specifically to avoid ever holding an assembled tangent in memory, not
because assembly was overlooked.

**The evidence that this isn't just a design story.** torch-fem's own
peak GPU memory at the same four resolutions: 5.7 GB (N=401) → 17.3 GB
(N=701) → 35.3 GB (N=1001) → **69.2 GB (N=1401)** — on an 80 GB A100,
already within about 13% of the card's own limit at the largest size
we tested. Our own solver ran that same N=1401 case (3.9M DOF) on the
same hardware family without hitting a comparable wall, because it
never allocates for a global matrix in the first place. This is the
actual, measured shape of the tradeoff: torch-fem wins on speed at
sizes it can reach; our solver exists for the sizes an assembled
approach can't reach on one GPU at all.

**I also tested whether a different solver choice would close the
gap, rather than assuming CG+Jacobi is already torch-fem's best
foot.** Per your own suggestion about direct solvers, I timed
torch-fem with a direct (LU) factorization instead of CG at N=401 and
N=701. It is not faster — it is **14× slower at N=401 and 33× slower
at N=701**, and the gap widens with N, not closes. So torch-fem's own
CG+Jacobi choice in this comparison isn't an unfairly weak baseline
I picked; it's the actually-faster option for this problem on this
library. (I didn't extend the direct-solver test to N=1001/1401 —
the trend by N=701 already argues against it, and a first attempt at
an untested factorization size felt like the wrong place to spend GPU
hours.)

**A full timing breakdown, as requested**: assembly is 43% of
torch-fem's own total time at N=401, falling to 24% at N=1401 — the
CG solve itself scales worse with problem size than assembly does.
Our own solver has no equivalent "assembly" line item to report
against this — every CG iteration is itself the matrix-vector product,
not a separate build step — which is the same point stated
differently.

**One more piece that bears on "competitive baseline," even though it
wasn't the original question**: the neural operator's own inference
cost is not flat with mesh size either. Measured directly at N=1401
for the first time (previously only checked up to N=49 in the
zero-shot study): 2.29 s/sample, about 500× the 4.6 ms/sample already
published at the study's standard resolution — a real, sublinear
(~n^0.74) but far from flat scaling. It's still 59× faster than
torch-fem's matched-precision solve at that same N, so the operator
remains the fastest option at every size checked so far — but its
accuracy at N=1401 was never validated (the zero-shot study's own
tested range stops at N=49), so that speed advantage doesn't yet come
with an accuracy guarantee at this scale.

**Where I land**: torch-fem is a genuinely faster baseline at the
sizes both solvers can run, and I'd rather say that plainly than
defend our own number past what it's earned. What I don't think
follows is that our solver is simply under-optimized — the memory
numbers above are the direct evidence that it's solving a different
problem (how far can this go on one GPU) than torch-fem is (how fast
can this go at a size that fits). If the report should lead with
torch-fem as the primary GPU-FEM baseline for the sizes it reaches,
and reserve our own matrix-free solver's role for the resolutions
beyond that, I'm glad to restructure it that way — that reflects what
the numbers actually show better than presenting one solver as a
strictly-better replacement for the other.

Best regards,

Omar
