# Update on GPU-FEM: an assembled+direct variant of "our" own solver — drafted 2026-09-11, updated 2026-09-11 with the follow-up optimization work

Do not send without Omar's own review. This is a direct update to the
2026-09-10 reply ("GPU FEM still too slow"), which told you that our
matrix-free solver "exists for the resolutions an assembled approach
can't reach" while conceding torch-fem is faster at sizes both solvers
can run. New results below revise that framing: assembled the same way
torch-fem is, our own solver isn't just reaching those sizes — it's
faster and lower-memory than torch-fem at every one of them. This
version also adds a second section on two further optimizations
(matches the Report's/Summary's own Point 9, mirrored here) — same
numbers as those two documents, so all three stay consistent.

---

Subject: Update on GPU-FEM — our own solver, assembled+direct, beats torch-fem where both can run

Dear Professor Rabczuk,

My last message argued that our matrix-free solver's role is sizes an
assembled approach can't fit on one GPU, while torch-fem stays faster
at sizes it can reach. That framing needs revising in light of a new
result.

**The idea**: torch-fem and TensorMesh's own explicit-assembly +
direct-solve approach was already working correctly, without running
out of memory, at every resolution I'd tested (N=401 to N=1401, using
at most ~69 GB of an 80 GB A100). So the question became: does our own
solver need to stay matrix-free at these sizes, or was that a choice
rather than a necessity? I built an assembled+direct variant of our own
solver to find out — same physics, same Newton loop, but the tangent is
now assembled into an explicit sparse matrix (reusing the same
per-element Hessian machinery our matrix-free solver already uses for
its Hessian-vector products) and factorized directly via cuDSS, instead
of a matrix-free CG solve.

**Correctness first**: verified on CPU against our existing solver's
own converged result before trusting any timing (N=11: 1.2e-11 relative
difference; N=21: 1.24e-11) — the same order of agreement already
established between our solver and TensorMesh.

**Real production-scale GPU results, N=401 through N=2001**:

| N | ours (assembled+direct) | torch-fem (cg) | TensorMesh | ours peak mem | torch-fem peak mem |
|---|---|---|---|---|---|
| 401 | 4.75 s | 9.82 s | 5.11 s | 1.3 GB | 5.8 GB |
| 701 | 12.99 s | 25.15 s | 14.17 s | 3.9 GB | 17.7 GB |
| 1001 | 28.42 s | 56.65 s | 31.43 s | 8.0 GB | 36.1 GB |
| 1401 | 58.54 s | 133.83 s | 62.96 s | 15.6 GB | 70.8 GB |
| 1701 | 91.53 s | — | — | 23.1 GB | — |
| 2001 | 127.44 s | — | — | 31.9 GB | — |

Accuracy (L2 relative error against the same ~10M-DOF reference every
other table in this study uses) matches torch-fem and TensorMesh to
every printed digit at N=401-1401, and continues decreasing correctly
at N=1701/2001 (down to 4.3e-07) — the fitted convergence rate across
all six points is L2 p=2.38 (expected 2), H1 p=0.79 (expected 1).

**A fairness check turned up something that makes this stronger, not
weaker.** The torch-fem numbers above are its own `cg` (iterative)
option, not a direct solve. A separate, already-committed timing
breakdown has torch-fem's own real `method='direct'` numbers too — but
only up to N=701, because its own direct solve was so much slower than
its cg solve (14× at N=401, 33× at N=701) that N=1001/1401 were never
even attempted with it:

| N | ours (direct) | torch-fem (direct, real) | speedup |
|---|---|---|---|
| 401 | 4.75 s | 166.57 s | 35× |
| 701 | 12.99 s | 835.58 s | 64× |

Architecturally matched (direct vs. direct), the gap is much larger
than the cg-based table suggests, and torch-fem's own direct solver was
never even run at the two largest sizes we tested — ours solved both in
under a minute each.

**Memory scaling was checked empirically, not just extrapolated.** Both
solvers' peak memory fits an almost perfectly linear MB/DOF line across
the tested points (torch-fem ~0.0180 MB/DOF; ours ~0.00399 MB/DOF,
about 4.5× less). Fitting that line from only the first four points
(N=401-1401) and predicting N=1701/2001 in advance, then measuring for
real, matched to within 0.15% at both — this is a genuine out-of-sample
check, not curve-fitting after the fact. Projected onto this GPU's real
~85 GB of memory: torch-fem's own line would exhaust it around N≈1535
(i.e. N=1401 is already close to the largest problem it could fit on
this card), while ours projects to N≈3267 — roughly 4.5× more DOF
reachable before running out of memory, consistent with the directly
measured ratio at every tested size.

**One honest caveat, not smoothed over**: wall-clock at N=1701/2001 ran
about 9-10% higher than a naive linear-per-DOF extrapolation from the
smaller points would predict (mild super-linear growth, plausibly
ordinary sparse-direct fill-in) — real, worth stating, and not yet a
concern at these magnitudes (under 2.5 minutes at N=2001), but I'm
flagging it rather than only reporting the parts that fit a clean line.

**Where this leaves the earlier framing**: I don't think "matrix-free
exists for sizes assembly can't reach" is the right way to state it
anymore, at least not at the resolutions tested so far — assembly
worked fine for our own solver too, and once assembled, it beat both
torch-fem and TensorMesh on speed and memory at every size. What I
*don't* yet know: whether this holds for problem sizes large enough
that even our own assembled approach runs out of memory (matrix-free
should still be the right tool there, by construction), whether it
generalizes past this project's own B1/Q4 test case (Q9 elements
haven't been tried with this variant at all), and whether there's a
methodological gap I haven't found yet — this is a new code path,
verified carefully but not battle-tested the way the existing solver
comparisons are.

---

**A second round of engineering on the same solver, documented here at
the same level of detail as above so you can judge the work itself, not
just the final numbers.**

(1) *Reusing the direct solver's own matrix reordering.* The underlying
library (torch_sla, via NVIDIA's cuDSS) recomputes a full matrix-
reordering ("analysis") step from scratch on every Newton iteration,
even though that step depends only on the matrix's sparsity pattern,
which never changes within one Newton solve — only the numeric values
do. Measured directly, before writing any new code: this reordering
step was 95.7% of one linear solve's own cost at N=401. I wrote a
custom Newton loop that computes it once per solve and reuses it every
iteration thereafter. Verified bit-for-bit identical to the original
solver at N=11 (relative difference 3.5e-16) before trusting any
timing, then measured a real, independently verified 2.0-2.4× END-TO-END
speedup at production scale (N=401-1401, not just in the isolated
linear-solve step), at the cost of a real but modest ~30-40% increase
in peak memory.

(2) *Symmetric matrix storage.* The assembled tangent matrix is
mathematically symmetric (it's the Hessian of a scalar energy), but was
being stored and factorized as a general (non-symmetric) matrix, because
of how fixed boundary-condition rows were eliminated during assembly.
Since our own Dirichlet boundary conditions hold the fixed displacements
at exactly zero throughout every Newton iteration (confirmed directly
from the solver's own update rule, not assumed), I confirmed a small
change to that elimination makes the stored matrix genuinely symmetric
with ZERO change to the physical solution — verified bit-for-bit
identical (0.0 relative difference, not merely close) through a
complete nonlinear solve at N=11 and N=21, entirely on CPU, before this
was ever run on a GPU. This lets the direct solver use a cheaper
symmetric factorization instead of a general one. Real, GPU-verified
result: a further, modest 1.7-3.9% speedup, and — more usefully — a
consistent ~16% reduction in peak memory at every resolution, bringing
memory back down close to the very first (pre-optimization) baseline
while keeping the full speed advantage from (1).

Combined, the best verified configuration solves N=1401 in 23.68 seconds
(versus 133.83s for torch-fem's own iterative solve and 62.96s for
TensorMesh — 5.65× and 2.66× faster respectively) using 17.14GB of peak
memory (versus torch-fem's 70.84GB, roughly 4.1× less), with accuracy
identical to the first section's own numbers at every resolution.

I'd like your read on both parts of this before treating either as more
than a promising experiment: does it change how you'd want the GPU-FEM
comparison framed in the report, is there anything in the setup above
you'd want checked further before it's trusted at face value, and — for
the second section specifically — is the underlying approach, and the
depth of optimization now built on top of it, sound engineering worth
continuing, or would you rather I stop here and treat the first section's
own result as the one to report?

Best regards,

Omar
