# Update on GPU-FEM: an assembled+direct variant of "our" own solver — drafted 2026-09-11

Do not send without Omar's own review. This is a direct update to the
2026-09-10 reply ("GPU FEM still too slow"), which told you that our
matrix-free solver "exists for the resolutions an assembled approach
can't reach" while conceding torch-fem is faster at sizes both solvers
can run. New results below revise that framing: assembled the same way
torch-fem is, our own solver isn't just reaching those sizes — it's
faster and lower-memory than torch-fem at every one of them.

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

I'd like your read on this before treating it as more than a promising
experiment: does it change how you'd want the GPU-FEM comparison framed
in the report, and is there anything in the setup above you'd want
checked further before it's trusted at face value?

Best regards,

Omar
