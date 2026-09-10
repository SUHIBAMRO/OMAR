# Round-9 full status email — drafted 2026-09-10

Do not send without Omar's own review. Supersedes the narrower
"still too slow" draft (`2026-09-10_reply_gpu_fem_still_too_slow.md`)
by folding its substance into a single, complete response covering
every item from both of Timon's round-9 messages (the torch-fem
comparison feedback and the follow-up email). Attach the updated
Report and Work Summary when sending.

---

Subject: Round-9 follow-up — all seven points, one open decision

Dear Professor Rabczuk,

Closing the loop on both your recent messages together. Every point
below now has a real, measured result behind it, run on the actual
comparison code rather than estimated — except one, which is a
decision for you rather than something I can resolve alone. I've
updated the attached Report and Work Summary to match everything
here.

**1. Matched-precision torch-fem comparison (FP64/1e-8, not
float32/1e-3).** You were right that the earlier comparison wasn't
fair. I found and fixed a real bug in torch-fem's own near-null-space
setup that was silently forcing float32/loose tolerance regardless of
the requested dtype, then re-ran everything at FP64/1e-8. Displacement,
L2, and H1 errors against the same ~10M-DOF reference are now IDENTICAL
to our own solver at every resolution tested, N=51 through N=1401.
Extended to every other QoI too — energy norm, peak stress — at the two
largest resolutions: all four quantities match to every printed digit,
ratio exactly 1.00x. At this matched precision, torch-fem is still
204–306× faster in wall-clock than our matrix-free solver — smaller
than the earlier unfair 418–1197×, but real and fully defensible now.

**2. Does the neural operator's inference cost stay flat at large
DOF?** No, not indefinitely. Measured directly at N=1401 for the first
time (previously only checked up to N=49): inference cost rises to
2.29 s/sample, about 500× the 4.6 ms/sample published at the study's
standard resolution — a real, sublinear (~n^0.74) but far from flat
scaling. It remains 59× faster than torch-fem's own matched-precision
solve at that same N, so the operator is still the fastest option at
every size checked so far, but its accuracy at N=1401 was never
validated — the zero-shot resolution-invariance study's own tested
range stops at N=49. I've added this caveat directly next to the
"essentially flat in mesh size" claim in the Report rather than leaving
it to stand unqualified.

**3. Table 6 (batch-size study) removed**, per your note that it was
confusing — along with its three figures, from both documents. Section
numbering and every cross-reference were fixed throughout so nothing is
left with a gap or a dangling pointer to it.

**4. torch-fem solver description, fixed.** You caught a real
inconsistency: one paragraph described torch-fem's iterative
(CG, Jacobi-preconditioned) setup, and the very next one described it
as a direct factorization — contradicting each other and not matching
what this comparison actually runs. Fixed to be consistent with the
real code path (CG, never a direct factorization, in this study).

**5. TensorMesh — confirmed correct, but a decision is needed on scope.**
I got a Newton + direct-solver (LU) case running, per your explicit
request not to use TensorMesh's own L-BFGS approach. Cross-checking
against our own solver first showed a real disagreement, which I
initially (and wrongly) attributed to a bug in TensorMesh's own
quadrature. Re-investigating rather than accepting that verdict found
the actual cause: TensorMesh's quadrilateral element expects node
ordering in tensor-product order, not the perimeter order this
project's own mesh generator (and meshio/VTK/torch-fem) use — confirmed
directly from TensorMesh's own official mesh-generator output, not
assumed. With that one-line fix, TensorMesh now matches our own solver
to about 10 significant digits at small scale (B1 × Neo-Hookean,
N up to 51).

Here's the open question: I measured, rather than guessed, how far this
setup scales. Real timings at N=3/11/21/31/51 (0.13s/0.21s/2.64s/14.53s/
105.27s) fit an empirical ~dof^2.2 growth. Extrapolating that to your
own torch-fem sweep's smallest production point, N=401 (321,602 DOF),
projects roughly 10 days for a single solve — TensorMesh's default
Jacobian path (a dense `torch.autograd.functional.jacobian`, confirmed
by reading the installed source directly) simply isn't built for that
scale. N=51 is the practical ceiling for what the current code can
produce. I can either (a) report TensorMesh as confirmed-correct at
this small scale and stop there, or (b) write an explicit sparse
Jacobian to reach the same production sizes used for the torch-fem
comparison — real additional engineering, not just a longer run. Let me
know which you'd prefer.

**6. Timing breakdown by phase, as requested.** Measured directly for
torch-fem (N=401–1401): assembly is 43% of total time at N=401, falling
to 24% at N=1401, as the CG solve itself scales worse with problem size
than assembly does. I also tried a direct (LU) solve per your own
suggestion, rather than assuming CG+Jacobi was already the best choice
— real, unexpected result: it is 14× slower at N=401 and 33× slower at
N=701 than CG+Jacobi, and the gap widens with N rather than closing (not
extended past N=701 given that trend). Our own solver has no equivalent
assembly phase to report against this — every CG iteration is itself
the matrix-vector product, via automatic differentiation, not a
separate build step.

**7. Tolerance sensitivity (optional).** Tested 1e-6 and 1e-7 against
the already-used 1e-8, at N=401 and N=1401. The L2 relative error is
identical to 4 significant digits across all three tolerances at both
resolutions, and wall-clock differs by only 2–10% between the tightest
and loosest — smaller than ordinary run-to-run GPU noise. 1e-8 costs
essentially nothing extra here, so I've kept it as the standard
throughout rather than trading a real precision guarantee for a
speed gain that doesn't actually materialize.

**On "GPU FEM still too slow" as a competitive baseline** — since it
underlies several of the points above: yes, torch-fem is genuinely
faster at every size both solvers can run. But torch-fem's own peak GPU
memory at those same resolutions (5.7/17.3/35.3/69.2 GB at N=401–1401)
is already within about 13% of an 80GB A100's own ceiling at the
largest size tested, while our own solver ran that same N=1401 case on
the same hardware without hitting a comparable wall, because it never
allocates for a global matrix at all. Combined with the direct-solver
finding in point 6, I don't read this as our solver being
under-optimized — the evidence points to two solvers built for
different questions (how fast at a size that fits, vs. how far can this
go on one GPU), not one being a strictly worse version of the other. If
it's useful, I'm glad to restructure the report so torch-fem is
presented as the primary GPU-FEM baseline at the sizes it reaches, with
our matrix-free solver's role reserved for sizes beyond that.

The attached documents reflect all of the above. The only thing I'm
waiting on from you is the point 5 decision — everything else is
final unless you see something that still doesn't look right.

Best regards,

Omar
