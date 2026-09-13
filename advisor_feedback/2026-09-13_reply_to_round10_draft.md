# DRAFT reply to Timon's round-10 email — NOT YET SENT

Review before sending. Two things to decide first:
1. Point 1's numbers are real and verified, but a retraining run is
   in progress to see if a wider training-resolution range closes some
   of the gap — you may want to wait for that before sending, or send
   now and follow up separately once it's done. The draft below sends
   now and says a follow-up is coming.
2. Point 4 asks Timon a direct question (confirm the B7 design) —
   this IS the "ask Timon first" step from PROJECT_STATUS.md's own
   standing rule. Sending this email is what discharges that requirement.

---

Subject: Round-10 follow-up: accuracy-matched comparison, throughput, profiling, and one question on the geometry example

Dear Timon,

Here is where things stand on your five round-10 points.

**1. Accuracy-matched comparison (NO vs. coarsest suitable FEM)**

The NO's own accuracy at N=1401, checked directly against a real FEM
ground truth there for the first time: displacement relative L2 error
44.65% (L2-norm 39.49%, H1 semi-norm 61.16%). This degrades smoothly
and monotonically away from the trained resolutions — best at N=33/37
(≈7.4%), rising to 8.5% at N=49, 15.0% at N=101, and up to the N=1401
number above. Ground truth convergence was confirmed at every single
resolution tested (relative residual 1.8e-11 to 1.0e-10 throughout).

Comparing this against native FEM (torch-fem) at the same low
resolutions, across the same QoI set (L2, H1, tangent energy, peak PK1
stress, reaction resultant): a near-degenerate FEM mesh (as coarse as
N=3-9, i.e. 9-81 nodes) already matches or beats the operator's own
best-case accuracy in L2, H1, energy, and reactions.

One QoI needs an explicit caveat rather than a clean number: peak PK1
stress, evaluated at a fixed location/value located once from a fine
reference (same convention for both methods). Both methods converge to
it very slowly — torch-fem's own error there barely improves from 83%
to 62% across the whole low-N range, while its other QoIs converge at
the expected rate over the same range. The located point sits
essentially at the domain corner where the fixed boundary meets the
free edge — a classic boundary-condition-transition location prone to
a stress singularity in elasticity, where the continuum target may not
even be well-posed for either method to converge to in the usual sense.
We think this metric should be reported with that caveat attached
rather than used as a clean deciding factor.

We are also mid-way through retraining the operator on a wider spread
of resolutions (adding N=101 and 201 to the original 21/33) to see how
much of the degradation above is an ordinary, addressable
resolution-generalization gap. Validation error has already dropped
substantially during training, but we have not yet re-run the same
rigorous FEM-referenced check on the new checkpoint — will follow up
with that once it's verified.

**2. Batch size and throughput at matched GPU memory**

At each method's own maximum feasible batch size on the same GPU: the
NO reaches batch size 8,192 (47.6 GB peak, 4,212.66 samples/s); FEM
reaches only batch size 256 (38.7 GB peak, 2.86 samples/s) before
running out of memory. Peak throughput favors the NO by roughly 1,473x.
Batching helps the NO substantially (throughput rises ~20x from bs=1 to
its own ceiling) while FEM barely benefits (~4.8x) — matching your own
stated expectation directly.

**3. NO inference optimization and profiling**

Pure GPU forward-pass time after warm-up (excluding data
transfer/preprocessing), fp32, batch size 1, N=1401: 2,290 ms/sample.
Peak GPU memory: 23.16 GB — notably higher than our own GPU-FEM
solver's peak memory at the same N (15.6-20.5 GB), so the NO is not the
more memory-frugal option here either. Profiler breakdown: ~67% of time
in batched matmuls (the slice-based attention operating on ~2M mesh
nodes), ~22% in the MLP layers — the cost is in genuine core compute at
this token count, not an obvious inefficiency.

We tried to speed this up before treating 2.29s as final:
`torch.compile` alone gives 1.07x (correctness-checked, negligible
output difference). Enabling TF32 matmul precision (Ampere tensor
cores) gives 4.72x alone and 5.80x combined with `torch.compile` — the
best result found, at a real but small accuracy cost (4.7e-3 relative
difference vs. strict fp32), checked directly rather than assumed.

**4. Complex-geometry example — one question before we commit training time**

Before spending real GPU-hours training a new checkpoint, we designed
and verified a candidate with real, direct evidence rather than a
hunch: extending our existing pressure-vessel-like ring case with a
single smooth local notch (a Gaussian dimple) in the inner wall — still
simply connected, no topology change, reusing our existing solver
machinery. A mesh-convergence study across four resolutions confirms a
genuine local stress concentration: peak stress at the notch is still
rising (+3.7% at the finest mesh tested) while the global displacement
field is essentially flat (+0.27%) — the specific signature you
described of a globally-converged field hiding an under-resolved local
quantity.

Before training a full model on this, we wanted to check with you
directly: does this count as an acceptable instance of your "pressure
vessel with local details" suggestion, or would you prefer we pursue a
different example instead?

**5. Break-even analysis**

Planned as the direct follow-on once item 1's numbers are finalized
(after the retraining check mentioned above) — an update here would be
premature before that.

Let me know if the framing above looks right, particularly on point 1's
stress-singularity caveat and on point 4.

Best regards,
Omar
