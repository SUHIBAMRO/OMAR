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
3. Points 2 and 3 were both re-measured with the corrected checkpoint
   and confirmed essentially identical to the original (wrong-checkpoint)
   measurement in both cases -- both resolved, no caveats remain.

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

| N | disp_rel_L2 | L2_rel | H1_semi_rel |
|---|---|---|---|
| 13 | 13.85% | 6.58% | 20.43% |
| 21 | 9.75% | 4.78% | 17.11% |
| 33 | 7.37% | 3.81% | 14.30% |
| 49 | 8.50% | 5.07% | 14.07% |
| 101 | 14.99% | 10.91% | 19.99% |
| 201 | 22.28% | 17.79% | 28.38% |
| 401 | 28.97% | 24.73% | 36.92% |
| 701 | 34.34% | 30.60% | 44.87% |
| 1001 | 38.82% | 34.86% | 51.82% |
| 1401 | 44.65% | 39.49% | 61.16% |

*(Table 1, abridged — full 16-row table available. See attached figure:
NO accuracy degradation, N=49 to N=1401.)*

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

Re-measured with the corrected checkpoint, including two additional
matched-memory scenarios (each method capped at the *other's* own
memory ceiling), N=21:

| Scenario | Max batch size | Peak GPU memory | Throughput |
|---|---|---|---|
| NO (own ceiling) | 8,192 | 47.6 GB | 4,203.79 samples/s |
| FEM (own ceiling) | 256 | 38.7 GB | 2.86 samples/s |
| FEM (capped at NO's 47.6 GB) | 256 | 38.7 GB | 2.86 samples/s |
| NO (capped at FEM's 38.7 GB) | 4,096 | 23.8 GB | 4,207.75 samples/s |

*(Figure attached: throughput and peak memory vs. batch size, NO vs.
FEM, own memory ceilings.)*

Peak throughput favors the NO by roughly 1,470x regardless of which
memory budget is used to cap it — FEM's own ceiling is already below
its saturation point, so giving it more memory (matching the NO's own
47.6 GB) changes nothing, while the NO barely gives anything up when
capped down to FEM's much smaller 38.7 GB (4,207.75 vs. 4,203.79
samples/s). Batching helps the NO substantially (throughput rises ~20x
from bs=1 to its own ceiling) while FEM barely benefits (~5x) —
matching your own stated expectation directly. Re-measured with the
corrected checkpoint and confirmed essentially identical to the
original (wrong-checkpoint) run — resolved, checkpoint-independent as
expected.

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

| Variant | ms/sample | Speedup vs. eager | Output diff vs. eager |
|---|---|---|---|
| eager (fp32) | 2,292.1 | 1.00x | — |
| torch.compile | 2,145.1 | 1.07x | 6.0e-7 |
| eager + TF32 | 491.2 | 4.67x | 3.2e-3 |
| compile + TF32 | 394.0 | 5.82x | 3.2e-3 |

*(Figure attached: same data as the table above. Re-measured with the
corrected checkpoint -- essentially identical to the original run,
confirming timing/speedup is checkpoint-independent as expected.)*

`torch.compile` alone gives 1.07x (correctness-checked, negligible
output difference). Enabling TF32 matmul precision (Ampere tensor
cores) gives 4.67x alone and 5.82x combined with `torch.compile` — the
best result found, at a real but small accuracy cost (~3.2e-3 relative
difference vs. strict fp32), checked directly rather than assumed.

**4. Complex-geometry example — one question before we commit training time**

Before spending real GPU-hours training a new checkpoint, we designed
and verified a candidate with real, direct evidence rather than a
hunch: extending our existing pressure-vessel-like ring case with a
single smooth local notch (a Gaussian dimple) in the inner wall — still
simply connected, no topology change, reusing our existing solver
machinery.

| Mesh resolution (elements) | Peak PK1 stress at notch | Max displacement |
|---|---|---|
| 72 | 11.70 | 0.008193 |
| 288 | 12.72 | 0.008384 |
| 1,152 | 13.45 | 0.008455 |
| 4,608 | 13.94 (+3.7%, still rising) | 0.008478 (+0.27%, essentially flat) |

A mesh-convergence study across four resolutions confirms a
genuine local stress concentration: peak stress at the notch is still
rising while the global displacement
field is essentially flat — the specific signature you
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
