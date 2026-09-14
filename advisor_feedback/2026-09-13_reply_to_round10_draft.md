# DRAFT reply to Timon's round-10 email — NOT YET SENT

Review before sending. All five points are now numerically complete and
verified with real GPU results, including a finished multi-resolution
retraining that substantially improved point 1's story and a real
accuracy-matched break-even for point 5. One thing to decide:
1. Point 4 asks Timon a direct question (confirm the B7 design) —
   this IS the "ask Timon first" step from PROJECT_STATUS.md's own
   standing rule. Sending this email is what discharges that requirement.

---

Subject: Round-10 follow-up: accuracy-matched comparison, throughput, profiling, and one question on the geometry example

Dear Timon,

Here is where things stand on your five round-10 points.

**1. Accuracy-matched comparison (NO vs. coarsest suitable FEM)**

Along the way we found and fixed a checkpoint-loading bug that had
caused the round-10 accuracy numbers to be measured against the wrong
model at one point; every number below is from the corrected,
fingerprint-verified checkpoint. We also went one step further: given
that accuracy degraded away from the original training resolutions
(N=21, 33), we retrained a new checkpoint on a wider spread (N=21, 33,
101, 201) to test whether that gap was addressable. It was, dramatically:

| N | Original checkpoint (N=21,33 only) | Retrained (N=21,33,101,201) |
|---|---|---|
| 13 | 13.85% | 11.74% |
| 33 | 7.37% | 3.50% |
| 49 | 8.50% | 2.32% |
| 101 | 14.99% | 3.55% |
| 201 | 22.28% | 4.85% |
| 701 | 34.34% | 5.88% |
| 1401 | 44.65% | 5.85% |

*(disp_rel_L2 vs. a real FEM ground truth at every N, both checkpoints;
full 16-row tables available. Ground-truth convergence confirmed at
every resolution, relative residual 4.6e-10 to 1.0e-10 throughout. See
attached figure: NO accuracy degradation, N=49 to N=1401.)*

Widening the training-resolution set does not just shrink the error —
it changes its whole shape: the original checkpoint degraded
monotonically the further N got from 21/33, up to 44.65% at N=1401. The
retrained checkpoint instead gets progressively more accurate up to
~N=45-49 (best 2.3%), then plateaus around 5.8-5.9% instead of
continuing to climb — a 7.6x reduction in error at N=1401.

Comparing the retrained checkpoint's full QoI set (L2, H1, tangent
energy, peak PK1 stress, reaction resultant) against native FEM at the
same low resolutions (N=3-49):

- For L2, H1, tangent energy, and reactions, a near-degenerate FEM mesh
  (N=3-9) still matches or beats the operator's accuracy — unchanged
  from before.
- Peak PK1 stress (evaluated at a fixed physical location located once
  from a fine reference, not the coarse mesh's own sample-max) tells a
  genuinely different story now: the retrained checkpoint's peak-stress
  accuracy (42-70% relative error, improved at every single N) is now
  BETTER than anything torch-fem achieves in the same low-N range (62%
  at best, N=49) for every NO resolution from N=29 upward. FEM would
  need N>49 (untested here) to match it.
- As a result, the "coarsest suitable FEM" is now bound by tangent
  energy rather than peak stress for most resolutions, and dropped from
  N=17-45 (original checkpoint) to a genuinely near-degenerate N=9-17
  (retrained checkpoint) for almost every resolution tested.

The same caveat as before still applies to peak stress specifically:
the located point sits essentially at the domain corner where the fixed
boundary meets the free edge — a classic boundary-condition-transition
location prone to a stress singularity, where the continuum target may
not be well-posed for either method to converge to cleanly. Both
methods still converge to it slowly. We think this metric should carry
that caveat rather than be read as an unqualified result — but the
improvement itself is real and directly measured against independent
FEM ground truth, not a training-validation artifact.

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

Using the retrained checkpoint's own coarsest-suitable FEM at N=1401
(N=11, from the crossover above) against the NO's real inference cost
there, and the real training wall-clock for the retrained checkpoint
(41,881s, ~11.6h — confirmed from its own training log; it stopped
itself via early stopping, 8 checks with no improvement, exactly as
configured):

| NO inference mode | ms/sample @ N=1401 | vs. FEM@N=11 (1,625.6 ms) | Break-even |
|---|---|---|---|
| eager fp32 (default) | 2,292.1 | 0.71x (slower) | **Never** — the accuracy-matched FEM mesh is already cheaper per sample |
| compile + TF32 | 394.0 | 4.13x (faster) | After 34,005 samples (~3.72 GPU-hours of NO inference) |

This is the one place in the whole investigation where the NO's default
(unoptimized) deployment mode does not come out ahead: run naively, it
never repays its own training cost against a FEM mesh that is only as
accurate as it needs to be. With the inference optimizations from point
3 (torch.compile + TF32, already verified, ~3.2e-3 relative accuracy
cost), it both wins per-sample (4.13x) and repays its training cost
cheaply — about 3.72 GPU-hours against 11.6 GPU-hours of training. We
think the honest framing is that the operator's economic case depends
on deploying it with these optimizations, not on its default settings.

Let me know if the framing above looks right, particularly on point 1's
stress-singularity caveat, point 4's design question, and point 5's
"only pays off with optimized inference" framing.

Best regards,
Omar
