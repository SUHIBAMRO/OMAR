# Draft email to Timon: B3 region-stress QoI -- full methodology, results, and open question

Drafted 2026-09-28. NOT YET SENT -- Omar's call, per the project's own
standing convention (never send a reply without his review first).

Every number below is from a real run (GPU training/evaluation logs,
real FEM solves) -- see PROJECT_STATUS.md's 2026-09-27/28 entries for
the full detail behind each one, including the exact scripts and
commit hashes. Nothing here is estimated or interpolated from an
earlier report.

---

Dear Timon,

A detailed update on B3 (the 3D rocking rubber-mount bushing): the
Transolver checkpoint now reproduces displacement, total strain
energy, and reaction moment very well, but regional Cauchy stress --
one of the four success criteria you named -- is far off target, and I
want to walk you through exactly how we measured that, what we ruled
out, and what's still open, before deciding how to proceed.

## 1. Setup

**Checkpoint**: `checkpoint_50000.pt`, the fourth training run. Deep
Energy Method (network minimizes total elastic strain energy directly,
no labeled displacement anywhere in the loss), Transolver architecture,
50,000 gradient steps, batch size 8. Material fields E(theta,r,z) and
nu(theta,r,z) vary spatially per sample (3D Gaussian random field,
E_mean=1000/std=200, nu_mean=0.45/std=0.02); the rocking angle phi
varies as a per-sample scalar. Mesh: 6,840 elements (21x20x19),
production resolution throughout.

**A real fix found and confirmed this cycle**: the first three training
runs showed unstable, oscillating accuracy (one displacement component
swinging 79-111% error across checkpoints, with the final checkpoint
sometimes worse than an earlier one). Root cause: E, nu, and phi were
fed to the network at their raw physical scales (~1000, ~0.45, ~0.05 --
three orders of magnitude apart), with no input normalization. Adding
per-channel normalization (fixed mean/std from a 200-sample reference
batch, the same convention already used in the 2D B1/B2 models) fixed
this: combined displacement error dropped from 26.2% to 1.88%, and the
oscillation disappeared entirely. This is now confirmed on a genuinely
disjoint 100-sample held-out set (seed never touched by training, this
was checked directly after an earlier held-out set turned out to
partially overlap with training seeds).

## 2. How each QoI was computed (no new FEM solve needed)

Every QoI is computed directly from a known displacement field (the
true FEM one, or the network's own prediction) via the exact same
energy functional the training loss already uses:

- **Total strain energy**: the training objective itself, evaluated
  on a given field.
- **Reaction moment**: since B3's entire load is prescribed
  displacement (no external nodal force anywhere), the reaction at a
  constrained node equals d(energy)/d(u) there, by the same
  variational argument any FEM solver uses internally to report
  reactions -- computed via autograd on a known field, not a new
  nonlinear solve.
- **Regional Cauchy stress**: sampled at Gauss points within a fixed
  physical region near the groove's deepest point (continuously
  bonded, no boundary-condition transition contamination). First
  Piola-Kirchhoff stress P = d(psi)/d(F) via autograd (exact for
  Neo-Hookean, not a hand-transcribed closed form), then
  sigma = (1/det F) P F^T.

**A real methodology problem found and fixed in the region-stress
metric itself**: the first version of this metric took a volume-
weighted SIGNED average of sigma_xx over the region, then computed
|pred-true|/|true| per sample. This blows up arbitrarily whenever the
true signed average happens to pass near zero -- which it does for
several samples, since the region genuinely contains both tension and
compression that partially cancel in a signed average. This produced
an apparent 1683% mean error that was significantly a measurement
artifact, not a real accuracy figure -- the same failure mode this
project already hit and fixed once before (2026-09-21) for the pure
FEM-vs-FEM mesh-convergence comparison. The fix, applied the same way
here: a volume-weighted, FULL-TENSOR relative field error (squares
each Gauss point's contribution before summing, so a near-zero signed
average cannot collapse the denominator). Verified with a real identity
check before trusting it: feeding the same field in as both "true" and
"predicted" gives exactly 0.0 for every sample, not just "small."

**A known resolution caveat, present from the start**: at 6,840
elements, only 6 Gauss points fall inside the fixed stress region --
far below the reliability threshold (20) this project's own
mesh-convergence work already established for percentile statistics.
A separate FEM-only convergence study (constant material, done
2026-09-21) found the region-average stress itself needs ~20,808
elements for 5% accuracy against a 243,360-element reference -- B3's
training/evaluation resolution is below that bar even for the "ground
truth" side of the comparison, not just for the network's own accuracy.

## 3. Results: the production checkpoint on 100 independent held-out samples

| QoI | Metric | Result |
|---|---|---|
| Total strain energy | mean rel. error | **0.30%** |
| | median | 0.24% |
| Displacement (ux, uy, uz combined) | mean rel. L2 | **1.88%** |
| | ux / uy / uz individually | 1.94% / **16.16%** / 1.42% |
| Reaction moment | mean rel. error | **2.67%** |
| | median | 1.98% |
| Regional Cauchy stress (naive signed-average metric) | mean rel. error | 1683.56% (artifact -- see above) |
| | median | 382.72% |
| | pooled-RMS (denominator-free) | 310.90% |
| **Regional Cauchy stress (corrected, full-tensor field metric)** | **mean rel. error** | **71.77%** |
| | **median** | **68.45%** |
| | std | 19.90% (n=100/100) |

Displacement/energy/reaction are all strong and reliable. Regional
Cauchy stress remains far from the 5-10% target even after the metric
fix removes the measurement artifact.

**One more pattern worth flagging**: uy's own individual relative L2
error (16.16%) is much worse than ux/uz (~1.4-1.9%), yet the combined
displacement metric (1.88%) looks excellent, because uy's physical
magnitude is 8-12x smaller than ux/uz and so contributes little to a
magnitude-weighted combined norm. This is the same qualitative pattern
as the region-stress result: a small-magnitude quantity is measurably
less accurate than the dominant ones, while global/combined metrics
mask it. We flagged this exact uy pattern to you separately in
September; it looks like the same underlying cause is showing up again
here, one level more local.

## 4. Follow-up diagnostics (before proposing any fix)

**Diagnostic A -- does finer EVALUATION change the picture?** Took 10
of the held-out samples, interpolated their already-generated material
fields onto a much finer mesh (43,400 elements, 36 region Gauss
points -- 6x more elements, 6x more region samples), solved a real new
FEM problem there, and queried the SAME trained network zero-shot at
that finer mesh (no retraining). Comparing the TRUE region-average
stress directly between the two resolutions, for the same physical
samples:

| Sample | True @ 6,840 el | True @ 43,400 el | Ratio |
|---|---|---|---|
| 0 | 0.309 | 0.584 | 1.9x |
| 4 | 0.007 | 0.542 | **78.6x** |
| 7 | -0.105 | 0.298 | **sign flip** |
| 8 | 0.044 | 0.907 | **20.5x** |
| (6 others) | | | ~1.7-2.8x |

The "ground truth" region-stress value itself is not stable at
production resolution -- confirming, with real per-sample evidence,
that the earlier FEM-only convergence study's conclusion (6,840
elements is well below what this QoI needs) is not just a theoretical
concern. But the network's own prediction barely moved between the two
evaluation resolutions of the same fixed checkpoint (mean -0.96 vs.
-0.87) -- it is not tracking the true, sign-variable local signal at
either resolution, so this does not by itself explain away the gap.

**Diagnostic B -- does finer TRAINING help?** A short (5,000-iteration,
not the full 50,000) pilot retrain at the same finer resolution
(43,400 elements), same architecture/normalization/distributions as
the production run -- to test whether giving the training's own energy
integral more resolution in the region reduces the gap, evaluated
against an independent FEM set solved fresh at that same resolution:

| QoI | Pilot result (5,000 iters @ 43,400 el) |
|---|---|
| Energy | 1.22% |
| Reaction moment | 3.57% |
| Regional Cauchy stress (corrected metric) | **233.72%** (worse than production's 71.77%) |

This result should NOT be read as "finer training doesn't help" --
the pilot changed two things at once (resolution AND a 10x cut in
iteration budget), and its loss curve was still visibly not converged
at 5,000 steps. A clean version of this test (same iteration budget,
resolution-only change) has not been run; the measured real cost is
0.784s/iteration at this resolution (5.6x the production rate), so a
full 50,000-iteration run here would cost roughly 11 hours on an A100.

## 5. Where this leaves us

- Displacement, energy, and reaction are all confirmed strong and
  reliable.
- Regional Cauchy stress has a real, now precisely quantified gap
  (71.77% mean, artifact-corrected), and it is not simply a stale
  measurement problem -- both the metric formula and the evaluation
  resolution have been checked and corrected/tested directly.
- The most likely explanation, not yet fully proven: an energy-only
  (Deep Energy Method) loss has little structural incentive to get a
  small-magnitude, locally sign-variable quantity right, whether that
  is the uy displacement component (flagged earlier) or the region-
  averaged local stress (this report) -- both look like the same
  underlying limitation showing up at different scales.
- We have not yet run the clean, decisive version of the resolution
  test (same iteration budget, resolution changed alone), and have not
  committed the ~11 GPU-hours a full run at finer resolution would
  cost.

**The question**: given the pattern above, do you think this is worth
the ~11-hour investment in a clean finer-resolution retrain, or would
you rather we treat this as a documented limitation of pure energy-only
training for this specific QoI and move on? If the former, is there a
training-strategy change (e.g. a light explicit local-stress
supervision term, similar to what we discussed for uy) you'd recommend
trying alongside resolution, rather than resolution alone?

Best regards,
Omar
