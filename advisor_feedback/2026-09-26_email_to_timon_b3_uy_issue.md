# Draft email to Timon: B3 Transolver accuracy issue (uy component)

Drafted 2026-09-26. NOT YET SENT -- Omar's call, per the project's own
standing convention (never send a reply without his review first).

Every number below is from a real run (GPU training logs, real FEM
solves, or the controlled local diagnostics listed), not estimated --
see PROJECT_STATUS.md's 2026-09-26 entries for the full detail behind
each one, including the exact scripts used. The two accuracy-table rows
were cross-checked directly against the raw result files on Drive
(`pfem_run/b3_training/eval_B3.json` and `eval_B3_50000.json`), not just
transcribed from a training log.

**On attachments/figures**: no plot of this specific finding exists yet
(these diagnostics were run and logged as numbers, not visualized) --
the tables below are the full result. `pfem_run/b3/B3_geometry.png` on
Drive shows the B3 bushing geometry itself (undeformed/deformed
configuration, groove detail) if a visual of the case is wanted alongside
this email; it is not a results plot. Say if a proper accuracy/training
figure would help and I will make one before sending.

---

Dear Timon,

A quick update on the B3 (3D rocking bushing) Transolver training, and a
question I'd like your input on.

**Context**: B3 uses the Deep Energy Method (same approach as B1/B2,
extended to 3D) -- the network minimizes the total elastic strain
energy directly, with no labeled FEM displacement data anywhere in the
loss. A first training run hit a real numerical instability (the
untrained network's initial output was far larger than the physical
displacement scale, pushing the energy into a singular region); I found
and fixed that, and confirmed the fix with a second run -- stable loss
throughout.

**The issue**: evaluating that checkpoint against 100 held-out real FEM
samples (never used in training) showed the stable loss curve did not
mean good accuracy, and training 25x longer did not fix it:

| Run | Gradient steps | ux rel. L2 | uy rel. L2 | uz rel. L2 | Combined | Inference speed vs. FEM |
|---|---|---|---|---|---|---|
| 2 (stability fix confirmed) | 2,000 | 32.0% | 99.9% | 36.8% | 35.7% | 823x |
| 3 (25x longer) | 50,000 | 29.4% | **106.1%** | 30.2% | 30.6% | 823x |

uy is stuck at essentially 100% error in both runs -- no better than the
network predicting zero everywhere -- and got slightly *worse*, not
better, with 25x more training. Inference speed is unaffected either way
and stays excellent.

Before assuming this needed more compute, I ran five separate controlled
diagnostics on a small fixed problem (fast to iterate on locally), each
isolating one candidate cause:

| # | Hypothesis tested | Method | Result |
|---|---|---|---|
| 1 | Just needs more training steps | The 50,000-step run itself (table above) | Ruled out -- uy got worse, not better |
| 2 | Vanishing/weak gradient for uy in the energy assembly | Measured d(energy)/d(u_net) directly via autograd, by component | Ruled out -- uy's gradient magnitude is comparable to ux/uz's (0.75x-1.0x) |
| 3 | Not enough training data diversity | Trained on a FIXED pool of 8 samples, repeated every step (pure memorization test) | Ruled out -- uy never converges below ~99-100% error even with nothing new to generalize to |
| 4 | Output-scaling constant (added for the stability fix) too small for uy | Tried the same scale, 10x larger, and a separately learnable scale, isolated to uy only | Ruled out -- final uy error stayed ~99.9-100.8% regardless; the network just shrinks its own raw output to compensate for a larger scale |
| 5 | Needs a training curriculum (ux/uz first, uy unfrozen after) | Trained ux/uz to their normal plateau with uy forced to exactly zero, then unfroze uy and continued | Ruled out -- uy collapsed straight back to ~99.8% error |

The pattern across all five: uy's true physical magnitude is genuinely
small relative to ux/uz (about 8-12x smaller, confirmed against a real
FEM solve of the same problem). My reading is that pure, unweighted
energy minimization has no structural reason to prioritize correcting a
small-magnitude component's large relative error over a large-magnitude
component's still-substantial (~30%) error, since the absolute energy
gain from fixing uy is small next to what is still on the table for
ux/uz. This looks like a real limitation of the pure physics-informed
(no-data) loss for a vector field whose components differ this much in
natural scale, not a bug in the implementation.

**The question**: I see two honest ways forward, and I'd like your view
before picking one:
1. Add a light supervised term for uy specifically, using the 100 FEM
   samples already generated (currently held out for evaluation only).
   A real, common enough fix for exactly this kind of imbalance in the
   PINN/DEM literature, but a genuine departure from the pure
   physics-informed framing we have used throughout.
2. Report this as a real, evidence-backed limitation of the pure-DEM
   approach for this geometry (a small out-of-plane field alongside a
   dominant in-plane rocking motion), rather than forcing a fix.

Happy to send the full diagnostic detail if useful.

Best regards,
Omar
