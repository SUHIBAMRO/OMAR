# Draft email to Timon: B3 Transolver accuracy issue (uy component)

Drafted 2026-09-26. NOT YET SENT -- Omar's call, per the project's own
standing convention (never send a reply without his review first).

Every number below is from a real run (GPU training logs, real FEM
solves, or the controlled local diagnostics listed), not estimated --
see PROJECT_STATUS.md's 2026-09-26 entries for the full detail behind
each one, including the exact scripts used.

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
samples showed the stable loss curve did not mean good accuracy. The
relative L2 error is very uneven across the three displacement
components: ~30% for the two components in the bushing's main rocking
plane (ux, uz), but the third (out-of-plane) component, uy, is stuck at
essentially 100% error -- no better than the network simply predicting
zero. Training 25x longer (2,000 to 50,000 real gradient steps) did not
close this gap and made uy slightly worse.

Before assuming this needed more compute, I ran five separate controlled
diagnostics, each isolating one candidate cause:
1. Step count -- ruled out by the 50,000-step run itself.
2. A vanishing/weak gradient specific to uy in the energy assembly --
   ruled out by measuring the actual gradient magnitude directly (it is
   comparable to ux/uz's).
3. Insufficient training diversity -- ruled out by a pure memorization
   test on a fixed pool of 8 samples (uy still failed to converge even
   with nothing new to generalize to).
4. The output-scaling constant added for the instability fix being too
   restrictive for uy specifically -- ruled out by testing a much larger
   and a separately-learnable scale for just that component; neither
   helped, and the optimizer simply shrank its own output further to
   compensate.
5. A curriculum schedule (train ux/uz first with uy frozen at zero, then
   unfreeze it) -- ruled out; uy collapsed back to ~100% error anyway.

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
