# Automated verification framework: status summary (DRAFT for Omar's review, not sent)

Dear Professor Rabczuk,

Following your suggestion to build an automated verification framework, this note summarises the current state. Every number below comes from runs in the repository (`Practical_Examples/omar_pfem/verification_agent/`, see its README).

## 1. What the framework does

1. **Deterministic gates decide every verdict.** An LLM never sets a verdict.
   - **Three-level check (Celik et al. 2008).** Observed order, GCI and Richardson extrapolation. The observed order must lie in the element's asymptotic window [0.5, 2k+1].
   - **Five-level check.** A least-squares fit v = a + b h^p over the finest five levels, in the spirit of Eça & Hoekstra 2014. It certifies a rate only if all of these hold: the sequence is monotone, the fit is good, p lies inside the window, and leave-one-out refits keep p inside the window.
   - **Strategy robustness.** Results must agree across different refinement strategies.
   - **Anchored robustness.** Every strategy must agree with the family's limit, not only with the other strategies.
   - **Singular-order check.** Observed orders are compared with the corner exponent from the Williams eigenproblem, solved numerically. The solver reproduces the known L-corner values 0.5445 and 0.9085.
2. **Problem and method trees with rules R1–R8.** Following the GRAFT paper, the rules state:
   - which checks every quantity requires;
   - which verdicts theory predicts.

   If an LLM plan omits a required check, code adds it and records that.
3. **Tolerances are derived from the purpose and set by code.** The required reference uncertainty is one third of the best recorded surrogate error for that quantity. It is capped by per-class ceilings, and the source of every surrogate error is recorded. LLM-proposed tolerances are replaced, and each replacement is recorded.
4. **LLM roles: formalizer, proposer, critic and advisor.** Code audits the advisor's statements against the gate verdicts. That includes claims of "convergence" or "monotonicity" for quantities that only passed the stability band.
5. **Adaptive refinement controller.** It stops refining once every quantity is decided, under a time budget.

## 2. B1 (Q4, uniform N = 11, 21, 41, 81, 161)

1. **All seven smooth quantities pass with a demonstrated rate from both convergence checks.** These are the five point and maximum displacements, the strain energy, and the interior region average of |P|. The fine-mesh uncertainty is at most 0.02%.
2. **The orders agree with corner theory.** For the clamped–free 90° corner (plane strain, ν = 0.3), the smallest exponent is λ = 0.7112.
   - Strain energy: observed order 1.43, against the predicted 2λ = 1.42.
   - Displacements: 1.47–1.92, against the predicted lower bound of 1.42.
3. **The peak stress grows like h^-0.275, against the theoretical h^-0.289 (λ − 1).** It is therefore not a usable reference value.
   - **Prediction made in advance.** A power law fitted on ten earlier meshes predicted the N = 161 peak within 0.4% before that mesh was solved.
   - **Two different meshes with the same smallest element give the same peak** (0.03% apart). The peak is set by the corner element size alone.

## 3. B3 (Q20 groove, seven levels from 2,376 to 45,000 elements)

1. **Global quantities converge with a rate.**
   - Reaction moment: order 1.32, GCI 0.21%; the five-level uncertainty is 0.35%.
   - Strain energy: order 1.37, GCI 0.19%.

   The two are linked (W ≈ ½Mφ), so they count as one piece of evidence. Both meet the requirement: the moment's surrogate error is 1.38%, so the required uncertainty is 0.46%.
2. **The groove-region averages over the stored sharp sphere are only stable within about 1–1.5%.** No rate can be shown.
   - **Cause.** A region-definition study on the same solutions traced this to sampling: only 2–36 Gauss points fall inside the sharp-edged sphere.
   - **Alternative definition.** A smooth Gaussian weight (length 2R₀) removes the oscillation: sign changes fall from 24 of 27 to 3 of 27. The normal-stress and pressure averages then converge with order about 2 and an uncertainty of about 0.2%.
3. **Pointwise peaks do not converge,** as reported earlier.
4. **Region averages of stress norms (Frobenius, von Mises) are unreliable.**
   - **Cause.** They pick up a Gauss-point pressure oscillation, which is consistent with near-incompressibility (ν = 0.45).
   - **Integration check.** A re-solve with full 27-point integration shows that the default reduced 2×2×2 rule does not cause this. With full integration the oscillation is the same or larger. Global quantities change by only 0.08% at 10,200 elements.
5. **Adequacy for the current comparison.** The surrogates' groove-stress errors are 600% or more, so even the sphere averages, accurate to about 1.5%, are adequate to show that the surrogates fail there.
6. **Adaptive controller.** For the global quantities and the Gaussian-weighted averages, it reaches the same verdicts as the full family with levels up to 23,520 elements, at 36% of the family's computing time.

## 4. Decisions I would ask you about

1. Should the B3 groove reference use the smooth Gaussian-weighted region average instead of the sharp sphere? The manuscript is unchanged so far.
2. Is a margin of 3 (reference uncertainty at most one third of the surrogate error) acceptable as the acceptance criterion?
3. Should the manuscript's wording be narrowed to "region-averaged normal stresses", since the shear and stress-norm averages are not reliable?
4. The surrogate active-learning part of the plan has not been started yet. I would like to discuss its scope with you first.

Best regards,
Omar
