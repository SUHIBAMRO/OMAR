# Automated verification framework: status summary (final version for Omar's review, not sent)

Review copy: Claude Doc "Automated verification framework: status summary"
(https://claude.ai/artifact/3TbzJAshrU6VZ2HviGZdeH). This file holds the same text.

Dear Professor Rabczuk,

Following your suggestion to build an automated verification framework, this note summarises where it stands. Every number below comes from runs in the repository (`Practical_Examples/omar_pfem/verification_agent/`, see its README).

## 1. What the framework does

1. **Deterministic checks decide every verdict.** An LLM never sets a verdict.
   - **Three-level check (Celik et al. 2008).** Observed order, GCI and Richardson extrapolation. The observed order must lie in the element's asymptotic window [0.5, 2k+1].
   - **Five-level check.** A least-squares fit v = a + b h^p over the finest five levels, in the spirit of Eça & Hoekstra 2014. It certifies a rate only if the sequence is monotone, the fit is good, p lies inside the window, and leave-one-out refits keep p inside the window.
   - **Combining the two.** Either check failing fails the quantity. A rate from the three-level check is not accepted when the five-level check finds the sequence non-monotone.
   - **Strategy robustness.** Results must agree across different refinement strategies.
   - **Anchored robustness.** Every strategy must agree with the family's limit, not only with the other strategies.
   - **Singular-order check.** Observed orders are compared with the corner exponent from the Williams eigenproblem, solved numerically. The solver reproduces the known L-corner values 0.5445 and 0.9085.
2. **Problem and method trees with rules R1–R8.** Following the GRAFT paper, the rules state which checks every quantity requires and which verdicts theory predicts. If an LLM plan omits a required check, code adds it and records that.
3. **Tolerances follow from the purpose and are set by code.** The required reference uncertainty is one third of the best recorded surrogate error for that quantity, capped by per-class ceilings. The source of every surrogate error is recorded. LLM-proposed tolerances are replaced, and each replacement is recorded.
4. **LLM roles: formalizer, proposer, critic and advisor.** Code audits the advisor's statements against the verdicts, including claims of "convergence" or "monotonicity" for quantities that only passed the stability band.
5. **Adaptive refinement controller.** It stops refining once every quantity is decided, under a time budget.

## 2. B1 (Q4, uniform N = 11, 21, 41, 81, 161)

1. **All seven smooth quantities pass with a demonstrated rate from both convergence checks.** These are the five point and maximum displacements, the strain energy, and the interior region average of |P|. The fine-mesh uncertainty is at most 0.02%.
   - **Smooth interior average.** The same |P| average taken with a Gaussian weight also passes with a rate. Its three- and five-level orders agree (1.61 and 1.79); for the sharp-edged region they do not (0.76 and 2.34).
2. **The orders agree with corner theory.** For the clamped–free 90° corner (plane strain, ν = 0.3), the smallest exponent is λ = 0.7112.

| Quantity | Observed order | Theory |
| --- | --- | --- |
| Strain energy | 1.43 | 2λ = 1.42 |
| Displacements | 1.47–1.92 | at least 1.42 |
| Peak stress growth | h^−0.275 | h^−0.289 (λ − 1) |

3. **The peak stress is therefore not a usable reference value.**
   - **Prediction made in advance.** A power law fitted on ten earlier meshes predicted the N = 161 peak within 0.4% before that mesh was solved.
   - **Two different meshes with the same smallest element give the same peak** (0.03% apart). The peak is set by the corner element size alone.

## 3. B3 (Q20 groove, eight levels from 2,376 to 59,400 elements)

1. **Global quantities converge with a rate.**
   - Reaction moment: order 1.35, GCI 0.18%; the five-level uncertainty is 0.15%.
   - Strain energy: order 1.41, GCI 0.16%; the five-level uncertainty is 0.14%.
   - The two are linked (W ≈ ½Mφ), so they count as one piece of evidence. Both meet the requirement: the moment's surrogate error is 1.38%, so the required uncertainty is 0.46%.
2. **The groove-region averages over the stored sharp sphere are only stable within about 1% under refinement, and up to 2.4% across refinement strategies.** No rate can be shown.
   - **Cause.** A region-definition study on the same solutions traced this to sampling: only 2–50 Gauss points fall inside the sharp-edged sphere.
   - **Alternative definition.** A smooth Gaussian weight (length 2R₀) removes the oscillation. The comparison below is on the same solutions.
   - **Independent strategies.** Besides axial grading toward the groove, three radial gradings toward the groove surface were added. Stronger radial grading inverts elements at the groove, so the strategies are limited to valid meshes.

| Region definition | Gauss points, coarsest to finest | Sign changes, finest five levels (11 quantities) | Normal-stress averages with a rate | Spread across six strategies (normal stresses) |
| --- | --- | --- | --- | --- |
| Sharp sphere (radius R₀) | 2–50 | 28 of 33 | none | up to 2.4%; σzz fails the 2% limit |
| Gaussian weight (length 2R₀) | 35–916 (effective) | 4 of 33 | 7 of 7: orders 2.2–2.8, uncertainty 0.11–0.18% | at most 0.5% |

3. **Pointwise peaks do not converge,** as reported earlier.
   - **On the groove surface itself.** The Gauss-point stresses were extrapolated to the surface nodes. At the node at the groove root, σxx, σzz and pressure are stable within 0.4–0.6% over the finest three levels, but no rate can be shown. The surface maxima near the root oscillate.
4. **Region averages of stress norms (Frobenius, von Mises) are unreliable.**
   - **Cause.** They pick up a Gauss-point pressure oscillation, consistent with near-incompressibility (ν = 0.45).
   - **Integration check.** A re-solve with full 27-point integration shows that the default reduced 2×2×2 rule does not cause this. With full integration the oscillation is the same or larger. Global quantities change by only 0.08% at 10,200 elements.
5. **Adequacy for the current comparison.** The surrogates' groove-stress errors are 600% or more. Even the sphere averages, accurate to about 2.5%, are adequate to show that the surrogates fail there.
6. **Adaptive controller.** Its convergence verdicts equal the full family's for all 28 quantities.
   - It decides the global quantities and five of the seven Gaussian-weighted normal averages with a rate at 15,960 elements, for 12% of the family's computing time.
   - The remaining two need levels up to 45,000 elements (66%).

## 4. Decisions I would ask you about

1. Should the B3 groove reference use the smooth Gaussian-weighted region average instead of the sharp sphere? The manuscript is unchanged so far.
2. Is a margin of 3 (reference uncertainty at most one third of the surrogate error) acceptable as the acceptance criterion?
3. Should the manuscript's wording be narrowed to "region-averaged normal stresses", since the shear and stress-norm averages are not reliable?
4. The surrogate active-learning part of the plan has not been started yet. I would like to discuss its scope with you first.

Best regards,
Omar
