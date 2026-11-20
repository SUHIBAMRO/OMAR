# CSM and CFM (Lone, Prabhakaran, (Tak,) Rabczuk, Zhuang): reading note, 2026-11-20

The two papers Prof. Rabczuk named in his reply of 19 November ("Both papers are now on arXiv").
The attachment had not arrived; both were found on arXiv and read (abstract, introduction,
framework, discretisation remarks, examples, conclusions).

- **CSM:** "Constitutive-Set Mechanics: variational mechanics on an admissible set of constitutive
  laws", arXiv:2610.07151 (5 Oct 2026, 33 pages).
  PDF: `advisor_feedback/2026-11-20_Rabczuk_CSM_Constitutive-Set_Mechanics_arXiv2610.07151.pdf`
- **CFM:** "Constitutive-fibre mechanics: mechanics-directed uncertainty in constitutive function
  space", arXiv:2611.22187 (17 Nov 2026, 53 pages).
  PDF: `advisor_feedback/2026-11-20_Rabczuk_CFM_Constitutive-fibre_mechanics_arXiv2611.22187.pdf`

## What they do

| | CSM | CFM |
| --- | --- | --- |
| Starts from | the constitutive model: an admissible set P of laws (thermodynamics, data within tolerance, certified regularity, material structure) | the data: the "constitutive fibre", all functions consistent with a finite test programme and a regularity bound |
| Computes | certified worst-case (upper) and, under global minimisation, two-sided bounds on the structural response over P | the range (half-width) of a quantity of interest over the fibre |
| Key idea | FE assembly makes the incremental energy linear in the law through the strain-occupation measure; equilibrium solves are the oracle of a concave maximisation; certificate needs at most n_vis + 1 states | adjoint representers; the part of the QoI sensitivity orthogonal to all observation sensitivities is hidden from the tests and sets the half-width |
| Next experiment | the experiment that most contracts the certified prediction (not where constitutive uncertainty is largest) | rank-one update ranks candidate tests; separates "more precise test" from "new type of test" |
| Examples | linear elasticity, data-constrained bar, phase-field fracture, finite-strain composite shell (4 of 226 directions visible), membrane | phase-field bar (3% hidden sensitivity -> 70% of the uncertainty), open-hole composite lamina |

## Where our verification agent fits

Both papers state the gap explicitly:

- CSM: the certificates are "value-level statements on a declared incremental problem"; the theory
  "is indifferent to how the mechanics is discretised". Extending the certificate from the energy
  to "functionals of the solution" (a margin, a residual strength) is named as the open step.
- CFM (open-hole lamina): one FE mesh (4,096 quadratic triangles, 64 divisions around the hole);
  the validity certificate "concerns one cell and the fixed discretisation; it is not a statement
  about ... the continuum problem", and the FE operators "have not been reproduced by an
  independent implementation". The constitutive discretisation "has not converged at degree 14"
  (compliance range 5.38 / 6.19 / 6.49% at degree 10 / 14 / 16).

That is the numerical half Prof. Rabczuk asked the agent to supply ("connect numerical error
control with CSM/CFM model and parameter uncertainty to obtain certified bounds on
engineering-relevant QoIs"):

1. **Total bound = constitutive range + numerical uncertainty.** For a QoI, CSM/CFM give the
   range over admissible laws on a fixed mesh; the agent gives the discretisation uncertainty
   of that QoI (refinement family, strategies, MMS, independent code). Requirement in the same
   spirit as our margin rule: numerical uncertainty <= 1/3 of the constitutive half-width.
2. **The QoIs match.** CFM's open-hole lamina reports hole-zone mean strains (mean shear strain,
   mean transverse strain in the hole zone) and compliance: region averages near a finite-radius
   hole, the same class as our plate-with-hole and B3 groove averages, where we showed that
   smooth (Gaussian) region weights converge with a rate and sharp regions only band-converge.
3. **Their discretisation questions are verification questions.** Both the FE mesh (one level)
   and the constitutive polynomial degree (not converged at 14) are refinement families the
   agent's gates and diagnosis could treat (CFM already reports three degrees; the widths are
   lower bounds, so convergence is monotone from below).
4. **Independent implementation.** CFM names the lack of an independent implementation of the
   FE operators; our second-code check (scikit-fem against torch-fem) is exactly that step.

## Points to raise with Prof. Rabczuk

1. A first joint test case: the CFM open-hole lamina (or our plate with a hole with an admissible
   set of laws), with the agent's numerical uncertainty reported next to the CFM half-width.
2. Whether the agent's QoIs (solution functionals) can serve as the "functionals of the solution"
   CSM names as its next step.
3. Access to the CSM/CFM code (CFM: "will be made openly available on GitHub upon publication").
