# OpenPINN/PINO (Wang, ..., Rabczuk, ..., Sun): reading note, 2026-11-20

Prof. Rabczuk's attachment to the 19 November reply. The full paper (21 pages + references) was read;
the PDF is `advisor_feedback/2026-11-19_Rabczuk_attachment_OpenPINN.pdf`.

## What it is

- **Framework.** An open-source PyTorch framework for PINNs and neural operators.
- **Shared contracts.** `ProblemSpec` covers geometry, physics, IC/BC and data; `MethodSpec` covers model, sampling, loss, optimisation and differentiation. Both drive PINN, operator and custom back-ends.
- **Five-module agent pipeline:**
  1. Problem Understanding;
  2. Solver Planning;
  3. Specification Validation (a compliance gate);
  4. Code Building (deterministic export);
  5. Diagnostic Evaluation (parses training logs and results; returns "concerns" and non-binding repair hints).
- **Demonstrations:**
  - Burgers PINN recipes: one method axis changed at a time.
  - FNO on periodic Poisson, with 16 held-out forcings.
  - Agent-assisted plate with a hole.

## The plate-with-hole case

- **Problem.** 2 m x 2 m plate, hole radius 0.25 m at (1,1), E = 210 GPa, nu = 0.3, plane stress. Left edge fixed; 1 MPa traction on the right edge.
- **Round 1.** The SI-unit run fails: displacement relative L2 = 3.65e4. The diagnostic agent flags the residual scale. A researcher nondimensionalises; result 0.758.
- **Round 2.** A second diagnosis flags x-momentum and hole-traction residuals. A researcher adds fixed-batch L-BFGS; final errors 3.51%, 4.13% and 6.31% (three seeds), against a 10% target.
- **Who repaired.** Both repairs were researcher-led; the agents located cues.
- **The reference.** An FEM solution on a single probe mesh, h = 0.025. The paper reports no convergence or uncertainty estimate for it.

## Relation to our framework

| | OpenPINN/PINO | Our verification framework |
| --- | --- | --- |
| Specification | `ProblemSpec` + `MethodSpec`, validated before code is built | `ProblemSpec` (same idea and name) + check plan, validated against a check catalogue, rules and tolerances |
| What is diagnosed | **surrogate training**: loss scale, residual imbalance, budget | **the numerical reference**: why a QoI does not converge (math/physics, discretisation, solver, QoI, surrogate) |
| Diagnosis output | concerns + repair hints; repairs by a researcher | supported/refuted hypotheses; cheap tests and confirmations run by the agent; next investigation named |
| Reference | one FEM mesh, taken as exact | refinement family, strategies, MMS, independent code; tolerance from the purpose (margin to surrogate error) |
| Success criterion | relative L2 below a stated target (10%) | the reference uncertainty must be at most 1/3 of the surrogate error (margin 3) |
| Human role | stated qualitatively | counted (59 interventions, by category and finder) |

**Complementary, not competing.**
- **Their diagnostic agent covers our "surrogate" cause class.** That is the class (U1) our engine leaves open.
- **Our verification would sit between their FEM reference and their metrics.** Their 10% target is meaningful only if the reference uncertainty is well below it.
- **Their plate with a hole is a natural held-out test for our agent.** The hole is a finite-radius concentration, as in B3. The clamped edge meets the free top and bottom edges at two corners: clamped/free singularities, as in B1. So the agent has to separate a converging hole peak from a diverging corner peak on one problem. This test is being run (task F9 in PROJECT_STATUS).

## Points to raise with Prof. Rabczuk

1. **A combined pipeline.** Should our verification agent become the reference-verification module of OpenPINN (the "Reference verification" box between computation and postprocessing)?
2. **Their reference.** Their plate-with-hole reference error (h = 0.025) could be quantified by our agent and reported next to the 3.5-6.3% surrogate errors.
3. **Shared vocabulary.** The `ProblemSpec` fields overlap with ours; aligning them would let both pipelines exchange problems.
