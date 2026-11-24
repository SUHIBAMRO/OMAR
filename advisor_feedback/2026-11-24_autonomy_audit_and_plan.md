# Autonomy audit and plan (after Professor Rabczuk's reply of 24 November)

His priorities, in order: (1) an autonomous end-to-end verification workflow — literature,
problem definition, meshing, MMS where suitable, simulation, convergence analysis, diagnosis —
in which the agent decides what to investigate next and learns from the results;
(2) new geometry families; (3) CSM/CFM at the very end. Next report: only the new results.

## Which steps are autonomous today (honest audit of the code as of commit 68c4573)

"Autonomous" = runs from the previous step's output with no person writing code, choosing
inputs or fixing anything. "Scripted" = works, but a person wrote problem-specific code or chose
what to run. "Manual" = a person does it.

| Step | Status | What runs by itself | What a person still does |
| --- | --- | --- | --- |
| 1. Literature | partly | `literature_agent`: from a given PDF and section number, an LLM fills a benchmark card; code checks every quote against the paper and rejects unsupported fields | finding and choosing the paper; naming the section |
| 2. Problem definition | partly | formalizer (LLM) turns the problem text into a specification; code checks it | turning the specification into something runnable: a data adapter or a `generic2d` description is written by hand |
| 3. Meshing | partly | `generic2d`: 2D domains built from unit squares, uniform Q2 and an independent P2 ladder; corner analysis and graded meshes chosen by the agent | every other geometry (curved, 3D, B3 groove): problem-specific mesh code |
| 4. MMS | scripted | sympy derives the body force for a chosen field (`finite_strain_exact`); exact Williams corner solutions are generated (`wedge_exact`) | deciding whether MMS is suitable, choosing the manufactured field, wiring it to the solver |
| 5. Simulation | partly | `generic2d` solves 2D linear elasticity end to end; cluster jobs are generated and their results imported by code (`job_factory`) | solver code for every other problem; submitting cluster jobs (kept manual on purpose) |
| 6. Convergence analysis | autonomous | rule plan, gates, QoI-class tolerances, independent cross-check, combined verdicts | — (only the rule changes themselves) |
| 7. Diagnosis | mostly | 15 hypotheses in 5 classes; cheap tests run by the agent; it reproduced the 16 causes found by hand | building the expensive tests (independent discretizations, other formulations) |
| 8. Deciding what next | partly | `followup` adds a finer level within a budget (wired for 2 problems); `generic2d` triggers the grading remedy | choosing every study of rounds K–N, every new benchmark, the spline levels |
| 9. Learning from results | minimal | `adaptive.outcome_prior` predicts verdicts from stored results | every rule change; no record of which actions worked |

Also manual: writing each pre-registration (predictions, frozen hashes) and each report section
(the numbers in it are checked automatically).

Counted in Section 16 of the report: 59 human interventions, 6 now fully automated.

## Plan (round O), in his order

- **O1 His decisions on the five questions** (labelled in the code as decisions of 24 November,
  made after the held-out results — so not blind):
  1. adopt the singular-order extension; pre-asymptotic convergence from EITHER direction
     (observed order above 2 lam, or still rising toward it from below) is INCONCLUSIVE until
     confirmed on further unseen benchmarks;
  2. absolute requirement U <= 1/3 of the constitutive half-width, with QoI-dependent limits
     (1% global and displacements, 5% region-averaged stresses);
  3. one more targeted spline refinement for the groove peak ONLY through the new experiment
     protocol (O3): the agent justifies it, predicts the outcome, then judges the hypothesis.
  4–5 (CSM/CFM coupling, geometry families / industrial models): deferred, as he asked.
- **O2 One command, paper to verdicts**: card -> specification -> generated `generic2d`
  description (LLM drafts it, code checks it against the card's quoted values) -> code
  verification (the agent decides whether MMS is suitable and generates the manufactured field
  on the same geometry) -> ladder -> convergence -> diagnosis.
- **O3 Experiment protocol**: for every next action the agent writes a hypothesis, a
  justification, a predicted outcome with a numeric criterion and a cost, freezes them (hash,
  commit) BEFORE running, runs within budget (or generates a cluster job), then scores the
  prediction. No person writes the pre-registration.
- **O4 Learning**: an action–outcome record (which action, on what kind of problem, was the
  prediction confirmed, what did it cost); the agent ranks the next actions with it, and the
  verdict prior uses it.
- **O5 Test**: the whole loop on unseen problems (pre-registered by us before the agent sees
  them); count the remaining human interventions per step and compare with the table above.
- **O6 Report**: a short PDF with only the new results.
