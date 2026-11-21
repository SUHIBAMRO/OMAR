# Verification agent

Deterministic verification of finite-element reference solutions, with an
optional LLM layer around it. It was built for the PFEM/Transolver
hyperelasticity project, and its design follows the GRAFT direction that
Prof. Rabczuk suggested: a problem tree, a method tree, rules, a
solved-case repository and LLM roles.

**Code decides every verdict.** LLMs may formalize a problem, propose
checks from a fixed catalogue, critique a plan and explain results. Code
validates every plan, and code audits every advisor statement against the
gates.

Run all commands from `Practical_Examples/`.

```
python -m pytest omar_pfem/verification_agent/test_llm_guards.py -q   # no LLM, no solver
python -m omar_pfem.verification_agent.agent rules                    # rule plan on B1 + B3, no LLM
python -m omar_pfem.verification_agent.agent b1-ladder --Ns 11,21,41,81,161
python -m omar_pfem.verification_agent.agent adaptive                 # B1, budget-aware
python -m omar_pfem.verification_agent.agent orchestrate --backend replay \
    --replay omar_pfem/verification_agent/fixtures/replay_b1_example.json --problem "B1 test"
python -m omar_pfem.verification_agent.region_study                   # B3 region-definition study
```

Live LLM backends:

- `--backend anthropic` needs an API key.
- `--backend claude-cli` uses headless `claude -p`.

**Both bill the account they run under.**

## Gates (`gates.py`)

| gate | decides | evidence |
|---|---|---|
| `qoi_convergence_gate` | finest three levels: Celik 2008 order and GCI. The order must lie in the element's window `[0.5, 2k+1]`. Fallback: stability band. A divergent sequence FAILs. | `rate` / `band` / `none` |
| `multilevel_convergence_gate` | least-squares fit `v = a + b h^p` over the finest 5 levels (needs at least 4). Rate evidence needs all of: monotone sequence, fit sigma ≤ 10% of the data range, `p` in the window and off the search bound, and leave-one-out refits that keep `p` in the window (only checked with ≥ 5 levels). Uncertainty: `U = 1.25·\|v_fine − a\| + sigma + \|finest residual\|`. | `rate` / `band` / `none`, plus `failed_criteria` |
| `strategy_robustness_gate` | spread across refinement strategies at comparable resolution | |
| `anchored_robustness_gate` | every strategy against the family's limit: the 3-level Richardson value, else the multilevel fit limit, else the finest value. INCONCLUSIVE when neither convergence gate passes: a family that did not converge has no limit. | |
| `mms_rate_gate` | observed vs theoretical rates against a manufactured solution | |
| `combine_verdicts` | one policy, shared by the orchestrator and the adaptive controller. Any FAIL fails the QoI. Evidence is the strongest among the convergence gates that passed, except that a 3-level rate is downgraded to band when the 5-level gate found the finest five levels non-monotone (evidence against an asymptotic regime). | |

**`band` is weaker than `rate`.** A band PASS means the QoI is stable
within the band. It does not show a convergence rate or monotonicity.

## Rules and tolerances (`trees.py`, `requirements.py`)

- **QoI classes:** global, displacement, region_average, pointwise_peak,
  error_norm.
- **Tolerances are set by code only** (`trees.check_params`). An LLM
  plan's parameters are replaced, and the replacements are recorded in
  the trace as `tolerances_overridden`. Each tolerance is the class
  ceiling, tightened to the requirement from the surrogate errors the
  reference has to resolve.
  - Requirement: `best recorded surrogate error / 3` (`requirements.py`;
    every number has its source).
  - A QoI with no recorded surrogate error keeps its class ceiling, and
    the requirement says so.
- **Class ceilings (upper bounds):**
  - global and displacement: band 0.2%, GCI 1%, robustness 1%;
  - all other classes: 1%, 2%, 2%.
- **Rules R1–R6** say which checks are required and which verdicts are
  predicted. The two convergence gates are required for every
  solution-verification QoI. Robustness checks are required for peaks and
  region averages near singular or near-singular features.
- **Rule R7** depends on the plan: any QoI checked for strategy
  agreement must also be compared against the family's limit, because
  strategies that agree can share one error.
- **The rules are authoritative.** If an LLM plan omits a required check,
  code adds it and records that. If a tolerance is looser than the class
  ceiling, code clamps it and records that.

## Data adapters (`orchestrate.py`)

- **B1** (`unit_square`):
  - Q4, uniform levels from the cache `b1_ladder_neo_hookean.json`;
  - corner-graded strategies a = 0.5 and 0.8 at N = 81;
  - solver: `b1_ladder.py`;
  - `interior_gk_PK1_frob`: the interior |P| average with a Gaussian weight
    (length 0.1), from the Gauss-point dumps in `b1_gauss_interior/`
    (`python -m omar_pfem.verification_agent.b1_ladder` writes them).
- **B3** (`bushing_groove`):
  - Q20, an 8-level family from 2,376 to 59,400 elements (the 8th level
    from `b3_global.py fine`; 100,815 elements does not fit in 15 GB);
  - six strategies at 10,200 elements: axial (z) grading z_beta = 1, 2, 3 and
    radial grading r_grading = 1.1, 1.2, 1.3 (`b3_global.py radial`, direct
    solver). r_grading >= 1.75 inverts elements at the groove and 1.5 is
    near-degenerate, so stronger radial grading is not used;
  - data in `b3_extended.json`, written by `b3_global.py`, which reproduces
    the stored solves.
  - Region averages use the stored sharp sphere of radius 2 × the groove
    curvature radius. `region_study.py` and `b3_gauss_near_groove/` compare
    other region definitions on the same solutions.
  - `gk2_*` QoIs average the same quantities with the smooth weight
    `exp(-(d/2R0)^2)`, built from the same dumps. This is a
    mesh-independent alternative to the sphere; it is **not** the
    manuscript's definition.
  - The Q20 elements use torchfem's 2×2×2 (reduced) rule.
    `b3_global.py fullint` re-solves two levels with the full 27-point
    rule and saves them to `b3_fullint.json`.
  - `surface_study.py`: stress ON the groove surface. The Gauss stresses
    (from the dumps) are extrapolated to all 20 nodes, so a node sits at
    the groove root on every level; 3- and 5-level gates on the root value
    and on the surface maximum within R0.

- **B2** (`quarter_ring`, HELD-OUT): quarter ring under internal pressure, Q4,
  fixed smooth fields (`mesh_convergence.AnalyticFieldB2`); levels N = 7-193
  (`b2_ladder.py`, query points are nodes on every level); radial-grading
  strategies at N = 49. Spec (smooth, Q4) fixed before any B2 verdict; rules,
  gates and tolerances applied unchanged.

## Further checks and tools

- `mms_q20.py`: manufactured-solution code verification of the Q20 element
  (3D, neo-Hookean, reduced and full rule, nu = 0.3 and 0.45).
- `verify_report.py`: recomputes every number of the technical report from
  the stored data and compares at the printed precision.
- `ablation.py`: LLM pipeline vs rule-only baseline over repeated runs
  (billed with a live backend; `--backend replay` tests the code for free).
- `diagnose.py`: diagnosis of every QoI without a rate. Five cause classes
  (mathematical/physical, numerical discretization, solver/implementation,
  QoI/post-processing, surrogate); 15 hypotheses (M1, M2, P1, D1-D4,
  S1-S3, Q1-Q3, U1), each with a test on stored data that returns
  supported / refuted / needs-data / not-applicable. A needs-data result
  names the next investigation and its cost; `--investigate` runs the cheap
  ones itself (registry `ACTIONS`: B2 Newton tolerance 1e-10, B3 direct
  linear solver; Gaussian averages of B3 shear components are computed on
  the fly from the Gauss-point dumps) and re-diagnoses. `consistency()`
  compares with the 16 causes established by hand (currently 16/16).
  Output: `diagnose_result.json`.

## Held-out problems, next agents and further tools (added 2026-11-20)

- `cook_*.py`, `plate_hole*.py`, `lamina*.py`, `lshape*.py`: held-out problems, each
  with a `*_preregister.py` that writes its predictions before any solve and scores
  them afterwards (`--compare`).
- `mixed_b3.py`: Q20 with F-bar (P0) and projected-dilatation (P1) formulations;
  `VERIF_LINSOLVE=amg` switches the linear solver to AMG-preconditioned CG (pyamg),
  which fits the 23,520-element level in memory.
- `literature_agent.py`: paper PDF -> quote-checked benchmark card (`card`, `eval`,
  `eval-unseen`; billed with a live backend).
- `active_learning.py`: verification-aware multi-fidelity active learning on the lamina.
- `goal_oriented.py`: one-mesh adjoint (enriched-dual) error estimate on the lamina.
- `iga_lamina.py`: independent spline (IGA-type, exact geometry) discretization of the
  lamina against the FEM ladder.
- Post hoc rule options, off by default: `VERIF_POSTHOC_SINGULAR=1` (singular-order
  check for degree-k elements, pre-asymptotic label), `VERIF_ABS_TARGETS=proposal`.

## Files

| file | role |
|---|---|
| `spec.py` | ProblemSpec and fingerprint |
| `repository.py`, `repository.jsonl` | append-only solved-case store. The latest record per problem name supersedes earlier ones. |
| `llm_roles.py` | check catalogue, schemas, backends, plan/spec validation, advisor audit |
| `orchestrate.py` | adapters, check execution, the full LLM pipeline |
| `adaptive.py` | similarity-weighted outcome prior; refinement controller |
| `traces/` | live LLM run traces |
| `fixtures/` | hand-written replay answers for tests (not model output) |
| `problems/` | problem texts |

## Known limits

- The advisor audit checks verdict labels only, not every sentence.
- Active learning exists for the lamina only (`active_learning.py`).
- The B3 region averages that get a rate (Gaussian kernel, `gk2_*`) are in
  the B3 adapter next to the sphere averages, but the manuscript still uses
  the sphere: switching would redefine the B3 validation target, which is
  Omar's and Prof. Rabczuk's decision.
- Mixed formulations (F-bar, P1) are in `mixed_b3.py`, outside torchfem; on strongly
  graded root meshes no formulation gives a converged groove peak.
- Two versions of the Gaussian groove average exist: the B3 adapter (Section 8 of the
  report) uses the Gauss points within 0.25 of the root (the stored dumps),
  `mixed_b3.py` all Gauss points; they differ by about 0.35% on the same mesh.

`PROJECT_STATUS.md` at the repository root holds the results and the
history.
