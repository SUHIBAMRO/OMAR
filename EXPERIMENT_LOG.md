# Experiment Log

Per Timon's note (2026-09-14): "please note the exact git commit and a
one-line description of the setup for every run you consider final,"
pending the more structured platform he is building. This file is the
interim version of that for every round-10 result currently cited in
the Report/Summary/reply draft. Going forward, every script that
produces a final number should also call `omar_pfem.run_manifest
.write_manifest(...)` (already the case for most of this project's
scripts; round-10's own gap is closed as of commit `7681812` below --
see the commits in that range for exactly which scripts gained the
call and why). A manifest entry captures far more than this file does
(exact command, full args, environment, timing) — this file is only the
one-line index into that, git-commit by git-commit, for a human
skimming what happened when.

Repo: `SUHIBAMRO/OMAR`, branch `claude/claude-code-question-d307wp`.

Extended 2026-09-14 to cover the whole project's history (round 4
through round 10), per Omar's follow-up ("يبدو إنه لكل المشروع بشكل
كامل بدو"). Built by a research pass over the full 518-commit git log
and all 8,481 lines of `PROJECT_STATUS.md` (this project's own
chronological, dated record of every real finding); every commit hash
below was independently verified to exist and its message spot-checked
against the description before merging. A handful of entries could not
be tied to a specific commit with confidence -- these are listed in
their own section at the end rather than guessed.

Extended AGAIN 2026-09-14, same day, after Omar caught that the above
pass still did not start "from the beginning of the project" as an
email to Timon now literally promises: it began at commit `bfcb67c0`
(2026-08-04), skipping the project's real first 34 commits
(2026-07-03 -- 2026-08-03), including an entire abandoned prototype
codebase and a full prior advisor-feedback round ("Round 3") that had
never been logged anywhere before this. The three sections immediately
below close that gap, read directly from `git log --reverse` back to
the repository's actual first commit (`f3d78f0`, 2026-07-03) -- every
commit hash in them re-checked with `git log -1 <hash>` against this
file's own description before being written down.

## Project origin -- VINO/FNO-based prototype, ABANDONED (2026-07-03 – 2026-07-07)

**Not behind any result in the current report or any later round below.**
The very first commits on this repo built B1/B2 hyperelasticity
benchmarks on top of vendored `eshaghi-ms/VINO` code (JAX/FNO
architecture, `Practical_Examples/omar/`) -- superseded three days
later (`a4f20e4`, 2026-07-09) by the PyTorch/PFEM/Transolver pipeline
(`Practical_Examples/omar_pfem/`) that every other entry in this log,
and the entire report, is actually about. Kept here only because Timon
asked for failures to stay visible, not just successes -- and this
prototype had real, instructive ones before it was set aside.

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `f3d78f0` | 2026-07-03 | Vendored `eshaghi-ms/VINO` code unmodified as the foundation for the first prototype. | -- |
| `0a43095` | 2026-07-03 | B1 (unit square) and B2 (quarter ring) hyperelasticity benchmarks added on top of VINO's unmodified FNO architecture/training loop; Arruda-Boyce added to `DemHyperelasticityLoss`. | -- |
| `44aec4b` | 2026-07-03 | Real FEniCSx (dolfinx) FEM data generator added; verified with 2-3 pilot samples/case (0 solver failures, BCs satisfied to float precision) before committing to full generation. | -- |
| `b4665ae` | 2026-07-03 | **Bug found+fixed**: Arruda-Boyce's raw invariant I1 has no lower bound under compression -- a full FEM generation run had all 550 samples "converge" to ~50-100% strain (~100x larger than Neo-Hookean/Mooney-Rivlin under the same load) instead of a physical response. Fixed by applying the polynomial series to the isochoric invariant I1_bar = I1/J instead (bounded below by 2, by 2D AM-GM). | -- |
| `59dd939` | 2026-07-04 | **Bug found+fixed, real negative result while it lasted**: B2's loss inherited B1's flat-domain derivative code unchanged, silently feeding d/dr, d/dtheta into the strain-energy functional as if they were d/dx, d/dy -- wrong physics for B2's curved (polar) geometry. After a full 1000-epoch Colab run: B1 test errors a reasonable 3.6-4.8%, but all three B2 materials at **153-183%** (worse than predicting zero). Training loss looked completely normal throughout -- the network was correctly converging to the WRONG energy functional. Fixed via the polar chain rule; the three pre-fix B2 runs were declared invalid and had to be redone. | -- |
| `fff41c6` | 2026-07-06 | Replaced the finite-difference/trapezoidal-quadrature (DEM) baseline with VINO's actual paper method (exact closed-form per-element energy integration) for B1; independently re-derived the existing Mooney-Rivlin formula from scratch with sympy and reproduced it byte-for-byte (0.00e+00 difference) before trusting the same method for Neo-Hookean/Arruda-Boyce. Cross-checked against the old DEM baseline on a smooth test field: all 6 material/geometry combinations agree to ~3.2%. | -- |
| `e03b6ee` | 2026-07-07 | B2's curved-geometry exact closed-form integration (replacing an interim Gauss-quadrature version), validated against direct numerical integration to ~1e-5/1e-4. **Known limitation, not a bug, kept as-is at the time**: an approximately constant ~35-40% discrepancy vs. the exact continuum energy of a smooth test field, confirmed NOT to shrink with mesh refinement (cross-validated to ~1e-7 against a brute-force reference, ruling out an implementation bug) -- an inherent property of the forward-difference bilinear-ansatz method on curved geometry. Moot within days once this whole prototype was abandoned. | -- |

## PFEM/Transolver pipeline bring-up (2026-07-09 – 2026-07-20)

**This is where the codebase behind every other entry in this log, and
the entire report, actually begins.**

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `a4f20e4` | 2026-07-09 | New, independent PFEM/Transolver pipeline started (`omar_pfem/`), per the professor's guidance to pursue PI-Transolver (Wang et al., PFEM, JMPS 2026) instead of pushing the FNO/VINO approach into complex geometries. B1 x Neo-Hookean only, real Q4 Gauss-quadrature FEM energy assembly (not autodiff, not VINO's closed-form method). Validated end-to-end on a 30-sample/15x15-mesh/200-epoch CPU run: no NaNs/crashes, physically sensible det(F), dominant-displacement test error drops from ~5.1 to 0.55-0.75 -- full GPU-scale convergence not yet attempted at this point. | -- |
| `1c04791` | 2026-07-09 | Generalized to all 3 materials x both B1/B2 (6 cases total) from scratch for B2. Fixed a real Mooney-Rivlin 2D reference-state calibration bug (`d=2(c1+c2)`, not 3D's `2(c1+2c2)`) that otherwise leaves a residual PK1 stress of `-0.5*mu` at the undeformed state. All 6 combinations validated: FEM solver converges in 2 Newton iterations/load step, zero residual stress at F=I, symmetry BCs to machine precision, genuine energy-minimization convergence (Pi: ~67 -> ~0.01 over 300 epochs, B1xMooney-Rivlin). | -- |
| `24cc9aa` | 2026-07-09 | Google Drive mounted for durable checkpoint storage (infrastructure only, not a result) -- local Colab disk doesn't survive a full runtime reset. | -- |

## Round 3 (Timon's third feedback round) -- device metrics, OOD, mesh convergence, GPU-native FEM solver, first resolution-invariance attempt (2026-07-23 – 2026-08-03)

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `6398d18` | 2026-07-23 | Per the advisor's review that 10,000 epochs wasn't necessary and the best result so far came around epoch 1000 (later epochs possibly hurt by optimization instability): replaced gradient-accumulation "batch size" (never actually reduced wall-clock time) with real mini-batching, added best-checkpoint tracking, configurable early stopping, and a systematic batch-size screening study. Verified locally (CPU, tiny data): bs=1 reproduces prior per-sample numerics exactly. | `screening_summary.json` |
| `c187f60` | 2026-07-29 | Device-level GPU memory (`torch.cuda.mem_get_info`) alongside allocator stats, standalone inference-latency benchmark, batch-size screening extended past 16 up to 256 with graceful OOM detection. | `inference_latency.json` |
| `b8dc04c` | 2026-07-29 | OOD test generation/evaluation added (shifted material/load distributions, same mesh/solver). **Bug found+fixed in the same commit**: the dataset loader crashed on `ntrain=0`, which never came up in normal training but is exactly this script's own call pattern. | -- |
| `8cce917` | 2026-07-29 | Mesh (h-refinement) convergence study for the reference FE solver, using fixed analytic fields (GRF's random-phase array isn't resolution-comparable). Confirms the reference solver is essentially converged at the N=21 mesh used throughout the study (e.g. B1/Mooney-Rivlin: 0.225% relative change from N=21->31). | -- |
| `ff46d33` | 2026-07-29 | GPU-native Newton-Raphson FEM solver (autodiff-based tangent, batched) for a fair GPU-to-GPU comparison. **Real correctness bug found+fixed during development**: naively reusing the training-side energy assembly averages material parameters per-NODE then converts, while the CPU reference solver converts per-ELEMENT-centroid then averages -- a genuinely different (nonlinear) formula, not just a different implementation. Fixed by precomputing per-element parameters with the CPU solver's own exact calls; final agreement with the CPU reference: ~1e-13 to ~1e-14 relative error. | -- |
| `b1a866f` | 2026-07-29 | First resolution-invariance study driver (Timon R3 item #7/7, completing code implementation for all 7 items of the advisor's third feedback round): 10 INDEPENDENT trainings, one per mesh resolution, compared side by side. **This design was later judged flawed by the advisor and replaced entirely -- see `589d8d0` below.** | -- |
| `50eaa37` | 2026-07-29 | Resolution-invariance driver fix: epoch budget 500->2000 (matching the main protocol) and an explicit "hit epoch cap, not converged" flag per resolution, to avoid a false "resolution X performs worse" reading of an undertrained run. | -- |
| `b4c8072` | 2026-07-29 | **Real bug found+fixed, could have inflated a headline OOD number**: `evaluate_ood.py` called the dataset loader with `ntrain=0` for the in-distribution split too, which reads `samples[0:ntest]` -- part of the TRAINING set for any real run, not the held-out test split. Would have measured train-set memorization as "in-distribution generalization," inflating both that number and the OOD degradation factor computed from it. Fixed via an explicit `--id_ntrain` flag; verified with a smoke test confirming the correct held-out indices are read. | -- |
| `4a63d45` | 2026-07-30 | **Real process bug found+fixed, live on Colab**: 3 of the advisor's 7 round-3 items (batch-size screening, inference latency, device-level GPU memory) silently never executed for the 6 main cases, because `train_hyperelastic_Q4()` returns early -- before reaching any of that new code -- whenever a case is already fully trained, which all 6 already were. The code was correct and locally verified; it simply never ran on the user's own Drive data until this fix (separate standalone cells/scripts that don't require retraining). | -- |
| `678a0f6` | 2026-08-01 | **Real bug found+fixed**: the mesh-convergence cell's skip check only tested whether an output file existed, not whether it contained every requested resolution -- an earlier interrupted run (B2 x Neo-Hookean, only N=6..26) was being skipped as "already done" on every subsequent run instead of completing to N=51. | -- |
| `6e9cb78` | 2026-08-03 | Exact, measured CPU/GPU FEM cost breakdown (assembly/solve timing, proper GPU sync, warm-up pass, analytical FLOPs estimate) per the advisor's fourth-round request, replacing a placeholder constant "8.0 s/sample" figure. | -- |
| `589d8d0` | 2026-08-04 | **Major correction, real negative finding about the project's OWN prior methodology**: the advisor correctly pointed out that training ten independent networks on ten meshes (`b1a866f`/`50eaa37`) does not demonstrate resolution invariance -- each network only ever learned its own resolution. Replaced with the actual test used ever since: train ONE Transolver model jointly on two mesh resolutions, then evaluate that SAME checkpoint with no retraining on unseen resolutions, scored against a common fine-mesh FEM reference. Required a new randomly-parametrized analytic field generator (`data/parametric_field.py`) since the GRF sampler's phase array isn't resolution-comparable. This is the direct methodological ancestor of every zero-shot resolution-invariance result reported in every later round. | -- |

## Round 4 / pre-Round-5 foundational results (2026-08-04 – 2026-08-19)

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `bfcb67c0` | 2026-08-04 | B1×Neo-Hookean Q4 convergence vs ~10M-DOF reference (N=2236, 74,871.6s solve): N=51→701 fitted rates L2 p=1.47/H1 p=0.72/energy p=0.78 -- L2 meets the 1e-4 target, H1/energy do not. | `pfem_ckpt/Q4_B1_neo_hookean_report.json` |
| `4b7813e8` | 2026-08-18 | Q4 sweep extended to N=1001/1401: L2 keeps improving (2.3e-6 @ N=1401) but H1/energy still above target; combined 7-pt fit L2 p=1.58/H1 p=0.73/energy p=0.87. | `pfem_run/Q4_B1_neo_hookean_report_extended.json` |
| `2c91f5661` | 2026-08-25 | B1×NH Q9 ~40M-DOF fine reference computed; fitted rates Q9 L2 p=1.57/H1 p=0.76/energy p=0.75 vs Q4 L2 p=1.39/H1 p=0.72/energy p=0.71. **Direct Q4-vs-Q9 agreement check FAILS the advisor's explicit <1e-5-in-all-norms criterion**: L2 2.06e-6 passes but H1 ~1.15e-3 and energy ~9.61e-4 miss by ~2 orders of magnitude. | `q4_vs_q9_B1_neo_hookean.json` |
| `58e429ae` | 2026-08-25 | CG-convergence audit: Q9's fine reference had 6/20 Newton iterations CG-fail (30%); Q4 and the small-N sweep clean. FAIL verdict stands; exact H1/energy numbers carry an unquantified margin. | (audit note only) |
| `3c83f3d8` | 2026-08-10 | **B2 accuracy root-cause #1 (original)**: B2's boundary force was a raw pressure×direction approximation, 13-16x too large vs. the FEM-consistent nodal force. | `omar_pfem/data_generate_B2.py` |
| `c8630cd9` | 2026-08-11 | Removed the spurious `/len(inner_edges)` division from B2's external-work term. | -- |
| `4780f7e7` | 2026-08-12 | Real fix: normalize training loss (not physics) by each sample's boundary-force scale (`--loss_force_norm 1`). | -- |
| `3f51a580` | 2026-08-12 | Bounded automated search for B2 accuracy fix, adopted recipe. | `pfem_run/B2_accuracy_search*/` |
| `1d121b2c` | 2026-08-15 | B2×Arruda-Boyce trial 1: 9.81% (missed self-imposed <9.00% target). | -- |
| `d677bdec` | 2026-08-15 | **Negative findings**: AB trial 2 (lr=0.005) failed at 76.9%; trial 3 (r_grading=2.5) trending ~83%+, manually stopped. | -- |
| `b3130d0c` | 2026-08-15 | B2×Mooney-Rivlin resolved at 7.28% (trial 1). | `pfem_run/B2_accuracy_search_mooney_rivlin/search_summary.json` |
| `fd35d61b` | 2026-08-16 | AB trial-1's 9.81% adopted as final. All 3 B2 materials resolved: NH 9.11%, MR 7.28%, AB 9.81% (vs B1's own 9.59%). | -- |
| `75f86c0d` | 2026-08-16 | Corrected B2 numbers (9.11/7.28/9.81%) propagated into Tables 5/7/11/14. | -- |

## Round 5 (2026-08-25 – 2026-08-28)

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `b92ee0cf` | 2026-08-26 | **Major correction**: B2×NH's real corrected training recipe costs MORE per sample (40.3s) than one native FEM solve (25.9s) -- training-cost speed-up flips from a stale 6.37x favorable to the real **0.64x**. | -- |
| `498b2d49` | 2026-08-26 | Table 7 B2×NH inference latency was reading a stale pre-fix checkpoint (4.673ms -> corrected 4.809ms); FLOPs found material-dependent (NH 5.88e7 vs MR/AB 9.72e7 assembly FLOPs/sample). | -- |
| `5811a891` | 2026-08-27 | Q9 CG caveat added to report §4.4. | -- |
| `0d22a5fa` | 2026-08-27 | Physical-quantities eval (H1, energy norm, PK1 stress, reaction forces) built for all 6 cases -- headline means (50 held-out samples, %): displacement B1 10.34-11.71/B2 7.21-10.47; H1 22.71-24.20/10.47-13.30; **peak‖P‖ 19.87-47.53 (B1, worst!) / 5.38-5.99 (B2)**. Finding: H1/energy/stress exceed displacement error in ALL SIX cases. | `omar_pfem/point5_results/` |
| `9346568d` | 2026-08-27 | **Major correction**: operator-vs-GPU-FEM speed-up was wrong by ~16x from an unmatched-batch-size comparison. Matched (bs=128 both): speed-up 73-80x -> **1,215-1,297x**. Deployment-realistic (bs=1) break-even: **1,133-19,410**, not the ~96,000 previously reported. | `omar_pfem/point3_inputs.json` |
| `da89fb47` | 2026-08-27 | **Real bug found+fixed**: GPU/matrix-free B2 solvers sampled material at element centroid while the CPU reference sampled per Gauss point. B2 FAIL (4.8e-5 abs) -> PASS (2.7-4.7e-16). | -- |
| `5d648d9` | 2026-08-27 | GPU-native FEM scaling sweep: 8 resolutions 0.02M -> 3.93M DOF. 3,925,602 DOF in 11.0h using 3,280MB/80GB (~4%). Memory model predicted 3,201MB, measured 3,280MB (2.4% out-of-sample). | Table 20 |
| `efe27e4f` | 2026-08-27 | OOD diagnosis: degradation entirely attributable to the material-stiffness shift, not the loading-magnitude shift. | `point6_results/` |
| `b1b74224` | 2026-08-28 | Point 2 Pareto, B1×NH: FEM@N=13 (0.608%) already 6.1x better than operator's best (3.69%@N=37); speed-up 1,630x-17,895x. | `point2_results/pareto_B1_neo_hookean.json` |
| `6ae8b99a` | 2026-09-01 | Point 7b (physics-informed vs data-driven) 2x2 complete: under a matched optimizer (AdamW+OneCycleLR), the physics-informed loss wins. | `point7b_results/` |
| `c21f38fd` | 2026-08-28 | MMS (Point 9), FEM half: Q4/Q9 vs analytic manufactured solution -- all rates hit theoretical value. | `omar_pfem/point9_results/` |
| `c3192fda` | 2026-08-28 | MMS operator (single member, N=17): operator/Q4 = 2.42x L2 but only 1.03x H1/stress -- **inverted norm ordering**. | `mms_operator_B1_neo_hookean.json` |
| `68572f30` | 2026-08-28 | OOD mitigation tested: normalization improves the degradation ratio at one cell but **costs 6.7% in-distribution and worsens absolute error in 13/18 cells** -- tested-and-did-not-work. | `point6_results/ood_mitigation_B1_neo_hookean.json` |

## Round 6 -- B2 root-cause saga, Pareto sweeps, MMS extensions, solver speedups (2026-08-29 – 2026-09-04)

**B2 zero-shot debugging chain** (sequential, real, verified negative findings before the true root cause):

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `641e3413` | 2026-08-27 | Original 3 B2 zero-shot cases found **INVALID**: relative errors 800-1450% (worse than predicting zero). Root cause: assembled load overstated by a mesh-dependent factor (13.1-13.3x @N=21, 20.8-21.0x @N=33). | `point7a_results/INVALID_B2_zeroshot.json` |
| `a45496b0` | 2026-08-28 | Load repaired for B2×MR/AB (verified consistent to 0.0075% across meshes). | -- |
| — | 2026-08-31 | Input normalization tested as B2 fix: **FALSIFIED** (0.9910 vs 0.9986 baseline, 0.8% move only). | -- |
| `c7f63c6` | 2026-08-31 | First diagnostic probe: model amplitude 2.5-4x too small (wrong shape, not size); U(pred) nearly flat regardless of input ("collapse"). | -- |
| `25557d3` | 2026-08-31 | Joint training **ruled out**: single-resolution retrain fails identically. | -- |
| `83469fe` | 2026-08-31 | **ROOT CAUSE CONFIRMED**: `per_component` validation metric ranks B2 checkpoints BACKWARDS -- rises while true error falls. Mechanism: per-sample rms(v)/rms(u) averages 1.90 while ratio-of-averages is 0.90 (skewed distribution). | -- |
| `dd077a50` | 2026-09-01 | **Fix propagated -- B2 works**: retrained with `--selection_metric both_components`: val error 0.9986 -> **0.0330**@epoch2750; zero-shot 0.87 -> 0.071-0.269, spread 3.8x vs B1's 2.1x. | `point7a_results/B2_zeroshot_fixedselection.json` |
| `ab72f8e8` | 2026-09-02 | B2×Mooney-Rivlin fixed-selection: MR wins at 3 coarse meshes, loses at 3 fine; MR spread 4.91x vs NH 3.78x. | `point7a_results/B2_mooney_rivlin_zeroshot_fixedselection.json` |
| `58ad302b` | 2026-09-02 | B2×Arruda-Boyce fixed-selection (completes Point 7, all 6 cases): worst of 3 B2 materials at 6/7 meshes but narrowest spread (3.62x). | `point7a_results/B2_arruda_boyce_zeroshot_fixedselection.json` |

**Pareto sweeps (Point 2), MMS family (Point 9), solver-speedup experiments:**

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `681424245` | 2026-09-01 | MMS family (16 members), Q4/Q9/operator: operator/Q4 @N=9/17/33 = 0.62x/2.59x/14.49x. **Negative finding, family-confirmed**: fitted rate for the operator is -0.28 (error GROWS with refinement) while Q4 falls 15.8x. | `point9_results/mms_family_fem_B1_neo_hookean.json` |
| `31708e14` | 2026-09-01 | MMS operator per-member reliability: std/mean L2 0.384-0.425 vs Q4's 0.000-0.007 -- operator far less reliable member-to-member. | `point9_results/mms_operator_per_member_B1_neo_hookean.json` |
| `2d93c9cb` | 2026-09-02 | B1×Arruda-Boyce Pareto complete: speed-up 3,452x-54,731x. | `point2_results/pareto_B1_arruda_boyce.json` |
| `67f46afe` | 2026-09-02 | B2×Neo-Hookean Pareto: speed-up 1,738x-27,392x. **Finding**: sharp local error minima exactly at N=21/33 (the joint-training resolutions), 4.30x/3.65x below neighbours -- not seen in B1's single-resolution-trained Pareto. | `point2_results/pareto_B2_neo_hookean.json` |
| `a254040e` | 2026-09-03 | B2×Arruda-Boyce Pareto (completes Point 2, all 6 cases): anchoring effect replicates a THIRD time. | `point2_results/pareto_B2_arruda_boyce.json` |
| `a999dbf3` | 2026-09-03 | Full numeric audit of all 48 report tables: one confirmed arithmetic error found (Table 24c, 13.32x->13.33x) -- everything else matched source JSON exactly. | -- |
| `d30b320f` | 2026-09-11 | **Point 8 real A100 result**: assembled+direct solver, N=401-1401, 1.94-2.29x faster than torch-fem, 1.08-1.11x faster than TensorMesh, ~4.5x less memory. | `assembled_direct_convergence_production_N401_1401.json` |
| `3c633579` | 2026-09-11 | **Point 9 real A100 result**: baseline/+reuse/+reuse+symmetric wall-clock @N=1401: 58.54/24.61/23.68s. Best config: 5.65x faster than torch-fem, 2.66x faster than TensorMesh, 4.13x less memory than torch-fem. | -- |
| `e79e780f` | 2026-09-11 | **Negative/null finding, reported honestly**: coalescing-reuse (3rd cuDSS optimization) verified correct but speed essentially IDENTICAL to pre-optimization (24.46s vs 24.61s) -- no material benefit, not reported as a "Point 10." | -- |

## Round 7/8 (2026-09-06 – 2026-09-09)

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `d43ce753` | 2026-09-07 | Multigrid preconditioner real GPU confirmation, N=401: cg_failures 20->0. | -- |
| `e3747807` | 2026-09-07 | **Real bug found**: peak-stress QoI "reference" was silently redefined at every mesh resolution instead of being a fixed target. | -- |
| `8465c8d4` | 2026-09-08 | Multigrid final: N=701/1001/1401 all reach full CG convergence; peak_stress_rel_err now decreases smoothly across all 7 resolutions (62.6%->7.7%). | `high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json` |
| `4178aba9` | 2026-09-08 | torch-fem vs "ours" wall-clock @N=401-1401 (OLD unmatched precision): torch-fem faster by 418x/655x/1197x/936x. **Superseded by round-9's matched-precision 204-306x result**, but was a real reported number at the time. | -- |
| `c41aba4b` | 2026-09-09 | MMS richer sine/cosine family extended to Arruda-Boyce (completes the richer-family x all-3-materials matrix). | `point9_results/mms_richer_B1_arruda_boyce.json` |

## Round 9 (2026-09-09 – 2026-09-10) -- matched-precision torch-fem study

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `6c042784` | 2026-09-10 | torch-fem fixed to run at matched FP64/1e-8 (root cause: `near_null_space()` hardcoded float32 `eye(3)`). | -- |
| `62145304` | 2026-09-10 | **Headline round-9 finding**: torch-fem vs "ours" wall-clock @N=401/701/1001/1401: 9.82/25.15/56.65/133.83s vs 2615.8/7205.4/17314.8/27257.4s -> **204x-306x speedup at MATCHED precision** (down from the old unmatched 418-1197x). Accuracy identical to every printed digit at all 7 resolutions. | `torchfem_convergence_vs_fine_reference_full.json` |
| `6f3fe476` | 2026-09-10 | torch-fem direct(LU) solve is 14x/33x SLOWER than CG+Jacobi @N=401/701 -- getting worse with N, opposite of the suggested fix. | `torchfem_timing_breakdown.json` |
| `9623de6f` | 2026-09-10 | NO inference @N=1401 = 2,286.69ms/sample -- ~500x slower than the operator's own Table 7 number, fitted scaling ~n^0.74 (sublinear, NOT flat). Still 59x faster than torch-fem's matched N=1401 solve, but accuracy at N=1401 was not yet validated at this point (precursor to round-10's later accuracy work). | -- |
| `f2336850` | 2026-09-10 | Batch-size study (Table 6) **removed** per Omar's decision (Timon found it confusing) -- an editorial decision, not a new measurement. | -- |
| `d3b6e9b8` | 2026-09-10 | TensorMesh N=51 dense-Jacobian ceiling broken via explicit sparse Jacobian: N=51 105.27s -> 0.42-0.49s (~215-250x faster). | -- |
| `19b5369c` | 2026-09-10 | TensorMesh production comparison closed, all 4 N: matches torch-fem to every printed digit; 1.77-2.13x faster at every N. | `tensormesh_convergence_production_N401_1401.json` |

## Round-10 final results

| Commit | Date | One-line description | Result file(s) |
|---|---|---|---|
| `4dd4484` | 2026-09-12 | **Bug, not a result**: fixed a checkpoint-resolution defect that had every round-10 NO accuracy script silently evaluating the wrong (data-driven) model instead of the physics-informed one. | `resolve_b1_checkpoint.py` |
| `c2586c1` | 2026-09-13 | NO accuracy at N=1401 vs. real FEM ground truth, original (N=21,33) checkpoint, corrected: 44.65% disp_rel_L2, degrading monotonically away from training resolutions. | `no_accuracy_degradation_sweep_B1_neo_hookean.json` |
| `cd043e4` | 2026-09-13 | Corrected peak-stress metric (fixed physical location from a fine reference, not the coarse mesh's own sample-max) and the resulting coarsest-suitable-FEM crossover, original checkpoint. | `no_peak_stress_fixed_location_B1_neo_hookean.json` |
| `01e50e4` | 2026-09-14 | Multi-resolution retrained checkpoint (N=21,33,101,201): N=1401 disp_rel_L2 drops from 44.65% to 5.85% (7.6x) vs. real FEM ground truth. | `no_accuracy_multires_retrain_comparison.json`, `no_accuracy_degradation_sweep_multires.json` |
| `086db1d` | 2026-09-14 | **Finding, not just an improvement**: the retrained checkpoint's fixed-location peak-stress accuracy now BEATS torch-fem's own best low-N (N<=49) peak-stress accuracy for every NO resolution N>=29 -- flips which QoI binds the coarsest-suitable-FEM crossover. | `no_peak_stress_fixed_location_multires.json` |
| `4467d34` | 2026-09-14 | Confirmed real final training wall-clock for the multi-res checkpoint: 41,881.28s (~11.6h), early-stopped exactly as configured (8 checks, no improvement). | `metrics_history_multires.json` |
| `ae7ee83` | 2026-09-14 | Batch-size/throughput and torch.compile/TF32 re-measured with the corrected (but still original N=21,33) checkpoint -- confirmed unchanged from the wrong-checkpoint-era numbers. | (Drive-side JSONs, see PROJECT_STATUS.md 2026-09-14 entries) |
| `732cf47` | 2026-09-14 | Batch-size/throughput re-verified specifically against the multi-res checkpoint: max-bs throughput 4,182.27 vs. 4,203.79 samples/s (0.5% difference) -- checkpoint-independent. | `max_feasible_batch_multires.json` |
| `9ba82d8` | 2026-09-14 | torch.compile/TF32 re-verified specifically against the multi-res checkpoint: all four variants within 0.01-0.8% of the original numbers -- checkpoint-independent. | `no_inference_torch_compile_N1401_multires.json` |
| `2feef0d` | 2026-09-14 | **Negative result, reported as such**: accuracy-matched break-even, NO@N=1401 vs. FEM@N=11 (its own coarsest-suitable mesh). In default eager fp32 mode the NO NEVER breaks even -- the accuracy-matched FEM is already cheaper per sample. Only with torch.compile+TF32 does it break even, after 34,005 samples (~3.72 GPU-hours) against the 11.6 GPU-hour training cost. | `break_even_accuracy_matched_N1401.json` |
| `7681812`..`10f7e78` | 2026-09-14 | Housekeeping: added `write_manifest()` calls to the round-10 scripts that lacked them (`gpu_fem_benchmark.py`, `no_accuracy_at_n1401.py`'s two sweep functions, both torch.compile cell scripts, the break-even cell script) so every future run of these captures git commit + full setup automatically, per Timon's own request. Also folded all of the above into the canonical Report/Summary deliverables and fixed one process gap (a claim written into the Report before its own verification had actually run -- caught and corrected with the real numbers once they existed). | n/a (code + document changes) |
| `d0ae5a78` | 2026-09-12 | B7 (ring+notch) feasibility check, 3 resolutions (72/288/1,152 elements): peak PK1 stress at the notch 11.70->12.72->13.45 (still rising, NOT converged) while max displacement is essentially flat (<1% change) -- the "smooth global field, under-resolved local quantity" signature Timon described. | `omar_pfem/b7_notch_stress_concentration_check.json` |
| `ce772938` | 2026-09-12 | 4th resolution (4,608 elements) confirms the trend: peak stress 13.9372 (+3.7%), displacement flat (+0.27%); stress/displacement change ratio widens 3.7x->6.7x->13.7x -- genuine local concentration, not generic slow convergence. Real evidence justifying the B7 case, but **training itself deferred pending Timon's confirmation of the design** -- not a final trained result. | same file |

## Known negative/failure results worth keeping visible

These are genuine, real findings that came out unfavorably (not
development bugs) -- kept here explicitly per Timon's own request not
to bury failures:

- **Abandoned VINO/FNO prototype (`omar/`, 2026-07-03 -- 07-07, not
  behind any current result)**: Arruda-Boyce's raw invariant produced
  unbounded, unphysical ~50-100% strain under compression for all 550
  samples of a full data-generation run before the fix (`b4665ae`); B2's
  loss silently fed polar-coordinate derivatives into the strain-energy
  functional as if they were Cartesian, producing 153-183% test error
  (worse than predicting zero) after a full 1000-epoch run, with
  completely normal-looking training loss throughout (`59dd939`); the
  curved-geometry closed-form energy integration showed a constant
  ~35-40% discrepancy vs. the true continuum energy that did NOT shrink
  with mesh refinement, an inherent limitation of the method rather than
  a bug (`e03b6ee`) -- moot days later when this whole codebase was set
  aside for the PFEM/Transolver pipeline actually used ever since.
- **The project's own first resolution-invariance study design was
  methodologically wrong and had to be replaced entirely**: the original
  version (`b1a866f`, `50eaa37`) trained 10 independent networks, one per
  mesh resolution, and compared them side by side -- which the advisor
  correctly pointed out does not demonstrate resolution invariance, since
  each network only ever learned its own resolution. Replaced (`589d8d0`)
  with the true zero-shot protocol used in every later round: one model
  trained jointly on two resolutions, evaluated with no retraining on
  unseen ones.
- **OOD in-distribution evaluation bug that could have inflated a
  headline number**: `evaluate_ood.py` read part of the TRAINING set as
  the "in-distribution test set" (`ntrain=0` on a loader whose convention
  needs the real ntrain to find the held-out slice), which would have
  measured train-set memorization as generalization and inflated the OOD
  degradation factor computed from it. Caught and fixed before any
  numbers were reported from it (`b4c8072`).
- **3 of the advisor's 7 round-3 feedback items silently never executed
  on real data for months**, despite being correctly written and locally
  verified: batch-size screening, inference latency, and device-level GPU
  memory all lived inside a training function that returns early whenever
  a case is already fully trained -- true for all 6 main cases by the
  time this code was added. Found only because the user was actively
  running the notebook (`4a63d45`).
- **Q4-vs-Q9 direct agreement check FAILS** the advisor's own
  <1e-5-in-all-norms criterion in H1/energy (misses by ~2 orders of
  magnitude), despite L2 passing (`2c91f5661`, `58e429ae`).
- **B2 accuracy catastrophically broken THREE separate times** across
  the project, each a distinct real bug: (1) raw-pressure boundary force
  13-16x too large, and the naive fix alone made it worse (32.46% ->
  94.08%) before the loss-normalization fix (`3c83f3d8`, `4780f7e7`);
  (2) zero-shot sample-cache load overstated by a mesh-dependent factor
  (13-21x), invalidating all 3 original B2 zero-shot cases, errors
  800-1450% (`641e3413`); (3) the `per_component` validation metric
  ranked checkpoints BACKWARDS, causing every B2 run to early-stop at
  its first validation event for months (`83469fe`).
- **B2xArruda-Boyce accuracy-search trials 2 and 3 failed outright**
  (76.9% and ~83%+, both abandoned) (`d677bdec`).
- **Operator-vs-GPU-FEM speed-up reported to Timon was wrong by ~16x**
  (unmatched batch sizes) before correction (`9346568d`).
- **B2xNeo-Hookean's corrected training recipe costs MORE per sample
  than one native FEM solve** (0.64x "speed-up," i.e. a slowdown) --
  a genuinely unfavorable training-cost finding (`b92ee0cf`).
- **MMS operator does not converge under mesh refinement** -- its L2
  error gets WORSE with refinement (fitted rate -0.28 to -0.59), both
  on a single member and confirmed on a 16-member family, while Q4/Q9
  converge at their theoretical rates (`c3192fda`, `681424245`).
- **OOD input-normalization mitigation tested and found NOT to work**:
  helped only 5/18 cells, hurt 13/18, cost 6.7% in-distribution
  accuracy (`68572f30`).
- **torch-fem's direct (LU) solver is 14-33x SLOWER than CG+Jacobi**,
  getting worse with N -- opposite of the advisor's suggested fix
  (`6f3fe476`).
- **cuDSS coalescing-reuse optimization: correct but produces no
  material speedup** (null result, deliberately not reported as a
  "Point 10") (`e79e780f`).
- **Cached-Hessian speedup did not close the GPU-solver performance
  gap** at production scale (qualitative negative result; exact commit
  unconfirmed, see "Unresolved" below).
- **NO's default (eager fp32) inference never economically justifies
  its own training cost** against a FEM mesh that is only as accurate
  as the NO itself (see `2feef0d` above). Only holds with the
  torch.compile+TF32 optimization applied.
- **Peak PK1 stress remains a slow-converging QoI for BOTH methods**
  (NO and native FEM) at every point this project has measured it --
  round-6/7's B1 mesh-convergence version (`e3747807`'s fixed-reference
  fix) and round-10's own version (`cd043e4`, `086db1d`) -- likely
  because the reference point sits at a domain-corner
  boundary-condition transition prone to a stress singularity. Neither
  method's peak-stress error should be read as a clean, well-posed
  target without this caveat.
- **B7 (ring+notch) case not yet trained**: feasibility confirmed real
  (`d0ae5a78`, `ce772938`), but training is on hold pending Timon's
  explicit sign-off on the design -- an open item, not a completed
  result, but worth surfacing since it is the one round-10 point
  without a final answer.

## Unresolved -- entries this pass could not confidently tie to one commit

- **Cached-Hessian production-scale negative result** (~2026-09-10/11):
  `PROJECT_STATUS.md` states a real GPU run at N=401-1401 "did NOT reach
  the goal" of closing the 204-306x gap vs. torch-fem, with no numeric
  table and no commit cited in the text. Only the CPU-verification
  commit (`419c1285`) and the notebook-build commit (`c491e075`) were
  found nearby. If this run's own JSON exists on Drive, it should be
  pulled in and given its own row.
- ~~B1xNeo-Hookean original Pareto "17,895x" vs. "25,676x"~~ **RESOLVED,
  not actually ambiguous**: checked the live Report table directly
  (`PFEM_Transolver_Report_2026-09-14.docx`, table index 40) --
  17,895x is N=41's own speed-up and 25,676x is N=49's, two different
  rows of the same Pareto sweep, not two conflicting numbers for one
  case. No fix needed.
- **Very early Round-4/pre-tracker results** (initial Tables 1-6 mesh
  convergence, Table 8 GPU memory, original Table 4a/7, the original
  pre-"revised" Table 12): the CODE behind these is now traced above
  (mesh convergence `8cce917`/`678a0f6`, device GPU memory `c187f60`,
  inference latency `4a63d45`) -- but the specific live-GPU-run commit
  that produced the exact numbers actually printed in these report
  tables predates `PROJECT_STATUS.md` itself (2026-08-15) and this
  project's one-commit-per-finding discipline, and was not confidently
  identified in this pass -- omitted rather than guessed at a commit.
- ~~OOD degradation factors 4.75x/5.47x/2.27x (Table 25)~~ **RESOLVED**:
  checked the live Report table directly (table index 36, "Case |
  Baseline | Loading @ k=3 | x | Material @ k=3 | x | Both @ k=3 | x").
  All three are the "Material @ k=3" column, one row each: B2 x
  Neo-Hookean = 4.75x, B2 x Mooney-Rivlin = 5.47x, B2 x Arruda-Boyce =
  2.27x. No fix needed.
