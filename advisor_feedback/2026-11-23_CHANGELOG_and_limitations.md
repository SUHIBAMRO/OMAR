# Round of 23 November 2026: change log, reproducible results, remaining limitations

Report: `advisor_feedback/2026-11-23_Verification_Framework_Technical_Report_v5.pdf` (68 pages; the
new material is Section 30, with pointers in Sections 1, 25 and 27). Everything below is committed on
branch `claude/claude-code-question-d307wp`. Post hoc changes are marked **[post hoc]**. Nothing was
sent to anyone, and no job was submitted to the cluster.

## 1. Changes, in the order of the five priorities

### P1. Audit of report and manuscript
- `manuscript_claims.py` (new): recomputes the 28 numbers of the manuscript's groove-stress passage from
  the stress-corrected data, and checks that each printed number is still present in `paper/main.tex`.
  Result: 28/28. Three surrogate-evaluator numbers are listed as not recomputed (unaffected by the fix).
- `paper/main.tex`: the Gaussian uncertainty is now stated on both bases. It is 0.11-0.18% of the regional
  stress magnitude, which is 0.8-1.0% of the averages themselves, and "about twice that uncertainty"
  is added next to the ~2% spline gap. The manuscript compiles (44 pages). No other manuscript change:
  every corrected Cauchy-stress number already matched the data.
- Report corrections:
  - Section 28: "82 quantities" changed to 89 (68 verifiable).
  - Section 28: the edge crack added to the held-out list.
  - Section 27: the claim that groove stresses are formulation-independent on uniform meshes is
    corrected (the spline gap contradicts it).
  - Section 27: the cross-check rule added to the post hoc list.
  - Section 25: the explanation given for the lamina contradiction is withdrawn (refuted, see P2).

### P2. Independent cross-checks
- New independent values (`crosscheck_extra.py`):
  - B1 Gaussian interior average: second FE code on Q2, N = 11-161. The stored Q2 values are
    reproduced exactly.
  - Cook: u_tip, v_tip, max|u| and the Gaussian average, on Q2, N = 4-128. This is the same code with a
    different element family, so it is an independent discretization, not an independent code.
  - The 3D benchmark is compared with its exact solution.
- Cross-check rule v2 **[post hoc, stricter]** (`crosscheck.py`): an independent sequence counts only if it
  passes the agent's own multilevel check (a rate, and its uncertainty within the same tolerance).
  Otherwise the quantity is INCONCLUSIVE (`independent_unconverged`). v1 is kept and reproduced.
  | | rate-certified | cross-checked | contradicted | independent not converged | none |
  |---|---|---|---|---|---|
  | v1 (22 Nov) | 34 | 21 | 5 | - | 8 |
  | v2, same 34 | 34 | 23 | 0 | 9 | 2 |
  | v2 + 3D benchmark | 38 | 27 | 0 | 9 | 2 |
- The five v1 contradictions:
  - B3 sigma_xx and sigma_zz Gaussian averages: unresolved. Both spline sequences are unconverged
    (three-level orders 0.47 and 0.75).
  - B3 sigma_yy and pressure: agree with the displacement splines. The independent uncertainty is
    1.5% and 3.0% of the values, so this is a weak confirmation.
  - Lamina sharp hole average: unresolved. The spline sequence's uncertainty is 9.5%.
    `lamina_zone_study.py` refuted my zone-definition explanation: the zone-area error at the
    compared levels is <= 4e-4, about 1e-5 in the average, against a 0.09% gap.
- `iga_b3.py` now also outputs the cylindrical rr/tt averages. `Practical_Examples/iga_b3_cyl_job.sh`
  (~8 h) is prepared for **manual** cluster submission, so the two quantities without an independent
  value can be checked.

### P3. Unseen 3D benchmark
- `cylinder3d.py` (new): a thick-walled cylinder in a quarter model, Q20 reduced (the B3 element),
  neo-Hookean with nu = 0.45, curved iso-parametric boundaries. The exact solution comes from a 1D
  boundary-value problem, checked against Lame theory at small strain (9e-6).
- `cylinder3d_preregister.py` / `cylinder3d_preregistration.json`: the specification, rule predictions,
  expectations, scoring tests and SHA-256 of the verdict files were committed **before** any loaded
  solve (commit 538f0fb).
- Results (`cylinder3d_result.json`):
  - Rules 3/3.
  - True error inside the certified uncertainty for 4/4 rate-certified quantities.
  - Expectations 7/8. The miss: the hoop peak is refused (U 2.9% > 2% tolerance; true finest error
    2.03%, so the refusal is correct).
  - The g = 2 graded strategy inverted a Gauss point. It is not used by the plan.
- Changes after the pre-registration:
  - Solver name "spsolve" to "direct" (a code fix).
  - crosscheck.py changed later (v2). It has no entry for this problem, and compare() allows exactly
    this one documented change.

### P4. Three-way evaluation
- `three_way.py` (new) compares, on the same stored solves, quantities, tolerances and labels (102
  quantities, 9 problems):
  - conventional GCI;
  - the deterministic framework without an LLM (as pre-registered, and with the post hoc extension);
  - the final cross-checked status;
  - the LLM agent (9 stored billed runs on B1-B3; **no new billed run**).
- Results:
  - Verifiable quantities certified (of 79): GCI 56; deterministic 56 (19 of its 23 non-certified are
    INCONCLUSIVE, not FAIL); with the extension 72; final 47. Missed failures: 0 for all.
  - B1-B3: GCI 22/41, deterministic with the extension 38/41.
  - LLM agent: specification 9/9; B3 coverage 44-71%; 2-9 false-alarm predictions per run.
  - Human-decision proxy on B1-B3: GCI 37, deterministic 9, LLM agent 19.
  - Unknowns solved (B3): GCI 1.76 M, deterministic 3.88 M, final 4.91 M.

### P5. Autonomous investigation **[post hoc]**
- `diagnose.py`:
  - New hypothesis C1 and a sixth class, "verification capability": a check that lacks theory for an
    element is not read as non-convergence.
  - Derived problem facts for problems without hand-written ones. Unknown facts lead to NEEDS_DATA,
    never to a guess.
  - D4 uses the drift of the pairwise orders.
  - Q3 (Gauss-point location) is generic.
- Accuracy on independently established causes: 12/12 from 7 of 9 problems, against 9/9 on 5 problems
  before. The 19 capability-gap cases are reported apart, because they are correct by construction.
  B1-B3 consistency with the manual findings is still 16/16.
- `followup.py` (new) acts on a D4 diagnosis. It estimates time and memory from its own solves, runs
  one finer level within a budget (300 s, 8 GB), stores it apart from the pre-registered ladder, and
  re-evaluates with the frozen checks.
- First use, L-shape v_right_top, predictions committed first:
  - next pairwise order 1.01 (predicted 0.97-1.15);
  - five-level order 0.938, so the verdict stays FAIL (predicted >= 0.94: missed by 0.002);
  - 41 s against an estimate of 29 s;
  - the next level would need ~19 GB, so the agent stops.
- `advisor_feedback/2026-11-23_automation_assessment.md`: which decisions are automated, the evidence of
  reliability, and the safeguards.

### Other files
- Report v5 (Claude doc and PDF): new Section 30, with pointers in Sections 1, 25 and 27; Figure 25.4
  labelled as rule v1.
- `verify_report.py`: Section 30 claims added; 890/890.
- `test_llm_guards.py`: 69 tests (6 new: cross-check v2, capability gap, pairwise orders, derived
  facts, follow-up budget).
- Email draft `2026-11-20_email_to_Rabczuk_reply_DRAFT.txt`: **one factual correction** in point 5.
  The cross-check counts (21/5/8, "five show disagreement") are replaced by the v2 result (23 pass,
  11 inconclusive, 0 contradicted; the ~2% gap neither confirmed nor excluded), because the attached
  report no longer supports "five show disagreement". Nothing else in the email was changed. Not sent.

## 2. Reproducing the results (from `Practical_Examples/`)

| Step | Command | Time |
|---|---|---|
| manuscript claims | `python -m omar_pfem.verification_agent.manuscript_claims` | 1 min |
| independent values | `python -m omar_pfem.verification_agent.crosscheck_extra b1 cook` | 12 min |
| cross-check v1 + v2 | `python -m omar_pfem.verification_agent.crosscheck` | 1 min |
| lamina zone study | `python -m omar_pfem.verification_agent.lamina_zone_study` | 10 min |
| 3D benchmark | `... cylinder3d exact`; `... cylinder3d solve`; `... cylinder3d_preregister --compare` | 4 min |
| three-way | `python -m omar_pfem.verification_agent.three_way` | 3 min |
| follow-up | `python -m omar_pfem.verification_agent.followup l_shape v_right_top` | 1 min (already stored: rerunning appends another level) |
| report claims | `python -m omar_pfem.verification_agent.verify_report` | 3 min |
| tests | `python -m pytest omar_pfem/verification_agent/test_llm_guards.py` | 1 min |

Cluster jobs, prepared and NOT submitted (for authorized manual submission): `iga_b3_job.sh` (spline
level 5, groove peak) and `iga_b3_cyl_job.sh` (cylindrical averages).

## 3. Remaining limitations (updated)

1. **The B3 Gaussian averages have no converged independent value.** The ~2% spline gap is neither
   confirmed nor excluded, and the groove peak is not certified. Both need the prepared cluster runs.
2. **The normalisation scale is loose.** The B3 Gaussian-region stress scale (3.10) is inflated by
   the reduced-integration pressure oscillation, so tolerances and uncertainties relative to it are
   6-7x looser for the normal components than relative to their values. The cross-check passes for
   sigma_yy and pressure rest on this. *Round J:* on their own values all seven B3 Gaussian
   uncertainties are 0.79-1.01%, inside the 2% tolerance; no certification depends on the scale.
3. **Certification by one refinement family bounds that family's sequence, not the continuum value.**
   The final, cross-checked status certifies 47 instead of 72 quantities. 11 are INCONCLUSIVE until
   independent solvers exist for them.
4. **Independent solvers, data adapters and problem facts are written by hand for each problem class.**
   The diagnosis works with derived facts, but most solver/physics tests then stay NEEDS_DATA.
   *Round J:* the solver and code tests (S1-S3) now have stored results for the L-shape, crack,
   lamina and both 3D problems (`followup_hooks.py`); physics tests remain hand-written.
5. **3D coverage.** One smooth 3D problem only: no singular or concentrated feature in 3D, no contact,
   no other element types. *Round J:* a second, singular 3D problem (clamped block) was added; vertex
   exponents are not computed, its energy is still pre-asymptotic, and it has no independent value.
6. **The LLM agent was measured on B1-B3 only** (9 billed runs). The other problems need approved
   billed runs. Its added value is specification extraction; its predictions add false alarms.
7. **The human-intervention numbers are a proxy** (set-up inputs plus quantities left to examine), not
   a timed study.
8. **Several criteria are post hoc:** the singular-order extension, the pre-asymptotic label, cross-check
   v1 and v2, the C1 hypothesis, the pairwise-order D4 test and the derived facts. Each was evaluated
   on at most one or two unseen cases after it was introduced (L-shape, crack, 3D benchmark, the
   follow-up).
9. **Strongly graded Q20 reduced-integration meshes are fragile.** An inverted Gauss point appeared in
   B3 and in the 3D benchmark. Hourglass control or full integration on such meshes is not implemented.
   *Round J, correction:* for the 3D benchmark this attribution is wrong. The full 3x3x3 rule inverts
   too; 40 load increments instead of 10 give the correct solution with either rule. The cause is a
   Newton solve that converged silently to an inverted state (see the round J section below).
10. **Independent human review** of the code is still missing. The claims checks (890/890, 28/28)
    check consistency with the stored data, not the correctness of the code that produced it.


---

# Round J (23 November 2026, later): everything that needs no approval

Report: Section 31 of `advisor_feedback/2026-11-23_Verification_Framework_Technical_Report_v6.pdf` (74 pages). No billed LLM run, no cluster submission, no
email. Post hoc changes are marked **[post hoc]**; no criterion, tolerance or order window changed.

## Changes
- **Graded-mesh failure explained** (`graded_q20_study.py`, `q20_full.py`, `graded_q20_study.json`):
  cylinder k = 4, g = 2.0 inverts with the reduced (min det F -1.03) and the full rule (-0.74) at 10
  increments, and is correct with both at 40 (min det F 1.004; reaction and displacement errors
  <= 1.1e-5). My reduced-quadrature hypothesis is refuted.
- **[post hoc] Admissibility guard** (`detf_guard.py`, `detf_guard.json`): 64 stored solves with min
  det F on seven problems; one inadmissible (that strategy), used by no verdict. Proposed rule for
  future runs (re-solve with 4x increments, else exclude): not applied to the frozen scoring.
- **Second unseen 3D benchmark** (`block3d.py`, `block3d_preregister.py`; pre-registered in b850330):
  default rules 1/3 predictions, post hoc extension 2/3, expectations 8/10; peak slope -0.281 against
  lam - 1 = -0.289. Misses: energy labelled pre-asymptotic (order 1.68 vs 1.42); mid-span average only
  band over five levels. Root average FAILs strategy robustness (3.5% > 2%).
- **Bug fix, pairwise orders** (`diagnose.py`): the formula assumed a constant refinement ratio; the 3D
  ladders have none (exact h^1.4 gave 3.28, 2.90, 0.33, 2.90). Now the three-level equation is solved;
  unchanged for 2D. No supported cause changed. New test.
- **3D follow-up** (`followup.py`): next level 1.5k, memory 76 kB/unknown (lower bound, measured at
  k = 6). Block energy: chooses k = 12, declines (983 s, >= 14.8 GB vs 300 s, 8 GB). Predictions
  committed before any solve (`block3d_followup.py`, `block3d_followup_preregistration.json`); job
  `Practical_Examples/block3d_followup_job.sh` prepared for manual submission.
- **Solver/code hooks** (`followup_hooks.py`): MMS through the L-shape/crack form, L2 2.98 / H1 2.00;
  tighter Newton tolerance changes <= 2.6e-15 (cylinder) and 9.6e-12 (block).
- **Normalisation analysis** (`normalisation_study.py`): 54 rate-certified, 7 on a larger scale (all B3
  Gaussian, ratio 5.7-7.3), all within tolerance on their own values.
- **Spline cylindrical rr/tt** (`iga_b3_cyl_local.py`, `iga_b3_cyl_result.json`): local run of the
  prepared job; the cross-check reads it. See Section 31.7 for the state.
- **Three-way on ten problems**: certified of 83 verifiable: GCI 60, deterministic 56, + extension 74,
  final 47; missed failures 0 everywhere; diagnosis 13/13 on independent references (22 C1 apart).
- Claims checks: Sections 25 and 30 now read snapshots `*_round_I.json`; 69 tests pass.

### Round J, second pass (after reviewing Section 31; report Section 31.10)
- **Block strategies at k = 6** (`block3d_strategy_k6.py`, stored apart): root average spread 1.82%
  (k = 4: 3.48%), 0.95% from the family limit -> the k = 4 FAIL reflects coarse strategy meshes. The
  pre-registered verdict stays FAIL; choosing the strategy level from the family's convergence is a
  candidate rule, not adopted.
- **Independent Q8 discretization for the block** (`block3d.run_q8`, `block3d_q8_ladder.json`, k = 2-16,
  109,395 unknowns): tip deflection (Q8 order 1.76, U 0.46%, gap 0.022%) and mid-span average (1.86,
  0.23%, 0.024%) cross-checked -> CERTIFIED. Same code, other element family.
- **[post hoc] Cross-check scope:** quantities certified only with the singular-order extension are now
  cross-checked in a separate list (`rows_posthoc_only`): 2 cross-checked, 14 without independent value
  (L-shape, crack, plate). Main summary unchanged. Linear-element order window 0.5-3 (same outcome
  with 0.5-5). Three-way final: 49 of 83 (was 47).
- **[post hoc] det F guard as a switch** (`detf_guard.guarded`, VERIF_POSTHOC_DETF=1, off by default) in
  the 3D solvers: re-solves with 4x increments (cylinder g = 2: -1.03 -> 1.004). The exclusion step
  needs a change in the frozen `orchestrate.py`; not made. New test (70 tests).
- Report wording: the mid-span "miss" is in the wording of my expectation; Sections 30.2/30.4 point to 31.
- Claims 957/957.

### Round J, third pass (report Section 31.11)
- **Independent P2 discretizations** for the L-shape, edge crack and plate (`independent_p2.py`, ladders
  `*_p2_ladder.json`, up to 526,850 unknowns): the problems' own code with 6-node triangles (each Q2
  quad split in two; plate hole mid-edge nodes on the exact circle). Of the 16 quantities certified
  only with the post hoc extension: 13 cross-checked, 3 independent-unconverged (L-shape corner
  averages, plate Gaussian hole average: P2 flat but no rate). Final status: 60 of 83 (was 49).
- **Report text check** (`report_text_check.py`): every claim's value is searched in the exported PDF.
  965 claims: 794 in their section, 91 elsewhere (section boundaries), 48 flags, 32 backing values with
  a recorded reason, 0 missing. Finds omissions, not proof (a number can match by coincidence).
- Claims 965/965; tests 71; all pre-registered scorings reproduce (block 8/10, cylinder 7/8, L-shape
  10/11, crack 11/12).

## Reproduce
    cd Practical_Examples
    python -m omar_pfem.verification_agent.graded_q20_study
    python -m omar_pfem.verification_agent.detf_guard
    python -m omar_pfem.verification_agent.block3d_preregister --compare
    python -m omar_pfem.verification_agent.followup_hooks
    python -m omar_pfem.verification_agent.followup clamped_block_3d strain_energy
    python -m omar_pfem.verification_agent.normalisation_study
    python -m omar_pfem.verification_agent.crosscheck
    python -m omar_pfem.verification_agent.three_way
    python -m omar_pfem.verification_agent.verify_report
    python -m pytest -q omar_pfem/verification_agent/test_llm_guards.py

## Limitations added or changed in round J
1. Inverted converged solves are detected, but the remedy is proposed, not in the frozen rules.
2. 3D vertex singularities are not computed; the block energy needs the k = 12 cluster level.
3. The 3D memory model is a lower bound from one level.
4. The independent values for the block (Q8) and the 2D singular problems (P2) are other element
   families in the same code, not second codes.
6. Not yet done (needs the memory the spline run uses): whether the B3 radial-grading inversions
   (>= 1.75) are also a load-step effect.
5. The 2D code checks use the unit square, not the singular geometries.

---

# Round K (23 November 2026): certification tested on a singular exact solution; the first remedy

Report: Section 32 of `advisor_feedback/2026-11-23_Verification_Framework_Technical_Report_v7.pdf`
(80 pages, Figures 32.1-32.2). No billed run, no cluster submission, no email. Pre-registered work:
commits 338d7f1 (exact benchmark + remedy) and be9e643 (remedy on L-shape and crack).

## Changes
- **Singular benchmark with an exact solution** (`lshape_exact.py`, `lshape_exact_preregister.py`,
  `extra_adapters.py`): L-shape loaded by the exact Williams solution u1 + 0.5 u2 (lam 0.5445, 0.9085) on
  the outer edges; exact energy two ways to 1.7e-14. Default rules 1/4, post hoc extension 4/4; **exact
  value inside the certified interval for 6/6**; expectations 9/10; P2 cross-checks 4/6.
  Amendment before scoring: interpolated query points on graded meshes (uniform values to round-off).
- **First remedy** (`remedy.py`): g > p/lam (3.7). Energy order 3.34; true energy error 1.5e-7 at
  99,330 unknowns against 3.7e-4 uniform at 395,266: ~2,400x smaller with a quarter of the unknowns.
- **Automatic corner analysis** (`remedy.corner_analysis`, `remedy_preregister.py`): finds every
  rate-limiting corner of the L-shape and the crack from polygon + edge conditions. Remedy on the real
  problems: 4/10 pre-registered predictions; the exponent-only order predictions missed (weak corners
  not yet dominant up to n = 64); U 4-80x smaller; energy cross-checked by P2 on graded meshes.
  Amendments before scoring: size-scaled inverse-map tolerance; absolute facet-selection tolerance
  (the modules' relative 1e-5 selected short facets across thin rows: top load 2.0 instead of 1.0).
- **Calibration** (`calibration.py`): every quantity with an exact value. Framework 12/12 covered
  (median ratio true error/U 0.07, max 0.70); conventional GCI 9/11 (false certifications on the two
  sharp-disc averages, ratios 2.6 and 17.1).
- **B3 graded inversion is geometric** (`b3_mesh_jacobian.py`): negative undeformed det J for
  r_grading >= 1.75 (480 / 840 points). Not a load-step effect; manuscript sentence correct as written.
- **Second code for the block** (`independent_block3d.py`, scikit-fem Hex27, Newton from the energy;
  report 32.6): five levels up to 47,775 unknowns. Tip deflection cross-checked by a second code (order
  1.90, U 0.10%, gap 0.021%); the mid-span average is flat in it (band, agrees to ~1e-6). Tangent checked
  against the formula and finite differences (test).
- **Frozen-file exemptions are now checked** (`prereg_util.py`): a scoring continues after a frozen file
  changed only if the functions it uses are unchanged since the pre-registration commit (AST via git).
- Report text check covers v7 (80 pages): 1,055 claims, 0 missing (numbers wrapped across lines handled).
  Claims 1,055/1,055; tests 75; all pre-registered scorings reproduce.

## Reproduce
    cd Practical_Examples
    python -m omar_pfem.verification_agent.lshape_exact exact
    python -m omar_pfem.verification_agent.lshape_exact_preregister --compare
    python -m omar_pfem.verification_agent.remedy_preregister --compare
    python -m omar_pfem.verification_agent.calibration
    python -m omar_pfem.verification_agent.b3_mesh_jacobian
    python -m omar_pfem.verification_agent.verify_report
    python -m omar_pfem.verification_agent.report_text_check

## Limitations added or changed in round K
1. One singular problem with an exact solution (coverage 6/6) - not a statistical guarantee.
2. Sequences converged to round-off give band evidence only: no interval, cannot be certified with a rate.
3. The remedy plans from exponents only; corner intensities must decide which corners to grade.
4. Long background runs stop whenever the session machine restarts (they resume from the last level).

---

# Round L (23 November 2026): intensity-aware remedy on a held-out problem

Report: Section 33 of `advisor_feedback/2026-11-23_Verification_Framework_Technical_Report_v8.pdf` (83 pages).
Pre-registered before any loaded solve: commit 4fd091f. No billed run, no cluster job, no email.

## Changes
- `remedy_intensity.py`: grading plan weighted by corner intensity (local strain-energy change within
  d < 0.25 of each rate-limiting corner between the two finest uniform levels; grade corners with share
  >= 0.10, each with its own g > p/lam). Constants fixed before use.
- `ushape.py` (held-out U-shaped panel, two re-entrant corners of equal exponent and different
  intensity, two clamped corners) and `ushape_preregister.py`.
- Result: plan shares 0.69 / 0.17 / 0.14 / 0.003 (upper clamped corner correctly left uniform);
  5/6 predictions met. Energy U 15x below uniform (tip deflection 71x), order 2.90, P2 cross-checks
  energy (gap 0.004%) and tip deflection. Miss: for the energy, grading only the two re-entrant corners
  gives a smaller U (3.8e-5 vs 6.5e-5); the intensity plan beats the one-exponent "all corners" plan for
  both quantities and is best of the three for the tip deflection.
- Claims 1,085/1,085; report text check (v8): 0 missing; tests 76.

## Reproduce
    cd Practical_Examples
    python -m omar_pfem.verification_agent.ushape_preregister --run
    python -m omar_pfem.verification_agent.ushape_preregister --compare

## Limitation changed
- The remedy now uses intensities, but one held-out problem and one threshold; trading the grading of
  each corner against its share (instead of a fixed 10% threshold) is the next step.

## Round M (2026-11-23): billed LLM-vs-rules ablation on 7 more problems (approved)

- 21 runs, USD 12.16; per-problem results and table in
  `Practical_Examples/omar_pfem/verification_agent/ablation_round_M/` (SUMMARY.md).
- Pattern of B1-B3 confirmed: specification recovered (18/21), 1-3 false alarms per run;
  the LLM caught the block failure the rules miss (3/3).
- New limitation found: strategy_robustness / anchored_robustness turn a FAILED SOLVE in one
  strategy (cylinder grading 2.0: inverted elements, NaN energy) into FAIL for every quantity.
  It should be INCONCLUSIVE. Not changed (would be post hoc); proposed for the next rule revision.
- Scoring script limitation: a degenerate run (no quantities chosen; crack run 2) is counted
  as 0 misses / 0 false alarms; excluded by hand in SUMMARY.md.
- FIXED (post hoc, labelled): check-level det F guard behind VERIF_POSTHOC_DETF (off by default).
  `detf_guard.admissible_strategies` + two guarded blocks in `orchestrate.run_check`: an inverted
  strategy solve is replaced by its guarded re-solve (`cylinder3d_strategy_guarded.json`, 40
  increments, min det F 1.004, energy within 3e-8 of uniform) or the check is INCONCLUSIVE.
  Cylinder: strategy checks FAIL 6 -> 1 quantity (only peak_hoop, the true failure).
- Frozen pre-registrations (block, exact L-shape) still score via a CHECKED exemption
  `prereg_util.unchanged_except_switch` (module == b850330 once the switch blocks are stripped;
  negative control refused); their stored results are unchanged.
- `round_m.py` -> `round_m_result.json`; degenerate runs excluded from the means.
- Report v9 (`2026-11-23_Verification_Framework_Technical_Report_v9.pdf`, 84 pages): Section 34 +
  Section 1 item 4. Claims 1,115/1,115. Tests 79/79.
- `report_text_check.py`: headings must start at column 0, so indented list items ("1. The ...")
  no longer split sections; on v9: section 1,001, elsewhere 28, NOT FOUND 0.
- Email draft renamed to `2026-11-23_email_to_Rabczuk_reply_DRAFT_v9.txt`; only change:
  "version 8" -> "version 9" (Omar asked for v9). Not sent.
- Remaining limitations: 3 runs per problem; the LLM plans of round M were not stored, so the
  guard's effect on the LLM's own agreement is inferred from the checks, not re-run (billed).

## 24 November: the three cluster jobs (report v10, Section 35)

- Jobs 3280560 (spline rr/tt; P1 L1-L4 + disp L4), 3280561 (block k = 12), 3280562 (spline L5,
  P1 + disp) ran on the cluster, submitted by Omar. GitHub is unreachable from the HPC: code went
  as `transfer/cluster_update.tar.gz`; results came back by terminal copy, md5-verified.
- Block k = 12: pre-registered predictions 6/6; energy INCONCLUSIVE -> PASS with rate (orders
  1.84, 1.53, 1.38, 1.32 -> 2 lam 1.42; five-level 1.43). `block3d_followup.py` scoring got the
  same checked orchestrate.py exemption (pre-registered hash == b850330).
- **Correction (factual):** the groove peak does NOT settle. P1: 1.71, 1.06, 1.00, **0.89**; disp:
  3.18, 1.68, 1.28, 1.04. Earlier "settles" statements (Sections 1, 15, 25, 28, Fig. 25.2,
  scope table) corrected in the doc.
- crosscheck.py now reads every stored spline level (1-5; data grew, rule unchanged). With five
  levels the leave-one-out test runs and fails for every spline Gaussian-average sequence: no
  rate. Cross-check: 25 cross-checked / 13 INCONCLUSIVE (was 27/11 with four levels; snapshot
  `crosscheck_result_round_M_4levels.json`); sigma_yy and pressure averages lose their
  cross-check. Three-way final: 58 of 83 (was 60; snapshot `three_way_result_round_M_4levels.json`,
  which Section 31 now reads), missed failures 0.
- P1 Gaussian sigma_xx keeps rising (0.5459 ... 0.5518), now 2.2% above the Q20 limit;
  displacement splines 0.16% above. The ~2% question stays open.
- Claims 1,155/1,155; text check NOT FOUND 0; tests 79/79.
- Email draft -> `2026-11-24_email_to_Rabczuk_reply_DRAFT_v10.txt`: factual corrections only
  (version 10; 25/13; 58; question 3 rewritten because the refinement it asked about has run).

## 24 November, round N (report v11, Section 36)

- **Wedge coverage (pre-registered, 32 exact Williams corner problems, 160 quantities, 128
  verifiable):** framework certifies 82, covers 81 (98.8%, Wilson 93.4-99.8%), median
  effectivity 0.10, worst 1.24; GCI certifies 120, covers 118, worst effectivity 10.3. No peak
  certified. Predictions 5/6 (P3 "GCI misses >= 3" missed: 2).
- **Generic 2D pipeline** (`generic2d.py`, JSON description -> verdicts): U-shape reproduced to
  1e-12; T-bracket pre-registered 6/6; uniform 4/7, graded 5/7 certified and cross-checked;
  U gains 28x energy, 18.8x tip, 10.9x, 3.6x. Amendment (before any graded result):
  graded solves use the size-scaled inverse-map tolerance of rounds K/L.
- **Finite strain** (12 manufactured neo-Hookean cases, up to 20% Green strain): framework 21/36
  certified, 21/21 covered (Wilson from 84.5%), median eff. 0.27, worst 0.35; GCI 36 certified,
  32 covered (worst 1.49). Predictions 3/4 (F2 liveness missed).
- **POST HOC** (`posthoc_window.py`, not in the frozen framework): finest-four refit when the
  five-level fit gives no rate. Finite strain 36/36 certified and covered (worst 0.60); 32
  corner problems unchanged (no refit passes). Adoption needs a new pre-registered test.
- **LLM vote (billed, USD 19.20):** 26 runs on 9 problems (usage limit killed crack x3 and one
  lshape run; cylinder and block re-run after the reset). Missed/false alarms: single mean
  8.3/21.7, majority 8/20, unanimous 10/15, rules 8/4. Predictions 4/4 but small effect.
- **Automation:** `job_factory.py` (make/import with md5 + no-overwrite merge); GitHub workflow
  (guard tests, claims check, text check) green after adding all third-party deps and a
  matching CPU torch/torchvision pair; dashboard artifact (fixed data files).
- **Not done:** p = 3 spline groove sequence (coarsest level > 14 min vs ~1 min at p = 2);
  needs a faster tangent assembly before any cluster job.
- Claims check 1,215/1,215; report text check 0 not found on v11 (90 pages).

## 24 November, round O (after Professor Rabczuk's reply; new-results PDF only)

- **Decisions Q1-Q3 implemented** as `adopted.py` (a layer; frozen modules unchanged): singular-order
  extension adopted; pre-asymptotic from below (order below theory, pairwise orders rising by
  > 0.02) is INCONCLUSIVE like from above; absolute targets 1% (global, displacement) / 5% (region
  averages), U <= half-width/3 (`requirements.ABSOLUTE_TARGETS_ADOPTED`, `constitutive_requirement`).
  Impact (`adopted_impact.json`): 1 of 83 verdicts changes (L-shape v_right_top FAIL ->
  INCONCLUSIVE); B* 58/83, 0 missed failures. NOT blind (decided after held-out results).
- **Autonomous loop** (`autonomous.py`): L literature -> D formalizer + modeler (code validation and
  number tracing) -> M pipeline2d -> V mms_auto -> S/C ladders + adopted rules + cross-check -> G
  diagnosis on derived facts -> N experiment protocol (`experiment.py`: candidates, prior x value /
  cost, frozen prediction with sha256, scored by code, `experience.json`). Each step logs its actor.
- `pipeline2d.py` (cell size, symmetry, pressure, plane stress, MMS inside the production solver;
  reproduces generic2d to 0); `corners_ext.py` (symmetry edges, complex exponents: clamped-free 270
  deg has Re 0.34; the old analysis gave 0.87 -- no earlier problem affected); `mms_auto.py`.
- **Autonomy test, pre-registered** (`autonomy_test/`): S1 stops 4/4, S2 MISSED (T4/T5 end to end,
  MMS PASS, energy certified, but flagged by prose assumptions), S3 met, S4 met (USD 1.19). Section
  choice wrong in 2 of 3 papers. **POST HOC** fixes: passage selector (`literature_select.py`), flag
  only assumed physical numbers, vertex tracing; post hoc rerun of T2/T3 right (USD 0.59).
- **Decision 3:** `groove_experiment.py` froze the agent's proposal (sha a909b1f9bd88): prediction
  |P1(L6)/P1(L5)-1| < 0.110 and gap < 0.151; cluster job `groove_l6` (1,029,192 unknowns, ~6 h per
  formulation, 18 h, 64 GB), code path smoke-tested; NOT submitted.
- New-results report: `2026-11-24_Autonomous_Workflow_New_Results.pdf` (6 pages), claims 36/36
  (`round_o_claims.py`, also in CI). Reply draft `..._DRAFT_round_O.txt` has a placeholder for CAS.
- Limits: 2D linear elasticity on unions of squares only; 3D/hyperelastic solvers not in the loop;
  memory of 12 experiments; priors not calibrated; no open paper in the test was solvable from text.

## 25 November: neo-Hookean in the loop, automatic literature search (new-results PDF Section 8)

- `pipeline2d_nh.py`: compressible neo-Hookean, plane strain, dead tractions (pressure and plane
  stress refused); load steps 4 -> 64 doubled on Newton failure or J <= 0; MMS (`ManufacturedNH`,
  ~10% strain) PASS (Q2 L2 3.05, H1 2.01). Newton stopping-criterion bug at tiny loads found by a
  new test and fixed (update at round-off level counts as converged).
- Loop: failed experiments are recorded (refuted) and banned per problem; new diagnostic action
  `small_load_check`. Demo (L-bracket, neo-Hookean): nothing certified at the stated load (energy
  order 0.69 vs linearized 1.09; min det F 0.87 -> 0.32), graded meshes invert at the corner, the
  agent's small-load hypothesis confirmed (order 1.17). n = 128 run killed by time limit (cost
  model underestimate) -> level capped at 64 for neo-Hookean; the killed run's two memory entries
  were removed (development run).
- `literature_search.py` (pre-registered): 66 papers, 27 abstract screen, 12 scouted, 1 positive,
  read and refused correctly (Mooney-Rivlin, Cook's membrane); 3/5 predictions (L2, L3 missed);
  USD 0.84. Log attribution bug ("chosen by a person") fixed post hoc (`chosen_by`).
- Cook's membrane named in 4 of 12 scouted papers -> first geometry family: inclined edges.
- New-results PDF now 8 pages (Section 8 added, Sections 3, 4 and 7 updated), claims 51/51.
- 25 Nov, groove L6 (job 3297326, md5 706016932817b103b0df0dfc2f6ab63e, imported with job_factory):
  P1 peak 0.92 (L5 0.89), disp 1.06; agent prediction CONFIRMED (change 3.6% < 11.0%, gap 0.135 <
  0.151) but the P1 sequence is not monotone -> peak stays not verified (frozen rule). L2-L6: moment
  and energy rate (1.51, 1.54 P1); Gaussian averages no rate; P1 sxx 2.3% and disp 0.60% above the
  Q20 limit, both rising -> Q20 Gaussian likely below the continuum value. Times 5.0 / 5.3 h (est. 6).
  Crosscheck/three-way of report v11 NOT recomputed with L6 (kept reproducible; L6 in new-results
  PDF Section 8.3). New-results PDF claims 66/66 (memory claims now matched as phrases).

## 25 November: general 2D geometry in the loop (inclined edges, holes, curved edges)

- `geometry2d.py` (gmsh 4.13.1): polygons with circular or polygonal holes, quadratic P2 triangles
  and independent Q2 quadrilaterals, curved edges on circles, points of interest embedded as nodes,
  corners and their exponents taken from the real meshed boundary (corners_ext), measured
  h = sqrt(area / elements). The modeler schema accepts `geometry_kind: polygon`.
- Self-similar refinement (fix): the size field is the same at every level, scaled by h = L/n
  (0.2 h at holes growing to h at L/2). Curvature-based sizing changed regime between levels and
  made the plate ladders irregular (nothing certified); now energy differences fall about 10x per level.
- BUG found and fixed: the ladder cache (pipeline2d_runs/<name>/ladders.json) was keyed by the problem
  name only, so a changed mesh generator silently reused old solutions (experiments "took 0.0 min").
  It is now tied to a sha256 fingerprint of the description and the solver/mesh code; a mismatched
  cache is moved aside (ladders_stale_*.json) and recomputed. Development memory entries and MMS
  records from the affected runs (10 and 8) were removed. The mms_auto key now includes geometry2d.py.
- Results (fresh solves): MMS PASS on both (Q2 L2 3.10-3.12, H1 2.04-2.08; P2 L2 3.08-3.18).
  Cook's membrane: v_C, strain energy and two corner averages certified; peak concluded unbounded
  (corner exponent 0.677 < 1); v_C 23.964 vs reference 23.96 at (48, 52). Plate with hole
  (10 x 10, r = 1): all 4 quantities certified (peak von Mises 3.0865 +- 0.0034). Kirsch check by hand
  (40 x 40): peak 2.946, 2.976, 2.990 -> about 3.005 vs 3.00. Record: geometry2d_validation.json.
- Tests 95 pass (point outside the domain refused, Cook mid-edge value, plate corners none and
  regular convergence, cache invalidation). CI installs gmsh and its system libraries.
