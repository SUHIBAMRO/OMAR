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
   sigma_yy and pressure rest on this.
3. **Certification by one refinement family bounds that family's sequence, not the continuum value.**
   The final, cross-checked status certifies 47 instead of 72 quantities. 11 are INCONCLUSIVE until
   independent solvers exist for them.
4. **Independent solvers, data adapters and problem facts are written by hand for each problem class.**
   The diagnosis works with derived facts, but most solver/physics tests then stay NEEDS_DATA.
5. **3D coverage.** One smooth 3D problem only: no singular or concentrated feature in 3D, no contact,
   no other element types.
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
10. **Independent human review** of the code is still missing. The claims checks (890/890, 28/28)
    check consistency with the stored data, not the correctness of the code that produced it.
