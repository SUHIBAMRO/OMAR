# PFEM / Transolver Project — Status Tracker

**Read this file FIRST at the start of any new conversation about this project.**
It is the single source of truth for where things stand — more reliable than
chat history, which resets between sessions. Update it whenever a task
finishes or a new one starts.

Last updated: 2026-08-28 (**point 5 is now written into both documents —
report v29 and the summary. See "Point 5 written in" below.**)

---

## Point 5 written into both documents (2026-08-28) — report v29

Point 5 was measured for all six cases but had never reached the report.
It is in now, as report **section 8.8, Tables 15/16/17**, and as **section 8**
of the parallel summary, plus a qualifying paragraph in each document's
conclusion. The old section 8.8 (training visualizations) became 8.9; nothing
cross-references "8.8", so that was safe.

Build scripts, all committed:
- `report_builders/point5_tables.py` — builds the three tables from the JSONs.
  **Both** document scripts import it, so the two documents cannot disagree;
  last round they were typed twice and compared afterwards.
- `report_builders/make_v29.py` — v28 → v29. Re-runnable.
- `report_builders/make_summary_v3.py` — reads
  `PFEM_Summary_Completed_Work.pre_v3.docx` and writes the live file, so
  re-running replaces section 8 instead of appending a second one.

The six result files are committed at
`Practical_Examples/omar_pfem/point5_results/`, with a README recording their
provenance and one incompleteness (below).

### Verification
- 216 table cells (108 per document) recomputed straight from the six JSONs by
  a separate script and compared against the saved .docx. 0 mismatches.
- The two documents' copies compared cell by cell against each other: 124
  cells, 0 differences.
- **Every cross-case sentence is asserted in `make_v29.py` before it is
  written.** This caught three false claims in the first draft:
  1. "the reaction resultant is as accurate as or more accurate than the
     displacement" — false for B2 × Mooney-Rivlin (12.61% against 7.21%) and
     for B2 × Neo-Hookean's θ=0 edge. It holds in **four of six** cases.
  2. "the H1 semi-norm is the worst of the integral measures" — true on B1,
     false on all three B2 cases, where the aggregate stress is the largest.
  3. "the tangent-energy error sits between the two throughout" — false for
     B2 × Mooney-Rivlin.
  A fourth was a code bug: the peak-stress standard deviation was reading the
  `mean` field, so the text said 20–48 percentage points when the real spread
  is 24–39.

### The headline numbers (means over 50 held-out samples, per cent)
| | B1 | B2 |
|---|---|---|
| displacement (report's own definition) | 10.34–11.71 | 7.21–10.47 |
| H1 semi-norm | 22.71–24.20 | 10.47–13.30 |
| tangent energy | 13.78–17.02 | 11.37–11.99 |
| aggregate PK1 stress (Frobenius) | 15.06–18.14 | 13.32–14.23 |
| peak ‖P‖ | **19.87–47.53** | **5.38–5.99** |
| reaction resultant | 4.66–8.41 | 6.26–12.61 |

Two findings worth carrying forward: H1, tangent energy and aggregate stress
exceed the displacement error in **all six** cases without exception, so a
displacement figure quoted alone is a lower bound; and peak stress splits the
benchmarks — best-in-section on B2, worst on B1, where the predicted peak
exceeds the reference peak in all three materials. No cause was isolated for
the B1 overshoot and none is claimed.

### One gap, not filled in
The Colab dump these JSONs came from printed the first 4,000 characters of
each file. The three B1 files are shorter and complete; the three B2 files
were cut part-way through their second symmetry edge's reaction block, so
`reaction_max_{pred,ref,rel_err}_edge1` is missing for B2. Nothing was
estimated to cover it: Table 17 uses only the resultant and nodal errors,
which are present everywhere, and the largest-single-nodal-reaction figure is
quoted for B1 only. Re-pulling those three files from Drive closes it and
changes nothing already written.

### Also found, NOT fixed
The report's Section 10 contains a reference to a **"Table 14" that does not
exist in the report**. The table itself exists only in the summary document
(final adopted B2 error for all three materials). Left alone deliberately —
the new tables are numbered 15/16/17 so that 14 is not silently absorbed. Ask
Omar whether to insert the missing table or reword the reference.

Also corrected: `physical_quantities_eval.py`'s comment about B2's symmetry
constraints had u_x and u_y the wrong way round. The **code** was right and
matches `train_B2` (`free_v[theta0_nodes]=0`, `free_u[thetahalfpi_nodes]=0`);
only the prose was wrong, so no number changes.

---

## State as of 2026-08-27

### Running on Colab right now
Three zero-shot notebooks, one per remaining B2 case, generating FEM data.
At the last report B2×MR was at N=33 train 75/400, ~136 s/sample, so roughly
16 h of generation remained for that case. B1×MR and B1×AB are also running
and were never affected by the B2 force bug. Each notebook resumes from
`samples_cache_N*.pt` on Drive if interrupted.

A fourth notebook is running the point-8 scaling sweep.

### Ready to run — scripts written, verified, committed
| Point | Script | Notes |
|---|---|---|
| 2 Pareto | `omar_pfem/pareto_analysis.py` | one run per finished zero-shot case; minutes, reuses the cached fine references |
| 6 OOD diagnosis | `omar_pfem/ood_diagnosis.py` | no new FEM solves at all |
| 8 scaling | `omar_pfem/gpu_fem_scaling_sweep.py` | needs a free GPU runtime; hours |

Ready-made Colab cells for each are in `zeroshot_notebooks/`. Every cell
should be self-contained (mount Drive, clone or pull, pip install) — a cell
that assumed `/content/OMAR` already existed failed with a bare `git` exit
128 in a fresh notebook.

### Done today
- Points 4 and 5 finished for all six cases.
- Point 3 recomputed at matched batch sizes and written into both documents
  (report v28 + summary), with every value re-verified.
- Point 8's Tensormesh question answered in the report.
- The B2 zero-shot force bug found and fixed; caches repaired in place.
- The GPU and matrix-free solvers fixed to match the CPU reference on B2;
  Table 9 regenerated.

### Blocked on Timon's reply — do not start
Points 7b (data-driven comparison) and 9 (MMS). The email asking about
scope was sent. Everything else has been done or is ready.

### If Google Drive is connected in your session
Results live under `MyDrive/pfem_run/`. Reading them directly saves the
copy-paste round trips this project has been doing all along; the per-run
`run_manifest.json` files record what produced each number.

---

## Dropped, do not restart

**B2 mesh-convergence study (the second case of the old round's point 1).**
Cancelled by the user on 2026-08-27: "خلص ملغي ما بدنا ياه". It was a
leftover from an earlier round and is not among Timon's round-5 requests.
Left here so a future session does not find it in an old task list and
revive it.

**Point 2's inputs were NOT already in hand (corrected 2026-08-27).** An
earlier note here said the Pareto only needed plotting. It does not. The
report's existing FEM accuracy-vs-cost curve (Table 6a) scores a fixed
analytic field against a ~10M-DOF reference, while the operator's error is
measured on random GRF fields against a same-mesh FEM solution — different
problems, different references, different hardware. Plotting them on shared
axes would look convincing and mean nothing.

`omar_pfem/pareto_analysis.py` measures both sides itself instead: same
problem instances (the parametric fields, same seeds), same fine-mesh
reference at `--fine_N`, same device, same batch size, with the operator
evaluated at every resolution from one checkpoint since resolution
invariance is the claim under test. It reuses the zero-shot eval's own
`fine_ref_cache_N*.pt`, so pointing it at a finished zero-shot case costs
nothing for the references. Smoke-tested end to end.

## CLOSED: the two solvers disagreed on B2 (found and fixed 2026-08-27)

`gpu_fem_solver.precompute_element_params_B2` sampled the material once at
each element's centroid while `solve_hyperelastic_TL_ring` sampled it at
every Gauss point, so the two were not solving the same B2 problem.

Not a design choice and not something the advisor asked for — an oversight
with a clear timeline. `gpu_fem_solver.py` and its validation script were
written 2026-07-29 (ff46d33), when both sides sampled at the centroid and
the validation genuinely passed. On 2026-08-10 (af7e67c) B2's CPU solver was
upgraded to per-Gauss-point sampling, deliberately and with B1 checked and
left alone, because "an element can span a genuine change in E/nu that a
single centroid sample misses". `gpu_fem_solver.py` was last touched
2026-08-03 and was never updated to follow, and nobody re-ran the
validation. Table 9 kept reporting the July result.

Evidence the table predates the change: it records `max|u_cpu| = 1.914e-2`
for B2 x NH; today's reference gives `1.9103e-2`. B1's `2.150e-3` is
unchanged.

**Fix:** `precompute_element_params_B2` now samples per Gauss point, at the
same `N @ Xe` locations and in the same Gauss order as the reference (the
ordering was verified identical across fem_core, gpu_fem_solver and
matrix_free_solver). It is also order-aware — Q4 gives (n_el, 4), Q9 gives
(n_el, 9) with the 3x3 rule — since `high_dof_convergence_study.py` calls it
for Q9 meshes too. Both energy functions in `gpu_fem_solver.py` and
`matrix_free_solver.py` accept either (n_el,) or (n_el, n_gauss).
`precompute_element_params_B1` is deliberately untouched: B1's own CPU
solver samples at the centroid, so matching the reference means staying
there.

**Verified, all six cases at N=11:**
| | before | after |
|---|---|---|
| B1 (all three materials) | 2.2–2.6e-16 PASS | unchanged, PASS |
| B2 (all three materials) | 4.8e-5 abs, 1.15e-3 rel, **FAIL** | **2.7–4.7e-16, PASS** |

The matrix-free solver was failing on B2 for the same reason and now also
passes (3.45e-15 vs the CPU reference, against its 1e-4 threshold).

### Consequences, all settled
1. **Table 9 regenerated** in both the report and the summary, from a fresh
   run of `validate_gpu_fem_solver.py` at N=11 (the configuration the table
   was originally produced at). All six rows PASS. The two documents'
   copies were compared cell by cell and are identical. Note the summary
   has *two* tables carrying a "Verdict" column — this one and the
   Q4-vs-Q9 convergence table — so match on the case names, not the header.
2. **Table 10's timings are unaffected** — where the material is sampled
   does not change how long a solve takes.
3. **Q9 / high-DOF B2 numbers: left as they are, by the user's decision
   (2026-08-27).** Tables 13/14 and the 10M/40M-DOF references were produced
   with centroid sampling on B2 and would move by roughly 0.1% if re-run.
   The Q4-vs-Q9 conclusion is a two-order gap, far larger than that, so it
   stands. Not worth the compute now; revisit only if a reason appears.
4. **The zero-shot notebooks are unaffected.** They call the CPU reference
   directly and never touch either GPU solver.

## Advisor's Round-5 feedback (2026-08-26) — 9 requests

Timon's framing: *"the results ... are very interesting. I think we can wrap
them up in a paper but I still have a few comments and requests."* So the work
is now aimed at a publication. He closes with *"There are many directions we
could pursue subsequently"* — this is a first round, so scope discipline
matters more than completeness.

**The email is stored verbatim at `advisor_feedback/2026-08-26_round5_timon.md`.**
Read it there before acting on any point; the table below is a summary and a
summary cannot settle a question about what was actually asked.

| # | Request | Status |
|---|---|---|
| 1 | Complete zero-shot resolution tests for the other five cases | 🟡 **restarted as 5 separate resumable notebooks** (the 3-notebook run lost its progress twice — see below) |
| 2 | Construct GPU-FEM vs Transolver accuracy/cost **Pareto** comparison | 🟡 **`omar_pfem/pareto_analysis.py` written and smoke-tested**; needs one run per finished zero-shot case |
| 3 | Recompute **break-even using GPU FEM** (not CPU) | ✅ **computed — see below** |
| 4 | Benchmark Transolver and GPU FEM under **identical batch sizes** | ⬜ needs Transolver inference at bs=8/32/128 (currently bs=1 only) |
| 5 | Error in **physically important quantities** beyond displacement: H1 semi-norm, energy, stress components, reaction forces, maxima (for the Transolver) | ✅ **done** — all six cases measured and written into report §8.8 (Tables 15–17) and summary §8; see "Point 5 written into both documents" at the top |
| 6 | Investigate **OOD robustness** — the 4–5× degradation is "probably the biggest obstacle to a strong physics-informed operator claim" | ⬜ research, not just measurement |
| 7 | Resolution invariance: train on 2, test on 5 **coarser AND finer**; the point being *"train on a very coarse grid and inference on a finer grid ... could provide computational savings"* — so **quantify the savings**, not just the flat error. Plus a **data-driven** comparison, its data *"from two different (fine enough) simulations"* (i.e. matched to the PI model's two training resolutions) | 🟡 7a covered by the per-case notebooks (7 resolutions, a superset of his 5); 7b pending |
| 8 | Test GPU-native FEM at **finer discretizations up to a few million DOFs**; and: *"Did you use Tensormesh or write the code yourself?"* | 🟡 **the Tensormesh question is answered in the report** (§8.5 now states it was written from scratch in PyTorch, not Tensormesh or any FE library); the 0.5/1/2/4M timing sweep is written as `omar_pfem/gpu_fem_scaling_sweep.py` and needs a free GPU runtime |
| 9 | Use **MMS** as ground truth instead of a baseline FEM solution, to test the operator *"compared to FEM"* — i.e. **both** are scored against the manufactured truth, which is the only way FEM itself gets graded (today it *is* the reference, so it cannot be) | ⬜ largest item; blocked on the energy functional having no body-force term, which MMS requires |

**Table 10 verified correct (2026-08-27).** Two things were checked here.

Which GPU timing set v27 uses: `gpu_fem_solver/`, confirmed from the report
itself — 7 decisive numbers match, 0 from the older `gpu_fem_timing_{B1,B2}`
set. This was the last claim resting on notes rather than a primary source.

Its speed-up column was then briefly and wrongly reported as an error. The
column is GPU FEM against each case's own **CPU** reference (Table 4a), which
its caption states plainly; it was misread as GPU FEM against the Transolver.
Read correctly, all six entries reproduce exactly — CPU_seconds*1000/GPU_ms
gives 71.72 / 149.39 / 138.89 / 73.04 / 171.52 / 159.27 against the printed
71.72 / 149.4 / 138.9 / 73.05 / 171.5 / 159.3. **Nothing in Table 10 needs
changing.** Recorded so the "error" is not rediscovered and acted on later.

The wide 71.7–171.5x spread is likewise real, not a symptom: it tracks CPU
cost, which varies 25.4–61.7 s/sample across materials, while GPU cost barely
moves (354.6–378.5 ms). Mooney-Rivlin and Arruda-Boyce cost 2.0–2.4x more on
CPU than Neo-Hookean, so they show the larger speed-ups.

**v27 -> v28 and the summary updated (2026-08-27).** The matched-batch
comparison is now written into both documents, in section 8.5 of the report
and section 5 of the summary. Build scripts are kept in
`Practical_Examples/report_builders/` so the edit is reproducible rather
than a one-off manual pass.

What changed in the report:
1. Section 8.5's closing paragraph claimed "a 73–80x GPU-to-GPU speed-up".
   It now says that figure compares the FEM solver at bs=128 against the
   operator at bs=1, and gives the matched figure of 1,215–1,297x.
2. New subsection "Operator vs. GPU-native FEM at identical batch sizes"
   with Tables 10a (operator latency by batch size), 10b (matched speed-up)
   and 10c (break-even). Numbered 10a-c rather than renumbering Tables
   11–14 and every cross-reference; the document already uses "Table 4a".
3. States both baselines rather than only the favourable one, and states
   two limitations: both sides measured at N=21 only, and B2's break-even
   is an order of magnitude later because its corrected recipe cost an
   order of magnitude more to train.

Verification: all 72 values across the three new tables recomputed from
`point3_inputs.json` and matched — one rounding slip caught this way (833.5
written as 834). The summary's three tables were then compared cell by cell
against the report's; 0 differences.

**Points 3 + 4 recomputed at MATCHED batch sizes (2026-08-27).** All inputs
and their exact Drive paths are recorded in
`Practical_Examples/omar_pfem/point3_inputs.json`; the calculation is
`omar_pfem/break_even_analysis.py`. Recomputing the *unmatched* comparison
from those inputs reproduces the figures already in the report (8,211 /
7,850 / 7,644 / 92,131 / 96,222 / 66,490 vs the recorded 8,211 / 7,847 /
7,644 / 92,165 / 96,275 / 66,523 — two exact, rest <0.06%), which is the
check that nothing was assumed.

Points 3 and 4 are one calculation. Doing 3 without 4 got it wrong: the
report compares GPU FEM at bs=128, where a GPU solver amortises its kernel
launches, against the Transolver at bs=1. Batching buys the network ~16x
(4.58–4.83 ms/sample at bs=1 down to 0.291–0.292 at bs=128), so the
comparison understated it by that factor.

**The speed-up in the report is wrong by ~16x. The break-even is not.**
| quantity | report (unmatched) | matched at bs=128 |
|---|---|---|
| speed-up vs GPU FEM | 73–80x | **1,215–1,297x** |
| break-even | 7,644 – 96,275 | **7,554 – 95,038** |
Break-even barely moves because FEM dominates the per-sample saving either
way; shrinking a term already worth ~1% of it changes little. The two are
easy to conflate — only the speed-up needs correcting.

**Break-even is not one number; it depends on the assumed batch size**, and
far more strongly than on anything else:
| case | bs=1 | bs=8 | bs=32 | bs=128 |
|---|---|---|---|---|
| B1 NH | 1,745 | 6,021 | 7,543 | 8,112 |
| B1 MR | 1,363 | 5,663 | 7,179 | 7,751 |
| B1 AB | **1,133** | 5,441 | 6,956 | 7,554 |
| B2 NH | 19,410 | 67,391 | 84,627 | 90,990 |
| B2 MR | 17,033 | 69,404 | 87,884 | 95,038 |
| B2 AB | 9,530 | 46,993 | 60,490 | 65,698 |
Full span 1,133 – 95,038, a factor of 84. The bs=128 figure assumes 128
problems are available to solve at once — but then FEM is batched too. In
the realistic deployment case, problems arriving one at a time, break-even
is **1,133 – 19,410**, not ~96,000. Quoting a break-even without naming the
batch size it assumes is misleading, so the report must state it.

Against the CPU baseline the report currently uses, break-even is 52 – 1,245
samples; both baselines should be shown side by side rather than only the
favourable one.
GPU-to-GPU speed-up of the trained operator over the GPU FEM solver is
73.3–79.7×. Per-case break-even vs GPU FEM: B1 NH 8,211 / B1 MR 7,847 /
B1 AB 7,644 / B2 NH 92,165 / B2 MR 96,275 / B2 AB 66,523. **Not yet written
into the report.**

**Point 8 — the answer to Timon's direct question:** the GPU FEM solver was
**written from scratch in PyTorch, not Tensormesh or any FEM library**. It
reuses the validated CPU solver's own force assembly and per-element material
evaluation, and gets the tangent by autodiff (`torch.func.hessian` + `vmap`)
rather than a hand-derived formula. See `gpu_fem_solver.py`'s docstring.
**Important limitation for point 8's second half:** that solver uses a DENSE
`torch.linalg.solve` per Newton step, which cannot reach millions of DOF (a
dense 3M×3M matrix is ~72 TB). The repo already has `matrix_free_solver.py`
— a matrix-free Newton-CG that never forms K and is what produced the
10M/40M-DOF references — so point 8 is a *timing sweep of the matrix-free
solver*, not new solver development.

**⚠️ Point 7 — affects the currently-running jobs.** Timon wants test
resolutions both **coarser and finer** than the training ones. The running
config trains on N=21,33 and tests on N=25,29,37,41,49 — **nothing is coarser
than 21**. Not a disaster: training (the expensive part) is unaffected, and
eval is a separate, cheap, re-runnable command on the same checkpoint, so the
fix is to re-run `eval` later with e.g. `--test_resolutions 13,17,25,29,41,49`.

**Point 5 progress (2026-08-26): `physical_quantities_eval.py` written.**
Computes, per held-out sample: displacement rel-L2 (for continuity with the
existing reports), H1 semi-norm, tangent-energy norm, PK1 stress per component
+ Frobenius + peak, and reaction forces on the constrained boundary (resultant,
nodal, max). Design notes:
- H1 and energy norms **reuse** `compute_l2_h1_errors_cross_order` and
  `compute_tangent_energy_error` from `high_dof_convergence_study.py`, so the
  operator is scored with the *same* norms as the Q4-vs-Q9 FE study; a
  prediction is just packaged into the same "solved field" dict shape.
- PK1 = dW/dF by autodiff of `materials_torch`'s energy density, so one code
  path covers all three materials (only Neo-Hookean has a closed-form PK1).
- Internal force assembled from the same Gauss-point stresses; on the
  constrained nodes external traction is zero in both benchmarks, so that IS
  the reaction. B1 fixes both components on `bottom_nodes`; B2's two radial
  edges are symmetry planes fixing one component each, handled separately.
- **Verified, not assumed.** Three bugs were caught during writing by checking
  the real signatures: `gauss_points_and_weights_physical` returns 6 values
  (not 4, and it hands back `N`/`dN_dX`, which removed a hacky interpolation
  workaround); `train_B2` uses the SAME symbol names as `train_B1` (the B2-
  specific names I first guessed do not exist); and `compute_tangent_energy_error`
  returns `tangent_energy_rel`, not `energy_rel`. Then smoke-tested: (a) u=0 →
  PK1 and reactions exactly 0; (b) uniform stretch → PK1 constant across all
  Gauss points and **matching a finite-difference of the energy density to
  4e-8 relative** (independent check of the autodiff path); (c) pred==ref →
  all errors exactly 0; (d) full end-to-end run on a toy checkpoint completes
  and writes its JSON.
- **Methodological caveat found while testing:** P12/P21 are near zero almost
  everywhere in both benchmarks, so their *relative* errors are huge even when
  absolute errors are negligible. Quote `P_rel_L2` (Frobenius) as the stress
  number; use `*_max_abs_err` for the shear components. Documented in the file.

---

Previous entry: 2026-08-26 (**v27 + summary finalized for sending to Timon**)

**v26 → v27 and summary finalization (2026-08-26).** Two things closed out:
1. **The Q9 CG caveat is now IN the report** (§4.4, end of the Q4-vs-Q9
   paragraph). It had been drafted much earlier but never actually inserted —
   verified by grepping v26, which contained no mention of it at all. Written
   in neutral numerical-methods language per the user's explicit instruction to
   state it factually without framing it as a defect ("reached its allotted
   iteration budget", not "failed"): 6 of 20 Newton iterations on the Q9 fine
   solve hit the CG budget before meeting cg_tol, while the Q4 fine solve and
   the entire small-N sweep for BOTH orders met it every time; Newton itself
   converged; the practical consequence is that this one comparison's norms are
   good to leading digits, not full precision; the margin is far below the
   two-order gap to 10⁻⁵ so the conclusion stands.
2. **The summary doc now opens with a plain-language narrative** ("Summary of
   what was done and what came out", 8 short paragraphs) and **closes with a
   Conclusion** (6 paragraphs: standing, the B2 result, the two conclusions that
   changed under scrutiny, efficiency, limitations stated plainly, remaining
   work). Previously it was tables+figures only, which the user found too bare
   to hand to an advisor.
Also drafted a short covering email for Timon (long version rejected as too
long; short version leads with the results and ends with two direct questions:
whether the Q4-vs-Q9 evidence is acceptable given 10⁻⁵ is met only in L2, and
whether B2's high-DOF study should be prioritised over finishing the five
resolution-invariance cases).

**Note on a fair challenge the user raised:** they asked whether numbers in the
tables that look suspiciously similar were fabricated or placeholder defaults.
Checked at full precision and they are not — the similarity is confined to
places where it is expected: (a) *settings* identical by construction (Newton
tol 1e-7, 30 iters, 10 load steps); (b) *analytical* FLOP counts, which depend
only on system size (solve FLOPs identical for all six; assembly identical
between MR and AB by construction of the hand-count); (c) GPU memory, which is
driven by the identical network/batch/mesh — and even there B1 reads
425.049088 while B2 reads 425.0496, i.e. genuinely different values that merely
round to the same 425.05, with the variation tracking *material* (extra
intermediate tensors in the energy computation) rather than geometry, which is
physically right. Every actually-measured quantity differs across all six cases.

Previous entry: 2026-08-26 (**full Drive audit completed → report v26**. User
correctly pushed back that the CODE should be saving its results; fixed that,
then had them run one Colab cell that dumped every remaining source file at
once. That closed the audit and caught 2 real errors — see below.)

**Full Drive audit + v25→v26 (2026-08-26).** `compare_q4_q9.py`'s `--out_json`
defaulted to `None`, so a multi-day Q4-vs-Q9 solve was run without it and its
numbers survived only as Colab stdout — **fixed: it now auto-saves to
`<checkpoint_dir>/q4_vs_q9_<geom>_<mat>_N<N>.json` unless you pass
`--out_json none`.** Then verified the remaining tables against a single
Colab dump of every source file. Newly verified this pass:
Table 4a all 6 rows (36 values ✅), Tables 1–6 mesh convergence (**193 ✅**),
Table 8 all 6 rows (24 ✅), Table 9 all 6 rows (30 ✅), Table 10 all 6 rows
(24 ✅), Table 11's 3 B1 OOD rows (9 ✅), Table 7 inference column (6).
**Cumulative: ~490 individual values checked against their Drive originals.**

**Two REAL errors found and fixed in v26 (not rounding):**
1. **Table 7, B2 × Neo-Hookean inference latency was 4.673 ms — the OLD
   pre-fix checkpoint** (`pfem_run/results/B2_neo_hookean/`). The corrected
   loss-normalized run's value is **4.809 ms**
   (`pfem_run/B2_accuracy_search/lossnorm/train/inference_latency.json`;
   confirmed as the right checkpoint by that case's own OOD report). B2×MR
   (4.908) and B2×AB (4.984) were already correctly taken from their lossnorm
   folders — only Neo-Hookean was stale. This is the same class of bug as the
   epoch/wall-clock staleness fixed in v25, just in a column we hadn't checked.
   Cascaded: inference speed-up range **5,545/5,546–12,575× → 5,387–12,575×**
   (3+2 = 5 occurrences across the report).
2. **FLOPs were reported as a single number but are material-dependent.**
   Assembly FLOPs/sample = **5.88×10⁷ for Neo-Hookean but 9.72×10⁷ for
   Mooney-Rivlin and Arruda-Boyce** (their autodiff-derived tangents cost
   ≈1.65× more per element); solve FLOPs ≈7.9×10⁵ for all. §4.2's sentence
   said only "approximately 5.88×10⁷" as if universal — now split by material,
   and the summary's Table 4b is a 3-row per-material table instead of one row.
Also corrected Table 10's B2×NH speed-up 73.04× → 73.05×.

**Everything else matched exactly.** The only remaining deviations are the 3
previously-noted ≤0.01 rounding-order artifacts, plus Table 10 speed-ups shown
to 4 significant figures (149.4/138.9/171.5/159.3 vs exact 149.41/138.88/
171.53/159.28) — display precision, not error.

Report file is now **v26**; summary doc rebuilt to match.

Previous entry: 2026-08-26 (**Drive-vs-summary number audit** — ~205 individual
values in the new summary doc checked directly against their original Drive
result files; see "Drive verification audit" below)

**Drive verification audit (2026-08-26):** Built a results-only summary doc
(`PFEM_Summary_Completed_Work.docx`, 22 tables + 16 figures, all pulled
programmatically from v25 so no value is retyped) and then verified its numbers
against the ORIGINAL Drive result files, not against v25. Verified live from
Drive this session:
| Table | Drive source file | values checked |
|---|---|---|
| Table 6 (batch-size sweep) | `pfem_run/.../fair_comparison_run_summaries.json` | 108 ✅ |
| Table 12 (zero-shot) | `pfem_run/zeroshot_B1_neo_hookean/zeroshot_eval_report.json` | 15 ✅ |
| Table 11, 3 B2 OOD rows | `B2_{material}_ood_report_corrected.json` ×3 | 9 ✅ |
| Table 4a, B1×NH row | `fem_cost_breakdown_B1_neo_hookean.json` | 7 ✅ |
| Table 4b (FLOPs) | same file (`flops_estimate`) | 2 ✅ |
| Table 10, B1×NH row | `gpu_fem_solver/B1_neo_hookean_timing.json` | 5 ✅ |
| Table 8, B1×NH row | `memory_profile_reruns/B1_neo_hookean/metrics_history.json` | 4 ✅ |
Plus, from Drive `metrics_history.json` files cached locally: Table 5 (24 ✅),
Table 7 opt-steps (6 ✅), Table 11 in-distribution column (6 ✅), Tables 13/14
(4 ✅). **Result: ~205 values checked, 202 exact matches.**
  The only 3 deviations are ≤0.01 last-digit **rounding-order** artifacts (a
  derived value computed from an already-rounded intermediate rather than from
  full precision), NOT data errors — confirmed numerically: B1×MR speed-up
  15.44 (table) vs 15.43 (exact), B2×MR speed-up 1.44 vs 1.45, B2×NH cost_full
  40.31 vs 40.30. Two of the three are what a reader dividing the table's own
  columns would get, so they were left as-is; flagged to the user.

**Two Drive findings worth remembering from that audit:**
1. **There are TWO different GPU-timing runs on Drive with slightly different
   numbers**: `gpu_fem_solver/{case}_timing.json` (2026-07-30/08-02) and
   `gpu_fem_timing_{B1,B2}.json` (2026-08-10, in the results folder). Table 10
   is sourced from the FORMER (B1×NH: 1651.6/477.9/381.3/354.6 — exact match);
   the latter gives 1649.9/479.2/382.6/354.8. Don't "correct" Table 10 against
   the wrong file. (Note `fem_cost_breakdown_*.json` ALSO contains GPU per-batch
   timings — a third set again — used for §4.2's cost breakdown, not Table 10.)
2. **No saved `q4_vs_q9_*.json` exists anywhere on Drive.** The Q4-vs-Q9
   comparison numbers in §4.4 (and summary Tables 6b/6c) live only as stdout in
   the Colab notebook `Untitled15.ipynb` (Drive id `1_IF1IbeqXWBlt3t0h9obOyNMUxkQJuX4`),
   which was read and verified earlier in this same session. That notebook has
   since grown to ~1 MB of accumulated output, so re-pulling it is impractical —
   **if this result is ever re-run, save its JSON to Drive properly.**

Still unverified against Drive (files located, just not pulled — same pattern
expected): Table 4a/8/10 rows 2–6, Table 9 (`validate_*.log`), Tables 1–6
mesh convergence, Table 11's 3 B1 OOD rows, and Table 7's inference-latency
column (~20 candidate `inference_latency.json` files, needs parent-folder
resolution; note the memory-profile rerun's copy reads 4.789 ms, which is a
3-epoch checkpoint and correctly NOT what Table 7 uses).

Previous update: 2026-08-26 (fixed a real 5x redundant-compute bug in
`resolution_invariance_zeroshot.py`'s eval command, discovered while
running the 3-notebook parallel plan for the 5 missing zero-shot cases —
see "Zero-shot eval redundant-solve fix" below)

**Run manifests + per-case notebooks (2026-08-27):** The user reported that the
3 running Colab notebooks had stopped twice, and that after restarting they
were back at the first cell with nothing saved. Investigating confirmed a real
data-loss bug and produced three changes:

1. **Sample generation was all-or-nothing.** In `cmd_train`, the
   `torch.save(...)` of the generated samples sat *outside* the
   `for N in train_resolutions` loop, and each resolution's 500 samples were
   built in a single list comprehension with no intermediate write. With
   generation measured at ~7.3 h for N=21 alone, any interruption before both
   resolutions finished discarded everything. Replaced with
   `_generate_samples_resumable()`, which writes `samples_cache_N{N}.pt` every
   `--gen_chunk` (default 25) samples via tmp-file + `os.replace`, and on
   restart prints `[resume] found X/400 ...` and continues from there. Seeds
   are unchanged (`10_000*N + i` train, `+500_000` val), so resumed data is
   bit-identical to the already-finished B1×NH run. Verified by truncating a
   cache mid-split and confirming the rerun resumed at the right index.

2. **Nothing recorded where a number came from.** New
   `omar_pfem/run_manifest.py` appends one record per run to
   `<out_dir>/run_manifest.json`: start/finish timestamps, duration, the exact
   command line, git commit + dirty flag, full environment (torch/CUDA/GPU/CPU
   — timings are meaningless without it), every argparse flag, the headline
   results, and every output file. Append-only, so re-running adds a record
   instead of erasing history. This is a direct response to two costs already
   paid: a stale pre-fix checkpoint's inference latency sat in the report for
   several revisions, and two GPU-timing runs with different numbers still
   coexist on Drive distinguishable only by folder name. Wired into the
   zero-shot `train` and `eval` stages, `physical_quantities_eval.py`, and
   `inference_latency_by_batch.py`. Both zero-shot stages verified end-to-end
   locally, including that a second run appends rather than overwrites.

3. **One notebook per case, and the slow phase separated from the fast one.**
   `Practical_Examples/zeroshot_notebooks/` holds five notebooks — B1×MR,
   B1×AB, B2×NH, B2×MR, B2×AB — generated by `make_zeroshot_notebooks.py` in
   that same directory (edit the generator, not the notebooks). Each is
   independent, so the five run in parallel on separate runtimes and one crash
   costs only its own case. Cells are split along the real cost structure:
   generation (hours, new `--stop_after_generation` flag) / training (minutes,
   already resumed from `train_state_latest.pt`) / eval / results. All output
   goes to `MyDrive/pfem_run/zeroshot_{case}/`, and cell 1 lists any *other*
   `*zeroshot*` folder on Drive that holds a sample cache, so an older run's
   hours aren't silently abandoned because the path changed.

   Note these notebooks also fold in point 7a: `--test_resolutions
   13,17,25,29,37,41,49` covers meshes both **coarser and finer** than the
   training pair (21, 33), which the earlier `25,29,37,41,49` did not.

**Zero-shot eval redundant-solve fix (2026-08-26):** While the 3-notebook
plan to run the 5 remaining resolution-invariance-zeroshot cases
(B1×mooney_rivlin/arruda_boyce, B2×neo_hookean/mooney_rivlin/arruda_boyce)
was already running, the user pasted a live Colab log showing the `train`
command's FEM data-generation step alone took 26,329s (~7.3h) just for
N=21's 500 samples (400 train + 100 val) — far longer than the ~1.7h
estimate quoted earlier, because that estimate came from `metrics_history`'s
`cumulative_wall_clock_s`, which (confirmed directly in the code, the timer
starts *after* the data-generation block) never included data-generation
time at all. Investigating further (user asked "is train or eval the
slower one?") found a genuine bug in `cmd_eval`: the common fine-mesh
reference solve at N=`fine_N` (N=101 by default, 10,201 nodes — expensive)
was being re-solved from scratch for every one of the 5 test resolutions,
even though it is the exact same physical problem (same seed) each time —
5x more expensive FEM solves than necessary. Fixed by caching each fine
solve by sample seed (persisted to
`fine_ref_cache_N{fine_N}.pt` next to `--out_json`, loaded on resume, same
pattern as the train command's own `samples_cache.pt`), so all 5 test
resolutions now share the same 20 fine-reference solves instead of doing
20x5=100. This was caught and fixed *before* any of the 5 running cases
reached their eval phase, so no wasted eval compute yet.
  Revised time expectation (previous estimate of "~2-2.5h/case" was wrong —
  it only counted the training loop, not data generation): total time per
  case is now expected to be dominated by two FEM-heavy phases — train's
  data generation (2 resolutions × 500 samples each, ~7.3h+ per resolution
  observed for N=21) and eval's now-fixed fine-reference solves (20 total
  instead of 100) — likely 10-20+ h/case depending on how N=33's generation
  and N=101's solve cost compare to N=21's. Not yet re-measured end-to-end
  post-fix; update this note once one of the 5 running cases finishes.

Previous update: 2026-08-26 (full read-through audit of v24 found 6 real internal
inconsistencies — stale numbers/text left over from earlier partial fixes — all
6 corrected in **v25**; see "v24→v25 correctness audit" below)

**v24→v25 correctness audit (2026-08-26):** User asked for a full read-through
of the whole report to check everything is correct and consistent, not just a
targeted check. Read all ~510 paragraphs/21 tables end-to-end and cross-checked
numbers against each other (not just against memory). Found and fixed 6 real
issues, all now in **v25**:
1. Abstract + Executive-Summary Table 1 row 7 still described the OLD,
   deprecated resolution-invariance method (10 independent trainings) even
   though §8.7/Table 12 already had the correct true zero-shot method and
   numbers (5.2–6.7%). Rewrote both to match §8.7.
2. **Table 5 (§8.1) and Table 7 (§8.3): B2 rows' Best-epoch/Final-epoch/
   Wall-clock/opt-steps figures were still the pre-accuracy-fix numbers**,
   while the accuracy numbers in the same rows were already the corrected
   ones — a real, provable internal contradiction (Table 5 said B2×NH final
   epoch 825 / 3257s, while §9.1's own text says 850 best-epoch / 32,244s for
   the same run). Pulled the authoritative `metrics_history.json` for all 3
   corrected B2 runs (`/tmp/fig1_data/B2_*.json`, the same files behind the
   Figure 1 regeneration) and recomputed exactly: B2×NH best=850/final=1050/
   wall=32,244s; B2×MR best=850/final=1050/wall=34,164s (already had this
   wall-clock right, only epoch/cost_epoch were off); B2×AB best=525/
   final=725/wall=24,847s (already correct, minor rounding refinement only).
   Recomputed cost_epoch/cost_full/speed-up from these exact numbers — the
   most consequential result: **B2×Neo-Hookean's corrected training recipe
   now costs MORE per sample (40.3s) than one native FEM solve (25.9s)** —
   speed-up flips from a stale 6.37× to the real 0.64×. This cascades into:
   the six-case wall-clock total (16,794s→**99,770s**, 4h40m→**27h43m**), the
   overall GPU-time total in §9.1 (8.2h→**31.2h**), and the break-even range
   (52–554→**52–1,245** new samples, both in §8.3 and the executive summary).
3. §10 Conclusion said "Table 11's OOD/degradation-factor columns remain
   pending" — but Table 11's own note already says that was resolved.
   Contradiction removed.
4. **§10 Conclusion silently omitted the B2 Q4-vs-Q9 ~10M/40M-DOF study
   (§4.4) from its "remaining items" list entirely**, even though §4.4 itself
   says it's still in progress for B2 — someone reading only the conclusion
   would think just 3 minor extensions remained. Added it as an explicit new
   bullet, marked as the one genuinely unfinished measurement (not a
   "scientifically motivated extension" like the other two).
5. "≈1,700×" inference-speedup-vs-CPU-FEM figure (in the executive summary
   and in §9.1) was a leftover from the old, already-replaced 8.0s/sample
   FEM placeholder (8.0/0.0046≈1,739). Real figure per Table 7 is
   **5,546–12,575×**. Fixed both occurrences.
6. Executive summary claimed "CPU-to-GPU FEM speed-up: 21–23× at large batch
   size" — contradicted Table 10's own data (71.7–171.5× at bs=128). Fixed
   (also fixed a matching "roughly 22×" repeat of the same stale figure in
   §9.1's discussion paragraph).

All 6 fixes verified by direct read-back of the saved .docx (python-docx) —
every changed cell/paragraph checked against its intended new text — and the
file passed the docx skill's XSD validator against v24 as baseline. (Note:
`soffice`/LibreOffice itself is currently broken in this sandbox — even the
unmodified v24 fails to convert to PDF — so this pass could not do a visual
PDF render; correctness was instead verified via python-docx content checks
and XSD validation only. Worth a visual spot-check next time soffice works.)

Report file: now **v25** (was v24), same handling as before — kept in the
scratchpad, delivered to the user via SendUserFile, not committed to this repo.

Previous update: 2026-08-25 (Point 1 closed for B1 in §4.4; Figures 8-10 (batch-size) and 11-16 (B2 diagnostics) embedded; Figure 1 regenerated with corrected B2 data; CG-convergence audit of every B1 Q4/Q9 solve behind Point 1 done — see notes below)

**CG-convergence audit of Point 1's B1 solves (2026-08-25):** User caught a
`cg_failures` field with nonzero values while poking at an old, unfinished
Drive file (`Q4_B1_neo_hookean_report_extended.json`, from 2026-08-18 — a
separate, still-incomplete attempt to extend the convergence sweep to
intermediate resolutions N=1001/1401 near the ~10M-DOF fine reference; both
rows there are 100% CG-failed and unusable, but this file was NEVER used in
the report). That raised a fair question about whether the numbers actually
IN the report (the small-N sweep rates, and the B1 Q4-vs-Q9 FAIL verdict at
the shared N=2236 fine references) might be similarly contaminated. Checked
every checkpoint's own embedded `stats` dict directly (the authoritative
source — `torch.load(path)['stats']`, not the summary JSONs, which don't
always carry it) for B1/neo_hookean:

| solve | N | cg_failures / newton_iters |
|---|---|---|
| Q4 fine reference | 2236 (~10M DOF) | 0 / 20 — clean |
| Q9 fine reference | 2236 (~10M DOF) | **6 / 20 — 30% failed** |
| Q4 coarse (small-N sweep) | 6, 11, 16, 21, 31, 41 | 0 / 20 at every N — clean |
| Q9 coarse (small-N sweep) | 6, 11, 16, 21, 31, 41 | 0 / 20 at every N — clean |

**Conclusion:** the small-N sweep convergence rates already in the report
(§4.4) are fully clean — no correction needed there. The B1 Q4-vs-Q9 FAIL
verdict against the advisor's 1e-5 criterion is very likely still correct
qualitatively (the observed differences exceed the threshold by ~2 orders
of magnitude, far more than a 30%-CG-failure margin could plausibly
explain) — but the *exact* reported numbers for that specific comparison
(H1-seminorm ~1.15e-3, energy ~9.61e-4) carry a known, unquantified error
margin from Q9's non-converged fine solve and should not be treated as
fully precise. Re-solving the Q9 fine reference at N=2236 cleanly (est.
24-48h GPU time, needs a fresh run since the existing checkpoint is marked
complete) would fix this exactly, but the user chose not to start that now
— **left as-is, not started**, tracked under the existing open Point 1 item
below (B1's ~10M-DOF Q9 reference specifically needs a clean re-solve
before its exact FAIL numbers can be called fully precise; the qualitative
FAIL conclusion itself is not in doubt).

**Figure 1 regeneration (v24):** `image1.png` (`all_cases_loss_curves.png`) was
stale — sourced from a file created 2026-07-27, weeks before the mid-August
B2 accuracy fix, so it still showed B2's old ~32% training curve while the
report's tables now show the corrected ~9-10% numbers. User confirmed
("نعم حدث الصور من الصور الي في درايف") to regenerate it. Rebuilt by
downloading all 6 `metrics_history.json` files from Drive (3 unchanged B1
originals + 3 corrected B2 `lossnorm` runs — the same adopted runs behind
Tables 5/7/11) and re-running the exact plotting logic from
`PFEM_Training_Colab.ipynb` cell 20 (2×3 grid, `semilogy(epochs, val_error)`,
same titles/dpi/figsize). New image is pixel-dimension-identical (2400×1350)
to the old one, so only `word/media/image1.png`'s bytes were swapped —
no XML/relationship changes needed for this one. Figures 2-7 were checked
and do NOT need updating (their captions' epoch/sid values already match
the corrected `lossnorm` folders).

Report file: `PFEM_Transolver_Report_vNN.docx` (latest: **v27**), kept in the
scratchpad, delivered to the user via SendUserFile after each update — not
committed to this repo.

Advisor: Prof. Timon Rabczuk (Bauhaus-Universität Weimar). Student: Omar Amro.

---

## Advisor's Round-4 feedback — 5 points

| # | Request (short) | Status |
|---|---|---|
| 1 | L2/H1/energy-norm error + convergence rate vs. a ~10M (or 1B) DOF reference; test Q4 vs. Q9; error ≤1e-4 in all norms | 🟡 **partial — B1 done, B2 still deferred; see below** |
| 2 | Exact CPU/GPU FEM cost breakdown (assembly/solve/IO, FLOPs, FP64, Newton/CG settings), GPU-native FEM comparison | ✅ done — §8.3–8.5, Tables 7–10 |
| 3 | Batch-size comparison with equal optimizer steps (not equal epochs) | ✅ done — §8.2, Table 6 |
| 4 | Exact mathematical definition of every reported error; investigate poor B2 accuracy | ✅ done — root cause, fix, and full propagation into Tables 5, 7, 11 for all 3 B2 materials |
| 5 | Resolution invariance = same trained model evaluated on unseen resolutions vs. a common fine reference (not 10 independently-trained networks) | ✅ done — §8.7 |

**Point 3 note (2026-08-25):** §8.2's text already claimed "Figures 8–10 plot
validation error against optimizer steps, processed samples, and wall-clock
time" (Table 6's equal-optimizer-step batch-size sweep), but the actual
images were never embedded in the .docx (confirmed: v20's `word/media/`
only had image1–7.png, matching Figures 1–7 from the training-curves
section — nothing for 8–10). Found the real plots on Drive
(`fair_comparison_vs_{opt_steps,processed_samples,wall_clock_s}.png`,
generated 2026-08-11, same run as `fair_comparison_run_summaries.json`
behind Table 6) and embedded them as Figures 8–10 in v22.

**B2 accuracy diagnostic plots: done as of v23.** Full image audit done —
user had me dump a complete Drive image listing (5034 PNG/JPG files
total, via a Colab `os.walk` script since the Drive connector kept
dropping in/out mid-session) to make sure nothing report-relevant was
missed. After categorizing all 5034: the overwhelming majority are
per-epoch training-visualization snapshots (`ux/uy/umag_combined_epoch*.png`,
repeated across dozens of run folders) and repeated `mesh_materials_forces_
{train,test}_sample0_sid0.png` sanity-check pairs — routine training
monitoring, not report content. Also found and explicitly excluded: 5
personal photos (`photo_*.jpg`, unrelated to the project, sitting in an
unrelated Drive folder), 15 `timoshenko_check_sample*.png` validation
images (no corresponding report section), and several superseded/failed
B2 trial folders (`lossnorm_lr5e3`, `B2_force_fix_ablation`,
`B2_neo_hookean_fixed`, `accuracy_diagnostics_B2_neo_hookean` pre-fix,
`B2_lr_test_2e-4`, `B2_force_fix_pilot_check`) and old superseded studies
(`resolution_study/`, `screening_B1_neo_hookean/`,
`screening_extended_B1_neo_hookean/`, `memory_profile_reruns/`) — all
replaced by later work already reflected in the report's tables.
  The only genuinely new, report-relevant images: the 6 diagnostic plots
  (`error_vs_parameters.png` + `worst_sample_error_contour.png`, 3
  material pairs) from the 3 *adopted* `lossnorm` trials —
  `pfem_run/B2_accuracy_search/lossnorm/diagnostics/` (Neo-Hookean, 9.11%),
  `pfem_run/B2_accuracy_search_mooney_rivlin/lossnorm/diagnostics/`
  (Mooney-Rivlin, 7.28%), `pfem_run/B2_accuracy_search_arruda_boyce/lossnorm/
  diagnostics/` (Arruda-Boyce, 9.81%). Embedded as new Figures 11–16 right
  after Table 14 in §9.1.

Round-3 items not repeated in Round 4 (Omar's Aug-3 reply claimed these were
addressed; not re-raised by Timon since):
- OOD evaluation (different material/load ranges) — ✅ confirmed done, §8.6 / Table 11, all 6 cases.
- Allocated-vs-reserved / peak GPU memory clarification — ✅ **confirmed done, 2026-08-26**
  (re-checked directly against v24's report text, not just memory): §8.4 reports all three
  quantities for all 6 cases — `torch.cuda.max_memory_allocated()` (≈425 MB, identical
  across cases), `torch.cuda.max_memory_reserved()` (≈680 MB), and device-level peak via
  `torch.cuda.mem_get_info()` (≈1.2 GB, the nvidia-smi-equivalent figure), with explicit text
  stating the first two are internal PyTorch-allocator statistics and do NOT match what
  nvidia-smi/the driver would report, while the third does.

---

## Point 1 detail (10M-DOF convergence, Q4 vs Q9) — the open item

- **B1 × Neo-Hookean, Q4**: ✅ done. Reference at N=2236 (n_dof=9,999,392≈10⁷,
  wall-clock 74,871.6s≈20.8h). Test points N=51→701 vs. that reference:
  L2 rate p=1.47, H1 rate p=0.72, energy rate p=0.78. L2 already satisfies
  the 1e-4 target (reaches 1.0e-5 by N=701); H1 and energy do not
  (~2–3×10⁻³ at N=701) — closing that gap by further test-mesh refinement
  alone would need N≈30,000–75,000, which exceeds the reference mesh itself
  and is not achievable within this project's compute budget. Written into
  the report as new **§4.4** (v16). Source data:
  `Google Drive: pfem_ckpt/Q4_B1_neo_hookean_report.json`.
  - Extended test points N=1001, N=1401 (vs. the same N=2236 reference)
    **finished** and are now in the report (v18, Table 6a): L2 continues
    to improve (down to 2.3e-6 at N=1401) but H1/energy are still above
    the 1e-4 target and improving slowly (H1 rel=1.63e-3, energy
    rel=7.82e-4 at N=1401). Combined 7-point least-squares fit: L2 p=1.58,
    H1 p=0.73, energy p=0.87. Two caveats flagged in the report: (1) CG
    hit its 2000-iter cap without reaching cg_tol on every Newton
    iteration at both N=1001 and N=1401 (Newton itself still converged,
    but adds some non-discretization error); (2) fine_N=2236 is only
    1.6–2.2× these two N values (recommended 4×), so the fitted rate,
    especially H1's, is likely a mild underestimate. Source:
    `Google Drive: pfem_run/Q4_B1_neo_hookean_report_extended.json`.
  - Also fixed a real gap in `high_dof_convergence_study.py`: the
    per-N coarse solve had no checkpointing or progress heartbeat (only
    the fine reference did), so a multi-hour coarse solve was invisible
    and would restart from zero on any interruption. Now wired through
    the same `--checkpoint_dir`/`--cg_progress_every` flags.
- **B1 × Neo-Hookean, Q9**: ✅ **done**. The ~40M-DOF fine reference (39,979,682
  DOF at N=2236 — ~4× Q4's 9,999,392 at the same N, from Q9's extra
  per-element edge/center nodes) finished after the multi-day CG effort
  noted below. `high_dof_convergence_study.py --orders Q4,Q9 --resolutions
  6,11,16,21,31,41 --fine_N 2236` then gave least-squares fitted convergence
  rates of **L2 p=1.57, H1 p=0.76, energy p=0.75** for Q9 vs. **L2 p=1.39,
  H1 p=0.72, energy p=0.71** for Q4 over the same six resolutions — Q9
  converges faster in all three norms, as FE theory predicts for a
  biquadratic vs. bilinear element. Written into the report §4.4 (v21).
  - **Direct Q4-vs-Q9 fine-solution agreement check** (`compare_q4_q9.py`,
    the advisor's explicit "difference smaller than 1e-5 in all norms"
    request, comparing the two *already-solved* fine fields directly
    against each other, no new solve): **FAIL**. L2 relative difference
    ≈2.06e-6 (meets the 1e-5 target), but H1-seminorm ≈1.11e-3/1.15e-3 and
    tangent-energy norm ≈9.61e-4/3.10e-4 (Q4-domain/Q9-domain) each miss it
    by ~2 orders of magnitude (worst = 1.15e-3). Attributed to Q4's own
    fine reference still carrying non-negligible discretization error
    relative to Q9's richer mesh at the same N=2236, not to a solver bug.
    Result file: `q4_vs_q9_B1_neo_hookean.json`. Also written into §4.4
    (v21).
  - Checkpoints used: `pfem_ckpt/fine_B1_neo_hookean_{Q4,Q9}_N2236.pt`.
- **B2 (both Q4 and Q9)**: ❌ **not started at all**. Zero files/results
  exist for a ~10M-DOF-referenced B2 study. Explicitly deprioritized/deferred
  by the user on 2026-08-15 ("لاحقًا" — do later, not now).

---

## Point 4 detail (B2 accuracy) — the other open item

Root cause (documented in report §9.1): B2's boundary force was a raw
pressure×direction approximation, 13–16× larger in magnitude than the
FEM-consistent nodal force. Fixing the force alone made things *worse*
(32.46%→94.08%) because the smaller, correct force gives too weak a
gradient signal in Π=U−W. Fix: normalize the training loss (not the
physics) by each sample's own boundary-force scale (`--loss_force_norm 1`
in `train_B2.py`) — provably preserves the true minimizer, restores
gradient conditioning.

- **B2 × Neo-Hookean**: ✅ resolved. 9.11% (vs. 32.46% original, 94.08%
  force-fix-alone regression). Confirmed at full production scale on Colab.
- **B2 × Mooney-Rivlin**: ✅ resolved. **7.28%** — reached on trial 1
  (`lossnorm`, same recipe as Neo-Hookean), no escalation needed. Best of
  all three B2 materials so far. Checkpoint:
  `pfem_run/B2_accuracy_search_mooney_rivlin/lossnorm/train/model_best.pt`.
  Full record: `pfem_run/B2_accuracy_search_mooney_rivlin/search_summary.json`.
- **B2 × Arruda-Boyce**: ✅ resolved (adopted). **9.81%** (trial 1, `lossnorm`
  — the same recipe as Neo-Hookean/Mooney-Rivlin). This technically missed
  the search tool's self-imposed <9.00% target (chosen to match B1's own
  9.59%, not an explicit number from the advisor — Timon only asked to
  "investigate the B2 accuracy gap," no numeric threshold), so the search
  auto-escalated to further trials:
  - Trial 2 (`lossnorm_lr5e3`, lr=0.005): failed badly, 76.9% (higher LR
    broke training).
  - Trial 3 (`lossnorm_graded`, r_grading=2.5): also trending badly
    (~83%+ at epoch 71) before being manually stopped.
  Decision (2026-08-15): adopt trial 1's 9.81% as final — it's close to
  the target and consistent with the other two B2 materials (9.11%,
  7.28%) and B1 itself (9.59%). Trial 3's Colab job was stopped manually;
  no further search needed unless a stricter target is requested later.
  Checkpoint: `pfem_run/B2_accuracy_search_arruda_boyce/lossnorm/train/model_best.pt`.
- All three B2 materials now resolved (9.11%, 7.28%, 9.81%) and
  **propagated into the report as of v17**: Table 5's "Best val. error"
  column and Table 11's "In-distribution val. err." column both updated;
  new **Table 14** added in §9.1 summarizing all three; §9.1's NOTE and
  the §10 bullet rewritten; the B1-vs-B2 narrative paragraph after Table 5
  rewritten (B2 is no longer "harder due to geometry" — all six cases now
  sit in the same 7–11% range).
  - **Table 7 (training cost): done as of v19.** Found the real
    `train.log` files for the corrected (`lossnorm`) Mooney-Rivlin and
    Arruda-Boyce runs on Drive (`pfem_run/B2_accuracy_search_{material}/lossnorm/train/train.log`)
    — same production-scale recipe as Neo-Hookean, not a separate
    confirmation run. Updated Table 7's two B2 rows: Mooney-Rivlin
    (840,000 opt. steps, cost_epoch=38.85ms, cost_full=42.71s,
    speed-up=1.44×, inference=4.908ms) and Arruda-Boyce (580,000 opt.
    steps, cost_epoch=42.89ms, cost_full=31.06s, speed-up=1.94×,
    inference=4.984ms). Native FEM cost columns unchanged (unaffected by
    the loss-fix). Propagated the resulting range changes into §8.3's
    narrative paragraph and the summary bullet (cost_full range
    2.80–4.07→3.48–42.71 s/sample; total wall-clock 2,237–3,257→
    2,784–34,164 s; break-even 36–126→52–554 new samples; inference
    speed-up 5,545–12,594→5,545–12,575×). Added a NOTE under Table 7
    flagging that these two rows now reflect the corrected recipe.
  - **Table 11's OOD / degradation-factor columns: done as of v20.**
    Confirmed via exhaustive Drive search that no OOD evaluation had been
    run on the 3 corrected checkpoints (existing `*_ood_report.json`
    files were all dated 2026-07-30, before the 2026-08-15 fix). Ran the
    3-step OOD pipeline (`data_generate_B2.py` → `convert_B2_quad.py` →
    `evaluate_ood.py`) on Colab for all three corrected B2 checkpoints
    (same OOD distribution shift as before: E_mean 1000→1500,
    p_mean 5.0→9.0). Results (`B2_{material}_ood_report_corrected.json`
    on Drive):
    - Neo-Hookean: ID=9.11%, OOD=48.25%, degradation=5.30×
    - Mooney-Rivlin: ID=7.28%, OOD=40.60%, degradation=5.58×
    - Arruda-Boyce: ID=9.81%, OOD=38.68%, degradation=3.94×
    Updated Table 11's three B2 rows, its NOTE (the old one flagged the
    OOD columns as stale — now resolved), and two narrative
    passages that had described the *old* B2 OOD numbers (an executive
    summary bullet, and the §8.6 discussion paragraph after Table 11) —
    both previously said B2 degrades "far less" than B1 (1.5–1.6×,
    an artifact of pairing corrected in-distribution accuracy with a
    stale pre-fix OOD run); now correctly say all six cases fall in a
    comparable 3.94–5.58× band. Also added checkpoint/resume support to
    `data_generate_B2.py`'s sample-generation loop (writes a
    `generation_progress.json` manifest + flushes the HDF5 file after
    every sample) since this OOD generation step has no such safety net
    before — matches the project's established checkpoint-everything
    convention.

---

## A genuinely useful discovery worth remembering

Google Drive (`pfem_ckpt/`, `pfem_run/`) already contained substantial
finished work from earlier sessions that was **not yet reflected in the
report** until this pass (v16) started incorporating it — in particular the
good Q4 N=51→701 convergence study. **Before assuming something needs to be
computed from scratch, search Drive first** (`pfem_ckpt`, `pfem_run`,
`pfem_data` folders under the user's Drive) — it may already exist.
Google Drive connector in chat sometimes shows `enabledInChat: false` even
when `connected: true`; this is a per-conversation toggle the user has to
flip from their client's connector settings — retrying the tool call does
not fix it, only re-checking after the user says they've toggled it does.

## Known pre-existing report issue (not caused by this project's edits, not yet fixed)

Table numbering in the report is **not globally unique** — e.g. "Table 3",
"Table 4", "Table 5", and "Table 7" each appear twice, once in §4.3's mesh-
convergence tables and again in §5–9's tables. This predates this project's
work. The new §4.4 table was deliberately labeled "Table 6a" (not "Table 7")
to avoid adding a *third* collision. A full renumbering pass (checking every
in-text "see Table N" cross-reference too) has not been done — flag to the
user if it becomes worth fixing.

---

## Environment / tooling notes

- Repo: `suhibamro/omar` (GitHub), branch `claude/claude-code-question-d307wp`.
  Local clone: `/home/user/OMAR`.
- Colab pattern used throughout: `pip install -q einops timm h5py jax tqdm`
  → `git clone -b claude/claude-code-question-d307wp
  https://github.com/SUHIBAMRO/OMAR.git /content/OMAR` → mount Drive →
  `cd /content/OMAR/Practical_Examples && python -u -m omar_pfem.<module>`.
- Everything long-running is checkpointed to
  `/content/drive/MyDrive/pfem_ckpt` or `pfem_run/...` and resumable by
  re-running the exact same command.
- User's GPU: A100, 80GB (per email to Timon, 2026-08-05).
