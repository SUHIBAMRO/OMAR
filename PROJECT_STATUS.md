# PFEM / Transolver Project — Status Tracker

**Read this file FIRST at the start of any new conversation about this project.**
It is the single source of truth for where things stand — more reliable than
chat history, which resets between sessions. Update it whenever a task
finishes or a new one starts.

Last updated: 2026-08-31 (Drive audit). **Read the master table immediately below first;
everything after it is detail.**

---

# ⛔ SETTLED — do not reopen, and do not list as "remaining work"

**Read this before ranking any priority or answering "what is left?".** Twice
now a session has read a *scientific caveat* in the report and reported it as
*unfinished work*. The report hedges properly; that is not the same as a gap.
This list is the authority.

| item | status | why it looks open but is not |
|---|---|---|
| **B2 mesh-convergence at ~10M/40M DOF** (§4.4) | **CANCELLED by Omar 2026-08-27** — *"خلص ملغي ما بدنا ياه"* | §4.4 still ends with a "NOTE — pending" line and §10 still lists it. **Both are stale and should be rewritten.** It was never among Timon's requests |
| **Data-driven vs physics-informed, other cases** (§8.9) | **COMPLETE** — point 7b, Table 21 | §8.9 closes with *"The comparison covers one case"*, which is a caveat, not a to-do. Timon's instruction was *"I'd start with one specific problem such as B1-Neo Hookean. **Based on the results, we can decide then.**"* One case was the instruction; extending is a decision that was never taken |

**The rule:** a sentence in the report that names a limitation is the report
being honest. Only this file says what is actually outstanding.

---

# MASTER TABLE — where every item stands

Current artefacts: report **v37**, summary mirrored (v10), branch
`claude/claude-code-question-d307wp`.

**Nothing measured is unwritten.** Point 7b's 2×2 is complete and in §8.9;
point 9's MMS is complete, three-way, and now across three meshes in
§8.11 (Tables 22–24b). Everything
measured is committed as JSON under `omar_pfem/point{2,5,6,7b,8,9}_results/`
AND written into both documents, verified by re-reading the .docx and
comparing cell by cell against the JSON.

## ✅ Done and in the report

| Item | Where |
|---|---|
| Mesh convergence, 6 cases, N=6→51 | §4.3, Tables 1–6 |
| Convergence vs ~10M-DOF reference, Q4 vs Q9 (B1 only) | §4.4, Table 6a |
| Batch-size sweep, equal optimizer steps | §8.2, Table 6 |
| GPU memory, three senses | §8.4, Table 8 |
| Measured native FEM cost + training cost | §4.2, §8.3, Tables 4a/7. **Table 4b is in the SUMMARY only, not the report** — see the numbering warning below |
| GPU-native FEM solver + machine-precision validation | §8.5, Table 9 |
| **R5-3** break-even vs GPU FEM | Table 10c |
| **R6** break-even, CPU and GPU side by side | **Table 10d** |
| **R5-4** identical batch sizes | Tables 10a/10b |
| **R5-5** physical quantities (H1, energy, stress, reactions) | §8.8, Tables 15–17 |
| **R5-2** Pareto, B1 × Neo-Hookean | §8.7, Table 18 |
| **R5-8a** "Tensormesh?" — written from scratch in PyTorch | §8.5 |
| B2 accuracy regression: root cause + fix (32.46% → 9.11%) | §9.1 |
| Exact definition of every reported error | §7.1 |
| Zero-shot resolution invariance, B1 × Neo-Hookean | §8.7, Table 12 |
| **R5-7b** physics-informed vs data-driven, the complete 2×2 | §8.9, **Table 21** |
| **R5-9** MMS, Q4 and Q9 against an analytic solution | §8.11, **Tables 22–23** |
| **R5-9** MMS, the operator third — the three-way is complete | §8.11, **Table 24** |
| **R6-1** progressive OOD: material vs loading, 0→3σ | §8.6, **Table 19** |
| **R6-1b** normalization tested as a mitigation — it does not work | §8.6, **Table 19a** |
| **R5-1 / R5-7a** zero-shot, three B1 materials, 7 resolutions | §8.7, **Table 12 (revised)** |
| **R5-8b** the CG counters and the corrected cost analysis | §8.5, **Table 20a** |
| **R5-8b** GPU-FEM scaling sweep, 0.02→3.93M DOF + cost breakdown | §8.5, **Table 20** |
| **R5-8b** CG allowed to converge — Table 20 understates by +28%/+18% | §8.5, **Table 20b** |
| **R5-9** the MMS operator across three meshes — it does not converge | §8.11, **Tables 24a/24b** |

## 🔵 Run, recorded, NOT yet in the report

| Item | State |
|---|---|
| **R5-1 / R5-7a** the three B2 zero-shot cases | **UNRESOLVED.** The old results are INVALID (mesh-dependent load). Caches repaired and verified, all three retrained with `loss_force_norm` — and they still sit at ~1.0, which is what predicting zero scores. Batch size ruled out. `point7a_results/B2_zeroshot_retrain_status.json`. v37 §8.7 states this plainly and quotes no B2 number |

All round-6 notebooks are self-contained, save to Drive incrementally, and
resume on re-run. All 12 repo notebooks pass `check_notebooks.py`.

## 🟡 Partial

| Item | State |
|---|---|
| **R5-1 / R5-7a** zero-shot, 6 cases | **3 of 6 valid** (B1×NH, B1×MR, B1×AB), all recorded in `point7a_results/`. **The three B2 cases are INVALID** — trained on a load overstated by a mesh-DEPENDENT factor (13.3× at N=21, 20.9× at N=33), giving relative errors of 8.0–14.5. Caches repaired for B2×MR and B2×AB on 2026-08-29 and the bad models deleted; B2×NH not yet confirmed. See `point7a_results/INVALID_B2_zeroshot.json` |
| ⚠️ **the two zero-shot protocols are not the same study** | Table 12 (B1×Neo-Hookean) trained at **N=21 only** and evaluated 5 resolutions, all FINER. The five new notebooks train at **N=21 and 33** and evaluate 7, including two COARSER (13, 17) — which is what round-5 item 7 actually asked for. So B1×MR cannot be added as another row of Table 12: material and protocol differ at once. Either B1×NH is re-run under the new protocol, or the new cases get their own table |
| **R5-2** Pareto, remaining cases | **B1×MR and B1×AB are unblocked now** — their checkpoints are valid; cell at `zeroshot_notebooks/cell_pareto_remaining_B1.py`, ~2 h per case (measured 1 h 54 m / 6 h 24 m on B1×NH). The three B2 cases stay blocked until their data is repaired and retrained |

## ⬜ Not started

| Item | Blocker |
|---|---|
| **R6** open-source the GPU-FEM + benchmark vs Tensormesh | Needs Omar's decision: separate repo? license? how much documentation? |
| Send Timon the correction + the B1×NH Pareto result | Drafted in the reading of the round-6 email; not sent |

## 🚫 Cancelled by Omar — DO NOT PROPOSE THESE AGAIN

**B2 mesh-convergence study** (the ~10M/40M-DOF Q4-vs-Q9 study of §4.4, for
the B2 geometry) · Tables 13/14 left as they are.

Cancelled 2026-08-27: *"خلص ملغي ما بدنا ياه"*. **It is not among Timon's
requests** — a leftover from an earlier round.

**⚠️ I proposed restarting it on 2026-08-31 and ranked it the top priority.**
That was wrong; Omar caught it. I had read §10's "remaining items" list and
§4.4's own "pending" note and had not opened this file, which exists precisely
so that does not happen. **Read this section before ranking any priority.**

**And two lines in the report still contradict this decision** and should be
rewritten so a reader does not think the study is coming:
* end of §4.4: *"NOTE — pending: this same ~10M-DOF-referenced convergence
  study for the B2 geometry (both element orders) remains in progress"*
* §10's remaining-items list carries the same item.

Both should say the study is deliberately confined to B1.

---

## The correction Timon needs

He is working from "approximately 7,600–96,000 samples" for the GPU
break-even, because that is what our email said. That range is the
**batch-size-128 column of Table 10c alone** — the least favourable of four.
The full range is **1,133–95,038**, and **1,133–19,410** at batch size 1,
which is the deployment case. His lower bound is 6.7× too pessimistic, and he
explicitly said the figure "clarifies where the neural operator is useful",
so it is shaping his judgement of the work.

## How to rebuild the documents

Builders live in `Practical_Examples/report_builders/`, each reading the
previous version, so the chain is v27 → v28 → v29 → v30 → v31 → v32 → v33.
**Run them from `/tmp`**, which is where the .docx files live:

    make_v28.py       matched batch sizes (Tables 10a-c)
    make_v29.py       physical quantities (Tables 15-17) + §10 qualification
    make_v30.py       break-even side by side (Table 10d)
    make_v31.py       Pareto (Table 18)
    make_v32.py       OOD attribution (Table 19)
    make_v33.py       GPU-FEM scaling sweep (Table 20)
    make_summary_v3.py … make_summary_v6.py   the parallel summary
                       (each expects a PFEM_Summary_Completed_Work.pre_vN.docx
                        copy of the current summary as its input)

`point5_tables.py` and `pareto_table.py` build their tables from the
committed JSONs and are imported by BOTH the report and summary builders, so
the two documents cannot disagree. The builders assert their own cross-case
claims before writing them, and `make_v33.py` additionally parses Table 4a
back out of the source .docx so the numbers it quotes from the rest of the
report are the report's own.

**That mechanism has now caught six false statements plus one code bug.** The
four earlier ones are listed in the sections below; the two from v33 were:

* the draft quoted **3,215 µs/DOF** for N=501 where the table prints the run's
  own **3,219** (`solve_s` is transcribed rounded, so dividing it by `n_dof`
  disagrees with the run's printed `us_per_dof` by up to 4 µs/DOF). **Quote
  the `us_per_dof` field, not a value re-derived from `solve_s`.**
* the draft cited **"the report's Table 4b"** for the CPU assembly-versus-solve
  split, at a factor of **74×**. See the numbering warning immediately below —
  the citation was wrong and the quantity was the wrong kind.

A third error was caught by hand while checking the same paragraph: the draft
said the *B2 geometry* costs ~2× more to assemble. It does not — B2 × NH is
within 2% of B1 × NH. The ~2× is a **material** effect: Neo-Hookean has an
analytic PK1 and tangent (`omar_pfem/data/materials.py`), while Mooney-Rivlin
and Arruda-Boyce use `jax.jacfwd(jax.grad(...))`
(`omar_pfem/data/material_models_jax.py`), costing 2.1–2.4× per Table 4a.

### Point 9 (MMS) — COMPLETE, all three legs (2026-08-28)

`omar_pfem/mms_study.py`, results in `omar_pfem/point9_results/`.

**The fork Timon left open is resolved as BODY FORCE.** A body-force-free
exact solution on this domain is a homogeneous deformation, which Q4
reproduces to machine precision — the study would measure round-off and
distinguish nothing. This was decided here, not confirmed by him, and it is
the first thing to raise if he wants the study shaped differently.

u* = 0.05·(sin πx sin πy, 0.7 sin πx sin πy), which **vanishes on the whole
boundary**, so homogeneous Dirichlet is exact and the shared solver needed no
inhomogeneous-Dirichlet support. The body force b = −Div P is derived by
nested autodiff and checked against a central finite difference (1.8e-10).

**Every convergence rate came out at its theoretical value**, which is what
validates the whole chain — a body force wrong by a sign or a factor would
collapse them:

| | L2 | H1 semi | stress | energy |
|---|---|---|---|---|
| Q4 | 1.98 (2) | 1.00 (1) | 1.00 (1) | 1.98 (2) |
| Q9 | 3.02 (3) | 2.01 (2) | 1.98 (2) | 3.98 (4) |

Also checked: the reported errors are **discretization** error, not algebraic
error — at Q4 N=9 they are identical to 12 significant digits across cg_tol
1e-6, 1e-8 and 1e-10.

**Q9 wins decisively at equal DOF**: 4.0× lower L2 at 162 DOF, 8.2× at 578.

**The operator third is MEASURED** — `mms_operator_B1_neo_hookean.json`,
report Table 24. `Round6_MMS_Operator` on an A100: N=17 (578 DOF), 16,000
optimizer steps, 8.2 min, a 64-member family, **no labels**. Existing
checkpoints could not be reused (no body-force term in Π, no body-force
input channel, wrong Dirichlet set), so it is a new physics-informed model
on the manufactured family: same architecture, same Adam recipe, scored by
`mms_study`'s own error routine so all three numbers are comparable.

| method | L2 | H1 semi | stress | energy |
|---|---|---|---|---|
| Q4 (same mesh, 578 DOF) | 3.403e-03 | 5.666e-02 | 5.724e-02 | 3.191e-03 |
| Q9 (same N, 2,178 DOF) | 5.163e-05 | 1.438e-03 | 1.495e-03 | 2.088e-06 |
| operator (578 DOF) | 8.238e-03 | 5.831e-02 | 5.886e-02 | 9.914e-03 |

**operator/Q4 = 2.42× in L2 — the ceiling holds.** The FEM rows are bit-for-bit
Table 22's N=17 rows; `make_v35.py` asserts that, so the two tables cannot
drift apart inside one document.

**The finding is that the four norms disagree**: 1.03× in H1 and 1.03× in
stress — effectively at the Q4 optimum — against 2.42× in L2 and 3.11× in
energy. That **inverts the usual ordering**, where L2 is the forgiving norm.
The loss is built from the deformation gradient, so strain and stress are
what it constrains hardest and the displacement is pinned only through them.
The same inversion appears in the independent N=9 CPU demo (1.35× vs 4.71×),
so it is not an artefact of one run. Stated plainly in §8.11: for a
physics-informed operator, an L2 displacement error overstates how wrong the
mechanics are.

What remains is **optimization** error, not discretization error: best
held-out L2 went 1.429e-02 → 8.826e-03 over the second half of training, a
further 38%, still falling slowly. Reported number is the best checkpoint
(epoch 1900), not the last — single-epoch scores span a factor of 12 over the
last twenty validations.

⚠️ **Provenance**: the run wrote its JSON to Drive and only stdout came back.
`point9_results/transcribe_operator_run.py` parses that stdout rather than
anyone retyping it, and the JSON's `provenance` block records that operator
values carry **printed** precision (4 s.f.) and which fields are **absent
rather than guessed** (the FEM refs' wall clocks; the training wall clock in
seconds — the cell printed 8.2 min only). The FEM references are taken at
full precision from `mms_B1_neo_hookean.json` after checking the run's
printed values agree.

Still missing from point 9: the operator at **more than one mesh** (so it has
no convergence rate of its own and is absent from Table 23 — that is a
training run per refinement, not a solve per refinement), a **common cost
axis** for GPU training against CPU Newton solves, and more than **one
geometry, one material, one scored member** (α=0.05, β=0.7).

**⚠️ The ceiling must be quoted with the result.** The operator minimizes the
*same* discrete functional over the *same* Q4 space as the Q4 solver, and the
minimizer of that functional **is** the Q4 solution. The operator therefore
**cannot beat Q4 at the same mesh** — that is arithmetic, not a finding.
Report the ratio **operator/Q4**: 1.0 means the network has fully solved the
variational problem. A ratio below 1.0 is a bug, not a win, and the runner
says so.

The functional is proved correct by `test_mms_operator.py`, not assumed: at
N=9 the Q4 solution lies in the operator's constrained space, the interpolant
of u* does not beat it, 36 admissible perturbations all raise Π, the excess
grows quadratically (ratio 4.000), and the deliberately wrong scaling —
dividing W by `len(top_edges)`, which is what train_B1 does for its traction
work and the natural mistake here — moves the minimum from scale 1.000 to
0.125, exactly 1/8 for a load weakened 8×. The test can fail, which is what
makes it worth running.

Labels are free in MMS (u* is analytic) but are **not used in training**; the
loss is the energy. They are only the scoring truth.

**Do not compare the training Π against the FEM solve's Π.** The training log
prints the mean of Π over the training *family*, whose members have genuinely
different energies because Π scales with the amplitude α. The FEM number is
one member (α=0.05, β=0.7). A short CPU run reached a family-mean Π of −8.99
while that member's Q4 minimum is −7.999, which looks like the network
beating the variational minimum and is nothing of the kind. The column is
labelled `trainPi(family mean)` for this reason. **The honest progress signal
is L2**, which on that run fell 1.71 → 0.204 over 300 epochs — converging, but
far from a reportable number; the production run is N=17 for 2000 epochs.

### 🛑 The B2 zero-shot trainer was missing `loss_force_norm` — caught mid-run

Found 2026-08-29 while the first B2 retrain was running, by checking §9.1
before waiting for the result.

**§9.1's documented root cause, in its own words**: *"Fixing the force alone
made things worse (32.46% → 94.08%) because the smaller, correct force gives
too weak a gradient signal in Π = U − W. Fix: normalize the training loss (not
the physics) by each sample's own boundary-force scale (`--loss_force_norm 1`
in `train_B2.py`)."*

**That is exactly what we had just done**: repaired the force so it is 13–21×
smaller and correct, then retrained — and
`resolution_invariance_zeroshot.py` **had no `loss_force_norm` at all**, while
`train_B2.py` has it and defaults it **on**.

The run was already reproducing the known regression when it was stopped:

| epoch | B2×NH retrain | B1×MR for comparison |
|---|---|---|
| 25 | **0.9587** | 0.4775 |
| 50 | **1.1298** | 0.4563 |

0.94–1.13 against the documented 94.08%. Same number.

**Fix applied**: the option is ported from `train_B2.py`, and its default is
**resolved from the geometry** rather than fixed — B2 → 1, B1 → 0 — and
printed at the top of every run. B1 stays at 0 because its force never had the
defect and `train_B1.py` has no such option, so the three completed B1
zero-shot cases stay reproducible. A B2 run with the scaling off now prints a
warning naming the 94.08% regression.

Checked, not assumed: dividing Π by a per-sample constant independent of uv
leaves the minimizer unchanged (`argmin c·f = argmin f`), verified numerically
alongside the tensor shapes.

**Why this nearly cost hours**: the knowledge lived in §9.1 and in
`train_B2.py`'s default, and nowhere in the path a zero-shot B2 run takes. It
now lives in the code that needs it.

### ✅ CG converged, and the prediction held to 0.4% — §8.5's model is verified

`point8_results/gpu_fem_cg_converged_B1_neo_hookean.json`, 2 h 3 m on an A100.
Identical settings to the point-8 sweep except `cg_max_iter` 2000 → 8000. **The
prediction was printed before the run**, so it could not be fitted afterwards.

| N | Newton | CG iters | failures | CG/Newton | predicted | error |
|---|---|---|---|---|---|---|
| 501 | 20 | 50,416 | **0** | 2,520.8 | 2,511 | **+0.4%** |
| 701 | 20 | 70,562 | **0** | 3,528.1 | 3,513 | **+0.4%** |

Four things confirmed at once:

1. **The 5.011 × N law**, fitted on N=101–301, holds at N=701 — seven times the
   largest mesh it was fitted on — to 0.4%.
2. **CG converged everywhere.** First genuinely converged solves this solver
   has produced at these sizes.
3. **Newton fell 30 → 20 at N=701**, exactly the 2 per load step every
   converged row showed. Predicted in advance: a truncated CG returns an
   inexact direction and costs extra Newton steps.
4. **Per-iteration cost matches the truncated runs** — 40.9 vs 40.4 ms and 74.9
   vs 74.8 ms. Two independent runs agree on the matvec cost, which is what the
   O(DOF) claim rests on.

**❗ And the open direction is settled: Table 20 UNDERSTATES.** v36's §8.5 says
the sign "is not one-signed … this study does not establish which effect is
larger." It does now: **N=501 +28%** (1,616 → 2,064 s), **N=701 +18%** (4,487 →
5,286 s). µs/DOF 3,219 → 4,109 and 4,566 → 5,379. The gap narrows with size
because the extra CG work is increasingly paid for by the Newton steps it
removes.

**Not measured**, and to be labelled as predictions wherever used: N=1001
+25% (→ 15,261 s) and N=1401 +5% (→ 41,646 s). N=1401 gains little because its
truncated run burned 67 Newton steps against the 20 a converged CG needs.

Memory unaffected: 1,122 and 1,567 MB against 1,123 and 1,568.

**In the report as of v37**: §8.5, Table 20b, with N=1001/1401 labelled as
predictions. Mirrored into the summary (v10).

### 🔬 The MMS operator has no convergence rate — and §8.11's ceiling was overstated

`point9_results/mms_operator_rate_B1_neo_hookean.json`, run 2026-08-29 on a T4.
N=9 and N=33 trained under exactly the N=17 protocol, giving three points.

| N | DOF | operator L2 | Q4 L2 | op/Q4 | operator H1 | Q4 H1 | op/Q4 |
|---|---|---|---|---|---|---|---|
| 9 | 162 | 5.035e-03 | 1.351e-02 | **0.37×** | 1.141e-01 | 1.132e-01 | 1.01× |
| 17 | 578 | 8.238e-03 | 3.403e-03 | 2.42× | 5.831e-02 | 5.666e-02 | 1.03× |
| 33 | 2,178 | 1.136e-02 | 8.525e-04 | **13.33×** | 3.850e-02 | 2.834e-02 | 1.36× |

Fitted rates in h: **operator L2 −0.59**, Q4 L2 1.99; operator H1 0.78, Q4 H1
1.00. **The Q4 control lands on Table 23's measured 1.98 and 1.00**, so this run
is comparable to that table.

**The operator does not converge.** Its L2 error gets *worse* with refinement.
Its error is dominated by **optimization** error, not discretization error:
refining reduces what limits Q4 and leaves the network where it was, while
enlarging the problem it must optimize. The crossover is visible — at N=9 Q4's
own error exceeds the network's and the operator is *ahead* in L2; by N=33 Q4
is 13× better.

**⚠️ And it corrects §8.11.** That section says *"a ratio below one would
indicate a defect in the Dirichlet mask, the quadrature or the work term rather
than an advance."* **Too strong.** The ceiling constrains **Π** — Q4 minimizes
Π over the Q4 space, so nothing in it reaches a lower Π. But Π is none of the
four reported error metrics. L2 against u\* is a different functional, and a
non-minimizer of Π can sit closer to u\* in L2 by partially cancelling Q4's
systematic discretization bias. That is what N=9 did, and the runner raised a
false alarm about it.

What *is* protected empirically: the derivative norms. op/Q4 in H1 semi is
1.01, 1.03, 1.36 and in stress 1.01, 1.03, 1.33 — above one at every mesh. For
a linear problem Galerkin optimality would guarantee that; this problem is
nonlinear so it does not formally transfer, but it held throughout. The
"energy" column is the relative error in a *scalar* strain energy, not the
energy norm, and carries none of that protection (0.89 at N=9).

**Fixed in the code already** so no future run repeats the false alarm:
`mms_operator.py`'s runtime message and `point9_results/make_readme.py`.
**Done in v37**: §8.11's ceiling is restated as a statement about Π that
transfers to the derivative norms empirically and not to L2, and Tables 24a
and 24b carry the three-mesh rate. Mirrored into the summary (v10).

### ⛔ The three B2 zero-shot cases are INVALID (2026-08-29)

`point7a_results/INVALID_B2_zeroshot.json`. Their eval reports are still on
Drive and must never be quoted.

**Relative errors of 8.0 to 14.5** — that is 800% to 1450%, against 5.0–10.6%
for the three valid B1 cases. A relative error above 1 means the prediction is
further from the truth than predicting zero everywhere. Second tell: each curve
is nearly **flat in N** (B2×MR moves 14.358 → 14.470 across a four-fold
refinement), and a model whose error ignores the mesh is not solving the
problem on that mesh.

**Root cause, and why it is the worst possible bug for this particular study**:
the assembled load was overstated by a factor that **depends on the mesh** —
13.1–13.3× at N=21, 20.8–21.0× at N=33. The study trains jointly at N=21 and
33 and then asks whether the operator transfers across resolution. With the two
training resolutions carrying loads inconsistent with each other by ~1.6×, the
model was fitted to two contradictory problems, and any "resolution invariance"
measured from it would have been measuring the bug.

**Repair** (commit `a45496b`, 2000 samples per case): applied to **B2×MR and
B2×AB only**. The check that matters passed — one fixed pressure field
assembled on each resolution now gives N=21 → 11.1775 and N=33 → 11.1784,
**0.0075% apart**. The models trained on the bad load were deleted.

**B2×Neo-Hookean: checked 2026-08-29 — its cache is ALREADY CORRECT.** The
dry run found an overstatement of 1.00×–1.00× over all 1,000 samples and the
mesh-independence check passes. Its stored loads equal the other two cases'
*repaired* values to four decimals (15.7568 at N=21, 15.9056 at N=33) — which
they must, since the load comes from the seed and the mesh, not the material.
Three cases agreeing is a stronger check than any one passing.

**And the mtimes settled it — there is NO model in that directory at all.**
The only files are `fine_ref_cache_N101.pt` (2026-08-27 02:53),
`zeroshot_eval_report.json` (**19:47:50**), `run_manifest.json` (19:48:09) and
`samples_cache.pt` (**19:48:14**). The eval report is stamped **24 seconds
before** the cache was rewritten. So an earlier repair ran immediately after
that eval, fixed the cache and deleted the model — `model_best.pt`,
`train_state_latest.pt`, `metrics_history.json` and `EARLY_STOPPED` are exactly
the set the old repair cell removes, and none of them is there.

**No mystery remains.** The 8.09 was a model trained on the bad load, scored
against freshly built correct references — the eval builds its samples fresh
and never reads the cache — which is precisely the mismatch that gives a large
error, flat in N. B2×NH needs the same treatment as the other two: retrain,
then re-evaluate.

Nothing was written and no model deleted — the diagnose-first cell
(`cell_b2_neo_hookean_repair.py`) stopped, and had it deleted the models the
way the older repair cell does, that evidence would be gone.

To make B2 admissible: retrain all three (all three models are gone), then
re-run eval. Cell: `zeroshot_notebooks/cell_b2_retrain_and_eval.py`.

**The eval is far cheaper than the B1 runs suggest.** Those took ~8 h each,
almost all of it solving twenty N=101 references — and those references are
cached per case in `fine_ref_cache_N101.pt` and are **unaffected by the load
bug**: `_get_fine_sample` builds each fine sample fresh and the FEM solver
assembles its own consistent force internally, so it never saw the bad field.
Where the cache is present the eval reduces to operator inference. The cell
prints the cached count per case before running anything. **Nothing else in the report is affected** — the bug lives
in the B2 zero-shot sample caches only, and Table 12 is B1.

### ✅ Full Drive audit, 2026-08-31 — every reported number checked at source

Prompted by a direct question about whether the results are right. The Google
Drive at `MyDrive/pfem_run/` was read **directly**, not via pasted stdout, and
every number the report and summary quote from a round-5/6 run was compared
against the run's own JSON.

**Verified identical, no discrepancy:**

| what | where on Drive | result |
|---|---|---|
| Table 12 — 21 zero-shot values, 3 B1 cases | `zeroshot_B1_*/zeroshot_eval_*.json` | **all 21 match, and all 3 checkpoint fingerprints match** |
| Table 20 / 20a — the four large rows | `gpu_fem_scaling_B1_neo_hookean.json` | solve times, Newton, CG, failures, peak memory all match |
| MMS operator N=9 and N=33 | `mms/operator_rate/*.json` | every operator, Q4 and Q9 figure matches |
| B2 retrain, best val error | `zeroshot_B2_*/metrics_history.json` | 0.9986 @25, 0.9752 @25, 1.0267 @225 — exact |
| B2 batch-size arms | `b2_batchsize_diagnostic/bs{1,8}/` | bs8 0.98880 @1,800 steps, bs1 0.94436 @4,800; both ran to 22,400 |

**Three discrepancies found and fixed:**

1. **`gpu_fem_cg_converged` N=501 `solve_s` was 2064.0; the run wrote
   2062.659.** The `us_per_dof` beside it (4108.87) was already correct, so the
   two fields in our own file contradicted each other. Corrected. **No
   conclusion moves** — +28% and +18% stand (2062.659/1616.061 = 1.276).
2. **`ms_per_cg_iter` was solve-time ÷ CG iterations, not CG-time ÷ CG
   iterations.** Both forms are now stored and named. The report quotes the
   solve-time form and now says why: the point-8 sweep recorded no `t_cg_s` at
   N=501/701, so that is the only like-for-like division, and CG is 99.8% of
   the converged solve, which bounds the substitution.
3. **`mms_operator.py` still wrote the too-strong ceiling into every result
   JSON.** Only the runtime *print* had been corrected, in `262eb0b` at 02:16 —
   **after** the N=9 (01:37) and N=33 (02:06) runs. Both JSONs on Drive
   therefore carry "the operator cannot beat it at this mesh". The `"ceiling"`
   field is now corrected in the code, with a `ceiling_note` saying those two
   files predate the fix.

**Also corrected in our own record:** `B2_zeroshot_retrain_status.json` said
the B2 eval errors were "flat to the fourth decimal". They are not —
Neo-Hookean moves in the third (0.87137 → 0.87270). The accurate figures are
now stored: all 21 values, and the spread over the mesh is **0.153%, 0.072%,
0.012%** against **85.7%, 111.3%, 79.1%** for the three B1 columns.

**And the three B2 evals had in fact completed** (02:13 on 08-31) — we only
had them as "0.87–0.89". The mesh-mean values are **0.8722, 0.8780, 0.8896**
and they are now in the report and the summary.

### ✅ B1 × Mooney-Rivlin Pareto is DONE — all nine resolutions

`point2_results/pareto_B1_mooney_rivlin.json`. Table 18's companion for the
second material.

| N | FEM error | FEM cost | operator error | operator cost | speed-up |
|---|---|---|---|---|---|
| 13 | 0.624% | 20.4 s | 8.87% | 5.70 ms | 3,574× |
| 21 | 0.280% | 56.9 s | 6.81% | 5.52 ms | 10,310× |
| 33 | 0.129% | 145.2 s | 4.77% | 5.58 ms | 26,042× |
| 49 | **0.062%** | 327.6 s | **3.72%** | 5.81 ms | **56,355×** |

**The operator improves at every single refinement** — 8.87% → 3.72%, no
minimum inside the range. That is the same shape Table 12 found for this
material *and this material only*: MR is the one case of three whose zero-shot
error keeps falling to the finest mesh. B1×NH instead bottoms at N=37 (3.69%)
and worsens after. **Two independent studies, same conclusion about the same
material.**

Speed-ups are **~2.2× larger than B1×NH's** 1,630×–25,676× throughout, because
MR's CPU assembly costs 2.1–2.4× more (Table 4a, autodiff tangent) while the
operator's forward pass is material-independent — 5.5 ms here, 5.5 ms there.

**⚠️ My cost estimate was wrong by ~7×.** The cell said "roughly two hours,
possibly more"; the manifest recorded **14 h 31 m**. The estimate was carried
over from B1×NH without allowing for MR's more expensive assembly, and the
sweep is almost entirely 20 CPU solves at each of nine meshes — the largest
5.5 minutes each. **Expect B1×Arruda-Boyce to take comparably long**, since its
assembly cost is in the same 2.1–2.4× band.

### 🔄 B1 × Arruda-Boyce Pareto is RUNNING — restarted 2026-08-31 at `65d5a65`

**Restarted deliberately, with resume protection active.** The first attempt
was killed at about 7 minutes (N=13 in flight) because it was executing from a
clone that predated the resume fix below. The restart runs
`omar_pfem.pareto_analysis` from `65d5a65`, so every completed resolution is
now written **and stamped**, and a disconnect costs only the resolution in
flight instead of the whole sweep. Expect **6.5 to 14 hours**; the lower
figure is the arithmetic from Mooney-Rivlin's measured per-solve times, the
upper is what its wall clock actually was.

**Nothing was lost in the restart.** The new `pareto_analysis.py` prints a
`[resume]` line whenever an output JSON exists, in every branch. The restart
printed none, so `pareto_B1_arruda_boyce.json` did not exist — N=13's row had
never reached disk. The cost of the restart was 7 minutes of compute and zero
saved work.

**Mooney-Rivlin was skipped and never opened for writing**, as intended: it is
complete at 9/9 on Drive.

**The stale-cell hazard is now closed from the repo side** (`omar_pfem/pareto_analysis.py`).
Partial results go to `pareto_<case>.json.progress`, and the real
`pareto_<case>.json` is written **only when every requested resolution is
present**, then the progress file is deleted. So the final file existing now
*means* the sweep finished, and even a cell that tests nothing but
`os.path.exists(out_json)` reaches the right answer. A partial `out_json` left
by the older code is still read and resumed, so nothing already on Drive is
stranded. `omar_pfem/test_pareto_resume.py` stubs the physics and checks the
file protocol end to end — killed run leaves no final JSON, restart resumes,
subset re-run does not delete other rows, a changed checkpoint forces a fresh
start. 11/11 pass.

**⚠️ COLAB CELLS ARE PASTED COPIES AND GO STALE — USE `Round6_RUN_THIS.ipynb`.**
This has now cost three runs: the Pareto cell printed `ALREADY DONE, will skip`
under a commit whose code says `COMPLETE (9/9)`; the single-resolution cell
recommended hours of FEM a later commit had ruled out; and the B1 metric cell
printed a verdict the checked-out commit had explicitly withdrawn. Each time
the NEW commit hash printed directly above the OLD output.
`zeroshot_notebooks/bootstrap_cell.py` (notebook: `Round6_RUN_THIS.ipynb`) has
no logic of its own to go stale — it updates the repo, prints the commit, and
`exec`s whichever `cell_*.py` you name as it exists on the branch right now.
Every cell should be run through it.

**⚠️ COLAB CELLS ARE PASTED COPIES AND GO STALE.** The restart's own output
proves it: the notebook printed the *old* pre-flight text (`ALREADY DONE, will
skip` / `expect roughly two hours`) even though `git log` in the same cell
showed `65d5a65` checked out. Only the **repo modules** the cell invokes are
fresh; the cell body itself is whatever was pasted into the notebook. This was
harmless here — the resume lives in `pareto_analysis.py`, which came from the
repo — but the stale cell still skips a material on `os.path.exists(out_json)`
alone, so **a future restart from that stale cell would read a partial file as
finished and silently drop the remaining resolutions.** Re-paste the cell from
`zeroshot_notebooks/cell_pareto_remaining_B1.py` before any restart.

**⚠️ A resume gap was found while answering that question, and fixed.**
`pareto_analysis.py` rewrites its JSON after **every resolution**, so a run
that dies at N=37 leaves N=13…33 safely on disk. But the loop started from an
empty `rows` list and re-ran **all** resolutions, overwriting what was there.
On a sweep where N=49 alone is 1.8 h of CPU and the whole thing is 14 h, a
Colab disconnect at hour ten cost everything.

Now completed resolutions are read back and skipped, guarded by the
**checkpoint fingerprint** plus `n_samples`, `fine_N`, `material` and
`geometry` — rows from a different model or protocol are never merged into a
new run, they force a fresh start with a printed reason. Rows are sorted by N
before writing, so a resumed file is ordered like a fresh one.

**The currently running job will not pick this up** — it is executing from a
clone made before the fix. The fix protects the restart if it dies.

### ❌ Input normalization FALSIFIED as the B2 cause (2026-08-31)

`--normalize_inputs 1` reached **0.9910** against the **0.9986** baseline. A
0.8% move on a metric where B1 sits at 0.066. Same failure shape: best at the
FIRST validation, worse after, early stop at 225.

The statistics installed were real (fx: mean 0.00668, std 0.04362 — the force
channels *were* lifted to unit variance), so the transform did what it was
meant to. It simply did not help.

**This is a real answer, not a dead end**: the falsification was written into
the notebook *before* the run. Two structural candidates remain — the Dirichlet
ramp (B2 has **two** ramps vanishing on **different** edges) and the parametric
family (ParametricFieldB2 varies with θ only, never with r). Note the ramp
candidate may already be weakened: the probe's stand-in assertion shows the
mask **can** reproduce `uv_exact` exactly, which should be checked before
spending another training run on it.

### ⛔ The B2 retrain did NOT fix it — the cases are still unusable

`point7a_results/B2_zeroshot_retrain_status.json`, written 2026-08-31. This is
the outcome of the "retrain all three" step the section above ends with.

All three were retrained on the repaired caches, under the B1 protocol, with
`--loss_force_norm` on (resolved from geometry B2 → 1).

| case | best combined val error | at epoch |
|---|---|---|
| B2 × Neo-Hookean | **0.9986** | 25 |
| B2 × Mooney-Rivlin | **0.9752** | 25 |
| B2 × Arruda-Boyce | **1.0267** | 225 |

Best at essentially the **first** validation and worse afterwards; early stop
fired in all three. Eval errors 0.87–0.89, flat in the mesh to four decimals.
The three B1 cases on the same trainer and protocol reach **0.0658–0.0827**.

**1.0 is not a random bad number.** The metric is
`0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v))`; substitute `uv_pred = 0` and it is
identically 1. The models are predicting approximately nothing — and a network
that minimizes Π goes wherever Π's minimum is, so this may be the optimizer
working correctly on data whose Π has its minimum near zero.

**Candidates, in the order they were tested:**

1. **`loss_force_norm` missing** — necessary, **not sufficient**. Added; the
   runs above are with it on.
2. **Batch size** (the 9.11% recipe in `b2_accuracy_search.py` calls
   `train_B2.py` at its default of 1; this trainer defaults to 8) — **ruled
   out.** Two arms at matched optimizer steps (22,400, chosen as a multiple of
   both arms' steps-per-epoch), early stopping off: **batch 8 → 0.9888, batch
   1 → 0.9444.** Noise on curves that swing 0.94–1.45.
3. **Π's minimum is not at `uv_exact` for this cache** — **RULED OUT, run
   2026-08-31 on commit `e12791c`.** Π(s·uv_exact) scanned over s on 3 samples
   at each training resolution: **the minimum landed at s = 1.0 in all 6**, no
   exceptions, no spread. And |W|/U at `uv_exact` came out **1.9951, 1.9985,
   2.0014, 1.9964, 2.0021, 1.9985** — a stationary point of Π = U − W has
   W = 2U, so three decimals on six independent samples is a second,
   independent confirmation.

   **So the cache is fine, the work term is fine, and the functional really is
   minimized by the FEM solution.** The data is exonerated. Everything left is
   in the training path.

   One detail worth carrying: the curve is flat near its minimum — Π(0.85) and
   Π(1.15) are only ~2% of |Π| above Π(1.0). A 15% amplitude error costs almost
   nothing in the objective. That does not explain a factor of ten, but it is
   worth stating when reading how hard Π pushes on amplitude.

4. **The training path itself** — **PENDING**, and it is what is left.

   The specific suspicion: `fun_material` is `(E, nu, f_x, f_y)` fed **RAW** —
   there is no normalization anywhere in `train_B2`'s energy function, which
   the zero-shot trainer re-exports unchanged. For B2 the load is an
   **inner-edge** traction, so `f` is exactly zero on every node off that
   boundary — about **95%** of them at N=21 — and the load repair made what
   remains **13–21× smaller**. `E` is around **1000**. If the two channels
   carrying the loading sit orders of magnitude below the one carrying
   stiffness and are nonzero on a twentieth of the nodes, the model may not see
   the load at all — which would give an error flat in N, flat across
   materials, and stuck near 1.0. **Suspicion, not yet measured.**

   **And a difference that should have been named earlier**: the 9.11% recipe
   and this study are **not on the same problem family**. `data_generate_B2.py`
   draws (E, ν, p) from a **2-D Gaussian random field in (θ, r)**;
   the zero-shot study uses `ParametricFieldB2`, a **two-harmonic Fourier
   series in θ alone**, chosen because it is resolution-independent by
   construction where a gridded GRF is not. So "9.11% is reachable on B2" was
   never transferable evidence about this trainer. B1 uses the same kind of
   parametric field and trains fine, so this is not on its own an explanation
   either.

   **First probe run 2026-08-31 (commit `c7f63c6`), 4 val samples at each of
   N=21 and N=33.** What it settled:

   * **The model is not predicting zero.** rms(pred) 2.51–3.34e-03 against
     targets 4.15e-03–1.28e-02. The 1.0 error is not a dead model.
   * **Its amplitude is 2.5–4× too small** — ratios 0.23, 0.30, 0.26, 0.61,
     0.56, 0.42, 0.35, 0.27, mean **0.375**.
   * **But rescaling would not fix it.** W/U at the prediction is 1.89–3.43,
     i.e. ≈ 2, which *is* the stationarity condition under rescaling. The
     model is not part-way down a ray with more to go — its **shape** is
     wrong, not just its size. (And the two disagree: if pred were s·uv_exact
     then W/U = 2/s implies s = 0.6–1.1, against the 0.23–0.61 measured. That
     mismatch is itself proof the prediction is off the solution ray.)
   * **U(pred) = 1.63–2.32e-02 on every sample and both meshes — a 1.42×
     spread — while the targets span 3.09× and Π(uv_exact) spanned 10×.** The
     model emits a field of nearly fixed strain energy whatever it is shown.
     That is the collapse, measured in the most physical variable available.
   * **It does read its input, about five times too weakly.** Prediction
     variability 0.134 / 0.100 against target variability 0.641 / 0.310. So
     "ignoring the fields" is ruled out; "responding far too weakly" replaces
     it. Correlation is erratic across samples: +0.87, +0.78, +0.45, +0.33,
     +0.16, +0.03, −0.02, −0.11.
   * **The load channel is 4–5 orders below the stiffness channel** —
     rms(f)/rms(E) = 7.2e-05 at N=21 and 2.3e-05 at N=33 — and nonzero on
     4.8% and 3.0% of nodes.

   **⚠️ Two faults in the probe itself, found by reading its own output, now
   fixed.** Both would have produced a confident wrong reading:

   1. It printed Π(pred) with **no Π(uv_exact) for the same sample**. The
      functional test's Π values are on `train_samples`; the probe reads
      `val_samples`, which are different problems. So Π(pred) could not be
      compared with anything. Now both are computed per sample and what is
      printed is the fraction of the available descent.
   2. It put the channel scales at N=21 and N=33 side by side **as if the
      difference were a mesh effect**. It is not separable that way: the cache
      uses `seed_base = 10_000 * N`, so those are different **draws** as well
      as different meshes. Now one fixed seed is rebuilt on both meshes, with
      the mesh-independent load total printed beside the per-node scale.

   **And the control that was missing**: B1 reaches 0.066 on the same trainer,
   architecture and protocol. Any account of B2's failure that applies equally
   to B1 explains nothing. The probe now runs **both arms** and prints them
   together.

### ❌ The Dirichlet ramp is EXONERATED (2026-08-31) — six candidates closed

It was the leading structural candidate and it is dead. The network's output
is `mask × raw`, so to produce `uv_exact` it must emit `uv_exact/mask`.
Representability was never the question — the probe's stand-in reconstructs
`uv_exact` through the mask to machine precision on **both** geometries. What
was measured is how large and how uneven that demanded raw field is:

| | B1 (works) | B2 (fails) | ratio |
|---|---|---|---|
| rms(raw demanded) / rms(output) | 1.72× | 2.00× | **1.16×** |
| peak/rms of the raw demanded | 2.44 | 2.46 | **1.01×** |

B2 has **two** ramps vanishing on **different** edges where B1 has one, and it
makes almost no difference to what the network is asked for — the unevenness is
identical to 1%. Against a **15×** gap in final error and a **3×** gap in
roughness, a 16% difference explains nothing.

**Closed so far:** the load · `loss_force_norm` · batch size · the
data-and-functional · input normalisation · the Dirichlet ramp.

### ❌ JOINT TRAINING IS NOT THE FAULT — B2 fails at ONE resolution too (2026-08-31)

Run on commit `25557d3`, A100, 12 m 8 s + 13 m 56 s. Same cache, same
trainer, same everything; only the joint-training half removed.

| arm | best | at epoch | last |
|---|---|---|---|
| **N=21 alone** | **0.9622** | 50 | 1.2255 |
| **N=33 alone** | **1.0372** | 50 | 1.1538 |
| N=21 and N=33 (joint, on Drive) | 0.9986 | — | — |
| B1, all three materials | **0.0658–0.0827** | — | — |

**Seven candidates are now closed** — the load, `loss_force_norm`, batch
size, the data-and-functional, input normalisation, the Dirichlet ramp, and
now joint training. The failure is fully present at a **single** resolution.

**And the run printed something sharper than its own verdict.** In BOTH arms
the best model is the **first validation event, epoch 50**, and all eight
after it are worse — the same shape as the input-norm run (best at the first
validation, early stop at 225) and every other B2 run on record:

```
N=21   0.9622(ep50) 1.3386 1.1442 1.1371 1.2015 1.2196 1.0999 1.1927 1.2255
N=33   1.0372(ep50) 1.1350 1.1439 1.1256 1.1162 1.2351 1.1744 1.1600 1.1538
```

Both early-stopped at epoch 450 of the requested 4,000, so the doubled epoch
budget never mattered — and the best checkpoint of each arm is 2,500 steps
old. **Training does not stall on B2; it moves the model away from the FEM
solution.** No closed candidate explains that: the load, the ramp and the
family are properties of the *problem*, and none would make 20,000 further
optimizer steps actively harmful.

### 🐛 `model_final.pt` was never saved on an early-stopped run — FIXED (2026-08-31)

The energy-vs-error probe ran and reported `model_final.pt missing` on both
arms. Cause: the save hung off a **`for`/`else`**, and `else` runs only when
the loop finishes **without `break`** — early stopping breaks. So **every
early-stopped run in this study kept `model_best.pt` alone** and silently
dropped the weights training actually ended on.

Not cosmetic: `model_best.pt` is whichever validation event scored lowest, so
on a run whose error *rises* with training — both B2 arms are best at their
**first** validation — the two checkpoints are the only record of which way
the objective moved, and one of them was being thrown away.

**Nothing on Drive was lost.** `train_state_latest.pt` is written at every
validation event and carries `model_state_dict`, so the epoch-450 weights are
there; the cell now unpacks them (with an assert that the state's epoch equals
the last validation event, so another run's state cannot be passed off as this
arm's endpoint). Seconds, no retraining. The trainer now saves
`model_final.pt` unconditionally after the loop.

### 📊 What the first probe run DID establish, on the best checkpoints

Both arms share one cache, so the two models were scored on **identical**
samples — which makes them directly comparable in a way their reported val
numbers are not (each arm's val error is measured on its own resolution's
samples).

| on the same samples | **N=21-trained** | **N=33-trained** |
|---|---|---|
| descent captured, mean | 44% | **59%** |
| roughness, mean | 2.32× | **2.04×** |
| correlation, N=21 block | −0.32, +0.81, −0.41, +0.81 | **+0.30, +0.65, +0.26, +0.91** |
| amplitude ratio | 0.22–0.46 | **0.34–0.77** |

**The N=33-trained model is better on every physical measure, including on
N=21's own samples** — better descent, smoother field, and correlations that
never go negative. Its reported val error (1.0372) is the *higher* of the two
only because the two numbers are measured on different sample sets and are
not comparable. Worth remembering before any of these val numbers is quoted
against another.

Both still sit far from B1, which captures ~100% of the descent at roughness
1.01×.

### ✅ CONFIRMED: the validation metric ranks B2's checkpoints backwards (2026-08-31)

All 100 val samples of each resolution, both arms, commit `83469fe`:

| arm | checkpoint | per_component | both_components |
|---|---|---|---|
| N=21 | epoch 50 | 0.9622 | 0.7743 |
| N=21 | epoch 450 | **1.2255** ↑ | **0.6858** ↓ |
| N=33 | epoch 50 | 1.0372 | 0.7043 |
| N=33 | epoch 450 | **1.1538** ↑ | **0.6822** ↓ |

The metric early stopping obeyed goes **up** while the error over both
components goes **down**, in both arms, on the full set. Every B2 run stopped
at its **first** validation event and kept the worse model.

**The mechanism, from its own printout:** per-sample `rms(v)/rms(u)` averages
**1.90** while the ratio of the *averaged* components is **0.90**. The
distribution is skewed, so the average reports its tail — samples where one
component is small and its relative error is therefore large, however well the
field as a whole is predicted.

**⚠️ WHAT THIS DOES NOT MEAN, and it matters more than the finding.** B2 is
still bad: best both-components error **0.68** against B1's **0.066**. The
metric cost B2 roughly 0.77 → 0.69 — **an eighth of the gap, not the gap.**
B2 zero-shot failing is **not** a metric artefact, and the report's conclusion
does not change. Anyone reading this later: do not turn this into "B2 works
after all".

**Fixed in `resolution_invariance_zeroshot.py`:** `evaluate_resolution` now
returns both metrics; `--selection_metric` (default **`both_components`**)
chooses which one drives `model_best.pt` and early stopping;
`metrics_history.json` records both plus which one was in force. The
per-component number is still printed and still stored as
`combined_val_error`, because every reported figure is in those units. A
resume from a pre-flag state starts selection afresh rather than comparing two
different metrics.

### ✅ B1 CHECKED — the report's numbers STAND, no table is restated (2026-08-31)

Three B1 cases, their own reported checkpoints, all 100 val samples:

| case | N | per_component | both_components | ratio | rms(v)/rms(u) |
|---|---|---|---|---|---|
| neo_hookean | 21 | 0.0772 | 0.0529 | 1.46 | 3.45 |
| neo_hookean | 33 | 0.0543 | 0.0360 | 1.51 | 3.54 |
| mooney_rivlin | 21 | 0.0984 | 0.0723 | 1.36 | 3.34 |
| mooney_rivlin | 33 | 0.0670 | 0.0466 | 1.44 | 3.42 |
| arruda_boyce | 21 | 0.0953 | 0.0565 | 1.69 | 4.06 |
| arruda_boyce | 33 | 0.0613 | 0.0360 | 1.71 | 4.17 |

**⚠️ The cell printed "B1 IS AFFECTED TOO ... the tables have to be restated".
THAT VERDICT WAS WRONG and is withdrawn.** It tested a threshold I picked
(1.15×) on the *offset* between the two metrics, when the B2 failure is an
*inversion of ordering*. One checkpoint per case cannot test ordering at all.

**What the numbers actually say.** `per_component` is 1.36–1.71× the
both-components number, **in the same direction every time**. That is a level
offset, and an expected one: `rms(v)/rms(u)` is 3.3–4.2 — the block is pulled
vertically, `u` is the small component, and dividing each component by its own
size lets the small one dominate. So **the reported B1 numbers are
conservative**: the true both-components error is 0.036–0.072, lower than the
0.054–0.098 reported.

**Nothing published is wrong.** §7.1 defines every reported error exactly, and
this file already recorded (in the `pareto_analysis` docstring correction) that
Tables 5/11/12 use the per-component average while §4.4 uses the combined norm,
"so the combined norm reads lower" on B1. This run measured that offset; it did
not find an error.

**Why B2 inverts and B1 does not** — the component ratio's *stability*, not its
size. B1's is 3.34–4.17, a tight band. B2's per-sample mean is 1.90 against an
aggregate 0.90, i.e. skewed, so the average reports its tail.

**Still open, and cheap:** whether B1's runs *also* early-stopped on a metric
that had begun to invert. If so B1 is better than reported — an improvement to
claim in v38, not an error to fix. The second half of
`Round6_B1_Metric_Recheck.ipynb` now recovers each B1 run's endpoint from its
own `train_state_latest.pt` and scores both metrics on both endpoints.

### ~~NEXT — `Round6_B1_Metric_Recheck.ipynb`, before the report is touched~~ — RAN, see above
Every B1 number in the report (5.0–10.6% zero-shot, 0.0658–0.0827 on the
training meshes) is the **same metric**.

* **B1's two metrics agree** → the reported numbers stand, and the metric
  inverting on the annulus but not the block is itself a finding about the
  geometry.
* **B1's disagree too** → every zero-shot number is in a metric that does not
  order models correctly, and the tables must be restated before v38.

CPU, minutes, trains nothing.

**Then, and only then:** retraining B2 with the fixed selection is worth
considering. At epoch 450 of 4,000 it was still improving on every measure —
Π falling, roughness falling, both-components error falling — so B2 has never
been trained to convergence. From the measured rate (450 epochs in 12 m 8 s),
4,000 epochs is about **1 h 50 m** per arm on an A100.

### ~~THE ENERGY-VS-ERROR RUN LANDED~~ — the reasoning that got here (2026-08-31)

Ran on `657deb0`, CPU, four checkpoints. **Π fell in both arms:**

| arm | Π(epoch 50) | Π(epoch 450) | descent | roughness |
|---|---|---|---|---|
| N=21 | −2.4788e−02 | **−4.3379e−02** | 44% → **76%** | 2.32× → **1.74×** |
| N=33 | −3.2643e−02 | **−4.3995e−02** | 59% → **80%** | 2.04× → **1.73×** |

So training works on its own objective, and the field gets **smoother** as
well as lower in energy.

**The cell's printed verdict — that Π must prefer some field other than the
FEM solution — does not survive the per-sample numbers and is WITHDRAWN.**
That branch was written assuming the error rose. On the samples the probe
actually looked at, it **fell**, epoch 50 → epoch 450:

```
N=21 arm, on N=21   0.9763->0.9023  0.7287->0.4474  0.9969->0.9330  0.5877->0.0930
N=33 arm, on N=33   0.4826->0.4832  0.6554->0.6250  0.7798->0.6734  0.7687->0.7175
```

Eight samples, better or level on all eight, one landing at **0.0930** —
B1 territory. And every negative correlation at epoch 50 (−0.32, −0.41) is
**positive** at epoch 450.

**Meanwhile the trainer says epoch 450 is 1.27× worse.** The two numbers are
different metrics:

| | |
|---|---|
| **the trainer** (`evaluate_resolution`) | `0.5·( rms(e_u)/rms(u) + rms(e_v)/rms(v) )` |
| **the probe** (`rel`) | `rms(e)/rms(uv_exact)`, both components at once |

The trainer's divides each component by **its own** size, so a small
component's ratio dominates the average however well the field as a whole is
predicted. It is also the metric **early stopping used** — and every B2 run on
record stops at its *first* validation event, which is exactly what a metric
that rises as the model improves would produce.

**🎯 NEXT — `Round6_Val_Metric_Check.ipynb`**, and it is the highest-value
thing outstanding. Both metrics on **all 100** val samples of each
resolution, for both checkpoints of both arms, with the per-component ratios
and component sizes beside them. CPU, minutes.

* **trainer up while combined down** → the metric ranks models backwards.
  Then: (1) the report's B2 zero-shot numbers are this metric and need
  re-reading; (2) **B1's numbers are the same metric and must be re-checked
  the same way**; (3) only then is retraining B2 worth it.
* **both up** → four samples were unrepresentative, the trainer is right, and
  the energy-vs-error reading stands as printed.

**Do not put the B2 zero-shot row into report v38 until this returns.**

### ~~NEXT, and it is free: does training LOWER Π while raising the error?~~ — RAN, see above

The objective is Π = U − W. `model_best.pt` (epoch 50) and `model_final.pt`
(epoch 450) are both saved for both arms, and `test_b2_zeroshot_model.py`
already prints Π(pred) beside Π(uv_exact) on the same sample. So this is the
existing probe on four existing checkpoints — **CPU, minutes, no training**.
Notebook: `Round6_B2_EnergyVsError.ipynb`.

| outcome | meaning |
|---|---|
| **Π down, error up** | the optimizer is descending correctly toward a field with **less energy than the FEM solution**. The discretised Π on B2 does not have `uv_exact` as its minimiser over what this network can reach. Structural and reportable — and it makes regenerating the cache from the GRF the **wrong** next spend, because the data family cannot cause an objective that prefers a different field |
| **Π up** | the optimizer is not minimising its own objective — an optimisation failure. Learning rate and the gradient through the two-ramp mask, both cheap to test |

**Why the functional check does not already answer this.** It scanned
Π(s·`uv_exact`) over a scalar `s` and found the minimum at s = 1.0 in 6/6
with W/U = 1.9951–2.0021. That is a scan along **one ray**: it proves
`uv_exact` is stationary under rescaling, and says nothing about whether some
field off that ray has lower Π. This measurement looks off the ray, at the
two fields training actually produced. The earlier entry's "the data is
exonerated, everything left is in the training path" was right about where to
look and overstated what a ray scan can settle.

**The GRF regeneration is NOT the next step** on this evidence, whatever the
single-resolution cell's own printed verdict said. It costs hours of FEM,
and the measurement above can rule it out for minutes.

### ~~NEXT: does B2 train at ONE resolution?~~ — ANSWERED ABOVE, kept for the reasoning

Five rounds of guessing between structural differences have closed five
candidates and cost five runs. **Two differences remain** from the B2 model
that *does* work — the 9.11% result `b2_accuracy_search.py` got from
`train_B2.py` on the same geometry, architecture and energy:

1. **the data family** — `data_generate_B2` draws (E, ν, p) from a **2-D GRF
   in (θ, r)**; `ParametricFieldB2` uses **two Fourier harmonics in θ alone**,
   so every field is constant along each radius;
2. **joint training** — the 9.11% run trains at **one** resolution; the
   zero-shot trainer trains at N=21 **and** N=33 together.

**Training B2 at N=21 alone and at N=33 alone SEPARATES them** instead of
guessing, and it is cheaper by orders of magnitude — half an hour against
hours of FEM regeneration.

| outcome | meaning |
|---|---|
| **both reach ~0.07** | joint training is the fault. Reportable on its own, since resolution invariance is the claim under test and **B1 does the same joint training successfully**. First thing to look at then: B2's per-node load scale differs **1.98×** between its two meshes against B1's 1.26× |
| **both stay ~1.0** | joint training is not the fault; **the parametric family is all that is left**. Testing it means regenerating the cache from the GRF — **hours of FEM**, and needs a decision first: B2 zero-shot is one cell of one table and the report is honest without it |
| **one of each** | a resolution-specific problem, narrower than either, and the failing mesh's cache is where to look |

Single arms get 4,000 epochs against the joint run's 2,000, because one
resolution is half the training set. Notebook: `Round6_B2_SingleResolution.ipynb`.

### 🎯 The B1 control arm ran, and it separates the two cleanly

Same trainer, same architecture, same optimizer, same protocol. B1 reaches
0.066; B2 sits at ~1.0. Both probed identically, 4 val samples at each of
N=21 and N=33.

| | **B1 (works)** | **B2 (fails)** |
|---|---|---|
| descent captured, Π(pred)/Π(uv_exact) | **99.8–100.8%**, mean 100% | **36.8–59.6%**, mean 47% |
| relative L2 vs uv_exact | 0.030–0.058 | 0.449–0.954 |
| correlation | **+0.997 to +0.9996**, every sample | +0.869 down to **−0.113**, erratic |
| amplitude ratio | 0.977–1.019 | **0.233–0.606** |
| prediction variability vs target's | 0.332/0.346, 0.157/0.159 — **matched** | 0.134/0.641, 0.100/0.310 — **a fifth to a third** |
| **roughness** | **1.01×** (0.99–1.05) | **3.00×** (1.67–4.80) |
| rms(f)/rms(E) in the input | 6.1e-04 – 8.2e-04 | **2.3e-05 – 7.2e-05** |

**Roughness is the sharpest number**: `(U_pred/U_exact) / (amplitude ratio)²`.
A field as smooth as the truth has strain energy scaling with amplitude
squared, so this is 1 — and B1 lands on 1.01 across all eight samples. B2
lands on **3.00**: its prediction carries three times the strain energy its
size warrants. **It is rough, not merely small**, which is why rescaling
cannot fix it — the Π(s·pred) scan puts the minimum at s = 1.0 on six of the
eight.

**And worth saying plainly: B2 has the BETTER-specified problem and the worse
result.** Its W uses the assembled force with no fudge factor, so its training
Π is *identical* to the FEM solver's — that is exactly why the functional test
found W/U = 2.000 and the minimum exactly at s = 1. B1's W is a trapezoid sum
over the raw pointwise traction divided by an edge count. **Any story in which
B2 fails because its data or its energy is wrong is now closed.**

**⚠️ A third fault in the probe, caught by the control.** It printed `sum(f)`
for both geometries and asserted in its own text that the two meshes "must"
agree — which on B1 came out as *"the TOTAL load agrees to 56.427% — it
must"*, a sentence that contradicts itself. The quantity that must be
mesh-invariant is the one each geometry's **own** work term uses, and they
differ. B1's own invariant, `sum(f)/n_edges`, is 4.6556 vs 4.5516 — 2.3%
apart, fine. The same confusion produced the *"that is impossible"* flag when
Π(pred) came out **below** Π(uv_exact) on six B1 samples: where the trainer's
Π and the solver's Π are not the same functional — as on B1 — `uv_exact` does
not minimize the trainer's Π, and a 0.1–0.8% gap is just that quadrature
difference. Both fixed.

5. **The input channels are unnormalized** — **PENDING**, and it is the one
   candidate the control singles out.

   On B2 the two channels carrying the **load** sit **10–30× quieter** relative
   to the one carrying **stiffness** than they do on B1, and are nonzero on
   3–5% of nodes. And the controlled mesh comparison (one fixed seed rebuilt on
   both meshes) shows B2's **per-node** load scale changing **1.98×** between
   N=21 and N=33 while the **total** is constant to 0.001% — the same physical
   loading presented to the network as two different numbers.

   **⚠️ Be clear what this is.** Neither the B1 nor the B2 runs used any input
   normalization — `train_B1` has the hook and its own docstring says it is a
   no-op by default and that every reported result was produced with it off. So
   **this is not the difference between the arms**; it is a candidate *remedy*
   for a condition that is measurably much worse on B2. **It may not work.**

   **Built**: `--normalize_inputs` wired through the zero-shot trainer and
   train_B2's two forward paths, sharing train_B1's single implementation and
   single module-level state rather than copying it. Statistics over the
   training samples of all resolutions, written to `input_norm.json`, reused on
   resume behind a drift assertion, and **loaded automatically from beside the
   checkpoint at eval** — a model trained on standardized inputs and scored on
   raw ones gives plausible garbage rather than an error. Off by default;
   verified a no-op when off.

   **What would falsify it**: if the normalized run also lands near 1.0, the
   input scaling is not the obstacle, and the next candidates are the Dirichlet
   ramp (B2 has **two** ramps, x/R_out and y/R_out, vanishing on different
   edges, against B1's single y/Ly) and the parametric family itself.

   Notebook: `Round6_B2_InputNorm.ipynb` — GPU, ~15 min, one arm.

   Probe: `omar_pfem/test_b2_zeroshot_model.py --geometry B1|B2`.
   **No training, CPU.** Notebook: `Round6_B2_Model_Probe.ipynb`.

**In the report**: v37 §8.7's "Two limits" paragraph and §10's bullet both say
this outright — data corrected, models retrained, still unusable, cause under
investigation, no B2 zero-shot number quoted anywhere.

### ❗ Table 12's caption is WRONG — resolved 2026-08-29

`point7a_results/B1_neo_hookean_OPEN_QUESTION.json`. File modification times
in `zeroshot_B1_neo_hookean/` order the events and settle it:

| when | file |
|---|---|
| 2026-08-10 11:57 | `samples_cache.pt` |
| 2026-08-10 13:09 | `model_best.pt` |
| 2026-08-10 23:11 | `metrics_history.json` — the **joint** history (21 and 33) |
| **2026-08-11 15:27** | **`zeroshot_eval_report.json` — Table 12's source** |
| 2026-08-27 21:12 | `zeroshot_eval_coarse_and_fine.json` — see below |
| 2026-08-28 05:00 | `pareto_B1_neo_hookean.json` — Table 18's source |

The eval was run a **day after** the joint training, on the `model_best.pt`
that joint training produced. There is no N=21-only model in the timeline.

**So Table 12's caption must be corrected.** It says *"a single checkpoint
(trained once at N=21), evaluated without retraining at five unseen
resolutions"*. The training set was **N=21 and N=33**. Everything else in the
caption is fine and **the five numbers are unaffected** — this is a caption
fix, not a data fix.

**And it settles the protocol question**: all three valid B1 cases are
joint-trained at 21 and 33. They share a protocol. The only difference left is
that B1×NH's eval lists 5 resolutions where the other two list 7.

### ✅ The B1 zero-shot table is COMPLETE — three cases, no compute needed

`zeroshot_eval_coarse_and_fine.json` was opened and it is exactly what its name
said: **the 7-resolution eval of B1×Neo-Hookean on the same joint
checkpoint**, run 2026-08-27, with a fingerprint. It had been invisible only
because every listing searched for `zeroshot_eval_report.json` by name.

It reproduces Table 12: across the five shared resolutions the worst relative
difference against the old file is **8.5e-07**, identical to six significant
figures, so **the report's four-decimal values do not change**. It adds the two
coarser meshes round-5 item 7 asked for.

All three B1 cases, joint-trained at N=21 and 33, same 7 resolutions, mean
relative L2 against the N=101 reference:

| N | Neo-Hookean | Mooney-Rivlin | Arruda-Boyce |
|---|---|---|---|
| 13 | 0.0967 | 0.1064 | 0.1011 |
| 17 | 0.0791 | 0.0885 | 0.0832 |
| 25 | 0.0574 | 0.0691 | 0.0647 |
| 29 | **0.0521** | 0.0628 | 0.0597 |
| 37 | 0.0525 | 0.0541 | **0.0564** |
| 41 | 0.0562 | 0.0515 | 0.0575 |
| 49 | 0.0670 | **0.0504** | 0.0630 |

**The finding**: training was at N=21 and 33. Two of the three bottom out near
that range and then get **worse** on the finest meshes — Neo-Hookean bottoms at
N=29 and rises **28.7%** by N=49; Arruda-Boyce bottoms at N=37 and rises
**11.6%**. Mooney-Rivlin alone keeps improving to the finest mesh tested. So
zero-shot transfer to much finer meshes is not free, it is material-dependent,
and reporting it on one material would have concealed that. The two coarser
meshes (13, 17) are uniformly the worst for all three, which is the
unsurprising half.

**One anomaly recorded, not smoothed**: B1×Neo-Hookean stopped 10 validation
events after its best epoch where the protocol's patience is 8. Its reported
errors are unaffected — they come from `model_best.pt`. The likely explanation
is that the trainer's own best-tracker sat 50 epochs later than the argmin of
`combined_val_error`, which would reconcile it exactly, but that is unverified.
This case also predates the manifest instrumentation, so it has no recorded
generation or training wall clock.

**What this unblocks**: Table 12 can be REPLACED by the 7-resolution version on
the same checkpoint — which fixes the wrong caption and adds the coarser meshes
in one edit — and the Pareto can run for B1×MR and B1×AB now, without waiting
on B2.

### R6-1b — normalization TESTED as an OOD mitigation (2026-08-29)

`point6_results/ood_mitigation_B1_neo_hookean.json`. Timon asked for the
mitigation Section 8.6 named to be tested. It was. **It is not a clean win,
and the runner's own headline overstates it.**

The cell printed *"Normalization materially reduces the degradation. Worth
reporting as a fix."* That rule fires on ONE cell — material at k=3, where the
degradation RATIO goes 5.90× → 3.85×. Three things cut against reading it that
way:

1. **In distribution it COSTS 6.7%** (0.0867 → 0.0925). That price is paid in
   every cell.
2. **On the absolute error it improved 5 of 18 cells and hurt 13.** Every cell
   at k ≤ 1.5 is worse by 11–23%; every loading cell is worse by 14–37%. The 5
   improvements are all at k ≥ 2.0 on material and both.
3. **The ratio is flattered by a worse denominator.** Degradation divides by
   each model's own in-distribution error, and the normalized model's is 6.7%
   larger, so part of every ratio gain is the denominator.

**And the one cell the headline rests on is the anomalous one.** Raw material
is strictly increasing in k (0.1409 → 0.5112). Normalized peaks at k=2.5 and
**falls** at k=3.0 (0.3973, 0.4030, **0.3565**). An error that stops growing as
the shift grows is what a prediction collapsing toward something
shift-independent looks like, not extrapolation.

**Loading is the control and it confirms §8.6.** Both models are nearly flat
under loading shift (raw 0.99–1.07×, normalized 1.07–1.38×). Normalization did
not move WHERE the sensitivity lives.

**Verdict for the report**: the mechanism in §8.6 stands — standardizing is an
affine rescaling, so a shifted E is still outside the trained range. Report it
as a tested-and-did-not-work mitigation, which is exactly what Timon asked for,
and name the untested remaining candidate (predicting a scaled quantity such as
u·E rather than u).

**✅ The training-budget confound is closed, and it cuts against
normalization.** Both `metrics_history.json` files were read on 2026-08-29:

| | best val | at steps | ran to | stopped |
|---|---|---|---|---|
| baseline | **0.09587** (epoch 550) | 55,000 | 75,000 | early |
| normalized | **0.1023** (epoch 850) | 85,000 | 105,000 | early |

The protocol WAS identical — both stopped under the same rule (patience 8
validation events, min_delta 1e-4; the normalized run stopped exactly 8 events
after its best). Different lengths are that rule's OUTPUT, not a deviation.
And the normalized model got **40% more optimizer steps** and found its best
**55% later**, and is still 6.7% worse. Extra training did not rescue it.

**Two independent metrics agree on the 6.7%**: the 200-sample training
validation set gives +6.71% (0.09587 → 0.1023); the OOD script's own 10-sample
in-distribution cell at N=21 gives +6.69% (0.0867 → 0.0925). Different sample
sets, different code path, 0.02 percentage points apart.

**Identity check**: the baseline's 0.09587 matches Table 21's
physics_informed entry (0.0959 at 75,000 steps) to 3.3e-05 — it is the same
checkpoint the 2×2 used.

### ✅ RESOLVED 2026-08-29 — CG never converged in the point-8 sweep

**The question is settled, and my earlier framing of it was wrong on two
counts.** Both Drive JSONs were read on 2026-08-29 and their `stats` blocks
are now committed into `point8_results/gpu_fem_scaling_B1_neo_hookean.json`.

What the counters say:

| N | Newton | CG iters | CG failures | CG per Newton | ms per CG iter |
|---|---|---|---|---|---|
| 101 | 20 | 10,084 | **0** | 504.2 | 38.8 |
| 201 | 20 | 20,168 | **0** | 1,008.4 | 39.5 |
| 301 | 20 | 30,240 | **0** | 1,512.0 | 40.4 |
| 401 | 20 | 40,000 | 20 | 2,000.0 | 41.5 |
| 501 | 20 | 40,000 | 20 | 2,000.0 | 40.4 |
| 701 | 30 | 60,000 | 30 | 2,000.0 | 74.8 |
| 1001 | 40 | 80,000 | 40 | 2,000.0 | 152.1 |
| 1401 | 67 | 134,000 | 67 | 2,000.0 | 296.6 |

**Wrong count 1: this is not the last-Newton-iteration effect.** At N≥401,
`cg_failures` EQUALS `newton_iters_total` — every CG solve hit the cap, not
one per load step. `cg_iters/newton_iters` is exactly 2000.0, the
`cg_max_iter` default.

**Wrong count 2: it is not an unreachable target either.** The three
converged rows fix the true requirement: **CG iterations per Newton solve =
5.011 × N** (the three constants are 4.992, 5.017, 5.023 — 0.6% apart). That
is the textbook rate: κ grows as 1/h², so CG needs O(1/h) = O(N) iterations.
CG at N≥401 was simply not given enough iterations. It is a budget shortfall,
not a broken stopping test.

Fraction of the required CG work actually performed: **N=401 99.5%, N=501
79.7%, N=701 56.9%, N=1001 39.9%, N=1401 28.5%.**

**Accuracy is untouched.** Newton's test is ABSOLUTE (‖R‖ < 1e-7), checked
before each step, and the counts stay far below `newton_max`=30 per load step.

**Newton's count is the tell.** It is exactly 20 (2 per load step) for every
row where CG did essentially all its work (N=101–501), and grows only as
truncation deepens: 30, 40, 67. An inexact direction costs extra Newton steps.

**⚠️ A sentence in report §8.5 is falsified by this.** It reads *"Above it,
the number of CG iterations required grows with refinement, because the
tangent's condition number scales with the inverse square of the element
size."* The MECHANISM is right and is now measured (5.011 × N), but the
measured CG count did NOT grow above N=401 — it was pinned at the cap. The
sentence must be rewritten, not patched.

**Also found: Table 20 is assembled from two runs, and N=501 is in both** —
identical settings, identical iteration counts (20 Newton / 40,000 CG / 20
failures), **13.0% apart in wall clock** (1,616.1 s vs 1,826.8 s). Table 20
quotes the first. That is the run-to-run variation every single-run timing in
the table silently carries.

**Direction of the error in Table 20: not one-signed.** Truncating CG makes
each Newton step cheaper than a converged one AND raises the Newton count.
My earlier note claiming the timings are "pessimistic" was unfounded.

**Still true and unaffected**: the memory model (2.4% out of sample — memory
does not depend on CG count), the 3.93M-DOF headline, and the point that a
matrix-free CG iteration contains assembly by construction.

The stopping test should still also accept a small absolute residual,
`‖r‖ < max(cg_tol*‖b‖, eps_abs)` — that is the real MMS bug, on tiny problems
where the relative target genuinely is unreachable. Not applied: it changes a
solver every committed result depends on, and it should be a deliberate,
separately validated change.

### ⚠️ Table 4b, and 74× vs 309× — read before touching §4.2 or §8.5

This trips up every session, so it is written out once, verified:

* **The REPORT has no Table 4b.** Its FLOP figures live in a plain, unnumbered
  paragraph directly after Table 4a.
* **The SUMMARY has a Table 4b** — a 3-row per-material table of FLOPs per
  sample (NH ≈5.88×10⁷ assembly / ≈7.9×10⁵ solve; MR and AB ≈9.72×10⁷ / same).
* **74× is the FLOP ratio** (5.88×10⁷ ÷ 7.9×10⁵), hand-counted, not measured.
* **309× is the measured wall-clock ratio** for B1 × Neo-Hookean from report
  Table 4a (25.343 s assembly ÷ 0.082 s solve); 290–692× across the six.
* So **74 is not a wrong number, it is a different quantity.** Never put it in
  a sentence about time, and never attribute it to the report.
  `advisor_feedback/2026-08-28_round6_timon.md` line 135 makes exactly this
  mistake — it says "the report's Table 4b" — and that note is what the v33
  draft copied from. The note is left as written because it is a record of
  what was thought at the time; this block is the correction.
* Loose end, not an error: the FLOP count implies the autodiff materials cost
  ≈1.65× more per element, while Table 4a's measured assembly time says
  2.1–2.4×. Both are stated on their own basis; the gap is unexplained in the
  text and nobody has looked into it.

---

## Point 2 (Pareto): first case measured TWICE, NOT yet in the report (2026-08-28)

`pareto_analysis.py` ran twice for **B1 × Neo-Hookean** — run3 (1 h 54 m) and
run4 (6 h 24 m). run4 was the **same configuration on a slower Colab runtime**,
not the seed/metric change described below, so that decision is still open.
Numbers and the full reading are in `omar_pfem/point2_results/`. The other five
cases have not been run.

Headline: the two methods never compete on accuracy — FEM at its coarsest
(N=13) is 0.608%, already 6.1× better than the operator at its best (3.69% at
N=37). The front is two branches with nothing between them. The operator's real
argument is the trend, not a point: its cost is flat in mesh size while FEM
grows superlinearly, so the speed-up climbs by an order of magnitude across the
sweep (1,630× → 17,895× on run4's numbers).

### What running it twice bought
* **Errors identical to every printed digit**, both sides, all nine
  resolutions, across two runs on different hardware. The accuracy half is
  fully reproducible.
* **Wall-clock is not, and systematically so**: run4's FEM is 2.887–2.925×
  slower at *every* resolution — a 1.3% spread, i.e. a different machine, not
  noise. Absolute milliseconds describe the Colab instance, not the method.
* **The ratio survives**: excluding N=49 the two runs' speed-ups agree within
  17%, because both sides slowed together. Quote the speed-up, not the
  milliseconds.
* **run4's timings are the ones consistent with the report.** Table 10a gives
  B1×NH at bs=1 as 4.582 ms; run4 at N=21 (the same 441-node mesh) gives
  4.584 ms. run3 gives 1.610 ms — a third of it. Use run4.
* **The N=49 anomaly did not reproduce.** run3's 4.613 ms outlier is 5.555 ms
  in run4, inside that run's own 4.584–5.563 ms band. It was a property of that
  run, not of N=49. Closed.
* Residual: run4's own N=21 is 17% faster than its other eight with no pattern,
  so assume ~20% jitter on any single bs=1 latency here.

**Not written into the report yet, deliberately** — one of six cases, and one
open decision below.

### A false claim in my own code, now fixed
`pareto_analysis.py`'s `rel_l2` docstring claimed its numbers were "comparable
to Table 12's". They are not, for two independent reasons:
1. **Metric.** It uses the combined relative L2 `‖e‖/‖u‖` (Section 4.4's
   convergence convention). Tables 5/11/12 use the per-component average
   `0.5*(rms(e_u)/rms(u)+rms(e_v)/rms(v))`. On B1 the loaded component v
   dominates, so the combined norm reads lower.
2. **Seeds.** Pareto draws `900_000 + i`; the zero-shot eval draws
   `20_000_000 + i`. Different physical problems entirely.

Both push the same direction, so no conversion factor between the two tables
can be quoted from this run. Docstring and seed line both carry the correction
now. **The Pareto result itself is unaffected** — within the run, both sides
use the same metric, samples and reference, which is all a Pareto plot needs.

### Open decision for Omar
Keep the combined-norm numbers (recommended — it is the right metric for a
convergence comparison, and re-running buys a metric change, not a better
measurement), or re-run with seed base `20_000_000` and the per-component
metric so the operator column can sit next to Table 12. Re-running costs about
**1 hour**, not the original 1 h 54 m, because those fine references are
already cached.

Re-running is now known to cost 1 h 54 m on a fast runtime and 6 h 24 m on a
slow one, so check what machine Colab hands out before starting.

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
| 2 Pareto | `omar_pfem/pareto_analysis.py` | one run per finished zero-shot case. **NOT minutes** — the two B1×NH runs took 1 h 54 m and 6 h 24 m for the same configuration on different Colab runtimes. It draws its own seeds (900_000+i) so it cannot borrow the zero-shot study's fine references; it computes its own |
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
| 4 | Benchmark Transolver and GPU FEM under **identical batch sizes** | ✅ **done** — Table 10a is operator latency at bs=1/8/32/128 for all six cases (measured, median of 50 repeats), 10b the matched speed-up, 10c the break-even. This row was left stale after the 2026-08-27 recompute |
| 5 | Error in **physically important quantities** beyond displacement: H1 semi-norm, energy, stress components, reaction forces, maxima (for the Transolver) | ✅ **done** — all six cases measured and written into report §8.8 (Tables 15–17) and summary §8; see "Point 5 written into both documents" at the top |
| 6 | Investigate **OOD robustness** — the 4–5× degradation is "probably the biggest obstacle to a strong physics-informed operator claim" | ⬜ research, not just measurement |
| 7 | Resolution invariance: train on 2, test on 5 **coarser AND finer**; the point being *"train on a very coarse grid and inference on a finer grid ... could provide computational savings"* — so **quantify the savings**, not just the flat error. Plus a **data-driven** comparison, its data *"from two different (fine enough) simulations"* (i.e. matched to the PI model's two training resolutions) | 🟡 7a covered by the per-case notebooks (7 resolutions, a superset of his 5), 1 of 6 cases finished; **7b ✅ complete** — the 2×2 is §8.9, Table 21, and the ranking flips |
| 8 | Test GPU-native FEM at **finer discretizations up to a few million DOFs**; and: *"Did you use Tensormesh or write the code yourself?"* | ✅ **both parts done** — §8.5 states the solver was written from scratch in PyTorch, not Tensormesh or any FE library; the sweep ran 0.02→3.93M DOF and is Table 20 |
| 9 | Use **MMS** as ground truth instead of a baseline FEM solution, to test the operator *"compared to FEM"* — i.e. **both** are scored against the manufactured truth, which is the only way FEM itself gets graded (today it *is* the reference, so it cannot be) | ✅ **complete and three-way** — report §8.11, Tables 22–24. The body-force blocker was removed by writing a separate operator (`mms_operator.py`) with a body-force channel and a body-force term in Π |

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

**Point 8 is now COMPLETE and written up (§8.5, Table 20).** Eight
resolutions, 0.02M → 3.93M DOF, one A100, FP64. Headline: **3,925,602 DOF in
11.0 h using 3,280 MB of 80 GB (~4%)**. Three findings worth keeping:

1. **µs/DOF is U-shaped** — falls 6.0× from 19,410 (N=101) to 3,219 (N=501),
   then rises 3.1× to 10,125 (N=1401). Two different causes: GPU
   under-occupancy below the minimum, growing CG iteration count above it
   (condition number ~1/h²). Large branch fits **DOF^1.54**, pairwise 1.52 /
   1.40 / 1.76 — the last interval is the steepest, so the exponent has NOT
   settled. **O(DOF) in memory, not in time.**
2. **The memory model made an out-of-sample prediction that held** — built
   before N=1401 ran, predicted 3,201 MB, measured 3,280 MB (2.4%). Caveat
   recorded in both documents: it is a two-point line through N=501 and
   N=1001, and N=701 sits 10% above it.
3. **Cost breakdown: assembly 0.1–0.6%, CG 99.4–99.9%.** This *superficially*
   confirms Timon's "the key cost should be the solver while the assembly
   should be minimal" — and both documents say explicitly that it must not be
   quoted that way. Matrix-free means every CG iteration IS a Hessian-vector
   product, i.e. an assembly-like pass over all elements; the assembly did not
   get cheap, it moved inside CG where this instrumentation cannot see it.

Remaining gap, judged not worth the compute: **no breakdown for N=501–1001**,
which were solved by commit `5d648d9` before the timing buckets existed and
are skipped on resume. Re-deriving them costs ~3 h of GPU for a number the
four smaller resolutions already establish.

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
