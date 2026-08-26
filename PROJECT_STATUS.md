# PFEM / Transolver Project — Status Tracker

**Read this file FIRST at the start of any new conversation about this project.**
It is the single source of truth for where things stand — more reliable than
chat history, which resets between sessions. Update it whenever a task
finishes or a new one starts.

Last updated: 2026-08-26 (**Drive-vs-summary number audit** — ~205 individual
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

Report file: `PFEM_Transolver_Report_vNN.docx` (latest: **v25**), kept in the
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
