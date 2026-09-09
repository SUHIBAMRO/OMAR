# PFEM / Transolver Project — Status Tracker

**Read this file FIRST at the start of any new conversation about this project.**
It is the single source of truth for where things stand — more reliable than
chat history, which resets between sessions. Update it whenever a task
finishes or a new one starts.

Last updated: 2026-09-09 (**one more gap closed in the Summary's new
top "Response to round-8" section**: Omar asked directly whether the
torch-fem comparison's two caveats -- float32/loose-tolerance vs. our
float64/tight, and "ours" own peak memory never measured (checkpoint-
resumed, not a fresh solve) -- were stated in that TOP section
specifically, not just somewhere else in the document. Checked and
they were NOT (they were only in the older narrative further down).
Since the whole point of that top section is "the answer to everything
in one place, no searching required," added a new "Extra -- GPU-native
solver vs. torch-fem (item #13, related to points 1 and 5)" entry right
after Point 7, before "Tables into figures," with both caveats stated
plainly. Figure count (39) reconfirmed unaffected. Sent to Omar.

Previous update, 2026-09-09 (**cross-checked Timon's two separate emails
against the real documents, per Omar's own question ("did these emails
get answered correctly, or not")** -- both his round-8 email (7 points)
and his separate Sep-6 email answering 3 of Omar's OWN earlier
questions (TensorMesh-vs-torch-fem preference: none, torch-fem is
fine; open-source license: Apache 2.0, wait until arxiv/journal;
GOEE/industrial-relevance paper). Checked systematically against
PROJECT_STATUS.md's own R6/R7/R8 entries and the real docx text, not
just recalled from memory. Result: every point across both emails IS
correctly reflected in the real documents already (continual-learning
citation added, GOEE paper confirmed and correctly NOT overclaimed as
true goal-oriented error estimation, license question resolved) --
**except one real, found bug**: the Summary's own "GPU-native FEM at
finer discretization" paragraph still had the ORIGINAL, now-stale
rhetorical question to Timon ("I would value your view on whether
[torch-fem] is an acceptable substitute... or whether Tensor Mesh was
named for a specific reason") verbatim, even though he'd already
answered exactly this in the Sep-6 email and that answer was already
used to justify choosing torch-fem for item #13 -- the Report's own
equivalent paragraph was already correctly updated to state this as
resolved, but the Summary's was never edited to match. **Fixed**:
replaced the dangling question with a factual sentence citing his
Sep-6 confirmation. Figure count (39) reconfirmed unaffected. Sent to
Omar.

Previous update, 2026-09-09 (**round-8 reply drafted, then restructured
directly into the Summary document itself**: after drafting a combined
reply email covering Timon's all 7 round-8 points
(`advisor_feedback/2026-09-09_reply_to_round8_combined.md`), Omar
pointed out the Summary's own top narrative section ("Summary of what
was done and what came out") does NOT actually answer Timon's 7 points
point-by-point -- checked this directly and confirmed it's true: that
narrative predates round-8, mixes in unrelated/older experiments (its
"Out-of-distribution behaviour" paragraph describes Table 11's single-
shift-level experiment, not Table 25's progressive multi-factor study
point 2 actually asked for), and never mentions the preconditioner fix,
richer MMS family, or Comparison A/B restructuring at all.

**Fixed by inserting a new, clearly-labeled section right after the
title**, before the existing narrative: "Response to Professor
Rabczuk's round-8 feedback (points 1-7)", using the SAME point numbers
Timon used, each with its own final result stated plainly (same content
as the drafted email, condensed) -- so he can read the answer to each
of his own questions in one place without hunting through the rest of
the document. The original narrative section stays below it unchanged,
as supporting detail. Verified the figure count is still 39 (unaffected
by this text-only insertion) before sending. Sent to Omar; the combined
email draft is unaffected/still separate for whenever he wants to
actually send something to Timon by email specifically.

Previous update, 2026-09-09 (**full-document sanity check, per Omar's own
worry about a mix-up**: after all 39 figures were embedded across both
documents, Omar explicitly asked to verify there was no duplication or
misplacement before going further. Checked programmatically, not just
by spot-checking a few: (1) exactly 39 "Figure N." captions in each
document, 1-39, no gaps, no duplicate numbers; (2) exactly 39 embedded
images in each, matched 1:1 to captions; (3) hashed every embedded
image's raw bytes -- zero duplicate image content in either document
(no picture accidentally embedded twice under two different figure
numbers); (4) printed every Figure-17-through-39 caption alongside the
nearest preceding table/paragraph in both documents and confirmed each
one topically matches its anchor (e.g. Figure 31-36's six OOD field
grids each sit right after Table 25/Table 19 as intended, in the
correct B1-NH/MR/AB, B2-NH/MR/AB reading order). Everything checked out
clean -- no fix needed, just confirmed.

Also, per Omar's request, added two new paragraphs to the informal
"plain summary of what was done" narrative at the very top of the
Summary document (right after the existing MMS paragraph, before the
numbered results sections begin) -- covering item #13 (torch-fem
comparison) and item #12 (all 39 figures), in the same plain-language
style as the existing narrative entries there. Deliberately did NOT
touch the Report's own Abstract/Executive Summary (formal paper prose,
different register, and Omar's own wording named only "the Summary"
specifically).

Previous update, 2026-09-09 (**item #12 is now COMPLETELY done, including
the notebook-generated figures**: the 10 remaining field-grid PNGs
fetched from Drive -- B1 mesh-convergence grid, B1 zero-shot grid, all
six OOD grids (B1/B2 x 3 materials, all now showing the corrected ring
hole), and both DD-NO coarse/fine grids -- are embedded as Figures
30-39 in both Report and Summary (`embed_more_figures.py`), on top of
the 13 local-script figures (17-29) from the previous entry. Both docx
files now carry all 39 figures; sent to Omar; the 10 PNGs + the script
pushed to GitHub alongside the earlier 13.

Anchor coverage checked directly per document rather than assumed: the
six OOD figures anchor to Table 25 in the Report (the only table
covering all six geometry x material combinations -- Table 19 there is
B1 x Neo-Hookean only) but to Table 19 in the Summary (which has no
Table 25/26 at all); DD-NO anchors to Table 26 in the Report but to its
own intro paragraph in the Summary (no dedicated table there).

**Real bug caught before trusting the result**: inserting several
figures against one SHARED anchor (the six OOD figures all anchor to
the same table) lands them in REVERSE of insertion-call order, since
each new pair splices in immediately after the anchor, pushing the
previous pair further down -- assigning figure numbers by call order
put "Figure 36" physically before "Figure 31" in an early draft. Fixed
by a placeholder-caption + second-pass-renumber scheme: insert all
figures first with a placeholder, THEN walk the document's own final
paragraph order once to assign real numbers, so numbering always
matches actual physical position regardless of how the insertions
themselves were ordered. Verified by reading the saved file back and
printing the six OOD captions in the order they actually appear.

Previous update, 2026-09-09 (**B2 OOD hole-rendering fix CONFIRMED for all
3 cases**: re-checked Drive after Omar's follow-up notebook re-run --
`fig_B2_neo_hookean_ood_grid.png`, `fig_B2_mooney_rivlin_ood_grid.png`,
`fig_B2_arruda_boyce_ood_grid.png` all now show the correct ring hole
and the clean title, downloaded and visually verified all three
directly (not just checked metadata/timestamps). The earlier note below
about Mooney-Rivlin/Arruda-Boyce being stale is RESOLVED -- that
notebook run only got through Neo-Hookean; a later re-run finished all
6 OOD cases (plus refreshed both DD-NO figures, B1-only so never
affected by the hole bug in the first place). Nothing further needed on
this specific item.

Separately, tried to upload the 13 report figures + both updated docx
to Google Drive via `mcp__Google_Drive__create_file` and confirmed this
is NOT technically feasible through this tool for files this size: it
requires the full base64 payload inlined as one generated tool-call
argument (no path-based upload), which is merely painful for the PNGs
(130-580 KB base64 each) but categorically impossible for the two docx
files (~5.7 MB each, ~7.6M base64 characters -- no single tool call can
carry that). Resolution: sent all 15 files directly to Omar via
SendUserFile instead (already done), for him to upload to Drive
himself if he wants them there -- safer than risking a corrupted
docx from a chunked-reassembly upload attempt.

Previous update, 2026-09-09 (**item #12 is now FULLY DONE**: all 13
table-based figures are embedded directly in the real Report and
Summary docx files as Figures 17-29, both promoted to new
`..._updated_2026-09-09.docx` canonical files, sent to Omar, pushed to
GitHub (script + PNGs, `.gitignore` amended with one narrow exception
for `report_builders/figures/*.png`), and a Drive upload of the same
13 PNGs + both docx files was kicked off in a background agent
(pfem_run/figures for the PNGs, pfem_run root for the docx -- first
time either docx has been placed on Drive; searched first and
confirmed neither existed there before).

Two real things found and fixed on the way, each caught by actually
checking rather than assuming:
1. **Style pass, per Omar's own direct feedback** ("colours I don't
   like, no numbers on bars/points, legend sits on the data"): built
   `report_builders/plot_style.py`, one shared palette (material colour
   is the same everywhere it appears; ditto geometry colour) plus bar/
   line label helpers, applied across all 13 scripts. Two real overlap
   bugs caught by re-inspecting each rendered PNG (not just re-running
   the script): a `bbox_to_anchor`-floated legend colliding with the
   axes title in two figures (OOD degradation, PI-vs-DD), fixed by
   keeping the legend inside the axes with generous y-headroom instead
   of trying to float it above; and two near-coincident line endpoints
   (zero-shot resolution) that no on-plot label placement could avoid
   overlapping, fixed by moving the number into the legend text itself.
2. **Figure-numbering pass, for the docx embedding**: originally
   planned to number the 13 new figures (17-29) in table-number order,
   but checked the ACTUAL physical position of each anchor first and
   found the Report and Summary do NOT lay out sections in ascending
   table-number order (MMS's tables physically precede the B2 fix-
   history tables), and don't even order sections the same way as each
   other -- confirmed the EXISTING Figures 1-16 aren't in strict
   physical order either, so this is normal for this document, but the
   NEW figures were still numbered by each one's own real anchor
   position per document (computed, not assumed) to keep them at least
   internally consistent.

**Also found, unrelated to the above, while spot-checking Drive for the
B2 hole-rendering fix's real result**: only `fig_B2_neo_hookean_ood_
grid.png` on Drive actually shows the corrected ring (downloaded and
visually confirmed) -- `fig_B2_mooney_rivlin_ood_grid.png` and
`fig_B2_arruda_boyce_ood_grid.png` are STILL the old, pre-fix images
(solid disk, no hole, old un-cleaned title with "Tables 19/25" baked
in) even after Omar's own notebook re-run. Section C of `Round6_
Project_Figures.ipynb` most likely didn't finish all 6 OOD cases in
that run. **Not yet told to Omar or investigated further** -- next
session (or later this one) should raise this specifically and get
those two regenerated before considering item #12's notebook-based
figures fully closed (the 13 local-script ones embedded above ARE
fully closed).

Previous update, 2026-09-09 (**real bug found by Omar himself, directly in
a rendered figure, and fixed**: the B2 OOD field-panel figure (`fig_B2_
neo_hookean_ood_grid.png` etc., Tables 19/25) rendered the ring domain
as if part of its inner hole were filled with material. Root cause
confirmed locally (zoomed before/after comparison, no GPU needed): B2
is a quarter ring, R_in=1 to R_out=2 (`data_generate_B2.py`), with a
genuinely empty hole from r=0 to r=1, but `panel_grid_plot.py`'s
`tricontourf` was only ever given the raw (x, y) node cloud, so it fell
back to a plain Delaunay triangulation that has no notion of the hole
— it filled a real, visible angular wedge of it with interpolated
colour, bounded by straight chords instead of the true circular
boundary. Fixed by teaching `panel_grid_plot.py` to accept the real Q4
element connectivity (`quad`, already available on every sample dict
as `s['quad']`) and build an exact `matplotlib.tri.Triangulation` from
it instead of guessing — wired into both `ood_progressive.py` (where
this was spotted) and `resolution_invariance_zeroshot.py` (same shared
utility, not yet hit in practice since Table 12/26 have only been run
on B1 so far, but would have the same bug on B2). Verified end-to-end
locally with a random-initialized model (no real checkpoint needed to
exercise the plotting path) plus a zoomed real-field-values comparison
showing the fix. B1 figures were never affected (no hole in that
geometry). **Committed and pushed; NOT yet regenerated with a real
checkpoint** — the 3 real B2 OOD PNGs (`fig_B2_{neo_hookean,mooney_
rivlin,arruda_boyce}_ood_grid.png`) still need Section C of `Round6_
Project_Figures.ipynb` re-run on Colab (fresh tab from GitHub, per the
known stale-tab lesson) to pick up this fix before they're resent.

Previous update, 2026-09-08 (**item #12 (tables → figures) is essentially
DONE for the whole tracked punch-list** — every table Omar named that
lacked a figure now has one, built with a safety-first, two-track
approach he set explicitly mid-session:

1. **Notebook-based, model-touching figures** (`Round6_Project_Figures.ipynb`,
   `zeroshot_notebooks/cell_project_figures.py`): additive, opt-in
   `--save_sample_plot` flags added directly to the ORIGINAL already-
   validated scripts (`resolution_invariance_zeroshot.py`,
   `ood_progressive.py`), re-invoked with their real original commands
   from `run_manifest.json` — never a separate reimplementation.
   Verified byte-identical numeric output on fresh vs. resumed runs
   before trusting it. Produces: FEM field grid, Table 12 zero-shot
   field grid, 6 OOD-case field grids, DD-NO coarse-vs-fine field grid.
   Titles cleaned of internal notes (item numbers, "no retraining",
   seed indices) per Omar's explicit "publication-ready title" request.
2. **Local scripts reading already-committed numbers**
   (`report_builders/make_figure_*.py`): no model, no new computation.
   For tables with a JSON source, read it directly. For the ~20 tables
   with NO separate JSON anywhere (mesh convergence, training cost,
   latency, break-even, ID/OOD, B2 fix history, operator-vs-FEM,
   PI-vs-DD), built `report_builders/docx_table_map.py`, a reusable
   utility that safely pairs every caption to its table by checking
   BOTH directions and inferring the convention from resolved
   neighbours — needed because one document block (Tables 1/1a/1b/
   2/2a/2b) uses "caption after table," the opposite of everywhere
   else, which would have silently mislabeled all six tables one slot
   off if only one direction were checked. Verified against 57
   captioned tables, zero ambiguity errors.

Real bugs caught and fixed BEFORE anything was sent to Omar, each by
visually inspecting the rendered PNG or double-checking caption text
rather than trusting a first pass: (a) a reaction-force panel silently
overwriting the stress panel in `make_figure_physical_quantities.py`
(a dropped `ax = axes[2]` line); (b) a naive dict-merge letting a
checkpoint-resume artifact (0.76s) silently overwrite the real N=401
solve time (2615.8s) in the solver-scaling figure; (c) the mesh-
convergence caption-direction trap described above; (d) the most
serious one — Table 10 is the GPU-native FEM SOLVER's own timing, NOT
the trained operator's, confirmed only by reading each table's own
caption text (the header rows are identical across Tables 10/10a/10b/
10c and would not have caught this) — an earlier combined figure had
mislabeled it, caught and fixed by splitting into correctly-labeled
`make_figure_operator_latency.py` (Tables 10/10a/10b) and
`make_figure_breakeven.py` (Table 10d, break-even vs. both CPU-FEM and
GPU-FEM).

Full list of local-script figures built and sent, this session: mesh
convergence (Tables 1/1a/1b/2/2a/2b), physical quantities (15/16/17),
zero-shot resolution (12/12b/12c), MMS convergence (22/22a/22b/23/23a),
solver scaling (20/20a/20c), torch-fem comparison (20d), training cost
(5/7/8), operator-vs-GPU-FEM latency (10/10a/10b), break-even (10c/10d),
ID-vs-OOD degradation (11), B2 fix history (13/14), operator-vs-FEM
accuracy/cost across resolutions (18/18a-e), physics-informed vs.
data-driven training (21/21a). Table 9 (correctness check, all PASS)
was explicitly skipped as not figure-worthy — validation only, no
trend to show.

**All scripts and the `docx_table_map.py` utility are committed and
pushed; all figures have been sent to Omar as PNGs via SendUserFile,
in batches, as they were produced and verified.** Nothing has been
embedded into the actual Report/Summary docx files yet (unlike item
#13's Table 20d, which WAS inserted) — that insertion step, plus Omar's
own review/approval of which figures he actually wants kept in the
final documents, is the only remaining work on item #12. Do NOT insert
anything into the docx without Omar's sign-off on the figure set first,
per his standing "explain before touching the real documents" rule.

Previous update, 2026-09-08 (**item #13 is FULLY DONE**, after five real
issues found and fixed one after another on the way there (pyvista/
IPython, two separate torch-fem internal device-default bugs, a stale
in-kernel module cache, and a stale browser tab never picking up any
fix until given a fresh `colab.research.google.com/github/...` link)
— see item #13's row for the full blow-by-blow. **The final result,
all four of Table 20's own resolutions (N=401/701/1001/1401)**:
torch-fem is dramatically faster in wall-clock than this project's own
matrix-free mgv solver — 418x/655x/1197x/936x, honestly NOT monotonic
(peaks at N=1001) — reported with two caveats stated as clearly as the
headline number: the two solvers are not run at matched precision/
tolerance (torch-fem forced to float32/loose vs. our float64/tight,
per Omar's own decision not worth more GPU-hours to close), and "ours"
own peak GPU memory was never captured (resumed from a checkpoint by
design). **Written into both real documents 2026-09-08** (new Table
20d + discussion, Report; matching paragraph + table, Summary),
explicitly tying the result to Timon's own round-8 prediction that an
assembled-matrix solver like TensorMesh would beat this project's own
matrix-free one on raw speed. Per Omar's own instruction, this same
architectural trade-off will also be named explicitly in the eventual
combined reply to Timon. **Next, per Omar's own sequencing ("خلص 13
بعدين الي بعده")**: item #12 (tables to figures) — scope still needs
Omar's own call (which tables, any example figure from Timon's prior
papers), asked and explicitly deferred by him for now.)

Previous update, 2026-09-08 (**item #4 is now fully DONE**: the full
N=701/1001/1401 run finished on Colab, 14.86h total — `cg_failures = 0`
at ALL 7 resolutions in the study (51/101/201/401/701/1001/1401), down
from 20/30/40/80 at 401/701/1001/1401 in the original run. Wall-clock
came in at or better than the rough pre-run estimates (701: ~2.00h vs.
~2.2h est.; 1001: ~4.81h vs. ~4.5h est.; 1401: ~7.57h vs. ~8.9h est.).
Got there across 2026-09-07/08 by: confirming the fix on N=401 first
(cg_failures 20→0, ~43.6 min), then catching and fixing a SECOND
infeasibility bug before ever touching N=701/1001/1401 (their coarsest
hierarchy levels are far too large for the exact dense solve N=401
used — added a size-based approximate-CG fallback, validated on CPU),
then running all three for real. One caveat carried forward, not yet
resolved: the fitted convergence RATES (not cg_failures/wall_clock) at
N=1001/1401 need a bigger fine_N reference to be fully trustworthy —
see item #4's row. Next: Omar's call on whether/how to write this into
the real report documents, and item #3 can now be closed using this
data. Earlier the same two days: items 5 and 8 done directly
in the report; items 6 and 7 (Colab notebooks built earlier the same
day, item 7's notebook had a stale-artifact bug found and fixed) have
SINCE BOTH FINISHED RUNNING and their results are written into both
real documents (item #6: Table 25 + §8.6; item #7: new §8.7 + Table
26) — nothing further pending on either; item 3
decided (Option A, current problem size is justified — see below) using
existing Table 6a data, then a peak-stress QoI was added to the
underlying script per Omar's own condition, and that real GPU sweep
FINISHED 2026-09-07 (2.59h) — **but the result was measuring a bug, not
a real finding**: Omar questioned the "peak stress just doesn't converge
cleanly" write-up instead of accepting it, which led to finding that the
peak-stress "reference" value was silently redefined at every mesh
resolution (max over a growing point set) instead of being one fixed
target. Fixed same day and verified on CPU on both geometries; the real
corrected sweep (combined with item #4's block2x2 re-run) is still
pending — see item 3's row for the full story; **item 9 (richer MMS family) and item 10 (MMS energy norm)
are both DONE** — implemented, safety-checked so nothing
already-published changed silently, full sweeps run and committed, all
rates matching theory on both element orders; **item 4 (block-Jacobi
preconditioner) is implemented, validated** against the dense CPU
reference on both B1 and B2 (including B2's mixed-fixed-DOF nodes), and
committed, kept strictly opt-in — the actual N=1001/N=1401 Colab re-run
this item exists to produce is still pending, needs Omar's GPU; **items
6, 7, and 14 are now fully DONE, including being written into the real
report and summary documents** (2026-09-07) — see each item's row.

**Important discovery while doing that writing-in, 2026-09-07**: Omar
uploaded the actual `.docx` files he sends to Timon directly to this
session (`PFEM_Transolver_Report.docx`, `Work_Summary_.docx`), resolving
the earlier "can't find the file" blocker. Checking them against the
known anchors from `report_builders/make_v53.py` through `make_v56.py`
showed these real files are **missing items #1 (continual-learning
citation), #2 (Table 21 DD-NO wall-clock rows), #5 (batch-size-1
labeling), and #8 (Table 21a budget comparison)** — i.e. the actual
document Timon has predates v53, even though this file's own earlier
entries (and the report_builders script history) describe those as
already done in "v53–v56". Those scripts were written and validated in
past sessions, but this session could never find the real `.docx` to
run them against, so it seems they were never actually applied to the
file Omar sends out.

**Resolved 2026-09-07 (Omar confirmed: yes, backfill all of them,
correctly, in the right place)**: items #1, #2, #5, #8 were then also
applied to the real file, alongside a fix to item #6's own placement
(the first pass had put the new OOD content BEFORE the sentence "has
not been measured for the other five cases," which read as
self-contradictory -- moved to come after it instead, so the sentence
poses the question the next paragraphs answer). One real content gap
found doing this: the real file has NO bracket-numbered References
section anywhere, so item #1's continual-learning citation was added
inline (author names + arXiv ID in the sentence itself), the same way
item #14's citation already was, rather than via a "[5]" cross-reference
into a bibliography that does not exist in this file. Item #5's point
turned out to already be stated in the Summary's own words (Table 10d's
text already calls batch-size-1 "the deployment case"), so nothing was
added there for #5. All of items #1/#2/#5/#6/#7/#8/#14 are now applied
to both real documents, via `report_builders/add_items_1_2_5_6_7_8_14_report.py`
and `add_items_1_2_6_7_8_14_summary.py` (superseding the earlier,
6/7/14-only scripts). The fully updated files were sent back to Omar
directly this session.

**Item #11 also DONE 2026-09-07** (extended MMS to Mooney-Rivlin AND
Arruda-Boyce, not just one material as the item asked) — caught and
fixed a real generalization bug in `mms_study.py` (hardcoded to exactly
two material parameters, which only meant something for Neo-Hookean)
the moment this was actually tried; verified as a pure generalization
(Neo-Hookean's own numbers unchanged) before trusting the two new
materials' results, both of which converge exactly at theory rates on
both element orders. See item #11's row for the full story.

**Item #4, time estimate for N=701/1001/1401 (2026-09-07, rough, explicitly uncertain)**:
a naive linear-in-DOF extrapolation from N=401's confirmed 2615.8s at
n_dof=321,602 (assuming multigrid's own iteration count per Newton
solve stays roughly flat with N, which is the theoretical hallmark of
correctly-working multigrid but not yet PROVEN to hold at this scale)
gives **N=701 (n_dof~982,802, 3.06x): ~2.2 h; N=1001 (n_dof~2,004,002,
6.23x): ~4.5 h; N=1401 (n_dof~3,925,602, 12.21x): ~8.9 h**. Treat these
as a floor, not a promise: N=701/1001/1401 all use the NEW approximate
coarse-solve fallback (see item #4's row) rather than N=401's cheap
exact one, and have fewer/shallower hierarchy levels (3-4 vs. N=401's
5), so the real per-iteration cost and the outer CG iteration count
could both come in higher than this straight-line extrapolation
assumes. Run N=701 for real before trusting the 1001/1401 numbers.

**Item #4, continued 2026-09-07**: block2x2 Stage 1 finished with a
NEGATIVE result (no cg_failures improvement at N=401/701) — rather than
accept that as a stated limitation, Omar directed building the actually-
correct fix, a genuine geometric multigrid V-cycle (`multigrid_precond.py`,
new file), validated on CPU (bugs found and fixed, iteration-count
advantage growing with N: 3.0x→5.2x→9.7x at N=9/33/65) before ever
touching GPU time. Omar then ran it for real on N=401 (`Round6_MGV_Recheck_N401.ipynb`):
**CG now genuinely converges (79 iterations, `converged=True`)** where
plain Jacobi and block2x2 both hit the iteration cap every time — the
core correctness win this whole item exists for. But each CG solve cost
~260s, ~82x more per-iteration than the old non-converging run (projected
~87 min total for N=401 vs. ~27 min old) — a real, diagnosed performance
regression, not a vague "multigrid is slower" shrug: `_build_dense_factor`
was building the coarsest level's dense tangent matrix via a SERIAL
Python loop of one autodiff call per free DOF, and `mg_max_levels=4` was
capping N=401's hierarchy at `[401,201,101,51]` instead of the
structurally-deepest `[401,201,101,51,26]`, leaving the coarsest level at
~5000 free DOFs instead of a genuinely small one — thousands of tiny
sequential GPU dispatches, each paying kernel-launch overhead far bigger
than its actual FLOP cost. **Fixed same day**: `_build_dense_factor` now
builds the dense matrix via chunked `torch.func.vmap(matvec)` batches
(same chunk-and-accumulate pattern `compute_jacobi_diagonal` already uses,
for the same reason) instead of a serial loop, and `mg_max_levels`'s
default raised from 4 to 8 so coarsening reaches the real structural
floor for a given N (N=401 now reaches N=26, not just N=51). **Re-verified
on CPU after the rewrite, same discipline as before**: all of
`validate_multigrid_precond.py`'s stages (1, 2b1, 2b2, 2c) reproduce
BYTE-IDENTICAL iteration counts and solutions to before the rewrite
(e.g. N=33: jacobi 734 / mgv 140 iters, rel diff 1.1e-12 — unchanged) —
proof the vmap rewrite is a pure performance change, not a behavior
change. The real GPU speedup on N=401 itself is NOT yet re-measured (CPU
can confirm correctness but not the GPU-dispatch-overhead saving this fix
specifically targets) — that needs Omar to pull this fix and resume
`Round6_MGV_Recheck_N401.ipynb`. Checkpointing is per-Newton-iteration
(confirmed earlier), so resuming after this code change safely continues
from N=401's last completed Newton iteration rather than restarting; only
N=401 was ever forced-fresh by that notebook, so nothing else is affected.
See item #4's row for the full detail.
**This work queue is now the single authoritative list —
read it first, before anything else in this file.**

---

# 🔢 WORK QUEUE — do these in order, one at a time

Everything actionable from today's two emails (round 7 and round 8),
merged into one list. Duplicates removed (round 8's points 1 and 5 are
one item, not two — see row 5). Status is exact: "done" means already in
report v54; nothing else below is started.

| # | Item | From | Status | Real dependency |
|---|---|---|---|---|
| 1 | Continual-learning citation added to §8.6 | R7 | ✅ **Actually applied to the real file 2026-09-07** — the "report v53" status was stale (the script existed and was validated in a past session, but that session never had access to the real .docx to run it against; see the version-gap note above). Wang et al. cited inline (this document has no bracket-numbered References section anywhere, so no "[5]" cross-ref was used) as a candidate future-work direction, right after the OOD section's new Table 25 content | — |
| 2 | DD-NO wall-clock time added to Table 21 | R8.4 | ✅ **Actually applied to the real file 2026-09-07** — same stale-status story as #1. Table 21 now has 5 rows: the two original error rows plus wall-clock (physics-informed 2,873.8s/3,108.9s; data-driven 1,458.3s/1,463.0s) | — |
| 3 | Decide: is B1/B2 at its current size a demanding-enough FEM problem for an NO to be worth using? (or scale up — e.g. Timon's tire example) | R8.1 | 🟡 **Provisionally answered 2026-09-06 (Option A: justify current size, not scale up), NOT final — Omar set two explicit conditions before this counts as closed for the paper.** Initial answer used Table 6a (B1×Neo-Hookean vs. ~10M-DOF reference, already in report v56): ≤1% error needs N=201 (~15 min), ≤0.5% needs N=401 (~30 min) — clean data, no caveats. A tighter ≤0.2%/N=1401 claim was checked and WALKED BACK: the report's own text says N=1001/1401 hit the CG iteration cap without reaching cg_tol (extra non-discretization error), and separately that H1/energy never reach the advisor's own already-stated 1e-4 target even at N=1401 — the honest framing is "doesn't even get there with tractable compute," not a clean "0.2% costs 3 hours" number. Revised draft reply written accordingly (`advisor_feedback/2026-09-06_reply_to_round8_point1.md`), NOT YET SENT (Omar's call, and it's explicitly a progress update, not a final answer). Omar's 2 conditions for treating this as final: **(a)** re-measure after the preconditioner fix (item #4) — makes #4 a hard prerequisite for closing #3, not just "nice before #13"; **(b)** add an engineering QoI like peak stress, since Table 6a's L2/H1/energy are all field norms, not the literal example ("maximum stresses") the advisor's email named. **(b) is now done at the code level**: `compute_peak_stress_error()` added to `high_dof_convergence_study.py` (peak Frobenius-norm PK1 stress + a stress-field L2 norm, reusing the same PK1-via-autograd recipe `physical_quantities_eval.py` already validated), wired into `main()`'s per-resolution loop and the JSON report/convergence-rate output. Verified end-to-end on CPU at tiny scale on BOTH geometries — and this caught a real bug before it shipped: the first version passed raw Cartesian (x,y) points straight to `AnalyticFieldB2`, which expects polar (theta, r) (confirmed against its own docstring and against `precompute_element_params_B2`'s already-validated conversion) — B1 would have been silently fine, B2 would have silently sampled the wrong material value at every point. Fixed and re-verified on both B1 (Neo-Hookean) and B2 (Neo-Hookean): sane, non-degenerate peak-stress values, decreasing error with refinement, convergence rate in the same range as H1's (physically expected, since stress derives from the same displacement gradient). **Colab notebook to run the real sweep is now built**: `zeroshot_notebooks/cell_high_dof_stress_qoi.py` / `Round6_HighDOF_PeakStress.ipynb`, registered in `make_round6_notebooks.py`, verified 40/40 via `check_notebooks.py`. Re-runs Table 6a's exact B1×Neo-Hookean setup (same resolutions, same ~10M-DOF reference) so the new column is directly comparable. **Original Table 6a checkpoint directory FOUND on Drive 2026-09-06** (Omar reported the run he started went fresh from N=2236, which meant the checkpoint path the notebook first guessed was wrong — searched Drive directly rather than guessing again): `/content/drive/MyDrive/pfem_ckpt` (a sibling of `pfem_run`, not inside it) contains `fine_B1_neo_hookean_Q4_N2236.pt` (the ~10M-DOF reference itself — the single most expensive solve in the study), plus `coarse_B1_neo_hookean_Q4_N1001.pt` and `_N1401.pt` (the two next most expensive). Checked and confirmed NOT present anywhere on Drive: checkpoints for N=51/101/201/401/701 — those five must still be solved fresh (~2.6 h total from Table 6a's own recorded timings). Notebook's `CHECKPOINT_DIR` default updated to this real path, rebuilt, re-verified (40/40). **Real expected cost with this fix: ~2.6 h, not 8–15+ h.** 🟢 **RUN FINISHED 2026-09-07** (2.59 h on A100): `highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean.json`, fetched from Drive and committed — **but Omar caught a real problem in it by asking "هل في طريقه صحيحه لعمله ولا المشكله باشي تاني؟" (is there a correct way to do this, or is the problem something else?) rather than accepting the first write-up's "this QoI is just inherently noisy" framing.** That framing was WRONG, or at least premature: `compute_peak_stress_error()`'s original implementation computed both the coarse prediction AND the reference target as `max(|P|)` over the COARSE mesh's OWN Gauss points — a point set that gets denser as the coarse N under test increases, so "peak_stress_ref" was silently redefined at every row instead of being one fixed number. Confirmed directly in the retracted data: `peak_stress_ref` itself climbed from 13.9 (N=51) to 39.3 (N=1401), a ~2.8x range — a "reference" is not supposed to move. This, not (primarily) inherent pointwise-QoI roughness, is what produced the directionless noise and the negative overall fit rate reported earlier today. **Fixed same day**: new `find_fine_peak_stress()` locates the fine reference's own true peak-stress point ONCE per order (from its own dense Gauss points); every coarse resolution is now compared against that SAME fixed physical point (exact FE point evaluation, reusing the already-validated `evaluate_fe_field_and_gradient` machinery on the coarse mesh too). Verified on CPU on both B1 and B2: `peak_stress_ref` is now identical across different coarse N in the same sweep (e.g. 12.4648592 at both N=6 and N=11 on B1; 12.2959739 at both N=6 and N=11 on B2), and the error now moves the expected direction as N grows (30.8%→20.2% on B1, 6.3%→1.4% on B2) instead of drifting incoherently. **The earlier "peak stress doesn't converge, may need GOEE" narrative is RETRACTED** — that conclusion was drawn from a buggy measurement; it might still turn out to be true once the corrected number is measured for real (pointwise QoIs genuinely are harder than global norms in general), but it can no longer be claimed from what was run. The GPU sweep needs to be re-run with the fixed code — combine this with item #4's `--precond_kind block2x2` re-run on N=401-1401 so both fixes land in one run instead of two. **Before asking for another GPU run, the fix itself was verified rigorously on CPU** (Omar explicitly asked "متأكد من الطريقه بشكل صحيح، بديش اقعد اشغل ساعات عالفاضي" — make sure the method is actually correct before hours are spent on it again): (1) an IDENTITY check (solve one mesh, treat it as both coarse AND fine) gives EXACTLY 0.0 relative error — proves the fixed-point logic has no bug, not just "looks plausible"; (2) a 3-point CPU sweep (N=6/11/17, B1) shows `peak_stress_ref` is now byte-identical across all 3 rows (13.282792), the error decreases monotonically (34.9%→24.6%→16.1%), and the fitted convergence rate is a sane, positive 0.65 — lower than H1's 0.98, which is exactly the textbook-expected relationship between a pointwise QoI and a global norm, not noise; (3) the peak location is physically consistent (lands near the same domain corner regardless of which fine-mesh resolution finds it). **Colab notebooks built for the real re-run, staged to avoid burning GPU hours on an unknown**: `zeroshot_notebooks/cell_precond_recheck_stage1.py` / `Round6_Precond_Recheck_Stage1.ipynb` (forces fresh block2x2 solves at ONLY N=401/701 — known costs from the original run, ~27+75 min — reuses everything else instantly) and `cell_precond_recheck_stage2.py` / `Round6_Precond_Recheck_Stage2.ipynb` (forces fresh block2x2 solves at N=1001/1401 — UNKNOWN cost, since these two have never been solved from scratch anywhere in this project's history; every past run free-rode on a pre-session checkpoint). Both registered in `make_round6_notebooks.py`, verified 42/42 via `check_notebooks.py`. **Explicit instruction in Stage 1's own printed output: only run Stage 2 if Stage 1's checklist looks right** (CG failures reduced at 401/701, peak_stress_ref identical across rows) — no point risking hours on the two most expensive, least-known resolutions before confirming block2x2 helps at all. **Stage 1 FINISHED 2026-09-07 — see item #4's row for the full result: block2x2 did NOT reduce cg_failures at N=401/701 (20/30, identical to plain Jacobi), so Stage 2 is not recommended as things stand. The peak-stress fix itself IS confirmed correct on this real GPU data** (`peak_stress_ref` identical across all 5 rows, error decreasing monotonically 62.6%→21.2%) — so condition (b) is genuinely done, but condition (a) (the preconditioner actually fixing CG convergence) is NOT met, meaning #3 still cannot be closed on today's numbers without either a better preconditioner or accepting the CG-cap limitation explicitly 🟢 **Condition (a) fully closed 2026-09-08 — Omar's explicit choice was to rebuild Table 6a itself, not just add a pointer note.** Table 6a's N=401/701/1001/1401 rows now use the genuinely CG-converged multigrid solves (item #4) instead of the old CG-capped plain-Jacobi ones. The rebuild landed almost exactly where the earlier text's own argument predicted it would: L2/H1/energy values at all four rows changed by at most a few parts in the last reported digit (e.g. N=1001's L2 moved from 5.18e-6 to 5.15e-6), and the least-squares fit across all seven resolutions moved from L2 p=1.58/H1 p=0.73/energy p=0.87 to L2 p=1.57/H1 p=0.73/energy p=0.86 -- within 0.01 everywhere -- confirming the report's own prior claim that Newton's absolute-residual convergence test had already bounded the damage from CG truncation. What DID change substantially is the wall-clock column, now reporting real converged costs instead of truncated-budget costs (and, at N=1001, replacing a checkpoint-resume-artifact placeholder that was never a real number): 401 1802.2s->2615.8s, 701 5940.1s->7205.4s, 1001 "0.9s*"->17314.8s, 1401 11871.5s->27257.4s. The table's own two-caveats footnote was rewritten to one: the CG-cap caveat is removed (resolved), the fine_N-insufficiency caveat for N=1001/1401's rate contribution is kept (a genuinely separate, still-open concern, unaffected by the preconditioner). Written into both real documents (`PFEM_Transolver_Report_updated_2026-09-08.docx`, `PFEM_Work_Summary_updated_2026-09-08.docx`), verified via python-docx readback before promoting as canonical. **Condition (b) was already done** (peak-stress QoI, see above). **Both of Omar's conditions for closing item #3 are now met.** **Omar's explicit decision 2026-09-08: do NOT send the round-8 point-1 reply yet, even though both conditions are met.** Hold it until every other outstanding point/item in the work queue is also done, then send ONE combined reply to Timon covering everything together, rather than one partial reply now and more later. So: item #3 itself is technically closable (both conditions met, Table 6a rebuilt), but the actual advisor-facing communication step is deliberately deferred — do not draft-finalize or send `advisor_feedback/2026-09-06_reply_to_round8_point1.md` (or any other pending reply) until told the work queue is fully done | None right now -- next action on this file is "wait," not a technical task. When the full work queue (items #12/#13 and anything else still open) is finished, revisit and rewrite this draft with final numbers (it currently still has RETRACTED placeholder numbers) as part of one combined reply, not before |
| 4 | Improve/benchmark the GPU-native solver's preconditioner; rerun N=1001 and N=1401 to full CG convergence | R8.1 + R8.5 (same concern) | ✅ **DONE 2026-09-08 — this item's actual deliverable is now complete: N=1001 and N=1401 (and N=701) all reach full CG convergence.** **A 2x2 block-Jacobi preconditioner is implemented, validated, and committed 2026-09-06; Stage 1 of the real re-run (N=51-701) FINISHED 2026-09-07 with a NEGATIVE result for this item's own goal — see below.** `matrix_free_solver.py` gained `compute_block_jacobi` (per-node 2x2 tangent-stiffness block, same vmap+hessian+scatter-add construction as the existing scalar `compute_jacobi_diagonal`, just a 2x2 local extraction instead of a diagonal one) and `make_block_jacobi_apply` (builds the CG apply closure, restricted to free DOFs). The real design risk was B2's boundary conditions: a symmetry line can fix only ONE of a node's two DOF components (confirmed in `validate_matrix_free_solver.py`: `theta0_nodes` fix only u_y, `thetahalfpi_nodes` fix only u_x), so a naive 2x2-per-node solve doesn't universally apply post-BC-elimination. Handled by detecting, per node, whether both DOFs are free (real 2x2 block-solve, closed-form inverse with a near-singular fallback to plain diagonal) or only one is (falls back to that DOF's own scalar diagonal entry — exactly today's Jacobi, so a partially-fixed node loses nothing). **Kept strictly opt-in, following the same discipline as item #9**: `solve_matrix_free`'s existing `use_jacobi` flag and every existing call site are untouched; a new `precond_kind="jacobi"` parameter defaults to the exact current scalar-Jacobi code path (byte-for-byte, since precond_kind only branches when explicitly changed), with `precond_kind="block2x2"` as the new opt-in choice. Threaded through to `high_dof_convergence_study.py`'s `solve_one()`/`main()` as a new `--precond_kind` CLI flag (default `jacobi`, unchanged behavior). **Validated against the dense CPU reference** (`validate_matrix_free_solver.py`, N=11, neo_hookean): B1 default-jacobi PASS (max rel diff 1.23e-13, 1860 CG iters, unchanged from before this change); B1 block2x2 PASS (1.18e-13, 1805 CG iters); B2 default-jacobi PASS (1.80e-13, 2754 CG iters); **B2 block2x2 PASS (1.86e-13, 2529 CG iters)** — this is the critical test since B2 is where the mixed-fixed-DOF fallback actually triggers, and it matches the dense reference to the same machine-precision level as plain Jacobi, confirming the fallback logic is correct. Also smoke-tested end-to-end through `high_dof_convergence_study.py --precond_kind block2x2` on B2 (tiny N, CPU) with no errors. CG iteration counts dropped modestly at this small N (B1 ~3%, B2 ~8%) — too small a problem to show the preconditioner's real value; the effect is expected to grow with problem size/heterogeneity, which is exactly what re-running N=1001/N=1401 (this item's actual deliverable) would show  **Stage 1 result (2026-09-07, real A100 run, `high_dof_stress_qoi_B1_neo_hookean_block2x2_stage1.json`, committed)**: `cg_failures` at N=401 is 20 and at N=701 is 30 — EXACTLY the same as the original plain-Jacobi run this item exists to improve on. Wall-clock is also essentially unchanged (1585.4s vs. 1603.2s at N=401; 4498.2s vs. 4494.6s at N=701, actually marginally slower). **The block2x2 preconditioner provides no measurable benefit for CG convergence at these resolutions.** Per Stage 1's own explicit checklist ("only run Stage 2 if cg_failures reduced"), this does not clear that bar — reporting honestly rather than spending Stage 2's much larger, unknown GPU cost (N=1001/N=1401) on a fix that demonstrably does not solve the problem it was built for. **Stage 2 is NOT recommended as things stand.** One genuinely good, separate finding from the same run: item #3's peak-stress bug fix is now confirmed on real GPU-scale data, not just the small CPU checks — `peak_stress_ref` is byte-identical across all 5 rows (43.47671959446411) and `peak_stress_rel_err` decreases monotonically (62.6%→54.1%→43.9%→32.0%→21.2%) as N grows. 🟢 **Second attempt: a genuine geometric multigrid V-cycle, built and validated 2026-09-07, replacing block2x2.** Omar's explicit direction after seeing block2x2 fail: build the actually-correct fix, not accept the limitation. IC0 (incomplete Cholesky) was considered and rejected -- it needs the real sparse tangent matrix, which conflicts with this whole solver's reason for existing (matrix-free, never forms K). Geometric multigrid needs no matrix either, only the ability to apply K's action on a hierarchy of coarser meshes of the same domain -- both B1 and B2 are already structured (i,j) grids (confirmed directly in build_mesh_and_bcs), exactly what classical geometric MG is built for. New module `multigrid_precond.py`: bilinear prolongation P (verified partition-of-unity), restriction R = P^T/4 (standard variational scaling), a symmetric V-cycle (damped-Jacobi pre/post-smooth, exact dense-LU coarsest-level solve factored ONCE per Newton iteration and reused for every V-cycle call within it -- an earlier version re-ran CG from scratch at the coarsest level on every V-cycle call instead, correct but far too slow, caught by a tiny smoke test hanging over a minute before this fix). **Two real bugs found and fixed during validation** (both caught by `validate_multigrid_precond.py`'s own checks, not assumed away): (1) the coarsest-level-CG performance bug just described; (2) a hierarchy-construction bug that reset a middle level's restriction operator to None on every loop iteration -- invisible with exactly 2 levels (a 2-level N=9 smoke test passed cleanly) but crashed immediately the moment a 3-level N=33 test ran, which is exactly why both smoke tests were run rather than stopping at the first PASS. **After both fixes, validated on CPU against the existing plain-Jacobi solve on BOTH B1 and B2** (including B2's own partially-fixed-DOF nodes, exactly where block2x2 needed its own special-case handling): same converged solution to ~1e-12 relative difference every time, zero cg_failures either way, and the iteration-count advantage over plain Jacobi GREW with problem size instead of staying flat -- 3.0x fewer iterations at N=9, 5.2x at N=33, 9.7x at N=65 (178→60, 734→140, 1471→152). A growing advantage with N, not a flat or shrinking one, is the actual signature of correctly-functioning multigrid (a real fix for the underlying conditioning problem), which is why this is worth spending real GPU time on where block2x2 was not. Wired into `high_dof_convergence_study.py` (`--precond_kind mgv`, builds the hierarchy via the same `build_mesh_and_bcs` every other resolution already uses) and a new Colab notebook, `Round6_MGV_Recheck_N401.ipynb`, forces a fresh mgv solve at ONLY N=401 (the cheapest resolution block2x2 failed at) before committing to N=701+. 🟢 **Real N=401 GPU run (2026-09-07): CG now genuinely CONVERGES (79 iterations, `converged=True`)** — the first time any preconditioner has actually fixed CG convergence at this resolution, where plain Jacobi and block2x2 both hit the 2000-iteration cap every time. This is the core correctness win. **But it came with a real performance cost**: each CG solve took ~260s, ~82x the per-iteration cost of the old (non-converging) run, projecting ~87 min total for N=401 vs. ~27 min old. Not accepted as an inherent multigrid trade-off without checking first -- diagnosed by re-reading `multigrid_precond.py` against the timing data: `_build_dense_factor` built the coarsest level's dense tangent matrix via a SERIAL Python loop (one autodiff `matvec` call per free DOF), and `mg_max_levels=4` in `high_dof_convergence_study.py`'s `solve_one()` capped N=401's hierarchy at `[401,201,101,51]` instead of the structurally-deepest `[401,201,101,51,26]` (`coarsen_N(26)` returns `None`, the true floor) -- leaving the coarsest level at ~5000 free DOFs, meaning ~5000 sequential GPU-dispatch-heavy autodiff calls per Newton iteration. **Fixed same day**: `_build_dense_factor` now applies `matvec` to chunked batches of unit basis vectors via `torch.func.vmap` (chunk_size=500, same accumulate-by-chunk pattern `compute_jacobi_diagonal` already uses for the identical GPU-dispatch-overhead reason) instead of one call per column; `mg_max_levels`'s default raised 4→8 so coarsening reaches the real structural floor instead of an arbitrary early cap. **Re-validated on CPU immediately after, before trusting the rewrite**: `validate_multigrid_precond.py` stages 1 (interpolation operators), 2b1 (B1, N=9), 2b2 (B2, N=9), and 2c (B1, N=33, exercises the 3-level hierarchy wiring) all reproduce the exact same iteration counts and converged solutions as before this change (e.g. N=33: jacobi 734 / mgv 140 CG iterations, relative solution difference 1.1e-12 -- identical to the pre-rewrite numbers), confirming this was a pure performance change with no behavior change. 🟢 **Re-run finished 2026-09-07, confirms the fix worked on real GPU data** (`highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json`, fetched from Drive and committed): at N=401, `cg_failures=0` (was 20 with both plain Jacobi and block2x2), `wall_clock_s=2615.8` (~43.6 min, down from the pre-fix rewrite's projected ~87 min, and only ~1.6x the old NON-converging Jacobi run's ~27 min) -- every one of the 20 Newton iterations' CG solves reports `converged=True`, with a steady-state cost around 130s/~153 iterations per solve that held essentially flat across all 10 load steps (not a fluke on one lucky step). `peak_stress_rel_err` at N=401 is 31.99%, consistent with block2x2 stage 1's own 32.0% at the same N -- the physics is sane and reproducible across preconditioner choices, only the CG behavior differs. Convergence rates across all 4 resolutions (51/101/201/401), now with N=401 genuinely converged instead of capped: L2 fit p=1.43, H1 fit p=0.77, energy fit p=0.76, peak-stress fit p=0.32. **This clears the notebook's own stated bar for proceeding** ("only continue to N=701+ if this shows a genuine reduction in cg_failures") cleanly -- 20 failures to 0, at a reasonable (not runaway) wall-clock cost.

🔴→🟢 **Before building the N=701/1001/1401 notebooks, a SECOND, more serious performance bug was found and fixed the same day, this time by checking the structural facts rather than assuming N=401's fix would generalize.** `coarsen_N` requires `(N-1)` even at every halving to keep the coarser grid's nodes an exact subset of the finer one -- so how many times a given N can coarsen depends on how many factors of 2 divide `(N-1)`, NOT on N's own size. Computed directly for every one of this study's own standard resolutions (`coarsen_N` chains, no assumption): N=401 -> `[401,201,101,51,26]` (lucky: 400=2^4*25, 4 halvings, coarsest ~1300 free DOF, cheap dense solve) but **N=701 -> `[701,351,176]` and N=1401 -> `[1401,701,351,176]` both bottom out at N=176 (~62,000 free DOF, since 700=2^2*175 and 1400=2^3*175 only carry 2-3 factors of 2), and N=1001 -> `[1001,501,251,126]` at N=126 (~32,000 free DOF)**. Building an exact dense LU factorization at 32,000-62,000 free DOF is not merely slower than N=401's ~1,300 -- it is computationally infeasible (dense LU is O(n^3) time / O(n^2) memory; a rough estimate put a SINGLE factorization at tens of hours, and a solve needs about 20 of them). Would have either crashed (OOM) or run for days on Colab, discovered before spending a minute of real GPU time on it. **Fixed by adding a size-based fallback to `build_mg_precond_apply`**: a new `coarse_direct_max_free` threshold (default 4000, comfortably above every already-validated case and comfortably below the 32k-62k problem cases) -- at or below it, the existing exact dense LU path is used completely unchanged; above it, the coarsest level is solved approximately instead, via a matrix-free CG call (reusing the solver's own `conjugate_gradient`) preconditioned by that level's own Jacobi diagonal and capped at a modest `coarse_cg_iters=50` (a standard, textbook "inexact coarse solve" multigrid variant -- not the earlier, already-rejected "full CG to tight tolerance every V-cycle call" design this file's own history describes and abandoned for being too slow even at SMALL coarsest sizes; here the cap is deliberately loose, cheap by design, not driven to convergence). A one-line print now announces which mode is active (`[mgv] coarsest-level solve: exact dense LU (n_free=...)` or `approximate CG fallback (...)`) so this is visible in Colab output, not silent. **Validated on CPU before trusting it**: (1) re-ran `validate_multigrid_precond.py` stages 1/2b1/2b2/2c with the new code and DEFAULT threshold -- byte-identical iteration counts/solutions to before this change (N=33: 140 mgv iterations, 1.136e-12 relative diff, unchanged), confirming zero effect on every already-validated case; (2) forced the new fallback path on artificially (via a monkeypatched near-zero threshold) on the already-known-good N=33 case, where the true coarsest level (N=9) is tiny enough that even the capped 50-iteration CG solve converges it almost exactly -- result: 140 mgv iterations, 2.55e-12 relative diff, statistically the same as the dense-exact baseline, confirming the fallback logic itself is correct, not just plausible. **The real GPU behavior at N=701/1001/1401's genuinely-large coarsest levels (where the fallback will be a truly approximate, not near-exact, coarse solve) is NOT yet measured** -- CPU validation proves the code path is correct, not what it costs or how many extra fine-level CG iterations an approximate coarse solve will need at real scale. 🟢 **N=701 FINISHED 2026-09-08, confirms the fallback works at real scale, and beat its own rough estimate**: `n_dof=982,802`, `cg_iters_total=2710` over 20 Newton iterations, **`wall_clock_s=7205.4` (~2.00h, better than the ~2.2h rough estimate)**, every iteration `converged=True` (cg_failures=0, same clean result as N=401). Steady-state cost per Newton iteration ~125-155 CG iterations at ~330-410s each -- the `approximate CG fallback (n_free=61600 > 4000, capped at 50 iters)` line confirmed the new threshold triggered exactly as designed. `peak_stress_rel_err` 21.2%, continuing the same monotonic decrease with N already seen at 401 (32.0%) and in the block2x2 stage-1 data. **A new, separate methodological finding surfaced right after N=701 finished, unrelated to the preconditioner itself**: the script's own built-in check warned `fine_N=2236 is only 2.2x N=1001 ... will flatten (underestimate) the measured convergence rate`, before N=1001 started -- the fine reference mesh (N=2236) that was sized to be a trustworthy reference for N up to ~401/701 is no longer big enough relative to N=1001/1401 to give a trustworthy fitted convergence RATE at those two resolutions (2236/1001=2.2x, 2236/1401=1.6x, both below the script's own stated 4x safety margin). This does NOT affect this item's own goal (cg_failures/wall_clock are reference-mesh-independent), but DOES mean the L2/H1/energy/peak-stress RATES reported for N=1001/1401 rows should be flagged as measured against an insufficiently-fine reference, not trusted at face value for item #3's own accuracy claims -- a bigger fine_N re-run would be needed later if those two rows' rates specifically need to be quoted. **N=1001 now running**: coarsest level correctly triggered the fallback (`n_free=31500 > 4000`), but its first two Newton iterations show a jump to 363 CG iterations (vs. N=701's steady ~125-155) -- worth watching once more steps land, not yet a conclusion.

🟢🟢🟢 **FULL RUN FINISHED 2026-09-08 -- item #4 is DONE.** `Round6_MGV_Recheck_N701_1001_1401.ipynb` ran to completion, 14.86h total on an A100, all three of N=701/1001/1401 solved in one process as designed. Fetched `high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json` from Drive, verified, and committed. **The headline number**: `cg_failures = 0` at every single one of the 7 resolutions in this study (51/101/201/401/701/1001/1401) -- the original run this whole item exists to fix had 20/30/40/80 CG failures at 401/701/1001/1401 respectively; that is now completely eliminated. Final wall-clock, each beating or landing close to its own rough pre-run estimate:

| N | cg_failures (old -> new) | wall_clock_s | pre-run estimate |
|---|---|---|---|
| 701 | 30 -> 0 | 7,205.4 (~2.00h) | ~2.2h (beat it) |
| 1001 | 40 -> 0 | 17,314.8 (~4.81h) | ~4.5h (~7% over) |
| 1401 | 80 -> 0 | 27,257.4 (~7.57h) | ~8.9h (beat it) |

`peak_stress_rel_err` decreases smoothly and monotonically across ALL 7 resolutions (62.6%->54.1%->43.9%->32.0%->21.2%->14.2%->7.7%) -- clean, physically sane behavior with no discontinuity introduced by the preconditioner swap or the coarse-solve fallback kicking in partway through. The `[mgv] coarsest-level solve: approximate CG fallback` line appeared exactly at N=701/1001/1401 as designed (`exact dense LU` never had to trigger there). One caveat carried over from the mid-run finding, unchanged: the fitted convergence RATES (not the cg_failures/wall_clock numbers) at N=1001/1401 specifically should be read with the fine_N=2236-is-too-close caveat in mind, not as fully trustworthy rate measurements -- the run's own final least-squares fit (`L2 p=1.57, H1 p=0.73, energy p=0.86, peak-stress p=0.58` across all 7 points) is reported here for completeness but is exactly the number that caveat applies to. **This item's own stated goal -- rerun N=1001 and N=1401 to full CG convergence -- is achieved.** Item #3's own condition (a) (re-measure after the preconditioner fix) is now satisfiable using this data.

**Written into both real documents 2026-09-08** (Omar confirmed: yes, write it in): new Table 20c (Report, §8.5 "Scaling to a few million degrees of freedom") / a matching compact table (Summary), reporting N=401/701/1001/1401 all at `0 of 20` capped CG solves under the ORIGINAL 2,000-iteration cap Table 20a used, not the raised one Table 20b needed. Written with the same measured-vs-predicted rigor the rest of that section already uses: multigrid needs 19-26x fewer CG iterations per Newton solve than Jacobi's O(N) law predicts, but each iteration costs more, so the net wall-clock comparison actually CROSSES OVER across the three new resolutions -- 36% slower than Table 20b's directly-measured N=701 Jacobi-converged result, 13% slower than the (unmeasured, predicted) N=1001 estimate, but 35% FASTER than the (unmeasured, predicted) N=1401 estimate -- stated as exactly that mix of measured/predicted comparisons, not smoothed into one number. A caveat paragraph explains the approximate coarsest-level solve N=701/1001/1401 needed (unlike N=401's exact one) as a plausible partial explanation for the smaller advantage at the two smaller of the three. A pointer sentence was also added right after Table 6a's own CG-cap caveat noting it is now resolved by this preconditioner, WITHOUT altering Table 6a's own already-published numbers (computed under the original plain-Jacobi run) -- a full re-fit of Table 6a using mgv-converged data is noted as a possible future refinement, not undertaken in this pass. Verified via python-docx readback in both documents before promoting `..._updated_2026-09-08.docx` as the new canonical deliverables (same LibreOffice-PDF-unavailable-in-sandbox situation as item #11's writing-in, same direct-content-verification substitute). | Item #4 is DONE and written into both real documents. Table 6a has SINCE been fully rebuilt with this data too -- see item #3's row. One remaining, separate decision for Omar: whether to commission a bigger fine_N re-run later, purely to get trustworthy convergence RATES at N=1001/1401 specifically (not needed for cg_failures/wall_clock, which are already trustworthy and in the report) | (superseding the separate per-N notebooks first drafted), running `high_dof_convergence_study` ONCE across `--resolutions 51,101,201,401,701,1001,1401` so N=701/1001/1401 solve in order in a single process (~15h combined, rough estimate) while N=51/101/201/401 reuse their existing checkpoints instantly. Made genuinely safe against a Colab disconnect mid-run: each target N's checkpoint is force-reset to a fresh mgv solve only the FIRST time the cell ever runs (guarded by a `mgv_reset_done_N{n}.marker` file on Drive) -- re-running the same cell after a disconnect does NOT re-delete an in-progress resolution's checkpoint (which a naive unconditional reset-then-solve script would have done, silently discarding real progress on a run this long), it resumes from `solve_matrix_free`'s own per-Newton-iteration checkpoint exactly where it left off. See the top summary for the reasoned, explicitly-uncertain time estimate for each resolution -- treat it as a ballpark, not a promise, precisely because all three take a structurally different code path (the new approximate coarse-solve fallback) than N=401 did. Read each resolution's own printed numbers as it finishes rather than assuming the estimates held |
| 5 | Label Table 10c/10d's batch-size-1 column as the primary single-query break-even comparison (editorial, quick) | R8.7-A | ✅ **Actually applied to the real Report file 2026-09-07** — same stale-status story as #1/#2. A sentence added right after "What this clarifies is where the operator is useful" naming batch-size-1 "the primary, single-query result," everything else "a separate throughput experiment." (The Summary document already made this point in its own words, in Table 10d's text — nothing needed there) | — |
| 6 | Extend the OOD progressive-shift study (Tables 19/19a) to the other five cases | R8.2 | ✅ **DONE — all 5 remaining cases finished on Colab 2026-09-07** (`Round6_OOD_Progressive_Remaining.ipynb`, ~8.5h total across B1×Mooney-Rivlin, B1×Arruda-Boyce, B2×Neo-Hookean, B2×Mooney-Rivlin, B2×Arruda-Boyce). **Consistent finding across every one of the 5 cases**: material-property shift is the dominant degradation driver, loading-magnitude shift is mild by comparison — matching the pattern the already-published B1×Neo-Hookean case (Tables 19/19a) established. B2 numbers: Neo-Hookean baseline 15.6%, degrades to 4.75x at material/3σ (loading only 0.99x); Mooney-Rivlin baseline 10.2%, degrades to 5.47x at material/3σ (the most fragile of the 3 B2 materials), loading 1.23x; Arruda-Boyce baseline 18.5%, degrades to only 2.27x at material/3σ (the most robust material found so far), and its loading-shift numbers are actually BELOW baseline at every σ tested (0.79-0.93x) — a genuinely interesting, consistent finding, not noise, worth naming directly in the report as a material-dependent robustness result rather than folding all 6 cases into one flat "4-5x degradation" number | **Written into both real documents 2026-09-07** (Table 25 + narrative, Report §8.6; new paragraph, Summary §6) — see note below the table for the version-gap context this uncovered. Nothing further needed for this item |
| 7 | New study: train a DD-NO on FEM labels from a coarse mesh vs. a finer mesh; measure how its accuracy/generalization across resolutions changes | R8.3 | ✅ **DONE — both coarse and fine finished 2026-09-07.** `Round6_DD_NO_Coarse_vs_Fine.ipynb`, B1×Neo-Hookean, N=13 (coarse) vs. N=33 (fine), zero-shot at Table 12's seven resolutions, identical Table-21 budget (800/200/75,000) for both, earlier stale-checkpoint bug fixed via `BUDGET_TAG`-keyed directories (see prior entries for the full story). **Coarse (N=13)**: mean rel. L2 at N=13/17/25/29/37/41/49 = 0.105/0.117/0.154/0.174/0.211/0.228/0.259 — error rises monotonically and substantially (2.5x) moving away from its own training resolution. **Fine (N=33)**: best val rel-L2 0.1448 (step 74,000, 5.65h label generation + 30min training); zero-shot at the same 7 resolutions = 0.137/0.128/0.118/0.115/0.111/0.110/0.109 — error DECREASES monotonically across the whole tested range (its own training resolution, N=33, wasn't itself in the 7 test points, but the trend continues past it) and stays in a tight 10.9-13.7% band. **Clean, direct answer to the item's own question**: training on a finer mesh yields a model that is more STABLE/robust across resolutions (worst case 13.7%) than training on a coarse mesh (worst case 25.9%), even though the coarse-trained model is more accurate exactly at its own training resolution (10.5% vs the fine model's 10.9-13.7% floor). This is a genuine, reportable finding — training-mesh resolution trades peak-accuracy for generalization robustness | **Written into both real documents 2026-09-07** (new §8.7 subsection "Training-mesh resolution: a coarse-versus-fine data-driven comparison" + Table 26, Report; new paragraph, Summary §7). Nothing further needed for this item |
| 8 | New break-even: physics-informed operator vs. DD-NO, total cost of ownership (data generation + training + N inferences) | R8.7-B | ✅ **Actually applied to the real Report file 2026-09-07** — same stale-status story as #1/#2/#5. New Table 21a: 800-solve label-generation cost (5.65 h) makes DD-NO more expensive by a FIXED 18,924 s (Adam) / 18,694 s (AdamW+OneCycle) for every N, not a break-even threshold, under the stated assumption that DD-NO's inference cost equals the PI operator's (not separately measured, but same architecture). Also added to the Summary (§10) | — |
| 9 | Richer manufactured-solution family (sum of several sine/cosine modes, boundary conditions preserved) | R8.6 | ✅ **Implemented, safety-checked, and verified 2026-09-06.** Extended `u_exact`/`grad_u_exact`/`P_exact`/`body_force_exact` (and everything downstream: `assemble_body_force`, `compute_errors`, `solve_mms`, `verify_derivation`) to accept a `modes` parameter — `MODES_RICHER` sums 3 sine/cosine spatial modes per component (different mode sets for u_x vs u_y, same reason beta != 1 mattered before: a bug that only gets one mode's derivative right can't hide), every term still vanishes exactly on the unit square boundary so homogeneous Dirichlet stays exact. **A real methodological risk was caught and fixed before committing anything**: the first pass made the richer family the new DEFAULT — but `report_builders/make_v34.py` etc. and `mms_operator.py`'s (alpha,beta) family already build on `mms_B1_neo_hookean.json`'s existing single-mode shape (Tables 22-24), so silently changing `u_exact`'s default would have invalidated already-published numbers without any of the report's usual revision process. Fixed by keeping `MODES_SINGLE` (the original field, byte-for-byte reproduced — verified `--verify` gives the IDENTICAL 1.828e-10 diff as before any of today's changes) as the default everywhere, and adding `MODES_RICHER` as strictly opt-in via a new `--richer_family` flag, which also switches the default output filename so it can never overwrite the reference file. Verified: `--verify` passes cleanly for both the default and `--richer_family`; `test_mms_operator.py`'s full existing sanity suite (4 independent checks against the FEM solution) still passes unchanged; a real Q4 convergence run (N=5/9/17) with `--richer_family` gives clean rates matching theory (L2 1.99≈2, H1 0.98≈1, energy_norm 0.99≈1, energy_rel 1.93≈2 superconverging as expected) — the richer field is a genuinely harder nonlinear problem (Newton needs ~20 iters vs. 15, CG work up several-fold) but converges at the textbook-correct rates regardless. **Full 8-row sweep DONE and committed 2026-09-07** (survived one interruption mid-run — the process was killed when the container paused for a usage-limit reset; `main()`'s own resumability guard, `done = {(order,N) for r in rows if "energy_norm_rel" in r}`, picked the sweep back up from the partial `/tmp` file and only re-solved the one row not yet finished, Q9 N=33, rather than redoing all 8). CPU, no Colab needed (same as item 10). Rates confirm the richer field converges exactly like the single-mode one despite being a harder nonlinear problem (Q9 N=33 took 1160s vs. the single-mode file's 379s): Q4 L2 1.99≈2, H1_semi 0.99≈1, stress 0.99≈1, energy 1.95≈2 (superconverges as expected), energy_norm 0.99≈1; Q9 L2 2.96≈3, H1_semi 1.98≈2, stress 1.99≈2, energy 3.97≈4 (superconverges), energy_norm 1.98≈2. `rate_check` is "as expected" for both orders. Committed at `point9_results/mms_richer_B1_neo_hookean.json` (a NEW file; `mms_B1_neo_hookean.json`, the reference every report table depends on, is untouched) | None — done. Next: fold the richer-family rates into whichever report section covers §4.9/MMS as a second, harder MMS instance alongside Tables 22-24's single-mode one, if Omar wants that in the report body rather than just as a validation artifact |
| 10 | Report the MMS energy NORM (not the energy value) — the computation already exists in this codebase (§4.4/Table 6a); apply it to the MMS study | R8.6 | ✅ **Done and verified 2026-09-06.** The earlier "not started" status was stale — this session found `torch` DOES work here (CPU, confirmed by running `mms_study.py --verify`, which passed clean). Added `energy_norm_rel`/`energy_norm_abs` to `mms_study.py`'s `compute_errors()`, kept alongside (not replacing) the existing `energy_rel` value comparison. **A real bug was found and fixed before trusting any numbers**: the first implementation reused `high_dof_convergence_study.py`'s matrix-free Hessian-vector machinery, evaluating u* at the mesh's own NODES — this measures `||u_h - I_h(u*)||_E` (against the nodal INTERPOLANT), which is a well-documented superconvergent quantity on uniform meshes (Wahlbin), and empirically showed rate ≈2 for Q4 — the SAME doubled rate as the already-known-buggy value comparison, defeating the entire point of adding a norm that should NOT superconverge. Caught by checking the measured rate against Cea's-lemma theory (a proper energy norm should track H1's rate, not double it) rather than trusting a plausible-looking number. **Fixed** by computing the norm as a direct quadrature integral against the CONTINUOUS exact solution instead — `int grad(e):C(F*):grad(e) dV`, C being the 4th-order tangent modulus (`tangent_modulus_batched`, new function, same nested-jacrev-then-vmap pattern this file's own `body_force_exact` already uses) — matching L2_rel/H1_semi_rel's own Gauss-point convention exactly, no nodal step. **FULL SWEEP NOW DONE, no Colab needed** — this study is tiny (N up to 33, not millions of DOF like B1/B2) so it ran to completion directly in this session, CPU only, ~15 minutes total for all 8 rows. Report-ready result committed at `omar_pfem/point9_results/mms_B1_neo_hookean.json` (replacing the old 6-row pre-energy-norm file — same L2/wall-clock values on the 3 overlapping resolutions, confirming reproducibility, now with N=33 added for both orders and `energy_norm_rel` throughout). **Both element orders confirm the fix cleanly against theory**: Q4 energy_norm rate 1.00 (matches H1's 1.00, expected 1) vs. energy_rel (value) rate 1.99 (expected 2, superconverges as it always did); Q9 energy_norm rate 2.00 (matches H1's 2.00, expected 2) vs. energy_rel rate 3.99 (expected 4). `rate_check` passes cleanly for both orders. **Genuinely done** — nothing left to run for this item. (Found in passing, unrelated, not fixed: `point9_results/make_readme.py`'s auto-README generator was already broken before this session — `mms_operator_per_member_...json` and `mms_operator_rate_...json`, both added in earlier commits, don't match the schema it expects. Pre-existing, out of scope here, flagged for whoever picks up the README next) | None — done. Next: fold `mms_B1_neo_hookean.json`'s energy_norm column into whichever report table covers §4.9/MMS (Tables 22-24's neighborhood) |
| 11 | Extend MMS to at least one other material/problem | R8.6 | ✅ **DONE for both remaining materials 2026-09-07, AND now actually written into both real documents 2026-09-07.** Mooney-Rivlin and Arruda-Boyce, not just one. Caught and fixed a real bug the moment this was first tried: `mms_study.py`'s core functions hardcoded exactly two material parameters (mu, lam), which only means something for Neo-Hookean — Mooney-Rivlin needs 4 (c, c1, c2, d), Arruda-Boyce needs 3 (mu_ab, N_ab, kappa_ab), so `--material mooney_rivlin --verify` immediately raised a `TypeError` from the material's own energy-density function. Fixed by generalizing every function (`_psi_and_P`, `tangent_modulus_batched`, `P_exact`, `body_force_exact`, `assemble_body_force`, `compute_errors`, `verify_derivation`, `solve_mms`) to thread a generic `params` tuple through instead, matching the pattern `matrix_free_solver.py` already used correctly elsewhere; also fixed the same hardcoded-(mu_e,lam_e) bug in `mms_operator.py`, `mms_operator_per_member.py`, and `test_mms_operator.py`, which called the now-changed functions with the old 2-argument convention. **Verified this is a pure generalization, not a behavior change**: `--verify` now passes for all 3 materials; `test_mms_operator.py`'s full 4-check sanity suite still passes unchanged; a fresh Neo-Hookean Q4 N=5 solve matches the already-committed `mms_B1_neo_hookean.json` (Tables 22-24's own reference) to floating-point noise (~1e-16). **Both new materials' full 8-row Q4/Q9 sweeps converge exactly at theory rates**: Mooney-Rivlin Q4 L2 1.98/H1 1.00/energy_norm 1.00, Q9 L2 3.01/H1 2.00/energy_norm 2.00; Arruda-Boyce Q4 L2 1.99/H1 1.00/energy_norm 1.00, Q9 L2 3.01/H1 2.00/energy_norm 2.00 — `rate_check` "as expected" for both orders, both materials. CPU only, ~10 min total for both sweeps. Committed at `point9_results/mms_B1_mooney_rivlin.json` and `mms_B1_arruda_boyce.json`. **Written into both real deliverables** (`PFEM_Transolver_Report_updated_2026-09-07.docx` §8.11, `PFEM_Work_Summary_updated_2026-09-07.docx` §11), per Omar's standing instruction to write every confirmed, finished result directly into the files so nothing gets lost: new Tables 22a (Mooney-Rivlin, 8 rows Q4/Q9) and 22b (Arruda-Boyce, 8 rows) mirror Table 22's exact columns/format, plus Table 23a (16-row combined convergence-rate table, both materials/orders/norms, all landing on theory) mirroring Table 23 — appended as lettered follow-ons after the existing Table 24/24a-e chain's own anchor point, so nothing already numbered needed renumbering. Verified via python-docx readback (table row/column counts, header/data correctness, and that the original following content resumes intact immediately after the insertion) in both documents before promoting them as the new canonical `..._updated_2026-09-07.docx` deliverables — LibreOffice's headless PDF converter would not run in this session's sandbox (failed even on a trivial one-paragraph test file, so this is an environment limitation, not a symptom of a bad edit) so the usual visual PDF-render check was substituted with this direct-content verification instead | None — written into both real files. Next real per-material report gap, if wanted: item #12 (tables to figures) still pending, applies equally to these new tables |
| 12 | Convert several tables into figures, matching Timon's previous papers' presentation | R8, general | 🟡 **Starting 2026-09-08, per Omar's own explicit direction to work through everything still left on the queue** while item #13's N=1401 GPU run is pending on his side. Scope is genuinely open -- Timon's own round-8 email says only "I think we should improve the presentation by converting several tables into figures as in my previous papers," with no specific table list and no example figure attached anywhere in this session's advisor_feedback files -- confirmed by re-reading the full round-8 email text directly, not assumed. Needs Omar's own scope call (which tables, and does he have example figures from Timon's prior papers to match the style) before real work starts, same as every other ambiguous-scope item this project has handled | Waiting on Omar's scope decision (see above). Independent of item #13's still-pending N=1401 number, since that adds a NEW table rather than reworking an already-published one -- can proceed in parallel |
| 13 | GPU-FEM vs. torch-fem efficiency comparison | R7.1 | 🟡 **Built and CPU-validated 2026-09-08, awaiting Omar's GPU run.** Omar's explicit scope choice: compare the large-scale matrix-free solver (Table 20/20a/20b/20c) against torch-fem at the SAME resolutions those tables use (N=401/701/1001/1401), not the small-scale batched dense solver compared against the operator elsewhere. New module `omar_pfem/torchfem_comparison.py`: reuses `high_dof_convergence_study.py`'s own `build_mesh_and_bcs` directly for BOTH solvers (same mesh, heterogeneous E/nu material field, boundary conditions, and top-edge traction load, not a re-derivation) -- confirmed by direct inspection that `generate_grid_Q4`'s own element node order (bottom-left, bottom-right, top-right, top-left) is IDENTICAL to torch-fem's `Quad1` convention, so `nodes`/`elements` pass through to torch-fem with zero reordering. **A real bug was found and fixed in torch-fem itself while validating this**: writing our Neo-Hookean energy with `torch.log(torch.linalg.det(F))` for ln(J) -- a completely standard way to write it, and what a first attempt naturally reached for -- makes torch-fem's own double-backprop tangent stiffness computation (`vmap(jacrev(jacrev(psi)))`) come out ALL-NaN at F=Identity, exactly the point every solve starts from, silently making the assembled tangent matrix "exactly singular" (confirmed via a direct, isolated test: computing the Hessian of the same psi both ways at F=I gives an all-NaN result with `torch.log(torch.linalg.det(.))` and a correct, finite one with `torch.linalg.slogdet(.)[1]` -- mathematically identical value, numerically stable gradient). This is a known sharp edge in `torch.linalg.det`'s double-backward, not a physics bug, but it took real debugging (tracing which F values reached the material's `.step()`, then isolating the Hessian computation itself) to separate from an initially-suspected setup/BC bug -- confirmed the mesh/BCs/forces were correct FIRST (torch-fem's own built-in linear-elastic material solved the identical mesh/BCs fine) before concluding the material function itself was the problem. **Also needed**: torch-fem's own `near_null_space()` hardcodes float32 and errors on a float64 model, so the torch-fem side runs in float32 (its own working precision here, not a compromise on our side) with correspondingly loosened solver tolerances (`stol=1e-4`, Newton `rtol=atol=1e-3`, vs. our own float64 solve's usual 1e-8). **CPU correctness validated on 2 resolutions** (`python -m omar_pfem.torchfem_comparison <N>`, N=11 and N=21): both solvers, given the identical problem, converge to the same displacement field, relative difference 4.6e-6 and 2.1e-6 respectively -- well inside torch-fem's float32 precision, confirming both solvers are genuinely solving the same physical problem before any timing comparison is trusted. Also smoke-tested `run_sweep_row` end-to-end on CPU with the mgv preconditioner path (tiny N=9, deliberately slow at this scale as already expected from item #4's own findings, but completed with `cg_failures=0`) to confirm no crashes before pointing it at real GPU sizes. **Comparison design**: our solver uses its best available preconditioner (multigrid, item #4); torch-fem uses its own best readily-available one (Jacobi -- AMG would need an extra pyamg/amgx dependency not assumed installed), explicitly documented as "best readily-available each," not each side's theoretical best. Instruments peak GPU memory via `torch.cuda.max_memory_allocated` for BOTH solvers, not just wall-clock, since torch-fem always explicitly assembles a sparse tangent matrix (confirmed by reading its source: `base.py`'s `assemble_matrix`/`self.K`, even in "cg" mode) while our own solver never forms K at all -- this architectural difference is expected to show up most clearly in memory at the largest resolutions, which wall-clock alone would miss. Colab notebook `Round6_TorchFEM_Comparison.ipynb` built, registered, verified 45/45 via `check_notebooks.py`, resumable across resolutions. **Caught and fixed a real design mistake before it cost real GPU time, prompted by Omar asking how long the run would take**: the first version of this sweep re-solved "ours" completely from scratch at all four resolutions, which would have re-spent essentially the same ~14.86h item #4's own real run already cost, purely to reproduce wall-clock/cg_iters numbers already committed in `highdof_stress_qoi_results/*.json`. Fixed by passing item #4's own checkpoint directory through: "ours" now resumes from the ALREADY-COMPLETED mgv solve at each N (near-instant, same checkpoint-reuse behavior every other notebook in this project already relies on) and the row uses the already-committed real numbers instead of the resume call's own (meaningless) near-zero timing. This makes the real cost of running this notebook just torch-fem's own solve time at each N, not ~15h of redundant recomputation -- unknown in advance since torch-fem has never been run at this scale, but expected to be far less than the original estimate implied **Also fixed 2026-09-08, before Omar's real run**: `import torchfem` itself failed on Colab specifically with `ModuleNotFoundError: No module named 'IPython.core.guarded_eval'`, raised inside pyvista (a plotting dependency this module never uses) -- traced to pyvista's own `_allow_ipython_completion`, which only runs `if 'IPython' in sys.modules` (true in a Colab notebook kernel, why this never appeared in local testing) and unconditionally imports a submodule Colab's installed IPython lacks. Confirmed by reading the function's full source that a harmless empty stub for that one missing submodule is sufficient (the function already defaults to `{}` via `getattr` if the module lacks the attribute it wants). Verified by simulating the exact Colab failure condition locally (fake IPython in `sys.modules`, the specific import patched to raise) before trusting the fix. **Also, prompted by Omar asking to shrink the resolutions tested to save time**: explained why that would NOT answer the same question and could plausibly flip the conclusion (item #4's own numbers already show our solver's relative standing changing with N; torch-fem's own scaling is unmeasured and there's no reason to assume it's flat) -- Omar agreed to the same cheapest-first staging used throughout this project instead. **Omar's own refinement 2026-09-08**: rather than N=401 alone, run the three cheaper resolutions together (401/701/1001) since "ours" is free regardless of how many N are listed, and hold back only N=1401 (the most expensive, and the one most likely to expose a real memory difference) for a separate decision -- exactly the same Stage-1/Stage-2 split already used for the block2x2 and mgv rechecks. `cell_torchfem_comparison.py`'s `RESOLUTIONS = [401, 701, 1001]` now, with a comment explaining how to add N=1401 once these three look right. **First real Colab attempt hit the pyvista/IPython import error above; a first, narrower fix (stubbing only the missing `IPython.core.guarded_eval` submodule) did NOT work on Omar's actual Colab run even after a runtime restart** -- fixed properly by instead replacing the ENTIRE `pyvista` module with `unittest.mock.MagicMock()` whenever `import pyvista` fails for any reason (nothing in this comparison ever touches pyvista's plotting API, so a permissive stand-in is safe), re-verified locally by patching only the first `import pyvista` attempt to fail then letting normal `sys.modules` caching apply — confirmed. **Omar confirmed this fix worked on the real Colab run** (N=11 CPU correctness check passed, N=401 "ours" resumed from checkpoint correctly) -- but immediately hit a SECOND, different real bug: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`, raised inside torch-fem's own `FEM.__init__` (`torchfem/base.py`), specifically its own `torch.arange(self.n_dof_per_node)` call, which has no explicit `device=` argument and therefore always defaults to CPU regardless of the model's other (CUDA) tensors -- a real bug in torch-fem itself, not something fixable by adjusting our own tensors, since the mismatch happens inside torch-fem's own constructor before any tensor comes back to our code. **Fixed** by wrapping the `Planar(nodes_t, elements_t, material)` call in `build_torchfem_model` with `with torch.device(device):`, which makes every device-less tensor creation inside that block -- including torch-fem's own internal `torch.arange` call -- default to the correct device automatically. Re-verified the CPU correctness check still passes after this change (`N=11: PASS, relative displacement-field difference 4.608e-06`, identical to before, confirming the fix is safe and behavior-preserving on CPU where `with torch.device('cpu'):` is a no-op). **Omar re-ran with this fix and got the EXACT SAME error again** -- traced this to a THIRD, different kind of bug, not a new real torch-fem issue: the traceback's own displayed source lines didn't match the current file's actual content at those line numbers (e.g. the frame blamed on `solve_theirs` showed source text that is really inside `solve_ours`'s body, and `build_torchfem_model`'s frame pointed at a blank line) -- the signature of a STALE in-kernel Python module cache, not a code bug. The cell's CPU correctness check runs as a fresh `subprocess` every time (always sees the latest file on disk, which is why it kept reporting PASS with the current numbers), but the sweep itself does a plain in-process `from omar_pfem.torchfem_comparison import run_sweep` -- and `git reset --hard` only changes files on disk, it does NOT clear Python's `sys.modules` cache in an already-running Colab kernel. Since this notebook (and others sharing the same long-lived Colab runtime, per "Drive already mounted" on each re-run) had already imported `omar_pfem.*` modules in an earlier cell execution, the sweep kept running the OLD pre-fix code every time, regardless of what was actually pushed to the branch. **Fixed in `cell_torchfem_comparison.py`**: right before importing `run_sweep`, delete every `omar_pfem`/`omar_pfem.*` entry from `sys.modules`, forcing a clean re-import from disk on every run of this cell, regardless of what an earlier cell (in this or an earlier notebook) already imported into the same kernel -- removes the dependency on the user remembering to restart the runtime, which was already tried once for the earlier pyvista bug and reportedly didn't fix that issue either (also potentially an instance of this same stale-cache class of problem). Rebuilt the notebook, re-verified 45/45 via `check_notebooks.py`. **Omar's next re-run (a genuinely fresh `git clone`, so the stale-cache class of bug could not recur) got past N=401's resume cleanly, then hit a FOURTH bug, same root cause as the earlier device-mismatch one but a different call site**: `RuntimeError: ... tensors is on cpu, different from other tensors on cuda:0`, this time inside torch-fem's own `near_null_space()`/`skew()` (`base.py`, called from inside `.solve()` itself, not the constructor `Planar(...)` already fixed) -- `torch.zeros`, `torch.arange`, and `torch.eye` there are ALSO built with no explicit device, and `skew()`'s own `torch.cat([r, torch.zeros(...)], dim=1)` fails once `r` (derived from `self.nodes`, correctly on cuda) meets that device-less `torch.zeros` (defaulting to cpu). **Fixed** by widening the existing `with torch.device(device):` wrap in `solve_theirs` from just the `Planar(...)` constructor call to the ENTIRE `model.solve(...)` call -- this also covers any other similar device-less internal call anywhere in torch-fem's whole `solve()` call tree, not just this one specific one, since the context manager stays active for every nested call for as long as the `with` block is open. Re-verified the CPU correctness check still passes unchanged (PASS, 4.608e-06 rel. diff) before pushing. **Real GPU sweep FINISHED 2026-09-08** (`torchfem_comparison_results/torchfem_comparison_B1_neo_hookean.json`, fetched from Drive and committed) -- but only after a THIRD environment issue, again not a code bug: Omar's re-run reproduced the exact same device error a third time despite the fix being pushed, traced to the Colab tab itself still holding a stale, previously-loaded copy of the notebook's own cell text (same long-lived kernel process ID across every attempt) -- `git`/`pip` correctly updated the files on disk each time, but the actual cell CODE running in that browser tab was never refreshed from GitHub, so none of the fixes committed after that tab was first opened ever actually ran. Fixed operationally, not with more code: gave Omar a direct `https://colab.research.google.com/github/...` link to force a genuinely fresh notebook load, which finally picked up every fix and ran clean end to end in 0.03h (torch-fem's own solve time really is that cheap at this scale). **The headline, honest result, N=401/701/1001 (n_dof 321,602/982,802/2,004,002)**: torch-fem is dramatically FASTER in wall-clock than this project's own matrix-free mgv solver -- 6.26s/11.00s/14.46s vs. 2615.8s/7205.4s/17314.8s, i.e. ~418x/655x/1197x, an ADVANTAGE THAT GROWS WITH N, the opposite direction from what the memory argument predicted. **This is real, but two honest caveats belong right next to it, not hidden**: (1) the two solvers are NOT run at matched precision/tolerance -- torch-fem's own float32 working precision forces stol=1e-4, Newton rtol=atol=1e-3 (see build_torchfem_model's own docstring for why), while "ours" runs at float64 with cg_tol=1e-8, newton_tol=1e-8 -- several orders of magnitude tighter, which plausibly explains a large share of the gap on its own (a known, already-disclosed asymmetry in the comparison's own design, not newly discovered here, but its likely SIZE wasn't visible until real numbers existed); (2) **`ours_peak_mem_mb` is `null` at all three rows** -- because "ours" resumed from item #4's own checkpoint (by design, to avoid ~15h of redundant GPU time) rather than doing a fresh solve, so the ONE measurement this comparison's own design doc said should show our solver's real architectural advantage most clearly (peak GPU memory, matrix-free vs. assembled-sparse) was never actually captured. torch-fem's own memory (3292.6/9669.8/19498.9 MB, growing roughly linearly with n_dof, consistent with its own confirmed sparse-matrix-assembly architecture) IS measured and committed, but there is currently nothing real to compare it against. Getting a real "ours" peak-mem number would need an actual (not resumed) solve at these sizes -- not free like this run was, and a decision Omar should make deliberately given the cost, not something to spend GPU hours on without asking first **Omar's own two decisions, asked directly once the real numbers and real costs were both on the table (2026-09-08)**: (1) on the memory gap -- Omar first (reasonably) asked whether real "ours" memory numbers already existed somewhere from item #4's own earlier runs; checked directly and confirmed they do NOT (`high_dof_convergence_study.py`/`matrix_free_solver.py` never instrumented `torch.cuda.max_memory_allocated` anywhere, and neither `highdof_stress_qoi_..._N401.json` nor `..._N701_1001_1401.json` has a memory field of any kind) -- once that was clarified, Omar's call was to accept the wall-clock-only comparison as sufficient for Timon's stated question and note the memory gap honestly as a deliberately open item, rather than spend the several additional GPU-hours a real (non-resumed) "ours" solve at these sizes would cost just to fill it in; (2) extend to N=1401 now -- approved, since the sweep turned out to cost almost nothing (0.03h total for the first three), so the fourth adds well under a minute of new real compute. `cell_torchfem_comparison.py`'s `RESOLUTIONS` updated to `[401, 701, 1001, 1401]`, header comment and `make_round6_notebooks.py`'s markdown rewritten to state the real result instead of the old staging plan, rebuilt and re-verified 45/45. 🟢🟢 **ITEM #13 IS NOW FULLY DONE, 2026-09-08.** N=1401 finished (Omar's re-run, same fresh-tab discipline): `ours_wall_clock_s=27257.4` (identical to item #4's own already-committed number, cg_iters=6377, cg_failures=0), `torchfem_wall_clock_s=29.1`, `torchfem_peak_mem_mb=38063.8`. **Final 4-row result, committed** (`torchfem_comparison_results/torchfem_comparison_B1_neo_hookean.json`): speedup (ours/torch-fem wall-clock) is 418x/655x/1197x/936x at N=401/701/1001/1401 -- honestly NOT monotonic (peaks at N=1001, dips back down at N=1401, worth stating plainly rather than smoothing into "grows with N" as an earlier partial read of the 3-row data suggested). torch-fem's own peak memory scales close to linearly with n_dof throughout (~0.0097-0.0102 MB/dof at all four points), consistent with its confirmed sparse-assembly architecture. **Written into both real documents 2026-09-08** (`report_builders/add_item_13_torchfem_result_report.py` / `_summary.py`, following the exact same anchor-paragraph/insert-table pattern item #4's own write-up used): new Table 20d (Report, right after Table 20c's own last paragraph, before "Three limits of this study should be stated") with the 4-row wall-clock/memory/speedup comparison plus two full paragraphs stating the precision/tolerance and "ours"-memory-not-measured caveats explicitly, and a closing paragraph naming this as a genuine matrix-free-vs-assembled architectural trade-off rather than a defect -- explicitly tying it back to Timon's own round-8 prediction ("presumably assembles explicitly once and factorizes") as a direct empirical confirmation of a concern he raised before this comparison was ever run. Matching compact paragraph + table added to the Summary (right after its own "Cost breakdown" paragraph, before the OOD heading). Verified via python-docx readback in both documents (table contents match the JSON exactly, surrounding paragraphs intact) before promoting `..._updated_2026-09-08b.docx` back onto the canonical `..._updated_2026-09-08.docx` name in both cases. **Per Omar's own direction 2026-09-08, the eventual combined reply to Timon should also name this same architectural trade-off explicitly** — see the new note just below this table for why | Item #13 is done, nothing further pending on it. Per Omar's own instruction (2026-09-08, "خلص 13 بعدين الي بعده"): move to item #12 next once he gives scope guidance (which tables / does he have an example figure from Timon's prior papers) -- he was asked and explicitly deferred that decision, so item #12 stays not-started until he answers |
| 14 | Get the GOEE/"trust" paper Timon says he attached — it never arrived in this session | R7 | 🟡 **Confirmed by Omar 2026-09-07 — this IS the paper Timon sent, but it is still NOT a GOEE paper by name or method.** `0000-2609.02982v1.pdf` (arXiv:2609.02982v1, "Equation Recast for Canonical Operator Learning Across Parametric PDEs," Cheng/Duruisseaux/Clauser et al., MIT/Caltech/RPI/Princeton, 2 Sep 2026) — Omar directly re-uploaded a fresh copy of the PDF and asked for confirmation; extracted its page-1 text (title/authors/institutions/arXiv ID/abstract, via `pdftotext` after installing `poppler-utils`, since `pdftoppm`/`pypdf` both failed in this environment) and it is byte-identical in content to the copy already reviewed on Drive, so this really is what Timon attached. It is a **parametric operator-learning / transfer-learning paper**: it analytically absorbs a PDE's parameter/geometry variation into an effective source term ("equation recast"), so ONE canonical neural operator (trained once at a reference parameter) can be reused zero-shot across the whole parameter family via a fixed-point iteration, demonstrated on 1D ADR/reaction-diffusion/Helmholtz, 2D Navier-Stokes, and a 4-geometry tokamak MHD case. **Grepped the full extracted text for "goal-oriented," "adjoint," "GOEE," and "error estimat*" as technical terms — zero hits, anywhere in the paper.** This is NOT goal-oriented error estimation in the rigorous FEM/adjoint sense (no dual problem, no QoI error bound, no certified estimate) — Timon's own round-7 email named GOEE explicitly as "adapted from FEM" (see the master table entry above), which this paper does not do. Its only trust-adjacent contribution is informal: "loss of convergence as an internal warning signal" — the fixed-point recast iteration's own iteration count/divergence (worst near a Helmholtz resonance) is used as a cheap, reference-free flag that a prediction is becoming unreliable, explicitly NOT backed by a computable Lipschitz/contraction bound ("we do not use such a bound as a predictive criterion... convergence is assessed directly from the observed iteration"). So: directly citable as prior art on "non-convergence of an internal solve as a runtime reliability flag for a neural-operator prediction" (a narrower, weaker claim than GOEE), but should not be presented as an actual goal-oriented error estimator | **Written into both real documents 2026-09-07** (new Conclusion bullet, Report; new note in §7, Summary), citing it only for the narrow "convergence-as-reliability-signal" claim, not as a GOEE method. Nothing further needed for this item |
| 15 | Open-source the GPU-FEM code (Apache 2.0) | R7.2 | 🚫 Do not do yet | **Blocked until the paper is submitted/on arXiv** — explicit instruction, not a technical dependency |

**New, explicit note for the eventual combined reply to Timon (Omar's own direction, 2026-09-08)**: once item #13 fully lands (N=1401 included), the combined reply should explicitly name the matrix-free-vs-assembled-matrix architecture trade-off as its own discussion point, not just report the raw torch-fem numbers. Context for why this belongs in the reply specifically: Timon's own round-8 email (point 5) already predicted this exact effect theoretically, before this comparison was ever run — he wrote that TensorMesh "presumably assembles explicitly once and factorizes," implying our own matrix-free solver's per-CG-iteration Hessian-vector cost (which does implicit, repeated element-level work Timon called out as hidden inside CG) would plausibly cost more than a solver built the traditional way. Item #13's real result (torch-fem ~418x-1197x faster in wall-clock, gap growing with N) is a direct empirical confirmation of that same concern, using a different assembled-matrix library (torch-fem, not TensorMesh, but the same fundamental architecture). The reply should say so plainly: this is not a defect to fix by rewriting the solver (doing so would forfeit the whole reason it was built matrix-free — reaching millions of DOF on a single GPU without ever materializing K, which is what let this project reach the multi-million-DOF references every other table's ground truth depends on, at problem sizes an assembled approach could plausibly not handle in memory at all), but a genuine, disclosed architectural trade-off, alongside the tolerance/precision caveat item #13's own row already documents.

**Two real orderings, not just independence:**
- **#3 before #4, #6, #7, #13** — not a hard technical block (the code/
  methodology from doing them now is not wasted), but if #3 concludes
  the benchmark needs to be fundamentally bigger, the specific
  measurements from #4/#6/#7/#13 on today's B1/B2 may need re-running on
  the new problem. Settling #3 first avoids that risk.
- **#4 before #13** — fix the GPU-native solver's own preconditioner
  before benchmarking it against torch-fem, so the comparison reflects
  the improved solver, not the version Timon already flagged as
  suboptimal.

**#8 was wrongly listed as needing #7 in the previous version of this
table — corrected above.** Rows 5, 6, 7, 8, 9, 10, 14 have no dependency
on anything and can start immediately, in any order.

---

# 📬 Timon's round-8 review (2026-09-06) — manuscript-level, mostly NEW WORK

Stored verbatim at `advisor_feedback/2026-09-06_round8_timon.md`, with a
full point-by-point reading. **This is a different kind of feedback than
rounds 5–7**: he opens with "I went now to the work and manuscript" —
this is a review of the report itself, not answers to open questions.
Most of the seven points require genuinely new measurement or
engineering work for the paper, not report edits. Full detail in the
table below ("🔬 Timon's round-8 review — new work needed for the
paper"); headline:

- **Point 1** (revised after Omar's review — the first pass wrongly
  assumed "GPU native FEM" meant the CPU reference solver at N=21; his
  own heading says GPU, not CPU): almost certainly the **same underlying
  concern as point 5** — the GPU-native matrix-free solver (Table
  20/20a) hides element-level ("assembly") work inside every CG
  Hessian-vector product, so the true cost of that work is large even
  though the instrumented "assembly" phase shows 0.1–0.6%. A better
  preconditioner is the shared concrete remedy for both points.
  Separately, point 1 also questions whether the benchmark problem is
  demanding enough for a neural operator to be worth using at all ("if
  the FE solution can be done in miliseconds, we do not need NOs any
  more") — a benchmark-design decision, not a bug fix. Suggests scaling
  to problems where FE takes minutes even academically, names a
  tire-tread industrial example.
- **Point 2**: repeat the OOD progressive-shift study (Tables 19/19a,
  currently B1×Neo-Hookean only) for the other five cases.
- **Point 3** (sharpened by Omar): not a generic "DD-NO resolution-
  invariance study" — the precise experiment is training the DD-NO on
  FEM labels generated at a COARSE mesh versus a FINER one, and
  measuring how its accuracy and zero-shot generalization across
  resolutions changes as a function of that label-generation mesh. This
  targets something the physics-informed operator structurally cannot
  suffer from (it never trains on FEM labels), since the DD-NO's
  accuracy ceiling is inherited directly from whatever mesh generated
  its data — that asymmetry is the real comparison being asked for, not
  just "does a DD-NO also generalize." New study, does not exist yet;
  present as a figure.
- **Point 4**: DD-NO wall-clock training time. **Done** — the numbers
  were already committed in `point7b_results/`, just not in Table 21.
  Added in report v54 (`make_v54.py`).
- **Point 5** (Omar's explicit caution, kept): the "assembly is
  negligible" misreading he warns against **is already explicitly
  addressed in the report**, near-verbatim, Section 8.5 — worth pointing
  him to that paragraph, but as confirmation alongside the real
  remaining work, never as a substitute for it. Two other parts of this
  point are real and open: rerun N=1001/1401 to CG convergence (only
  N=501/701 done, per Table 20b), and improve/benchmark the
  preconditioner before drawing scaling conclusions — which is also the
  concrete fix for point 1 above.
- **Point 6**: a richer manufactured-solution family (multiple sine/
  cosine modes) and — technically correct, checked directly against the
  source JSON — the current MMS "Energy" column is the relative error of
  the scalar energy VALUE (superconverges, rate = 2× the H1 rate), not
  the ENERGY NORM he's asking for. The proper energy-norm computation
  already exists in this codebase (Section 4.4's Table 6a, "tangent/
  incremental energy norm") — this is applying existing code to a new
  section, not building new theory.
- **Point 7**: restructure the break-even comparison — primary
  single-query comparison at batch size 1 for both methods (Table
  10c/10d's bs=1 column already is this, just not labeled as primary),
  plus a NEW total-cost-of-ownership break-even between the
  physics-informed operator and a DD-NO specifically (needs point 3's
  study and point 4's wall-clock numbers as inputs). Batched throughput
  numbers stay, but presented separately, not as the headline.

**None of points 1, 2, 3, 5 (preconditioner + rerun), 6 (new family),
or 7 (Comparison B) have been started.** These are substantial —
multiple new studies, not edits — and need Omar's prioritization before
work begins, the same way earlier blocked items waited for a decision
rather than being guessed at.

---

# 📬 Timon's round-7 reply (2026-09-06) — see the table further below

Stored verbatim at `advisor_feedback/2026-09-06_round7_timon.md`. Full
resolution table under "✅ Answered by Timon, round 7" further down this
file. Headline: the torch-fem GPU-FEM comparison is unblocked, the
open-source license question is resolved (Apache 2.0 leaned on, no
approval needed) but publishing is gated on the paper being submitted/on
arXiv first, the continual-learning citation was given
(arxiv.org/abs/2605.04832), and Timon confirmed genuine industrial
relevance for the commercialization question, naming trust/goal-oriented
error estimation as the key future direction — with a paper attachment
on that topic that did not come through and still needs to be requested
from Omar.

---

# ✅ Repo root README fixed; VINO's real (small) connection to this project documented

Omar noticed the repo's root `README.md` was still describing VINO's
paper, not this project, and asked why. Unshallowed the git history
(this session's clone had been shallow) to find out for certain rather
than guessing:

- **2026-07-03** (`f3d78f0`): an early exploration vendored
  `eshaghi-ms/VINO`'s code as-is — the files now under
  `Comparative_Examples/`, `Integration/`, `Practical_Examples/utils/`,
  and the top-level `Practical_Examples/*.py` scripts.
- **2026-07-06** (`fff41c6`): that prototype (`Practical_Examples/omar/`,
  now abandoned) briefly switched to VINO's own closed-form
  energy-integration method.
- **2026-07-09**: a separate pipeline, `Practical_Examples/omar_pfem/`,
  started "isolated from omar/". **This is the only codebase behind
  every result in the report.** It uses ordinary Gauss-quadrature energy
  assembly (already correctly described in the report's Section 5.2),
  not VINO's method.

My first read of this (before unshallowing and checking `omar_pfem/`
itself) overstated the connection — I initially told Omar the whole
project's training method came from VINO. Checked further and corrected
that before acting on it: the only real, current connection is that
`materials_torch.py`'s Mooney-Rivlin and Arruda-Boyce strain-energy
densities were cross-checked against VINO's implementation for
correctness (its own docstring says so) — not that VINO's method was
adopted.

**Fixed, scaled to match the real (smaller) finding:**
- `README.md` at the repo root now describes this project, with an
  honest "Third-party code: VINO" section naming exactly which commits
  and directories are vendored, what the real connection is, and what it
  is not.
- VINO's original README preserved unmodified at
  `Comparative_Examples/VINO_README.md`, so its own attribution isn't
  lost.
- `report_builders/make_v52.py`: adds one sentence after Section 2.4
  (right where the Mooney-Rivlin/Arruda-Boyce derivative computation is
  already discussed) documenting the cross-check honestly, and adds VINO
  as reference [4]. Section 5.2's description of the actual training
  method is untouched — it was already correct.
- The summary was NOT touched — it has no references section or
  material-derivation discussion, so there was nothing there to fix.

Current artefact: report **v52**. Summary stays at v24 (unaffected by
this fix).

---

# ✅ Table formatting made consistent across both documents — report v51, summary v24

Omar asked how to make every table look the same in Word. Checked
directly rather than guessing: table borders were already identical
everywhere (newer `make_vN.py` scripts copy `tblPr` from an existing
table when building a new one, which carries the border settings). What
was NOT copied is header-row shading and font size, because those live
on the cell (`tcPr`) and the run, not on the table-level `tblPr` — so
every table added after the border-copying convention started kept its
borders but lost the styled look.

**Report**: 20 of 50 tables had a styled header (dark fill `#1F2937`,
bold white text, 9pt) and 9pt body text throughout; the other 30 had no
header shading, no bold/white header text, and no explicit font size at
all (rendering in whatever size Word's Normal style defaults to).
**Summary**: same problem, different numbers — 21 of 48 tables styled
(lighter fill `#E8EAED`, bold, 8pt), 27 bare.

Fixed by applying the report's own header style (`#1F2937`, bold, white,
9pt) to the header row of every table in BOTH documents, and 9pt to
every other cell, via `make_v51.py` and `make_summary_v24.py`. Chose the
report's style rather than the summary's own so the two documents match
each other too, not just internally. Verified: all 50 report tables and
all 48 summary tables now share one identical formatting signature; text
content untouched (spot-checked Table 24c's N=33 row, still `13.33×`);
`check_report_tables.py` still `ALL CLEAR` on both.

---

# ✅ Stale "still open" sentences found and fixed — report v50, summary v23

Omar quoted a sentence straight out of the summary: *"Still open: the
same high-DOF study for B2, and the five remaining resolution-invariance
cases."* Checking it found this was one of **five** leftover sentences
in the summary, plus **two** more of the same kind in the report, all
describing a project status from well before this session — never
updated even after the rest of the same document moved past it.

**In the summary** (`make_summary_v23.py`):
- Para 9 ("Resolution invariance…") still described the single-
  resolution, five-unseen-resolution protocol and said "The other five
  cases are still running."
- Para 10 — the sentence Omar quoted.
- Para 12 (the "1. Mesh refinement" item) said "B2's DOF-referenced
  study is not started yet" — it was **cancelled** by Omar on
  2026-08-27, a scope decision, not a pending measurement.
- Para 69 (the "7. Resolution invariance" item) said "done for 1 of 6
  cases. The other 5 are running now" — three paragraphs later the same
  document already says all six have a valid result.
- Para 168 ("Remaining work.") named the out-of-distribution attribution
  extension as the one open item left — see below, that closed too.

**In the report** (`make_v49.py`, `make_v50.py`):
- The point-mapping overview table at the top of the report (row 7,
  "Resolution invariance") still described the old single-resolution
  protocol on B1 × Neo-Hookean only and said "Extending the same
  zero-shot protocol to the other 5 cases is in progress." Section 8.7's
  actual body has long since covered all six cases under the current
  two-resolution joint-training protocol; the overview row was never
  updated to match.
- Section 10's "remaining items" list had one bullet ("Extend the
  out-of-distribution evaluation to isolate the individual contributions
  of the material-stiffness shift and the loading-magnitude shift") that
  read as still open. It is not — Table 19 already isolates exactly
  those two factors (at seven shift levels each) and Table 19a tests
  normalization as a mitigation. With this fixed, all four bullets in
  that list now state they are closed, so the list's own heading ("The
  remaining items are:") was rewritten too, since nothing in it remains.

All five summary sentences and two report sentences now agree with the
rest of their own document. Verified with `check_report_tables.py` on
both outputs afterward — still `ALL CLEAR`.

**Also found while fixing this:** the "SETTLED — do not reopen" table
further below still warned that "§4.4 still ends with a 'NOTE — pending'
line and §10 still lists it. **Both are stale and should be rewritten.**"
— that warning is itself now stale: §4.4 already says the B2 study "is
not planned" and (as of this fix) §10 no longer references it at all.
Corrected in place rather than left to mislead the next session.

---

# ✅ Full table audit, all 48 tables — one error found and fixed — report v48, summary v22

Omar asked directly: check EVERYTHING, and fix whatever is wrong. This
went beyond the earlier duplicate-numbering pass and beyond the earlier
Colab-notebook cross-check (which covered 7 of the 48 tables) — every
single captioned table in the report was matched against its source,
cell by cell.

**Method.** Built a full inventory: 48 numbered tables, mapped in
document order to the 48 raw table objects in the `.docx` (the caption
paragraph is NOT always immediately before its table — several sections,
e.g. Tables 11/12/18/19, place the table BEFORE its own caption, with
analysis paragraphs in between — so position-relative-to-caption is
unreliable and document ORDER was used instead, verified against every
table's column headers).

- **27 tables backed by committed JSON** (`point{2,5,6,7a,7b,8,9}_results/`)
  were checked value-by-value against that JSON: Tables 11, 12, 12b, 12c,
  13 (its final row only — see caveat below), 18–18e, 19, 19a, 20, 20a,
  20b, 21, 22, 23, 24, 24a, 24b, 24c, 24d, 24e.
- **Cross-table identities** were checked where two tables report the
  same underlying measurement: Table 4a's Total(s) column against Table
  7's Native-FEM(s) column (exact match, all 6 cases); Table 5's epoch
  counts against Table 7's Opt.-steps column (exact match for the three
  B1 cases at 100 steps/epoch — the three B2 cases use a different,
  unverified steps-per-epoch, flagged below, not asserted as wrong);
  Table 5 against Table 11's in-distribution column (exact match, all 6);
  Table 5's B2×Neo-Hookean value against Table 13's final row (exact:
  both 9.11%); Table 7's own Speed-up column re-derived from its Native-
  FEM and training-time columns (exact match, all 6 cases).
- **21 tables have no committed source JSON** — mesh convergence (Tables
  1, 2, 1a, 1b, 2a, 2b, 6a), hyperparameters/protocol (3, 4), batch-size
  sweep (6), GPU memory (8), solver agreement (9), GPU-native FEM timing/
  latency/speed-up/break-even (10, 10a–10d), and Table 4a itself. These
  predate the `record_*.py`-plus-JSON convention this project later
  adopted. They were checked for internal and cross-table arithmetic
  consistency (all of which passed, see above) but could not be
  independently re-derived from a source file in this session — this is
  a real, honest limit, not a claim that they contain errors.

**The one error found.** Table 24c (Q4/Q9/operator over the operator's
own 16-member test family), N=33 row, last column
("operator/Q4, single member"): printed as **13.32×**. The correct value
is **13.33×** — computed from the single (non-family) operator and Q4 L2
values at N=33 (0.011362 / 0.00085254, both from
`mms_operator_rate_B1_neo_hookean.json`), which is the same source Table
24a and 24b already draw from. Table 24b, the very next table in the
same report, already states this exact ratio correctly as 13.33×, and
the source JSON's own precomputed field
(`ratios_operator_over_Q4.L2["33"]`) says 13.33 too — so Table 24c's
13.32× was disagreeing with its own neighboring table, not just with the
source. Fixed in both documents: `report_builders/make_v48.py` and
`report_builders/make_summary_v22.py`. Verified afterward by re-reading
both `.docx` files and by re-running `check_report_tables.py` on both
(still `ALL CLEAR`, the fix didn't touch any caption).

**Everything else checked out exactly.** No other numeric discrepancy
was found anywhere in the 27 JSON-backed tables or in the cross-table
identities above.

---

# ✅ Numeric cross-check: Colab notebook output vs. report v47 tables

Omar ran `Round6_Print_All_Final_Results.ipynb` on Colab and pasted the
full output back. Checked it directly against the live `.docx`, table by
table, by extracting each table's actual cell contents from the file:
**Table 18 series, 19, 20, 20b, 22/23, 24, 24a all match exactly**,
number for number (e.g. Table 20's µs/DOF column, Table 20b's predicted
vs. measured CG-per-Newton values, Table 24a's rate-in-h row
-0.59/1.99/+0.78/1.00).

One gap found: the notebook's Table 20 print block only pulled
`N`/`n_dof`/`solve_s`/`us_per_dof` from `gpu_fem_scaling_B1_neo_hookean.json`,
with no block for the report's separate **Table 20a** (Newton iters / CG
iters / CG solves that hit the cap / CG per Newton solve / ms per CG
iteration). Confirmed the same JSON file already holds every field
needed (`stats.newton_iters_total`, `stats.cg_iters_total`,
`stats.cg_failures`, and `stats.t_cg_s` where a timing breakdown exists,
falling back to `solve_s` for the four earliest-commit rows that only
recorded total solve time) — derived all 8 rows from it and verified
each one against the report's Table 20a cell-by-cell (all 8 rows,
including the non-obvious "ms per CG iteration" values 38.8/39.5/40.4/
41.5/40.4/74.8/152.1/296.6). Added the missing block to
`zeroshot_notebooks/cell_print_all_final_results.py`, tested locally
against the real JSON first (same practice as before), rebuilt the
notebook via `make_round6_notebooks.py`, verified 37/37 via
`check_notebooks.py`.

This remaining audit was then completed in full — see the section above
("Full table audit, all 48 tables") for the result: one error found and
fixed (Table 24c), everything else confirmed correct.

---

# ✅ Duplicate table numbers found and fixed — report v47, summary v21

While confirming to Omar that every number in the documents is correct
(not just the ones added this session), built
`report_builders/check_report_tables.py` — scans a `.docx` for every
"Table N." caption (including "Table N (word)." variants; the first
version of the script missed those and was fixed before being trusted)
and flags any number used for more than one table.

**It found three real collisions in the report**, present since long
before this session: Table 3, 4 and 5 each captioned two completely
different tables —

| Number | Mesh-convergence meaning (0 inline references) | The OTHER meaning (heavily referenced) |
|---|---|---|
| 3 | B1 × Mooney-Rivlin mesh convergence | Transolver hyperparameters |
| 4 | B1 × Arruda-Boyce mesh convergence | Training protocol (2 references) |
| 5 | B2 × Mooney-Rivlin mesh convergence | Best validation error (11 references) |

(Table 6 had the same problem — mesh convergence vs. the batch-size
sweep, "Table 6 (revised)" — caught only after fixing the checker's
regex to see that caption form at all.)

**Fix:** renumbered only the mesh-convergence quartet — 1a, 1b, 2a, 2b,
alongside Table 1/2 (B1/B2 × Neo-Hookean) they belong with — since it had
zero inline references anywhere, versus 13+ references for the other
meanings. A compound "(Tables 1–6)" sentence that depended on the old
numeric range was reworded to name the new labels directly. Applied
identically to the summary so both documents agree on what "Table 1a"
etc. means, even though the summary's own Table 3/4 were not technically
ambiguous there (it never had separate hyperparameters/protocol tables).

**Verified:** re-ran the checker on both outputs — `ALL CLEAR, no
duplicate table numbers`. Run this checker again after any future table
addition or renumbering, on both files, before treating either as final.

This audit — checking every OTHER number in every table against its
source JSON, not just numbering collisions — was completed afterward; see
"Full table audit, all 48 tables" above for the result.

---

# ✅✅✅ POINT 2 (ACCURACY/COST PARETO) IS COMPLETE — ALL SIX CASES

B2 × Arruda-Boyce finished last (9/9 resolutions, 7h9m20s wall clock).
Fetched directly from Google Drive (file id
`1dEn57bytN4DkkMDkQ7VnMtY4plT3b0yo`), all nine rows cross-checked against
the run's own stdout independently — exact match.
`point2_results/pareto_B2_arruda_boyce.json`,
`record_pareto_b2_arruda_boyce.py`. Folded into **report v46** (Table
18e) and **summary v19**; both also retext every stale "still running"
sentence left over from the Neo-Hookean and Mooney-Rivlin additions (the
main summary line, Table 18c's and 18d's own speed-up sentences, and the
summary's "Remaining work" closing paragraph) — verified by re-reading
the built `.docx` against the source JSON.

**The headline finding: the training-resolution anchoring effect now
replicates in THREE OF THREE B2 materials.**

| | N=21 anchor ratio | N=33 anchor ratio |
|---|---|---|
| Neo-Hookean | 4.30× | 3.65× |
| Mooney-Rivlin | 3.07× | 3.17× |
| Arruda-Boyce | 2.99× | 2.47× |

Every B2 material's operator error has a sharp local minimum exactly at
N=21 and N=33 — the two resolutions the checkpoint was jointly trained
on — smallest for Arruda-Boyce, largest for Neo-Hookean, but the same
shape every time. **This is no longer "a candidate pattern in one case"
— it is a property of B2's two-resolution training protocol**, observed
at every material tried. What remains genuinely unresolved: whether the
effect traces to the training protocol or to the B2 geometry itself — no
B1 checkpoint trained jointly at two resolutions exists to separate the
two, so every available B1-vs-B2 comparison still differs in both at
once. Not fixable without a B1 run that was never done; stated as an
open question in both documents, not smoothed over.

Speed-up 3,725×–59,234×, same order of magnitude as every other case in
the report.

**With this, every one of Timon's nine round-5 points is now measured
and written up.** See the "Timon's 9 points" status further down for the
full mapping — nothing on that list is outstanding except the two
items Omar owns separately (open-sourcing the GPU-FEM code, sending the
correction email), which were never on Timon's numbered list to begin
with.

---

# ✅ Table 18c is written — report v43, summary v16 (2026-09-02)

`point2_results/pareto_B2_neo_hookean.json` (the training-resolution
anchoring finding, committed earlier) was recorded as data but had not yet
been folded into the actual `.docx` files. Closed now:

- `report_builders/make_v43.py` → `PFEM_Transolver_Report_v43.docx`: adds
  Table 18c (full 9-row Pareto for B2 × Neo-Hookean) plus three paragraphs
  stating the anchoring finding — sharp local minima at N=21 and N=33 (the
  two jointly-trained resolutions), 4.30×/3.65× below neighbouring meshes —
  and explicitly flagging it as a **candidate explanation, not established**
  (B1 vs B2 checkpoints differ in both geometry and training protocol at
  once). Retexts the old "not yet extended to B2" line to "one of three B2
  materials so far."
- `report_builders/make_summary_v16.py`: mirrors the same table and finding
  into `PFEM_Summary_Completed_Work.docx`.
- Both scripts assert every number against the committed JSON before
  writing (`anchor[21] > 3`, `anchor[33] > 3`, `0.8 < b1_ratio_21 < 1.3`) and
  were independently verified afterward by re-reading the built `.docx`
  paragraph/table content back out and checking it against the source JSON.

**Still open, cannot be closed by me:** Timon's email mentions "our paper
on continual learning ... now on arXiv" with **no title and no arXiv ID**
(checked `advisor_feedback/2026-08-28_round6_timon.md` directly). Nothing
is added for this — a fabricated citation would be worse than none. Omar
needs to get the actual reference from Timon first.

**Still to fold in once they finish:** B2 × Mooney-Rivlin's Pareto and
B2 × Arruda-Boyce's Pareto (both running on Colab as of this update).

---

# ✅ B2 WORKS. The "failure" was our own early-stopping metric (2026-09-01)

The fixed-selection run finished: **3 h 18 m 8 s**, early-stopped at epoch
3500 of 4000, best at **epoch 2750** (275,000 steps).

**📁 Now a data file, not just a table here:**
`point7a_results/B2_zeroshot_fixedselection.json`, written by
`record_b2_fixedselection.py`. It carries both eval columns for all seven
meshes, the superseded columns beside them, the late validation curve at full
precision, and a `provenance` field saying it is transcribed from Colab stdout
while the run's own JSONs live on Drive. The builder asserts the rerun is
better on **both** metrics at every mesh before it will write.

| | old selection | **fixed selection** |
|---|---|---|
| val, per_component | 0.9986 | **0.0330** |
| val, both_components | — | **0.0214** |

**Zero-shot eval, the seven unseen meshes, N=101 reference, 20 samples:**

| N | old per_comp | **new per_comp** | old both | **new both** |
|---|---|---|---|---|
| 13 | 0.87137 | **0.23142** | 0.72109 | 0.20462 |
| 17 | 0.87176 | **0.11422** | 0.72133 | 0.10914 |
| 25 | 0.87222 | **0.07202** | 0.72143 | 0.05954 |
| 29 | 0.87235 | **0.07111** | 0.72144 | 0.06550 |
| 37 | 0.87254 | **0.08082** | 0.72142 | 0.07377 |
| 41 | 0.87259 | **0.15113** | 0.72141 | 0.14319 |
| 49 | 0.87270 | **0.26912** | 0.72138 | 0.25963 |

B1 on the same seven meshes: **0.050–0.106** per_component.

**Read it honestly — this is not "B2 now equals B1".**

* **In the mid-range B2 is comparable to B1**: 0.071–0.081 at N=25–37,
  against B1's 0.052–0.067 there.
* **B2 degrades at both ends**: 0.231 at N=13 and 0.269 at N=49, against
  B1's 0.097 and 0.067. Training was at N=21 and N=33, so B2 falls off
  moving away from the training meshes and B1 does not. **B2's spread across
  the meshes is 3.8×; B1's is 2.1×.** That is the honest statement of its
  resolution invariance: real, and weaker than B1's.
* **And the old flatness is explained.** The previous B2 was 0.871–0.873,
  identical to three decimals at every mesh. That was never invariance — it
  was a model emitting nearly the same field whatever it was shown. The new
  model's error *varies* with the mesh precisely because it now tracks the
  problem.

**The cause was one line of ours.** Early stopping and `model_best.pt`
selection used `0.5·(rms(e_u)/rms(u) + rms(e_v)/rms(v))`, which on B2's skewed
component ratio rises while the model improves. Every B2 run stopped at its
first or second validation event — epochs 25, 25 and 225 — so every downstream
diagnosis measured a model trained for 25–50 epochs. Not the physics, not the
data, not the architecture.

### ⚠️ OPEN, and it affects B1: is patience 8 too tight on a curve this noisy?

B2's own history is the evidence. Its both_components error was **0.0369 at
epoch 950**, then rose and fell repeatedly, and only reached **0.0214 at epoch
2750** — 1,800 epochs later. On that curve a patience of 8 validation events
would have stopped it in a local dip and called it converged.

**The three B1 runs used patience 8 and stopped at epochs 900, 775 and 900.**
Their endpoints were verified worse than their best on *both* metrics, so
those stops were not the inversion failure — but "the next 8 events were
worse" is not the same as "no better model exists 1,800 epochs later", and B2
just demonstrated the difference. The B1 numbers are **not known to be wrong**;
they are **not known to be converged either**. Re-running the three B1 cases
with patience 15 would cost about 3 h each. Omar's call, and it should be made
with the report deadline in view rather than on principle.

### ✅ POINT 9's FAMILY HALF IS CLOSED — Q4 and Q9 scored on the 16 members (2026-09-01)

**📁 Now in the repo:** `point9_results/mms_family_fem_B1_neo_hookean.json`,
written by `record_mms_family.py` (the run's own JSON is on Drive at
`pfem_run/mms/family/`). It records all 16 members by name, both rate
estimators, the per-interval rates, and states in `provenance` that the
operator columns come from the operator run and not from this sweep.

Members drawn with the operator run's own call,
`sample_family(16, 31_000_000 + 1)`, verified identical to what
`mms_operator.py` draws, and (0.05, 0.7) is **not** among them — so the FEM and
operator means are over the **same 16 problems** and this is a new measurement,
not a re-dressing of Tables 22–24.

**Family means, L2_rel:**

| N | Q4 | Q9 | operator | operator/Q4 |
|---|---|---|---|---|
| 9 | 1.3515e-02 | 4.1550e-04 | 8.3792e-03 | **0.62×** |
| 17 | 3.4049e-03 | 5.1628e-05 | 8.8261e-03 | **2.59×** |
| 33 | 8.5298e-04 | 6.4423e-06 | 1.2358e-02 | **14.49×** |

**The finding: the operator does not converge, and refining makes it
relatively worse.** Over N=9→33 the family mean L2 rates are Q4 **1.99**, Q9
**3.01**, operator **−0.28** — its error *grows* 1.47× while Q4's falls 15.8×.
That is the honest three-way statement Timon's point 9 was asking for, and it
is now on a family rather than a single member.

**⚠️ Those rates were first written here as 2.13 / 3.21 / −0.30, computed with
h ∼ 1/N. That is wrong.** h is L/(N−1) for N *nodes* per side, and only that
convention reproduces **Table 23's measured Q4 rates, 1.98 in L2 and 1.00 in
H1** — the sweep's own control — and the committed `fitted_rates_in_h` in
`mms_operator_rate_B1_neo_hookean.json`. The corrected numbers are in
`point9_results/mms_family_fem_B1_neo_hookean.json`, whose builder now
*asserts* the Q4 control rather than trusting it. **Quote 1.99 / 3.01 / −0.28.**

**And do not read the two whole-range estimators agreeing as evidence of a
straight line.** N=9, 17, 33 are equally spaced in log h, so the least-squares
slope reduces algebraically to the endpoint slope and the middle mesh cancels
— they agree to the last digit for every method by construction. Per interval:

| method | N=9→17 | N=17→33 |
|---|---|---|
| Q4 L2 | 1.989 | 1.997 |
| Q9 L2 | 3.009 | 3.003 |
| **operator L2** | **−0.075** | **−0.486** |

Q4 and Q9 hold their textbook rates on both halves, which is what makes them a
control. The operator is negative on **both** intervals and steepens — so what
this sweep establishes is the **sign**, everywhere, not a single magnitude.
The single-member run fitted −0.59 on the same quantity.

**⚠️ The 0.62× at N=9 is NOT a bug, and a previous session raised a false alarm
about exactly this.** The ceiling argument constrains **Π**: the operator
minimises the same discrete functional over the same Q4 space, so nothing it
produces can have lower Π than the Q4 solution. **Π is not L2.** A field that
does not minimise Π can sit closer to u\* in L2 by partly cancelling Q4's own
discretisation bias, and that is what N=9 does. §8.11 already carries this
correction; do not "fix" it.

**Table 24 is vindicated at N=17**: its single-member 2.42× against the family's
2.59×, so the published comparison was representative there. At **N=9** the
single member was materially easier — operator 5.0351e-03 against the family's
8.3792e-03, **66% worse on the family** — so any N=9 statement should quote the
family. N=33 moves 9%.

**Q4's spread across the family is negligible** — stdev/mean 0.002–0.003 in L2,
0.000 in H1, 0.007 in stress. The FEM is essentially member-independent.

### ✅ CLOSED — the operator's spread over the family, no retraining needed (2026-09-01)

The open question above turned out not to need re-running anything: the
three already-trained checkpoints (N=9, 17, 33) were re-scored per member
(`omar_pfem/mms_operator_per_member.py`, `Round6_MMS_Operator_PerMember.ipynb`)
in minutes on an A100, no training. Recorded at
`point9_results/mms_operator_per_member_B1_neo_hookean.json`
(`record_mms_operator_per_member.py`), which asserts its means reproduce
`operator_family_mean` from the family sweep to 4 significant figures before
writing — they do, exactly, which is the check that this is the same
computation and not a different family or checkpoint.

**Answer: no, not consistent, in any norm, at any mesh — and in two of the
four norms it is not even close.** Q4's std/mean is 0.000–0.007 across all
four metrics and all three meshes (essentially member-independent, already
known). The operator's:

| N | L2 | H1 semi | stress | energy |
|---|---|---|---|---|
| 9 | 0.384 | 0.003 | 0.006 | 0.549 |
| 17 | 0.258 | 0.016 | 0.015 | 0.370 |
| 33 | 0.425 | 0.134 | 0.120 | 0.736 |

(Q4's own std/mean for comparison: L2 0.002–0.003, H1 0.000, stress 0.007,
energy 0.003, essentially flat across the family and the mesh.)

**H1 and stress start indistinguishable from Q4's own spread at the
coarsest mesh** (0.003–0.006 at N=9, against Q4's 0.000–0.007) **and grow
away from it with refinement** (0.12–0.13 by N=33) — the same
ceiling-proximity effect Table 24d already reports for the *mean* ratio,
now shown to affect per-member *reliability* too: when the operator sits
close to the Q4 optimum it inherits some of Q4's member-independence, and
as optimization error comes to dominate at finer meshes both the mean error
and its spread across the family grow together. **L2 and energy are never
close** — 0.26–0.55× already at N=9, far above Q4's 0.002–0.003 — the same
two metrics Table 24d shows diverging outright in the mean.

**In the report as of v40**: §8.11, Table 24e, next to Table 24d. Mirrored
into the summary (v13). Both builders assert the same mean-agreement check
as `record_mms_operator_per_member.py` before writing.

### ✅ REPORT v39 AND SUMMARY v12 ARE WRITTEN (2026-09-01)

Builders committed: `report_builders/make_v38.py`, `make_summary_v11.py`,
`make_v39.py`, `make_summary_v12.py`. All ran clean; the documents are at
`/tmp/PFEM_Transolver_Report_v39.docx` and
`/tmp/PFEM_Summary_Completed_Work.docx` (previous states preserved as
`.pre_v11.docx`, `.pre_v12.docx`). All read the same JSONs, so they cannot
disagree.

**v39/v12 add Table 24d**: `mms_family_fem_B1_neo_hookean.json` (built
2026-09-01, see "POINT 9's FAMILY HALF IS CLOSED" below) always carried H1
semi-norm, stress and energy on the 16-member family alongside L2 — v38/v11
only read the L2 field into Table 24c. Nothing new was measured; v39 reads
the same file's other three fields. **The finding is not the same shape in
every norm**: operator/Q4 at N=33 is 14.49× in L2 and 8.02× in energy, but
only 1.45× in H1 semi-norm and 1.40× in stress — H1 and stress stay near the
variational ceiling through the whole refinement (fitted rate +0.74 H1,
+0.77 stress, both still improving, against Q4's 1.00) while L2 diverges
outright (fitted rate −0.28). Per-interval rates for H1/stress stay positive
on both halves (+0.96/+0.97 then +0.52/+0.57), so the direction does not
reverse — it just slows as Q4's own error falls faster. This is the same
inversion Table 24a/24b already reported on the single member (H1 and stress
protected near the ceiling, L2 not); Table 24d confirms it on 16 members
rather than one.

**A claim checked and found already correct, not missing**: the GPU-FEM
break-even correction (1,133–95,038 against the GPU baseline, not the
7,600–96,000 Timon is working from) has been in both documents' text and
tables since `make_v28.py`/`make_v30.py` (report) and `make_summary_v4.py`
(summary) — verified 2026-09-01 by reading the compiled .docx files directly,
not the builder source. **What is still outstanding is sending it to Timon**,
which is Omar's own action, not a document edit — see "The correction Timon
needs" below.

**What changed in the report:**

| where | from | to |
|---|---|---|
| §8.7 ¶"Two limits" | "do not reach a usable accuracy… the cause is under investigation" | seven paragraphs: the load repair, the metric defect, the rerun, **Table 12b** (all seven meshes, both conventions, before/after), and what it does *not* establish |
| §8.11, after "Two qualifications" | — | **Table 24c** + four paragraphs: Q4/Q9/operator over the 16-member family |
| §8.11 closing limitation | "every error … scored on the single member" | the family answers half; what remains is one geometry, one material |
| ¶6 scope note | "point 7 covers B1 × Neo-Hookean only" | four of six; the two B2 materials are the only ones outstanding |
| §4.4 "NOTE — pending" | promises a B2 ~10M-DOF study | says it is deliberately confined to B1 and not planned |
| §10 first bullet | "Complete the B2 ~10M-DOF study… the item the advisor explicitly requested" | re-run the two B2 materials |
| §10 last bullet | "the retrained models do not yet reach a usable accuracy" | four of six done, the other two are the bullet above |

**⚠️ A claim removed for lack of evidence.** §10 said the B2 ~10M-DOF study was
"the item the advisor explicitly requested". **Neither stored advisor email —
`advisor_feedback/2026-08-26_round5_timon.md` nor `2026-08-28_round6_timon.md`
— contains that request**, checked directly. Earlier rounds are not stored
here, so this file's stronger claim ("never among Timon's requests") is **not
verified for rounds 1–4**; v38 simply drops the attribution rather than
asserting the opposite.

**Both builders assert before they write.** v38 refuses to emit Table 24c
unless the family sweep's Q4 control reproduces Table 23's measured 1.98/1.00,
and refuses to emit Table 12b unless the rerun beats the superseded run on
**both** metrics at **every** mesh. v38 also recomputes B1's span and spread
from `zeroshot_B1_*.json` rather than quoting v37's prose.

### ✅ B2 × Mooney-Rivlin fixed-selection: DONE, both training and eval (2026-09-01)

**📁 Now a data file:** `point7a_results/B2_mooney_rivlin_zeroshot_fixedselection.json`,
written by `record_b2_mooney_rivlin_fixedselection.py`. Second of the two B2
materials gated behind Neo-Hookean's rerun — **only Arruda-Boyce is left.**

Ran the full 4000-epoch budget (did not early-stop), best at epoch 3350,
`combined_val_error` 3.4406e-02 — comparable to Neo-Hookean's 2.14e-02 and
to B1's own range. Zero-shot on the seven unseen meshes, per_component:

| N | 13 | 17 | 25 | 29 | 37 | 41 | 49 |
|---|---|---|---|---|---|---|---|
| Mooney-Rivlin | 0.1946 | 0.0790 | 0.0661 | 0.0730 | 0.1220 | 0.1997 | 0.3246 |
| Neo-Hookean (recorded) | 0.2314 | 0.1142 | 0.0720 | 0.0711 | 0.0808 | 0.1511 | 0.2691 |

**The finding: a clean crossover, not a uniform win or loss.** Mooney-Rivlin
beats Neo-Hookean at all three coarse meshes (13, 17, 25) and loses at all
three fine meshes (37, 41, 49) — training was at N=21 and 33 for both
materials, so this is not explained by one training closer to the fine end.
**Mooney-Rivlin's spread is 4.91×**, wider than Neo-Hookean's 3.78×, both far
past B1's worst case 2.11×. B2's resolution invariance is real for both
materials tried so far and gets weaker each time it's checked more closely.

⚠️ **Wall clock is incomplete, and said so in the record.** This run resumed
from epoch 1651/4000 after the container restart that lost track of it (see
below); the recorded `train_wall_clock_s_this_segment_only` covers only that
segment, not the true total from epoch 1.

**Arruda-Boyce started immediately after** (same cell, same session) — epoch
200/4000 as of this update, still in the noisy early phase every B2 run
shows (see B2×NH's own history: 0.037 at 950, converging only by 2750).
Ceiling ~3h46m from a fresh start if it runs the full budget.

### ⚠️ Container restart lost /tmp and confused (but did not lose) a live Colab tab (2026-09-01)

The session's container restarted mid-conversation: `/tmp` was wiped (the
report/summary .docx builds live only there, see below) and the local git
checkout briefly reverted to an old commit (fixed by `git merge --ff-only`
against origin — nothing was lost on GitHub). Separately, this made a
still-running B2 training Colab tab (commit `69b594c`, running continuously
since early in the session) appear to be gone from Colab's "Active sessions"
list, which led to relaunching it in a second tab — a real but
non-destructive duplicate-write risk (both tabs would have periodically
overwritten the same `train_state_latest.pt`/`metrics_history.json` on
Drive) that was caught and resolved by closing one tab. **Lesson for next
time: before concluding a Colab tab is dead, check its own scrollback first
— "not in Active Sessions" is not proof of that.**

**The .docx artifacts (report v39, summary v12 at the time) were recovered
from Omar's own copy** (sent to him earlier via chat) rather than rebuilt
from scratch, since the builder chain has no committed base document before
`make_v28.py`'s `PFEM_Transolver_Report_v27.docx` and none of v1–v27 was
ever committed to git — **the .docx files are a single point of failure
outside the repo.** Whoever holds the report/summary should keep the latest
copies somewhere durable; this file cannot substitute for that.

### ✅✅✅ POINT 7 IS COMPLETE — ALL SIX CASES, ZERO OUTSTANDING (2026-09-01)

**Arruda-Boyce finished too**, same session, right after Mooney-Rivlin.
**📁** `point7a_results/B2_arruda_boyce_zeroshot_fixedselection.json`
(`record_b2_arruda_boyce_fixedselection.py`). This closes the item every
report revision since v35 has named as "the only outstanding" one — there
is no longer a B1/B2 × material combination without a valid zero-shot
result.

**This run's early stop actually fired** (the other two ran the full
4000-epoch budget): best at epoch 1450, stopped at 2200 — exactly
`best_epoch + patience(15) × validate_every(50)`. First direct confirmation
that patience 15 works as designed on a real B2 curve, not just in theory.
Note patience 8 would have stopped by epoch 1850 on the SAME run, before
1450's best was even reached retrospectively knowable — one more data point
for the still-open "is patience 8 too tight" question, though it does not
settle it (this run used patience 15 throughout).

**All three B2 materials, final numbers:**

| material | old (broken) val @ epoch | new val @ epoch | zero-shot per_component |
|---|---|---|---|
| neo_hookean | 0.9986 @ 25 | 0.0330 @ 2750 | 0.0711–0.2691 |
| mooney_rivlin | 0.9752 @ 25 | 0.0344 @ 3350 | 0.0661–0.3246 |
| arruda_boyce | 1.0267 @ 225 | 0.0580 @ 1450 | 0.0940–0.3402 |

B1 (all three materials, same seven meshes): 0.0504–0.1064.

**Arruda-Boyce is the worst of the three B2 materials at 6/7 meshes** — the
only exception is N=17, where it beats Neo-Hookean. Its baseline error is
consistently higher than the other two's. **But its spread (3.62×) is
narrower than both Mooney-Rivlin's (4.91×) and Neo-Hookean's (3.78×)** —
being the least accurate B2 material and having the most even error across
resolutions are two different properties, and this case is the one that
separates them. **All three B2 materials exceed B1's worst-case spread
(2.11×).** B2's resolution invariance is real for every material tried and
weaker than B1's for every one.

**No more B2 zero-shot cells to run.** `Round6_B2_FixedSelection_All.ipynb`
has nothing left to do (both its GATE and its two targets are complete).

### ✅ REPORT v41 AND SUMMARY v14 ARE WRITTEN (2026-09-01)

Builder: `report_builders/make_v41.py` / `make_summary_v14.py`. Adds Table
12c (all three B2 materials side by side, mirroring Table 12's own layout),
retexts the ¶6 scope note and both of §10's remaining-items bullets from
"four of six" / "outstanding" to done, and replaces §8.7's stale "only
Neo-Hookean has been re-run" paragraph with the crossover/spread/patience
findings above. Built on the .docx Omar handed back after the container
restart; verified against the source .docx before building (Table 24d
presence) and all four edits verified landed correctly afterward.

### ✅✅ B1 × Arruda-Boyce Pareto is DONE — all three B1 Pareto sweeps complete (2026-09-01)

**📁** `point2_results/pareto_B1_arruda_boyce.json`
(`record_pareto_b1_arruda_boyce.py`). Fetched **directly from Google Drive**
(file id `10Jt2W0sCIHhoVG1Tb9NALMeEEhjaW3ua`), not transcribed from stdout —
every row is the run's own recorded value, no transcription risk.

All 9/9 resolutions, checkpoint fingerprint `bff6d7f2...`. Speed-up
3,452×–54,731×, between Neo-Hookean's 1,630×–25,676× and Mooney-Rivlin's
3,574×–56,355× — all three B1 materials in the same order-of-magnitude
band. **The operator error bottoms at N=37 (3.53%) then rises to N=49
(4.27%)** — the same shape as Neo-Hookean (also bottoms at N=37); **Mooney-Rivlin,
whose error falls monotonically all the way to N=49, is the exception among
the three, not the rule.** As with the other two materials, the cheapest
FEM solve (N=13, 0.474%) is already 7.4× more accurate than the operator's
best.

**Point 2's B1 side is now fully done**: Neo-Hookean, Mooney-Rivlin and
Arruda-Boyce all have complete 9-resolution Pareto sweeps. Not yet in the
report (Table 18 currently covers Neo-Hookean and Mooney-Rivlin only — see
report_builders for the next version to add).

### ✅ B2 × Neo-Hookean Pareto is DONE — and it shows something B1's never did (2026-09-02)

**📁** `point2_results/pareto_B2_neo_hookean.json`
(`record_pareto_b2_neo_hookean.py`). Fetched directly from Drive (file id
`1hjWfVmpXDmNdrTjrwk7E8QVecweNzTOc`). 9/9 resolutions, speed-up
1,738×–27,392×. 7h11m wall clock.

**Finding: the operator's error is not smooth in N.** It has sharp local
minima exactly at N=21 and N=33 — the two meshes this checkpoint was
**jointly trained on** — 4.30× and 3.65× lower than the mean of each
point's immediate neighbours. **B1 × Neo-Hookean's Pareto, from a
checkpoint trained at N=21 only, shows no such dip at N=21** (ratio 1.01×,
flat). Candidate explanation, stated as a candidate and not established:
joint training at two specific resolutions leaves two visible anchor
points that single-resolution training does not. Cannot be disentangled
from a possible B2-geometry effect, since the two compared checkpoints
differ in both geometry and training protocol at once.

**Mooney-Rivlin's B2 Pareto started immediately after, same session.
Arruda-Boyce has not started.**

### 🎯 NEXT

1. **B2×Mooney-Rivlin Pareto** — running now, started right after Neo-Hookean
   finished (same Colab session).
2. **B2×Arruda-Boyce Pareto** — not started yet; needs its own launch (or
   the same session to reach it after Mooney-Rivlin).
3. **Fold B2 Pareto into the report** once at least Mooney-Rivlin (and
   ideally Arruda-Boyce) finish — same pattern as Table 18/18a/18b for B1.

### ✅ REPORT v42 AND SUMMARY v15 ARE WRITTEN (2026-09-01)

Builder: `report_builders/make_v42.py` / `make_summary_v15.py`. Adds Table
18a (Mooney-Rivlin) and 18b (Arruda-Boyce) — Mooney-Rivlin's Pareto had been
sitting recorded in `point2_results/` for days without ever reaching the
report, a gap only noticed while adding Arruda-Boyce's. Retexts the stale
"run for B1 × Neo-Hookean only" closing line and the summary's section-9
header. **Caught one authoring bug before it shipped**: the first version
of `make_summary_v15.py` was missing the `new_table(...)` call for
Arruda-Boyce entirely (a plain omission, not a tool issue) — verified by
counting tables with the Pareto header (got 2, expected 3), found the
missing line by reading the script, rebuilt from the `.pre_v15.docx`
backup rather than patching the already-wrong output.

### 🎯 NEXT

1. **B2 Pareto, all three materials** — every B2 checkpoint now exists, so
   `cell_pareto_B2.py` (`Round6_Pareto_B2.ipynb`) will run for all three on
   its next launch instead of reporting "no checkpoint yet" for two of them.
   This is the only Pareto work left — B1's is done for all three materials
   as of v42.

---

# 🔴 ~~SUPERSEDED — every B2 "failure" conclusion in this file is provisional~~ — RESOLVED ABOVE, the list below still says what is void

**Read this before quoting any B2 number or diagnosis from anywhere below.**

On 2026-08-31 the cause of B2's apparent failure was found, and it was ours:
early stopping and `model_best.pt` selection used the per-component metric
`0.5·(rms(e_u)/rms(u) + rms(e_v)/rms(v))`, which on B2's skewed component ratio
**rises while the model improves**. Every B2 run in this project stopped at its
**first validation event**. With `--selection_metric both_components` the same
case reached **0.0598** per_component at epoch 950 — against its old 0.9986,
and against B1 × Neo-Hookean's 0.0657 on the identical metric.

**So every diagnosis below that measured a B2 model was measuring a model
trained for 25–50 epochs**, and none of it supports a claim about B2:

| conclusion below | what it is worth now |
|---|---|
| "B2 zero-shot fails, ~0.87–0.89 eval, ~1.0 val" | **superseded** — that was the crippled run |
| batch size / `loss_force_norm` / input normalisation ruled out | **void as evidence** — both arms were cut short |
| joint training ruled out (single-resolution arms) | **void** — those arms early-stopped at 450 on the same metric |
| "the model under-responds to its input", roughness 3.0×, correlation erratic | **describes a 25-epoch model**, not B2 |
| the load repair, the Π/functional check, the Dirichlet ramp | ✅ **still valid** — physical and geometric measurements, no metric involved |
| **every B1 result** | ✅ **untouched and valid** — B1's metric was checked and does not invert |

**Nothing about B2 goes into the report until the fixed-selection run finishes
and its zero-shot eval is in.** The other two B2 materials (Mooney-Rivlin,
Arruda-Boyce) carry the identical defect and need the same rerun before any
B2 row is quoted for them either.

**The rerun for those two is ready and gated:**
`zeroshot_notebooks/cell_b2_fixed_selection_all.py`
(`Round6_B2_FixedSelection_All.ipynb`). It **refuses to start** until
`zeroshot_B2_neo_hookean_fixedsel/zeroshot_eval.json` holds all seven
resolutions, and prints those numbers first — the two cases are ~7 h 32 m of
A100 spent on the premise that the fix holds, so the premise is checked by the
cell rather than remembered by a person. Per case: a new directory, both caches
copied in, `--selection_metric both_components`, patience 15, 4,000 epochs
(~3 h 46 m at the measured 3.387 s/epoch), then the zero-shot eval. Resumable
at every validation event; a case whose eval is complete is skipped. The gate
logic was exercised against partial, complete, unreadable and missing files.

---

# ⛔ SETTLED — do not reopen, and do not list as "remaining work"

**Read this before ranking any priority or answering "what is left?".** Twice
now a session has read a *scientific caveat* in the report and reported it as
*unfinished work*. The report hedges properly; that is not the same as a gap.
This list is the authority.

| item | status | why it looks open but is not |
|---|---|---|
| **B2 mesh-convergence at ~10M/40M DOF** (§4.4) | **CANCELLED by Omar 2026-08-27** — *"خلص ملغي ما بدنا ياه"* | Was stale in an earlier revision (§4.4 ended with a "NOTE — pending" line and §10 listed it); **fixed as of report v50** — §4.4 now says "not planned" and §10 no longer references it. It was never among Timon's requests |
| **Data-driven vs physics-informed, other cases** (§8.9) | **COMPLETE** — point 7b, Table 21 | §8.9 closes with *"The comparison covers one case"*, which is a caveat, not a to-do. Timon's instruction was *"I'd start with one specific problem such as B1-Neo Hookean. **Based on the results, we can decide then.**"* One case was the instruction; extending is a decision that was never taken |

**The rule:** a sentence in the report that names a limitation is the report
being honest. Only this file says what is actually outstanding.

---

# MASTER TABLE — where every item stands

Current artefacts: report **v56**, summary mirrored (v24), branch
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
| **R5-2** Pareto, remaining cases | **B1×MR and B1×AB are unblocked now** — their checkpoints are valid; cell at `zeroshot_notebooks/cell_pareto_remaining_B1.py`, ~2 h per case (measured 1 h 54 m / 6 h 24 m on B1×NH). **B2×Neo-Hookean is unblocked too, as of 2026-09-01** (fixed-selection checkpoint exists) — `zeroshot_notebooks/cell_pareto_B2.py` runs it now, does not wait for the other two, and reports "no checkpoint yet" and skips a case rather than asserting. B2×MR and B2×AB stay blocked on their fixed-selection retrain finishing |

## ✅ Answered by Timon, round 7 (2026-09-06) — see `advisor_feedback/2026-09-06_round7_timon.md`

| Item | Resolution |
|---|---|
| **R6** benchmark the GPU-FEM solver, computational efficiency | **Unblocked.** No preference between TensorMesh and torch-fem — "torch-FEM is also fine," the only requirement is "an efficient GPU implementation... for a fair comparison to a NO," which torch-fem satisfies. The comparison can now be built |
| **R6** open-source the GPU-FEM code — approval | **Resolved: no institutional/advisor sign-off needed.** "There is no approval necessary" |
| **R6** open-source the GPU-FEM code — license | **Resolved, leans Apache 2.0**: "both MIT or Apache License are fine... Apache is more general and also TensorMesh is based on Apache license" |
| **New gate, not previously known**: when to actually publish | **Do not open-source or make any repository public yet**, license question notwithstanding: "I'd wait though until the paper is on arxiv and submitted to a journal... not before" |
| Continual-learning citation | **Done.** Fetched the real title/authors from arXiv (not guessed): "Replay-Based Continual Learning for Physics-Informed Neural Operators" (Wang, Eshaghi, Zhuang, Rabczuk, Liu, arXiv:2605.04832) — uses the same Transolver architecture this report does. Added as reference [5] and one sentence at the end of §8.6, framed accurately as incremental adaptation with some retraining budget, distinct from this report's own zero-shot (no retraining) approach. `report_builders/make_v53.py`, report **v53** |
| Commercial/industrial relevance (Omar's question) | **Confirmed as real**, not speculative: cites ANSYS SIMAI as an existing commercial neural-operator product (his own characterization, not confirmed fact) and "several start-up companies about CAE acceleration." Names **trust / accuracy verification without ground truth** as the key obstacle, proposes **goal-oriented error estimation (GOEE)** adapted from FEM as a candidate future direction — explicitly framed as future work, not a request to act now |
| GOEE/trust paper Timon says he attached | **Not received in this session** — his email says "I attach a recent paper" but no attachment came through. Do not assume or act on its content until Omar shares the actual file |

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

### ✅ DONE, see top of file — B1 × Arruda-Boyce Pareto finished 2026-09-01. History below is how it got there.

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

**✅ ANSWERED — B1 DOES NOT INVERT** (run on `5bccf12`, endpoints recovered
from each run's own `train_state_latest.pt`):

| case | best → endpoint, per_component | best → endpoint, both_components |
|---|---|---|
| neo_hookean | 0.0657 → 0.2512 (up) | 0.0444 → **0.1761 (up)** |
| mooney_rivlin | 0.0827 → 0.1316 (up) | 0.0594 → **0.0730 (up)** |
| arruda_boyce | 0.0783 → 0.0926 (up) | 0.0462 → **0.0525 (up)** |

Both metrics agree in all three cases: `model_best.pt` really is the better
checkpoint, and B1 genuinely degrades after it (Neo-Hookean by 4×). **Early
stopping did the right thing on B1 and the wrong thing on B2**, and the
difference is the *stability* of the component ratio — 3.34–4.17 and tight on
B1, skewed on B2 (per-sample mean 1.90 against 0.90 aggregate).

**So the picture is now closed and consistent:**

| | B1 | B2 |
|---|---|---|
| does the metric invert? | **no** | **yes** |
| did early stopping keep the right model? | **yes** | no — first validation event |
| reported numbers | **stand**, conservative by 1.36–1.71× | come from an inverted selection |

### 🧰 A full Drive audit, and a real save gap in the MMS family sweep (2026-08-31)

**`zeroshot_notebooks/cell_drive_audit.py`** (notebook `Round6_Drive_Audit.ipynb`)
walks the whole `pfem_run` tree rather than pattern-matching a list of
expected filenames the way `cell_check_results.py` does. Every file with size
and time; every JSON opened and summarised **by shape, not by name**; every
checkpoint fingerprinted and every result file tied back to the checkpoint
that produced it, with an orphaned fingerprint flagged; every study checked
for completeness (Pareto 9, zero-shot eval 7, MMS family 3) so a partial file
is never read as finished; leftover `.tmp`, `EARLY_STOPPED` and unreadable
JSON all reported. Exercised against all 26 result files in the repo — it
found one real shape it could not read (the operator-rate rows store
`Q4: {L2: ...}` where the family sweep stores `Q4: {L2_rel: {mean: ...}}`),
which is fixed, and then handled all 26 with no failures.

**`mms_family_fem.py` had no save until the very end** — one `json.dump` after
all three meshes. A run killed anywhere before that left **nothing** and
restarted from the first member, and N=33 is over an hour and runs last. It
now appends each member's four numbers to a `.progress` file as they are
computed and skips whatever is already there on restart, keyed by the drawn
members so another family's results cannot be inherited. `out_json` is still
written only when every mesh is done, so its existence still means finished,
and the progress file is deleted at that point. Verified by killing a run
mid-sweep: no final JSON, progress held `{'5': [0, 1]}`, and the restart
skipped exactly those two and finished.

### ⚠️ THREE OF THE SEVEN CLOSED B2 CANDIDATES WERE CLOSED WITH THE FAULTY SELECTION

This has to be said plainly before anyone treats the candidate list as
settled. Every candidate judged by a *val number* was judged by the metric now
known to rank B2's checkpoints backwards, on runs that early-stopped at their
first validation event. The list splits cleanly:

| candidate | how it was closed | still safe? |
|---|---|---|
| mesh-dependent load | physical: total load agrees to **0.001%** across meshes | ✅ safe — no metric involved |
| data and functional | physical: Π's minimum at s=1.0 in 6/6, W/U = 1.9951–2.0021 | ✅ safe |
| Dirichlet ramp | geometric: peak/rms 2.46 against B1's 2.44 | ✅ safe |
| `loss_force_norm` | val number | ⚠️ judged by the faulty metric |
| batch size | val number, 0.9888 vs 0.9444 at matched steps | ⚠️ judged by the faulty metric |
| **input normalisation** | val number, **0.9910 vs 0.9986**, early stop at 225 | ⚠️ **judged by the faulty metric, and both runs stopped at their first validation** |
| joint training | val number, 0.9622 / 1.0372 vs 0.9986 | ⚠️ judged by the faulty metric — though the probe later confirmed both arms really were poor |

**The four ⚠️ comparisons are weak, not wrong.** Each compared two arms under
the *same* metric, so the ordering may well hold; what they cannot support is
any statement about where those configurations *converge*, because none of
them was trained to convergence. "Input normalisation moves 0.9986 → 0.9910"
is really "after ~50 epochs, with selection inverted, the two are close".

**Input normalisation is the one worth re-testing**, and it is the most
physically plausible remaining cause of the symptom that is actually measured:
the model under-responds to its input (prediction variability 0.10–0.24
against targets' 0.31–0.64). The load channels sit at `rms(f)/rms(E)` =
2.3e-05 to 7.2e-05 and are exactly zero on 95–97% of nodes, so two of the four
input channels carry the loading at 10⁻⁵ of the scale of the one carrying
stiffness. That is a real candidate mechanism, and the run that was supposed
to rule it out never converged.

**Order of operations:** let the current fixed-selection run finish first. It
is the same configuration minus the defect, so it sets the honest baseline any
re-test must beat. Only then is a second run (`--normalize_inputs 1
--selection_metric both_components`, same ~3 h 36 m) worth the GPU.

### 🟢🟢 B2 IS NOT BROKEN. IT WAS BEING STOPPED AT ITS FIRST VALIDATION EVENT (2026-08-31)

From the Drive audit, `zeroshot_B2_neo_hookean_fixedsel/metrics_history.json`,
the fixed-selection run **still in progress** at epoch 1200 of 4000:

```
24 validation events, epochs 50..1200
best combined_val_error (per_component)  0.05978  at epoch 950
best both_components                     0.0369   at epoch 950
[selected on both_components]
```

**Against the same B2 case's previous best of 0.9986, and against B1 ×
Neo-Hookean's 0.0657 on the identical per-component metric.** B2 is now at
**0.0598** — the same league as B1, from a run that has not finished.

**⚠️ I predicted this would not work, and I was wrong.** The estimate in this
file and in the cell header — "not expected to rescue B2 … expect it to land
near 0.68, an eighth of the gap" — was extrapolated from the single-resolution
arms, and those arms *also* early-stopped on the inverted metric at epoch 450.
Reasoning from runs that were themselves cut short by the defect under test
was the error.

**The cause of the entire B2 "failure" was ours, and it was one line.** Early
stopping and `model_best.pt` selection used
`0.5·(rms(e_u)/rms(u) + rms(e_v)/rms(v))`, which on B2's skewed component ratio
*rises while the model improves*. Every B2 run stopped at its first validation
event with patience exhausted, and every downstream diagnosis — the load, the
ramp, the family, the batch size, the functional, joint training — was
measuring models that had been trained for 25 to 50 epochs.

**What this does to the report.** §8.7's B2 zero-shot row, §9.1's account, the
`~0.87` eval figures and every sentence describing B2 as failing are now
provisional. **Nothing about B2 goes into v38 until this run finishes and its
zero-shot eval is in.** The B1 results are untouched and stand.

**What is still to come from the run:** epochs 1200→4000 (or early stop at
patience 15 on the metric that orders correctly), then the zero-shot eval at
the seven unseen meshes against N=101, which is the number the report quotes.

### 📌 B2 BASELINE RE-SCORED ON BOTH METRICS, and it reproduces exactly (2026-08-31)

Stage 1 of the retrain cell, on `5091b02`. The existing B2 × Neo-Hookean
checkpoint, the seven unseen meshes, 20 samples, N=101 reference:

| N | per_component | both_components |
|---|---|---|
| 13 | 0.87137 | 0.72109 |
| 17 | 0.87176 | 0.72133 |
| 25 | 0.87222 | 0.72143 |
| 29 | 0.87235 | 0.72144 |
| 37 | 0.87254 | 0.72142 |
| 41 | 0.87259 | 0.72141 |
| 49 | 0.87270 | 0.72138 |

**The per_component column reproduces the stored figures to all five digits**
(0.87137 → 0.87270 is exactly what this file already recorded), so the eval is
deterministic and this baseline is the report's own number, not a re-derivation
of it. The both-components column is new: **0.721**, against B1's 0.050–0.106.

**Two things worth carrying into v38.** First, B2's per-sample **standard
deviation is 0.328** — a third of the mean. The figure is an average over
wildly unequal samples, not a tight result. Second, the flatness across the
mesh (0.153% spread) is **not** resolution invariance in the sense the study
is claiming: it is the model emitting nearly the same field whatever mesh it
is shown, which the probe already measured directly (U(pred) spread 1.15–1.66×
against targets spanning 3×). B1's columns move by 85.7%, 111.3% and 79.1%
across the same meshes *because* B1 tracks the problem. That contrast should
be stated rather than left for a reader to misread as B2 being the more
resolution-invariant of the two.

### 🔄 B2 RETRAIN WITH FIXED SELECTION — decided, cell ready (2026-08-31)

Omar chose to re-derive the number rather than caveat it. Cell:
`zeroshot_notebooks/cell_b2_fixed_selection.py`, run through
`Round6_RUN_THIS.ipynb`.

**Configuration** — joint N=21+33, 400 train / 100 val per resolution, batch 8,
4,000 epochs, `--early_stop_patience 15`, **`--selection_metric
both_components`**, then a zero-shot eval at the seven unseen meshes
(13,17,25,29,37,41,49) against N=101 with 20 samples. Output goes to a NEW
directory `zeroshot_B2_neo_hookean_fixedsel`; the sample cache and the N=101
fine-reference cache are **copied** in, so nothing is regenerated (the sample
cache alone is 7+ hours of FEM) and the existing run is only read.

**Cost** from the measured rate: the single-resolution arm did 450 epochs of
400 samples in 12 m 8 s = 1.618 s/epoch, so two resolutions at 800 samples an
epoch is ~3.24 s/epoch and 4,000 epochs is about **3 h 36 m** on an A100. Less
if it plateaus. Resumable at every validation event.

**⚠️ It is not expected to rescue B2** and the cell says so in its own header.
B2's best both-components error is 0.68 against B1's 0.044; the selection
defect cost it 0.77 → 0.69, an eighth of the gap. What it buys is being able
to write *"B2 was trained to convergence under a selection criterion that
orders checkpoints correctly, and this is where it lands"* instead of *"B2's
number came from a run our own metric stopped early"*.

**Verified before the cell was written, by running it** — a real
two-resolution training run end to end on the new code; both metrics printed
each validation event with the selected one named; `model_best.pt` tracking
the both-components number while `combined_val_error` still records the
per-component one at that checkpoint; resume from `train_state_latest.pt`
continuing at the right epoch against the right stored best; **`model_final.pt`
written on the early-stopped path**; `--selection_metric per_component`
reproducing the old behaviour; and eval reporting both metrics per resolution.

`cmd_eval` now records `mean_combined_rel_L2_vs_fine_reference` beside the
existing per-component field, which is unchanged — so every existing table
stays comparable and the new column can simply be added. Rows resumed from a
pre-change file print `not recorded` rather than being filled in.

**The one live consequence for v38:** the report's B2 zero-shot figure
(0.9986) was selected by the inverted metric. Either it is re-derived from a
run with `--selection_metric both_components`, or the report states the figure
with the selection defect named. The first needs ~1 h 50 m of A100 per arm
(from the measured 450 epochs in 12 m 8 s); the second costs nothing.

~~Still open, and cheap:~~ whether B1's runs *also* early-stopped on a metric
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
