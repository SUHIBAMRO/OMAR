# PFEM / Transolver Project — Status Tracker

**Read this file FIRST at the start of any new conversation about this project.**
It is the single source of truth for where things stand — more reliable than
chat history, which resets between sessions. Update it whenever a task
finishes or a new one starts.

Last updated: 2026-08-15

Report file: `PFEM_Transolver_Report_vNN.docx` (latest: **v16**), kept in the
scratchpad, delivered to the user via SendUserFile after each update — not
committed to this repo.

Advisor: Prof. Timon Rabczuk (Bauhaus-Universität Weimar). Student: Omar Amro.

---

## Advisor's Round-4 feedback — 5 points

| # | Request (short) | Status |
|---|---|---|
| 1 | L2/H1/energy-norm error + convergence rate vs. a ~10M (or 1B) DOF reference; test Q4 vs. Q9; error ≤1e-4 in all norms | 🟡 **partial — see below** |
| 2 | Exact CPU/GPU FEM cost breakdown (assembly/solve/IO, FLOPs, FP64, Newton/CG settings), GPU-native FEM comparison | ✅ done — §8.3–8.5, Tables 7–10 |
| 3 | Batch-size comparison with equal optimizer steps (not equal epochs) | ✅ done — §8.2, Table 6 |
| 4 | Exact mathematical definition of every reported error; investigate poor B2 accuracy | 🟡 **partial — see below** |
| 5 | Resolution invariance = same trained model evaluated on unseen resolutions vs. a common fine reference (not 10 independently-trained networks) | ✅ done — §8.7 |

Round-3 items not repeated in Round 4 (Omar's Aug-3 reply claimed these were
addressed; not re-raised by Timon since):
- OOD evaluation (different material/load ranges) — ✅ confirmed done, §8.6 / Table 11, all 6 cases.
- Allocated-vs-reserved / peak GPU memory clarification — not re-verified this pass.

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
  - A bounded, cheap follow-up (reuses the existing checkpointed reference,
    ~11–13h) extending test points to N=1001, N=1401 was **launched but not
    confirmed finished** — check
    `Google Drive: pfem_run/Q4_B1_neo_hookean_report_extended.json`.
- **B1 × Neo-Hookean, Q9**: 🟡 **in progress**. The ~10M-DOF fine reference
  computation (10 load steps) is very slow — individual CG solves inside a
  single Newton iteration have taken ~10+ hours each. No error/rate numbers
  exist yet (the reference itself isn't done, so nothing downstream of it
  can be computed). Checkpoint: `pfem_ckpt/fine_B1_neo_hookean_Q9_N2236.pt`
  (+ `.cg_state`), resumable — just re-run the same
  `high_dof_convergence_study.py` command and it picks up where it left off.
  Consider re-targeting the reference to a DOF count matched to Q4's (~10M)
  instead of Q4's *N* value (which gives Q9 ~4× more DOF for the same N,
  since Q9 has 9 nodes/element vs. Q4's 4) — this was discussed but not
  yet executed as of this writing.
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
- **B2 × Mooney-Rivlin**: 🟡 in progress. Launched via
  `b2_accuracy_search.py --material mooney_rivlin --target 0.09 --max_trials 3
  --out_dir .../pfem_run/B2_accuracy_search_mooney_rivlin`. Check
  `TRIAL_DONE.json` / `search_summary.json` in that folder for status.
- **B2 × Arruda-Boyce**: 🟡 in progress. Same tool,
  `--material arruda_boyce`, `--out_dir .../pfem_run/B2_accuracy_search_arruda_boyce`.
- Once both finish: propagate the corrected numbers through report **Tables
  5, 7, and 11** (currently still showing the old, uncorrected ~31–32%
  numbers for these two B2 cases) — flagged as a pending NOTE in §9.1 of
  the report itself.

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
