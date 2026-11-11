"""Standalone memo: what's left from the broader roadmap (Items F, G,
H2, and Point 6/H1/I) -- separate from the main re-evaluation report,
since Timon's 2026-11-10 email itself is now fully closed (see that
report's Section 6.5). Same plain house style as the main report.
"""
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor

OUT_DIR = "/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/report"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "Remaining_Roadmap_Items_2026-11-11.docx")

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.left_margin = Inches(1.25)
section.right_margin = Inches(1.25)
section.top_margin = Inches(1.0)
section.bottom_margin = Inches(1.0)

HEADING1_COLOR = RGBColor(0x36, 0x5F, 0x91)
HEADING2_COLOR = RGBColor(0x4F, 0x81, 0xBD)


def title(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)


def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = HEADING1_COLOR


def para(text, italic=False, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)


title("Remaining Roadmap Items (Beyond Timon's 2026-11-10 Email)")
para("Prepared 2026-11-11. Timon's 2026-11-10 email (groove-stress convergence, SCF, "
     "far-field PI-vs-DD comparison) is now fully closed -- see the separate "
     "B1_B3_Independent_Reference_Reevaluation_Report, Section 6.5. This memo covers only "
     "the broader, longer-running roadmap items that are NOT part of that email: Items F "
     "and G (from Timon's 2026-10-20 email) and Item H2 / Point 6 (from his 2026-11-04 "
     "email).")

h1("Item F -- Geometry families + industrial nonlinearities")
para("What Timon asked (2026-10-20): move from a fixed geometry to geometry FAMILIES, with "
     "more industrial nonlinearities.")
para("Status: NOT STARTED as new work -- but two existing, unfinished candidates from an "
     "earlier (September) round already exist in this codebase and were never trained on:")
bullet("B7: a smooth local Gaussian notch cut into the B2 ring geometry. Real CPU evidence "
       "already exists that its peak stress does NOT converge across 4 mesh resolutions "
       "(11.70 -> 12.72 -> 13.45 -> 13.94, still rising) while displacement stays flat -- "
       "exactly the local-stress-concentration signature Timon described. Deliberately "
       "paused pending Timon's confirmation that this is an acceptable instance of his "
       "ask; that confirmation was never requested or received (the relevant email thread "
       "appears to have gone unanswered for about 2 months).")
bullet("B8: a real, published 3D laminated elastomeric seismic bearing (Kalantari & Rofooei "
       "2010 -- cited dimensions, real Neo-Hookean rubber + linear-elastic steel shim "
       "materials). CPU convergence is clean; no GPU production run or Transolver training "
       "has been done yet.")
para("What's left / the decision needed: either (a) resume and finish B7 and/or B8's "
     "FEM-side work and train Transolver on whichever is ready -- reusing real existing "
     "work rather than starting over, or (b) treat this as wanting something different "
     "and design a fresh benchmark, or (c) simply ask Timon whether B7/B8 already satisfy "
     "this item before doing anything else. Not yet decided.")

h1("Item G -- AMG-Krylov / multi-GPU faster iterative solvers")
para("What Timon asked (2026-10-20): faster iterative solvers via AMG-Krylov with "
     "multi-GPU/domain decomposition.")
para("AMG half -- tested, real negative result. A direct CPU benchmark on this project's "
     "actual B3 problem (same solve code, only the preconditioner changed) showed Jacobi "
     "beating AMG at every tested size (1,320 to 18,200 elements), with the gap GROWING "
     "with size -- the opposite of the naive expectation. Most likely explanation: this is "
     "a geometrically nonlinear (Neo-Hookean, multi-increment Newton) problem, so AMG must "
     "rebuild its multigrid hierarchy from scratch at every Newton iteration, a real, "
     "repeated cost Jacobi does not pay. This may not hold at GPU scale (AmgX's own cost "
     "model differs), but testing that needs building NVIDIA AmgX from source, not done.")
para("Multi-GPU/domain-decomposition half -- real mechanism built and CPU-verified, one "
     "real bug found and fixed, NOT YET CONFIRMED on actual GPU hardware. A genuine "
     "domain-decomposed distributed CG (partitioned storage, neighbor halo exchange, "
     "all_reduce for global dot products) was built and verified correct on CPU (gloo "
     "backend) across several process counts, agreeing with a reference solve to ~1e-15. "
     "Ported to real GPUs (NCCL backend): the first attempt hung for 13+ minutes (confirmed "
     "genuinely stuck, not just slow, via CPU-time-vs-wall-clock tracking) -- root cause was "
     "a known NCCL deadlock risk in using separate send/recv calls instead of the batched "
     "point-to-point API; fixed, and the CPU correctness suite re-verified unchanged after "
     "the fix. The fixed version has NOT yet been re-submitted and confirmed working on "
     "real GPU hardware -- this is the single concrete next action, needs an HPC job.")
para("What's left: (1) confirm the NCCL fix actually works on real GPUs (next concrete "
     "step, needs Omar's HPC); (2) a graph-partitioned (not simple slab) decomposition of a "
     "real unstructured FEM mesh; (3) swap the toy Laplacian test problem for this "
     "project's own real B1/B3 tangent stiffness. None of (2)/(3) are done -- this remains "
     "a proven mechanism, not a finished multi-GPU solver.")

h1("Item H2 -- Goal-oriented a-posteriori error estimation")
para("What Timon asked (2026-11-04, explicitly 'a more realistic second direction' toward "
     "trustworthy-AI certification): use the PI solution as the primal approximation, and "
     "build QoI-specific error estimators from the governing residual and an adjoint "
     "solution (adjoint computed via FEM first, later compared against DD/hybrid adjoint "
     "approximations).")
para("Status: NOT STARTED. This builds directly on the resolution-theory/model-selection "
     "understanding from Point 5 (already done) -- a natural next step once Item F and/or "
     "G are closed out, but no design or code exists yet.")

h1("Point 6 / Items H1 and I -- the long-term trustworthy-AI vision")
para("Timon's 2026-11-04 email was explicit that this is 'not an action item... for "
     "discussion only' -- the two long-term directions are H1 (a-priori error bounds, "
     "'extremely challenging' in his own words) and I (a large multi-fidelity database "
     "built via active learning / multi-fidelity Bayesian optimization / meta-learning / "
     "eventually RL, 'could form a substantial PhD direction'). Nothing to execute here "
     "yet -- flagged only so it stays on the radar for a future discussion with Timon, not "
     "because it is overdue.")

h1("Suggested order")
para("Item G is closest to a clean close (one HPC job: confirm the NCCL fix on real GPU "
     "hardware). Item F needs a decision from Omar (or a quick question to Timon) before "
     "any new compute is spent, since real, reusable work already exists for it. Item H2 "
     "is a genuinely new undertaking, best started once F and/or G are settled. Point 6 "
     "stays a discussion topic for the next direct conversation with Timon.")

doc.save(OUT_PATH)
print(f"Wrote {OUT_PATH}")
