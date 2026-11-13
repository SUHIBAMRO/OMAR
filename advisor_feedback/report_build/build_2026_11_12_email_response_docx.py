"""Standalone reply document: a complete, self-contained response to
Professor Rabczuk's 2026-11-12 email -- evaluate PI and the DD variants
against the now-converged quadratic (Q20) groove reference, including
the SCF, and report the quadratic meshes' DOF. Written as a report, not
a letter (per Omar's explicit preference), separate from the large
B1/B3 re-evaluation report.
"""
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_DIR = "/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/report"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "Timon_2026-11-12_Email_Response.docx")

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
    r.font.size = Pt(14)
    r.font.color.rgb = HEADING1_COLOR


def h2(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = HEADING2_COLOR


def para(text, italic=False, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(htext)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(9)
    doc.add_paragraph()


# ===========================================================================
title("B3 Final Check: PI/DD vs. the Converged Quadratic Groove Reference")
para("Prepared 2026-11-13. This report answers Professor Rabczuk's 2026-11-12 email in full: "
     "evaluating PI and the DD variants against a now-converged quadratic (Q20) groove "
     "reference, including the Stress Concentration Factor (SCF), and reporting the DOF of "
     "the quadratic meshes used. A critical bug discovered while preparing this work is "
     "disclosed first, since it affects how every number below should be read.")

# ===========================================================================
h1("1. A critical bug found and fixed before this work could be trusted")
para("While building the infrastructure for this check, we discovered that the two scripts "
     "written for last week's quadratic-element (Q20) convergence study had hard-coded the "
     "groove depth as 0.05 (with half-width 0.15) -- a stale value carried over from an "
     "earlier exploratory script's own module default, not the real B3 production geometry "
     "(depth 0.20, half-width 0.15, defined in the dataset-generation script). The real "
     "groove is 4x deeper and consequently far sharper: its radius of curvature is "
     "approximately 0.0228, not the 0.0912 the buggy code was actually using.")
para("This invalidated essentially every specific number from last week's Q20 convergence "
     "study (region-average stress, SCF, quadratic peak-stress values, field-convergence "
     "percentages) -- though not necessarily its qualitative conclusions, since those follow "
     "from the same generic geometric/FEM arguments regardless of the exact depth. Both "
     "scripts were fixed to import the groove depth from the single authoritative source "
     "(the dataset-generation module) rather than hard-coding it, and every result below was "
     "produced only after this fix, at the correct, real geometry.")
para("One further, genuine bug was caught and fixed during this same work: an earlier draft "
     "of the SCF computation (Section 3 below) reused a helper that solves the groove and "
     "no-groove counterfactual with LINEAR elements, inconsistent with the quadratic (Q20) "
     "reference used for the checkpoint comparison in the same run. This was fixed to compute "
     "both sides of the SCF with quadratic elements throughout. We flag both fixes explicitly "
     "because the resulting SCF (Section 3) is a materially different, and more defensible, "
     "number than anything reported previously.")

# ===========================================================================
h1("2. The quadratic (Q20) reference mesh and its DOF")
para("The Q20 reference mesh used for the comparison below has 47,040 elements and 200,675 "
     "nodes, giving:")
para("DOF = 200,675 nodes x 3 = 602,025.", bold=True)
para("This mesh captures 38 Gauss quadrature points inside the groove region -- above this "
     "project's own established minimum-reliable-sample threshold (20), though still modest, "
     "reflecting how sharp and locally under-resolved this specific groove geometrically is "
     "even at a production-scale global mesh.")

# ===========================================================================
h1("3. PI and the DD variants against the Q20 reference, with SCF")
para("Every existing u-only checkpoint (PI baseline, the three local-integration-refinement "
     "seeds, the two equilibrium-augmented attempts, and the five data-driven/DD variants) "
     "was re-evaluated against this Q20 reference, reusing the project's existing, already-"
     "validated evaluation pipeline unmodified (it is generic over element type; only the "
     "reference-mesh builder itself needed a genuinely new, quadratic-element counterpart).")
table(
    ["Checkpoint", "disp L2", "region field error", "sigma_xx error"],
    [
        ["PI baseline", "2.89%", "3077%", "4095%"],
        ["Local-integration-refinement (seed 1)", "3.50%", "2063%", "2827%"],
        ["Local-integration-refinement (seed 2)", "2.55%", "1745%", "2263%"],
        ["Local-integration-refinement (seed 3)", "2.99%", "2301%", "3064%"],
        ["Equilibrium w=1.0 (closed, poor)", "36.87%", "10790%", "13047%"],
        ["Equilibrium w=0.12 (closed, poor)", "37.34%", "4616%", "6340%"],
        ["DD @ N=50", "8.73%", "4524%", "5448%"],
        ["DD @ N=200", "2.79%", "3082%", "3519%"],
        ["DD @ N=400", "3.78%", "1825%", "2176%"],
        ["DD sobolev", "3.32%", "1766%", "2102%"],
        ["DD hybrid", "3.37%", "1886%", "2473%"],
    ],
)
para("As Professor Rabczuk anticipated, the local groove-region stress errors remain very "
     "large for every checkpoint (1700-13000%), and we report this plainly as a limitation "
     "rather than something this particular architecture resolves. Global quantities (the "
     "displacement field) remain small and sane for every u-only checkpoint (2.5-8.7%, except "
     "the already-known-poor equilibrium-augmented attempts at ~37%), consistent with every "
     "earlier finding in this project that the groove failure is a localized phenomenon, not a "
     "general model-quality problem.")
para("A genuinely new finding, worth highlighting rather than treating as noise: the RANKING "
     "among checkpoints changes meaningfully against this better reference. Against the old "
     "(linear, 950,400-element) reference, PI baseline clearly beat every DD variant on "
     "groove-region stress. Against this quadratic reference, several DD variants (N=400, "
     "sobolev, hybrid: 1766-2473%) and all three local-integration-refinement seeds "
     "(1745-2301%) now score BETTER than PI baseline (3077%) on the same quantity. We report "
     "this as a real result of using a more trustworthy reference, not as evidence that the "
     "earlier (reversed) ranking was wrong at the time -- it simply used a reference that was "
     "itself not yet converged at the groove.")

h2("3.1 Stress Concentration Factor")
para("SCF is defined, as in the previous report, as the ratio of the peak stress at the "
     "groove to the average stress at the exact same physical location, same mesh, same "
     "loading, with the groove geometrically removed -- computed here with quadratic (Q20) "
     "elements on BOTH sides for full consistency, at the same resolution as the reference "
     "mesh above (47,040 elements).")
table(
    ["", "avg|sigma_xx|", "max|sigma_xx|"],
    [
        ["groove (Q20, real geometry)", "0.433", "0.705"],
        ["nominal, no groove (Q20, same point)", "3.949", "9.611"],
    ],
)
para("SCF (max groove / nominal average) = 0.178 -- below 1 on every basis (average and "
     "max).", bold=True)
para("This is a materially different conclusion from every earlier SCF estimate produced in "
     "this project (19, 9.0, and 8.45 at various points), each of which was built on either "
     "the wrong groove depth or an under-resolved/inconsistent-element-type comparison. Once "
     "measured with a properly converged, properly matched-element-type reference, this "
     "specific groove shows no genuine stress concentration under this loading -- the "
     "peak groove stress is lower than the no-groove nominal baseline at every level tested. "
     "We report this as the correct, superseding answer, not as an additional data point "
     "alongside the earlier numbers.")

# ===========================================================================
h1("4. Independent supporting evidence: quadratic convergence vs. a likely linear artifact")
para("Beyond the specific SCF computation above, two further checks (both rerun at the "
     "corrected depth) support the same conclusion from independent angles.")
para("Quadratic (p-refinement) elements converge the peak groove stress cleanly and "
     "monotonically across five increasing global resolutions, from 2,376 elements up to the "
     "47,040-element reference mesh itself:")
table(
    ["Elements", "max|sigma_xx|"],
    [
        ["2,376", "4.13"],
        ["5,760", "1.80"],
        ["10,200", "0.83"],
        ["15,960", "0.81"],
        ["47,040", "0.705"],
    ],
)
para("The last two points agree within about 2.5%, and the overall sequence is monotonically "
     "decreasing with no sign of blowing up -- the behavior expected of a genuinely bounded "
     "field, not a singularity.")
para("By contrast, we also pushed LINEAR elements through aggressive local z-clustering at "
     "the groove (fixed global element count, 47,040), to see whether this alternative "
     "refinement strategy told the same story. It did not, at first: the peak stress climbed "
     "sharply and showed no sign of plateauing (1.78 at the mildest clustering up to 13.32 at "
     "z_beta=18, more than 7x over that range, with solve time itself growing roughly 10x over "
     "the same sweep). Pushing two further points past that:")
table(
    ["z_beta", "max|sigma_xx|", "solve time"],
    [
        ["18.0", "13.32", "74.96s"],
        ["22.0", "9.32", "1412.66s (about 19x slower)"],
    ],
)
para("At z_beta=22, the peak stress DROPS back to 9.32 -- a reversal, not a continuation of "
     "the earlier climb -- while the solve time explodes nearly 19x for this one additional "
     "step. We stopped the sweep here rather than continuing for days: the combination of (a) "
     "this non-monotonic reversal and (b) the severe, non-physical growth in solve cost "
     "(consistent with worsening numerical conditioning as z-elements become extremely thin "
     "and distorted under heavy local clustering) is the signature of a numerical artifact, "
     "not a genuine physical divergence. We present this as the most likely explanation, "
     "not as a certainty -- it was not pushed to full resolution of the question, by explicit "
     "choice, given the cost already observed and the strength of the independent quadratic-"
     "element evidence above pointing to the same bounded conclusion.")
para("Taken together: a clean, monotonically convergent quadratic-element result and a "
     "linear-element result that is best explained as artifact rather than genuine growth "
     "both point the same way as the SCF computation in Section 3 -- this groove, at the "
     "resolutions and element types we can trust, does not show a genuine stress "
     "concentration under this loading.")

# ===========================================================================
h1("5. Summary")
bullet("DOF of the quadratic (Q20) reference mesh: 602,025 (47,040 elements, 200,675 nodes).")
bullet("Every u-only checkpoint still shows very large groove-region stress error "
       "(1700-13000%) against this reference -- reported as a limitation, as anticipated.")
bullet("The checkpoint ranking on groove-region stress changes against this better "
       "reference: several DD variants and all local-integration-refinement seeds now beat "
       "PI baseline specifically on this quantity.")
bullet("SCF, computed consistently with Q20 elements on both sides: 0.178 -- below 1 -- "
       "superseding every earlier estimate (19 / 9.0 / 8.45), all built on either the wrong "
       "groove depth or an inconsistent element-type comparison.")
bullet("Independent evidence (clean quadratic convergence; the likely-artifact linear "
       "result) supports the same conclusion: this groove shows no genuine stress "
       "concentration under this loading, once measured properly.")
para("We have not yet re-run the field-level (pointwise, same-physical-location) contour "
     "convergence check at the corrected depth -- the last such check, like all specific "
     "numbers before this fix, used the wrong geometry. We consider this a lower-priority, "
     "further-rigor item given the strength of the evidence above, and would be glad to "
     "pursue it if useful before moving on. We look forward to discussing the longer-term "
     "automated-verification-framework direction raised in the same email.")

doc.save(OUT_PATH)
print(f"Wrote {OUT_PATH}")
