"""DRAFT reply to Timon's round-13 email (items 1-4), asking him to
choose between the two prepared 3D candidates before dataset generation,
training, and the FEM-vs-operator comparison begin. Omar's own explicit
instruction (2026-09-21): the candidate must be chosen WITH Timon first;
nothing downstream (dataset design, training, comparison) starts before
his answer.

REVISED 2026-09-21c: Omar's own second, line-by-line review of the
first draft caught that this email still described the Cauchy-tensor
field-error metric's earlier ~17% "plateau" as a settled finding, when
it was actually a methodology bug (an asymmetric comparison) found and
fixed after that draft was written -- see
add_round14_b3_tire_candidates_to_report.py's own module docstring for
the full seven-point list. Paragraph 4(A) below is rewritten around the
corrected, CPU-validated result instead, and a sentence is added making
explicit that the FEM-vs-operator comparison itself has not started for
either candidate, so Timon does not read the 3D item as further along
than it actually is.
"""
from docx import Document

OUT = '/home/user/OMAR/advisor_feedback/2026-09-21c_reply_to_round13_candidate_choice_draft.docx'

doc = Document()


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True


def para(text):
    doc.add_paragraph(text)


note("DRAFT reply to Timon's round-13 email -- NOT YET SENT. Review before sending.")
note(
    "Items 1-3 are complete and reported in the attached Report/Summary. Item 4's "
    "mesh-convergence half is also complete for the rubber-mount bushing (B3). What "
    "is genuinely NOT done yet is the dataset generation, operator training, and "
    "FEM-vs-operator comparison item 4 also asks for -- that is gated on Timon "
    "confirming which geometry to proceed with, which this draft asks directly "
    "rather than assuming."
)
para("")
para("Subject: Round-13 items 1-3 complete; item 4 -- two candidates prepared, your call on which one to take forward")
para("")
para("Dear Professor Rabczuk,")
para(
    "Items 1-3 from your last email are complete; the attached Report and Summary "
    "cover them in full."
)
para(
    "1. The accuracy-vs-N comparison now uses only the lower-N range (N=1401 "
    "dropped), the corrected final checkpoints, and a common reference across all "
    "six cases. The new table shows, for 1%/2%/5% accuracy thresholds, the required "
    "FEM resolution and the operator's break-even point, separately for "
    "displacement L2, H1/tangent energy, reaction force, and the fixed-region "
    "Cauchy stress -- so the practical comparison is shown explicitly to depend on "
    "which accuracy level and which QoI are actually required, as you asked."
)
para(
    "2. Our own GPU-native finite-element solver is now named explicitly as the "
    "primary timing baseline everywhere it has been measured; torch-fem/TensorMesh "
    "remain secondary comparisons, and the same-N comparison is kept as its own "
    "separate table, since it answers a different question from the accuracy-"
    "matched one."
)
para("3. IGA/NURBS is dropped from this paper and noted as future work only.")
para(
    "4. For the harder 3D example, we prepared two candidates in parallel so you "
    "could see both before we commit real GPU time to one:"
)
para(
    "(A) A rocking elastomeric rubber-mount bushing with a smooth, finite-radius "
    "stress concentration -- essentially the geometry you suggested. This one has "
    "been taken to full rigor already: mesh convergence is confirmed on a real "
    "A100 run (the fixed-region Cauchy stress -- the hardest QoI here -- changes by "
    "only 0.506% between a 243k- and a 424k-element reference, within the intended "
    "0.5-0.7% convergence band), and a full required-resolution table exists for "
    "displacement, H1, energy, reaction force/moment, and the regional Cauchy "
    "stress. A stricter, full-tensor pointwise version of the Cauchy-stress field "
    "error initially appeared to plateau around 17%; on closer review this turned "
    "out to be a real bug in how the comparison was built (comparing two different "
    "representations of the field, not a fair test), now fixed and re-tested -- the "
    "corrected version shows clean, physically sensible convergence at CPU scale, "
    "and we're re-confirming it at the same GPU reference scale as the average "
    "statistic before finalizing the number."
)
para(
    "(B) A tire sector -- a real torus segment, not an extrusion -- prepared only "
    "as a lightweight preliminary alternative (geometry, boundary conditions, a "
    "real solve, and a basic convergence check), since you had mentioned it "
    "separately as a possible harder problem. It is not brought to the bushing's "
    "own level of rigor unless you'd prefer it."
)
para(
    "Both are described in full in the attached Summary. To be clear about where "
    "this stands: only the geometry, mesh convergence, and a real solve are done "
    "for either candidate -- no dataset has been generated, no operator trained, "
    "and no FEM-vs-operator comparison run yet. Before we start that (dataset "
    "generation, operator training, and the displacement/reaction/energy/"
    "regional-Cauchy-stress comparison item 4 asks for), could you confirm which "
    "of the two you'd like us to take forward -- the bushing, the tire sector, or "
    "something else you have in mind?"
)
para(
    "Once you confirm, we'll generate the dataset, train the operator, and report "
    "back with that same comparison used for the other two cases -- and then start "
    "writing the paper."
)
para("Best regards,")
para("Omar")

doc.save(OUT)
print('Saved', OUT)
check = Document(OUT)
print('paragraphs:', len(check.paragraphs))
