"""DRAFT reply to Timon's round-14 follow-up email ("B3 looks
promising... What parameters would vary in VINO?..."). Covers only
points 1 and 2 of that email (the VINO/Transolver naming, and what
would vary per training sample) -- Omar's own explicit instruction
(2026-09-22): clarify and verify these two before starting ANY dataset-
generation work, and hold off on the resolution/tolerance points until
separately decided.

Point 2's answer was NOT just proposed -- it was checked directly
against the actual geometry code before writing anything down:
`boundary_node_sets` constrains the y=0 plane's own uy to zero
(`constraints[sym, 1] = True`), which is only a VALID symmetry
condition because `rigid_rotation_displacement`'s rotation is
specifically about the y-axis (confirmed in its own docstring: "ux, uz
depend on x0, z0 only; uy stays 0 for every point"). Varying the
rocking AMPLITUDE (phi's magnitude) preserves this exactly -- no
geometry/BC/mesh code changes needed. Varying the rocking AXIS/
DIRECTION would break this symmetry and require the FULL domain
(roughly double the elements) plus a genuinely new, more general
rotation formula -- not something to bundle in casually. This matches
this project's own established B1/B2 convention of varying one load/
field parameter per sample while keeping material fixed (Omar's own
earlier explicit choice for B3: material stays Neo-Hookean).
"""
from docx import Document

OUT = '/home/user/OMAR/advisor_feedback/2026-09-22_reply_to_round14_followup_draft.docx'

doc = Document()


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True


def para(text):
    doc.add_paragraph(text)


note("DRAFT reply to Timon's round-14 follow-up email -- NOT YET SENT. Review before sending.")
note(
    "Covers points 1 and 2 of that email (naming, and what varies per "
    "training sample), plus the geometry figure he asked for (now described "
    "in the email body itself, not just sent as an unexplained attachment). "
    "The resolution/tolerance points (123k/424k 'fairly modest,' 10^5-10^6 "
    "elements, neglecting 1%/2%) are intentionally NOT addressed here -- "
    "Omar's own instruction was to hold those for a separate decision on how "
    "much more GPU time to commit."
)
para("")
para("Subject: Re: B3 follow-up -- operator naming, training-sample design, and geometry figure")
para("")
para("Dear Professor Rabczuk,")
para(
    "Thank you for the feedback -- a few clarifications and the geometry "
    "figure you asked for."
)
para(
    "1. This project's pipeline follows PFEM (\"Pretrain Finite Element "
    "Method,\" Yizheng Wang et al., JMPS 2026) combined with the Transolver "
    "architecture -- that is the operator we actually train and report "
    "results for throughout. VINO was an early prototype tried in the "
    "project's very first week and abandoned; it survives in the repository "
    "only as a vendored, unmodified third-party reference, used once to "
    "cross-check two material-model formulas for correctness. It is not the "
    "architecture behind any of our own trained results. Happy to use "
    "whichever name is clearer going forward -- just wanted to flag the "
    "distinction in case VINO was meant generically."
)
para(
    "2. For the training dataset, the parameter that would vary per sample "
    "is the rocking AMPLITUDE (the size of the core's tilt), with the "
    "rotation axis, material (Neo-Hookean), and geometry all fixed -- the "
    "same pattern as B1/B2's own convention of varying one load/field "
    "parameter per sample. This choice is not arbitrary: the current mesh "
    "exploits a mirror symmetry (only half the cylinder is modeled, with "
    "the y=0 plane's own out-of-plane displacement constrained to zero), "
    "and that symmetry is only valid because the rocking is specifically a "
    "rotation about one fixed axis. Varying the amplitude preserves this "
    "exactly, at no extra cost. Varying the rocking DIRECTION as well is "
    "possible in principle, but would require modeling the full cylinder "
    "(roughly double the mesh) and a more general rotation formula -- a "
    "real increase in scope, not a free addition. We'd suggest amplitude-"
    "only as the default unless you specifically want direction to vary "
    "too, in which case we'd treat it as a separate, explicit extension."
)
para(
    "3. Geometry figure (attached, B3_geometry.png). Three panels: (left) "
    "the undeformed geometry -- the outer housing (fixed, blue) and the "
    "inner core (red) that receives the rocking rotation; (middle) the same "
    "geometry after the rocking rotation is applied, at the same amplitude "
    "(0.05 rad) used throughout the mesh-convergence study -- this shows "
    "the true rigid-body tilt, not a small-angle approximation; (right) a "
    "zoomed meridian cross-section showing the groove -- the one explicit, "
    "smooth, finite-radius feature where the stress concentration is "
    "measured, with its radius of curvature (0.0912, in the same "
    "nondimensional units as the rest of the geometry) marked directly on "
    "the plot."
)
para(
    "We'll hold your resolution/tolerance points (the 123k/424k "
    "discretization, the 10^5-10^6 element target, and neglecting the "
    "1%/2% tolerances) for a separate follow-up once we've worked out how "
    "much additional GPU time makes sense."
)
para("Best regards,")
para("Omar")

doc.save(OUT)
print('Saved', OUT)
check = Document(OUT)
print('paragraphs:', len(check.paragraphs))
