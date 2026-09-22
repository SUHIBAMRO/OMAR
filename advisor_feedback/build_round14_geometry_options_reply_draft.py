"""DRAFT reply to Timon's latest email (confirms he meant the physics-
informed Transolver; says he's not sure what geometry would need
10^5-10^6 elements for 5-10% QoI error, and is "quite open"). Presents
two candidate directions, per Omar's own decision (2026-09-22): a
modified (sharper-groove) B3, and a new laminated annular elastomeric
seismic bearing -- and asks Timon to pick, rather than guessing and
committing scope to one without confirming first.
"""
from docx import Document

OUT = '/home/user/OMAR/advisor_feedback/2026-09-22b_reply_to_round14_geometry_options_draft.docx'

doc = Document()


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True


def para(text):
    doc.add_paragraph(text)


note("DRAFT reply to Timon's latest email -- NOT YET SENT. Review before sending.")
note(
    "Presents two candidate directions for the 10^5-10^6-element, 5-10%-error "
    "target he described, and asks him to choose -- same pattern as the "
    "earlier B3-vs-tire choice, since committing real design/GPU time to one "
    "without confirming first has already cost a round-trip once this "
    "project."
)
para("")
para("Subject: Re: B3 follow-up -- two directions for a genuinely 10^5-10^6-element problem")
para("")
para("Dear Professor Rabczuk,")
para(
    "Good to confirm the physics-informed Transolver is what we've been "
    "training and reporting all along."
)
para(
    "On the geometry: the current B3 bushing converges faster than that "
    "target -- its region-Cauchy-stress error is already down to about 1.3% "
    "at 123,000 elements, well inside your 5-10% band rather than still "
    "sitting in it at 10^5-10^6 elements. That's a direct consequence of an "
    "earlier design choice: B3's groove was deliberately made smooth (no "
    "sharp corner) specifically to avoid a true stress singularity. A "
    "genuinely harder problem in your target range needs a sharper local "
    "feature or a more demanding geometry. Two directions:"
)
para(
    "(A) A modified B3: same rocking bushing, with the groove made "
    "noticeably sharper (a smaller radius of curvature) -- still smooth, "
    "no singularity, but a stronger local stress gradient that should push "
    "the required mesh resolution up substantially. This reuses the "
    "existing, already-validated geometry and code; we can test cheaply on "
    "CPU whether it actually lands in your target range before committing "
    "any GPU time."
)
para(
    "(B) A new geometry: a 3D laminated annular elastomeric seismic "
    "bearing -- thin rubber layers between internal shim plates, a central "
    "hole, under combined compression and shear/rocking. This is a "
    "realistic, industrially standard isolator design, and its thin, "
    "near-incompressible layers with free edges at each interface are a "
    "much more demanding target for FEM resolution than a single smooth "
    "feature. To keep the single-material scope we've used throughout, we "
    "would model the shim plates as rigid (a boundary condition, not a "
    "second deformable material) and let only the rubber layers deform -- "
    "this is a new geometry, not an extension of B3, and a substantially "
    "bigger design effort than (A)."
)
para(
    "(A) is quick for us to test and report back on; (B) is a real, "
    "worthwhile commitment if you'd rather have the more realistic "
    "geometry. Could you let us know which direction you'd like us to take, "
    "or if you'd like to see (A)'s numbers first before deciding?"
)
para("Best regards,")
para("Omar")

doc.save(OUT)
print('Saved', OUT)
check = Document(OUT)
print('paragraphs:', len(check.paragraphs))
