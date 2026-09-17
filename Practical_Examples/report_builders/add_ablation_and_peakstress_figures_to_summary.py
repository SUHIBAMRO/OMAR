"""Companion to add_ablation_and_peakstress_figures_to_report.py -- same
two figures, added right after their own table captions in the
Summary."""
import os

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

FIG = '/home/user/OMAR/Practical_Examples/report_builders/figures'
DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17b.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17c.docx')

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def insert_figure_after(anchor_para, image_path, caption_text, width_in=5.5):
    img_p = doc.add_paragraph()
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_p.add_run().add_picture(image_path, width=Inches(width_in))
    img_elem = img_p._p
    img_elem.getparent().remove(img_elem)

    cap_p = doc.add_paragraph()
    cap_run = cap_p.add_run(caption_text)
    cap_run.italic = True
    cap_run.font.size = Pt(9)
    cap_elem = cap_p._p
    cap_elem.getparent().remove(cap_elem)

    anchor_para._p.addnext(cap_elem)
    anchor_para._p.addnext(img_elem)
    return Paragraph(cap_elem, anchor_para._parent)


anchor_ablation = find_para_exact(
    "Table R10-5. Direct-N1401 training ablation vs. multi-resolution zero-shot, "
    "B1 x Neo-Hookean."
)
insert_figure_after(anchor_ablation,
    os.path.join(FIG, 'fig_direct_n1401_ablation.png'),
    "Figure R10-E. Direct-N1401 training vs. multi-resolution zero-shot, "
    "training cost and accuracy (Table R10-5).")

anchor_peak = find_para_exact(
    "Table R10-6. Peak-stress QoI at N=1401, five cases. B1's own two rows "
    "(78.05%, 78.25%) are structurally worse than every B2 case "
    "(39.97%-49.48%), consistent with B1's own boundary-corner stress-"
    "singularity caveat noted above. B2xArruda-Boyce (39.97%) has the BEST "
    "peak-stress accuracy of all six cases despite failing the break-even "
    "above -- unrelated findings: that failure is torch-fem's own reference "
    "solve running out of memory, not a property of the operator's own "
    "accuracy."
)
insert_figure_after(anchor_peak,
    os.path.join(FIG, 'fig_peak_stress_qoi.png'),
    "Figure R10-F. Peak-stress QoI vs. displacement error, five cases "
    "(Table R10-6).")

doc.save(DST)
print('Saved', DST)
