"""Companion to add_resolution_matched_breakeven_figure_to_report.py --
same figure, added right after Table R10-4''s own footnote in the
Summary."""
import os

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

FIG = '/home/user/OMAR/Practical_Examples/report_builders/figures'
DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17b.docx')

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


anchor = find_para_exact(
    "Table R10-4'. Resolution-matched break-even, both methods at N=1401, all six "
    "cases. At matched resolution the operator wins by 57-89x even unoptimized -- "
    "unlike Table R10-4's own accuracy-matched comparison, where the default mode "
    "does not break even at all. All four non-Arruda-Boyce numbers are from a "
    "single clean re-run (2026-09-17) after fixing a real memory-cleanup bug in "
    "the sweep script (an exception-chain reference kept a failed case's own GPU "
    "tensors pinned into the next case) -- this re-run succeeded for every "
    "non-Arruda-Boyce case in one pass, confirming the fix on real GPU. Both "
    "Arruda-Boyce cases fail genuinely and consistently across every run: "
    "torch-fem's own Newton-Raphson solve does not converge at N=1401, "
    "root-caused to a real CUDA OOM inside torch-fem's own Hessian assembly for "
    "this specific material."
)

insert_figure_after(anchor,
    os.path.join(FIG, 'fig_resolution_matched_breakeven.png'),
    "Figure R10-D. Resolution-matched speedup, all six cases (Table R10-4'). "
    "(Figure R10-C already exists for Table R10-3's own data.)")

doc.save(DST)
print('Saved', DST)
