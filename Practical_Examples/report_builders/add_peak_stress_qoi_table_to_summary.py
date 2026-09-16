"""Companion to add_peak_stress_qoi_table_to_report.py -- same content
(Task #22 peak-stress QoI, the five cases beyond B1xNeo-Hookean),
condensed to the Summary's own terser style. See that script's own
docstring for the full checkpoint-version caveat and rationale."""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-16c.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-16d.docx')

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def insert_after(anchor, text):
    new_p_el = copy.deepcopy(anchor._p)
    anchor._p.addnext(new_p_el)
    np = Paragraph(new_p_el, anchor._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


def insert_table_after(anchor_para, header, rows):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = doc.tables[0].style
    pr = doc.tables[0]._tbl.find(qn('w:tblPr'))
    if pr is not None:
        old = tbl._tbl.find(qn('w:tblPr'))
        if old is not None:
            tbl._tbl.remove(old)
        tbl._tbl.insert(0, copy.deepcopy(pr))
    for j, h in enumerate(header):
        c = tbl.cell(0, j)
        c.text = ''
        r = c.paragraphs[0].add_run(h)
        r.bold = True
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            tbl.cell(i, j).text = str(v)
    tbl._tbl.getparent().remove(tbl._tbl)
    anchor_para._p.addnext(tbl._tbl)
    return tbl


def insert_after_table(table, style_para, text):
    new_p_el = copy.deepcopy(style_para._p)
    table._tbl.addnext(new_p_el)
    np = Paragraph(new_p_el, style_para._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


anchor = find_para_exact(
    "Table R10-4'. Resolution-matched break-even, both methods at N=1401, all six "
    "cases. At matched resolution the operator wins by 57-89x even unoptimized -- "
    "unlike Table R10-4's own accuracy-matched comparison, where the default mode "
    "does not break even at all. * B2 x Neo-Hookean reuses an earlier clean "
    "measurement: the freshest re-run's own attempt at this case failed with a "
    "CUDA OOM caused by leftover GPU memory from the immediately preceding B1 x "
    "Arruda-Boyce failure (81.27 GB already allocated on a 79.25 GB device before "
    "this case even started) -- a memory-cleanup gap between cases in the sweep "
    "script, not a real finding about this case. Both Arruda-Boyce cases fail "
    "genuinely and consistently across every run: torch-fem's own Newton-Raphson "
    "solve does not converge at N=1401, root-caused to a real CUDA OOM inside "
    "torch-fem's own Hessian assembly for this specific material."
)

p1 = insert_after(anchor,
    "Task #22's peak-stress QoI, the five cases beyond B1xNeo-Hookean (Table "
    "R10-6). Caveat: B1xMooney-Rivlin, B1xArruda-Boyce and B2xMooney-Rivlin here "
    "reflect their ORIGINAL pre-retrain checkpoints (their disp_rel_L2 matches the "
    "\"OLD\" column in Table R10-2 exactly) -- the peak-stress metric has not been "
    "recomputed against this week's new retrained checkpoints for those three. "
    "B2xNeo-Hookean and B2xArruda-Boyce have only one checkpoint each, no caveat.")

header = ['Case', 'disp_rel_L2 @N1401', 'peak_stress_rel_err @N1401']
t = insert_table_after(p1, header, [
    ['B1 x Mooney-Rivlin', '39.20%', '78.05% (orig. checkpoint)'],
    ['B1 x Arruda-Boyce', '45.62%', '78.25% (orig. checkpoint)'],
    ['B2 x Neo-Hookean', '46.37%', '49.48%'],
    ['B2 x Mooney-Rivlin', '49.47%', '48.13% (orig. checkpoint)'],
    ['B2 x Arruda-Boyce', '39.73%', '39.97%'],
])
p2 = insert_after_table(t, p1,
    "Table R10-6. Peak-stress QoI at N=1401, five cases. B1's own two rows "
    "(78.05%, 78.25%) are structurally worse than every B2 case (39.97%-49.48%), "
    "consistent with B1's own boundary-corner stress-singularity caveat noted "
    "above. B2xArruda-Boyce (39.97%) has the BEST peak-stress accuracy of all six "
    "cases despite failing the break-even above -- unrelated findings: that "
    "failure is torch-fem's own reference solve running out of memory, not a "
    "property of the operator's own accuracy.")

doc.save(DST)
print('Saved', DST)
