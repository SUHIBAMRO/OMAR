"""Companion to add_resolution_matched_breakeven_to_report.py -- same
new content (resolution-matched break-even, operator vs. torch-fem both
AT N=1401, all six cases), condensed to the Summary's own terser style.
See that script's own docstring for the full rationale, exact number
provenance, and the B2xNeo-Hookean memory-cleanup caveat; not repeated
in full here.

Naming note: the Summary's own existing Table R10-4 caption (paragraph
75) cross-references "the earlier matched-resolution break-even (Point
3 above)" -- that is the Section-8.4-style matched-BATCH-SIZE break-even
from Point 2/3's own throughput material, a DIFFERENT comparison from
the one added here. The new table below is explicitly named and
described so it cannot be confused with either the existing Table R10-4
(accuracy-matched, operator@N=1401 vs FEM@N=11) or that earlier
cross-reference.

Inserted after paragraph 76 (Table R10-4's own interpretation
paragraph) and before "Summary of what was done and what came out" --
the natural continuation point."""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-16.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-16b.docx')

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
    "This is the one place in the whole round-10 investigation where the operator's "
    "default (unoptimized) mode does not come out ahead: run naively, it never "
    "repays its own training cost against a FEM mesh that is only as accurate as it "
    "needs to be. With the point-3 optimizations (torch.compile + TF32), it both "
    "wins per-sample (4.13x) and repays its training cost cheaply -- about 3.72 "
    "GPU-hours against 11.6 GPU-hours of training. The operator's economic case "
    "depends on deploying it with these optimizations, not on its default settings "
    "-- reported honestly rather than only the favorable case."
)

p1 = insert_after(anchor,
    "The advisor's round-11 feedback asked to keep a SECOND break-even comparison "
    "too: both methods at the SAME resolution, N=1401, rather than each at its own "
    "accuracy-matched mesh -- expected, correctly, to look much more favourable to "
    "the operator. Table R10-4', all six cases, operator's own default eager fp32 "
    "forward pass at N=1401 vs. torch-fem's own real GPU solve time at the same N.")

header = ['Case', 'torch-fem @N=1401', 'Operator @N=1401', 'Speedup', 'Break-even']
t = insert_table_after(p1, header, [
    ['B1 x Neo-Hookean', '135.18 s', '2292.1 ms', '59.0x', '315 samples'],
    ['B1 x Mooney-Rivlin', '134.16 s', '2347.5 ms', '57.2x', 'training cost unknown'],
    ['B1 x Arruda-Boyce', 'FAILED (see note)', '-', '-', '-'],
    ['B2 x Neo-Hookean', '205.92 s *', '2353.5 ms *', '87.5x *', 'training cost unknown'],
    ['B2 x Mooney-Rivlin', '207.57 s', '2338.7 ms', '88.8x', 'training cost unknown'],
    ['B2 x Arruda-Boyce', 'FAILED (see note)', '-', '-', '-'],
])

p2 = insert_after_table(t, p1,
    "Table R10-4'. Resolution-matched break-even, both methods at N=1401, all six "
    "cases. At matched resolution the operator wins by 57-89x even unoptimized -- "
    "unlike Table R10-4's own accuracy-matched comparison, where the default mode "
    "does not break even at all. * B2 x Neo-Hookean reuses an earlier clean "
    "measurement: the freshest re-run's own attempt at this case failed with a CUDA "
    "OOM caused by leftover GPU memory from the immediately preceding B1 x "
    "Arruda-Boyce failure (81.27 GB already allocated on a 79.25 GB device before "
    "this case even started) -- a memory-cleanup gap between cases in the sweep "
    "script, not a real finding about this case. Both Arruda-Boyce cases fail "
    "genuinely and consistently across every run: torch-fem's own Newton-Raphson "
    "solve does not converge at N=1401, root-caused to a real CUDA OOM inside "
    "torch-fem's own Hessian assembly for this specific material.")

doc.save(DST)
print('Saved', DST)
