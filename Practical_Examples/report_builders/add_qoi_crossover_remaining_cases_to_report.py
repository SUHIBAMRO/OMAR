"""Answers the advisor's round-11 point 2 directly from real GPU data:
does the same QoI/norm determine the coarsest-suitable-FEM crossover
for every case, or does it vary? Already answered for B1xNeo-Hookean
(N=11, bound by the tangent-energy norm, Table 18-R10e's own context
paragraph above). This adds the same analysis for the five remaining
cases (Round11_QoI_Crossover_RemainingCases.ipynb, real GPU run
2026-09-17, second attempt after fixing a real bug in the crossover's
own NO-accuracy-JSON indexing -- see PROJECT_STATUS.md same day).

CHECKPOINT-VERSION CAVEAT, same as Table 18-R10k above: this crossover
reuses each case's own NO accuracy sweep from task #22, which predates
all four multi-resolution retrains -- B1xMooney-Rivlin, B1xArruda-Boyce,
B2xMooney-Rivlin and B2xNeo-Hookean are therefore measured here against
their ORIGINAL (pre-retrain) checkpoints, not the newer, more accurate
ones now in the Report. B2xArruda-Boyce has only ever had one
checkpoint, so no such caveat applies to that row.

Inserted right after the peak-stress QoI interpretation paragraph
(Table 18-R10k's own discussion), the natural continuation of the same
"extend B1xNeo-Hookean's analysis to the other five cases" narrative,
and before the document moves on to the older round-5 Pareto-sweep
material (Table 18/18a/18b).
"""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-17d.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-17e.docx')

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
    "Two patterns are worth stating plainly rather than left for the reader to "
    "notice. First, B1's own two cases here (78.05%, 78.25%) are structurally "
    "worse on this metric than every B2 case (39.97%-49.48%) -- consistent with "
    "the B1xNeo-Hookean finding above that the located peak-stress point sits near "
    "a boundary-condition-transition corner prone to a stress singularity, a "
    "property of the B1 geometry's own boundary conditions rather than of any "
    "particular material. Second, B2xArruda-Boyce (39.97%) has the BEST "
    "peak-stress accuracy of all six cases in this report, despite being one of "
    "the two cases that fails outright on the resolution-matched break-even above "
    "-- the two findings are unrelated: the break-even failure is torch-fem's own "
    "reference solve running out of memory, not a property of the operator's own "
    "accuracy, which this table shows is genuinely strong for this case."
)

p1 = insert_after(
    anchor,
    "The advisor's own round-11 point 2 asked explicitly whether the same QoI/norm "
    "determines the coarsest-suitable-FEM crossover for every case, or whether it "
    "varies -- and expected it would NOT be the same metric every time. Repeating "
    "the same crossover analysis already published for B1xNeo-Hookean (coarsest "
    "suitable N=11, bound by the tangent-energy norm) for the five remaining "
    "cases, against each case's own NO accuracy at N=1401, gives Table 18-R10m. "
    "Checkpoint-version caveat, same as Table 18-R10k above: this crossover reuses "
    "each case's own NO accuracy sweep from task #22, which predates all four "
    "multi-resolution retrains -- B1xMooney-Rivlin, B1xArruda-Boyce, "
    "B2xMooney-Rivlin and B2xNeo-Hookean are measured here against their ORIGINAL "
    "(pre-retrain) checkpoints, not the newer, more accurate checkpoints now used "
    "elsewhere in this report. B2xArruda-Boyce has only ever had one checkpoint, "
    "so no such caveat applies to that row."
)

HEADER = ['Case', 'Coarsest suitable N', 'Binding QoI']
t = insert_table_after(p1, HEADER, [
    ['B1 x Mooney-Rivlin', '3', 'L2, H1 semi-norm, tangent energy, peak PK1 stress, reaction resultant (all five simultaneously)'],
    ['B1 x Arruda-Boyce', '3', 'L2, H1 semi-norm, tangent energy, peak PK1 stress, reaction resultant (all five simultaneously)'],
    ['B2 x Neo-Hookean', '4', 'tangent energy'],
    ['B2 x Mooney-Rivlin', '4', 'tangent energy'],
    ['B2 x Arruda-Boyce', '4', 'tangent energy'],
])

caption_para = insert_after_table(
    t, p1,
    "Table 18-R10m. Coarsest suitable finite-element mesh and its binding QoI per "
    "case, N=1401, the same crossover analysis already published for "
    "B1xNeo-Hookean (N=11, bound by tangent energy). N=3 is the coarsest "
    "resolution tested in this sweep (N = 3, 4, 5, 6, 9, 11, ..., 49), so the true "
    "crossover for the two B1 rows below may lie even coarser than N=3, untested "
    "here."
)

# Interpretation paragraph, anchored right after the table's own caption.
insert_after(
    caption_para,
    "The advisor's own question turns out to have a genuinely mixed answer, not "
    "uniform in either direction. For B2, all three materials -- Neo-Hookean, "
    "Mooney-Rivlin, and Arruda-Boyce -- are bound by the SAME metric, the "
    "tangent-energy norm, exactly matching the already-published B1xNeo-Hookean "
    "finding: this looks like a property of the tangent-energy norm's own "
    "sensitivity (or of the B2 ring geometry) rather than a coincidence of one "
    "material. For B1's other two materials the picture differs in kind, not just "
    "in which single metric binds: all five metrics -- L2, H1 semi-norm, tangent "
    "energy, peak stress, and reaction resultant -- cross simultaneously at N=3, "
    "the coarsest mesh tested. This follows directly from how poor the "
    "(pre-retrain) operator's own accuracy is for these two cases at N=1401 "
    "(L2_rel 33.65% for Mooney-Rivlin, 44.57% for Arruda-Boyce): torch-fem's own "
    "N=3 mesh, the cheapest finite-element discretization tested anywhere in this "
    "report, already beats it on every quantity of interest measured. Whether an "
    "even coarser mesh (N<3, untested) would also suffice is not answered here."
)

doc.save(DST)
print('Saved', DST)
