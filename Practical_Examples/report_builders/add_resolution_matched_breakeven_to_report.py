"""Adds the resolution-matched break-even comparison (operator vs.
torch-fem, BOTH evaluated at N=1401 -- as opposed to Table 18-R10e's
own accuracy-matched comparison, operator@N=1401 vs. FEM@N=11) to the
Report. This was an explicit advisor request kept alive across two
emails: "Please keep both break-even comparisons... the [resolution-
matched] one should now be much more favourable to the NO" (round-11,
point 1) -- computed with real GPU numbers (task #21, "FULLY DONE for
all 6 cases") but never actually transcribed into the Report itself,
per the 2026-09-16 audit that found this gap.

Numbers are taken from a real GPU log pasted directly into the working
conversation (Round6_ResolutionMatchedBreakEven_AllCases.ipynb, a fresh
re-run), NOT retyped from PROJECT_STATUS.md's own slightly earlier
numbers -- the two runs agree to within ordinary timing noise (e.g.
58.5x vs. 59.0x for B1xNeo-Hookean) except for one real problem this
script documents explicitly below rather than silently picking a
number: the fresh run's own B2xNeo-Hookean case failed with a CUDA OOM,
but the GPU-memory printout in that same log shows 81.27 GB already
allocated on a 79.25 GB device BEFORE that case's own forward pass even
ran -- i.e. leftover memory from the immediately preceding B1xArruda-
Boyce failure was never released before the next case started. That is
a memory-cleanup gap in the sweep script between cases, not a real
finding about B2xNeo-Hookean, which completed cleanly in an earlier,
unaffected run (205.92s torch-fem, 2353.5ms operator, 87.5x). This
script therefore reports the fresh run's own numbers for every case
except B2xNeo-Hookean, where it uses that earlier clean measurement,
and says so explicitly in the table's own footnote paragraph -- exactly
the kind of caveat this project's own standing discipline (never
silently pick a number) requires.

Both B1xArruda-Boyce and B2xArruda-Boyce genuinely fail in BOTH runs:
torch-fem's own Newton-Raphson solve does not converge at N=1401,
root-caused (via the real __cause__ chain in the exception, not
guessed) to a CUDA OOM inside torch-fem's own Hessian assembly for this
material specifically -- a real, reported finding, not a defect in this
project's own comparison code.

Inserted directly after Table 18-R10e's own caption (paragraph 403),
before "The same comparison for the other two B1 materials..." (a
different topic, the accuracy-cost Pareto tables) -- the natural place
for a companion table contrasting the two break-even notions Timon
asked to keep side by side.
"""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-16.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-16b.docx')

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
    'Table 18-R10e. Accuracy-matched break-even: operator@N=1401 vs. finite-element '
    'solver@N=11 (its coarsest suitable mesh for the retrained checkpoint). Distinct '
    'from the matched-resolution/matched-batch-size break-even of Section 8.4: this '
    'comparison intentionally runs the two methods at different N, matched by '
    'accuracy rather than by mesh.'
)

p1 = insert_after(anchor,
    "The advisor's round-11 feedback asked to keep BOTH break-even comparisons in "
    "the report, expecting the second kind -- both methods evaluated at the SAME "
    "resolution, N=1401, rather than at each method's own accuracy-matched mesh -- "
    "to look substantially more favourable to the operator. Table 18-R10e' (below) "
    "gives exactly that: the operator's own default eager fp32 forward pass at "
    "N=1401 against torch-fem's own real GPU solve time at the SAME N=1401, for all "
    "six (geometry, material) cases.")

HEADER = ['Case', 'torch-fem @N=1401', 'Operator @N=1401 (eager fp32)', 'Speedup', 'Break-even']

t = insert_table_after(p1, HEADER, [
    ['B1 x Neo-Hookean', '135.18 s', '2292.1 ms', '59.0x', '315 samples'],
    ['B1 x Mooney-Rivlin', '134.16 s', '2347.5 ms', '57.2x', 'training cost unknown'],
    ['B1 x Arruda-Boyce', 'FAILED (see note)', '-', '-', '-'],
    ['B2 x Neo-Hookean', '205.92 s *', '2353.5 ms *', '87.5x *', 'training cost unknown'],
    ['B2 x Mooney-Rivlin', '207.57 s', '2338.7 ms', '88.8x', 'training cost unknown'],
    ['B2 x Arruda-Boyce', 'FAILED (see note)', '-', '-', '-'],
])

p2 = insert_after_table(t, p1,
    "Table 18-R10e'. Resolution-matched break-even: operator vs. torch-fem, both "
    "evaluated at N=1401, all six cases. At this matched resolution the operator "
    "wins by 57x-89x even in its default, unoptimized deployment mode -- markedly "
    "more favourable than the accuracy-matched comparison above (Table 18-R10e), "
    "where the same default mode does not break even at all and the operator's "
    "economic case instead rests on the torch.compile+TF32 optimization from point "
    "3. \"Break-even (samples)\" is only reported where that case's own retrained-"
    "checkpoint training wall-clock is known; for the three cases marked \"training "
    "cost unknown\" only the per-sample speedup is available, since those "
    "checkpoints predate this project's own training-time logging. * B2 x "
    "Neo-Hookean's row uses an earlier, independently clean measurement rather than "
    "the same session's freshest re-run of this sweep: that re-run's own attempt at "
    "this case failed with a CUDA out-of-memory error, but the sweep's own printed "
    "GPU-memory state shows 81.27 GB already allocated on a 79.25 GB device before "
    "this case's forward pass even started -- memory left over from the immediately "
    "preceding B1 x Arruda-Boyce failure, never released before the next case began. "
    "That is a memory-cleanup gap between cases in the sweep script, not a real "
    "finding about this case, which is why the earlier clean number is used here "
    "instead of silently reporting the corrupted re-run's own failure as if it were "
    "a genuine result.")

p3 = insert_after(p2,
    "Both Arruda-Boyce cases fail at N=1401 in every run attempted so far, for a "
    "genuine and already root-caused reason distinct from the memory-cleanup issue "
    "above: torch-fem's own Newton-Raphson solve does not converge (\"did not "
    "converge in increment 8 after 10 cutbacks\"), and the real underlying cause, "
    "read directly from the exception's own cause chain rather than assumed, is a "
    "CUDA out-of-memory error inside torch-fem's own Hessian assembly for this "
    "material specifically -- Arruda-Boyce's own strain-energy density is more "
    "expensive to differentiate twice than Neo-Hookean's or Mooney-Rivlin's, and at "
    "N=1401's problem size that cost alone exhausts the GPU before torch-fem's own "
    "solve can complete. This is reported as a real finding about torch-fem's own "
    "scaling limit for this material at this resolution, not a defect in this "
    "project's own comparison code.")

doc.save(DST)
print('Saved', DST)
