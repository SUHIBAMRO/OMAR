"""Item #4, Summary counterpart: write the final geometric-multigrid
preconditioner result (N=401/701/1001/1401 all reach full CG
convergence, cg_failures=0) into the REAL, already-updated Summary
deliverable (PFEM_Work_Summary_updated_2026-09-07.docx). Mirrors
add_item_4_mgv_result_report.py's data but in the Summary's own
terser voice, placed right after its Table 20/CG-cap discussion.
"""
import copy
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-07.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-08.docx')

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


d401 = json.load(open(os.path.join(
    PF, 'highdof_stress_qoi_results', 'high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json')))
d_rest = json.load(open(os.path.join(
    PF, 'highdof_stress_qoi_results',
    'high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json')))

row401 = next(r for r in d401['orders']['Q4']['rows'] if r['N'] == 401)
rows_rest = {r['N']: r for r in d_rest['orders']['Q4']['rows'] if r['N'] in (701, 1001, 1401)}

assert row401['cg_failures'] == 0
for n in (701, 1001, 1401):
    assert rows_rest[n]['cg_failures'] == 0

all_rows = [row401, rows_rest[701], rows_rest[1001], rows_rest[1401]]


def fmt_time(s):
    if s < 3600:
        return f'{s/60:.1f} min'
    return f'{s/3600:.2f} h'


table_rows = []
for r in all_rows:
    cg_iters = r['cg_iters']
    newton_iters = r['newton_iters']
    cg_per_newton = cg_iters / newton_iters
    ms_per_cg = r['wall_clock_s'] * 1000 / cg_iters
    table_rows.append([
        r['N'], f"{r['n_dof']:,}", f'{cg_per_newton:,.1f}', f'{ms_per_cg:,.1f}',
        fmt_time(r['wall_clock_s']),
    ])

HEADER = ['N', 'DOF', 'CG per Newton solve', 'ms per CG iteration', 'Solve time']

anchor = find_para_exact(
    'That caveat has since been closed. N = 501 and 701 were re-run with the CG cap '
    'raised from 2,000 to 8,000 and nothing else changed. CG converged at both zero '
    'capped solves and the O(N) law, fitted on meshes no larger than N = 301, '
    'predicted 2,511 and 3,513 iterations per Newton solve against the 2,520.8 and '
    '3,528.1 measured, 0.4% at both. The prediction was printed before the runs were '
    'launched. Truncation does cost Newton steps as expected at N = 701 the count '
    'fell from 30 to 20 but not enough to pay for the missing iterations: Table 20 '
    'UNDERSTATES the converged cost, by 28% at N = 501 (1,616 s → 2,063 s) and 18% at '
    'N = 701 (4,487 s → 5,286 s). The per-CG-iteration cost agrees between the '
    'truncated and converged runs to 1.3%, which is two independent measurements of '
    'the quantity the O(DOF) claim rests on. N = 1001 and 1401 were not re-run; the '
    'same model puts them at +25% and +5%, and those two are predictions.'
)

p1 = insert_after(
    anchor,
    'That prediction is now superseded by a real fix, not another cap raise. A 2x2 '
    'block-Jacobi preconditioner was tried first and measured to do nothing (cg_failures '
    'unchanged at N=401/701). A geometric multigrid V-cycle was built instead -- '
    'validated on small CPU meshes first (matches plain Jacobi to 1e-12, advantage '
    'growing 3.0x to 9.7x with size) -- and run for real at all four resolutions that '
    'hit the cap.'
)

t = insert_table_after(p1, HEADER, table_rows)
p2 = insert_after_table(
    t, p1,
    'Every one converges within the ORIGINAL 2,000-iteration cap (0 of 20 hit it), '
    'including N=1001 and N=1401 which the cap-raise approach could only predict. '
    'Multigrid needs 19-26x fewer iterations per Newton solve than Jacobi\'s O(N) law '
    'predicts (135.5/261.0/318.9 against 3,513/~5,016/~7,020 at N=701/1001/1401), '
    'flat and sub-linear where Jacobi grows without bound -- the real signature of '
    'working multigrid. Each iteration costs more, though: against Table 20b\'s '
    'directly measured N=701 (5,286 s), multigrid is 36% slower; against the same '
    'law\'s unmeasured, predicted N=1001 (~15,349 s), 13% slower; against its '
    'unmeasured, predicted N=1401 (~41,675 s), multigrid is 35% FASTER. The crossover '
    '-- a net loss at 701, roughly break-even at 1001, a clear win at 1401 -- is '
    'multigrid\'s flatter iteration count starting to outrun its larger fixed '
    'overhead as N grows, which is the reason to prefer it going forward even though '
    'it is not a clean win everywhere tested. Caveat: N=701/1001/1401\'s coarsest '
    'mesh level (~32,000-62,000 free DOF) is too large to factor exactly like '
    'N=401\'s own ~1,300-DOF coarsest level, so those three solve their coarsest '
    'level approximately (a capped 50-iteration CG pass) -- a standard multigrid '
    'variant, and a plausible partial explanation for the smaller advantage at 701/1001.'
)

doc.save(DST)
print('Saved', DST)
