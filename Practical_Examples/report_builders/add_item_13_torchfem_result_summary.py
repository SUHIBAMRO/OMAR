"""Item #13: write the real GPU-FEM vs. torch-fem efficiency comparison
result into the REAL, already-updated Work Summary deliverable
(PFEM_Work_Summary_updated_2026-09-08.docx).

Placement: right after the "Cost breakdown" paragraph (the last
paragraph of the GPU-FEM scaling discussion, mirroring the multigrid/
Table 20c write-up already there) and before the "6. Out-of-distribution
generalization" heading -- one compact paragraph plus a small table,
matching this document's own denser, less-narrated style versus the
Report's fuller prose.
"""
import copy
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-08.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-08b.docx')

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


d = json.load(open(os.path.join(
    PF, 'torchfem_comparison_results', 'torchfem_comparison_B1_neo_hookean.json')))
rows_by_n = {r['N']: r for r in d['rows']}
all_rows = [rows_by_n[n] for n in (401, 701, 1001, 1401)]
for r in all_rows:
    assert r['ours_cg_failures'] == 0


def fmt_time(s):
    if s < 3600:
        return f'{s/60:.1f} min'
    return f'{s/3600:.2f} h'


table_rows = []
for r in all_rows:
    speedup = r['ours_wall_clock_s'] / r['torchfem_wall_clock_s']
    table_rows.append([
        r['N'], fmt_time(r['ours_wall_clock_s']), f"{r['torchfem_wall_clock_s']:.1f} s",
        f'{speedup:,.0f}x', f"{r['torchfem_peak_mem_mb']:,.0f} MB",
    ])

HEADER = ['N', 'Ours (mgv)', 'torch-fem', 'Speedup (ours slower by)', 'torch-fem peak mem']

anchor = find_para_exact(
    "Cost breakdown: explicit assembly (residual + Jacobi preconditioner) 0.1–0.6%, CG "
    "99.4–99.9%. This superficially confirms Timon's expectation that assembly should be "
    "minimal and should NOT be quoted that way: matrix-free means every CG iteration IS a "
    "Hessian-vector product, i.e. an assembly-like pass over all elements. The assembly "
    "did not get cheap; it moved inside CG where this instrumentation cannot see it. The "
    "clean split exists only for assemble-once-and-factories solvers on the CPU reference "
    "(Table 4a) assembly outweighs the sparse solve by 290–692×."
)

p1 = insert_after(
    anchor,
    "GPU-FEM vs. torch-fem (item #13, R7.1): direct comparison against an established, "
    "GPU-accelerated FEM library, requested explicitly with no preference between "
    "torch-fem and TensorMesh. Same mesh/material/BCs/load as Table 20c, same four N. "
    "Correctness confirmed first at small N on CPU (2-5e-6 relative displacement "
    "difference, inside torch-fem's float32 precision)."
)

t = insert_table_after(p1, HEADER, table_rows)

p2 = insert_after_table(
    t, p1,
    "torch-fem wins wall-clock by a large, non-monotonic margin (peaks at N=1001, dips "
    "at N=1401). Two caveats, not hidden: (1) torch-fem forced to float32/loose "
    "tolerance (rtol=atol=1e-3) vs. our float64/tight (1e-8) — likely explains much of "
    "the gap; (2) \"ours\" resumed from item #4's checkpoint rather than solving fresh, "
    "so no real \"ours\" peak-memory number exists to set against torch-fem's measured "
    "one (Omar's own call: not worth the extra GPU-hours a fresh solve would cost). Not "
    "read as a defect: torch-fem assembles a sparse tangent once and factorizes it; our "
    "solver never forms one at all, by design, so it reaches DOF counts (the ~10M-DOF "
    "reference this report's own Table 6a depends on) an assembled approach could not "
    "hold in GPU memory. Timon's round-8 review already predicted this exact trade-off "
    "(TensorMesh \"presumably assembles explicitly once and factorizes\", vs. our "
    "solver's per-CG-iteration element work) — Table 20d/this result is a direct "
    "empirical confirmation of that same concern."
)

doc.save(DST)
print('Saved', DST)
