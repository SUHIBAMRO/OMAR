"""Item #13: write the real GPU-FEM vs. torch-fem efficiency comparison
result (N=401/701/1001/1401, the same four resolutions Table 20/20a/20b/
20c already uses) into the REAL, already-updated Report deliverable
(PFEM_Transolver_Report_updated_2026-09-08.docx).

Placement: a new Table 20d + discussion, inserted right after Table
20c's own last paragraph (the coarsest-level exact-vs-approximate solve
caveat) and before "Three limits of this study should be stated" --
staying inside the same §8.5 scaling section Table 20c already lives
in, following the same idiom the rest of this section already uses
(measured numbers, explicit caveats, no smoothing over an unfavorable
result).
"""
import copy
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08b.docx')

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
        r['N'], f"{r['n_dof']:,}", fmt_time(r['ours_wall_clock_s']),
        f"{r['torchfem_wall_clock_s']:.1f} s", f'{speedup:,.0f}x',
        f"{r['torchfem_peak_mem_mb']:,.0f}",
    ])

HEADER = ['N', 'DOF', 'Ours (mgv), wall clock', 'torch-fem, wall clock',
          'Speedup (ours slower by)', 'torch-fem peak GPU memory (MB)']

anchor = find_para_exact(
    'One implementation detail affects this comparison and is stated for completeness. '
    'The multigrid hierarchy\'s coarsest level is solved exactly wherever it is small '
    'enough (N=401\'s own hierarchy bottoms out at a ~1,300-free-DOF mesh, cheap to '
    'factor directly); N=701 and N=1401\'s hierarchies bottom out at a ~62,000-free-DOF '
    'mesh and N=1001\'s at ~32,000, both too large for an exact dense factorization to '
    'be practical (a rough estimate put a single such factorization at tens of hours, '
    'needed roughly once per Newton iteration), so those three resolutions instead '
    'solve their coarsest level approximately, via a capped 50-iteration matrix-free '
    'CG pass -- a standard inexact-coarse-solve multigrid variant, not the exact solve '
    'N=401 enjoys. This is one plausible source of the smaller apparent advantage at '
    'N=701/1001 relative to N=1401, though disentangling it from ordinary per-size '
    'variation in a single run each would need repeated trials this study\'s budget did '
    'not allow.'
)

p1 = insert_after(
    anchor,
    'A direct comparison against an established GPU FEM library, requested explicitly: '
    '"an efficient GPU implementation... necessary for a fair comparison to a NO," with '
    'no preference stated between torch-fem and TensorMesh. torch-fem, a PyTorch-native, '
    'GPU-accelerated FEM library, was run on the identical mesh, heterogeneous material '
    'field, boundary conditions, and load this study already uses, at the same four '
    'resolutions as Table 20c, using its own best readily-available preconditioner '
    '(Jacobi; AMG would need an additional dependency not assumed installed). '
    'Correctness was validated first, independent of any timing claim: at two small '
    'resolutions (N=11, N=21, solved on CPU), both solvers converge to the same '
    'displacement field to within 2-5e-6 relative difference, comfortably inside '
    'torch-fem\'s own float32 working precision.'
)

p2 = insert_after(
    p1,
    'Table 20d. Wall-clock and torch-fem\'s own peak GPU memory at the same four '
    'resolutions as Table 20c, on the same NVIDIA A100. torch-fem is dramatically '
    'faster in wall-clock at every size tested, by a margin that is large even by the '
    'standards of this comparison and does not move in one direction monotonically with N.'
)

t20d = insert_table_after(p2, HEADER, table_rows)

p3 = insert_after_table(
    t20d, p2,
    'Two caveats belong beside this result, not after it. First, the two solvers are not '
    'run at matched precision or convergence tolerance: torch-fem\'s own near-null-space '
    'construction (needed for its preconditioner setup) hardcodes float32, forcing a '
    'correspondingly loosened Newton/CG tolerance (rtol=atol=1e-3, stol=1e-4) against '
    'this project\'s own float64 solve at rtol=atol=1e-8 -- several orders of magnitude '
    'tighter, which plausibly accounts for a large share of the gap on its own. Second, '
    '"ours" own peak GPU memory is not reported here: to reuse item #4\'s own already-'
    'converged checkpoints rather than re-spending its ~14.86h of GPU time, each row\'s '
    '"ours" solve resumes from a saved state instead of running fresh, which reports the '
    'correct converged wall-clock and CG-iteration counts (identical to Table 20c\'s own '
    'values) but does not allocate the memory a fresh solve would, so no real "ours" '
    'memory figure exists yet to set beside torch-fem\'s measured one.'
)

p4 = insert_after(
    p3,
    'This gap is not read here as a defect to be engineered away. Two architectural '
    'families are being compared, not two implementations of the same one: torch-fem '
    'always explicitly assembles a sparse tangent stiffness matrix once per Newton '
    'iteration and factorizes/solves it with cuSPARSE-backed routines, while this '
    'project\'s own solver never forms that matrix at all, obtaining every Hessian-vector '
    'product it needs by automatic differentiation instead -- the design this project\'s '
    'multi-million-DOF references (including the ~10M-DOF mesh Table 6a compares '
    'against) depend on, since an explicitly assembled tangent at that scale would not '
    'fit in a single GPU\'s memory. This trade-off was anticipated, not discovered here: '
    'the round-8 review already noted that a solver like TensorMesh "presumably assembles '
    'explicitly once and factorizes," in contrast to this solver\'s per-CG-iteration '
    'element-level work; Table 20d is a direct empirical confirmation of that same '
    'concern, measured against torch-fem rather than TensorMesh specifically. The honest '
    'reading is that this project\'s matrix-free solver trades wall-clock speed at the '
    'sizes tested here for the ability to reach substantially larger problems than an '
    'explicitly-assembled approach can hold in GPU memory at all -- both properties worth '
    'stating plainly, not one to the exclusion of the other.'
)

doc.save(DST)
print('Saved', DST)
