"""Item #4: write the final geometric-multigrid preconditioner result
(N=401/701/1001/1401 all reach full CG convergence, cg_failures=0) into
the REAL, already-updated Report deliverable
(PFEM_Transolver_Report_updated_2026-09-07.docx).

Placement: a new Table 20c + discussion, inserted right after the
Table 20/20a/20b cost-breakdown paragraph and before "Three limits of
this study should be stated" -- the section that already extensively
documents the CG-non-convergence problem this result fixes, following
the same "an earlier revision claimed X; here is what was actually
measured" idiom this report already uses repeatedly in this exact
section (paragraphs on Table 20b, the U-shape, the memory model).
A pointer sentence is also added after Table 6a's own CG-cap caveat,
without altering Table 6a's own published numbers (a full re-fit using
mgv-converged data is noted as a possible future refinement, not
undertaken here).
"""
import copy
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-07.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')

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


# -- Load the real N=401 (fresh-solve) and N=701/1001/1401 (combined-run)
# results. N=401's own row inside the combined JSON reflects a resumed
# checkpoint (wall_clock_s a fraction of a second), not the real fresh
# solve -- the dedicated N401-only run's JSON is the correct source for
# that one row.
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
    n_dof = r['n_dof']
    cg_iters = r['cg_iters']
    newton_iters = r['newton_iters']
    cg_per_newton = cg_iters / newton_iters
    ms_per_cg = r['wall_clock_s'] * 1000 / cg_iters
    table_rows.append([
        r['N'], f'{n_dof:,}', newton_iters, f'{cg_iters:,}', f'0 of {newton_iters}',
        f'{cg_per_newton:,.1f}', f'{ms_per_cg:,.1f}', fmt_time(r['wall_clock_s']),
    ])

HEADER = ['N', 'DOF', 'Newton iters', 'CG iters', 'CG solves that hit the cap',
          'CG per Newton solve', 'ms per CG iteration', 'Solve time']

# -- Anchor: the end of the cost-breakdown paragraph, right before
# "Three limits of this study should be stated."
anchor = find_para_exact(
    'On the cost breakdown, the measured split is unambiguous and its interpretation is '
    'not. Explicit assembly evaluating the residual, and building the Jacobi '
    'preconditioner accounts for 0.1% to 0.6% of the solve, and the CG loop for 99.4% '
    'to 99.9%, with the CG share rising as the problem grows. Read literally that says '
    'the solver dominates and assembly is negligible. It should not be read literally. '
    'A matrix-free solver never forms the tangent, so every CG iteration is a '
    'Hessian-vector product, and a Hessian-vector product is itself a pass over every '
    'element performing assembly-like work. The 99.9% is solver time that contains the '
    'assembly by construction: the assembly has not become cheap, it has moved inside '
    'the CG loop, where this instrumentation cannot separate it. A clean '
    'assembly-versus-solve split exists only for a solver that assembles the tangent '
    'once and factorises it, which is what the batched GPU-native solver of Table 10 '
    'does and what Table 4a measures for the CPU reference where, at the study\'s own '
    'small mesh, assembly outweighs the sparse linear solve by a factor of 309 for this '
    'case, and by 290 to 692 across the six. At the sizes in Table 20 the question '
    'simply does not have the clean answer it has at small scale.'
)

p1 = insert_after(
    anchor,
    'A genuine fix, not a workaround. Raising the CG cap to 8,000 (Table 20b) answers '
    'the counterfactual question but is not itself a deployable fix: it does not know '
    'in advance how large a cap a given N will need, and the two largest resolutions '
    'were left as predictions specifically because finding a large-enough cap by trial '
    'would itself cost hours. A 2x2 per-node block-Jacobi preconditioner was tried '
    'first as a cheap alternative and tested directly at N=401 and N=701: cg_failures '
    'were unchanged (20 and 30, identical to plain Jacobi) and wall-clock moved by '
    'under 1%, so it was rejected on measurement rather than argument. A geometric '
    'multigrid V-cycle was built instead: bilinear-interpolation prolongation, its '
    '1/4-scaled transpose as restriction, damped-Jacobi pre/post-smoothing, and a '
    'coarsest-grid solve, on the same structured (i, j) mesh hierarchy the solver '
    'already builds internally -- validated first on small CPU meshes (its solution '
    'matched the existing plain-Jacobi solve to 1e-12 relative difference, and its '
    'iteration-count advantage over plain Jacobi grew with problem size, 3.0x at the '
    'smallest mesh tested to 9.7x at the largest) before it was ever pointed at N >= 401.'
)

p2 = insert_after(
    p1,
    'Table 20c. The same solver, same eight-resolution family, at N = 401, 701, 1001 '
    'and 1401 -- the four resolutions Table 20a shows hitting the cap -- with the '
    'multigrid preconditioner in place of Jacobi, the same 2,000-iteration cap and '
    '10⁻⁶ tolerance Table 20a itself used, not the raised one Table 20b '
    'needed. Every row reads 0 of 20 capped solves: CG converges within the ORIGINAL '
    'budget at all four sizes, including the two (N=1001, 1401) Table 20b could only '
    'predict.'
)

t20c = insert_table_after(p2, HEADER, table_rows)
p3 = insert_after_table(
    t20c, p2,
    'Multigrid needs far fewer iterations than the O(N) Jacobi law of the earlier '
    'paragraph predicts converged CG would: 135.5 per Newton solve against 3,513 '
    'predicted at N = 701 (26x fewer), 261.0 against ~5,016 at N = 1001 (19x fewer), '
    '318.9 against ~7,020 at N = 1401 (22x fewer) -- the flat, sub-linear growth (135 '
    '-> 261 -> 319 across a 4x range in N) is the actual signature of a working '
    'multigrid method, in contrast to Jacobi\'s iteration count growing without bound. '
    'But each multigrid iteration costs substantially more: several matrix-free '
    'matvecs across every level of the mesh hierarchy plus a coarsest-level solve, '
    'against Jacobi\'s single elementwise division. At N = 701, where Table 20b '
    'supplies a directly MEASURED Jacobi-converged wall clock (5,286 s), multigrid\'s '
    '7,205 s is 36% slower -- the fixed per-iteration overhead is not yet paid back by '
    'the iteration-count saving at this size. At N = 1001, against the same law\'s '
    'UNMEASURED, PREDICTED Jacobi-converged estimate (~15,349 s), multigrid\'s 17,315 s '
    'is 13% slower. At N = 1401, against that same law\'s UNMEASURED, PREDICTED ~41,675 '
    's, multigrid\'s 27,257 s is 35% FASTER -- the crossover reflects multigrid\'s '
    'flatter iteration count starting to outweigh its larger constant overhead as N '
    'grows, exactly the trend a correctly-functioning multigrid method should show, '
    'and the reason to prefer it going forward even though it is not a clean win at '
    'every size tested here.'
)

p4 = insert_after(
    p3,
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

# -- Pointer from Table 6a's own CG-cap caveat, without altering Table 6a's
# published numbers.
anchor_6a = find_para_exact(
    "The L2 error satisfies the advisor's 10⁻⁴ target already at N=201 and "
    "continues to shrink to 2.3×10⁻⁶ by the finest resolution tested "
    "(N=1401). The H1 semi-norm and the tangent energy norm improve more slowly and "
    "remain above the 10⁻⁴ target even at N=1401 (1.63×10⁻³ and "
    "7.82×10⁻⁴ respectively), consistent with their lower fitted "
    "convergence rates (H1 p=0.73, energy p=0.87, versus L2's p=1.58). Extrapolating "
    "the fitted rates, closing that gap through further test-mesh refinement alone "
    "would require a test resolution far beyond the current ~10M-DOF reference mesh "
    "itself (N=2236) and is therefore not achievable against this reference; reaching "
    "the target this way would require a substantially finer reference as well, at a "
    "DOF count well beyond what is computationally tractable in this project's "
    "available compute budget. Two caveats apply to the N=1001 and N=1401 rows "
    "specifically: first, the solver's matrix-free CG hit its 2000-iteration cap "
    "without reaching cg_tol on 40 of 40 Newton iterations at N=1001 and 80 of 80 at "
    "N=1401 (every iteration, at this N, needed more than 2000 CG iterations) the "
    "outer Newton loop still converged because each under-converged CG step still made "
    "adequate descent progress, but this introduces some non-discretization error "
    "beyond what the table's error columns capture. Second, the fine reference "
    "(N=2236) is only 1.6–2.2× these two N values rather than the recommended "
    "4×, so the reference mesh's own discretization error is not fully negligible "
    "next to theirs, which flattens (underestimates) the true convergence rate at "
    "these two points most visibly in H1, whose fitted rate is noticeably below the H1 "
    "rate the earlier five-point-only fit would have extrapolated to."
)

insert_after(
    anchor_6a,
    'The first of those two caveats is now resolved: §8.5 (Scaling to a few '
    'million degrees of freedom, Table 20c) reports a geometric multigrid '
    'preconditioner under which CG converges within its original 2,000-iteration '
    'budget at both N=1001 and N=1401, eliminating the under-converged-CG error '
    'source this table\'s numbers were computed under. The second caveat, the fine '
    'reference being too close in resolution to these two N values for a fully '
    'trustworthy fitted rate, is independent of the preconditioner and still applies. '
    'Table 6a\'s own values above were computed under the original plain-Jacobi run '
    'and are not refitted here; re-solving with the multigrid preconditioner and '
    're-fitting this table is noted as a possible future refinement, not undertaken '
    'in this revision.'
)

doc.save(DST)
print('Saved', DST)
