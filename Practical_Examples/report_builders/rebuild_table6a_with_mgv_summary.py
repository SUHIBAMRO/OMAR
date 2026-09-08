"""Item #3 (condition a), Summary counterpart: rebuild the Summary's own
Table 6a with the genuinely CG-converged multigrid data, mirroring
rebuild_table6a_with_mgv_report.py. Modifies the REAL, already-updated
Summary deliverable (PFEM_Work_Summary_updated_2026-09-08.docx, which
already has item #4's compact table from the previous pass).
"""
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-08.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_updated_2026-09-08b.docx')

doc = Document(SRC)


def text_of(el):
    return ''.join(t.text or '' for t in el.iter(qn('w:t')))


def set_para_text(p, text):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.add_run(text)


d401 = json.load(open(os.path.join(
    PF, 'highdof_stress_qoi_results', 'high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json')))
d_rest = json.load(open(os.path.join(
    PF, 'highdof_stress_qoi_results',
    'high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json')))

row401 = next(r for r in d401['orders']['Q4']['rows'] if r['N'] == 401)
rows_by_n = {701: None, 1001: None, 1401: None}
for r in d_rest['orders']['Q4']['rows']:
    if r['N'] in rows_by_n:
        rows_by_n[r['N']] = r
rows_by_n[401] = row401
for n, r in rows_by_n.items():
    assert r['cg_failures'] == 0


def fmt_err(v):
    s = f'{v:.2e}'
    mantissa, exp = s.split('e')
    sign = exp[0]
    digits = exp[1:].lstrip('0') or '0'
    return f'{mantissa}e{sign}{digits}'


def fmt_wall(s):
    return f'{s:.1f} s'


new_cells = {}
for n in (401, 701, 1001, 1401):
    r = rows_by_n[n]
    new_cells[n] = {
        'L2 rel.': fmt_err(r['l2_rel_error']),
        'H1 rel.': fmt_err(r['h1_semi_rel_error']),
        'Energy (tangent) rel.': fmt_err(r['energy_rel_error']),
        'Wall-clock': fmt_wall(r['wall_clock_s']),
    }

rates = d_rest['orders']['Q4']['convergence_rates']
l2_p, h1_p, energy_p = rates['l2_fit'], rates['h1_semi_fit'], rates['energy_fit']

body = doc.element.body
children = list(body)

cap_idx = None
for i, ch in enumerate(children):
    if ch.tag == qn('w:p') and text_of(ch).strip().startswith('Table 6a.'):
        cap_idx = i
        break
assert cap_idx is not None
tbl_idx = None
for j in range(cap_idx - 1, cap_idx - 4, -1):
    if children[j].tag == qn('w:tbl'):
        tbl_idx = j
        break
assert tbl_idx is not None

tbl = Table(children[tbl_idx], doc)
header = [c.text for c in tbl.rows[0].cells]
col_idx = {h: k for k, h in enumerate(header)}
for row in tbl.rows[1:]:
    n_val = int(row.cells[0].text.replace(',', ''))
    if n_val in new_cells:
        for col_name, new_val in new_cells[n_val].items():
            row.cells[col_idx[col_name]].text = new_val

caption_para = Paragraph(children[cap_idx], doc)
old_caption = caption_para.text
assert old_caption == 'Table 6a. B1 x Neo-Hookean, Q4 vs ~10M-DOF reference'
set_para_text(
    caption_para,
    'Table 6a. B1 x Neo-Hookean, Q4 vs ~10M-DOF reference. Every row is now solved '
    'with the geometric multigrid preconditioner (see the CG-convergence note below), '
    f'so N=401-1401 are genuinely converged, not the CG-capped solves an earlier '
    f'revision used. Fitted rates: L2 p={l2_p:.2f}, H1 p={h1_p:.2f}, energy '
    f'p={energy_p:.2f} -- within 0.01 of the earlier CG-capped fit; only the '
    'wall-clock column changes substantially.'
)

doc.save(DST)
print('Saved', DST)
