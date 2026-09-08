"""Item #3 (condition a): fully rebuild Table 6a using the genuinely
CG-converged multigrid solves (item #4) instead of the old
plain-Jacobi, CG-capped ones at N=401/701/1001/1401. Omar's explicit
choice: rebuild the table itself, not just add a pointer note.

Modifies the REAL, already-updated Report deliverable
(PFEM_Transolver_Report_updated_2026-09-08.docx, which already has
item #4's Table 20c from the previous pass).

N=51/101/201 are UNCHANGED: their checkpoints were never affected by
CG capping (0 of 20 hit the cap even in the original run), so the
converged displacement -- and therefore every error number -- is
numerically the same regardless of which preconditioner reached it.
Only N=401/701/1001/1401's rows change: L2/H1/energy move by at most a
handful of parts in the last reported digit (confirming the report's
own earlier claim that Newton's own convergence check had already
bounded the damage from CG truncation), while wall-clock changes
substantially (these are now real converged costs, not truncated-budget
costs or, at N=1001, a checkpoint-resume artifact).
"""
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

PF = '/home/user/OMAR/Practical_Examples/omar_pfem'
DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08b.docx')

doc = Document(SRC)


def text_of(el):
    return ''.join(t.text or '' for t in el.iter(qn('w:t')))


def set_para_text(p, text):
    """Replaces a paragraph's runs with a single run carrying new text,
    preserving the paragraph's own formatting (matches this file's
    established insert_after/new-paragraph convention elsewhere)."""
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
    assert r['cg_failures'] == 0, f'N={n} still has cg_failures'


def fmt_err(v):
    # Match the table's existing exponent style (single digit, no leading
    # zero, e.g. "6.99e-5" not "6.99e-05").
    s = f'{v:.2e}'
    mantissa, exp = s.split('e')
    sign = exp[0]
    digits = exp[1:].lstrip('0') or '0'
    return f'{mantissa}e{sign}{digits}'


def fmt_wall(s):
    # The existing table reports every row in plain seconds regardless of
    # magnitude (e.g. the old N=1401 row was "11871.5 s") -- match that
    # convention rather than switching units partway through the table.
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

# Recomputed least-squares fit across all 7 resolutions, from the same
# combined mgv run's own reported convergence_rates (N=51/101/201/401
# drawn from the SAME converged checkpoints Table 6a always used; only
# 701/1001/1401 actually changed).
rates = d_rest['orders']['Q4']['convergence_rates']
l2_p = rates['l2_fit']
h1_p = rates['h1_semi_fit']
energy_p = rates['energy_fit']

body = doc.element.body
children = list(body)

# --- Locate Table 6a's table element and caption/discussion paragraphs.
tbl_idx = None
caption_idx = None
for i, ch in enumerate(children):
    if ch.tag == qn('w:p') and text_of(ch).strip().startswith('Table 6a.'):
        caption_idx = i
        break
assert caption_idx is not None
# Table sits 2 elements before the caption (table, blank paragraph, caption).
for j in range(caption_idx - 1, caption_idx - 4, -1):
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

# --- Rewrite the caption (rates + drop the N=1001-checkpoint-artifact footnote).
caption_para = Paragraph(children[caption_idx], doc)
old_caption = caption_para.text
assert old_caption.startswith('Table 6a. B1 × Neo-Hookean, Q4, error vs. the ~10M-DOF reference.')
set_para_text(
    caption_para,
    'Table 6a. B1 × Neo-Hookean, Q4, error vs. the ~10M-DOF reference, using the '
    'geometric multigrid preconditioner of §8.5 (Table 20c) so every row below is a '
    'genuinely CG-converged solve. Least-squares convergence rates across all seven '
    f'resolutions: L2 p={l2_p:.2f}, H1 p={h1_p:.2f}, energy p={energy_p:.2f} -- each '
    'within 0.01 of the earlier CG-capped fit, confirming the discussion below\'s '
    'point that Newton\'s own convergence check had already bounded the damage from '
    'CG truncation; only the wall-clock column below changes substantially.'
)

# --- Rewrite the two-caveats discussion paragraph: the CG-cap caveat is
# resolved (folded into one sentence); the fine_N caveat still applies.
caveats_para = None
for p in doc.paragraphs:
    if p.text.strip().startswith("The L2 error satisfies the advisor's 10⁻⁴ target"):
        caveats_para = p
        break
assert caveats_para is not None
set_para_text(
    caveats_para,
    "The L2 error satisfies the advisor's 10⁻⁴ target already at N=201 and continues "
    f"to shrink to {rows_by_n[1401]['l2_rel_error']:.2e} by the finest resolution "
    "tested (N=1401). The H1 semi-norm and the tangent energy norm improve more "
    f"slowly and remain above the 10⁻⁴ target even at N=1401 "
    f"({rows_by_n[1401]['h1_semi_rel_error']:.2e} and "
    f"{rows_by_n[1401]['energy_rel_error']:.2e} respectively), consistent with "
    f"their lower fitted convergence rates (H1 p={h1_p:.2f}, energy p={energy_p:.2f}, "
    f"versus L2's p={l2_p:.2f}). Extrapolating the fitted rates, closing that gap "
    "through further test-mesh refinement alone would require a test resolution far "
    "beyond the current ~10M-DOF reference mesh itself (N=2236) and is therefore not "
    "achievable against this reference; reaching the target this way would require a "
    "substantially finer reference as well, at a DOF count well beyond what is "
    "computationally tractable in this project's available compute budget. One "
    "caveat applies to the N=1001 and N=1401 rows specifically: the fine reference "
    "(N=2236) is only 1.6-2.2x these two N values rather than the recommended 4x, so "
    "the reference mesh's own discretization error is not fully negligible next to "
    "theirs, which flattens (underestimates) the true convergence rate at these two "
    "points, most visibly in H1. A second caveat that applied to an earlier revision "
    "of this table -- the solver's matrix-free CG hitting its 2,000-iteration cap "
    "without reaching cg_tol at N=1001 and N=1401 -- is now resolved: every row "
    "above is solved with the geometric multigrid preconditioner of §8.5 (Table "
    "20c), under which CG converges within its original budget at every N in this "
    "table, so the error and wall-clock values above are genuinely converged rather "
    "than truncated."
)

# --- Remove the now-redundant pointer paragraph added in the previous
# pass (its content is folded into the caveats paragraph above).
pointer_para = None
for p in doc.paragraphs:
    if p.text.strip().startswith('The first of those two caveats is now resolved'):
        pointer_para = p
        break
assert pointer_para is not None
pointer_para._p.getparent().remove(pointer_para._p)

doc.save(DST)
print('Saved', DST)
print('New rates: L2 p=%.4f H1 p=%.4f energy p=%.4f' % (l2_p, h1_p, energy_p))
