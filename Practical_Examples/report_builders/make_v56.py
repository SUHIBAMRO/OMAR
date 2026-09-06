"""v55 -> v56. Timon's round-8 point 7, "Comparison B": total budget of
M FEM runs (data generation, data-driven only) + training + N
inferences, computed for the physics-informed operator versus the
data-driven one (not against FEM) -- "For VINO, there is no data
generation."

Every number this needs is already committed: the data-driven model's
800-solve label-generation cost (5.65 h, Table 21's own section, §8.9),
both models' training wall-clock at each matched optimiser (Table 21),
and the physics-informed model's inference cost (Table 10a). No new
experiment -- this is a calculation over existing numbers, which is why
it was flagged in PROJECT_STATUS.md as startable immediately rather than
gated on the new coarse-vs-fine DD-NO study (a separate ask, point 3).

The one figure NOT independently measured: the data-driven checkpoint's
own inference latency. It shares the physics-informed model's
architecture exactly (same Transolver, same size), so this assumes equal
inference cost between the two -- stated explicitly, not silently. Under
that assumption the per-inference term cancels between the two totals,
so the "break-even" is not a crossover in N at all: it is a fixed gap
(the label-generation cost minus the training-time difference) that
holds for every N, not just below some threshold. That is itself the
finding worth reporting, not a limitation of the calculation.
"""
import json
import os

from docx import Document

SRC, DST = 'PFEM_Transolver_Report_v55.docx', 'PFEM_Transolver_Report_v56.docx'
HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, '..', 'omar_pfem')

D = json.load(open(os.path.join(PF, 'point7b_results', 'comparison_B1_neo_hookean.json')))
runs = D['runs']

pi_adam = runs['physics_informed']['train_wall_clock_s']
pi_onecycle = runs['physics_informed_adamw_onecycle']['train_wall_clock_s']
dd_adam = runs['data_driven_matched_optimizer']['train_wall_clock_s']
dd_onecycle = runs['data_driven_own_optimizer']['train_wall_clock_s']
label_cost_h = runs['data_driven_matched_optimizer']['label_generation_cost_h']
assert label_cost_h == runs['data_driven_own_optimizer']['label_generation_cost_h'] == 5.65
label_cost_s = label_cost_h * 3600.0

assert (pi_adam, pi_onecycle, dd_adam, dd_onecycle) == (2873.8, 3108.9, 1458.3, 1463.0)

dd_total_adam = label_cost_s + dd_adam
dd_total_onecycle = label_cost_s + dd_onecycle
gap_adam = dd_total_adam - pi_adam
gap_onecycle = dd_total_onecycle - pi_onecycle

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para(prefix):
    hits = [p for p in ORIGINAL if p.text.strip().startswith(prefix)]
    assert len(hits) == 1, f'{len(hits)} paragraphs start with {prefix!r}'
    return hits[0]


anchor = find_para('The comparison covers one case, B1 × Neo-Hookean')

new_table_caption = (
    'Table 21a. Total cost before the first inference, physics-informed '
    'versus data-driven, at each matched optimiser: training time plus '
    '(data-driven only) the 800-solve label-generation cost of Table '
    '4a. The physics-informed model pays no label-generation cost.'
)
rows = [
    ('Adam, lr 2×10⁻³', f'{pi_adam:,.1f}', '—', f'{pi_adam:,.1f}',
     f'{dd_adam:,.1f}', f'{label_cost_s:,.1f}', f'{dd_total_adam:,.1f}',
     f'{gap_adam:,.1f}'),
    ('AdamW lr 10⁻³ + OneCycleLR', f'{pi_onecycle:,.1f}', '—',
     f'{pi_onecycle:,.1f}', f'{dd_onecycle:,.1f}', f'{label_cost_s:,.1f}',
     f'{dd_total_onecycle:,.1f}', f'{gap_onecycle:,.1f}'),
]
header = ['Optimiser', 'PI training (s)', 'PI label-gen (s)',
          'PI total (s)', 'DD-NO training (s)', 'DD-NO label-gen (s)',
          'DD-NO total (s)', 'DD-NO − PI (s)']

new_p = anchor.insert_paragraph_before(new_table_caption)

tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
tbl.style = doc.tables[0].style
from docx.oxml.ns import qn
pr = doc.tables[0]._tbl.find(qn('w:tblPr'))
if pr is not None:
    import copy
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
        tbl.cell(i, j).text = v

# Move the freshly-built table (currently appended at the very end of the
# document body) to sit right after its caption paragraph.
tbl._tbl.getparent().remove(tbl._tbl)
new_p._p.addnext(tbl._tbl)

explanation = anchor.insert_paragraph_before(
    'The data-driven model is more expensive by a fixed amount — '
    f'{gap_adam:,.0f} s ({gap_adam/3600:.2f} h) under matched Adam, '
    f'{gap_onecycle:,.0f} s ({gap_onecycle/3600:.2f} h) under matched '
    'AdamW+OneCycleLR — for every number of future inferences, not just '
    'below some threshold. This assumes the two checkpoints’ inference '
    'cost is equal, which was not separately measured for the '
    'data-driven model but follows from it sharing the physics-informed '
    'model’s architecture exactly; under that assumption the '
    'per-inference term is identical on both sides of the comparison '
    'and cancels, so what remains is the label-generation cost minus '
    'the (small, and inconsistently signed) training-time difference — '
    'a constant, not a break-even point in the usual sense. The '
    'label-generation cost dominates by more than an order of magnitude '
    'in both pairings, which is why the sign of that constant does not '
    'depend on which optimiser is used.'
)

doc.save(DST)
print('wrote', DST)
print(f'gap_adam={gap_adam:.1f}s ({gap_adam/3600:.3f}h)')
print(f'gap_onecycle={gap_onecycle:.1f}s ({gap_onecycle/3600:.3f}h)')
