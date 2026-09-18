"""Fills the B1xNeo-Hookean 'n/a' cells in Table 18-R10p (Report) / R10-10
(Summary) with real numbers from the follow-up sweep
(B1NH_FinalCheckpoint_LowN_Accuracy.ipynb, 2026-09-18, 46s of real GPU
compute) that was built specifically to close this gap -- see
cell_no_accuracy_degradation_sweep_b1nh_final_lowN.py and
PROJECT_STATUS.md for the full story. Numbers read directly from the
downloaded result JSON, never hand-transcribed.

Also updates the three surrounding paragraphs (Table 18-R10o's own
caption, the cross-case snapshot's intro, and Table 18-R10p's own
caption) that described this as an open gap -- it no longer is.
"""
import json
import os

from docx import Document

DELIV = '/home/user/OMAR/advisor_feedback'
DATA_PATH = ('/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/'
             'scratchpad/round12/no_accuracy_degradation_sweep_B1_neo_hookean.json')

REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-18.docx')
SUMMARY = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-18.docx')

with open(DATA_PATH) as f:
    SWEEP = json.load(f)

EXPECTED_FINGERPRINT = 'cb318c4694d820152d018bcb3a2caa6be4528f3654a59cc8aa2482e9cd495f86'
assert SWEEP['checkpoint_fingerprint'] == EXPECTED_FINGERPRINT, \
    'sweep file fingerprint does not match the retrained checkpoint fingerprint printed in the run log'

ROW13 = next(r for r in SWEEP['rows'] if r['N'] == 13)['fp32']


def pct(x):
    return f'{x * 100:.2f}%'


NEW_ROW = [
    'B1 x Neo-Hookean',
    f"0.35% / {pct(ROW13['L2_rel'])}",
    f"4.85% / {pct(ROW13['H1_semi_rel'])}",
    f"4.07% / {pct(ROW13['energy_rel'])}",
    f"0.22% / {pct(ROW13['reaction_resultant_rel_err'])}",
]
print('New B1xNeo-Hookean row:', NEW_ROW)


def find_para_exact(paras, text):
    hits = [p for p in paras if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def replace_paragraph_text(para, new_text):
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    para.add_run(new_text)


def update_qoi_table(doc):
    for t in doc.tables:
        header = [c.text for c in t.rows[0].cells]
        if header == ['Case', 'L2 (FEM/op.)', 'H1 (FEM/op.)', 'Energy (FEM/op.)', 'Reaction (FEM/op.)']:
            for row in t.rows[1:]:
                if row.cells[0].text == 'B1 x Neo-Hookean':
                    for j, v in enumerate(NEW_ROW):
                        row.cells[j].text = v
                    return True
    return False


def old_r10o_caption(o_label):
    return (
        f"Table {o_label}. Fixed-region Cauchy-stress error vs. resolution, B1 x "
        "Neo-Hookean, FEM and operator both vs. the same fine reference (N=201), "
        "all sixteen resolutions tested. A genuine gap, stated honestly rather "
        "than papered over: this case's own operator-side L2/H1/energy/reaction "
        "sweep is entirely missing from Drive at this resolution range (every "
        "row null) -- only the region-Cauchy metrics, computed fresh regardless "
        "of that cache, are available here for the flagship case."
    )


def new_r10o_caption(o_label, p_label):
    return (
        f"Table {o_label}. Fixed-region Cauchy-stress error vs. resolution, B1 x "
        "Neo-Hookean, FEM and operator both vs. the same fine reference (N=201), "
        "all sixteen resolutions tested. This table reports the region-Cauchy "
        "metrics only; the same case's established L2/H1/energy/reaction QoIs, "
        f"alongside every other case, are in Table {p_label} below."
    )


OLD_QOI_INTRO = (
    "Cross-case snapshot at N=13 (the lowest resolution in this sweep where "
    "the operator's own L2/H1/energy sweep exists for five of six cases -- "
    "B1xNeo-Hookean's own gap, noted above, means its FEM/op. cells below "
    "are FEM-only): the established accuracy QoIs, FEM vs. operator, both "
    "vs. the same fine reference."
)


def new_qoi_intro(p_label):
    return (
        "Cross-case snapshot at N=13: the established accuracy QoIs, FEM vs. "
        "operator, both vs. the same fine reference, all six cases. "
        "B1xNeo-Hookean's own operator-side sweep at this resolution range was "
        "genuinely missing when this table was first assembled -- closed with a "
        f"dedicated follow-up run against the same final checkpoint (see Table "
        f"{p_label}'s own caption)."
    )


def old_r10p_caption(p_label, o_label):
    return (
        f"Table {p_label}. Displacement L2, H1 semi-norm, tangent-energy norm, and "
        f"reaction-force error at N=13, FEM vs. operator, all six cases, same "
        f"fine reference (N=201) as Table {o_label}. B2's reaction cells are n/a, "
        "matching this project's own established convention (no reaction "
        "resultant defined for the B2 ring geometry)."
    )


def new_r10p_caption(p_label, o_label):
    return (
        f"Table {p_label}. Displacement L2, H1 semi-norm, tangent-energy norm, and "
        f"reaction-force error at N=13, FEM vs. operator, all six cases, same "
        f"fine reference (N=201) as Table {o_label}. B2's reaction cells are n/a, "
        "matching this project's own established convention (no reaction "
        "resultant defined for the B2 ring geometry). B1xNeo-Hookean's own "
        "operator-side cells come from a dedicated follow-up sweep (same final "
        "checkpoint, same LOW_N range and fine reference, 46s of real GPU "
        "compute) run specifically to close a gap found while assembling this "
        "table -- its own operator-side L2/H1/energy/reaction sweep did not "
        "previously exist at this resolution range for the retrained checkpoint."
    )


LABELS = {
    REPORT: ('18-R10o', '18-R10p'),
    SUMMARY: ('R10-9', 'R10-10'),
}

for path, label in [(REPORT, 'Report'), (SUMMARY, 'Summary')]:
    o_label, p_label = LABELS[path]
    doc = Document(path)
    updated = update_qoi_table(doc)
    assert updated, f'{label}: QoI table not found or B1xNeo-Hookean row not found'

    paras = list(doc.paragraphs)
    replace_paragraph_text(find_para_exact(paras, old_r10o_caption(o_label)),
                            new_r10o_caption(o_label, p_label))
    paras = list(doc.paragraphs)
    replace_paragraph_text(find_para_exact(paras, OLD_QOI_INTRO), new_qoi_intro(p_label))
    paras = list(doc.paragraphs)
    replace_paragraph_text(find_para_exact(paras, old_r10p_caption(p_label, o_label)),
                            new_r10p_caption(p_label, o_label))

    doc.save(path)
    print(f'Saved {label}:', path)
