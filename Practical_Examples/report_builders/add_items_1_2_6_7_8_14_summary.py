"""Adds items #1 (continual-learning note) and #2+#8 (DD-NO wall-clock
+ total-cost-of-ownership numbers) to Summary_updated.docx (which
already has items #6/#7/#14 from the prior pass). Item #5's point
(batch-size-1 = the deployment case) is already stated explicitly in
this document's own section 5 (Table 10d's own text), so nothing is
added for it here.
"""
import copy
import json

from docx import Document
from docx.text.paragraph import Paragraph

SRC = 'Summary_updated.docx'
DST = 'Summary_full_updated.docx'
PF = '/home/user/OMAR/Practical_Examples/omar_pfem'

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def insert_after(anchor_para, text):
    new_p_el = copy.deepcopy(anchor_para._p)
    anchor_para._p.addnext(new_p_el)
    np = Paragraph(new_p_el, anchor_para._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


# ---------------------------------------------------------------------
# Item #1: continual-learning future-work note, right after the item #6
# addition (end of the OOD section).
# ---------------------------------------------------------------------
anchor1 = find_para_exact(
    'The same per-factor isolation has now been run for the other '
    'five geometry × material combinations (Table 25 in the report). '
    'Loading-shift degradation stays mild everywhere, 0.81×–1.35× at '
    'k=3σ, and is at or below baseline for two cases (B2 × '
    'Neo-Hookean 0.99×, B2 × Arruda-Boyce 0.81×). Material-shift '
    'degradation dominates every case, 2.27×–5.90× at k=3σ — an order '
    'of magnitude worse than loading throughout. Mooney-Rivlin is the '
    'most fragile material on both geometries (4.71× on B1, 5.47× on '
    'B2); Arruda-Boyce is the most robust (4.85× on B1, only 2.27× on '
    'B2). B2 × Arruda-Boyce is also the only case where the '
    'loading-shift degradation measures below baseline at EVERY σ '
    'tested, 0.79×–0.93×, not just at k=3 — the same sign at all six '
    'shift magnitudes, which rules out sampling noise.'
)

insert_after(
    anchor1,
    'Candidate future work on this: continual learning, which adapts '
    'a trained operator to new distributions incrementally instead of '
    'insisting on one fixed zero-shot model. Wang, Eshaghi, Zhuang, '
    'Rabczuk, and Liu, "Replay-Based Continual Learning for '
    'Physics-Informed Neural Operators" (arXiv:2605.04832), do exactly '
    'this — replay-and-distillation, no labeled data — on the same '
    'Transolver architecture used throughout this report. Not '
    'implemented or tested here; named as a candidate direction only.'
)

# ---------------------------------------------------------------------
# Items #2 and #8: DD-NO wall-clock and total-cost-of-ownership,
# appended to the end of section 10.
# ---------------------------------------------------------------------
D7b = json.load(open(f'{PF}/point7b_results/comparison_B1_neo_hookean.json'))
runs = D7b['runs']
pi_adam = runs['physics_informed']['train_wall_clock_s']
pi_onecycle = runs['physics_informed_adamw_onecycle']['train_wall_clock_s']
dd_adam = runs['data_driven_matched_optimizer']['train_wall_clock_s']
dd_onecycle = runs['data_driven_own_optimizer']['train_wall_clock_s']
assert (pi_adam, pi_onecycle, dd_adam, dd_onecycle) == (2873.8, 3108.9, 1458.3, 1463.0)
label_cost_h = runs['data_driven_matched_optimizer']['label_generation_cost_h']
assert label_cost_h == 5.65
label_cost_s = label_cost_h * 3600.0
gap_adam = (label_cost_s + dd_adam) - pi_adam
gap_onecycle = (label_cost_s + dd_onecycle) - pi_onecycle

anchor2 = find_para_exact(
    'What survives the flip: the data-driven model needs 800 FEM '
    'solves = 5.65 h of CPU for labels before training starts; the '
    'physics-informed one needs none. That is a property of the '
    'principle, not of the recipe, and it does not move between '
    'columns.'
)

p_wallclock = insert_after(
    anchor2,
    f'Training wall-clock (Table 21): physics-informed {pi_adam:,.1f}s '
    f'(Adam) / {pi_onecycle:,.1f}s (AdamW+OneCycle); data-driven '
    f'{dd_adam:,.1f}s / {dd_onecycle:,.1f}s — before adding the label-'
    'generation cost above.'
)

insert_after(
    p_wallclock,
    'Total cost before the first inference (Table 21a): data-driven is '
    f'more expensive by a FIXED amount — {gap_adam:,.0f}s '
    f'({gap_adam/3600:.2f}h) under matched Adam, {gap_onecycle:,.0f}s '
    f'({gap_onecycle/3600:.2f}h) under matched AdamW+OneCycle — for '
    'every future inference count, not a break-even threshold. Assumes '
    'equal inference cost between the two (same architecture, not '
    'separately measured); under that assumption the per-inference '
    'term cancels and the label-generation cost dominates by more than '
    'an order of magnitude in both pairings.'
)

doc.save(DST)
print('wrote', DST)
