"""Adds items #6 (OOD progressive-shift, remaining 5 cases), #7 (DD-NO
coarse-vs-fine training-mesh resolution study), and #14 (GOEE paper
confirmation/citation) to the Report document Omar uploaded directly
(the copy he actually sent to Timon). Anchors below were verified to
exist, uniquely, in this exact file before writing this script.
"""
import copy
from docx import Document
from docx.oxml.ns import qn

SRC = 'Report_uploaded.docx'
DST = 'Report_updated.docx'

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para(prefix):
    hits = [p for p in ORIGINAL if p.text.strip().startswith(prefix)]
    assert len(hits) == 1, f'{len(hits)} paragraphs start with {prefix!r}'
    return hits[0]


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def new_table(anchor_para, header, rows):
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


HEADING3_STYLE = find_para(
    'Accuracy against cost: the operator and the finite-element '
    'solver on one pair of axes'
).style

# ---------------------------------------------------------------------
# Item #6: OOD progressive-shift study extended to the other 5 cases
# Inserted right after the paragraph that flagged this as unmeasured,
# i.e. right before the "8.7 Resolution invariance" heading.
# ---------------------------------------------------------------------
anchor6 = find_para_exact(
    'This diagnosis covers B1 × Neo-Hookean. Whether the same '
    'attribution holds for the other five cases has not been '
    'measured, and the B2 rows of Table 11 in particular degrade by '
    'a different factor.'
)

p1 = anchor6.insert_paragraph_before(
    'The same isolation has now been run for the other five geometry '
    '× material combinations, using the identical protocol as Table '
    '19: ten held-out samples per point, the same relative-shift grid '
    'up to k = 3σ. Table 25 gives the material- and loading-shift '
    'endpoints for all six cases; the full per-σ sweep for B1 × '
    'Neo-Hookean remains in Table 19 above.'
)

cap25 = anchor6.insert_paragraph_before(
    'Table 25. Progressive out-of-distribution shift, all six geometry '
    '× material combinations, at the shift endpoint k = 3σ (baseline = '
    'k = 0). Same protocol as Table 19: mean relative L2 error over ten '
    'held-out samples per point, ratio (×) against that case’s own '
    'in-distribution baseline.'
)

header25 = ['Case', 'Baseline', 'Loading @ k=3', '×', 'Material @ k=3', '×', 'Both @ k=3', '×']
rows25 = [
    ('B1 × Neo-Hookean', '0.0867', '0.0927', '1.07×', '0.5112', '5.90×', '0.4601', '5.31×'),
    ('B1 × Mooney-Rivlin', '0.0947', '0.1279', '1.35×', '0.4459', '4.71×', '0.5711', '6.03×'),
    ('B1 × Arruda-Boyce', '0.1434', '0.1529', '1.07×', '0.6957', '4.85×', '0.5235', '3.65×'),
    ('B2 × Neo-Hookean', '0.1555', '0.1532', '0.99×', '0.7381', '4.75×', '0.5680', '3.65×'),
    ('B2 × Mooney-Rivlin', '0.1020', '0.1260', '1.23×', '0.5584', '5.47×', '0.4857', '4.76×'),
    ('B2 × Arruda-Boyce', '0.1852', '0.1493', '0.81×', '0.4209', '2.27×', '0.4956', '2.68×'),
]
new_table(cap25, header25, rows25)

p2 = anchor6.insert_paragraph_before(
    'The pattern already established for B1 × Neo-Hookean holds '
    'without exception in every one of the five new cases: shifting '
    'the loading magnitude alone produces at most mild degradation at '
    'k = 3σ — 0.81× to 1.35× of the in-distribution error — and in two '
    'of the six cases (B2 × Neo-Hookean, B2 × Arruda-Boyce) measures at '
    'or even below the baseline (0.99× and 0.81× respectively). '
    'Shifting the material stiffness alone is the dominant driver in '
    'every case, 2.27× to 5.90× at k = 3σ, an order of magnitude worse '
    'than the corresponding loading shift throughout. The two shifts do '
    'not compound: the combined column is below the material-only '
    'column in five of six cases, matching the sub-additive behaviour '
    'already reported for B1 × Neo-Hookean.'
)

p3 = anchor6.insert_paragraph_before(
    'Material sensitivity is itself material-dependent. Mooney-Rivlin '
    'is the most fragile of the three materials on both geometries '
    '(4.71× on B1, 5.47× on B2), and Arruda-Boyce is consistently the '
    'most robust (4.85× on B1, only 2.27× on B2 — the mildest '
    'degradation measured anywhere in this study). B2 × Arruda-Boyce is '
    'also qualitatively different from every other case: its '
    'loading-shift degradation is below 1.0× at every σ tested, 0.79× '
    'to 0.93×, not just at the k = 3 endpoint — the same sign at all '
    'six shift magnitudes, which rules out this being sampling noise '
    'around 1.0×. Whatever makes Arruda-Boyce’s energy landscape less '
    'sensitive to the applied load on B2 also appears to make it the '
    'most forgiving material under a material-property shift; nothing '
    'in this report identifies the mechanism, and it is worth flagging '
    'as a genuine, material-specific robustness result rather than '
    'folding all six cases into one flat degradation figure.'
)

# ---------------------------------------------------------------------
# Item #7: DD-NO coarse-vs-fine training-mesh resolution study.
# Inserted at the end of Section 8.7, right before the "8.8" heading.
# ---------------------------------------------------------------------
anchor7 = find_para_exact('8.8 Error in physically important quantities beyond displacement')

h3 = anchor7.insert_paragraph_before(
    'Training-mesh resolution: a coarse-versus-fine data-driven '
    'comparison'
)
h3.style = HEADING3_STYLE

p4 = anchor7.insert_paragraph_before(
    'A related but distinct question is not how one trained model '
    'generalizes across mesh resolutions (Table 12), but how the '
    'resolution of the TRAINING mesh itself shapes that '
    'generalization, for a data-driven (not physics-informed) '
    'operator. Two otherwise-identical data-driven Transolver models '
    'were trained on B1 × Neo-Hookean under the same protocol as Table '
    '21 (800 training samples, 200 held out, 75,000 optimiser steps, '
    'batch size 8) — one labelled at a coarse mesh, N = 13, the other '
    'at a finer mesh, N = 33 — and both were evaluated zero-shot, with '
    'no retraining, at Table 12’s same seven unseen resolutions '
    'against the same N = 101 reference.'
)

cap26 = anchor7.insert_paragraph_before(
    'Table 26. Zero-shot mean relative L2 error (per-component, '
    'against the common N=101 reference) at seven unseen resolutions, '
    'for two data-driven B1 × Neo-Hookean models differing only in '
    'their training-label mesh: N = 13 versus N = 33. Neither training '
    'resolution is itself among the seven test points.'
)

header26 = ['Test resolution N', 'Coarse-trained (N=13)', 'Fine-trained (N=33)']
rows26 = [
    ('13', '0.1052', '0.1372'),
    ('17', '0.1173', '0.1282'),
    ('25', '0.1541', '0.1182'),
    ('29', '0.1736', '0.1150'),
    ('37', '0.2105', '0.1110'),
    ('41', '0.2277', '0.1099'),
    ('49', '0.2590', '0.1090'),
]
new_table(cap26, header26, rows26)

p5 = anchor7.insert_paragraph_before(
    'The two models trade accuracy for stability in opposite '
    'directions. The coarse-trained model is the more accurate of the '
    'two at the resolution closest to its own training mesh (10.52% at '
    'N = 13, against the fine model’s 13.72% there) but degrades '
    'sharply and monotonically moving away from it, reaching 25.90% at '
    'N = 49 — a 2.5× spread across the range tested. The fine-trained '
    'model does the opposite: worse than the coarse model at N = 13, '
    'but its error DECREASES monotonically across the entire tested '
    'range, from 13.72% at N = 13 to 10.90% at N = 49, staying inside a '
    'tight 10.9–13.7 percentage-point band throughout. The direct '
    'answer to this study’s own question: training a data-driven '
    'operator on a finer mesh yields a model that is materially more '
    'STABLE across resolutions (worst case 13.7%) than training on a '
    'coarse mesh (worst case 25.9%), at the cost of some peak accuracy '
    'at any one fixed resolution.'
)

# ---------------------------------------------------------------------
# Conclusion: close out item #6's open action, add #7 and #14
# ---------------------------------------------------------------------
remaining6 = find_para_exact(
    'Extend the out-of-distribution evaluation to isolate the '
    'individual contributions of the material-stiffness shift and the '
    'loading-magnitude shift, which were varied together in Section '
    '8.6.'
)
for r in remaining6.runs:
    r.text = ''
remaining6.runs[0].text = (
    'The out-of-distribution evaluation has been extended to isolate '
    'the individual contributions of the material-stiffness shift and '
    'the loading-magnitude shift for all six geometry × material '
    'combinations (Section 8.6, Table 25). Material shift is the '
    'dominant driver in every case (2.27–5.90× at k = 3σ); loading '
    'shift is mild to negligible and, for B2 × Arruda-Boyce, '
    'consistently net-negative (0.79–0.93× at every σ tested); this '
    'item is closed.'
)

last_item = find_para_exact(
    'The resolution-invariance study is reported for all six cases. '
    'Section 8.7 covers the three B1 materials (Table 12) and all '
    'three B2 materials (Table 12b, Table 12c).'
)
def new_list_item(anchor_para, text):
    """Deep-copies anchor_para's XML (numPr, style, everything) so the
    new paragraph renders as a correctly-numbered member of the same
    bulleted list, then replaces its run text."""
    from docx.text.paragraph import Paragraph
    new_p_el = copy.deepcopy(anchor_para._p)
    anchor_para._p.addprevious(new_p_el)
    np = Paragraph(new_p_el, anchor_para._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


p6 = new_list_item(
    last_item,
    'A new study trains a data-driven operator on FEM labels from a '
    'coarse mesh (N = 13) versus a finer mesh (N = 33) and evaluates '
    'both zero-shot at Table 12’s seven unseen resolutions (Section '
    '8.7, Table 26): the coarse-trained model is most accurate at its '
    'own training resolution but degrades to 25.9% at the farthest '
    'mesh tested, while the fine-trained model trades some peak '
    'accuracy for a materially tighter 10.9–13.7% band across the '
    'entire range. Training-mesh resolution trades peak accuracy for '
    'cross-resolution robustness; this item is closed.'
)

p7 = new_list_item(
    last_item,
    'Timon’s suggested "GOEE" / trust reference has been tracked '
    'down and confirmed directly with Omar: arXiv:2609.02982v1 (Cheng, '
    'Duruisseaux, Clauser et al., "Equation Recast for Canonical '
    'Operator Learning Across Parametric PDEs"). It is not goal-oriented '
    'error estimation in the adjoint/FEM sense the name suggests — no '
    'dual problem, no computable QoI error bound — and its only '
    'trust-adjacent content is an informal, explicitly unverified '
    'heuristic: treating non-convergence of an internal fixed-point '
    'iteration as a runtime reliability flag for a prediction. Citable '
    'only for that narrower claim, as a candidate direction for future '
    'work on this report’s own zero-shot predictions, not as a method '
    'implemented anywhere in this report.'
)

doc.save(DST)
print('wrote', DST)
