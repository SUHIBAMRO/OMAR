"""Adds items #6 (OOD progressive-shift, remaining 5 cases), #7 (DD-NO
coarse-vs-fine training-mesh resolution study), and #14 (GOEE paper
confirmation) to the Work Summary document Omar uploaded directly (the
copy he actually sent to Timon). Anchors verified unique in this file
before writing this script.
"""
from docx import Document

SRC = 'Summary_uploaded.docx'
DST = 'Summary_updated.docx'

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


# ---------------------------------------------------------------------
# Item #6: append to the end of section 6 (Out-of-distribution
# generalization), right before section 7's heading.
# ---------------------------------------------------------------------
anchor6 = find_para_exact('7. Resolution invariance (zero-shot)')

anchor6.insert_paragraph_before(
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

# ---------------------------------------------------------------------
# Item #7: append to the end of section 7 (Resolution invariance),
# right before section 8's heading.
# ---------------------------------------------------------------------
anchor7 = find_para_exact('8. Error in physically important quantities beyond displacement')

anchor7.insert_paragraph_before(
    'Separate from the above: a new study asks how the resolution of '
    'the TRAINING mesh itself (not the trained model’s own '
    'generalization) shapes zero-shot accuracy, for a data-driven '
    '(not physics-informed) operator. Two B1 × Neo-Hookean '
    'data-driven models were trained under the Table 21 protocol '
    '(800/200 samples, 75,000 steps, batch size 8), one labelled at '
    'N=13, the other at N=33, and both evaluated zero-shot at the '
    'same seven resolutions as Table 12. The coarse-trained model is '
    'most accurate at its own resolution (10.5% at N=13) but '
    'degrades to 25.9% at N=49, a 2.5× spread. The fine-trained model '
    'is worse at N=13 (13.7%) but its error DECREASES monotonically '
    'to 10.9% at N=49, staying inside a tight 10.9–13.7% band '
    'throughout. Training on a finer mesh trades some peak accuracy '
    'for materially better cross-resolution stability — the direct '
    'answer to this study’s own question.'
)

anchor7.insert_paragraph_before(
    'Note on the "GOEE" / trust reference (separate from the above '
    'two items): the paper Timon sent has been confirmed directly '
    'with Omar as arXiv:2609.02982v1 (Cheng, Duruisseaux, Clauser et '
    'al., "Equation Recast for Canonical Operator Learning Across '
    'Parametric PDEs"). It is not goal-oriented error estimation in '
    'the adjoint/FEM sense used to name it — no dual problem, no '
    'computable QoI error bound. Its only trust-adjacent content is '
    'an informal, explicitly unverified heuristic: treating '
    'non-convergence of an internal fixed-point iteration as a '
    'runtime reliability flag for a prediction. Usable only for that '
    'narrower claim, as a candidate direction for future work, not as '
    'a method implemented anywhere in this report.'
)

doc.save(DST)
print('wrote', DST)
