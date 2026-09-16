"""Replaces the "still mid-training, not yet included" placeholder for
the advisor's round-11 point 4 (a B1 x Neo-Hookean checkpoint trained
DIRECTLY at N=1401, as an ablation against the multi-resolution
zero-shot result) with the real, now-complete result -- task #24
finished on real A100 GPU (all 5 steps: generate/train/accuracy-check/
inference-timing/comparison, one job, no crash) on 2026-09-16.

Unlike every other script in this directory, this one EDITS an existing
paragraph's own text in place (paragraph 388, a single plain run, style
"Normal") rather than only inserting new content after an anchor --
the old text was a forward-looking placeholder for exactly this result
and is now simply wrong (the ablation is no longer "still
mid-training"), so leaving it untouched and only appending new material
after it would leave a stale, self-contradicting sentence in the
document. The edit is narrow and mechanical: clear the paragraph's one
run and add a replacement run with the corrected text, changing nothing
about the paragraph's own style/formatting. B2xNeo-Hookean's own
"still mid-training" status is UNCHANGED (still genuinely true as of
this writing) and is preserved, just no longer bundled with the
ablation in the same sentence.

Real numbers (direct_n1401_vs_multires_comparison.json, transcribed
verbatim, not estimated):
    training wall-clock: direct=28,791.9s (8.00h), multi-res=41,881.28s (11.63h)
    disp_rel_L2 @ N=1401: direct=36.65%, multi-res=5.85%
    inference (eager fp32): direct=2318.8 ms/sample, multi-res=2292.1 ms/sample
"""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-16b.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-16c.docx')

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para_exact(text):
    hits = [p for p in ORIGINAL if p.text.strip() == text]
    assert len(hits) == 1, f'{len(hits)} paragraphs exactly match {text!r}'
    return hits[0]


def replace_paragraph_text(para, new_text):
    """In-place text replacement: clear this paragraph's own runs and add
    one new run, preserving the paragraph's own style/formatting exactly
    (this document's relevant paragraph has a single plain "Normal"-style
    run, verified before writing this function)."""
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    para.add_run(new_text)
    return para


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


anchor = find_para_exact(
    "Two cases remain open as of this writing and are not included in the tables "
    "above: B2 x Neo-Hookean's own multi-resolution retrain was still mid-training "
    "(most recently observed at epoch 575 of a 2000-epoch cap, validation error "
    "still improving, no sign of a plateau yet), and the advisor's own round-11 "
    "point 4 -- a B1 x Neo-Hookean checkpoint trained directly at N=1401, as an "
    "ablation against the zero-shot multi-resolution result -- was likewise still "
    "mid-training. Both will be added to this section once complete, rather than "
    "reported early from an unfinished run."
)

replace_paragraph_text(anchor,
    "One case remains open as of this writing and is not included in the tables "
    "above: B2 x Neo-Hookean's own multi-resolution retrain was still mid-training "
    "(most recently observed at epoch 575 of a 2000-epoch cap, validation error "
    "still improving, no sign of a plateau yet). It will be added to this section "
    "once complete, rather than reported early from an unfinished run.")

p1 = insert_after(anchor,
    "The advisor's own round-11 point 4 -- a B1 x Neo-Hookean checkpoint trained "
    "DIRECTLY at N=1401, kept as an ablation against the zero-shot multi-resolution "
    "result above -- has since completed (Table 18-R10j). 100 training samples at "
    "N=1401 alone (a reduced budget relative to the multi-resolution recipe's own "
    "400 samples per resolution, chosen given N=1401's real per-sample generation "
    "cost) were used, with early stopping (patience 8, unchanged) stopping the run "
    "itself at epoch 36, best epoch 20.")

HEADER = ['Metric', 'Direct training at N=1401', 'Multi-res (zero-shot to N=1401)']
t = insert_table_after(p1, HEADER, [
    ['Training wall-clock', '28,791.9 s (8.00 h)', '41,881.28 s (11.63 h)'],
    ['disp_rel_L2 @ N=1401', '36.65%', '5.85%'],
    ['Inference (eager fp32)', '2318.8 ms/sample', '2292.1 ms/sample'],
])
p2 = insert_after_table(t, p1,
    "Table 18-R10j. Direct-N1401 training ablation vs. the multi-resolution "
    "zero-shot checkpoint, same architecture, same material (B1 x Neo-Hookean).")

p3 = insert_after(p2,
    "The result is unambiguous: training directly at the target resolution is "
    "cheaper (about 30% less GPU time) but produces a model 6.3x LESS accurate "
    "(36.65% vs. 5.85% relative L2 error) than training on four cheaper "
    "resolutions (N=21,33,101,201) and zero-shot generalizing to N=1401 -- despite "
    "the direct model never having to generalize across resolutions at all, only "
    "fit the single resolution it is evaluated on. Inference cost is essentially "
    "identical either way (2318.8 ms vs. 2292.1 ms per sample): the training "
    "data's own resolution range does not change the deployed model's own "
    "architecture or parameter count, so it does not change inference cost. This "
    "is a genuine result in favour of the multi-resolution training strategy, not "
    "merely a demonstration that it also works zero-shot: it produces a "
    "substantially better model than direct training at the target resolution, "
    "for less than 50% more training time. A plausible explanation, stated as a "
    "hypothesis rather than an independently confirmed cause: 100 training "
    "samples at N=1401 alone is a considerably smaller and less varied effective "
    "training set than 400 samples spread across four resolutions, which may make "
    "the direct model more prone to over/under-fitting its own narrow, "
    "single-resolution training distribution.")

doc.save(DST)
print('Saved', DST)
