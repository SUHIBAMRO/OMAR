"""Full, final rebuild of round-12 point 1 (2026-09-18) using the new,
self-consistent data source (round12_consistent_field_qoi_<case>.json,
produced by run_qoi_study_consistent_field_b1/b2). This replaces BOTH
earlier versions of this section:

  1. The original version (Tables 18-R10o/p/q): incomplete "for each
     resolution" coverage, only one case had a full sweep.
  2. The "fixed" version (rebuild_round12_point1_full.py, then
     fix_stale_classical_qoi.py): fixed coverage and a stale-checkpoint
     bug, but still compared the operator against FEM-solved-AT-THE-SAME-N
     (not a genuine fine reference) for classical QoIs, and -- caught only
     after Omar pushed back a second time, rejecting a text-only caveat --
     compared FEM's own numbers (AnalyticField) against the operator's own
     numbers (ParametricField) as if they were the same physical problem,
     for BOTH classical and Cauchy QoIs.

This version has FEM and the operator solved on the EXACT SAME
ParametricField(seed) realization, both scored against ONE real fine
reference (N=201), for every QoI -- L2, H1, energy, reaction (B1 only),
and the region-Cauchy stress statistics. Same table numbering as before
(18-R10o..z, two tables per case in case order B1xNH/B1xMR/B1xAB/B2xNH/
B2xMR/B2xAB) since the STRUCTURE was already right; only the underlying
computation changes.
"""
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

DELIV = '/home/user/OMAR/advisor_feedback'
FIG = '/home/user/OMAR/Practical_Examples/report_builders/figures'
SCRATCH = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/round12c'

REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-18.docx')

CASE_LABELS = {
    'B1_neo_hookean': 'B1 x Neo-Hookean',
    'B1_mooney_rivlin': 'B1 x Mooney-Rivlin',
    'B1_arruda_boyce': 'B1 x Arruda-Boyce',
    'B2_neo_hookean': 'B2 x Neo-Hookean',
    'B2_mooney_rivlin': 'B2 x Mooney-Rivlin',
    'B2_arruda_boyce': 'B2 x Arruda-Boyce',
}
CASE_ORDER = list(CASE_LABELS)
CASE_FILE = {
    'B1_neo_hookean': 'b1_nh', 'B1_mooney_rivlin': 'b1_mr', 'B1_arruda_boyce': 'b1_ab',
    'B2_neo_hookean': 'b2_nh', 'B2_mooney_rivlin': 'b2_mr', 'B2_arruda_boyce': 'b2_ab',
}

DATA = {}
for case_id, fname in CASE_FILE.items():
    with open(os.path.join(SCRATCH, f'round12_consistent_field_qoi_{fname}.json')) as f:
        d = json.load(f)
    DATA[case_id] = {r['N']: r for r in d['rows']}
    assert len(DATA[case_id]) == 16, case_id

LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

LETTERS = list('opqrstuvwxyz')
TABLE_ID = {}
for i, case_id in enumerate(CASE_ORDER):
    TABLE_ID[(case_id, 'classical')] = f'18-R10{LETTERS[2 * i]}'
    TABLE_ID[(case_id, 'cauchy')] = f'18-R10{LETTERS[2 * i + 1]}'


def pct(x):
    return f'{x * 100:.2f}%' if x is not None else 'n/a'


CLASSICAL_HEADER = ['N', 'FEM L2', 'Op. L2', 'FEM H1', 'Op. H1', 'FEM Energy', 'Op. Energy', 'FEM Reaction', 'Op. Reaction']
CAUCHY_HEADER = ['N', 'FEM avg', 'Op. avg', 'FEM p99', 'Op. p99', 'FEM max', 'Op. max']


def classical_rows(case_id):
    rows = []
    for N in LOW_N:
        r = DATA[case_id][N]
        fem, no = r['fem'], r['no']
        rows.append([
            str(N),
            pct(fem['l2_rel']), pct(no['l2_rel']),
            pct(fem['h1_semi_rel']), pct(no['h1_semi_rel']),
            pct(fem['energy_rel']), pct(no['energy_rel']),
            pct(fem['reaction_resultant_rel_err']), pct(no['reaction_resultant_rel_err']),
        ])
    return rows


def cauchy_rows(case_id):
    rows = []
    for N in LOW_N:
        r = DATA[case_id][N]
        fem, no = r['fem'], r['no']
        rows.append([
            str(N),
            pct(fem['cauchy_avg_rel_err']), pct(no['cauchy_avg_rel_err']),
            pct(fem['cauchy_p99_rel_err']), pct(no['cauchy_p99_rel_err']),
            pct(fem['cauchy_max_rel_err']), pct(no['cauchy_max_rel_err']),
        ])
    return rows


# =======================================================================
# docx surgery helpers
# =======================================================================
def body_children(doc):
    return list(doc.element.body)


def elem_text(el):
    if el.tag != qn('w:p'):
        return None
    return ''.join(t.text or '' for t in el.iter(qn('w:t')))


def find_index(children, text):
    matches = [i for i, el in enumerate(children)
               if elem_text(el) is not None and elem_text(el).strip() == text]
    assert len(matches) == 1, f'{len(matches)} matches for {text!r}'
    return matches[0]


def _elem_of(x):
    if hasattr(x, '_p'):
        return x._p
    if hasattr(x, '_tbl'):
        return x._tbl
    return x


def remove_range_return_anchor(doc, start_text, end_text):
    children = body_children(doc)
    i0 = find_index(children, start_text)
    i1 = find_index(children, end_text)
    assert i1 >= i0
    anchor = children[i0 - 1]
    for el in children[i0:i1 + 1]:
        el.getparent().remove(el)
    return anchor


def insert_paragraph_after(doc, anchor, text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    p_elem = p._p
    p_elem.getparent().remove(p_elem)
    _elem_of(anchor).addnext(p_elem)
    return p_elem


def insert_table_after(doc, anchor, header, rows):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = doc.tables[0].style
    for j, h in enumerate(header):
        tbl.rows[0].cells[j].text = h
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            tbl.rows[i].cells[j].text = v
    tbl_elem = tbl._tbl
    tbl_elem.getparent().remove(tbl_elem)
    _elem_of(anchor).addnext(tbl_elem)
    return tbl_elem


def insert_figure_after(doc, anchor, image_path, caption_text, width_in):
    img_p = doc.add_paragraph()
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_p.add_run().add_picture(image_path, width=Inches(width_in))
    img_elem = img_p._p
    img_elem.getparent().remove(img_elem)

    cap_p = doc.add_paragraph()
    cap_run = cap_p.add_run(caption_text)
    cap_run.italic = True
    cap_run.font.size = Pt(9)
    cap_elem = cap_p._p
    cap_elem.getparent().remove(cap_elem)

    anchor_elem = _elem_of(anchor)
    anchor_elem.addnext(cap_elem)
    anchor_elem.addnext(img_elem)
    return cap_elem


# =======================================================================
# content text
# =======================================================================
INTRO_TEXT = (
    "Round-12 point 1 (Cauchy-stress fixed-region QoI, final checkpoints): the advisor "
    "asked for one final accuracy-versus-resolution table -- for EACH resolution, FEM "
    "vs. operator vs. the same fine reference, in displacement L2, H1/energy norm, "
    "reaction force, and stress -- with Cauchy stress (not PK1) as the main engineering "
    "quantity, explicitly rejecting the bare pointwise maximum as the primary QoI "
    "(singular/mesh-dependent at corners even for FEM) in favour of a FIXED physical "
    "region around the stress concentration, reporting a robust local statistic (a "
    "weighted average or 99th percentile) with the true max still available separately, "
    "the region fixed in physical space across every resolution. All new code "
    "(fixed-region selection, the Cauchy push-forward from the already-validated PK1 "
    "stress, and the weighted-percentile/top-fraction statistics) was verified on CPU "
    "before any GPU time was spent (5 checks, all passed). Below: two tables per case, "
    "all six cases, all sixteen resolutions tested (N=3..49) -- the established "
    "accuracy QoIs, then the new Cauchy-stress QoI. Scope, stated plainly: these "
    "resolutions are the LOW_N range already established for this project's own "
    "coarsest-suitable-mesh crossover analysis above, against a fine reference at "
    "N=201 -- this does NOT extend to N=1401, which would need a fresh, expensive "
    "fine_N=2236 reference for the five cases that have never had one, a separate, "
    "bigger ask that should be scoped on its own rather than silently bundled in here."
)

GAP_NOTE = (
    "Two real problems were caught and fixed in this section before it was finalized, "
    "both by Omar's own review rather than a scheduled check, and both are stated here "
    "plainly rather than silently absorbed. First (2026-09-18, from a real GPU run): "
    "the operator's own classical QoIs were being compared against FEM solved at the "
    "SAME low resolution, not against a genuine fine reference -- the Cauchy-stress "
    "table alone used a real fine reference correctly. Second, and deeper (caught the "
    "same day when Omar rejected a proposed text-only caveat and asked for the "
    "computation itself to be fixed): FEM's own numbers in this section were computed "
    "under a fixed, deterministic material/load field (AnalyticFieldB1/B2, the field "
    "this project's separate Table-6a-style mesh-convergence studies use by design), "
    "while the operator's own numbers used a randomly-seeded field family "
    "(ParametricFieldB1/B2) -- the same field the operator was actually trained and "
    "tested on. Same table, same row, same N, but two different physical problems, not "
    "merely two different mesh resolutions of the same one. Both problems are now fixed "
    "at the code level (not worked around): FEM and the operator are solved on the "
    "EXACT SAME ParametricFieldB1/B2(seed) realization, and both are scored against ONE "
    "real fine reference (N=201) for every QoI below -- L2, H1, energy, reaction "
    "(B1 only), and the region-Cauchy statistics. Verified on CPU before any GPU time "
    "was spent (an identity check gives exactly zero error when a solution is compared "
    "against itself; a real coarse-vs-fine pair gives sane, non-degenerate numbers), "
    "then run for real on all six cases. Some values below differ substantially from "
    "earlier drafts of this table as a direct result of this fix -- most notably "
    "B2xNeo-Hookean, where the operator's own displacement L2 at N=3 is 210.82% against "
    "the correct fine reference (it was 260.62% in an intermediate, still-partially-"
    "wrong draft, and separately anywhere from 12.71% to 53.33% in even earlier, "
    "differently-stale drafts of just this one cell) -- every number below is now "
    "traceable to one single, self-consistent GPU run, not a patchwork of separately "
    "fixed pieces."
)

CLOSING_DISCUSSION = (
    "Two findings worth stating plainly, now on the corrected, self-consistent data. "
    "First, the true max is markedly noisier and slower-converging than the region "
    "average for FEM -- B1xNeo-Hookean's own FEM region average falls to 0.30% by N=49 "
    "(displacement L2 is already down to 0.06% there), while its FEM true max is still "
    "64.5% at N=3 and only reaches 26.6% at N=49 -- a real, data-grounded illustration "
    "of exactly why the advisor asked to avoid the bare pointwise maximum as the "
    "primary design QoI: even FEM's own true max, with no operator involved at all, "
    "converges far more slowly than the region-averaged statistic he asked to use "
    "instead. Second, the operator's own region-averaged stress error is consistently "
    "worse than its displacement L2 error at matched resolution (e.g. B1xMooney-Rivlin "
    "at N=13: L2 10.39% vs. region average 10.69%) -- stress is a genuinely harder, "
    "noisier target for the operator than displacement, not merely a rescaled version "
    "of the same accuracy. B2xNeo-Hookean is the extreme case: its operator's own "
    "region-averaged stress error reaches 245.59% at N=3 and stays in the 3.2%-245.6% "
    "range across the whole sweep, never settling anywhere close to FEM's own sub-1% "
    "region average at the comparable resolutions -- confirming that displacement "
    "accuracy alone should not be treated as a stand-in for stress accuracy. Also "
    "worth flagging honestly rather than smoothing over: two cases now show a "
    "NON-monotonic operator error with resolution (B2xNeo-Hookean's own L2 rises again "
    "from 3.87% at N=29 to 18.92% at N=49; B2xArruda-Boyce's own L2 rises from 4.79% at "
    "N=29 to 38.86% at N=49) -- unlike FEM, which converges smoothly and monotonically "
    "at every N for every case, the operator's own accuracy does not degrade smoothly "
    "outside a well-behaved middle range, a genuine, real property of this checkpoint's "
    "own zero-shot generalization, not a data artefact (both cases' own fine reference "
    "and FEM sweep converge cleanly at every N checked)."
)

FLAGSHIP_FIGURE_CAPTION = (
    f"Round-12 Figure A. B1 x Neo-Hookean, region-average vs. true-max Cauchy stress "
    f"error, FEM and operator, both vs. the same fine reference (Table "
    f"{TABLE_ID[('B1_neo_hookean', 'cauchy')]})."
)


def classical_caption(case_id):
    tid = TABLE_ID[(case_id, 'classical')]
    label = CASE_LABELS[case_id]
    extra = ''
    if case_id.startswith('B2'):
        extra = ' Reaction cells are n/a for both methods (no reaction resultant defined for the B2 ring geometry).'
    return (
        f"Table {tid}. {label}: displacement L2, H1 semi-norm, tangent-energy norm, "
        f"and reaction-force error, FEM vs. operator, BOTH solved on the same "
        f"ParametricField(seed) realization and scored against the same fine "
        f"reference (N=201), all sixteen resolutions tested.{extra}"
    )


def cauchy_caption(case_id):
    tid = TABLE_ID[(case_id, 'cauchy')]
    label = CASE_LABELS[case_id]
    return (
        f"Table {tid}. {label}: fixed-region Cauchy-stress error (region-weighted "
        f"average, 99th percentile, true max), FEM vs. operator, BOTH solved on the "
        f"same ParametricField(seed) realization and scored against the same fine "
        f"reference and fixed-region convention as the L2/H1/energy/reaction table "
        f"above, all sixteen resolutions tested."
    )


# =======================================================================
# apply to the Report
# =======================================================================
doc = Document(REPORT)
paras = list(doc.paragraphs)
before_p, before_t, before_i = len(doc.paragraphs), len(doc.tables), len(doc.inline_shapes)

START_TEXT = paras[[i for i, p in enumerate(paras)
                     if p.text.strip().startswith('Round-12 point 1 (Cauchy-stress')][0]].text.strip()
END_TEXT = [p.text for p in paras if 'Two findings worth stating plainly' in p.text][0]

anchor = remove_range_return_anchor(doc, START_TEXT, END_TEXT)

p = insert_paragraph_after(doc, anchor, INTRO_TEXT)
p = insert_paragraph_after(doc, p, GAP_NOTE)

for case_id in CASE_ORDER:
    p = insert_paragraph_after(doc, p, CASE_LABELS[case_id] + ':', bold=True)
    p = insert_paragraph_after(doc, p, classical_caption(case_id))
    p = insert_table_after(doc, p, CLASSICAL_HEADER, classical_rows(case_id))
    p = insert_paragraph_after(doc, p, cauchy_caption(case_id))
    p = insert_table_after(doc, p, CAUCHY_HEADER, cauchy_rows(case_id))
    if case_id == 'B1_neo_hookean':
        p = insert_figure_after(doc, p, os.path.join(FIG, 'fig_round12_cauchy_flagship.png'),
                                 FLAGSHIP_FIGURE_CAPTION, width_in=6.0)

p = insert_paragraph_after(doc, p, CLOSING_DISCUSSION)

doc.save(REPORT)
print('Saved', REPORT)

check = Document(REPORT)
after_p, after_t, after_i = len(check.paragraphs), len(check.tables), len(check.inline_shapes)
print(f'paragraphs: {before_p}->{after_p}, tables: {before_t}->{after_t}, images: {before_i}->{after_i}')
