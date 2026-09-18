"""Fixes a real data-staleness bug Omar's own pasted GPU run log surfaced
(2026-09-18): the classical-QoI ("Op." L2/H1/Energy/Reaction) columns in
Tables 18-R10q/s/u/w/y (B1xMooney-Rivlin, B1xArruda-Boyce, B2xNeo-Hookean,
B2xMooney-Rivlin, B2xArruda-Boyce) were populated from cached
no_accuracy_degradation_sweep_<case>.json files that predated the
checkpoint-fingerprint safety check entirely -- i.e. from an earlier,
pre-final-retrain checkpoint, exactly the same class of bug already found
and fixed for B1xNeo-Hookean earlier in this project.

This was only discovered because Omar's Remaining5_LowN_Accuracy.ipynb run
(closing the N=3,4,5,6,9,11 gap) triggered the sweep function's own
discard-and-recompute safety net for all five cases (their old files had
no recorded fingerprint at all), and the freshly recomputed N=13 value for
B1xMooney-Rivlin (10.48%) did not match what was already written in the
Report (7.39%).

Fix: replace the "Op." L2/H1/Energy/Reaction columns for ALL sixteen N (not
only the six that were previously "n/a") with the freshly fetched,
fingerprint-verified sweep data for all five cases. FEM columns and every
Cauchy-stress column are untouched -- they were never sourced from these
stale files.
"""
import json
import os

from docx import Document
from docx.oxml.ns import qn

DELIV = '/home/user/OMAR/advisor_feedback'
SCRATCH = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad'
ROUND12 = os.path.join(SCRATCH, 'round12')
ROUND12B = os.path.join(SCRATCH, 'round12b')

TARGETS = [
    os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-18.docx'),
]

# ---------------------------------------------------------------------
# merge fresh, fingerprint-verified data for the 5 stale cases
# ---------------------------------------------------------------------
with open(os.path.join(ROUND12, 'round12_final_accuracy_cauchy_summary.json')) as f:
    DATA = json.load(f)

FIX_FILES = {
    'B1_mooney_rivlin': 'no_accuracy_degradation_sweep_B1_mooney_rivlin.json',
    'B1_arruda_boyce': 'no_accuracy_degradation_sweep_B1_arruda_boyce.json',
    'B2_neo_hookean': 'no_accuracy_degradation_sweep_B2_neo_hookean.json',
    'B2_mooney_rivlin': 'no_accuracy_degradation_sweep_B2_mooney_rivlin.json',
    'B2_arruda_boyce': 'no_accuracy_degradation_sweep_B2_arruda_boyce.json',
}

EXPECTED_FINGERPRINTS = {
    'B1_mooney_rivlin': 'd7c9a8c4bdb2c3c1c075200b6a4b060dbac6d980e1e163b0fea566f166aeb78e',
    'B1_arruda_boyce': '73cea5f12960c5cdb06954a8be05063b5cebe58adc0c6be991dc31c93283336e',
    'B2_neo_hookean': 'dd2e244d5ac4d4454419fdc55b981d1e78c532d1d7d6458894ea1a74edb3287f',
    'B2_mooney_rivlin': '7681830b40f18ef266f5a890eb54c8925368509c92df188e0646da489917389c',
    'B2_arruda_boyce': '3424f961455b635d0656b53d8438d53627700575864c385dafeb14d73e8d3f45',
}

# reported (approx) fingerprint from the pasted run log for B1_mooney_rivlin,
# used only as an extra plausibility check that the OLD cached value really
# was stale and not a transcription mistake on our own part.
CHANGED_N13_L2 = {}

for case_id, fname in FIX_FILES.items():
    with open(os.path.join(ROUND12B, fname)) as f:
        fix = json.load(f)
    assert fix['checkpoint_fingerprint'] == EXPECTED_FINGERPRINTS[case_id], \
        f'{case_id}: fingerprint mismatch, refusing to trust this file'
    fix_by_n = {r['N']: r['fp32'] for r in fix['rows']}
    assert len(fix_by_n) == 16, f'{case_id}: expected 16 rows, got {len(fix_by_n)}'
    rows = DATA['cases'][case_id]
    assert len(rows) == 16
    for row in rows:
        fx = fix_by_n[row['N']]
        if row['N'] == 13:
            CHANGED_N13_L2[case_id] = (row['no_l2_rel'], fx['L2_rel'])
        row['no_l2_rel'] = fx['L2_rel']
        row['no_h1_semi_rel'] = fx['H1_semi_rel']
        row['no_energy_rel'] = fx['energy_rel']
        row['no_reaction_rel_err'] = fx.get('reaction_resultant_rel_err')

print('N=13 L2_rel changes (old cached -> fresh, fingerprint-verified):')
for case_id, (old, new) in CHANGED_N13_L2.items():
    print(f'  {case_id}: {old*100:.2f}% -> {new*100:.2f}%')

STALE_CASES = list(FIX_FILES)


def pct(x):
    return f'{x * 100:.2f}%' if x is not None else 'n/a'


def classical_rows(case_id):
    rows = []
    for r in DATA['cases'][case_id]:
        rows.append([
            str(r['N']),
            pct(r['fem_l2_rel']), pct(r['no_l2_rel']),
            pct(r['fem_h1_semi_rel']), pct(r['no_h1_semi_rel']),
            pct(r['fem_energy_rel']), pct(r['no_energy_rel']),
            pct(r['fem_reaction_rel_err']), pct(r['no_reaction_rel_err']),
        ])
    return rows


CAPTION_STARTS = {
    'B1_mooney_rivlin': 'Table 18-R10q. B1 x Mooney-Rivlin: displacement L2',
    'B1_arruda_boyce': 'Table 18-R10s. B1 x Arruda-Boyce: displacement L2',
    'B2_neo_hookean': 'Table 18-R10u. B2 x Neo-Hookean: displacement L2',
    'B2_mooney_rivlin': 'Table 18-R10w. B2 x Mooney-Rivlin: displacement L2',
    'B2_arruda_boyce': 'Table 18-R10y. B2 x Arruda-Boyce: displacement L2',
}


def body_children(doc):
    return list(doc.element.body)


def elem_text(el):
    if el.tag != qn('w:p'):
        return None
    return ''.join(t.text or '' for t in el.iter(qn('w:t')))


def find_table_after_caption(doc, caption_start):
    children = body_children(doc)
    idx = None
    for i, el in enumerate(children):
        txt = elem_text(el)
        if txt is not None and txt.strip().startswith(caption_start):
            idx = i
            break
    assert idx is not None, f'caption not found: {caption_start!r}'
    for el in children[idx + 1:]:
        if el.tag == qn('w:tbl'):
            return el
    raise AssertionError(f'no table found after caption: {caption_start!r}')


def set_table_values(tbl_elem, rows):
    trs = tbl_elem.findall(qn('w:tr'))
    assert len(trs) == 1 + len(rows), f'expected {1 + len(rows)} rows, found {len(trs)}'
    for i, row in enumerate(rows, start=1):
        tcs = trs[i].findall(qn('w:tc'))
        assert len(tcs) == len(row), f'expected {len(row)} cols, found {len(tcs)}'
        for j, val in enumerate(row):
            ts = tcs[j].findall('.//' + qn('w:t'))
            assert len(ts) >= 1, 'no text run in cell'
            ts[0].text = val
            for extra in ts[1:]:
                extra.text = ''


def replace_paragraph_text(para, new_text):
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    para.add_run(new_text)


OLD_GAP_NOTE = (
    "A genuine, pre-existing gap, disclosed rather than hidden: the operator's own "
    "classical accuracy QoIs (L2, H1, energy, reaction) were never evaluated below "
    "N=13 for any case before this round -- its own zero-shot-validated range has "
    "always started there. B1xNeo-Hookean's own \"Op.\" columns below are complete "
    "down to N=3 only because a dedicated follow-up run (2026-09-18, 46s of real GPU "
    "compute) closed that gap specifically for the flagship case; the other five "
    "cases still show \"n/a\" for the operator at N=3,4,5,6,9,11, an honest, "
    "unfilled gap rather than a guess. The Cauchy-stress metric, by contrast, is "
    "complete for every case at every resolution, since it was computed fresh in "
    "this round's own sweep regardless of any older cache."
)
NEW_GAP_NOTE = (
    "This gap is now fully closed for every case. A first follow-up run "
    "(2026-09-18, 46s of GPU compute) closed it for the flagship B1xNeo-Hookean case "
    "alone; a second follow-up run, covering the other five cases together (6m30s "
    "total), closed the N=3,4,5,6,9,11 gap there too. That second run also revealed a "
    "real data-staleness bug worth stating plainly: its own checkpoint-fingerprint "
    "safety check found that the five cases' previously cached classical-QoI sweep "
    "files predated the fingerprint check entirely (i.e. were computed against an "
    "earlier, pre-final-retrain checkpoint) and discarded them, recomputing all "
    "sixteen resolutions fresh against the current final checkpoints -- the same class "
    "of bug already caught and fixed for B1xNeo-Hookean earlier in this project. The "
    "\"Op.\" L2/H1/Energy/Reaction columns below are therefore now complete and "
    "checkpoint-verified for every case at all sixteen resolutions; some values differ "
    "from an earlier draft of this table as a direct result (e.g. B1xMooney-Rivlin at "
    "N=13: L2 was 7.39% against the stale cache, 10.48% against the verified final "
    "checkpoint). The Cauchy-stress metric was unaffected throughout, since it was "
    "always computed fresh in this round's own sweep regardless of any older cache."
)

OLD_CLOSING_SNIPPET = (
    "the operator's own region-averaged stress error is consistently worse than its "
    "displacement L2 error at matched resolution (e.g. B1xMooney-Rivlin at N=13: L2 "
    "7.39% vs. region average 10.69%)"
)
NEW_CLOSING_SNIPPET = (
    "the operator's own region-averaged stress error is consistently worse than its "
    "displacement L2 error at matched resolution (e.g. B1xMooney-Rivlin at N=13: L2 "
    "10.48% vs. region average 10.69%)"
)

for path in TARGETS:
    doc = Document(path)
    before_p, before_t, before_i = len(doc.paragraphs), len(doc.tables), len(doc.inline_shapes)

    for case_id in STALE_CASES:
        tbl_elem = find_table_after_caption(doc, CAPTION_STARTS[case_id])
        set_table_values(tbl_elem, classical_rows(case_id))

    gap_hits = [p for p in doc.paragraphs if OLD_GAP_NOTE in p.text]
    assert len(gap_hits) == 1, f'expected 1 GAP_NOTE match, found {len(gap_hits)}'
    replace_paragraph_text(gap_hits[0], gap_hits[0].text.replace(OLD_GAP_NOTE, NEW_GAP_NOTE))

    closing_hits = [p for p in doc.paragraphs if OLD_CLOSING_SNIPPET in p.text]
    assert len(closing_hits) == 1, f'expected 1 CLOSING snippet match, found {len(closing_hits)}'
    replace_paragraph_text(closing_hits[0], closing_hits[0].text.replace(OLD_CLOSING_SNIPPET, NEW_CLOSING_SNIPPET))

    doc.save(path)
    after = Document(path)
    after_p, after_t, after_i = len(after.paragraphs), len(after.tables), len(after.inline_shapes)
    print(f'{path}: paragraphs {before_p}->{after_p}, tables {before_t}->{after_t}, images {before_i}->{after_i}')
    assert (before_p, before_t, before_i) == (after_p, after_t, after_i), 'structural change detected, unexpected!'
