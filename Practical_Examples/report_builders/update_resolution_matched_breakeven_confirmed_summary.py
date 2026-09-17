"""Companion to update_resolution_matched_breakeven_confirmed.py -- same
in-place table/footnote update to the Summary. See that script's own
docstring for the full rationale."""
import os

from docx import Document

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-16e.docx')
DST = os.path.join(DELIV, 'PFEM_Work_Summary_2026-09-17.docx')

doc = Document(SRC)

TARGET_HEADER = ['Case', 'torch-fem @N=1401', 'Operator @N=1401', 'Speedup', 'Break-even']
NEW_ROWS = {
    'B1 x Neo-Hookean': ['135.00 s', '2292.1 ms', '58.9x', '316 samples'],
    'B1 x Mooney-Rivlin': ['133.92 s', '2361.9 ms', '56.7x', 'training cost unknown'],
    'B2 x Neo-Hookean': ['205.14 s', '2351.1 ms', '87.3x', 'training cost unknown'],
    'B2 x Mooney-Rivlin': ['207.21 s', '2356.0 ms', '88.0x', 'training cost unknown'],
}

found_table = False
for t in doc.tables:
    header = [c.text for c in t.rows[0].cells]
    if header != TARGET_HEADER:
        continue
    found_table = True
    for row in t.rows[1:]:
        case = row.cells[0].text
        if case in NEW_ROWS:
            new_vals = NEW_ROWS[case]
            for j, v in enumerate(new_vals, start=1):
                row.cells[j].text = v
    break
assert found_table, "Table R10-4' not found -- check the header text matches exactly"


def replace_paragraph_text(para, new_text):
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    para.add_run(new_text)


OLD_FOOTNOTE = (
    "Table R10-4'. Resolution-matched break-even, both methods at N=1401, all six "
    "cases. At matched resolution the operator wins by 57-89x even unoptimized -- "
    "unlike Table R10-4's own accuracy-matched comparison, where the default mode "
    "does not break even at all. * B2 x Neo-Hookean reuses an earlier clean "
    "measurement: the freshest re-run's own attempt at this case failed with a "
    "CUDA OOM caused by leftover GPU memory from the immediately preceding B1 x "
    "Arruda-Boyce failure (81.27 GB already allocated on a 79.25 GB device before "
    "this case even started) -- a memory-cleanup gap between cases in the sweep "
    "script, not a real finding about this case. Both Arruda-Boyce cases fail "
    "genuinely and consistently across every run: torch-fem's own Newton-Raphson "
    "solve does not converge at N=1401, root-caused to a real CUDA OOM inside "
    "torch-fem's own Hessian assembly for this specific material."
)
NEW_FOOTNOTE = (
    "Table R10-4'. Resolution-matched break-even, both methods at N=1401, all six "
    "cases. At matched resolution the operator wins by 57-89x even unoptimized -- "
    "unlike Table R10-4's own accuracy-matched comparison, where the default mode "
    "does not break even at all. All four non-Arruda-Boyce numbers are from a "
    "single clean re-run (2026-09-17) after fixing a real memory-cleanup bug in "
    "the sweep script (an exception-chain reference kept a failed case's own GPU "
    "tensors pinned into the next case) -- this re-run succeeded for every "
    "non-Arruda-Boyce case in one pass, confirming the fix on real GPU. Both "
    "Arruda-Boyce cases fail genuinely and consistently across every run: "
    "torch-fem's own Newton-Raphson solve does not converge at N=1401, "
    "root-caused to a real CUDA OOM inside torch-fem's own Hessian assembly for "
    "this specific material."
)

replaced = False
for p in doc.paragraphs:
    if p.text.strip() == OLD_FOOTNOTE:
        replace_paragraph_text(p, NEW_FOOTNOTE)
        replaced = True
        break
assert replaced, "footnote paragraph not found -- check the text matches exactly"

doc.save(DST)
print('Saved', DST)
