"""Omar's explicit request, 2026-10-05: "put them in the report too" (the
multi-seed training-uncertainty results, already added to the paper this
same day). Section 11.10 (PFEM_Transolver_Report_2026-10-04.docx) closed
out the B3 local-integration-refinement result as a single training run,
with training-seed variability left unquantified. Two further independent
local-integration-refinement training runs (seeds 2 and 3, identical
configuration, only the weight-initialization seed differing) have since
been trained on Colab and evaluated; this script adds one new subsection
(11.11) reporting the real 3-seed mean+-std, and updates the Executive
Summary row 9 cross-link, following this project's own established
convention (see add_b3_training_evaluation_local_refinement_and_breakeven.py).

Every number below is copied verbatim from the real Colab run output
(seed 2 and seed 3 training logs + region_local_refine_convergence.json/
global_qois_and_jacobian.json, pasted directly by Omar) and from the
actual B3_Multiseed_Aggregate.ipynb notebook run (its own printed
mean+-std, numpy's default population std, ddof=0) -- no new computation
beyond that aggregation, no new GPU time, this is a writing task.
"""
import copy
import os

from docx import Document
from docx.oxml.ns import qn

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-10-04.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-10-05.docx')

doc = Document(SRC)
REF_TABLE_STYLE = doc.tables[0].style
REF_TABLE_TBLPR = doc.tables[0]._tbl.find(qn('w:tblPr'))


def add_table(header, rows):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = REF_TABLE_STYLE
    if REF_TABLE_TBLPR is not None:
        old = tbl._tbl.find(qn('w:tblPr'))
        if old is not None:
            tbl._tbl.remove(old)
        tbl._tbl.insert(0, copy.deepcopy(REF_TABLE_TBLPR))
    for j, h in enumerate(header):
        c = tbl.cell(0, j)
        c.text = ''
        r = c.paragraphs[0].add_run(h)
        r.bold = True
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            tbl.cell(i, j).text = str(v)
    return tbl


# ===========================================================================
# 11.11 Multi-Seed Training Uncertainty
# ===========================================================================
doc.add_heading('11.11 Multi-Seed Training Uncertainty', level=2)
doc.add_paragraph(
    'The local-integration-refinement result reported in 11.8-11.10 rests '
    'on a single trained checkpoint (one weight-initialization seed; '
    'train_B3.py has no --seed argument and never calls '
    'torch.manual_seed itself, so weight initialization uses PyTorch’s '
    'default unseeded RNG while the training-data sampling schedule is '
    'deterministic). To quantify how much of that result depends on '
    'training-seed luck rather than the method itself, two further '
    'independent local-integration-refinement training runs were trained '
    '(seeds 2 and 3, identical configuration throughout -- same '
    'architecture, 50,000 iterations, local_refine_n_sub=10, same reused '
    'input normalization -- only torch.manual_seed differs), each '
    'evaluated on the same 100-sample validation set as seed 1.'
)
add_table(
    ['Quantity', 'Seed 1 (original)', 'Seed 2', 'Seed 3', 'Mean +/- std (population)'],
    [
        ('Regional stress (pooled Frobenius, n_sub=10)', '29.77%', '16.95%', '47.33%', '31.35% +/- 12.45%'),
        ('Displacement (combined)', '2.40%', '1.10%', '1.96%', '1.82% +/- 0.54%'),
        ('u_x', '2.52%', '1.02%', '2.06%', '1.87% +/- 0.63%'),
        ('u_y', '21.10%', '6.22%', '16.59%', '14.64% +/- 6.23%'),
        ('u_z', '1.74%', '1.03%', '1.45%', '1.41% +/- 0.29%'),
        ('Energy (pooled RMS)', '0.42%', '0.115%', '0.278%', '0.27% +/- 0.12%'),
        ('Reaction moment (pooled RMS)', '2.72%', '2.06%', '2.42%', '2.40% +/- 0.27%'),
        ('Predicted min J', '0.803', '0.8063', '0.8169', '0.809 +/- 0.006'),
    ],
)
doc.add_paragraph(
    'This is a real, substantial finding, not a minor caveat. The '
    'headline "62% to 29.8%" regional-stress improvement from local '
    'integration refinement is itself seed-dependent: across the 3 '
    'independent training runs it actually ranges 16.95%-47.33% (mean '
    '31.35%, standard deviation 12.45 percentage points, i.e. roughly '
    '40% of the mean). The original run’s 29.77% turns out to sit '
    'almost exactly at the 3-seed mean, but the true range is far wider '
    'than a single run suggests. u_y shows the same pattern (standard '
    'deviation 6.23 points on a mean of 14.64%); energy and reaction are '
    'comparatively more stable relative to their own means (standard '
    'deviation under 50% of the mean in both cases).'
)
doc.add_paragraph(
    'The deformation-validity check (J = det F, every element and Gauss '
    'point, 5,472,000 sampled points) was re-run for all 3 seeds: the '
    'true FEM field itself is identical across seeds (min J = 0.8007, '
    'since it is the same ground truth every time, 1st percentile 0.977, '
    'median 0.9999, zero points with J <= 0). The 3 local-refinement '
    'seeds’ own predicted min J stays in a tight 0.803-0.817 band '
    '(0.809 +/- 0.006) regardless of seed, with zero points with J <= 0 '
    'in every one of the 3 runs. Unlike the regional-stress result above, '
    'deformation validity is therefore robust to training-seed variation.'
)
doc.add_paragraph(
    'The baseline checkpoint, which supplies the displacement (1.9%), '
    'energy (0.3%), and reaction (2.7%) figures and the break-even '
    'analysis throughout 11.6-11.9, was not re-run at additional seeds in '
    'this pass; its own training-seed variability remains unquantified '
    'and is noted here as a limitation, not as a claim that its specific '
    'numbers are seed-independent. The local-integration-refinement '
    'checkpoint’s own seed variability, by contrast, has now been '
    'directly measured and is substantial for regional stress and u_y '
    'specifically, as reported above.'
)

# ===========================================================================
# Cross-link: Executive Summary feedback-tracking table (row 9) -- same
# consistency-audit pattern this project has repeatedly applied after
# adding new B3 content.
# ===========================================================================
table = doc.tables[0]
header_cells = [c.text.strip() for c in table.rows[0].cells]
assert header_cells == ['#', 'point', 'Headline result'], header_cells
row9 = None
for r in table.rows[1:]:
    if r.cells[0].text.strip() == '9':
        row9 = r
        break
assert row9 is not None, 'row 9 (3D candidate) not found -- has the table changed?'
current = row9.cells[2].text
expected_tail = (
    'Regional Cauchy stress, after local integration refinement, is '
    'reported as a documented limitation (62.2% reduced to 29.8%, still '
    'above the 5-10% target) -- see 11.8 and 11.10.'
)
assert expected_tail in current, 'row 9 text has changed since the script was written -- re-check before overwriting'
row9.cells[2].text = current.replace(
    expected_tail,
    'Regional Cauchy stress, after local integration refinement, is '
    'reported as a documented limitation: a single run gave 62.2% '
    'reduced to 29.8%, still above the 5-10% target, but a 3-seed '
    'uncertainty check (11.11) found this itself ranges 16.95%-47.33% '
    '(mean 31.35% +/- 12.45%) -- see 11.8, 11.10, and 11.11.'
)

doc.save(DST)
print('Saved', DST)

check = Document(DST)
print('paragraphs:', len(check.paragraphs))
print('tables:', len(check.tables))
print('table[0] row 9 col 2:', check.tables[0].rows[-1].cells[2].text[:160] + '...')
print('last heading:', [p.text for p in check.paragraphs if p.style.name == 'Heading 2'][-1])
