"""Omar's explicit request, 2026-10-04: "add everything not yet added to
the report." Section 11 (PFEM_Transolver_Report_2026-09-24e.docx) ends at
11.5 with candidate selection and VINO dataset scope decided, but
explicitly states "No dataset generation or neural-operator training has
started" -- everything since (dataset generation, training, the input-
normalization fix, the full QoI evaluation, the regional-stress metric
investigation, local integration refinement on both the evaluation and
training sides, the advisor's decisive reply, and the accuracy-matched
FEM/VINO comparison + break-even analysis) is real, already done, and
verified in PROJECT_STATUS.md, but has never been written into the
report itself. This script adds five new subsections (11.6-11.10)
continuing directly from 11.5, plus the same two consistency cross-links
(Executive Summary table, Section 10 remaining-items list) this project
has twice already had to retrofit after the fact (rounds 15-17) -- done
proactively here instead.

Every number below is copied from this project's own already-verified,
already-committed sources (PROJECT_STATUS.md's 2026-09-27 through
2026-10-02 entries, qois_50000.json, region_local_refine_convergence.json,
region_local_refine_n10.json, break_even_B3.json) -- no new computation,
no new GPU time, this is a writing task.
"""
import copy
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DELIV = '/home/user/OMAR/advisor_feedback'
SRC = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-09-24e.docx')
DST = os.path.join(DELIV, 'PFEM_Transolver_Report_2026-10-04.docx')

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
# 11.6 Dataset Generation and Operator Training
# ===========================================================================
doc.add_heading('11.6 Dataset Generation and Operator Training', level=2)
doc.add_paragraph(
    'Following the scope decided in 11.5, a Deep Energy Method (DEM) '
    'Transolver operator was trained for Candidate A (B3) at the fixed '
    'production resolution (21, 20, 19 in the theta/r/z directions) = '
    '6,840 elements, 7,980 nodes, with the groove geometry held fixed '
    '(groove depth 0.20, half-width 0.15, uniform radial grading) exactly '
    'as decided in 11.5. As in B1/B2, the training objective is the total '
    'potential energy functional directly (Pi = U for B3, since the '
    'applied load is a prescribed rigid-body rotation rather than an '
    'external force, so there is no external-work term); no labeled FEM '
    'displacement field ever appears in the loss. The held-out FEM '
    'dataset used for every accuracy number below (100 samples, disjoint '
    'seeds from training) is used only for evaluation, exactly as in B1/B2.'
)
doc.add_paragraph(
    'The first production training run, with the raw E/nu/phi material '
    'and loading values fed directly to the network (no input '
    'standardization), was unstable: the best checkpoint reached only '
    '26.2% combined displacement relative-L2 error on the held-out set, '
    'with the smallest-magnitude component (uy, the axial displacement '
    'driven by the rocking rotation) at 93.6% -- worse than a trivial '
    'all-zero prediction. The root cause was isolated, not assumed: the '
    'raw inputs span three orders of magnitude (E around 1,000, nu around '
    '0.45, phi around 0.05), and a controlled fixed-8-sample A/B test '
    '(identical seed, identical 4,000 iterations, raw inputs versus '
    'standardized-to-mean-0/std-1 inputs) showed combined error dropping '
    'from 10.7% to 1.1% and uy specifically from 54.1% to 4.0% under '
    'normalization alone, with a visibly smoother, monotonic loss curve '
    'replacing the earlier run’s oscillation between healthy and '
    '~1.0-error states.'
)
doc.add_paragraph(
    'A full production retrain with input normalization enabled (50,000 '
    'iterations, same architecture and material/loading distributions as '
    'the unstable run) confirmed this fix generalizes to the genuinely '
    'disjoint held-out set, not just the fixed-pool memorization test: '
    'combined displacement relative-L2 error reached 1.88% at the final '
    'checkpoint (ux 1.94%, uy 16.16%, uz 1.42%), improving cleanly and '
    'mostly monotonically across training rather than oscillating. Real '
    'GPU training wall-clock was 6,980.3 s (1 h 57 m), 0.140 s/iteration, '
    'essentially identical per-iteration cost to the unnormalized run, '
    'confirming normalization adds no meaningful overhead. uy remains the '
    'least accurate of the three displacement components, consistent with '
    'it being the smallest-magnitude term and therefore contributing '
    'proportionally less to a pure energy-minimization loss than the '
    'other two components’ own errors.'
)

# ===========================================================================
# 11.7 Accuracy on Displacement, Energy, and Reaction
# ===========================================================================
doc.add_heading('11.7 Accuracy on Displacement, Energy, and Reaction', level=2)
doc.add_paragraph(
    'Beyond raw displacement, the advisor named total strain energy, '
    'reaction force/moment, and regional Cauchy stress as the real '
    'success criteria for this benchmark. Energy and reaction are '
    'computed directly from a known displacement field (true FEM or the '
    'network’s own prediction) via the same total-potential-energy '
    'functional training already uses, with no new FEM solve: the '
    'reaction at a Dirichlet-constrained node is exactly the derivative '
    'of total potential energy with respect to displacement there, by the '
    'same variational argument any FEM solver uses internally to report '
    'reactions, computed here via automatic differentiation on the known '
    'field. A translation-invariance identity (hyperelastic energy cannot '
    'see a uniform rigid translation of the whole field, so this '
    'quantity, summed over every node, must be approximately zero for ANY '
    'field, equilibrium or not) was used only as an implementation-'
    'correctness sanity check -- confirmed at machine precision (~1e-15) '
    '-- not as a proof of solution quality.'
)
add_table(
    ['QoI', 'Mean relative error (checkpoint_50000, 100-sample held-out set)'],
    [
        ('Total strain energy', '0.30%'),
        ('Reaction moment (about the rocking axis)', '2.67%'),
    ],
)
doc.add_paragraph(
    'Both are well-reproduced quantities, consistent with the operator’s '
    'strong displacement accuracy above. Regional Cauchy stress, the '
    'third named criterion, required substantially more investigation and '
    'is reported separately in 11.8.'
)

# ===========================================================================
# 11.8 Regional Cauchy-Stress: Metric Methodology and Local Integration
# Refinement
# ===========================================================================
doc.add_heading(
    '11.8 Regional Cauchy-Stress: Metric Methodology and Local Integration Refinement',
    level=2)
doc.add_paragraph(
    'The first region-stress number computed for this checkpoint, a naive '
    'relative error on the per-sample signed average of sigma_xx within '
    'the fixed groove-neighborhood region, came back at a mean of 1,683.6% '
    '-- not a credible measure of a genuinely well-behaved checkpoint that '
    'had already reproduced displacement, energy, and reaction well. This '
    'was investigated rather than reported at face value or dismissed: '
    'the signed-average metric can divide by a near-zero denominator '
    'whenever tension and compression partially cancel within the region '
    'for a given sample -- the same failure mode this project had already '
    'documented once before on a different benchmark. Replacing it with a '
    'volume-weighted, quadrature-based full stress-TENSOR relative field '
    'error (squaring each quadrature point’s contribution before '
    'summing, so a near-zero signed average cannot collapse the '
    'denominator) gave a mean of 71.77% / median of 68.45% -- far lower '
    'than the naive number, confirming it was substantially a metric '
    'artifact, but still far above the 5-10% target. A further '
    'contributing factor was identified: at the production resolution '
    '(6,840 elements), only 6 Gauss-quadrature points fall inside the '
    'fixed groove-neighborhood region, below this project’s own '
    'established minimum sample count (20) for reliable statistics.'
)
doc.add_paragraph(
    'This was reported to the advisor in full, together with an open '
    'question of whether to invest further GPU time in a resolution-'
    'changing retrain. His reply redirected the investigation before any '
    'such retrain: (1) clarify the exact computation of the field-error '
    'metric; (2) report all six independent Cauchy-stress components '
    'separately, with their typical magnitudes, plus a tensor error using '
    'one common Frobenius-norm normalization across the whole dataset -- '
    'to rule out a small component inflating the aggregate (the true '
    'Frobenius norm of a symmetric tensor double-counts the off-diagonal '
    'shear terms: ||sigma||_F^2 = sigma_xx^2 + sigma_yy^2 + sigma_zz^2 + '
    '2(sigma_xy^2 + sigma_yz^2 + sigma_xz^2)); (3) confirm what the '
    '43,400-element mesh used in an earlier finer-evaluation diagnostic '
    'represents; (4) test, FIRST, whether refining the LOCAL numerical '
    'integration near the groove -- independently of the operator’s own '
    'mesh discretization -- closes the gap, before any resolution-'
    'changing retrain or loss-function change; (5) explicitly defer both '
    'a groove-specific loss term and multi-resolution training until the '
    'region is properly resolved; (6) keep the single-resolution model '
    'plus zero-shot finer-resolution evaluation as the primary '
    'discretization-invariance check, not superseded by a resolution-'
    'changing retrain.'
)
doc.add_paragraph(
    'An evaluation-side local-integration-refinement tool was built '
    'first, implementing point (4) without any retraining: torch-fem’s '
    'own shape-function evaluator accepts arbitrary local coordinates, so '
    'a finer internal Gauss-Legendre grid re-samples the SAME already-'
    'known coarse-element displacement field (true FEM or the network’s '
    'own prediction) at many more internal points, going from 6 to up to '
    '994 region points with the operator’s own 6,840-element '
    'discretization completely unchanged -- no new FEM solve, no new '
    'network query. A convergence sweep over the refinement level gave:'
)
add_table(
    ['Refinement level (points per axis)', 'Region quadrature points', 'Pooled Frobenius-norm relative error', 'Change from previous level'],
    [
        ('2', '6', '64.04%', '--'),
        ('4', '36', '60.32%', '5.81%'),
        ('6', '124', '62.39%', '3.44%'),
        ('8', '292', '62.30%', '0.14%'),
        ('10', '566', '62.21%', '0.15%'),
        ('12', '994', '62.32%', '0.18%'),
    ],
)
doc.add_paragraph(
    'The error stabilizes from a refinement level of 8 onward (successive '
    'changes under 0.2%), settling at approximately 62.2-62.3%: per the '
    'advisor’s own stated decision criterion, this is a real gap, not a '
    'local-integration-count artifact (a true artifact would keep '
    'drifting toward zero, or keep changing, as points are added; instead '
    'the error converges and stays elevated). The per-component '
    'breakdown directly answers point (2)’s concern as well: the error '
    'is dominated by the large-magnitude normal stresses (sigma_xx, '
    'sigma_yy, sigma_zz, true RMS magnitude roughly 14 to 18), not by the '
    'much smaller shear components (sigma_xy, sigma_yz, sigma_xz, true '
    'RMS magnitude roughly 0.1 to 2.6) -- so the aggregate is not an '
    'artifact of a negligible component dominating a combined metric.'
)
doc.add_paragraph(
    'The same local-refinement idea was then implemented on the TRAINING '
    'side: the DEM background-integration quadrature used to compute the '
    'strain-energy contribution of elements touching the groove region '
    'was refined to 10x10x10 points (instead of the standard 2x2x2), for '
    'ONLY the four elements (of 6,840) that intersect the fixed region, '
    'with the operator’s own mesh, node count, and element count left '
    'completely unchanged otherwise -- directly decoupling the DEM '
    'integration resolution from the operator’s own discretization, '
    'exactly as the advisor asked. A real performance regression was '
    'found and fixed during this work, disclosed here rather than only '
    'in internal notes: an early implementation recomputed shape '
    'functions for the FULL 6,840-element mesh at every fine quadrature '
    'point, inside an unvectorized Python loop, on every single training '
    'iteration -- 1.685 s/iteration on real GPU data, worse than even a '
    '6.3x increase in the operator’s own resolution, even though only '
    'the four region elements’ results were ever used. This was fixed by '
    'precomputing shape functions once, before training, from a reduced '
    'model built from only the region elements’ own node connectivity, '
    'and fully vectorizing the fine-point energy combination -- a 32.5x '
    'speedup measured directly against a reconstruction of the original '
    'implementation, confirmed to reproduce an identical energy value to '
    'about 1e-12 relative precision, and confirmed again on real GPU '
    'training data afterward (0.1525 s/iteration, matching the baseline '
    'run’s own 0.1396 s/iteration).'
)
doc.add_paragraph(
    'Retraining for the same 50,000-iteration budget with this locally-'
    'refined DEM integration reduced the converged regional-stress error '
    'substantially:'
)
add_table(
    ['Stress component', 'True RMS magnitude', 'Baseline checkpoint', 'Local-integration-refinement training'],
    [
        ('sigma_xx', '14.02', '67.0%', '32.2%'),
        ('sigma_yy', '14.94', '63.1%', '29.9%'),
        ('sigma_zz', '18.24', '59.6%', '28.4%'),
        ('sigma_xy', '0.113', '42.9%', '100.9%'),
        ('sigma_yz', '0.205', '33.7%', '43.1%'),
        ('sigma_xz', '2.603', '28.7%', '21.0%'),
        ('Pooled Frobenius norm (all six components)', '--', '62.4%', '29.8%'),
    ],
)
doc.add_paragraph(
    'The three large-magnitude normal stresses, which dominate the pooled '
    'Frobenius norm, all improved by roughly half. The two smallest-'
    'magnitude shear components (sigma_xy, sigma_yz) instead got worse in '
    'relative terms, while sigma_xz improved; given their tiny absolute '
    'magnitude and sign-changing behavior within the region, this is '
    'reported as-is rather than over-interpreted, and does not meaningfully '
    'move the pooled metric, which remains dominated by the normal '
    'stresses as confirmed above. Refining only the DEM background-'
    'integration quadrature in the groove region during training -- '
    'without changing the operator’s own mesh, node count, or element '
    'count at all -- roughly halves the regional-stress error, directly '
    'confirming the advisor’s hypothesis that the operator’s '
    'discretization and the DEM integration resolution had been '
    'conflated in the original setup.'
)
doc.add_paragraph(
    'Reporting this real, substantial, but partial result, the advisor’s '
    'reply was decisive: report the regional Cauchy-stress QoI, as '
    'measured (62.2-62.3% reduced to 29.7-29.9% by local integration '
    'refinement, still above the 5-10% target), as a documented '
    'limitation of this benchmark for the present report, rather than a '
    'fully resolved result or an item still being actively chased. '
    'Further work on it -- a stress/equilibrium-aware physics loss, a '
    'mixed displacement-stress formulation, PINO-type residual '
    'enforcement, a local or hierarchical representation with more '
    'capacity in stress-critical regions, data-driven stress supervision, '
    'a hybrid physics-informed-plus-supervised approach, or further study '
    'of separating the physics-integration resolution from the '
    'operator’s own resolution -- is explicitly deferred to a planned '
    'follow-up publication, not attempted in this report.'
)

# ===========================================================================
# 11.9 Accuracy-Matched FEM/VINO Comparison and Break-Even Analysis
# ===========================================================================
doc.add_heading(
    '11.9 Accuracy-Matched FEM/VINO Comparison and Break-Even Analysis',
    level=2)
doc.add_paragraph(
    'Per the advisor’s request to finish this report, the accuracy-'
    'matched FEM/VINO comparison and break-even analysis already built '
    'for B1/B2 (8.6) was completed for the three QoIs reported above as '
    'well-reproduced (displacement, energy, reaction); regional stress is '
    'excluded from this comparison, reported separately as the '
    'limitation in 11.8.'
)
doc.add_paragraph(
    'A genuinely apples-to-apples accuracy-matched comparison requires '
    'the operator’s own error to be measured against an INDEPENDENT fine '
    'reference, not against the same 6,840-element mesh it trained on: '
    'that mesh itself still carries real discretization error against a '
    'converged solution (this project’s own required-resolution study '
    'needs 9,464 elements for 1% displacement accuracy). Ten held-out '
    'samples were therefore re-solved at the already-established '
    '43,400-element fine evaluation mesh (interpolating each sample’s '
    'own material field onto the finer mesh, a genuinely new nonlinear '
    'FEM solve, clean Newton convergence throughout, force-residual '
    'around 1e-14), and the SAME trained network was queried zero-shot at '
    'that finer mesh’s own node positions:'
)
add_table(
    ['QoI', 'Operator error vs. its own 6,840-element training mesh (not apples-to-apples)', 'Operator error vs. the independent 43,400-element fine reference (correct comparison)'],
    [
        ('Displacement (combined)', '1.88%', '2.03%'),
        ('Energy (pooled RMS)', '0.30%', '2.30%'),
        ('Reaction moment (pooled RMS)', '2.67%', '2.22%'),
    ],
)
doc.add_paragraph(
    'Displacement accuracy barely changed (the 6,840-element mesh was '
    'already close to converged for this quantity); the energy comparison '
    'moved more, since the 6,840-element FEM’s own energy value is only '
    'within about 0.8% of the fine reference per the required-resolution '
    'table, so part of the earlier 0.30% number was being measured '
    'against a not-fully-converged reference. Both numbers are now '
    'genuinely comparable to the cheap-FEM errors used in the break-even '
    'analysis below, which are likewise measured against the same kind '
    'of independent fine reference.'
)
doc.add_paragraph(
    'Two separate break-even comparisons are reported together, '
    'following the same convention as 8.6: resolution-matched (operator '
    'and FEM both at the operator’s own 6,840-element resolution) and '
    'accuracy-matched (the cheapest FEM mesh whose own error against the '
    'independent fine reference is at least as good as the operator’s).'
)
add_table(
    ['Comparison', 'Operator error', 'Matched FEM mesh', 'FEM time per sample', 'Speed-up', 'Break-even (samples)'],
    [
        ('Resolution-matched (both at 6,840 elements)', '--', '6,840 elements', '5,060.4 ms', '800.7x', '1,381'),
        ('Accuracy-matched: displacement', '2.03%', '3,240 elements', '4,083.9 ms', '646.2x', '1,712'),
        ('Accuracy-matched: energy', '2.30%', '600 elements', '3,723.1 ms', '589.1x', '1,878'),
        ('Accuracy-matched: reaction moment', '2.22%', '9,464 elements', '4,672.1 ms', '739.3x', '1,496'),
    ],
)
doc.add_paragraph(
    'The one-off training cost (6,980.3 s, under 2 GPU-hours) is repaid '
    'within fewer than 1,900 solves for every one of the three QoIs, even '
    'judged against the cheapest FEM mesh that matches the operator’s own '
    'accuracy rather than the full production mesh -- a small fraction of '
    'the number of solves typically needed in any real deployment or '
    'dataset-generation setting. This completes the accuracy-matched '
    'comparison and break-even analysis requested for this benchmark.'
)

# ===========================================================================
# 11.10 Status
# ===========================================================================
doc.add_heading('11.10 Status', level=2)
doc.add_paragraph(
    'Dataset generation, operator training, and the full FEM-versus-'
    'operator comparison on displacement, energy, reaction, and regional '
    'Cauchy-stress QoIs (the four numerical steps 11.5 identified as '
    'remaining) are now complete. Displacement, energy, and reaction are '
    'well reproduced, with a completed accuracy-matched comparison and '
    'break-even analysis (11.9). Regional Cauchy-stress is reported as a '
    'documented limitation (11.8): local integration refinement, applied '
    'on both the evaluation and training sides, roughly halves the error '
    '(62.2-62.3% to 29.7-29.9%) without changing the operator’s own '
    'discretization, confirming the gap was partly, but not entirely, a '
    'DEM-integration-resolution artifact; the remaining gap above the '
    '5-10% target is left as an explicitly deferred item for a planned '
    'follow-up publication, per the advisor’s own decision, rather than '
    'pursued further in this report.'
)

# ===========================================================================
# Cross-links: Executive Summary feedback-tracking table (row 9) and
# Section 10 "remaining items" bullet -- the same consistency-audit
# pattern this project has twice already had to retrofit (rounds 15-17),
# applied proactively here instead of waiting to be caught again.
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
row9.cells[2].text = (
    'Dataset generation, operator training, and the full FEM-versus-'
    'operator comparison are now complete for B3 (11.6-11.9): '
    'displacement (2.03%), energy (2.30%), and reaction (2.22%) are well '
    'reproduced against an independent fine reference, with a completed '
    'accuracy-matched break-even analysis (589-739x speed-up, break-even '
    'under 1,900 solves). Regional Cauchy stress, after local integration '
    'refinement, is reported as a documented limitation (62.2% reduced to '
    '29.8%, still above the 5-10% target) -- see 11.8 and 11.10.'
)

paras = doc.paragraphs
last_item_idx = None
for i, p in enumerate(paras):
    if (p.style.name == 'List Paragraph'
            and 'Dataset generation and operator training for it have not started' in p.text):
        last_item_idx = i
if last_item_idx is None:
    for i, p in enumerate(paras):
        if (p.style.name == 'List Paragraph'
                and 'Section 11 for the full mesh-convergence results' in p.text):
            last_item_idx = i
assert last_item_idx is not None, 'could not find the round-17 "remaining items" bullet to update'
paras[last_item_idx].text = (
    'The new, harder, more realistic 3D example (item 4) is now complete '
    'through training and evaluation (11.6-11.9): displacement, energy, '
    'and reaction are well reproduced with a completed accuracy-matched '
    'break-even analysis; regional Cauchy stress is reported as a '
    'documented limitation after local integration refinement roughly '
    'halved its error, with further improvement explicitly deferred to a '
    'planned follow-up publication per the advisor’s own decision (11.10).'
)

doc.save(DST)
print('Saved', DST)

check = Document(DST)
print('paragraphs:', len(check.paragraphs))
print('tables:', len(check.tables))
print('table[0] row 9 col 2:', check.tables[0].rows[-1].cells[2].text[:80] + '...')
