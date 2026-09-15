"""Builds a standalone status-update email/memo for Timon, per Omar's
explicit request: full detail on (1) the round-8 point 6 fix (richer
MMS family + energy norm, now in the Report/Summary), (2) everything
currently in progress, and (3) everything queued after that -- all in
ONE file, formal address ("Dear Professor Rabczuk", never his first
name alone), no admission/concession language.
"""
import os

from docx import Document
from docx.shared import Pt

DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
OUT = os.path.join(DELIV, 'Timon_Status_Update_2026-09-15.docx')

doc = Document()


def title(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(15)
    return p


def draft_note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10)
    return p


def para(text):
    return doc.add_paragraph(text)


def bold_lead(lead, rest):
    p = doc.add_paragraph()
    r1 = p.add_run(lead)
    r1.bold = True
    p.add_run(rest)
    return p


def section(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    return p


title('Status update — manufactured-solution verification, retraining in progress')
draft_note('DRAFT — do not send without Omar’s own review.')
para('')
para('Dear Professor Rabczuk,')
para('')
para('An update covering the point you raised, the work currently running, and what '
     'follows once it completes.')
para('')

section('1. Round-8, point 6 — richer manufactured-solution family and the energy norm')
para('')
para('Tables 22c, 22d, and 22e have been added to both the Report and the Summary, '
     'covering Neo-Hookean, Mooney-Rivlin, and Arruda-Boyce respectively. Each is '
     'evaluated against a manufactured field that combines three sine/cosine spatial '
     'modes per displacement component, rather than a single mode varied only in '
     'amplitude:')
para('')
p = doc.add_paragraph()
p.add_run('u_x = 0.05·[sin(πx)sin(πy) + 0.5·sin(2πx)sin(πy) + '
          '0.3·sin(πx)sin(3πy)]').italic = True
p2 = doc.add_paragraph()
p2.add_run('u_y = 0.035·[sin(πx)sin(πy) − 0.4·sin(3πx)sin(2πy) '
           '+ 0.2·sin(2πx)sin(2πy)]').italic = True
para('')
para('Same four resolutions and both element orders (Q4, Q9) as Tables 22/22a/22b. Each '
     'row reports the internal-energy value (the original metric) and the energy norm '
     'side by side, so the two can be compared directly:')
para('')
p3 = doc.add_paragraph()
p3.add_run('energy norm = sqrt( ∫ grad(e):C(F*):grad(e) dV / ∫ grad(u*):C(F*):'
           'grad(u*) dV )').italic = True
para('with e = u_h − u*, and C the fourth-order tangent modulus at the exact solution '
     '— the same tangent/incremental norm used elsewhere in the Report (Table 6a), '
     'here expressed as a direct quadrature integral against the continuous exact field.')
para('')
para('Table 23b gives the corresponding convergence rates for all three materials. The '
     'energy-norm rate now matches the H1 semi-norm’s own theoretical rate (1 at Q4, '
     '2 at Q9) rather than double it, consistent with Céa’s lemma for a norm rather '
     'than a value. All 24 rows land on their theoretical rate.')
para('')
para('Figure 28a (Report) / 27a (Summary) plots the same convergence data.')
para('')
bold_lead('Representative numbers: ',
          'Neo-Hookean, Q4, N=33 — L2 = 2.011e-03, H1 semi-norm = 5.392e-02, energy '
          'value = 2.257e-03, energy norm = 4.803e-02. The energy value is consistently '
          'one to two orders of magnitude tighter than every other column at matched '
          'rows; the energy norm tracks the H1 semi-norm closely at every resolution, as '
          'expected for a norm rather than a scalar comparison.')
para('')
para('Tables 22/22a/22b and 24 were left unchanged: Table 24’s operator-vs-Q4 ratios '
     'are computed directly from Table 22’s own numbers, and revising the '
     'manufactured field there would require re-running that separate study, which is '
     'outside the scope of this point.')
para('')
para('The underlying richer-family solves were already computed and verified earlier; '
     'this update wires that data into the Report and Summary themselves. Both updated '
     'documents are attached.')
para('')

section('2. Currently in progress')
para('')
bold_lead('B1 × Mooney-Rivlin and B1 × Arruda-Boyce, multi-resolution '
          'retraining. ',
          'Both checkpoints (originally trained on N=21,33 only) are being retrained on '
          'the wider resolution set (N=21,33,101,201) that corrected B1 × '
          'Neo-Hookean’s own accuracy at N=1401. Both are training now and have '
          'already reached validation results better than their respective originals. '
          'Once each finishes, a direct accuracy comparison against the original '
          'checkpoint runs automatically at sixteen resolutions from N=13 to N=1401.')
para('')
bold_lead('B1 × Neo-Hookean, direct training at N=1401 (ablation). ',
          'A separate checkpoint trained directly at the target resolution, as a '
          'comparison point against the zero-shot approach (train on N=21,33,101,201, '
          'evaluate at N=1401 without retraining). Ground-truth data generation at '
          'N=1401 is in progress; training follows once it completes.')
para('')

section('3. Queued next')
para('')
bold_lead('B2 × Neo-Hookean and B2 × Mooney-Rivlin, multi-resolution '
          'retraining. ',
          'Same protocol as the B1 retraining above, extended to the B2 geometry. Not '
          'yet started; will begin once the current jobs finish.')
para('')
bold_lead('Complex-geometry example (point 4). ',
          'To proceed once the checkpoints above are finalized, per your own request '
          'that this follow the accuracy/break-even work rather than precede it.')
para('')
bold_lead('Final consolidation. ',
          'Once all of the above are complete, the results will be folded into the '
          'Report and Summary and a final, complete reply covering every open point '
          'will follow.')
para('')
para('Best regards,')
para('Omar')

doc.save(OUT)
print('Saved', OUT)
