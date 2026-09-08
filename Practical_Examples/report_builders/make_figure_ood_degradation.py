"""Item #12: in-distribution vs. out-of-distribution error figure
(Table 11), built from the numbers already published in the real
Report docx -- no separate JSON source exists for this table, so the
report's own already-verified numbers are the only available source,
extracted directly via docx_table_map.py rather than retyped by hand.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from docx_table_map import build_table_map, get_rows

DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

doc, tmap = build_table_map(REPORT)
rows = {r['Case']: r for r in get_rows(tmap['Table 11'])}

CASES = ['B1 × Neo-Hookean', 'B1 × Mooney-Rivlin', 'B1 × Arruda-Boyce',
          'B2 × Neo-Hookean', 'B2 × Mooney-Rivlin', 'B2 × Arruda-Boyce']

x = np.arange(len(CASES))
width = 0.35
id_err = [float(rows[c]['In-distribution val. err.']) * 100 for c in CASES]
ood_err = [float(rows[c]['OOD val. err.']) * 100 for c in CASES]

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
ax.bar(x - width / 2, id_err, width, label='In-distribution', color='#1f77b4')
ax.bar(x + width / 2, ood_err, width, label='Out-of-distribution', color='#d62728')
for xi, case in zip(x, CASES):
    deg = rows[case]['Degradation factor (OOD / ID)']
    ax.text(xi, max(id_err[xi], ood_err[xi]) + 1.5, deg, ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('×', 'x') for c in CASES], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Validation error (%)')
ax.set_title('In-distribution vs. out-of-distribution generalization error')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()

out_path = os.path.join(OUT, 'fig_ood_degradation.png')
fig.savefig(out_path)
print('Saved', out_path)
