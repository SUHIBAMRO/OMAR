"""Item #12: B2 fix-attempt history and final adopted results figure
(Tables 13/14), built from the numbers already published in the real
Report docx -- no separate JSON source exists for these, so the
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
rows13 = get_rows(tmap['Table 13'])
rows14 = get_rows(tmap['Table 14'])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)

ax = axes[0]
labels = ['Original\nbaseline', 'Force-consistency\nfix alone', '+ Loss-scale\nnormalization']
vals = [float(r['Mean rel. L2 error'].rstrip('%')) for r in rows13]
colors = ['#7f7f7f', '#d62728', '#2ca02c']
ax.bar(labels, vals, color=colors)
ax.axhline(9.0, color='black', linewidth=1, linestyle='--', label='9.00% target')
for i, v in enumerate(vals):
    ax.text(i, v + 1.5, f'{v:.2f}%', ha='center', fontsize=9)
ax.set_ylabel('Mean relative L2 error (%)')
ax.set_title('B2 fix-attempt history (Neo-Hookean)')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis='y', alpha=0.25)

ax = axes[1]
materials = [r['Case'].split(' × ')[1] for r in rows14]
vals14 = [float(r['Mean rel. L2 error'].rstrip('%')) for r in rows14]
colors14 = ['#1f77b4' if v < 9.0 else '#ff7f0e' for v in vals14]
ax.bar(materials, vals14, color=colors14)
ax.axhline(9.0, color='black', linewidth=1, linestyle='--', label='9.00% target')
for i, v in enumerate(vals14):
    ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=9)
ax.set_ylabel('Mean relative L2 error (%)')
ax.set_title('Final adopted B2 results')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis='y', alpha=0.25)

fig.suptitle('B2 (ring geometry) accuracy fix history and final results', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out_path = os.path.join(OUT, 'fig_b2_fix_history.png')
fig.savefig(out_path)
print('Saved', out_path)
