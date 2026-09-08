"""Item #12: training cost/efficiency figure (Tables 5/7/8), built from
the numbers already published in the real Report docx -- these three
tables have no separate JSON source anywhere in the repo, so the
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
rows5 = get_rows(tmap['Table 5'])
rows7 = get_rows(tmap['Table 7'])
rows8 = get_rows(tmap['Table 8'])

CASES = ['B1 × Neo-Hookean', 'B1 × Mooney-Rivlin', 'B1 × Arruda-Boyce',
          'B2 × Neo-Hookean', 'B2 × Mooney-Rivlin', 'B2 × Arruda-Boyce']
by_case5 = {r['Case']: r for r in rows5}
by_case7 = {r['Case']: r for r in rows7}
by_case8 = {r['Case']: r for r in rows8}
short = [c.replace('×', 'x') for c in CASES]

x = np.arange(len(CASES))
colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 3

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=200)

ax = axes[0]
wall = [float(by_case5[c]['Wall-clock (s)']) for c in CASES]
ax.bar(x, wall, color=colors)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Training wall-clock (s)')
ax.set_title('Training cost')
ax.grid(True, axis='y', alpha=0.25)

ax = axes[1]
speedup = [float(by_case7[c]['Speed-up'].rstrip('×')) for c in CASES]
ax.bar(x, speedup, color=colors)
ax.axhline(1.0, color='gray', linewidth=1, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Speed-up vs. native FEM (×)')
ax.set_title('Per-sample inference speed-up')
ax.grid(True, axis='y', alpha=0.25)

ax = axes[2]
mem = [float(by_case8[c]['Peak device / nvidia-smi-equiv. (MB)']) for c in CASES]
ax.bar(x, mem, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Peak GPU memory (MB)')
ax.set_title('Training memory footprint')
ax.grid(True, axis='y', alpha=0.25)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#1f77b4', label='B1'), Patch(color='#ff7f0e', label='B2')],
           loc='upper right', frameon=False, fontsize=9, ncol=2)

fig.suptitle('Training cost and efficiency across geometries and materials', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out_path = os.path.join(OUT, 'fig_training_cost.png')
fig.savefig(out_path)
print('Saved', out_path)
