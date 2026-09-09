"""Item #12: physics-informed vs. data-driven training figure (Tables
21/21a), built from the numbers already published in the real Report
docx -- no separate JSON source exists for these, so the report's own
already-verified numbers are the only available source, extracted
directly via docx_table_map.py rather than retyped by hand.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from docx_table_map import build_table_map, get_rows
from plot_style import PRIMARY, SECONDARY, add_bar_labels

DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

doc, tmap = build_table_map(REPORT)
rows21 = get_rows(tmap['Table 21'])
rows21a = get_rows(tmap['Table 21a'])

by21 = {r['Training loss']: r for r in rows21}
OPTS = ['Adam, lr 2×10⁻³', 'AdamW lr 10⁻³ + OneCycleLR']
OPT_SHORT = ['Adam', 'AdamW+OneCycleLR']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)

ax = axes[0]
x = np.arange(len(OPTS))
width = 0.35
pi_err = [float(by21['Physics-informed (energy)'][o]) * 100 for o in OPTS]
dd_err = [float(by21['Data-driven (relative L2 to FEM)'][o]) * 100 for o in OPTS]
bars_pi = ax.bar(x - width / 2, pi_err, width, label='Physics-informed', color=PRIMARY)
bars_dd = ax.bar(x + width / 2, dd_err, width, label='Data-driven', color=SECONDARY)
add_bar_labels(ax, bars_pi, fmt='{:.2f}%', fontsize=8)
add_bar_labels(ax, bars_dd, fmt='{:.2f}%', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(OPT_SHORT, fontsize=9)
ax.set_ylabel('Held-out relative L2 error (%)')
ax.set_title('Validation error by training loss (B1 × NH)')
ax.grid(True, axis='y', alpha=0.25)
ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=2)

ax = axes[1]
by21a = {r['Optimiser']: r for r in rows21a}
pi_total = [float(by21a[o]['PI total (s)'].replace(',', '')) for o in OPTS]
dd_total = [float(by21a[o]['DD-NO total (s)'].replace(',', '')) for o in OPTS]
bars_pit = ax.bar(x - width / 2, pi_total, width, label='Physics-informed total', color=PRIMARY)
bars_ddt = ax.bar(x + width / 2, dd_total, width,
                   label='Data-driven total\n(incl. label generation)', color=SECONDARY)
add_bar_labels(ax, bars_pit, fmt='{:.0f}s', fontsize=8)
add_bar_labels(ax, bars_ddt, fmt='{:.0f}s', fontsize=8)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(OPT_SHORT, fontsize=9)
ax.set_ylabel('Total cost before first inference (s)')
ax.set_title('Total cost before first inference')
ax.grid(True, axis='y', alpha=0.25)
ax.set_ylim(top=ax.get_ylim()[1] * 4)
ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=1)

fig.suptitle('Physics-informed vs. data-driven training: accuracy and total cost', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out_path = os.path.join(OUT, 'fig_pi_vs_dd.png')
fig.savefig(out_path)
print('Saved', out_path)
