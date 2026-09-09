"""Item #12: break-even figure (Table 10d), built from the numbers
already published in the real Report docx -- no separate JSON source
exists for this table, so the report's own already-verified numbers
are the only available source, extracted directly via
docx_table_map.py rather than retyped by hand.

Table 10d gives, per case, the number of new problem instances needed
before the operator's training cost is repaid: one column against the
CPU-FEM baseline (batch size 1 only), and four columns against the
GPU-native FEM solver at batch sizes 1/8/32/128 (identical to Table
10c, reused here rather than duplicated in a separate figure).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from docx_table_map import build_table_map, get_rows
from plot_style import NEUTRAL, HIGHLIGHT, GOOD, SECONDARY, PRIMARY, add_bar_labels

DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

doc, tmap = build_table_map(REPORT)
rows = get_rows(tmap['Table 10d'])

CASES = ['B1 × Neo-Hookean', 'B1 × Mooney-Rivlin', 'B1 × Arruda-Boyce',
          'B2 × Neo-Hookean', 'B2 × Mooney-Rivlin', 'B2 × Arruda-Boyce']
by_case = {r['Case']: r for r in rows}

COLS = [('vs. CPU FEM', 'CPU FEM (bs=1)'), ('vs. GPU FEM, bs=1', 'GPU FEM, bs=1'),
        ('bs=8', 'GPU FEM, bs=8'), ('bs=32', 'GPU FEM, bs=32'), ('bs=128', 'GPU FEM, bs=128')]
COLORS = [NEUTRAL, HIGHLIGHT, SECONDARY, GOOD, PRIMARY]

x = np.arange(len(CASES))
width = 0.15

fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
all_bars = []
for i, (key, label) in enumerate(COLS):
    vals = [float(by_case[c][key].replace(',', '')) for c in CASES]
    bars = ax.bar(x + (i - 2) * width, vals, width, label=label, color=COLORS[i])
    all_bars.append(bars)

# Labelling all 30 bars would be unreadable at this density -- label only
# the two "bookend" series (cheapest baseline, most demanding baseline),
# the two numbers a reader actually needs to see at a glance.
add_bar_labels(ax, all_bars[0], fmt='{:.0f}', fontsize=6.5)
add_bar_labels(ax, all_bars[-1], fmt='{:,.0f}', fontsize=6.5, rotation=90)

ax.set_yscale('log')
ax.set_ylim(top=ax.get_ylim()[1] * 3)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('×', 'x') for c in CASES], rotation=0, ha='center', fontsize=9)
ax.set_ylabel('Break-even (new problem instances)')
ax.set_title('Break-even against CPU-FEM and GPU-native FEM baselines')
ax.legend(frameon=False, fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.10))
ax.grid(True, axis='y', which='both', alpha=0.25)

fig.tight_layout(rect=[0, 0.06, 1, 1])

out_path = os.path.join(OUT, 'fig_breakeven.png')
fig.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
