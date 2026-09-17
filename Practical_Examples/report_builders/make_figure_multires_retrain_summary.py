"""Round-10 Figure D: old-vs-new disp_rel_L2 at N=1401, across every
(geometry, material) case whose multi-resolution retrain (N=21,33
-> N=21,33,101,201) has actually completed as of this figure's build
date. Mirrors Round-10 Figure A/B/C's own convention of NOT taking a
main-sequence figure number, since this is a direct, same-subsection
extension of that round-10 material, not a new numbered section.

Numbers transcribed directly from PROJECT_STATUS.md's own real-GPU
entries (B1xNeo-Hookean, B1xMooney-Rivlin, B1xArruda-Boyce) and real GPU
logs pasted directly into the working conversation (B2xMooney-Rivlin,
B2xNeo-Hookean) -- see add_multires_retrain_extension_to_report.py for
the exact same numbers used in the accompanying tables, so the figure
and the tables cannot silently drift apart.

B2xNeo-Hookean ADDED 2026-09-17 once its own retrain finished (was
previously excluded as still mid-training) -- the fifth and last
completed case (B2xArruda-Boyce has no retrain planned).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_style import add_bar_labels

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

CASES = ['B1 x Neo-Hookean', 'B1 x Mooney-Rivlin', 'B1 x Arruda-Boyce',
         'B2 x Mooney-Rivlin', 'B2 x Neo-Hookean']
OLD = [44.65, 39.20, 45.62, 49.47, 46.37]
NEW = [5.85, 15.04, 25.05, 13.10, 36.05]

x = np.arange(len(CASES))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
bars_old = ax.bar(x - width / 2, OLD, width, label='Old checkpoint (N=21,33 only)', color='#C0392B')
bars_new = ax.bar(x + width / 2, NEW, width, label='New checkpoint (N=21,33,101,201)', color='#27AE60')

add_bar_labels(ax, bars_old, fmt='{:.1f}%', fontsize=9)
add_bar_labels(ax, bars_new, fmt='{:.1f}%', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(CASES, fontsize=9)
ax.set_ylabel('disp_rel_L2 at N=1401 (%)')
ax.set_title('Multi-resolution retrain fix at N=1401, every completed case')
ax.set_ylim(top=max(OLD) * 1.18)
ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout(rect=[0, 0.06, 1, 1])

out_path = os.path.join(OUT, 'fig_multires_retrain_summary.png')
fig.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
