"""Draft figure for Table 18-R10k (peak-stress QoI at N=1401, five
cases beyond B1xNeo-Hookean) -- for review before adding to the
Report/Summary. Grouped bars: disp_rel_L2 next to peak_stress_rel_err
per case, so the "B1 structurally worse on peak stress than B2"
pattern and the "B2xArruda-Boyce best on peak stress despite failing
break-even" point are both visible at a glance.

Checkpoint-version caveat (already stated in the table's own text, not
repeated visually here): B1xMooney-Rivlin, B1xArruda-Boyce and
B2xMooney-Rivlin reflect their ORIGINAL pre-retrain checkpoints.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_style import add_bar_labels

# Two consistent colors for the QoI being compared -- teal/purple pair,
# distinct from the direct-N1401 ablation figure's own palette so the two
# figures are never visually confused for showing the same comparison.
L2_COLOR = '#3B7EA1'
PEAK_STRESS_COLOR = '#D4A017'

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

CASES = ['B1 x\nMooney-Rivlin', 'B1 x\nArruda-Boyce', 'B2 x\nNeo-Hookean',
         'B2 x\nMooney-Rivlin', 'B2 x\nArruda-Boyce']
DISP_L2 = [39.20, 45.62, 46.37, 49.47, 39.73]
PEAK_STRESS = [78.05, 78.25, 49.48, 48.13, 39.97]

x = np.arange(len(CASES))
width = 0.35

fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=200)
bars0 = ax.bar(x - width / 2, DISP_L2, width, label='disp_rel_L2', color=L2_COLOR)
bars1 = ax.bar(x + width / 2, PEAK_STRESS, width, label='peak_stress_rel_err', color=PEAK_STRESS_COLOR)

add_bar_labels(ax, bars0, fmt='{:.1f}', fontsize=8)
add_bar_labels(ax, bars1, fmt='{:.1f}', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(CASES, fontsize=9)
ax.set_ylabel('Relative error (%)')
ax.set_title('Peak-stress QoI at N=1401 vs. displacement error, five cases')
ax.set_ylim(top=max(PEAK_STRESS + DISP_L2) * 1.2)
ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout(rect=[0, 0.06, 1, 1])

out_path = os.path.join(OUT, 'fig_peak_stress_qoi.png')
fig.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
