"""Draft figure for Table 18-R10j (direct-N1401 training ablation vs.
multi-resolution zero-shot, B1xNeo-Hookean) -- for review before adding
to the Report/Summary. Two panels: training cost (hours) and accuracy
(disp_rel_L2 %), since the headline finding is "cheaper to train but
much less accurate," best shown as two side-by-side comparisons rather
than one dual-axis chart.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_style import PRIMARY, SECONDARY, add_bar_labels

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

LABELS = ['Direct\n@N=1401', 'Multi-res\n(zero-shot)']
TRAIN_HOURS = [8.00, 11.63]
ACCURACY_PCT = [36.65, 5.85]

fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=200)

bars0 = axes[0].bar(LABELS, TRAIN_HOURS, color=[SECONDARY, PRIMARY], width=0.55)
add_bar_labels(axes[0], bars0, fmt='{:.2f}h')
axes[0].set_ylabel('Training wall-clock (hours)')
axes[0].set_title('Training cost')
axes[0].set_ylim(top=max(TRAIN_HOURS) * 1.25)
axes[0].grid(True, axis='y', alpha=0.25)

bars1 = axes[1].bar(LABELS, ACCURACY_PCT, color=[SECONDARY, PRIMARY], width=0.55)
add_bar_labels(axes[1], bars1, fmt='{:.1f}%')
axes[1].set_ylabel('disp_rel_L2 @ N=1401 (%)')
axes[1].set_title('Accuracy (lower is better)')
axes[1].set_ylim(top=max(ACCURACY_PCT) * 1.25)
axes[1].grid(True, axis='y', alpha=0.25)

fig.suptitle('Direct-N1401 training vs. multi-resolution zero-shot, B1 x Neo-Hookean')
fig.tight_layout()

out_path = os.path.join(OUT, 'fig_direct_n1401_ablation.png')
fig.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
