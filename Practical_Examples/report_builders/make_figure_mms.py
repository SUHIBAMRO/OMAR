"""Item #12: MMS convergence-rate figures (Tables 22/22a/22b + 23/23a),
built entirely from already-committed JSON result files -- no model to
load, no new solving, so this runs locally with no Colab/GPU needed at
all. Log-log convergence plots, one row per material, Q4/Q9 side by
side, matching the log-log style seen in Timon's own group's papers
(VINO Fig. 3b).

Clean, publication-style titles only: no internal implementation notes
(item numbers, "no retraining", etc.) inside the image itself -- those
belong in the paper's own caption text, not the figure.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PF = 'omar_pfem/point9_results'
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

MATERIALS = [
    ('neo_hookean', 'Neo-Hookean', 'mms_B1_neo_hookean.json'),
    ('mooney_rivlin', 'Mooney-Rivlin', 'mms_B1_mooney_rivlin.json'),
    ('arruda_boyce', 'Arruda-Boyce', 'mms_B1_arruda_boyce.json'),
]
SERIES = [
    ('L2_rel', 'L2', '#1f77b4', 'o'),
    ('H1_semi_rel', 'H1 semi-norm', '#ff7f0e', 's'),
    ('stress_rel_L2', 'Stress', '#2ca02c', '^'),
    ('energy_norm_rel', 'Energy norm', '#d62728', 'd'),
]

fig, axes = plt.subplots(3, 2, figsize=(9, 10.5), dpi=200)

for mi, (mat_key, mat_label, fname) in enumerate(MATERIALS):
    d = json.load(open(os.path.join(PF, fname)))
    for oi, order in enumerate(('Q4', 'Q9')):
        ax = axes[mi, oi]
        rows = [r for r in d['rows'] if r['order'] == order]
        rows.sort(key=lambda r: r['h'])
        h = np.array([r['h'] for r in rows])
        for key, label, color, marker in SERIES:
            y = np.array([r[key] for r in rows])
            rate = d['convergence_rates'][order][
                {'L2_rel': 'L2', 'H1_semi_rel': 'H1_semi', 'stress_rel_L2': 'stress',
                 'energy_norm_rel': 'energy_norm'}[key]]['rate']
            ax.loglog(h, y, marker=marker, color=color, linewidth=1.6, markersize=5,
                       label=f'{label} (p={rate:.2f})')
        ax.set_title(f'{mat_label}, {order}', fontsize=10)
        ax.set_xlabel('h')
        if oi == 0:
            ax.set_ylabel('Relative error')
        ax.legend(frameon=False, fontsize=7, loc='upper left')
        ax.grid(True, which='both', alpha=0.25)

fig.suptitle('Method of manufactured solutions: convergence rates', fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])

out_path = os.path.join(OUT, 'fig_mms_convergence.png')
fig.savefig(out_path)
print('Saved', out_path)
