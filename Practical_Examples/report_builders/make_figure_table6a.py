"""Item #12: first figure, converting Table 6a's own numbers (L2/H1/energy
relative error vs. N, B1 x Neo-Hookean) into a log-log convergence plot,
in the style of Timon's own VINO paper (Fig. 3b: log-log, one line per
series, resolution on x). Table 6a itself is left completely untouched --
this figure is an ADDITION alongside it, not a replacement (VINO's own
Table 1 + Fig. 3a live side by side the same way).

Generated first, standalone, for a style check before batch-producing
the rest of item #12's figures.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PF = 'omar_pfem/highdof_stress_qoi_results'
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

d1 = json.load(open(os.path.join(PF, 'high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json')))
d2 = json.load(open(os.path.join(PF, 'high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json')))
rows = d1['orders']['Q4']['rows'] + [r for r in d2['orders']['Q4']['rows'] if r['N'] in (701, 1001, 1401)]
rows.sort(key=lambda r: r['N'])

N = np.array([r['N'] for r in rows], dtype=float)
l2 = np.array([r['l2_rel_error'] for r in rows])
h1 = np.array([r['h1_semi_rel_error'] for r in rows])
en = np.array([r['energy_rel_error'] for r in rows])


def fit_rate(N, err):
    p = np.polyfit(np.log(N), np.log(err), 1)
    return -p[0]


rates = {'L2': fit_rate(N, l2), 'H1 semi-norm': fit_rate(N, h1), 'Energy': fit_rate(N, en)}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.axisbelow': True,
})

fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=200)

series = [
    ('L2', l2, '#1f77b4', 'o'),
    ('H1 semi-norm', h1, '#ff7f0e', 's'),
    ('Energy', en, '#2ca02c', '^'),
]
for label, y, color, marker in series:
    ax.loglog(N, y, marker=marker, color=color, linewidth=1.8, markersize=6,
               label=f'{label} (fitted rate p={rates[label]:.2f})')

ax.set_xlabel('Mesh resolution N (B1, Q4 elements)')
ax.set_ylabel('Relative error')
ax.set_title('Convergence of L2, H1 semi-norm, and energy-norm error with mesh resolution\n(B1, Neo-Hookean; Table 6a)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()

out_path = os.path.join(OUT, 'fig_table6a_convergence.png')
fig.savefig(out_path)
print('Saved', out_path)
