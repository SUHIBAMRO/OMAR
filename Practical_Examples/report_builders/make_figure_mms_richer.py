"""Round-8 point 6 (Timon): the MMS verification (Tables 22/22a/22b,
Figure 27/28) used essentially one spatial mode (u* = alpha*sin(pi x)
sin(pi y) family, varied only through alpha/beta amplitudes) and
reported the scalar internal-energy VALUE error (which superconverges
at double the true rate -- not a meaningful discretization-error
indicator). Timon asked for a richer family combining several spatial
sine/cosine modes, and for the actual ENERGY NORM of the error instead.

Both already exist in code (mms_study.py's MODES_RICHER and
compute_errors' energy_norm_rel) and were verified correct on a real
CPU run for all three materials back on 2026-09-09
(point9_results/mms_richer_B1_*.json, rate_check "as expected" for
both Q4 and Q9) -- but that verification only ever reached a side note
in the Summary's "Response to round-8" text, never the Report itself,
which is why Timon still sees this as open: the Report's own Tables
22/22a/22b and Figure 27/28 still show the single-mode/energy-value
study. This script builds the companion figure for the NEW richer-
family tables (22c/22d/22e, rate table 23b) being added to the Report
by add_richer_mms_to_report.py, using the exact same plotting
convention as make_figure_mms.py (so the two figures are visually
consistent) but pointing at the richer-family JSON files and plotting
energy_norm_rel, which is what the original figure always plotted too
-- only the DATA SOURCE changes here, not the metric choice.
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
    ('neo_hookean', 'Neo-Hookean', 'mms_richer_B1_neo_hookean.json'),
    ('mooney_rivlin', 'Mooney-Rivlin', 'mms_richer_B1_mooney_rivlin.json'),
    ('arruda_boyce', 'Arruda-Boyce', 'mms_richer_B1_arruda_boyce.json'),
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
    assert d['richer_family'] is True
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

fig.suptitle('Method of manufactured solutions, richer multi-mode family: '
             'convergence rates', fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])

out_path = os.path.join(OUT, 'fig_mms_richer_convergence.png')
fig.savefig(out_path)
print('Saved', out_path)
