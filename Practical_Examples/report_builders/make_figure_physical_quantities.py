"""Item #12: displacement/stress/reaction-force error figure
(Tables 15/16/17), built entirely from already-committed JSON result
files (point5_results/physical_quantities_*.json) -- no model, no new
computation, runs locally.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PF = 'omar_pfem/point5_results'
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

CASES = [
    ('B1', 'neo_hookean', 'B1 x NH'), ('B1', 'mooney_rivlin', 'B1 x MR'),
    ('B1', 'arruda_boyce', 'B1 x AB'), ('B2', 'neo_hookean', 'B2 x NH'),
    ('B2', 'mooney_rivlin', 'B2 x MR'), ('B2', 'arruda_boyce', 'B2 x AB'),
]

data = {}
for geometry, material, label in CASES:
    fname = f'physical_quantities_{geometry}_{material}.json'
    d = json.load(open(os.path.join(PF, fname)))
    data[label] = d['metrics']

x = np.arange(len(CASES))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=200)

ax = axes[0]
disp = [data[lbl]['disp_rel_L2']['mean'] * 100 for _, _, lbl in CASES]
disp_std = [data[lbl]['disp_rel_L2']['std'] * 100 for _, _, lbl in CASES]
ax.bar(x, disp, yerr=disp_std, capsize=3, color='#1f77b4')
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, _, lbl in CASES], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Relative error (%)')
ax.set_title('Displacement error')
ax.grid(True, axis='y', alpha=0.25)

ax = axes[1]
stress = [data[lbl]['P_rel_L2']['mean'] * 100 for _, _, lbl in CASES]
stress_std = [data[lbl]['P_rel_L2']['std'] * 100 for _, _, lbl in CASES]
ax.bar(x, stress, yerr=stress_std, capsize=3, color='#ff7f0e')
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, _, lbl in CASES], rotation=45, ha='right', fontsize=8)
ax.set_title('First Piola-Kirchhoff stress error')
ax.grid(True, axis='y', alpha=0.25)

def reaction_mean_std(m):
    # B1 has one fixed edge (reaction_resultant_rel_err); B2 has two
    # (edge0/edge1, its two symmetry boundaries) -- averaged here for a
    # single comparable bar per case, clearly noted rather than silently
    # picking one edge and dropping the other.
    if 'reaction_resultant_rel_err' in m:
        return m['reaction_resultant_rel_err']['mean'], m['reaction_resultant_rel_err']['std']
    means = [m[f'reaction_resultant_rel_err_edge{i}']['mean'] for i in (0, 1)]
    stds = [m[f'reaction_resultant_rel_err_edge{i}']['std'] for i in (0, 1)]
    return float(np.mean(means)), float(np.mean(stds))


reaction, reaction_std = zip(*[reaction_mean_std(data[lbl]) for _, _, lbl in CASES])
reaction = [v * 100 for v in reaction]
reaction_std = [v * 100 for v in reaction_std]
ax = axes[2]
ax.bar(x, reaction, yerr=reaction_std, capsize=3, color='#2ca02c')
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, _, lbl in CASES], rotation=45, ha='right', fontsize=8)
ax.set_title('Reaction-force error')
ax.grid(True, axis='y', alpha=0.25)

fig.suptitle('Trained-operator error: displacement, stress, and reaction force', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT, 'fig_physical_quantities.png')
fig.savefig(out_path)
print('Saved', out_path)
