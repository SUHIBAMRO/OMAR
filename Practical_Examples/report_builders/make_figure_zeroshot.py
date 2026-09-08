"""Item #12: zero-shot resolution-invariance figure (Table 12 revised
+ 12b/12c), built entirely from already-committed JSON result files --
no model, no new computation, runs locally.

B2's per-material files carry an explicit, documented caveat worth
respecting here rather than silently averaging over it: Neo-Hookean's
own first fixed-selection attempt was defective (see its own
'the_defect' field) and was corrected in a later run -- this script
reads the corrected 'B2_{material}_zeroshot_fixedselection.json' /
'B2_zeroshot_fixedselection.json' files (the ones whose own 'reading'
field confirms the corrected numbers), not any superseded ones sitting
alongside them.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PF = 'omar_pfem/point7a_results'
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

B1_FILES = {
    'Neo-Hookean': 'zeroshot_B1_neo_hookean.json',
    'Mooney-Rivlin': 'zeroshot_B1_mooney_rivlin.json',
    'Arruda-Boyce': 'zeroshot_B1_arruda_boyce.json',
}
B2_FILES = {
    'Neo-Hookean': 'B2_zeroshot_fixedselection.json',
    'Mooney-Rivlin': 'B2_mooney_rivlin_zeroshot_fixedselection.json',
    'Arruda-Boyce': 'B2_arruda_boyce_zeroshot_fixedselection.json',
}
COLORS = {'Neo-Hookean': '#1f77b4', 'Mooney-Rivlin': '#ff7f0e', 'Arruda-Boyce': '#2ca02c'}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)

for ax, geometry, files in [(axes[0], 'B1', B1_FILES), (axes[1], 'B2', B2_FILES)]:
    for material, fname in files.items():
        d = json.load(open(os.path.join(PF, fname)))
        rows = sorted(d['rows'], key=lambda r: r['N'])
        N = [r['N'] for r in rows]
        err = [r['mean_rel_L2_vs_fine_reference'] * 100 for r in rows]
        ax.semilogy(N, err, marker='o', color=COLORS[material], linewidth=1.6,
                     markersize=5, label=material)
    ax.set_xlabel('N (test resolution)')
    ax.set_ylabel('Mean relative L2 error (%)')
    ax.set_title(geometry)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which='both', alpha=0.25)

fig.suptitle('Zero-shot resolution invariance (one checkpoint, seven unseen resolutions)',
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out_path = os.path.join(OUT, 'fig_zeroshot_resolution.png')
fig.savefig(out_path)
print('Saved', out_path)
