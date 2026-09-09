"""Item #12: operator-vs-FEM accuracy and speed-up across mesh
resolutions (Tables 18/18a-e), built from the numbers already
published in the real Report docx -- no separate JSON source exists
for these, so the report's own already-verified numbers are the only
available source, extracted directly via docx_table_map.py rather than
retyped by hand.

Caption text (not just header text, which is identical across all six
tables) was used to confirm the geometry/material mapping: Table 18=B1
NH, 18a=B1 MR, 18b=B1 AB, 18c=B2 NH, 18d=B2 MR, 18e=B2 AB.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from docx_table_map import build_table_map, get_rows
from plot_style import MATERIAL_COLOR

DELIV = '/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad/deliverables'
REPORT = os.path.join(DELIV, 'PFEM_Transolver_Report_updated_2026-09-08.docx')
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

doc, tmap = build_table_map(REPORT)

CASES = [
    ('Table 18', 'B1', 'Neo-Hookean'), ('Table 18a', 'B1', 'Mooney-Rivlin'),
    ('Table 18b', 'B1', 'Arruda-Boyce'), ('Table 18c', 'B2', 'Neo-Hookean'),
    ('Table 18d', 'B2', 'Mooney-Rivlin'), ('Table 18e', 'B2', 'Arruda-Boyce'),
]
data = {}
for label, geometry, material in CASES:
    rows = sorted(get_rows(tmap[label]), key=lambda r: int(r['N']))
    data[(geometry, material)] = rows

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=200)

for ax, geometry in zip(axes[:2], ('B1', 'B2')):
    for material, color in MATERIAL_COLOR.items():
        rows = data[(geometry, material)]
        N = [int(r['N']) for r in rows]
        fem = [float(r['FEM rel. L2 (%)']) for r in rows]
        op = [float(r['Operator rel. L2 (%)']) for r in rows]
        ax.loglog(N, fem, '-', marker='o', color=color, linewidth=1.6, markersize=5,
                    label=f'{material} (FEM)')
        ax.loglog(N, op, '--', marker='s', color=color, linewidth=1.6, markersize=5,
                    label=f'{material} (operator)')
    ax.set_xlabel('N')
    ax.set_ylabel('Relative L2 error (%)')
    ax.set_title(f'{geometry}: FEM vs. operator accuracy')
    ax.legend(frameon=False, fontsize=6.5, ncol=1)
    ax.grid(True, which='both', alpha=0.25)

ax = axes[2]
for geometry, style, marker in [('B1', '-', 'o'), ('B2', '--', 's')]:
    for material, color in MATERIAL_COLOR.items():
        rows = data[(geometry, material)]
        N = [int(r['N']) for r in rows]
        speedup = [float(r['Speed-up'].rstrip('×').replace(',', '')) for r in rows]
        ax.loglog(N, speedup, style, marker=marker, color=color, linewidth=1.6, markersize=5,
                    label=f'{geometry} × {material}')
ax.set_xlabel('N')
ax.set_ylabel('Speed-up over CPU FEM (×)')
ax.set_title('Operator speed-up')
ax.legend(frameon=False, fontsize=6.5, ncol=1)
ax.grid(True, which='both', alpha=0.25)

fig.suptitle('Operator vs. FEM: accuracy and cost across mesh resolutions', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT, 'fig_operator_vs_fem.png')
fig.savefig(out_path)
print('Saved', out_path)
