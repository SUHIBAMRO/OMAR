"""Item #12: operator-vs-GPU-FEM latency scaling figure (Tables
10/10a/10b), built from the numbers already published in the real
Report docx -- no separate JSON source exists for these, so the
report's own already-verified numbers are the only available source,
extracted directly via docx_table_map.py rather than retyped by hand.

IMPORTANT distinction (verified against each table's own caption text,
not assumed from the header row alone -- the header row is identical
across Tables 10/10a/10b/10c and would have caused a real mislabeling
here): Table 10 is the GPU-NATIVE FEM SOLVER's own per-sample cost at
each batch size; Table 10a is the TRAINED OPERATOR's per-sample cost at
those same batch sizes; Table 10b is the operator's speed-up over the
solver (Table 10 / Table 10a, row by row, both hardware-matched).
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
rows10 = {r['Case']: r for r in get_rows(tmap['Table 10'])}
rows10a = {r['Case']: r for r in get_rows(tmap['Table 10a'])}
rows10b = {r['Case']: r for r in get_rows(tmap['Table 10b'])}

CASES = ['B1 × Neo-Hookean', 'B1 × Mooney-Rivlin', 'B1 × Arruda-Boyce',
          'B2 × Neo-Hookean', 'B2 × Mooney-Rivlin', 'B2 × Arruda-Boyce']
BATCH = [1, 8, 32, 128]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)


def series_10(r):
    return [float(r[f'bs={b} (ms/sample)']) for b in BATCH]


def series_10a(r):
    return [float(r[f'bs={b}']) for b in BATCH]


def series_10b(r):
    return [float(r[f'bs={b}'].rstrip('×').replace(',', '')) for b in BATCH]


for ax, rows, getter, title, ylabel, logy in [
    (axes[0], rows10, series_10, 'GPU-native FEM solver', 'Latency (ms/sample)', True),
    (axes[1], rows10a, series_10a, 'Trained operator', 'Latency (ms/sample)', True),
    (axes[2], rows10b, series_10b, 'Operator speed-up', 'Speed-up over GPU-FEM (×)', True),
]:
    for case in CASES:
        geometry, material = case.split(' × ')
        style = '-' if geometry == 'B1' else '--'
        marker = 'o' if geometry == 'B1' else 's'
        ax.plot(BATCH, getter(rows[case]), style, marker=marker,
                 color=MATERIAL_COLOR[material], linewidth=1.6, markersize=5,
                 label=case.replace('×', 'x'))
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Batch size')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)

axes[0].legend(frameon=False, fontsize=7, loc='upper right')

fig.suptitle('GPU-native FEM vs. trained-operator latency across batch sizes', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT, 'fig_operator_latency.png')
fig.savefig(out_path)
print('Saved', out_path)
