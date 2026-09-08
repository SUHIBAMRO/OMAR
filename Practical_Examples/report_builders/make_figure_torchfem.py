"""Item #12: GPU-FEM vs. torch-fem comparison (Table 20d / item #13),
built entirely from the already-committed JSON result file -- no model,
no new computation, runs locally.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PF = 'omar_pfem/torchfem_comparison_results'
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

d = json.load(open(os.path.join(PF, 'torchfem_comparison_B1_neo_hookean.json')))
rows = sorted(d['rows'], key=lambda r: r['N'])

N = [r['N'] for r in rows]
ours = [r['ours_wall_clock_s'] for r in rows]
theirs = [r['torchfem_wall_clock_s'] for r in rows]
mem_theirs = [r['torchfem_peak_mem_mb'] for r in rows]

x = np.arange(len(N))
width = 0.38

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)

ax = axes[0]
ax.bar(x - width / 2, ours, width, label='Ours (matrix-free, mgv)', color='#1f77b4')
ax.bar(x + width / 2, theirs, width, label='torch-fem', color='#ff7f0e')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in N])
ax.set_ylabel('Wall-clock time (s)')
ax.set_title('Solve time')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis='y', alpha=0.25)

ax = axes[1]
ax.bar(x, mem_theirs, width * 1.3, color='#ff7f0e')
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in N])
ax.set_ylabel('Peak GPU memory (MB)')
ax.set_title('torch-fem peak memory')
ax.grid(True, axis='y', alpha=0.25)

fig.suptitle('GPU-native solver vs. torch-fem (B1, Neo-Hookean)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT, 'fig_torchfem_comparison.png')
fig.savefig(out_path)
print('Saved', out_path)
