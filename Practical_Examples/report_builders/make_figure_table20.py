"""Item #12: GPU-native matrix-free solver scaling (Table 20/20a/20c),
built entirely from already-committed JSON result files -- no model to
load, no new solving, runs locally with no Colab/GPU needed.

Real, CONVERGED values only, combined from the two source files the
same way Table 6a's own rebuild already did: N=51/101/201 from the
original plain-Jacobi run (real, uncapped at these sizes already) and
N=401/701/1001/1401 from the geometric-multigrid re-run (item #4) --
NOT the placeholder/checkpoint-resume artifacts each file also
contains at the other's resolutions (e.g. the original file's N=1001/
1401 rows are a "0.9s*"-style resume artifact, not a real timing; the
mgv file's own N=51/101/201 rows are similarly a resume artifact, not
a real timing).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PF = 'omar_pfem/highdof_stress_qoi_results'
OUT = 'report_builders/figures'
os.makedirs(OUT, exist_ok=True)

orig = json.load(open(os.path.join(PF, 'high_dof_stress_qoi_B1_neo_hookean.json')))['orders']['Q4']['rows']
d1 = json.load(open(os.path.join(PF, 'high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json')))['orders']['Q4']['rows']
d2 = json.load(open(os.path.join(PF, 'high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json')))['orders']['Q4']['rows']

by_n_orig = {r['N']: r for r in orig}
# d1 is the N=401-dedicated mgv run (its own N=401 row is the real fresh
# solve); d2 is the N=701/1001/1401 combined run, whose OWN N=401 row is
# just a fast checkpoint-resume artifact (a duplicate dict key here would
# silently prefer whichever file is merged last) -- keep each file's real
# row for its own dedicated resolution(s) rather than blindly merging.
by_n_d1 = {r['N']: r for r in d1}
by_n_d2 = {r['N']: r for r in d2}

rows = []
for n in (51, 101, 201):
    rows.append(by_n_orig[n])
rows.append(by_n_d1[401])
for n in (701, 1001, 1401):
    rows.append(by_n_d2[n])

N = np.array([r['N'] for r in rows])
n_dof = np.array([r['n_dof'] for r in rows])
wall = np.array([r['wall_clock_s'] for r in rows])
cg = np.array([r['cg_iters'] for r in rows])
assert all(r['cg_failures'] == 0 for r in rows), 'expected every row to be a real converged solve'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)

ax = axes[0]
ax.loglog(n_dof, wall, marker='o', color='#1f77b4', linewidth=1.8)
ax.set_xlabel('Degrees of freedom')
ax.set_ylabel('Wall-clock time (s)')
ax.set_title('Solve time vs. problem size')
ax.grid(True, which='both', alpha=0.25)
for x, y, n in zip(n_dof, wall, N):
    ax.annotate(f'N={n}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=7)

ax = axes[1]
ax.loglog(n_dof, cg, marker='s', color='#d62728', linewidth=1.8)
ax.set_xlabel('Degrees of freedom')
ax.set_ylabel('Total CG iterations')
ax.set_title('CG iteration count vs. problem size')
ax.grid(True, which='both', alpha=0.25)
for x, y, n in zip(n_dof, cg, N):
    ax.annotate(f'N={n}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=7)

fig.suptitle('GPU-native matrix-free solver scaling (B1, Neo-Hookean)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT, 'fig_table20_scaling.png')
fig.savefig(out_path)
print('Saved', out_path)
