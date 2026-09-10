# =====================================================================
#  CELL -- torch-fem timing breakdown by phase: total / assembly /
#  solve(-or-factorization) / nonlinear iterations / peak memory
#  (Timon round-9, item 3, 2026-09-10).
#
#  WHY: Timon asked to "report total time together with assembly,
#  solve/factorization, nonlinear iterations and peak memory" -- the
#  existing comparison only ever reported one aggregate wall-clock
#  number per side. solve_theirs_with_breakdown (torchfem_comparison.py)
#  instruments torch-fem's own Newton loop from the outside (no source
#  changes): each Newton iteration calls eval_residual (assembly:
#  model.assemble_matrix + model.integrate_material) then, only if not
#  converged, the module-level torchfem.sparse.sparse_solve (the linear
#  solve/factorization) -- both monkeypatched with timers for the
#  duration of one solve call only, then restored.
#
#  "cg" (matched to the main accuracy/convergence study) is run at
#  every resolution; "direct" (a real factorization, Timon's own
#  suggestion) is ALSO tried, but only up to N=701 -- direct sparse
#  solves scale far worse than CG at large DOF (fill-in), and this has
#  never been tried above a tiny toy size before, so N=1001/1401 are
#  deliberately skipped for "direct" rather than risk an untested
#  multi-hour factorization or an OOM.
#
#  "ours" (matrix-free) needs no new instrumentation here: nonlinear-
#  iteration and CG-iteration counts already live in
#  highdof_stress_qoi_results/*.json (newton_iters, cg_iters columns),
#  and it has NO separate assembly/factorization phase to report by
#  architecture (every CG iteration IS the Hessian-vector product) --
#  this cell's own printed analysis says so explicitly rather than
#  forcing a number that doesn't correspond to anything real.
#
#  RESUMABLE: run_breakdown_sweep checks its own out_json and skips
#  (N, method) pairs already present.
# =====================================================================
import json
import os
import subprocess
import sys


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU)')

R = '/content/drive/MyDrive/pfem_run'
OUT_JSON = f'{R}/torchfem_timing_breakdown.json'
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

RESOLUTIONS = [401, 701, 1001, 1401]
DIRECT_MAX_N = 701  # "direct" tried only up to here -- see header comment

from omar_pfem.torchfem_comparison import run_breakdown_sweep
rows = run_breakdown_sweep(RESOLUTIONS, OUT_JSON, direct_max_n=DIRECT_MAX_N)

print('\nDone. Results:', OUT_JSON)

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

cg_rows = sorted([r for r in rows if r['method'] == 'cg'], key=lambda r: r['N'])
direct_rows = sorted([r for r in rows if r['method'] == 'direct'], key=lambda r: r['N'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=200)

# Panel 1: assembly vs solve time, stacked, CG method, all N
ax = axes[0]
N_cg = [r['N'] for r in cg_rows]
x = np.arange(len(N_cg))
assembly = [r['assembly_time_s'] for r in cg_rows]
solve = [r['solve_time_s'] for r in cg_rows]
b1 = ax.bar(x, assembly, 0.5, label='Assembly', color=PRIMARY)
b2 = ax.bar(x, solve, 0.5, bottom=assembly, label='Solve (CG)', color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in N_cg])
ax.set_ylabel('Time (s)')
ax.set_title('torch-fem: assembly vs. solve time (CG)')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis='y', alpha=0.25)
totals = [a + s for a, s in zip(assembly, solve)]
for xi, t in zip(x, totals):
    ax.annotate(f'{t:.1f}s', xy=(xi, t), xytext=(0, 3), textcoords='offset points',
                ha='center', fontsize=7)

# Panel 2: CG vs direct total time, where both exist
ax = axes[1]
common_N = [r['N'] for r in direct_rows]
cg_by_n = {r['N']: r for r in cg_rows}
if common_N:
    x2 = np.arange(len(common_N))
    cg_t = [cg_by_n[n]['total_time_s'] for n in common_N]
    direct_t = [r['total_time_s'] for r in direct_rows]
    bb1 = ax.bar(x2 - 0.2, cg_t, 0.4, label='CG (Jacobi)', color=PRIMARY)
    bb2 = ax.bar(x2 + 0.2, direct_t, 0.4, label='Direct (LU)', color=SECONDARY)
    ax.set_xticks(x2)
    ax.set_xticklabels([f'N={n}' for n in common_N])
    add_bar_labels(ax, bb1, fmt='{:.2f}s', fontsize=7)
    add_bar_labels(ax, bb2, fmt='{:.2f}s', fontsize=7)
ax.set_ylabel('Total time (s)')
ax.set_title(f'torch-fem: CG vs. direct (up to N={DIRECT_MAX_N})')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis='y', alpha=0.25)

# Panel 3: peak memory, CG, all N
ax = axes[2]
mem = [r['peak_mem_mb'] / 1024 for r in cg_rows]
b3 = ax.bar(x, mem, 0.5, color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in N_cg])
ax.set_ylabel('Peak memory (GB)')
ax.set_title('torch-fem: peak GPU memory (CG)')
ax.grid(True, axis='y', alpha=0.25)
add_bar_labels(ax, b3, fmt='{:.1f}', fontsize=7)

fig.suptitle('torch-fem timing breakdown by phase (Timon round-9, item 3)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
FIG_PATH = f'{R}/fig_torchfem_timing_breakdown.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
for r in cg_rows:
    frac_assembly = 100 * r['assembly_time_s'] / r['total_time_s']
    print(f"  N={r['N']}: assembly is {frac_assembly:.0f}% of total time "
          f"({r['assembly_time_s']:.2f}s of {r['total_time_s']:.2f}s), "
          f"{r['n_nonlinear_iters']} Newton iterations, "
          f"peak memory {r['peak_mem_mb']/1024:.1f} GB.")
if direct_rows:
    print()
    for r in direct_rows:
        cg_row = cg_by_n[r['N']]
        speedup = cg_row['total_time_s'] / r['total_time_s']
        print(f"  N={r['N']}: direct (LU) is {speedup:.2f}x "
              f"{'faster' if speedup > 1 else 'slower'} than CG+Jacobi "
              f"({r['total_time_s']:.2f}s vs {cg_row['total_time_s']:.2f}s).")
print()
print('  Note on "ours" (matrix-free): no separate assembly/factorization')
print('  phase exists by architecture -- every CG iteration IS a Hessian-')
print('  vector product computed via automatic differentiation, not a')
print('  separate matrix build. "ours" own nonlinear/CG iteration counts')
print('  and wall-clock are already committed in highdof_stress_qoi_')
print('  results/*.json; no new measurement needed on that side.')
