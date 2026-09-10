# =====================================================================
#  CELL -- torch-fem tolerance sensitivity: 1e-6/1e-7 vs. the already-
#  used 1e-8 (Timon round-9, item 4/optional: "If you wish you can
#  also test 10^-6 or 10^-7 and report the difference").
#
#  Explicitly optional in Timon's own email ("if you wish"), unlike the
#  FP64/1e-8 request itself ("we certainly should") -- run after
#  everything else because it's a nice-to-have, not because it's hard.
#
#  WHY separate out_json files per tolerance rather than one shared
#  file: run_convergence_study's own resumability keys on N alone, not
#  (N, tol) pairs -- reusing the SAME out_json across different
#  tolerances at the same N would make the second call see "N already
#  done" and silently skip it, using the FIRST tolerance's stale row.
#  Using one file per tolerance sidesteps this entirely without
#  touching already-verified code.
#
#  Only N=401 and N=1401 (small and large ends of the already-measured
#  range) -- representative without re-running the whole sweep at two
#  more tolerances.
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
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
RESOLUTIONS = [401, 1401]
TOLS = [1e-6, 1e-7]  # 1e-8 already done, reused from the existing full result file

from omar_pfem.torchfem_comparison import run_convergence_study

all_rows = {}
for tol in TOLS:
    out_json = f'{R}/torchfem_tolerance_{tol:.0e}.json'
    print(f'\n=== tol={tol:.0e} ===')
    rows = run_convergence_study(RESOLUTIONS, out_json, checkpoint_dir=CHECKPOINT_DIR,
                                  fine_N=2236, tol=tol)
    all_rows[tol] = {r['N']: r for r in rows}

# Reuse the already-committed tol=1e-8 numbers (no re-solve needed)
EXISTING_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_convergence_vs_fine_reference_full.json'
with open(EXISTING_JSON) as f:
    existing = json.load(f)
all_rows[1e-8] = {r['N']: r for r in existing['rows'] if r['N'] in RESOLUTIONS}

print('\nDone.')

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

tols_sorted = sorted(all_rows.keys(), reverse=True)  # 1e-6, 1e-7, 1e-8
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

for N, ax_col in zip(RESOLUTIONS, [0, 1]):
    ax = axes[ax_col]
    l2s = [all_rows[t][N]['l2_rel'] for t in tols_sorted]
    times = [all_rows[t][N]['torchfem_wall_clock_s'] for t in tols_sorted]
    x = np.arange(len(tols_sorted))
    ax2 = ax.twinx()
    b1 = ax.bar(x - 0.2, l2s, 0.4, color=PRIMARY, label='L2 rel error')
    b2 = ax2.bar(x + 0.2, times, 0.4, color=SECONDARY, label='wall-clock (s)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.0e}' for t in tols_sorted])
    ax.set_xlabel('tolerance')
    ax.set_ylabel('L2 relative error', color=PRIMARY)
    ax2.set_ylabel('wall-clock (s)', color=SECONDARY)
    ax.set_title(f'N={N}')
    lines = [b1, b2]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc='upper left')

fig.suptitle('torch-fem tolerance sensitivity: accuracy vs. cost', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
FIG_PATH = f'{R}/fig_torchfem_tolerance_sensitivity.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
for N in RESOLUTIONS:
    print(f'\n  N={N}:')
    for t in tols_sorted:
        r = all_rows[t][N]
        print(f'    tol={t:.0e}: l2_rel={r["l2_rel"]:.4e}  wall_clock={r["torchfem_wall_clock_s"]:.2f}s')
    fastest = min(tols_sorted, key=lambda t: all_rows[t][N]['torchfem_wall_clock_s'])
    tightest = 1e-8
    speedup = all_rows[tightest][N]['torchfem_wall_clock_s'] / all_rows[fastest][N]['torchfem_wall_clock_s']
    print(f'    Loosening from 1e-8 to {fastest:.0e} saves {speedup:.2f}x wall-clock at this N.')
print('\n  Report per Timon\'s own suggestion: does the accuracy actually change enough between')
print('  1e-6/1e-7/1e-8 to justify choosing a looser one, or does 1e-8 cost little extra for a')
print('  materially tighter guarantee? Read the two numbers above together, not the speed alone.')
