# =====================================================================
#  CELL -- torch-fem vs "ours" on ALL QoIs (not just L2/H1), at large
#  DOF (Timon round-9, item 9, 2026-09-10: "What about all QoIs,
#  particularly for large DOFs (in the range of millions)?").
#
#  The earlier convergence study (Round6_TorchFEM_Convergence_vs_
#  Fine_Reference.ipynb + its N=1001/1401 extension) only compared
#  L2/H1 displacement error. This extends that SAME methodology
#  (evaluated against the identical fine ~10M-DOF reference) with the
#  energy norm and peak-stress QoIs high_dof_convergence_study.py
#  already computes for "ours" own Table 6a-adjacent work
#  (compute_tangent_energy_error, find_fine_peak_stress /
#  compute_peak_stress_error) -- so torch-fem's numbers land in the
#  same units/convention as "ours" own already-published ones.
#
#  "ours" own numbers need NO new computation -- they already exist in
#  highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean_mgv_
#  N701_1001_1401.json (energy_rel_error, peak_stress_rel_err columns)
#  at exactly N=1001 and N=1401, the "large DOF" resolutions Timon
#  named. This cell reuses that file directly rather than re-deriving.
#
#  RESUMABLE: run_qoi_study checks its own out_json and skips
#  resolutions already present.
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
OUT_JSON = f'{R}/torchfem_qoi_large_dof.json'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'

# Timon's own phrasing: "particularly for large DOFs" -- N=1001
# (2,004,002 DOF) and N=1401 (3,925,602 DOF). Smaller N left out
# deliberately: L2/H1 already fully covered there by the earlier
# convergence study; this cell's real job is the NEW QoIs (energy,
# peak stress) at the two largest, already-matched-precision sizes.
RESOLUTIONS = [1001, 1401]

from omar_pfem.torchfem_comparison import run_qoi_study
rows = run_qoi_study(RESOLUTIONS, OUT_JSON, checkpoint_dir=CHECKPOINT_DIR, fine_N=2236)

print('\nDone. Results:', OUT_JSON)

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

OURS_JSON = f'{REPO}/Practical_Examples/omar_pfem/highdof_stress_qoi_results/high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json'
with open(OURS_JSON) as f:
    ours_rows = {r['N']: r for r in json.load(f)['orders']['Q4']['rows']}

Ns = [r['N'] for r in rows]
x = np.arange(len(Ns))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=200)

qoi_specs = [
    ('l2_rel', 'l2_rel_error', 'L2 relative error', axes[0]),
    ('energy_norm_rel', 'energy_rel_error', 'Energy-norm relative error', axes[1]),
    ('peak_stress_rel_err', 'peak_stress_rel_err', 'Peak-stress relative error', axes[2]),
]
for torchfem_key, ours_key, title, ax in qoi_specs:
    ours_vals = [ours_rows[n][ours_key] for n in Ns]
    tf_vals = [r[torchfem_key] for r in rows]
    b1 = ax.bar(x - width / 2, ours_vals, width, label='Ours', color=PRIMARY)
    b2 = ax.bar(x + width / 2, tf_vals, width, label='torch-fem', color=SECONDARY)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in Ns])
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)
    add_bar_labels(ax, b1, fmt='{:.2e}', fontsize=6, rotation=90, pad=5)
    add_bar_labels(ax, b2, fmt='{:.2e}', fontsize=6, rotation=90, pad=5)
    ax.set_ylim(top=ax.get_ylim()[1] * 20)

fig.suptitle('All QoIs at large DOF: ours vs. torch-fem (Timon round-9, item 9)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
FIG_PATH = f'{R}/fig_torchfem_all_qois_large_dof.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
for n in Ns:
    o = ours_rows[n]
    t = next(r for r in rows if r['N'] == n)
    print(f"\n  N={n} ({o['n_dof']:,} DOF):")
    print(f"    L2 rel error:          ours={o['l2_rel_error']:.3e}   torch-fem={t['l2_rel']:.3e}")
    print(f"    H1 semi rel error:     ours={o['h1_semi_rel_error']:.3e}   torch-fem={t['h1_semi_rel']:.3e}")
    print(f"    Energy-norm rel error: ours={o['energy_rel_error']:.3e}   torch-fem={t['energy_norm_rel']:.3e}")
    print(f"    Peak-stress rel error: ours={o['peak_stress_rel_err']:.3e}   torch-fem={t['peak_stress_rel_err']:.3e}")
    l2_ratio = t['l2_rel'] / o['l2_rel_error']
    energy_ratio = t['energy_norm_rel'] / o['energy_rel_error']
    stress_ratio = t['peak_stress_rel_err'] / o['peak_stress_rel_err']
    print(f"    torch-fem/ours ratio -- L2: {l2_ratio:.2f}x, energy: {energy_ratio:.2f}x, "
          f"peak-stress: {stress_ratio:.2f}x")
print('\n  A ratio near 1.0 across all QoIs (not just L2/H1) at these large-DOF')
print('  resolutions is the direct answer to "what about all QoIs at large DOF":')
print('  the two solvers agree, or clearly disagree, consistently across every')
print('  quantity checked, not just the one(s) already reported.')
