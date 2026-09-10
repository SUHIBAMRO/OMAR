# =====================================================================
#  CELL -- reaction-force resultant + per-component PK1 stress (P11/P12/
#  P21/P22), ours vs. torch-fem at large DOF (N=1001, N=1401).
#
#  Closes a real gap Omar flagged (2026-09-10) after re-reading the
#  Summary himself: the earlier "all QoIs at large DOF" result (task #9,
#  cell_torchfem_all_qois_large_dof.py) covered L2, H1, energy norm, and
#  peak (Frobenius) stress -- NOT reaction force or the per-component PK1
#  stress tensor, both of which are established QoIs this project already
#  tracks at the standard resolution (physical_quantities_eval.py's own
#  reaction_errors/stress_errors). Calling the earlier result "all QoIs"
#  was true of what was checked, not literally true -- this cell adds the
#  two missing ones so the claim can actually be made without a caveat.
#
#  New library functions (high_dof_convergence_study.py, 2026-09-10),
#  smoke-tested locally at N=11 before this cell was ever written:
#    - pk1_component_errors_at_point: per-component P11/P12/P21/P22 error
#      at the SAME fixed peak-stress point compute_peak_stress_error
#      already locates (Frobenius norm only, until now).
#    - compute_reaction_resultant_error: total reaction-force resultant
#      on the fixed boundary (B1: bottom edge, both components), compared
#      between two DIFFERENT meshes by their RESULTANT (a mesh-independent
#      equilibrium quantity), not node-by-node like the same-mesh
#      NO-vs-FEM reaction_errors utility.
#  compute_peak_stress_error itself also gained P11-P22 FIELD errors
#  (not just the peak point), purely additive to its existing return dict.
#
#  BOTH sides of the comparison get these new fields from the same code
#  path: "ours" via a normal re-run of high_dof_convergence_study's own
#  CLI (RESUMES from the existing coarse_B1_neo_hookean_Q4_N1001/1401.pt
#  checkpoints already on Drive from the original mgv sweep -- no new
#  multi-hour solve, just the new QoI computation on an already-solved
#  field), torch-fem via run_qoi_study (resolves torch-fem fresh --
#  cheap, ~2-3 minutes total per the already-measured timing breakdown).
#
#  Separate OUT_JSON files from the original task #9 run: run_qoi_study's
#  own resumability keys on N alone, so reusing torchfem_qoi_large_dof.json
#  would see "N=1001/1401 already done" and skip recomputing with the new
#  fields, silently keeping the OLD rows that lack them.
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
RESOLUTIONS = [1001, 1401]

# ---- "ours" side: resumes the existing coarse_*_N1001/1401.pt checkpoints,
# just adds the new QoI fields on top of the already-solved fields. ----
OURS_OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_reaction_pk1_N1001_1401.json'
run([
    sys.executable, '-u', '-m', 'omar_pfem.high_dof_convergence_study',
    '--geometry', 'B1', '--material', 'neo_hookean',
    '--resolutions', '1001,1401', '--fine_N', '2236', '--orders', 'Q4',
    '--precond_kind', 'mgv', '--checkpoint_dir', CHECKPOINT_DIR,
    '--out_json', OURS_OUT_JSON,
])

# ---- torch-fem side: fresh solve (cheap at these two N, per the already
# -measured timing breakdown), same fine reference, same new QoI fields. ----
TF_OUT_JSON = f'{R}/torchfem_qoi_reaction_pk1_N1001_1401.json'
from omar_pfem.torchfem_comparison import run_qoi_study
tf_rows = run_qoi_study(RESOLUTIONS, TF_OUT_JSON, checkpoint_dir=CHECKPOINT_DIR, fine_N=2236)

print('\nDone.')

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

with open(OURS_OUT_JSON) as f:
    ours_rows = {r['N']: r for r in json.load(f)['orders']['Q4']['rows']}
tf_by_n = {r['N']: r for r in tf_rows}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

# Panel 1: reaction-resultant relative error
ax = axes[0]
x = np.arange(len(RESOLUTIONS))
o_vals = [ours_rows[n]['reaction_resultant_rel_err'] for n in RESOLUTIONS]
t_vals = [tf_by_n[n]['reaction_resultant_rel_err'] for n in RESOLUTIONS]
b1 = ax.bar(x - 0.2, o_vals, 0.4, label='Ours', color=PRIMARY)
b2 = ax.bar(x + 0.2, t_vals, 0.4, label='torch-fem', color=SECONDARY)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in RESOLUTIONS])
ax.set_title('Reaction-force resultant, relative error')
ax.legend(frameon=False, fontsize=8)
add_bar_labels(ax, b1, fmt='{:.2e}', fontsize=7, rotation=90, pad=5)
add_bar_labels(ax, b2, fmt='{:.2e}', fontsize=7, rotation=90, pad=5)
ax.set_ylim(top=ax.get_ylim()[1] * 20)

# Panel 2: per-component PK1 stress field errors (P11/P12/P21/P22), N=1401 only
ax = axes[1]
comps = ['P11', 'P12', 'P21', 'P22']
o_vals = [ours_rows[1401][f'{c}_field_l2_rel'] for c in comps]
t_vals = [tf_by_n[1401][f'{c}_field_l2_rel'] for c in comps]
x = np.arange(len(comps))
b1 = ax.bar(x - 0.2, o_vals, 0.4, label='Ours', color=PRIMARY)
b2 = ax.bar(x + 0.2, t_vals, 0.4, label='torch-fem', color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(comps)
ax.set_title('Per-component PK1 stress field error, N=1401')
ax.legend(frameon=False, fontsize=8)
add_bar_labels(ax, b1, fmt='{:.2e}', fontsize=7, rotation=90, pad=5)
add_bar_labels(ax, b2, fmt='{:.2e}', fontsize=7, rotation=90, pad=5)
ax.set_ylim(top=ax.get_ylim()[1] * 20)

fig.suptitle('Reaction force + per-component PK1 stress: ours vs. torch-fem, large DOF', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
FIG_PATH = f'{R}/fig_torchfem_reaction_pk1_components.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
for n in RESOLUTIONS:
    o, t = ours_rows[n], tf_by_n[n]
    print(f'\n  N={n} ({o["n_dof"]:,} DOF):')
    print(f'    Reaction resultant: ours={o["reaction_resultant_pred"]}  '
          f'ref={o["reaction_resultant_ref"]}  rel_err={o["reaction_resultant_rel_err"]:.3e}')
    print(f'    torch-fem reaction resultant: pred={t["reaction_resultant_pred"]}  '
          f'ref={t["reaction_resultant_ref"]}  rel_err={t["reaction_resultant_rel_err"]:.3e}')
    for c in ['P11', 'P12', 'P21', 'P22']:
        print(f'    {c}_field_l2_rel: ours={o[f"{c}_field_l2_rel"]:.3e}  '
              f'torch-fem={t[f"{c}_field_l2_rel"]:.3e}  '
              f'ratio={t[f"{c}_field_l2_rel"] / (o[f"{c}_field_l2_rel"] + 1e-30):.2f}x')
print('\n  A ratio near 1.0x on reaction force AND every PK1 stress component, not just the')
print('  Frobenius-norm peak already reported, is what actually justifies calling this an')
print('  "all QoIs" comparison -- read the two numbers above together with the earlier')
print('  L2/H1/energy/peak-stress result, not as a separate, smaller claim.')
