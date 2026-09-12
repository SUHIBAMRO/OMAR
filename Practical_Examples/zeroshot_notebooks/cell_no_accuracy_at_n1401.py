# =====================================================================
#  CELL -- NO accuracy (L2, H1, energy, PK1 stress, reactions) at N=1401,
#  against a real FEM ground truth there for the first time (Timon
#  round-10, item 1): "the NO accuracy at N=1401 itself, including these
#  QoIs has not been checked. So the 10 times speed-up at N=1401 is not
#  yet an accuracy matched comparison."
#
#  Ground truth comes from omar_pfem.no_ground_truth_fast.solve_b1_fast_gpu,
#  a fast GPU-vectorized path CPU-verified (N=11: 3.9e-12, N=21: 6.7e-12
#  relative displacement difference) against the slow CPU reference that
#  made a real N=1401 ground truth previously look prohibitively
#  expensive ("would cost hours"). Reuses the SAME QoI machinery already
#  used for Table 15-17 (physical_quantities_eval.py), applied here to
#  NO-vs-FEM instead of FEM-vs-FEM.
#
#  Plumbing already smoke-tested with a random-init model at N=21 on CPU
#  (no crashes/shape errors) -- this cell is the first run against the
#  REAL checkpoint at the REAL N=1401.
#
#  Ground truth now goes through assembled_direct_solver.solve_assembled_
#  direct (switched from solve_matrix_free after that solver's own real
#  N=1401 cost turned out to be ~7.5 hours -- see PROJECT_STATUS.md), which
#  needs torch-sla + a pinned nvmath-python for a real cuDSS direct solve
#  on CUDA -- same two packages every other assembled+direct notebook in
#  this project already installs (e.g. cell_assembled_direct_speedup_
#  production.py), installed here for the same reason.
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

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'nvmath-python[cu12]==0.9.0'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

# Confirm the direct solver is ACTUALLY available before spending any real
# GPU time -- same discipline as every other assembled+direct notebook.
from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError(
        "cuDSS is NOT available after installing nvmath-python[cu12] -- the ground-truth "
        "solve below would silently fall back to an iterative solver, not the direct one "
        "this notebook is meant to use. Check the pip install output above for the real error.")
print('cuDSS (real direct solver on CUDA) is available.')

R = '/content/drive/MyDrive/pfem_run'
CKPT = f'{R}/results/checkpoints/B1_neo_hookean/model_best.pt'
if not os.path.exists(CKPT):
    CKPT = f'{R}/data_driven/B1_neo_hookean/model_best.pt'
assert os.path.exists(CKPT), f'checkpoint not found, update CKPT: tried {CKPT}'
print('Using checkpoint:', CKPT)

device = torch.device('cuda')

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import evaluate_no_accuracy_at_n1401
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model = build_model(args, device)
model.load_state_dict(torch.load(CKPT, map_location=device))
print('Checkpoint loaded.')

N_TEST = 1401
rec = evaluate_no_accuracy_at_n1401(model, args, device, N=N_TEST, seed=0,
                                     material='neo_hookean')

OUT_JSON = f'{R}/no_accuracy_at_n1401.json'
with open(OUT_JSON, 'w') as f:
    json.dump(rec, f, indent=2)
print('Saved:', OUT_JSON)

print('\n' + '=' * 70)
print('RESULT -- NO accuracy at N=1401 (first real ground-truth check there)')
print('=' * 70)
print(json.dumps(rec, indent=2))
print('\nCompare fp32 disp_rel_L2/L2_rel/H1_semi_rel/energy_rel/P_rel_L2 above against this '
      "checkpoint's own already-published numbers at N=21/N=1001/N=1401 (FEM-vs-FEM) in "
      'Tables 15-17, to see how much accuracy degrades this far past the training '
      'resolution -- this is the number needed before the 10x speed-up at N=1401 can be '
      "called an accuracy-matched comparison, and before finding the coarsest FEM N with "
      'comparable accuracy (the next step, not done in this cell).')
if 'bf16' in rec:
    print(f"\nbf16 autocast (Timon round-10, item 3 follow-up -- 5.69x faster than fp32 "
          f"in the profiling cell, 4.6% self-consistency gap there): disp_rel_L2="
          f"{rec['bf16']['disp_rel_L2']:.4e} vs. fp32's {rec['fp32']['disp_rel_L2']:.4e}, "
          f"against the SAME real ground truth -- this is the real accuracy verdict the "
          f"profiling cell's own self-consistency check could not give.")

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

qoi_keys = ['disp_rel_L2', 'L2_rel', 'H1_semi_rel', 'energy_rel', 'P_rel_L2',
            'reaction_resultant_rel_err']
qoi_labels = ['disp L2', 'L2', 'H1 semi', 'energy', 'PK1 stress', 'reaction']

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
x = np.arange(len(qoi_keys))
width = 0.35 if 'bf16' in rec else 0.6
fp32_vals = [rec['fp32'][k] for k in qoi_keys]
bars1 = ax.bar(x - (width / 2 if 'bf16' in rec else 0), fp32_vals, width,
               label='fp32', color=PRIMARY)
if 'bf16' in rec:
    bf16_vals = [rec['bf16'][k] for k in qoi_keys]
    bars2 = ax.bar(x + width / 2, bf16_vals, width, label='bf16', color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(qoi_labels, rotation=20, ha='right')
ax.set_ylabel('Relative error')
ax.set_title(f'NO accuracy vs. real FEM ground truth at N={N_TEST}')
ax.legend()
ax.grid(True, axis='y', alpha=0.25)
add_bar_labels(ax, bars1, fmt='{:.2e}', fontsize=7, rotation=90, pad=6)
if 'bf16' in rec:
    add_bar_labels(ax, bars2, fmt='{:.2e}', fontsize=7, rotation=90, pad=6)
ax.set_ylim(top=ax.get_ylim()[1] * 1.6)
fig.tight_layout()
FIG_PATH = f'{R}/fig_no_accuracy_at_n1401.png'
fig.savefig(FIG_PATH)
print('\nSaved figure:', FIG_PATH)
