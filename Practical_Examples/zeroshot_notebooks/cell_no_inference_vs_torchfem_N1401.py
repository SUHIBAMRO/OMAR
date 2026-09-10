# =====================================================================
#  CELL -- neural operator (NO) inference time at N=1401 vs. torch-fem's
#  own solve time there (Timon round-9, item 12, 2026-09-10: "Did you
#  try for N=1401 NO inference time and compare it to the 30s of
#  torch-FEM?").
#
#  Not previously measured anywhere in this project: the operator's own
#  Table 7 inference-latency number was measured at the study's
#  standard training/eval resolution (N=21), not at N=1401 -- a mesh
#  size far beyond anything the zero-shot resolution-invariance study
#  tested (up to N=49). N=1401 (3,925,602 DOF) is the SAME resolution
#  torch-fem's own timing sweep uses.
#
#  HOW: build_sample_b1(N=1401, seed=0, material='neo_hookean',
#  solve_fem=False) (resolution_invariance_zeroshot.py) builds the mesh/
#  BC/material/sample structure at N=1401 WITHOUT solving the FEM
#  ground truth (which would cost hours at this size and isn't needed
#  for a timing measurement) -- feeds directly into
#  benchmark_inference_latency_Q4 (train_B1.py), the exact same timing
#  protocol/function that produced Table 7's own number.
#
#  torch-fem's own N=1401 number to compare against: 133.83s at matched
#  FP64/1e-8 precision (this project's own real, matched-precision
#  measurement, 2026-09-10) -- NOT the old 29.1s/"~30s" unmatched
#  float32/1e-3 number Timon's email cites, which predates the
#  precision fix. Both are reported here so Omar can see which one
#  Timon meant once this lands in front of him.
#
#  RISK, stated plainly: N=1401 is far beyond any resolution this
#  operator has ever been evaluated at (zero-shot study: up to N=49).
#  This measures whatever happens at that scale honestly -- fast,
#  slow, or an outright OOM -- rather than assuming it will "just work"
#  because Transolver is architecturally resolution-flexible.
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

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- results would not be comparable to the GPU-measured Table 7 number')

R = '/content/drive/MyDrive/pfem_run'
# Same checkpoint item #4/Table 7's own inference-latency number used --
# the physics-informed operator for B1 x Neo-Hookean.
CKPT = f'{R}/results/checkpoints/B1_neo_hookean/model_best.pt'
if not os.path.exists(CKPT):
    # fall back to the data-driven one used for the round-8 point-7
    # latency check, if the physics-informed path differs in this Drive
    CKPT = f'{R}/data_driven/B1_neo_hookean/model_best.pt'
assert os.path.exists(CKPT), f'checkpoint not found, update CKPT: tried {CKPT}'
print('Using checkpoint:', CKPT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

from omar_pfem.train_B1 import benchmark_inference_latency_Q4
from omar_pfem.resolution_invariance_zeroshot import build_sample_b1
from omar_pfem.measure_inference_latency import build_model
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)

model = build_model(args, device)
state_dict = torch.load(CKPT, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print('Checkpoint loaded.')

N_TEST = 1401
print(f'\nBuilding N={N_TEST} sample (solve_fem=False -- no FEM ground truth needed for timing)...')
sample, _ = build_sample_b1(N_TEST, seed=0, material='neo_hookean', Lx=args.Lx, Ly=args.Ly,
                             solve_fem=False)
print(f'  mesh: {sample["xy"].shape[0]} nodes, {sample["quad"].shape[0]} elements')

print('\nMeasuring inference latency (this is where an OOM would show up, if any)...')
stats = benchmark_inference_latency_Q4([sample], model, args, device, dtype,
                                        n_repeats=200, n_warmup=20)
print('Result:', stats)

OUT_JSON = f'{R}/no_inference_vs_torchfem_N1401.json'
TORCHFEM_N1401_MATCHED_S = 133.83   # this project's own real FP64/1e-8 number
TORCHFEM_N1401_OLD_UNMATCHED_S = 29.108236074447632  # float32/1e-3, pre-fix

result = {
    "N": N_TEST, "n_nodes": int(sample["xy"].shape[0]), "n_elements": int(sample["quad"].shape[0]),
    "no_inference_ms_per_sample": stats["inference_ms_per_sample"],
    "no_inference_device": stats["inference_device"],
    "torchfem_N1401_matched_fp64_1e8_s": TORCHFEM_N1401_MATCHED_S,
    "torchfem_N1401_old_unmatched_float32_1e3_s": TORCHFEM_N1401_OLD_UNMATCHED_S,
}
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print('\nSaved:', OUT_JSON)

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

no_s = result["no_inference_ms_per_sample"] / 1000.0
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
labels = ['NO inference\n(this checkpoint)', 'torch-fem\n(matched FP64/1e-8)', 'torch-fem\n(old, unmatched)']
values = [no_s, TORCHFEM_N1401_MATCHED_S, TORCHFEM_N1401_OLD_UNMATCHED_S]
colors = [PRIMARY, SECONDARY, '#bbbbbb']
bars = ax.bar(labels, values, color=colors)
ax.set_yscale('log')
ax.set_ylabel('Time (s), log scale')
ax.set_title(f'NO inference vs. torch-fem solve time, N={N_TEST}')
ax.grid(True, axis='y', alpha=0.25)
add_bar_labels(ax, bars, fmt='{:.4f}s', fontsize=8)
ax.set_ylim(top=ax.get_ylim()[1] * 5)
fig.tight_layout()
FIG_PATH = f'{R}/fig_no_inference_vs_torchfem_N1401.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
speedup_matched = TORCHFEM_N1401_MATCHED_S / no_s
speedup_old = TORCHFEM_N1401_OLD_UNMATCHED_S / no_s
print(f'  NO inference at N={N_TEST}: {result["no_inference_ms_per_sample"]:.4f} ms/sample '
      f'({no_s:.6f} s) on {result["no_inference_device"]}.')
print(f'  torch-fem solve at N={N_TEST}: {TORCHFEM_N1401_MATCHED_S}s (matched FP64/1e-8, real) / '
      f'{TORCHFEM_N1401_OLD_UNMATCHED_S:.1f}s (old, unmatched float32/1e-3 -- likely what')
print(f'  Timon\'s own "~30s" referred to, from before the precision fix).')
print(f'  NO is {speedup_matched:,.0f}x faster than torch-fem (matched) and '
      f'{speedup_old:,.0f}x faster than torch-fem (old unmatched) at this single N.')
print('  Caveat to state alongside this number: this compares one FEM SOLVE against one')
print('  NO INFERENCE call -- the NO paid its accuracy-for-this-mesh-size cost once, at')
print('  training time (and, per the zero-shot study, was never actually validated for')
print(f'  accuracy at N={N_TEST} specifically -- only up to N=49). A large speed advantage')
print('  at inference time is not itself evidence the prediction is trustworthy at a mesh')
print('  size this far outside anything the zero-shot study covered.')
