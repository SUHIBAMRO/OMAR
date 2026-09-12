# =====================================================================
#  CELL -- torch.compile as an actual optimization attempt for NO
#  inference at N=1401, before treating 2.29s as the final number
#  (Timon round-10, item 3: "there is probably still room for
#  optimising the NO at inference. Could you please check this before
#  considering the 2.29s as the final inference number.")
#
#  The profiling cell (Round6_NO_Inference_Profile_N1401.ipynb) measured
#  WHERE the time goes (~90% in bmm/einsum/GEMM) but never tried to make
#  it faster. This cell tries the one safe, correctness-preserving
#  optimization that directly targets what the profiler's own numbers
#  pointed at: 4,800 cudaLaunchKernel calls / 2,893 "Command Buffer
#  Full" events across only 30 repeats -- torch.compile's kernel fusion
#  is built exactly for this kind of dispatch overhead, without
#  changing the architecture or a single weight.
#
#  Correctness-checked before any speedup is trusted: compares the
#  compiled model's own output against eager mode's on the same input.
#  If torch.compile fails or does not help on this architecture, that
#  is reported honestly -- eager mode stays the answer, not assumed
#  successful.
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
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__)

R = '/content/drive/MyDrive/pfem_run'
CKPT = f'{R}/results/checkpoints/B1_neo_hookean/model_best.pt'
if not os.path.exists(CKPT):
    CKPT = f'{R}/data_driven/B1_neo_hookean/model_best.pt'
assert os.path.exists(CKPT), f'checkpoint not found, update CKPT: tried {CKPT}'
print('Using checkpoint:', CKPT)

device = torch.device('cuda')
dtype = torch.float32

from omar_pfem.resolution_invariance_zeroshot import build_sample_b1
from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_inference_torch_compile import profile_with_torch_compile
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
print(f'\nBuilding N={N_TEST} sample...')
sample, _ = build_sample_b1(N_TEST, seed=0, material='neo_hookean', Lx=args.Lx, Ly=args.Ly,
                             solve_fem=False)

print('\nRunning eager baseline + torch.compile attempt (this can take a few minutes -- '
      'the FIRST compiled call triggers real compilation, not counted in the timing)...')
result = profile_with_torch_compile(sample, model, args, device, dtype,
                                     n_repeats=200, n_warmup=20, compile_warmup=5)

OUT_JSON = f'{R}/no_inference_torch_compile_N1401.json'
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print('Saved:', OUT_JSON)

print('\n' + '=' * 70)
print('RESULT -- torch.compile attempt at N=1401')
print('=' * 70)
print(json.dumps(result, indent=2))
if result['compile_succeeded']:
    print(f"\ntorch.compile: {result['compiled_ms_per_sample']:.4f} ms/sample vs. eager "
          f"{result['eager_ms_per_sample']:.4f} ms/sample -- {result['speedup_vs_eager']:.2f}x, "
          f"output relative difference vs. eager: {result['compiled_vs_eager_rel_diff']:.3e}")
    print("If the relative difference above is at floating-point noise level (~1e-5 or "
          "tighter for fp32), this speedup is a real, correctness-preserving optimization "
          "and the compiled number, not 2.29s, should be the one considered for finalizing.")
else:
    print(f"\ntorch.compile did not produce a usable result on this model: {result['error']}")
    print("Eager mode's 2.29s stays the answer -- this was a genuine attempt, honestly "
          "reported, not assumed to succeed.")

# ---- Figure ------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import PRIMARY, SECONDARY, add_bar_labels

fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
labels = ['eager (fp32)']
values = [result['eager_ms_per_sample']]
colors = [PRIMARY]
if result['compile_succeeded']:
    labels.append('torch.compile')
    values.append(result['compiled_ms_per_sample'])
    colors.append(SECONDARY)
bars = ax.bar(labels, values, color=colors)
ax.set_ylabel('ms/sample')
title = 'NO inference, N=1401: eager vs. torch.compile'
if not result['compile_succeeded']:
    title += '\n(compile FAILED -- see printed error)'
ax.set_title(title)
ax.grid(True, axis='y', alpha=0.25)
add_bar_labels(ax, bars, fmt='{:.1f}')
fig.tight_layout()
FIG_PATH = f'{R}/fig_no_inference_torch_compile_N1401.png'
fig.savefig(FIG_PATH)
print('\nSaved figure:', FIG_PATH)
