# =====================================================================
#  CELL -- does the NO's accuracy break down SMOOTHLY between N=49 (known
#  good, 5-10% error) and N=1401 (catastrophic, 640% error), or is the
#  N=1401 result an isolated, suspicious one-off?
#
#  WHY THIS EXISTS: Omar asked, reasonably, whether the N=1401 result
#  might itself be wrong -- there is real history in this project of
#  results that looked like genuine findings turning out to be bugs (the
#  float32/float64 mesh-mismatch bug that produced a false
#  "non-convergence" alarm across six consecutive runs, found and fixed
#  2026-09-12). The single N=1401 number, however trustworthy the
#  ground-truth solve itself now is (relative_residual=1.030e-10,
#  converged_likely=True, matching the in-loop step-10 diagnostic
#  exactly), is still just ONE data point. This cell runs the EXACT SAME,
#  already-debugged pipeline (no code changes, same ground-truth solver,
#  same convergence check, same QoI scoring) at several resolutions IN
#  BETWEEN N=49 and N=1401. A smooth, monotonic-ish rise in error is the
#  expected signature of an operator being pushed past its trained
#  resolution range -- real, independent evidence, not a re-explanation
#  of the same single number.
#
#  COST: cheap. The ground-truth solver (assembled+direct) is fast even
#  at N=1401 (see assembled_direct_convergence_production_N401_1401.json:
#  58.54s wall-clock for a SINGLE-shot solve there; this cell's nsteps=10
#  load-stepping costs some multiple of that, still on the order of a
#  few minutes at the largest N tested here). All intermediate N below
#  are far cheaper than N=1401. Expect the whole sweep well under 30
#  minutes on a real GPU.
#
#  RESUMABLE: run_accuracy_degradation_sweep skips any N already present
#  in its output JSON.
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
from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model = build_model(args, device).to(torch.float32)
model.load_state_dict(torch.load(CKPT, map_location=device))
print('Checkpoint loaded, cast to float32.')

# N=49 is the top of the zero-shot study's own validated range (known
# good, 5-10% error); N=1401 is the already-measured catastrophic point.
# These fill in between, at increasing distance past the trained
# resolutions (21, 33).
RESOLUTIONS = [49, 101, 201, 401, 701, 1001, 1401]

OUT_JSON = f'{R}/no_accuracy_degradation_sweep.json'
rows = run_accuracy_degradation_sweep(model, args, RESOLUTIONS, OUT_JSON, device)

print('\n' + '=' * 70)
print('RESULT -- NO accuracy vs. N, N=49..1401 (real ground truth at every point)')
print('=' * 70)
print(f"{'N':<8}{'converged_likely':<20}{'disp_rel_L2':<16}{'L2_rel':<14}{'H1_semi_rel':<14}")
for r in rows:
    gt = r['ground_truth_convergence']
    print(f"{r['N']:<8}{str(gt['converged_likely']):<20}{r['fp32']['disp_rel_L2']:<16.4e}"
          f"{r['fp32']['L2_rel']:<14.4e}{r['fp32']['H1_semi_rel']:<14.4e}")

not_converged = [r['N'] for r in rows if not r['ground_truth_convergence']['converged_likely']]
if not_converged:
    print(f"\n*** WARNING: ground truth did NOT converge at N={not_converged} -- "
          f"those rows compare the NO against a WRONG reference, exclude them "
          f"before drawing any conclusion. ***")
else:
    print('\nGround truth converged at every N tested -- every row above is a genuine '
          'NO-vs-real-FEM comparison.')

print(json.dumps(rows, indent=2))

# ---- Figure: error vs. N, log-log, with the training resolutions marked ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import PRIMARY, SECONDARY

Ns = [r['N'] for r in rows]
disp = [r['fp32']['disp_rel_L2'] for r in rows]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
ax.loglog(Ns, disp, 'o-', color=PRIMARY, label='fp32 disp_rel_L2')
for train_n in (21, 33):
    ax.axvline(train_n, color='gray', linestyle=':', alpha=0.6)
ax.axvline(49, color=SECONDARY, linestyle='--', alpha=0.8,
           label='top of zero-shot validated range (N=49)')
ax.set_xlabel('N')
ax.set_ylabel('Relative displacement error (disp_rel_L2)')
ax.set_title('NO accuracy degradation from N=49 to N=1401')
ax.grid(True, which='both', alpha=0.25)
ax.legend()
fig.tight_layout()
FIG_PATH = f'{R}/fig_no_accuracy_degradation_sweep.png'
fig.savefig(FIG_PATH)
print('\nSaved figure:', FIG_PATH)
print('Saved data:', OUT_JSON)
