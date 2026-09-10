# =====================================================================
#  CELL -- TensorMesh's own ACCURACY and MESH CONVERGENCE at PRODUCTION
#  scale (N=401/701/1001/1401, matching torch-fem's own sweep exactly),
#  now that the sparse-Jacobian fix makes this reachable.
#
#  WHY THIS EXISTS: this project's TensorMesh comparison was measured
#  (2026-09-10) to be intractable past N=51 with nonlinear_solve's own
#  DEFAULT dense-then-sparsify Jacobian (a dense torch.autograd.
#  functional.jacobian call) -- extrapolating the real N=3..51 timings
#  projected roughly 10 days for a single N=401 solve. Per Omar's
#  explicit instruction ("صلح تينسور وخلينا نكملها كما هو مطلوب" --
#  fix TensorMesh and let's finish it as required), an explicit sparse
#  jac_fn was written instead (tensormesh_comparison.py's own
#  build_sparse_jac_fn), reusing THIS PROJECT'S OWN already-correct,
#  already-fast per-element Hessian machinery (matrix_free_solver.py's
#  vmap+hessian local tangent, the same one "ours" own matrix-free
#  solver already uses for its Hessian-vector products) rather than
#  re-deriving assembly from scratch or asking TensorMesh's own
#  ElementAssembler for a tangent it does not expose.
#
#  REAL, MEASURED RESULT of the fix (CPU, this environment, before this
#  notebook ever ran): N=51 dropped from 105.27s (dense) to 0.42-0.49s
#  (sparse) -- a ~215-250x speedup at the SAME accuracy (correctness
#  re-verified against "ours" own solver: relative displacement-field
#  difference 1.212e-11, unchanged from the dense-Jacobian result).
#  Pushed further on CPU alone: N=401 (torch-fem's own smallest
#  production point) solved in 62.77s -- previously projected at ~10
#  DAYS. This notebook runs the same sweep on GPU, where it should be
#  faster still, up to torch-fem's full N=401-1401 range.
#
#  ONE REAL CAVEAT WORTH WATCHING, confirmed by reading torch_sla's own
#  installed source directly (not assumed): CUDA_ITERATIVE_THRESHOLD =
#  2_000_000 in torch_sla/backends/__init__.py -- direct solvers (cuDSS)
#  are only used below ~2M DOF on CUDA; N=1001 (2,004,002 DOF) sits
#  right at this line and N=1401 (3,925,602 DOF) is well past it, so
#  linear_method='lu' at those two sizes may silently fall back to an
#  iterative method instead of a true direct factorization -- watch the
#  printed wall-clock times for a discontinuity between N=701 and
#  N=1001 as a sign this happened, and report it plainly either way
#  rather than assuming 'lu' was honored.
#
#  RESUMABLE: run_tensormesh_convergence_study checks its own out_json
#  and skips resolutions already present. The fine reference RESUMES
#  from "ours" own already-converged checkpoint.
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

run([sys.executable, '-m', 'pip', 'install', '-q', 'tensormesh-fem', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- Runtime > Change runtime type > GPU strongly recommended')

# Cheap correctness re-check before spending any real GPU time on the
# sweep below -- same discipline as every other GPU run in this project.
run([sys.executable, '-m', 'omar_pfem.tensormesh_comparison', '11'])

R = '/content/drive/MyDrive/pfem_run'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
OUT_JSON = f'{R}/tensormesh_convergence_production_N401_1401.json'
os.makedirs(R, exist_ok=True)

RESOLUTIONS = [401, 701, 1001, 1401]

run([
    sys.executable, '-u', '-m', 'omar_pfem.tensormesh_comparison', 'convergence',
    ','.join(str(n) for n in RESOLUTIONS), OUT_JSON, CHECKPOINT_DIR, '2236',
])

print('\nDone. Results:', OUT_JSON)

# ---- Comparison table against torch-fem's own already-committed numbers
# at the SAME resolutions and fine reference. ----
with open(OUT_JSON) as f:
    tm_rows = {r['N']: r for r in json.load(f)['rows']}

TF_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_convergence_vs_fine_reference_full.json'
tf_rows = {}
if os.path.exists(TF_JSON):
    with open(TF_JSON) as f:
        tf_rows = {r['N']: r for r in json.load(f)['rows']}

print('\nN      TensorMesh L2_rel   torch-fem L2_rel   TensorMesh wall_s   torch-fem wall_s')
for N in RESOLUTIONS:
    t = tm_rows.get(N)
    f_ = tf_rows.get(N)
    if t and f_:
        print(f"{N:<6} {t['l2_rel']:<20.3e} {f_['l2_rel']:<18.3e} "
              f"{t['tensormesh_wall_clock_s']:<19.2f} {f_['torchfem_wall_clock_s']:<16.2f}")
    elif t:
        print(f"{N:<6} {t['l2_rel']:<20.3e} {'(no torch-fem row)':<18} "
              f"{t['tensormesh_wall_clock_s']:<19.2f}")

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, add_bar_labels

Ns_common = [n for n in RESOLUTIONS if n in tm_rows and n in tf_rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

ax = axes[0]
x = np.arange(len(Ns_common))
tm_l2 = [tm_rows[n]['l2_rel'] for n in Ns_common]
tf_l2 = [tf_rows[n]['l2_rel'] for n in Ns_common]
b1 = ax.bar(x - 0.2, tm_l2, 0.4, label='TensorMesh', color=PRIMARY)
b2 = ax.bar(x + 0.2, tf_l2, 0.4, label='torch-fem', color=SECONDARY)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in Ns_common])
ax.set_title('L2 relative error vs. fine reference')
ax.legend(frameon=False, fontsize=8)
add_bar_labels(ax, b1, fmt='{:.2e}', fontsize=7, rotation=90, pad=5)
add_bar_labels(ax, b2, fmt='{:.2e}', fontsize=7, rotation=90, pad=5)
ax.set_ylim(top=ax.get_ylim()[1] * 20)

ax = axes[1]
tm_t = [tm_rows[n]['tensormesh_wall_clock_s'] for n in Ns_common]
tf_t = [tf_rows[n]['torchfem_wall_clock_s'] for n in Ns_common]
b1 = ax.bar(x - 0.2, tm_t, 0.4, label='TensorMesh', color=PRIMARY)
b2 = ax.bar(x + 0.2, tf_t, 0.4, label='torch-fem', color=SECONDARY)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in Ns_common])
ax.set_title('Wall-clock time')
ax.set_ylabel('seconds, log scale')
ax.legend(frameon=False, fontsize=8)
add_bar_labels(ax, b1, fmt='{:.1f}', fontsize=7, rotation=90, pad=5)
add_bar_labels(ax, b2, fmt='{:.1f}', fontsize=7, rotation=90, pad=5)
ax.set_ylim(top=ax.get_ylim()[1] * 20)

fig.suptitle('TensorMesh (sparse jac_fn) vs. torch-fem: production-scale accuracy and cost', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
FIG_PATH = f'{R}/fig_tensormesh_convergence_production.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
for N in Ns_common:
    t, f_ = tm_rows[N], tf_rows[N]
    speedup = f_['torchfem_wall_clock_s'] / t['tensormesh_wall_clock_s']
    print(f'\n  N={N} ({t["n_dof"]:,} DOF):')
    print(f'    L2 rel error: TensorMesh={t["l2_rel"]:.3e}  torch-fem={f_["l2_rel"]:.3e}')
    print(f'    Wall-clock:   TensorMesh={t["tensormesh_wall_clock_s"]:.2f}s  '
          f'torch-fem={f_["torchfem_wall_clock_s"]:.2f}s  '
          f'({"TensorMesh" if speedup > 1 else "torch-fem"} '
          f'{max(speedup, 1/speedup):.2f}x faster)')
print('\n  Watch for a jump in TensorMesh wall-clock between N=701 and N=1001 -- that would')
print('  be torch_sla\'s own CUDA_ITERATIVE_THRESHOLD (2,000,000 DOF) silently switching')
print('  linear_method=\'lu\' from a true direct solve to an iterative fallback, not a')
print('  regression in the sparse Jacobian itself. Report whichever happened plainly.')
