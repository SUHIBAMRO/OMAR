# =====================================================================
#  CELL -- EXPERIMENT (Omar's own request, 2026-09-11): torch-fem and
#  TensorMesh's memory-heavy explicit-matrix-assembly + direct-solve
#  approach already works correctly and fast at every resolution "ours"
#  own matrix-free solver has been tested at (up to N=1401, ~69GB of an
#  80GB A100 -- comfortably inside the limit, never crashing). Since
#  matrix-free is a choice made to avoid an out-of-memory failure that,
#  at these exact sizes, never actually happens for the other two
#  solvers, this notebook tries the same thing for "ours": assemble the
#  global sparse tangent explicitly (build_sparse_jac_fn -- already
#  "our own" element-energy Hessian, not TensorMesh's), factorize it
#  directly (cuDSS via torch_sla, the SAME library/threshold-forcing
#  already proven for TensorMesh), and see whether it becomes
#  competitive in speed AND memory too, instead of assuming matrix-free
#  is the only option at this scale.
#
#  NOT TENSORMESH -- omar_pfem/assembled_direct_solver.py's own
#  solve_assembled_direct builds a torch_sla.SparseTensor directly from
#  a (values, row, col, shape) COO triple (confirmed possible by reading
#  torch_sla's installed sparse_tensor/core.py __init__ directly -- no
#  mesh/assembler object required), so this drops the TensorMesh
#  dependency entirely for what is otherwise "our own" solver, just
#  assembled instead of matrix-free.
#
#  CORRECTNESS ALREADY VERIFIED ON CPU (before this notebook ever ran):
#  N=11 rel_diff=1.196e-11, N=21 rel_diff=1.240e-11 against solve_matrix_
#  free's own converged result -- the same order of agreement already
#  established between "ours" and TensorMesh (1.269e-11 at N=3).
#
#  EXPERIMENTAL, NOT A FINALIZED RESULT. Per the standing reminder in
#  PROJECT_STATUS.md (2026-09-10, which applies equally to this idea --
#  the same category of change as the cached-Hessian speedup): this must
#  be verified here on real GPU hardware, and then run past Timon,
#  BEFORE being finalized, applied broadly, or presented as an official
#  project result. Nothing here changes solve_matrix_free's own default
#  behavior or any already-published number.
#
#  WHAT THIS TESTS, IN ORDER:
#  1. Correctness re-check at N=11 on THIS device (fails loudly if it
#     does not match solve_matrix_free's own result).
#  2. Accuracy + speed + peak-GPU-memory at N=401/701/1001/1401 --
#     matching torch-fem's and TensorMesh's own sweeps exactly, against
#     the same fine ~10M-DOF reference.
#  3. A genuine three-way comparison table (wall-clock AND peak memory)
#     against the already-committed real torch-fem and TensorMesh
#     numbers at those same N, plus "ours" own matrix-free numbers
#     where available.
#
#  ONE REAL CAVEAT, same one already documented for the TensorMesh sweep:
#  CUDA_ITERATIVE_THRESHOLD=2_000_000 in torch_sla's own backends module
#  would silently drop 'lu' to an iterative fallback above 2M DOF if not
#  forced past -- solve_assembled_direct forces linear_solver='cudss'
#  explicitly on CUDA (same policy as solve_tensormesh), so this should
#  NOT happen here, but watch the printed wall-clock times for a
#  discontinuity between N=701 and N=1001 as a sign it happened anyway.
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

# torch-sla (imported directly, no TensorMesh package needed for THIS
# experiment) plus the same nvmath-python pin already proven necessary
# for a real cuDSS direct solve on CUDA -- see cell_tensormesh_convergence_
# production.py's own docstring for the two real bugs (missing dependency,
# then a version mismatch) this pin avoids re-hitting.
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
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- Runtime > Change runtime type > GPU required for a meaningful result here')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Confirm the direct solver is ACTUALLY available before spending any real
# GPU time -- same discipline as the TensorMesh notebook.
from torch_sla.backends import is_cudss_available
if device.type == 'cuda' and not is_cudss_available():
    raise RuntimeError(
        "cuDSS is NOT available after installing nvmath-python[cu12] -- "
        "the sweep below would silently fall back to an iterative solver "
        "on CUDA, not the direct solver this experiment is meant to test. "
        "Check the pip install output above for the real error.")
print('cuDSS (real direct solver on CUDA) is available.' if device.type == 'cuda'
      else '(CPU run -- cuDSS not relevant, using scipy LU.)')

# ---- Step 1: correctness re-check at small N, on THIS device ----------
print('\n' + '=' * 70)
print('STEP 1: correctness re-check (N=11, this device, vs. solve_matrix_free)')
print('=' * 70)
run([sys.executable, '-m', 'omar_pfem.assembled_direct_solver', '11'])

# ---- Step 2+3: production-scale accuracy, speed, and peak memory ------
print('\n' + '=' * 70)
print('STEP 2+3: production-scale accuracy/speed/memory, assembled_direct')
print('=' * 70)

R = '/content/drive/MyDrive/pfem_run'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
OUT_JSON = f'{R}/assembled_direct_convergence_production_N401_1401.json'
os.makedirs(R, exist_ok=True)

RESOLUTIONS = [401, 701, 1001, 1401]

run([
    sys.executable, '-u', '-m', 'omar_pfem.assembled_direct_solver', 'convergence',
    ','.join(str(n) for n in RESOLUTIONS), OUT_JSON, CHECKPOINT_DIR, '2236',
])

print('\nDone. Results:', OUT_JSON)

# ---- Four-way comparison against already-committed real numbers -------
with open(OUT_JSON) as f:
    ad_rows = {r['N']: r for r in json.load(f)['rows']}

TF_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_convergence_vs_fine_reference_full.json'
TM_JSON = f'{REPO}/Practical_Examples/omar_pfem/tensormesh_convergence_production_N401_1401.json'
tf_rows, tm_rows = {}, {}
if os.path.exists(TF_JSON):
    with open(TF_JSON) as f:
        tf_rows = {r['N']: r for r in json.load(f)['rows']}
if os.path.exists(TM_JSON):
    with open(TM_JSON) as f:
        tm_rows = {r['N']: r for r in json.load(f)['rows']}

print('\n' + '=' * 70)
print('WALL-CLOCK COMPARISON (seconds)')
print('=' * 70)
print(f'{"N":<6} {"ours(assembled)":<16} {"torch-fem":<12} {"TensorMesh":<12}')
for N in RESOLUTIONS:
    a = ad_rows.get(N, {}).get('assembled_direct_wall_clock_s')
    tf = tf_rows.get(N, {}).get('torchfem_wall_clock_s')
    tm = tm_rows.get(N, {}).get('tensormesh_wall_clock_s')
    a_s = f'{a:.2f}' if a is not None else '(n/a)'
    tf_s = f'{tf:.2f}' if tf is not None else '(n/a)'
    tm_s = f'{tm:.2f}' if tm is not None else '(n/a)'
    print(f'{N:<6} {a_s:<16} {tf_s:<12} {tm_s:<12}')

print('\n' + '=' * 70)
print('PEAK GPU MEMORY (MB)')
print('=' * 70)
print(f'{"N":<6} {"ours(assembled) MB":<20} {"torch-fem MB":<14}')
for N in RESOLUTIONS:
    a_mem = ad_rows.get(N, {}).get('assembled_direct_peak_mem_mb')
    tf_mem = tf_rows.get(N, {}).get('torchfem_peak_mem_mb')
    a_mem_s = f'{a_mem:.1f}' if a_mem is not None else '(n/a, not CUDA)'
    tf_mem_s = f'{tf_mem:.1f}' if tf_mem is not None else '(n/a)'
    print(f'{N:<6} {a_mem_s:<20} {tf_mem_s:<14}')

print('\n' + '=' * 70)
print('ACCURACY (L2 relative error vs. the same fine ~10M-DOF reference)')
print('=' * 70)
print(f'{"N":<6} {"ours(assembled)":<16} {"torch-fem":<12} {"TensorMesh":<12}')
for N in RESOLUTIONS:
    a = ad_rows.get(N, {}).get('l2_rel')
    tf = tf_rows.get(N, {}).get('l2_rel')
    tm = tm_rows.get(N, {}).get('l2_rel')
    a_s = f'{a:.3e}' if a is not None else '(n/a)'
    tf_s = f'{tf:.3e}' if tf is not None else '(n/a)'
    tm_s = f'{tm:.3e}' if tm is not None else '(n/a)'
    print(f'{N:<6} {a_s:<16} {tf_s:<12} {tm_s:<12}')

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import PRIMARY, SECONDARY, GOOD, add_bar_labels, legend_below

Ns_common = [n for n in RESOLUTIONS if n in ad_rows and n in tf_rows and n in tm_rows]
if Ns_common:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    x = np.arange(len(Ns_common))

    ax = axes[0]
    a_t = [ad_rows[n]['assembled_direct_wall_clock_s'] for n in Ns_common]
    tf_t = [tf_rows[n]['torchfem_wall_clock_s'] for n in Ns_common]
    tm_t = [tm_rows[n]['tensormesh_wall_clock_s'] for n in Ns_common]
    b1 = ax.bar(x - 0.27, a_t, 0.27, label='ours (assembled+direct)', color=PRIMARY)
    b2 = ax.bar(x, tf_t, 0.27, label='torch-fem', color=SECONDARY)
    b3 = ax.bar(x + 0.27, tm_t, 0.27, label='TensorMesh', color=GOOD)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in Ns_common])
    ax.set_title('Wall-clock time')
    ax.set_ylabel('seconds, log scale')
    add_bar_labels(ax, b1, fmt='{:.1f}', fontsize=6, rotation=90, pad=5)
    add_bar_labels(ax, b2, fmt='{:.1f}', fontsize=6, rotation=90, pad=5)
    add_bar_labels(ax, b3, fmt='{:.1f}', fontsize=6, rotation=90, pad=5)
    ax.set_ylim(top=ax.get_ylim()[1] * 30)

    ax = axes[1]
    a_l2 = [ad_rows[n]['l2_rel'] for n in Ns_common]
    tf_l2 = [tf_rows[n]['l2_rel'] for n in Ns_common]
    tm_l2 = [tm_rows[n]['l2_rel'] for n in Ns_common]
    b1 = ax.bar(x - 0.27, a_l2, 0.27, label='ours (assembled+direct)', color=PRIMARY)
    b2 = ax.bar(x, tf_l2, 0.27, label='torch-fem', color=SECONDARY)
    b3 = ax.bar(x + 0.27, tm_l2, 0.27, label='TensorMesh', color=GOOD)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in Ns_common])
    ax.set_title('L2 relative error vs. fine reference')
    add_bar_labels(ax, b1, fmt='{:.2e}', fontsize=6, rotation=90, pad=5)
    add_bar_labels(ax, b2, fmt='{:.2e}', fontsize=6, rotation=90, pad=5)
    add_bar_labels(ax, b3, fmt='{:.2e}', fontsize=6, rotation=90, pad=5)
    ax.set_ylim(top=ax.get_ylim()[1] * 30)

    legend_below(ax, ncol=3, fontsize=8)
    fig.suptitle('EXPERIMENTAL: "ours" assembled+direct vs. torch-fem vs. TensorMesh', fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    FIG_PATH = f'{R}/fig_assembled_direct_convergence_production.png'
    fig.savefig(FIG_PATH)
    print('\nSaved figure:', FIG_PATH)
else:
    print('\n(No figure: fewer than one N has all three of ours/torch-fem/TensorMesh -- '
          'check that the torch-fem and TensorMesh JSONs above were found.)')

# ---- Printed analysis --------------------------------------------------
print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
print('This is an EXPERIMENT, not a finalized result. If "ours (assembled+direct)" is now')
print('competitive with or faster than torch-fem/TensorMesh at some N, while using peak memory')
print('within the same safe envelope (well under the 80GB A100 limit those two already fit')
print('inside at these sizes), that is real evidence the matrix-free architecture is a CHOICE')
print('at this scale, not a necessity -- matrix-free would then remain the right default only')
print('for problem sizes where the assembled matrix itself would not fit in memory, not for')
print('every size unconditionally. If it is NOT competitive, or if it uses substantially more')
print('memory than the accuracy gain justifies, report that plainly too -- either outcome is')
print('useful, and per the standing PROJECT_STATUS.md reminder, this must be discussed with')
print('Timon before any of it is treated as a finalized project result either way.')
