# =====================================================================
#  CELL -- does the cached-Hessian Hv speedup (verified on CPU: 15.6x at
#  N=11, 22.3x at N=21, correctness to 1e-13/1e-14) hold at GPU
#  production scale? Per Omar's own request: make "ours" own solver
#  faster than BOTH torch-fem and TensorMesh, not just the fallback for
#  sizes beyond their memory ceiling.
#
#  WHAT THIS TESTS, IN ORDER:
#  1. Correctness re-check at small N (matches the CPU result: final
#     solution should match the existing 'autodiff' path to ~1e-13).
#  2. Speed at N=401/701/1001/1401 (matching torch-fem's and
#     TensorMesh's own sweeps exactly), 'autodiff' (the OLD, every-
#     already-published-number path) vs 'cached_hessian' (the NEW path)
#     -- run mgv-preconditioned, matching "ours" own already-published
#     Table 6a/20-series methodology, not a new preconditioner.
#  3. A genuine three-way speed comparison at those same N: "ours"
#     (both hvp_methods), torch-fem, and TensorMesh, all already-
#     committed real numbers for the other two.
#
#  WHY THIS MIGHT NOT MATCH THE CPU RATIO: GPU already parallelizes the
#  autodiff-heavy 'autodiff' path far more than CPU does (many small
#  per-element ops run genuinely in parallel on a GPU, not just
#  vectorized), so the relative benefit of removing that overhead could
#  be smaller than on CPU -- or similarly large, if autodiff dispatch/
#  kernel-launch overhead (not raw FLOPs) is itself the GPU bottleneck
#  for this many small per-element operations. Genuinely unknown until
#  measured here -- this notebook exists to find out, not to confirm an
#  assumed answer.
#
#  DOES NOT reuse "ours" own already-converged checkpoints for N=401+ --
#  those were produced with hvp_method='autodiff' and precond_kind='mgv'
#  starting from u=0 each load step; testing a DIFFERENT hvp_method
#  needs a fresh solve from the same starting point to be a fair
#  wall-clock comparison (resuming a converged autodiff checkpoint and
#  then re-solving with cached_hessian would just detect convergence
#  immediately, which measures nothing).
# =====================================================================
import json
import os
import subprocess
import sys
import time


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
import numpy as np
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- Runtime > Change runtime type > GPU required for a meaningful result here')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs, solve_one
from omar_pfem.multigrid_precond import build_mg_hierarchy, coarsen_N

# ---- Step 1: correctness re-check at small N, on THIS device ----------
print('\n' + '=' * 70)
print('STEP 1: correctness re-check (small N, this device)')
print('=' * 70)
N_check = 21
r_auto = solve_one('B1', 'Q4', N_check, 'neo_hookean', device, torch.float64,
                    1e-6, 1e-7, hvp_method='autodiff')
r_cached = solve_one('B1', 'Q4', N_check, 'neo_hookean', device, torch.float64,
                      1e-6, 1e-7, hvp_method='cached_hessian')
rel_diff = np.linalg.norm(r_auto['u'] - r_cached['u']) / (np.linalg.norm(r_auto['u']) + 1e-30)
print(f'  N={N_check}: autodiff wall_clock={r_auto["wall_clock_s"]:.2f}s, '
      f'cached_hessian wall_clock={r_cached["wall_clock_s"]:.2f}s, '
      f'speedup={r_auto["wall_clock_s"] / r_cached["wall_clock_s"]:.2f}x')
print(f'  relative difference in final solution: {rel_diff:.3e}')
if rel_diff > 1e-6:
    raise RuntimeError(f'cached_hessian result does not match autodiff (rel_diff={rel_diff:.3e}) -- '
                        f'STOP, do not trust any timing below until this is understood.')
print('  PASS -- both hvp_methods agree; proceeding to production-scale timing.')

# ---- Step 2+3: production-scale speed, both hvp_methods, vs. torch-fem
# and TensorMesh's own already-committed numbers -------------------------
print('\n' + '=' * 70)
print('STEP 2+3: production-scale speed, ours (both methods) vs. torch-fem vs. TensorMesh')
print('=' * 70)

R = '/content/drive/MyDrive/pfem_run'
os.makedirs(R, exist_ok=True)
RESOLUTIONS = [401, 701, 1001, 1401]
MG_MIN_COARSE_N = 13

results = {}
for N in RESOLUTIONS:
    results[N] = {}
    for method in ['autodiff', 'cached_hessian']:
        print(f'\n  N={N}, hvp_method={method} ...')
        r = solve_one('B1', 'Q4', N, 'neo_hookean', device, torch.float64,
                       cg_tol=1e-8, newton_tol=1e-8, precond_kind='mgv',
                       mg_min_coarse_n=MG_MIN_COARSE_N, hvp_method=method,
                       cg_progress_every=None)
        results[N][method] = {
            'wall_clock_s': r['wall_clock_s'],
            'cg_iters_total': r['stats']['cg_iters_total'],
            'newton_iters_total': r['stats']['newton_iters_total'],
            'cg_failures': r['stats']['cg_failures'],
        }
        print(f'    wall_clock={r["wall_clock_s"]:.2f}s, cg_iters={r["stats"]["cg_iters_total"]}, '
              f'cg_failures={r["stats"]["cg_failures"]}')
    speedup = results[N]['autodiff']['wall_clock_s'] / results[N]['cached_hessian']['wall_clock_s']
    results[N]['speedup_cached_vs_autodiff'] = speedup
    print(f'  --> speedup at N={N}: {speedup:.2f}x')

OUT_JSON = f'{R}/cached_hessian_speedup_production_N401_1401.json'
with open(OUT_JSON, 'w') as f:
    json.dump({'geometry': 'B1', 'material': 'neo_hookean', 'order': 'Q4', 'device': str(device),
               'precond_kind': 'mgv', 'rows': results}, f, indent=2)
print(f'\nSaved: {OUT_JSON}')

# ---- Three-way comparison table against already-committed real numbers -
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
print('THREE-WAY COMPARISON (wall-clock, seconds)')
print('=' * 70)
print(f'{"N":<6} {"ours(autodiff)":<16} {"ours(cached)":<14} {"torch-fem":<12} {"TensorMesh":<12}')
for N in RESOLUTIONS:
    o_a = results[N]['autodiff']['wall_clock_s']
    o_c = results[N]['cached_hessian']['wall_clock_s']
    tf = tf_rows.get(N, {}).get('torchfem_wall_clock_s')
    tm = tm_rows.get(N, {}).get('tensormesh_wall_clock_s')
    tf_s = f'{tf:.2f}' if tf is not None else '(n/a)'
    tm_s = f'{tm:.2f}' if tm is not None else '(n/a)'
    print(f'{N:<6} {o_a:<16.2f} {o_c:<14.2f} {tf_s:<12} {tm_s:<12}')

print('\n' + '=' * 70)
print('ANALYSIS')
print('=' * 70)
print('If "ours(cached)" is now below torch-fem and/or TensorMesh at some N, that N is a real')
print('case where the matrix-free solver -- previously 204-306x slower at matched precision --')
print('is now competitive with or faster than an assembled-matrix approach, while STILL never')
print('forming a global matrix (the memory-scaling property this solver exists for in the first')
print('place is unchanged by this optimization -- only the per-CG-iteration cost changed).')
print('If the speedup on GPU is much smaller than the 15.6x/22.3x measured on CPU, that itself')
print('is real, useful information: it would mean GPU already parallelizes the autodiff path')
print('well enough that removing it matters less there than on CPU -- report that plainly too.')
