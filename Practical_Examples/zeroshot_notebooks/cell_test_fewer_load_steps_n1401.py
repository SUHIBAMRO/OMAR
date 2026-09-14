# =====================================================================
#  QUICK TEST -- does B1 x Neo-Hookean at N=1401 still converge with
#  FEWER incremental load steps (nsteps=5 or nsteps=3) than the current
#  default (nsteps=10)?
#
#  WHY: task #24's data generation is running at ~587s/sample (10x the
#  originally-assumed 58.5s), because it needs nsteps=10 -- a prior
#  finding (2026-09-12) showed nsteps=1 (single-shot) does NOT converge
#  at N=1401 with random material fields, so 10 incremental steps were
#  chosen. But every one of those 10 steps converges to a relative
#  residual ~1e2-1e3x TIGHTER than the required tolerance (see the real
#  console output already collected: e.g. 1.441e-09 vs. the 1e-7
#  tolerance) -- suggesting Newton isn't struggling within each step,
#  which makes it PLAUSIBLE (not proven) that fewer, larger steps would
#  still converge. Nothing between nsteps=1 (fails) and nsteps=10
#  (works) has ever been tested.
#
#  THIS CELL IS CHEAP AND SAFE: only 3 samples at each of nsteps=5 and
#  nsteps=3 (6 solves total, ~well under 30 min even if nsteps=5 takes
#  half of nsteps=10's time), run in a SEPARATE Colab tab so it cannot
#  interfere with the already-running full data-generation job. Nothing
#  here writes to the real output directory -- pure diagnostic, no
#  training data produced by this cell.
#
#  DECISION RULE: if all 3 samples converge cleanly (converged_likely
#  True, comparable relative residual to the nsteps=10 case) at a given
#  nsteps, that value is safe to use for the REMAINING samples in the
#  real job (resume with --nsteps <value> once this confirms it). If
#  even one sample fails to converge or looks borderline, do NOT use
#  that value for real data -- ground-truth quality matters more than
#  saved time.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
import time

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    import subprocess
    subprocess.run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
                     'https://github.com/SUHIBAMRO/OMAR.git', REPO], check=True)
else:
    import subprocess
    subprocess.run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'], check=True)
    subprocess.run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'], check=True)
    subprocess.run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'], check=True)

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'einops', 'timm', 'h5py', 'jax', 'tqdm'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'nvmath-python[cu12]==0.9.0'], check=True)

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

from omar_pfem.no_ground_truth_fast import solve_b1_fast_gpu

N = 1401
MATERIAL = 'neo_hookean'
SEEDS = [0, 1, 2]  # 3 quick trials per nsteps value, not the real training seeds

results = {}
for nsteps in (10, 5, 3):
    print(f'\n{"="*70}\nnsteps={nsteps}\n{"="*70}')
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        u_flat, nodes_gt, elems_gt, conv = solve_b1_fast_gpu(
            N, seed, MATERIAL, torch.device('cuda'), torch.float64, nsteps=nsteps)
        dt = time.time() - t0
        print(f'  seed={seed}: wall_clock={dt:.1f}s  converged_likely={conv.get("converged_likely")}  '
              f'relative_residual={conv.get("relative_residual"):.3e}')
        rows.append({'seed': seed, 'wall_clock_s': dt,
                      'converged_likely': conv.get('converged_likely'),
                      'relative_residual': conv.get('relative_residual')})
    results[nsteps] = rows

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
baseline_mean = sum(r['wall_clock_s'] for r in results[10]) / len(results[10])
for nsteps, rows in results.items():
    mean_t = sum(r['wall_clock_s'] for r in rows) / len(rows)
    all_converged = all(r['converged_likely'] for r in rows)
    worst_res = max(r['relative_residual'] for r in rows)
    speedup = baseline_mean / mean_t if mean_t > 0 else float('nan')
    verdict = 'SAFE to use' if all_converged else 'DO NOT USE -- at least one sample failed to converge'
    print(f'  nsteps={nsteps:<3} mean={mean_t:.1f}s  speedup_vs_nsteps10={speedup:.2f}x  '
          f'all_converged={all_converged}  worst_relative_residual={worst_res:.3e}  -> {verdict}')

print('\nIf a lower nsteps shows "SAFE to use" above, resume the REAL data-generation '
      'job (task #24) with --nsteps <that value> for the samples not yet generated -- '
      'do not restart from scratch, the existing 12+ samples at nsteps=10 stay valid '
      '(only the LOAD-STEP COUNT changes, not the physics/mesh/material).')
