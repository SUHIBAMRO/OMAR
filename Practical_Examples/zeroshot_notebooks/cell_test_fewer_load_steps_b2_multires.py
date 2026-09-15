# =====================================================================
#  QUICK TEST -- can B2's multi-res data generation (N=21,33,101,201)
#  use FEWER incremental load steps (nsteps=5 or nsteps=3) than the
#  current default (nsteps=10), same question already answered YES for
#  B1 x Neo-Hookean at N=1401 (Test_FewerLoadSteps_N1401.ipynb,
#  2026-09-14, verified SAFE and 1.64x faster there).
#
#  WHY THIS IS DIFFERENT FROM THE N=1401 CASE: that verification was
#  done AT N=1401 specifically -- it says nothing about whether fewer
#  steps are safe at these much smaller resolutions. B2's own data
#  generation already only takes ~3h10m total (all 4 resolutions,
#  500 samples each, real measured number from B1's own manifest at
#  the same sample counts) -- far cheaper than N=1401's ~587s/sample,
#  so the ABSOLUTE time this could save is much smaller too. Worth
#  checking anyway since Omar asked directly, but expectations should
#  be modest (maybe an hour saved, not many).
#
#  THIS CELL IS CHEAP AND SAFE: tests 3 seeds x both B2 materials
#  (neo_hookean, mooney_rivlin) x all 4 train resolutions x 3 nsteps
#  values (10 baseline, 5, 3) = 72 solves total, but at these small
#  resolutions each solve is expected to take at most a few seconds,
#  so the whole sweep should finish in well under 30 minutes. Runs in
#  a SEPARATE Colab tab, writes nothing to any real output directory --
#  pure diagnostic.
#
#  DECISION RULE: a given nsteps value is "SAFE to use" only if EVERY
#  seed at EVERY resolution for EVERY material converges cleanly. If
#  even one combination fails or looks borderline, do NOT use that
#  value for the real B2 multi-res retrain jobs.
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

import numpy as np
from omar_pfem.no_ground_truth_fast import (solve_b2_fast_gpu, check_convergence,
                                             precompute_element_params_B2)
from omar_pfem.data.data_generate_B2 import assemble_traction_inner_curved
from omar_pfem.data.parametric_field import ParametricFieldB2

RESOLUTIONS = [21, 33, 101, 201]  # B2 multi-res retrain's own train_resolutions
MATERIALS = ['neo_hookean', 'mooney_rivlin']  # B2's own two materials (no Arruda-Boyce)
SEEDS = [0, 1, 2]  # 3 quick trials per (N, material, nsteps) combo
R_IN, R_OUT = 1.0, 2.0


def independent_convergence_check(nodes_gt, elems_gt, u_full, seed, material, device, dtype):
    # Same reason as the B1/N=1401 version of this check: solve_b2_fast_gpu's
    # own 4th return value is a hardcoded None, so an independent, post-hoc
    # check (the SAME pattern already used everywhere else in this project)
    # is required rather than trusting a value that was never populated.
    E_fn = ParametricFieldB2("E", seed)
    nu_fn = ParametricFieldB2("nu", seed)
    p_fn = ParametricFieldB2("p", seed)
    tolx = 1e-9
    theta0_nodes = np.where(np.abs(nodes_gt[:, 1]) < tolx)[0]
    thetahalfpi_nodes = np.where(np.abs(nodes_gt[:, 0]) < tolx)[0]
    fixed_dofs = np.concatenate([2 * theta0_nodes + 1, 2 * thetahalfpi_nodes])
    ndof = 2 * len(nodes_gt)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    fext_full = assemble_traction_inner_curved(nodes_gt, elems_gt, R_IN, p_fn)
    mat_params = precompute_element_params_B2(nodes_gt, elems_gt, E_fn, nu_fn, material)
    return check_convergence(nodes_gt, elems_gt, free_dofs, fext_full, mat_params,
                              u_full, material, "Q4", device, dtype)


results = {}
for material in MATERIALS:
    for N in RESOLUTIONS:
        for nsteps in (10, 5, 3):
            key = (material, N, nsteps)
            print(f'\n{"="*70}\nmaterial={material}  N={N}  nsteps={nsteps}\n{"="*70}')
            rows = []
            for seed in SEEDS:
                t0 = time.time()
                u_flat, nodes_gt, elems_gt, _ = solve_b2_fast_gpu(
                    N, seed, material, torch.device('cuda'), torch.float64, nsteps=nsteps)
                dt = time.time() - t0
                conv = independent_convergence_check(nodes_gt, elems_gt, u_flat, seed,
                                                      material, torch.device('cuda'), torch.float64)
                print(f'  seed={seed}: wall_clock={dt:.2f}s  converged_likely={conv["converged_likely"]}  '
                      f'relative_residual={conv["relative_residual"]:.3e}')
                rows.append({'seed': seed, 'wall_clock_s': dt,
                              'converged_likely': conv['converged_likely'],
                              'relative_residual': conv['relative_residual']})
            results[key] = rows

print('\n' + '=' * 70)
print('SUMMARY (SAFE only if every material/N/seed converges for that nsteps)')
print('=' * 70)
for nsteps in (10, 5, 3):
    all_rows = [r for (material, N, ns), rows in results.items() if ns == nsteps for r in rows]
    baseline_rows = [r for (material, N, ns), rows in results.items() if ns == 10 for r in rows]
    mean_t = sum(r['wall_clock_s'] for r in all_rows) / len(all_rows)
    baseline_mean = sum(r['wall_clock_s'] for r in baseline_rows) / len(baseline_rows)
    all_converged = all(r['converged_likely'] for r in all_rows)
    worst_res = max(r['relative_residual'] for r in all_rows)
    speedup = baseline_mean / mean_t if mean_t > 0 else float('nan')
    verdict = ('SAFE to use' if all_converged
               else 'DO NOT USE -- at least one (material, N, seed) failed to converge')
    print(f'  nsteps={nsteps:<3} mean={mean_t:.2f}s (all resolutions/materials pooled)  '
          f'speedup_vs_nsteps10={speedup:.2f}x  all_converged={all_converged}  '
          f'worst_relative_residual={worst_res:.3e}  -> {verdict}')

print('\nPer-(material, N) breakdown, so a value can be adopted for only some '
      'resolutions if it is not uniformly safe:')
for material in MATERIALS:
    for N in RESOLUTIONS:
        for nsteps in (10, 5, 3):
            rows = results[(material, N, nsteps)]
            all_ok = all(r['converged_likely'] for r in rows)
            worst = max(r['relative_residual'] for r in rows)
            mean_t = sum(r['wall_clock_s'] for r in rows) / len(rows)
            print(f'  {material:<15} N={N:<4} nsteps={nsteps:<3} mean={mean_t:.2f}s  '
                  f'all_converged={all_ok}  worst_residual={worst:.3e}')

print('\nIf a lower nsteps shows "SAFE to use" above (uniformly across every '
      'material and resolution), the real B2 multi-res retrain notebooks can '
      'pass --nsteps <that value> to their data-generation command -- do this '
      'BEFORE starting them, not after, since this only affects samples not '
      'yet generated.')
