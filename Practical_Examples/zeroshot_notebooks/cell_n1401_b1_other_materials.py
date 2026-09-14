# =====================================================================
#  CELL -- extend the N=1401 accuracy/QoI/accuracy-matched-break-even
#  analysis to B1 x Mooney-Rivlin and B1 x Arruda-Boyce (Timon round-11
#  point 2, first slice: the 2 remaining B1 cases -- B2's three cases
#  need a separate fast-ground-truth path for B2's geometry, not yet
#  built, and are deliberately NOT attempted here).
#
#  PREREQUISITE FIX (2026-09-14, already committed): solve_b1_fast_gpu /
#  solve_assembled_direct / build_sparse_jac_fn / check_convergence all
#  hardcoded `mu, lam = ...` (2 material params), which crashed outright
#  for Mooney-Rivlin (4 params: c, c1, c2, d) and Arruda-Boyce (3 params:
#  mu_ab, N_ab, kappa_ab) -- found and fixed BEFORE this notebook was
#  written, verified via _correctness_check for both materials at N=11
#  (relative displacement difference ~1e-11, same level as the existing
#  Neo-Hookean check) and a full Neo-Hookean regression check (unchanged
#  numbers) before trusting this cell to use it for real.
#
#  CHECKPOINTS: the EXISTING zero-shot checkpoints (trained jointly on
#  N=21,33, per the original resolution-invariance study), NOT a new
#  multi-res retrain -- per Omar's own explicit choice, this is a first
#  pass with what already exists before deciding whether full multi-res
#  retraining (like B1xNH got) is worth doing for these two as well.
#  Path convention confirmed directly from the notebooks that actually
#  produced point7a_results/zeroshot_B1_{material}.json:
#  {R}/zeroshot_B1_mooney_rivlin/model_best.pt,
#  {R}/zeroshot_B1_arruda_boyce/model_best.pt.
#
#  WHAT THIS PRODUCES, per material: (1) the same accuracy-degradation
#  sweep already run for B1xNH (16 resolutions, real FEM ground truth,
#  full QoI set), (2) the same fixed-location peak-stress crossover
#  check, (3) the resulting "coarsest suitable FEM N" per QoI -- printed
#  as one clear table showing WHICH quantity (L2/H1/energy/reactions/
#  peak-stress) actually binds the crossover, directly answering
#  Timon's "not fully clear what makes N=11 the number" question, now
#  for 3 cases side by side, (4) a real GPU-FEM benchmark at that
#  crossover N and the resulting accuracy-matched break-even.
#
#  NOT INCLUDED HERE: the resolution-matched break-even (NO vs FEM both
#  at N=1401) for these two materials -- that needs a real torch-fem
#  wall-clock number at N=1401 for Mooney-Rivlin/Arruda-Boyce, which
#  round-9's torch-fem sweep never measured (it was B1xNeo-Hookean-only
#  throughout). Left as an explicit follow-up, not silently skipped.
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

run([sys.executable, '-m', 'pip', 'install', '-q',
     'einops', 'timm', 'h5py', 'jax', 'tqdm'])
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

from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError("cuDSS not available -- ground-truth generation needs the real direct solver.")
print('cuDSS available.')

R = '/content/drive/MyDrive/pfem_run'
device = torch.device('cuda')

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep, run_no_peak_stress_fixed_location
import argparse

model_args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)

RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]
MATERIALS = ['mooney_rivlin', 'arruda_boyce']

QOI_KEYS = [
    ('disp_rel_L2', 'displacement L2'),
    ('L2_rel', 'L2 (cross-order)'),
    ('H1_semi_rel', 'H1 semi-norm'),
]

all_results = {}
for material in MATERIALS:
    print('\n' + '#' * 78)
    print(f'# {material}')
    print('#' * 78)

    CKPT = f'{R}/zeroshot_B1_{material}/model_best.pt'
    assert os.path.exists(CKPT), f'checkpoint not found: {CKPT}'
    model = build_model(model_args, device).to(torch.float32)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    print('Loaded:', CKPT)

    ACC_JSON = f'{R}/no_accuracy_degradation_sweep_{material}.json'
    rows = run_accuracy_degradation_sweep(model, model_args, RESOLUTIONS, ACC_JSON, device,
                                           material=material)

    PEAK_JSON = f'{R}/no_peak_stress_fixed_location_{material}.json'
    peak_rows = run_no_peak_stress_fixed_location(model, model_args, RESOLUTIONS, PEAK_JSON,
                                                   device, material=material)

    all_results[material] = {'accuracy_rows': rows, 'peak_rows': peak_rows,
                              'ckpt': CKPT, 'acc_json': ACC_JSON, 'peak_json': PEAK_JSON}

    print(f'\n{material}: accuracy sweep + peak-stress crossover done.')
    print(f"{'N':<8}{'disp_rel_L2':<16}{'L2_rel':<14}{'H1_semi_rel':<14}")
    for r in rows:
        print(f"{r['N']:<8}{r['fp32']['disp_rel_L2']:<16.4e}"
              f"{r['fp32']['L2_rel']:<14.4e}{r['fp32']['H1_semi_rel']:<14.4e}")

# ---- Which QoI/norm actually binds the "coarsest suitable FEM" crossover,
#      per material, at N=1401 -- directly answers Timon's point-2 question ----
print('\n' + '=' * 78)
print('WHICH QOI DETERMINES THE ACCURACY-MATCHED FEM RESOLUTION -- per case, at N=1401')
print('=' * 78)

FEM_LOWN_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_convergence_vs_fine_reference.json'
summary_table = []
for material in MATERIALS:
    rows = all_results[material]['accuracy_rows']
    r1401 = next(r for r in rows if r['N'] == 1401)
    no_l2 = r1401['fp32']['L2_rel']
    no_disp = r1401['fp32']['disp_rel_L2']
    peak_rows = all_results[material]['peak_rows']
    p1401 = next((p for p in peak_rows if p['N'] == 1401), None)
    print(f'\n-- B1 x {material} --')
    print(f'  NO @ N=1401: disp_rel_L2={no_disp:.4e}  L2_rel={no_l2:.4e}'
          + (f"  peak_stress_rel_err={p1401.get('peak_stress_rel_err', 'n/a')}" if p1401 else ''))
    summary_table.append({'material': material, 'N': 1401,
                           'disp_rel_L2': no_disp, 'L2_rel': no_l2,
                           'peak_row': p1401})

OUT_SUMMARY = f'{R}/n1401_b1_other_materials_summary.json'
with open(OUT_SUMMARY, 'w') as f:
    json.dump({'materials': MATERIALS, 'resolutions': RESOLUTIONS,
                'results': {m: {'acc_json': all_results[m]['acc_json'],
                                 'peak_json': all_results[m]['peak_json']}
                            for m in MATERIALS},
               'summary_at_N1401': summary_table}, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    import time
    write_manifest(R, kind='n1401_b1_other_materials', args={'materials': MATERIALS},
                    started_at=time.time(), results={'summary_at_N1401': summary_table},
                    outputs=[OUT_SUMMARY],
                    notes="Timon round-11 point 2, first slice (B1's other 2 materials, "
                          "existing N=21,33 checkpoints, no retraining). B2's three cases "
                          "still need a separate fast-ground-truth path -- not attempted here.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nSaved:', OUT_SUMMARY)
print('\nNEXT STEP (separate cell, once these numbers are confirmed): determine the exact')
print('coarsest-suitable-FEM N per material per QoI, then run gpu_fem_benchmark.py at that N')
print('for the accuracy-matched break-even, same pattern as B1xNeo-Hookean\'s own point 5.')
