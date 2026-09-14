# =====================================================================
#  CELL -- extend the N=1401 accuracy/QoI analysis (Timon round-11 point 2)
#  to ALL 5 remaining cases: B1 x Mooney-Rivlin, B1 x Arruda-Boyce,
#  B2 x Neo-Hookean, B2 x Mooney-Rivlin, B2 x Arruda-Boyce -- the full
#  scope, per Omar's explicit request ("even if it takes time, do them
#  properly"), not just B1's 2 remaining materials.
#
#  PREREQUISITES (2026-09-14, all already committed, all independently
#  verified before this cell was written):
#  1. solve_b1_fast_gpu / solve_assembled_direct / build_sparse_jac_fn /
#     check_convergence all hardcoded exactly 2 material params -- fixed
#     to accept any arity (*mat_params), verified via _correctness_check
#     for mooney_rivlin/arruda_boyce at N=11 (~1e-11 rel. diff) plus a
#     full Neo-Hookean regression check (unchanged numbers).
#  2. B2 had NO fast GPU ground-truth path at all. Built solve_b2_fast_gpu
#     (reusing build_sample_b2's own grid/BCs/traction assembler), verified
#     via a new _correctness_check_b2 against the slow CPU reference for
#     all 3 materials at N=11 (3.14e-11 / 7.88e-12 / 4.19e-11 rel. diff).
#  3. Built evaluate_no_accuracy_at_n1401_b2 / run_accuracy_degradation_
#     sweep_b2 / run_no_peak_stress_fixed_location_b2 (B1's own versions
#     were hardcoded throughout -- build_sample_b1, ParametricFieldB1,
#     B1's own BCs/energy function). Reaction-force QoI deliberately
#     omitted for B2 (no established convention in this project for its
#     two single-component symmetry edges -- honest gap, not guessed).
#     Smoke-tested end-to-end on CPU (tiny mesh, untrained model) before
#     ever being pointed at a real checkpoint or N=1401.
#
#  CHECKPOINTS: existing zero-shot checkpoints (trained jointly on
#  N=21,33), NOT new multi-res retrains -- per Omar's own explicit
#  choice, this is a first pass with what already exists. Path
#  conventions confirmed directly from the notebooks/cells that actually
#  produced each case's own point7a_results/zeroshot_*.json:
#    B1: {R}/zeroshot_B1_{material}/model_best.pt
#    B2: {R}/zeroshot_B2_{material}_fixedsel/model_best.pt (the
#        "_fixedsel" suffix is real and required -- these are the
#        corrected checkpoints from Round 6's per_component-vs-
#        both_components selection-metric bug fix; the un-suffixed
#        zeroshot_B2_{material}/ checkpoints are the KNOWN-WORSE
#        pre-fix ones and must not be used here.)
#
#  WHAT THIS PRODUCES, per case: the same accuracy-degradation sweep and
#  fixed-location peak-stress crossover check already run for B1xNH (16
#  resolutions, real FEM ground truth, full QoI set for B1; L2/H1/
#  energy/stress but no reaction QoI for B2), then one clear summary
#  table at N=1401 across ALL 5 cases showing which QoI actually binds
#  each case's own "coarsest suitable FEM" answer.
#
#  NOT INCLUDED HERE: the resolution-matched break-even (NO vs. FEM both
#  at N=1401) for any of these 5 cases -- torchfem_comparison.py's own
#  build_torchfem_model hardcodes both the Neo-Hookean psi function
#  (mu, lam only) AND a B1-specific "both displacement components fixed"
#  BC-constraint assumption that is FALSE for B2's symmetry edges (each
#  fixes only one component). Generalizing this safely, without
#  regressing round-9's already-published B1xNH headline numbers, is a
#  separate, comparably-sized piece of work -- not started yet, flagged
#  honestly rather than attempted in a rush.
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
from omar_pfem.no_accuracy_at_n1401 import (
    run_accuracy_degradation_sweep, run_no_peak_stress_fixed_location,
    run_accuracy_degradation_sweep_b2, run_no_peak_stress_fixed_location_b2,
)
import argparse

RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]

# (geometry, material, checkpoint path, sweep fn, peak-stress fn)
CASES = [
    ('B1', 'mooney_rivlin', f'{R}/zeroshot_B1_mooney_rivlin/model_best.pt',
     run_accuracy_degradation_sweep, run_no_peak_stress_fixed_location),
    ('B1', 'arruda_boyce', f'{R}/zeroshot_B1_arruda_boyce/model_best.pt',
     run_accuracy_degradation_sweep, run_no_peak_stress_fixed_location),
    ('B2', 'neo_hookean', f'{R}/zeroshot_B2_neo_hookean_fixedsel/model_best.pt',
     run_accuracy_degradation_sweep_b2, run_no_peak_stress_fixed_location_b2),
    ('B2', 'mooney_rivlin', f'{R}/zeroshot_B2_mooney_rivlin_fixedsel/model_best.pt',
     run_accuracy_degradation_sweep_b2, run_no_peak_stress_fixed_location_b2),
    ('B2', 'arruda_boyce', f'{R}/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt',
     run_accuracy_degradation_sweep_b2, run_no_peak_stress_fixed_location_b2),
]

model_args_b1 = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model_args_b2 = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, R_in=1.0, R_out=2.0,
)

all_results = {}
for geometry, material, ckpt, sweep_fn, peak_fn in CASES:
    case_id = f'{geometry}_{material}'
    print('\n' + '#' * 78)
    print(f'# {case_id}')
    print('#' * 78)

    assert os.path.exists(ckpt), f'checkpoint not found: {ckpt}'
    model_args = model_args_b1 if geometry == 'B1' else model_args_b2
    model = build_model(model_args, device).to(torch.float32)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print('Loaded:', ckpt)

    ACC_JSON = f'{R}/no_accuracy_degradation_sweep_{case_id}.json'
    rows = sweep_fn(model, model_args, RESOLUTIONS, ACC_JSON, device, material=material)

    PEAK_JSON = f'{R}/no_peak_stress_fixed_location_{case_id}.json'
    peak_rows = peak_fn(model, model_args, RESOLUTIONS, PEAK_JSON, device, material=material)

    all_results[case_id] = {'accuracy_rows': rows, 'peak_rows': peak_rows,
                             'ckpt': ckpt, 'acc_json': ACC_JSON, 'peak_json': PEAK_JSON}

    print(f'\n{case_id}: accuracy sweep + peak-stress crossover done.')
    print(f"{'N':<8}{'disp_rel_L2':<16}{'L2_rel':<14}{'H1_semi_rel':<14}")
    for r in rows:
        print(f"{r['N']:<8}{r['fp32']['disp_rel_L2']:<16.4e}"
              f"{r['fp32']['L2_rel']:<14.4e}{r['fp32']['H1_semi_rel']:<14.4e}")

# ---- Which QoI/norm actually binds the "coarsest suitable FEM" crossover,
#      per case, at N=1401 -- directly answers Timon's point-2 question ----
print('\n' + '=' * 78)
print('WHICH QOI DETERMINES THE ACCURACY-MATCHED FEM RESOLUTION -- all 5 cases, at N=1401')
print('=' * 78)

summary_table = []
for geometry, material, *_ in CASES:
    case_id = f'{geometry}_{material}'
    rows = all_results[case_id]['accuracy_rows']
    r1401 = next(r for r in rows if r['N'] == 1401)
    no_l2 = r1401['fp32']['L2_rel']
    no_disp = r1401['fp32']['disp_rel_L2']
    peak_rows = all_results[case_id]['peak_rows']
    p1401 = next((p for p in peak_rows if p['N'] == 1401), None)
    print(f'\n-- {geometry} x {material} --')
    print(f'  NO @ N=1401: disp_rel_L2={no_disp:.4e}  L2_rel={no_l2:.4e}'
          + (f"  peak_stress_rel_err={p1401.get('peak_stress_rel_err', 'n/a')}" if p1401 else ''))
    summary_table.append({'geometry': geometry, 'material': material, 'N': 1401,
                           'disp_rel_L2': no_disp, 'L2_rel': no_l2,
                           'peak_row': p1401})

OUT_SUMMARY = f'{R}/n1401_all_remaining_cases_summary.json'
with open(OUT_SUMMARY, 'w') as f:
    json.dump({'cases': [f'{g}_{m}' for g, m, *_ in CASES], 'resolutions': RESOLUTIONS,
               'results': {k: {'acc_json': v['acc_json'], 'peak_json': v['peak_json']}
                           for k, v in all_results.items()},
               'summary_at_N1401': summary_table}, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    import time
    write_manifest(R, kind='n1401_all_remaining_cases',
                    args={'cases': [f'{g}_{m}' for g, m, *_ in CASES]},
                    started_at=time.time(), results={'summary_at_N1401': summary_table},
                    outputs=[OUT_SUMMARY],
                    notes="Timon round-11 point 2, full scope: all 5 remaining cases "
                          "(B1's other 2 materials + all 3 of B2), existing N=21,33 "
                          "checkpoints, no retraining. Resolution-matched break-even for "
                          "these 5 cases still needs torchfem_comparison.py generalized "
                          "past B1xNeo-Hookean -- not attempted here, flagged separately.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nSaved:', OUT_SUMMARY)
