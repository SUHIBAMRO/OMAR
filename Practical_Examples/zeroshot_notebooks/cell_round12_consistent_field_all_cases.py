# =====================================================================
#  CELL -- Round-12 point 1, PROPERLY fixed (2026-09-18). Omar caught two
#  real, separate problems in the original table, then pushed back a
#  second time ("بدي تحسب الحسبة بشكل صحيح") rather than accept a text
#  caveat for the second one -- so this rebuilds the actual computation,
#  not just the wording:
#
#  1. The "Op." classical-QoI columns (L2/H1/energy/reaction) compared
#     the operator against FEM solved AT THE SAME LOW N, not against a
#     genuine fine reference -- unlike the Cauchy-stress table, which
#     already used a real fine reference correctly.
#
#  2. Even after fixing (1), FEM's own numbers (from run_qoi_study,
#     AnalyticFieldB1/B2 -- the field built for the separate Table-6a
#     mesh-convergence study) and the operator's own numbers (from
#     ParametricFieldB1/B2(seed) -- the field the operator was actually
#     trained/tested on) were computed on TWO DIFFERENT PHYSICAL
#     PROBLEMS placed side by side in the same row, not just two
#     different mesh resolutions of the same one.
#
#  run_qoi_study_consistent_field_b1/b2 (new, in no_accuracy_at_n1401.py)
#  fixes both at once: FEM and the operator are now solved on the EXACT
#  SAME ParametricFieldB1/B2(seed) realization, and BOTH are scored
#  against ONE real fine reference (fine_N=201) for L2, H1, energy,
#  reaction (B1 only), and the region-Cauchy stress QoI. Verified on CPU
#  first (identity check: coarse==fine gives exactly 0 error on every
#  QoI; a real CPU-solved coarse-vs-fine pair gives sane, finite,
#  non-degenerate numbers) before spending any GPU time -- see the chat
#  transcript for that CPU verification.
#
#  COST: comparable order to the Remaining5_LowN_Accuracy.ipynb run
#  (6m30s for 5 cases' classical QoI alone), but doing MORE work per row
#  now (FEM's own solve at every N too, not reused from an old cache,
#  plus the Cauchy computation for both sides) across all SIX cases, so
#  expect roughly 15-30 minutes total.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import hashlib
import json
import subprocess
import sys
import time

_started = time.time()


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

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'nvmath-python[cu12]==0.9.0'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError(
        "cuDSS is NOT available after installing nvmath-python[cu12] -- the fine-reference "
        "and per-N FEM solves below would silently fall back to an iterative solver, not the "
        "direct one this notebook is meant to use. Check the pip install output above.")
print('cuDSS (real direct solver on CUDA) is available.')

R = '/content/drive/MyDrive/pfem_run'
device = torch.device('cuda')


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import (
    run_qoi_study_consistent_field_b1, run_qoi_study_consistent_field_b2)
import argparse

# Same checkpoint paths every other round-12 cell uses -- final retrained
# checkpoint where one exists, the original run where it does not
# (B2xArruda-Boyce only ever had one).
CASES = [
    ('B1', 'neo_hookean', f'{R}/zeroshot_B1_neo_hookean_multires/model_best.pt'),
    ('B1', 'mooney_rivlin', f'{R}/zeroshot_B1_mooney_rivlin_multires/model_best.pt'),
    ('B1', 'arruda_boyce', f'{R}/zeroshot_B1_arruda_boyce_multires/model_best.pt'),
    ('B2', 'neo_hookean', f'{R}/zeroshot_B2_neo_hookean_multires/model_best.pt'),
    ('B2', 'mooney_rivlin', f'{R}/zeroshot_B2_mooney_rivlin_multires/model_best.pt'),
    ('B2', 'arruda_boyce', f'{R}/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt'),
]

BASE_ARGS = dict(model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
                  mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
                  use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_in=1.0, R_out=2.0)

LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
FINE_N = 201  # same fine reference already established/cost-verified for this project's
              # own low-N crossover study and the round-12 Cauchy-only sweep.

all_results = {}
for geometry, material, ckpt_path in CASES:
    case_id = f'{geometry}_{material}'
    print('\n' + '#' * 78)
    print(f'# {case_id}  (checkpoint: {ckpt_path})')
    print('#' * 78)
    assert os.path.exists(ckpt_path), f'{case_id}: checkpoint not found at {ckpt_path}'

    ckpt_fingerprint = sha256_of(ckpt_path)
    print(f'Checkpoint fingerprint (sha256): {ckpt_fingerprint}')

    args = argparse.Namespace(**BASE_ARGS)
    model = build_model(args, device).to(torch.float32)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print('Checkpoint loaded.')

    out_json = f'{R}/round12_consistent_field_qoi_{case_id}.json'
    fn = run_qoi_study_consistent_field_b1 if geometry == 'B1' else run_qoi_study_consistent_field_b2
    rows = fn(model, args, LOW_N, out_json, device, material=material,
              fine_N=FINE_N, checkpoint_fingerprint=ckpt_fingerprint)
    all_results[case_id] = rows

    print(f'\n--- {case_id} result, N=3..49 (FEM and operator, SAME field, SAME fine reference) ---')
    for r in rows:
        fem, no = r['fem'], r['no']
        print(f"  N={r['N']:<5} FEM L2={fem['l2_rel']*100:6.2f}%  Op L2={no['l2_rel']*100:6.2f}%   "
              f"FEM cauchy_avg={fem['cauchy_avg_rel_err']*100:6.2f}%  "
              f"Op cauchy_avg={no['cauchy_avg_rel_err']*100:6.2f}%")

print('\nAll six cases done.')

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='round12_consistent_field_qoi_all_cases',
                    args={'low_N': LOW_N, 'fine_N': FINE_N,
                          'cases': [c[0] + '_' + c[1] for c in CASES]},
                    started_at=_started, results={'cases': list(all_results)},
                    outputs=[f'{R}/round12_consistent_field_qoi_{c[0]}_{c[1]}.json' for c in CASES],
                    notes="Round-12 point 1, properly fixed: FEM and the operator now scored "
                          "on the SAME ParametricField(seed) realization, both against ONE "
                          "real fine reference (N=201), for classical QoI AND region-Cauchy "
                          "stress -- replaces the earlier version's same-N-not-fine-reference "
                          "and FEM/operator field-mismatch bugs, both caught by Omar.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
