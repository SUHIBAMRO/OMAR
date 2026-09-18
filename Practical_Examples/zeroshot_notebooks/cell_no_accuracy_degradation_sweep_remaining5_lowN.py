# =====================================================================
#  CELL -- closes the LAST real gap in round-12 point 1 (caught by Omar,
#  2026-09-18, before sending): Timon's own wording was "for EACH
#  resolution ... compare FEM and NO" -- the operator's own classical
#  QoIs (L2, H1, energy, reaction) are still "n/a" at N=3,4,5,6,9,11 for
#  the five cases beyond B1xNeo-Hookean (whose own gap at this same N
#  range was already closed by a dedicated follow-up run). This cell
#  closes it for the remaining five.
#
#  NOT a new training run and NOT the fine-reference re-solve the Cauchy
#  sweep needed -- this is pure evaluation, reusing the exact same
#  already-debugged pipeline (run_accuracy_degradation_sweep[_b2]) that
#  already produced these five cases' own N=13..49 rows. Each case's own
#  existing no_accuracy_degradation_sweep_<case>.json is RESUMED, not
#  replaced: passing the full LOW_N list lets the function skip every N
#  already present (13..49) and compute only the six missing ones
#  (3,4,5,6,9,11) -- the same resumable, checkpoint-fingerprint-checked
#  design already used for B1xNeo-Hookean's own follow-up.
#
#  COST: cheap, same order as B1xNeo-Hookean's own follow-up (46s of
#  real compute, 2m18s total) -- five cases at six new N each, all
#  N<=11, the cheapest end of the whole sweep.
# =====================================================================
import hashlib
import json
import os
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
        "cuDSS is NOT available after installing nvmath-python[cu12] -- the ground-truth "
        "solve below would silently fall back to an iterative solver, not the direct one "
        "this notebook is meant to use. Check the pip install output above for the real error.")
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
    run_accuracy_degradation_sweep, run_accuracy_degradation_sweep_b2)
import argparse

# (geometry, material, FINAL checkpoint -- same paths every other round-12
# cell uses) -- B1xNeo-Hookean deliberately excluded, already done.
CASES = [
    ('B1', 'mooney_rivlin', f'{R}/zeroshot_B1_mooney_rivlin_multires/model_best.pt'),
    ('B1', 'arruda_boyce', f'{R}/zeroshot_B1_arruda_boyce_multires/model_best.pt'),
    ('B2', 'neo_hookean', f'{R}/zeroshot_B2_neo_hookean_multires/model_best.pt'),
    ('B2', 'mooney_rivlin', f'{R}/zeroshot_B2_mooney_rivlin_multires/model_best.pt'),
    ('B2', 'arruda_boyce', f'{R}/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt'),
]

BASE_ARGS = dict(model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
                  mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
                  use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_in=1.0, R_out=2.0)

# Full LOW_N (not just the six new ones): the resumable sweep skips
# whatever it finds already in each case's own out_json, so passing the
# full list is both correct and simpler than hand-tracking which six are
# actually missing per case.
LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

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

    out_json = f'{R}/no_accuracy_degradation_sweep_{case_id}.json'
    fn = run_accuracy_degradation_sweep if geometry == 'B1' else run_accuracy_degradation_sweep_b2
    rows = fn(model, args, LOW_N, out_json, device, material=material,
              checkpoint_fingerprint=ckpt_fingerprint)
    all_results[case_id] = rows

    print(f'\n--- {case_id} result, N=3..49 ---')
    for r in rows:
        gt = r['ground_truth_convergence']
        fp32 = r['fp32']
        print(f"  N={r['N']:<5} converged={gt['converged_likely']!s:<6} "
              f"L2_rel={fp32['L2_rel']:.4e}  H1_semi_rel={fp32['H1_semi_rel']:.4e}  "
              f"energy_rel={fp32['energy_rel']:.4e}  "
              f"reaction_rel_err={fp32.get('reaction_resultant_rel_err')}")

print('\nAll five remaining cases done.')

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='no_accuracy_degradation_sweep_remaining5_lowN',
                    args={'low_N': LOW_N, 'cases': [c[0] + '_' + c[1] for c in CASES]},
                    started_at=_started, results={'cases': list(all_results)},
                    outputs=[f'{R}/no_accuracy_degradation_sweep_{c[0]}_{c[1]}.json' for c in CASES],
                    notes="Closes the last real gap in round-12 point 1 (Omar's own catch, "
                          "2026-09-18): the operator's classical QoIs (L2/H1/energy/reaction) "
                          "at N=3,4,5,6,9,11, for the five cases beyond B1xNeo-Hookean (whose "
                          "own gap at this range was already closed separately). Pure "
                          "evaluation against each case's own final checkpoint, resuming each "
                          "case's own existing sweep file rather than replacing it.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
