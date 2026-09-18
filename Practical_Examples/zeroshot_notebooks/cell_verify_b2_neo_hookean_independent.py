# =====================================================================
#  CELL -- INDEPENDENT third-party re-verification of the B2xNeo-Hookean
#  classical-QoI staleness finding (2026-09-18). Omar was not satisfied
#  with cross-referencing existing Drive files as proof and asked for a
#  completely fresh, from-scratch recomputation.
#
#  This writes to a BRAND-NEW output file
#  (no_accuracy_degradation_sweep_B2_neo_hookean_VERIFY.json) that has
#  never existed before, so the resumable sweep function has nothing to
#  skip and nothing cached to reuse -- every one of the sixteen
#  resolutions below is computed fresh, in front of you, this run.
#
#  What this checks: does the operator's own classical L2 error at N=13,
#  loading the SAME final checkpoint every other round-12 cell uses
#  (zeroshot_B2_neo_hookean_multires/model_best.pt), come out around
#  53.33% (the value now in the Report) or around 12.71% (the old,
#  stale, pre-final-retrain value that used to be in the Report)?
#
#  COST: same order as the B2xNeo-Hookean row alone in the earlier
#  Remaining5 run -- a small fraction of that notebook's own 6m30s
#  total (that run did five cases; this is one), so expect roughly
#  1-2 minutes of real GPU compute.
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
from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep_b2
import argparse

# Same path every other round-12 cell uses for this case (the Cauchy sweep
# that produced round12_final_accuracy_cauchy_summary.json, and the
# Remaining5_LowN run that first surfaced this discrepancy).
CKPT_PATH = f'{R}/zeroshot_B2_neo_hookean_multires/model_best.pt'
assert os.path.exists(CKPT_PATH), f'checkpoint not found at {CKPT_PATH}'

ckpt_fingerprint = sha256_of(CKPT_PATH)
print(f'Checkpoint path: {CKPT_PATH}')
print(f'Checkpoint fingerprint (sha256): {ckpt_fingerprint}')
EXPECTED_FINGERPRINT = 'dd2e244d5ac4d4454419fdc55b981d1e78c532d1d7d6458894ea1a74edb3287f'
if ckpt_fingerprint == EXPECTED_FINGERPRINT:
    print('  -> matches the fingerprint from both the 2026-09-17 post-retrain sanity '
          'sweep AND today\'s Remaining5_LowN_Accuracy.ipynb run.')
else:
    print('  *** WARNING: fingerprint does NOT match the earlier two runs -- the '
          'checkpoint file itself has changed since then. Investigate before trusting '
          'anything below. ***')

BASE_ARGS = dict(model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
                  mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
                  use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_in=1.0, R_out=2.0)
args = argparse.Namespace(**BASE_ARGS)
model = build_model(args, device).to(torch.float32)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print('Checkpoint loaded.')

LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

# Brand-new filename -- has never existed before, so nothing is skipped and
# nothing cached is reused. Every row below is computed fresh, this run.
OUT_JSON = f'{R}/no_accuracy_degradation_sweep_B2_neo_hookean_VERIFY.json'
assert not os.path.exists(OUT_JSON), (
    f'{OUT_JSON} already exists from a previous run of this exact cell -- delete it on '
    f'Drive first if you want a guaranteed from-scratch recompute, or this run will '
    f'correctly (but confusingly for this specific verification purpose) skip rows it '
    f'already computed.')

rows = run_accuracy_degradation_sweep_b2(
    model, args, LOW_N, OUT_JSON, device, material='neo_hookean',
    checkpoint_fingerprint=ckpt_fingerprint)

print('\n--- B2 x Neo-Hookean, independent from-scratch recomputation ---')
for r in rows:
    gt = r['ground_truth_convergence']
    fp32 = r['fp32']
    print(f"  N={r['N']:<5} converged={gt['converged_likely']!s:<6} "
          f"L2_rel={fp32['L2_rel']*100:.2f}%  H1_semi_rel={fp32['H1_semi_rel']*100:.2f}%  "
          f"energy_rel={fp32['energy_rel']*100:.2f}%")

n13 = next(r for r in rows if r['N'] == 13)
l2_13 = n13['fp32']['L2_rel'] * 100
print(f"\n=== THE DISPUTED NUMBER: N=13, L2_rel = {l2_13:.2f}% ===")
print(f"    Old, stale (pre-final-retrain checkpoint) value that used to be in the Report: 12.71%")
print(f"    Value now written into the Report and Summary (from 2026-09-18's fix):         53.33%")
if abs(l2_13 - 53.33) < 0.5:
    print("    -> This from-scratch, independent recomputation CONFIRMS the 53.33% figure.")
elif abs(l2_13 - 12.71) < 0.5:
    print("    -> This from-scratch, independent recomputation instead matches the OLD "
          "12.71% figure -- STOP, this contradicts the fix and needs investigation "
          "before trusting either number.")
else:
    print("    -> This from-scratch recomputation matches NEITHER prior number -- STOP, "
          "something else has changed (checkpoint, code, or environment) and needs "
          "investigation before trusting any of the three numbers.")

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='verify_b2_neo_hookean_independent',
                    args={'low_N': LOW_N, 'checkpoint': CKPT_PATH,
                          'checkpoint_fingerprint': ckpt_fingerprint},
                    started_at=_started, results={'N13_L2_rel_pct': l2_13},
                    outputs=[OUT_JSON],
                    notes="Independent, from-scratch (brand-new output file, nothing "
                          "resumed/reused) re-verification of the B2xNeo-Hookean "
                          "classical-QoI staleness finding, requested by Omar because "
                          "cross-referencing existing Drive files was not convincing "
                          "enough on its own.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
