# =====================================================================
#  CELL -- fills a real gap found while writing round-12 point 1 into the
#  Report/Summary: B1xNeo-Hookean (the flagship case) has NO operator-side
#  L2/H1/energy/reaction sweep at all in the LOW_N=[3..49] range for its
#  FINAL (multi-resolution-retrained) checkpoint. A same-named file
#  (no_accuracy_degradation_sweep.json, no case suffix) DOES exist on
#  Drive, but was checked (by hand, via its own stored
#  checkpoint_fingerprint and its N=1401 disp_rel_L2 of ~39-45%, wildly
#  inconsistent with the retrained checkpoint's own published 5.85%) and
#  confirmed to be from the OLD, pre-retrain checkpoint -- using it here
#  would silently mix two different model versions into one "final
#  checkpoint" table, exactly the kind of mistake this project's own
#  checkpoint-fingerprint safety net exists to catch. This cell reruns
#  the same sweep, same code path (run_accuracy_degradation_sweep,
#  unchanged), against the CURRENT retrained checkpoint instead, at the
#  SAME LOW_N range as the round-12 Cauchy sweep (Table 18-R10o/R10-9),
#  so this case's own "n/a" cells in Table 18-R10p/R10-11 (Report) and
#  R10-10/R10-11 (Summary) can be filled with a real number instead.
#
#  COST: cheap. Every resolution here (N<=49) is far below the N=1401
#  point that historically dominates this same sweep's cost (~27-34 min
#  for the full 16-point N=13..1401 range, per this project's own
#  run_manifest.json history) -- this cell's own 16 points top out at
#  N=49, and the round-12 Cauchy sweep already showed the SAME LOW_N
#  range (FEM sweep + a fresh fine-reference solve) completing in
#  13-25 SECONDS per case. Expect low minutes here, not tens of minutes.
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

# The FINAL (multi-resolution-retrained) checkpoint -- the SAME path every
# other round-12 cell uses -- NOT resolve_b1_neo_hookean_checkpoint (that
# helper verifies against the OLD zero-shot checkpoint's own fingerprint,
# which this retrained checkpoint will not and should not match).
CKPT = f'{R}/zeroshot_B1_neo_hookean_multires/model_best.pt'
assert os.path.exists(CKPT), f'checkpoint not found: {CKPT}'


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


ckpt_fingerprint = sha256_of(CKPT)
print(f'Checkpoint: {CKPT}')
print(f'Checkpoint fingerprint (sha256): {ckpt_fingerprint}')

OLD_PRERETRAIN_FINGERPRINT = '86030f4f05ea74f83079cee6b74485b30f2c1a5acac6acd5bc7a06e3adaa88f4'
if ckpt_fingerprint == OLD_PRERETRAIN_FINGERPRINT:
    raise RuntimeError(
        "This checkpoint's fingerprint matches the OLD, pre-retrain zero-shot "
        "checkpoint -- the multi-resolution retrain fix does not appear to be "
        "reflected at this path. Stopping rather than silently reproducing the "
        "exact stale numbers this cell exists to replace.")
print('Confirmed: this fingerprint DIFFERS from the old pre-retrain checkpoint '
      f'({OLD_PRERETRAIN_FINGERPRINT}) -- this really is the retrained model.')

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep
from omar_pfem.gpu_memory_monitor import GPUMemoryMonitor
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model = build_model(args, device).to(torch.float32)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()
print('Checkpoint loaded, cast to float32.')

# Same LOW_N range as the round-12 Cauchy sweep (Table 18-R10o/R10-9), so
# this fills in exactly the rows that table's own N-13-snapshot needs
# (Table 18-R10p/R10-q, Summary R10-10/R10-11) -- not the old sweep's own
# wider N=13..1401 range, which is not needed here and would waste time
# on the expensive N=101..1401 tail.
LOW_N = [3, 4, 5, 6, 9, 11, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

OUT_JSON = f'{R}/no_accuracy_degradation_sweep_B1_neo_hookean.json'
# Deliberately NOT the old unsuffixed no_accuracy_degradation_sweep.json --
# that name is the stale pre-retrain file and stays untouched; this is a
# new file under the per-case name round-12's own cell already expects.

gpu_monitor = GPUMemoryMonitor(device, interval_s=0.5)
gpu_monitor.__enter__()
rows = run_accuracy_degradation_sweep(model, args, LOW_N, OUT_JSON, device,
                                       checkpoint_fingerprint=ckpt_fingerprint)
gpu_monitor.__exit__(None, None, None)
GPU_MEM_FIG = f'{R}/fig_b1nh_final_lowN_accuracy_gpu_memory.png'
gpu_monitor.save_plot(GPU_MEM_FIG, title='GPU memory over time -- B1xNeo-Hookean final-checkpoint LOW_N accuracy sweep')

print('\n' + '=' * 78)
print('RESULT -- B1 x Neo-Hookean, FINAL (retrained) checkpoint, LOW_N=[3..49]')
print('=' * 78)
print(f"{'N':<6}{'converged':<12}{'disp_rel_L2':<14}{'L2_rel':<12}{'H1_semi_rel':<14}"
      f"{'energy_rel':<12}{'reaction_rel_err':<18}")
for r in rows:
    gt = r['ground_truth_convergence']
    fp32 = r['fp32']
    print(f"{r['N']:<6}{str(gt['converged_likely']):<12}{fp32['disp_rel_L2']:<14.4e}"
          f"{fp32['L2_rel']:<12.4e}{fp32['H1_semi_rel']:<14.4e}{fp32['energy_rel']:<12.4e}"
          f"{fp32['reaction_resultant_rel_err']:<18.4e}")

not_converged = [r['N'] for r in rows if not r['ground_truth_convergence']['converged_likely']]
if not_converged:
    print(f"\n*** WARNING: ground truth did NOT converge at N={not_converged} -- "
          f"exclude those rows before drawing any conclusion. ***")
else:
    print('\nGround truth converged at every N tested.')

print('\nSaved:', OUT_JSON)
print('This file uses the EXACT filename the round-12 Cauchy cell '
      '(cell_round12_final_accuracy_cauchy_all_cases.py) already looks for '
      '-- rerunning that notebook later will now find it automatically.')

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='no_accuracy_degradation_sweep_b1nh_final_lowN',
                    args={'low_N': LOW_N, 'checkpoint': CKPT, 'checkpoint_fingerprint': ckpt_fingerprint},
                    started_at=_started, results={'rows': len(rows)},
                    outputs=[OUT_JSON, GPU_MEM_FIG],
                    notes="Fills the real gap found while writing round-12 point 1 into the "
                          "Report/Summary: B1xNeo-Hookean's own operator-side L2/H1/energy/"
                          "reaction sweep, LOW_N=[3..49], FINAL retrained checkpoint (the "
                          "existing no_accuracy_degradation_sweep.json on Drive was verified "
                          "to be from the OLD pre-retrain checkpoint and intentionally not "
                          "reused).")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
