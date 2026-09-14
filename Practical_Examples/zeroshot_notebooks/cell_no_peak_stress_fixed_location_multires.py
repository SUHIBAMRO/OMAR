# =====================================================================
#  CELL -- same fixed-location peak-stress check as
#  cell_no_peak_stress_fixed_location.py, but for the NEW multi-resolution
#  checkpoint (trained on N=21,33,101,201) instead of the old one (N=21,33
#  only). Built 2026-09-14 right after the multi-res retrain's own
#  accuracy-degradation comparison showed a dramatic improvement (N=1401
#  disp_rel_L2: 44.65% -> 5.85%) -- this fills in the one QoI that
#  comparison did not cover (peak PK1 stress at a fixed physical location),
#  needed to redo the "coarsest suitable FEM" crossover correctly for the
#  new checkpoint instead of reusing the old checkpoint's numbers.
#
#  Loads the checkpoint by its KNOWN path (the multi-res retrain notebook's
#  own OUT_NAME), then verifies its sha256 against the fingerprint already
#  recorded in no_accuracy_degradation_sweep_multires.json
#  (cb318c4694d820152d018bcb3a2caa6be4528f3654a59cc8aa2482e9cd495f86) before
#  trusting it -- same checkpoint-identity discipline as every other cell
#  in this project, not a new hardcode-and-hope.
#
#  COST: cheap, same as the original cell (one N=1401 ground-truth solve to
#  locate x_star, then 16 fast NO forward passes) -- a few minutes.
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
OUT = f'{R}/zeroshot_B1_neo_hookean_multires'
CKPT = f'{OUT}/model_best.pt'
assert os.path.exists(CKPT), f'multi-res checkpoint not found: {CKPT}'

from omar_pfem.resolve_b1_checkpoint import sha256_of
EXPECTED_FP = 'cb318c4694d820152d018bcb3a2caa6be4528f3654a59cc8aa2482e9cd495f86'
actual_fp = sha256_of(CKPT)
print(f'{CKPT}\n  sha256={actual_fp}')
assert actual_fp == EXPECTED_FP, (
    f'Checkpoint fingerprint mismatch! Expected {EXPECTED_FP} (recorded in '
    f'no_accuracy_degradation_sweep_multires.json), got {actual_fp}. Refusing to '
    f'proceed with an unverified checkpoint -- see PROJECT_STATUS.md for why this '
    f'check exists (a real wrong-checkpoint bug found 2026-09-12).')
print('Checkpoint identity verified by fingerprint -- this IS the multi-res retrained model.')

device = torch.device('cuda')

from omar_pfem.measure_inference_latency import build_model
from omar_pfem.no_accuracy_at_n1401 import run_no_peak_stress_fixed_location
import argparse

args = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model = build_model(args, device).to(torch.float32)
model.load_state_dict(torch.load(CKPT, map_location=device))
print('Checkpoint loaded, cast to float32.')

RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]

OUT_JSON = f'{OUT}/no_peak_stress_fixed_location_multires.json'
rows = run_no_peak_stress_fixed_location(model, args, RESOLUTIONS, OUT_JSON, device,
                                          fine_N_for_peak=1401)

print('\n' + '=' * 70)
print('RESULT -- NO (multi-res checkpoint) peak-stress error (fixed location)')
print('=' * 70)
for r in rows:
    print(f"  N={r['N']:<6} peak_stress_rel_err={r['peak_stress_rel_err']:.3e}")
print(json.dumps(rows, indent=2))

# ---- Compare against the OLD checkpoint's own fixed-location peak-stress numbers ----
OLD_JSON = f'{REPO}/Practical_Examples/omar_pfem/no_peak_stress_fixed_location_B1_neo_hookean.json'
if os.path.exists(OLD_JSON):
    with open(OLD_JSON) as f:
        old_rows = {r['N']: r['peak_stress_rel_err'] for r in json.load(f)['rows']}
    new_rows_by_n = {r['N']: r['peak_stress_rel_err'] for r in rows}
    print('\n' + '=' * 70)
    print('OLD vs NEW checkpoint -- fixed-location peak-stress error')
    print('=' * 70)
    print(f"{'N':<8}{'OLD':<12}{'NEW':<12}{'better?':<10}")
    for N in sorted(new_rows_by_n):
        if N in old_rows:
            better = 'YES' if new_rows_by_n[N] < old_rows[N] else 'no'
            print(f"{N:<8}{old_rows[N]:<12.4f}{new_rows_by_n[N]:<12.4f}{better:<10}")

# ---- Re-run the multi-QoI crossover with the NEW checkpoint's full QoI set ----
FEM_JSON = f'{REPO}/Practical_Examples/omar_pfem/torchfem_full_qoi_low_N_result.json'
NO_JSON = f'{REPO}/Practical_Examples/omar_pfem/no_accuracy_degradation_sweep_multires.json'
if os.path.exists(FEM_JSON) and os.path.exists(NO_JSON):
    with open(FEM_JSON) as f:
        fem_rows = sorted(json.load(f)['rows'], key=lambda r: r['N'])
    with open(NO_JSON) as f:
        no_rows = {r['N']: r for r in json.load(f)['rows']}
    no_peak = {r['N']: r['peak_stress_rel_err'] for r in rows}

    METRIC_PAIRS = [
        ('L2_rel', 'l2_rel', 'L2'),
        ('H1_semi_rel', 'h1_semi_rel', 'H1 semi-norm'),
        ('energy_rel', 'energy_norm_rel', 'tangent energy'),
        ('reaction_resultant_rel_err', 'reaction_resultant_rel_err', 'reaction resultant'),
    ]
    print('\n' + '=' * 78)
    print('MULTI-QoI CROSSOVER, NEW (multi-res) checkpoint vs. torch-fem low-N')
    print('=' * 78)
    for N in sorted(no_rows):
        if N not in no_peak:
            continue
        print(f"\nNO (multi-res) at N={N}:")
        per_metric_crossover = {}
        for no_key, fem_key, label in METRIC_PAIRS:
            no_val = no_rows[N]['fp32'].get(no_key)
            match = next((fr for fr in fem_rows if fr.get(fem_key) is not None
                          and fr[fem_key] <= no_val), None)
            if match:
                per_metric_crossover[label] = match['N']
                print(f"    {label:<18} NO={no_val:.3e}  ->  torch-fem matches at N={match['N']}")
        no_peak_val = no_peak[N]
        match = next((fr for fr in fem_rows if fr.get('peak_stress_rel_err') is not None
                      and fr['peak_stress_rel_err'] <= no_peak_val), None)
        if match:
            per_metric_crossover['peak PK1 stress'] = match['N']
            print(f"    {'peak PK1 stress':<18} NO={no_peak_val:.3e}  ->  torch-fem matches at N={match['N']}")
        else:
            print(f"    {'peak PK1 stress':<18} NO={no_peak_val:.3e}  ->  torch-fem needs N > "
                  f"{fem_rows[-1]['N']} (slow-converging, possibly near a domain-corner "
                  f"singularity -- see PROJECT_STATUS.md)")
        if per_metric_crossover:
            coarsest_suitable = max(per_metric_crossover.values())
            binding = [k for k, v in per_metric_crossover.items() if v == coarsest_suitable]
            print(f"  => COARSEST SUITABLE FEM for NO(multi-res)@N={N}: N={coarsest_suitable} "
                  f"(binding metric: {', '.join(binding)})")
else:
    print(f"\n(FEM full-QoI sweep or NEW NO accuracy sweep not found locally -- pull/commit "
          f"them first for the crossover table)")

print('\nSaved:', OUT_JSON)
