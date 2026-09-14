# =====================================================================
#  CELL -- re-verification of point 2 (max feasible batch size /
#  throughput) against the NEW multi-res checkpoint, built 2026-09-14.
#
#  WHY THIS EXISTS: the original max-feasible-batch-size measurement
#  (cell_max_feasible_batch_size.py) used the OLD checkpoint (trained on
#  N=21,33 only) -- the correct, fingerprint-verified one, but not the
#  new multi-res checkpoint the draft's point 1 now recommends. Timing
#  should be checkpoint-independent (same architecture, same parameter
#  count, only the WEIGHTS differ) -- but this project's own standing
#  rule is to verify that rather than assume it, exactly as was done
#  earlier for the wrong-vs-correct-checkpoint question. This cell is
#  that verification, pointed at the multi-res checkpoint instead.
#
#  Identical logic to cell_max_feasible_batch_size.py otherwise -- see
#  that file's own docstring for the full four-step rationale. Loads the
#  checkpoint by its KNOWN path and verifies its sha256 against the
#  fingerprint already recorded in no_accuracy_degradation_sweep_
#  multires.json (cb318c4694d820152d018bcb3a2caa6be4528f3654a59cc8aa
#  2482e9cd495f86) before trusting it.
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

R = '/content/drive/MyDrive/pfem_run'
OUT = f'{R}/zeroshot_B1_neo_hookean_multires'
CKPT = f'{OUT}/model_best.pt'
assert os.path.exists(CKPT), f'multi-res checkpoint not found: {CKPT}'

from omar_pfem.resolve_b1_checkpoint import sha256_of
EXPECTED_FP = 'cb318c4694d820152d018bcb3a2caa6be4528f3654a59cc8aa2482e9cd495f86'
actual_fp = sha256_of(CKPT)
print(f'{CKPT}\n  sha256={actual_fp}')
assert actual_fp == EXPECTED_FP, (
    f'Checkpoint fingerprint mismatch! Expected {EXPECTED_FP}, got {actual_fp}. '
    f'Refusing to proceed with an unverified checkpoint.')
print('Checkpoint identity verified by fingerprint -- this IS the multi-res retrained model.')

DATA = f'{R}/results/datasets/B1_neo_hookean/hyperelastic_training_data_q4.npz'
assert os.path.exists(DATA), f'dataset not found, update DATA: tried {DATA}'
print('Using dataset:', DATA)

OUT_DIR = f'{OUT}/max_feasible_batch'
os.makedirs(OUT_DIR, exist_ok=True)


def run_no(out_json, mem_budget_gb=None):
    cmd = [sys.executable, '-u', '-m', 'omar_pfem.inference_latency_by_batch',
           '--geometry', 'B1', '--material', 'neo_hookean',
           '--checkpoint', CKPT, '--dataset', DATA,
           '--ntrain', '800', '--ntest', '200',
           '--find_max_batch', '--out_json', out_json]
    if mem_budget_gb is not None:
        cmd += ['--mem_budget_gb', str(mem_budget_gb)]
    run(cmd)
    return json.load(open(out_json))


def run_fem(out_json, mem_budget_gb=None):
    cmd = [sys.executable, '-u', '-m', 'omar_pfem.gpu_fem_benchmark',
           '--geometry', 'B1', '--material', 'neo_hookean', '--N', '21',
           '--find_max_batch', '--out_json', out_json]
    if mem_budget_gb is not None:
        cmd += ['--mem_budget_gb', str(mem_budget_gb)]
    run(cmd)
    return json.load(open(out_json))


print('\n' + '=' * 70)
print('STEP 1/4: NO (multi-res), own natural max feasible batch size')
print('=' * 70)
no_natural = run_no(f'{OUT_DIR}/no_max_batch_natural.json')

print('\n' + '=' * 70)
print('STEP 2/4: FEM, own natural max feasible batch size')
print('=' * 70)
fem_natural = run_fem(f'{OUT_DIR}/fem_max_batch_natural.json')

no_peak_gb = no_natural['rows'][-1]['peak_memory_mb'] / 1024.0
fem_peak_gb = fem_natural['rows'][-1]['peak_memory_mb'] / 1024.0

print('\n' + '=' * 70)
print(f'STEP 3/4: FEM, capped at the NO\'s own peak memory ({no_peak_gb:.2f} GB)')
print('=' * 70)
fem_matched = run_fem(f'{OUT_DIR}/fem_max_batch_matched_to_no.json', mem_budget_gb=no_peak_gb)

print('\n' + '=' * 70)
print(f'STEP 4/4: NO (multi-res), capped at the FEM\'s own peak memory ({fem_peak_gb:.2f} GB)')
print('=' * 70)
no_matched = run_no(f'{OUT_DIR}/no_max_batch_matched_to_fem.json', mem_budget_gb=fem_peak_gb)

print('\n' + '=' * 70)
print('SUMMARY -- maximum feasible batch size and throughput, N=21 (multi-res checkpoint)')
print('=' * 70)
for name, res in [('NO multi-res (own memory ceiling)', no_natural),
                  ('FEM (own memory ceiling)', fem_natural),
                  (f'FEM (capped at NO\'s {no_peak_gb:.2f} GB)', fem_matched),
                  (f'NO multi-res (capped at FEM\'s {fem_peak_gb:.2f} GB)', no_matched)]:
    last = res['rows'][-1]
    print(f'  {name}: max_feasible_bs={res["max_feasible_batch_size"]}, '
          f'peak_mem={last["peak_memory_mb"]:.1f} MB, '
          f'throughput={last["throughput_samples_per_s"]:.2f} samples/s '
          f'(bs=1 throughput: {res["rows"][0]["throughput_samples_per_s"]:.2f} samples/s)')

# ---- Compare against the OLD checkpoint's own numbers ----
OLD_JSON = f'{R}/max_feasible_batch/no_max_batch_natural.json'
if os.path.exists(OLD_JSON):
    with open(OLD_JSON) as f:
        old = json.load(f)
    old_bs1 = old['rows'][0]['throughput_samples_per_s']
    new_bs1 = no_natural['rows'][0]['throughput_samples_per_s']
    old_max_thr = old['rows'][-1]['throughput_samples_per_s']
    new_max_thr = no_natural['rows'][-1]['throughput_samples_per_s']
    print('\n' + '=' * 70)
    print('OLD (N=21,33 checkpoint) vs. NEW (multi-res checkpoint) -- NO throughput')
    print('=' * 70)
    print(f"  bs=1 throughput:   old={old_bs1:.2f} samples/s   new={new_bs1:.2f} samples/s")
    print(f"  max-bs throughput: old={old_max_thr:.2f} samples/s   new={new_max_thr:.2f} samples/s")
    rel_diff = abs(new_max_thr - old_max_thr) / old_max_thr
    print(f"  relative difference at max bs: {rel_diff:.3e}")

print('\nAll four JSON files saved under', OUT_DIR)
