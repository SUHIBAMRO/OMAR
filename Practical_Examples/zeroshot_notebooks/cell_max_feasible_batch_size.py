# =====================================================================
#  CELL -- maximum feasible batch size and throughput (samples/s) for
#  the NO vs. the GPU-native FEM solver (Timon round-10, item 2):
#  "It would be interesting to additionally report the maximum feasible
#  batch size and throughput (samples/s) for both approaches at the
#  same GPU memory. I expect that the NO should benefit from batching
#  but this should be demonstrated."
#
#  Both inference_latency_by_batch.py (NO) and gpu_fem_benchmark.py
#  (FEM) already existed (they produced Tables 10a-c's own numbers at
#  fixed batch sizes 1/8/32/128), but neither tracked peak GPU memory
#  or searched for a maximum feasible batch size -- both gained a new
#  --find_max_batch flag (shared search logic in
#  omar_pfem/max_feasible_batch.py, doubling batch size from 1 until a
#  real OOM or a --mem_budget_gb cap, reporting peak memory and
#  throughput at every size actually run) rather than a new script.
#
#  Runs FOUR configurations, all at N=21 (matching Tables 10a-c's own
#  convention -- both sides measured at the study's standard
#  resolution):
#    1. NO,  its own natural max feasible batch size (no memory cap)
#    2. FEM, its own natural max feasible batch size (no memory cap)
#    3. FEM, capped at the NO's own peak memory from (1) -- the "same
#       GPU memory" comparison Timon asked for, FEM's side
#    4. NO,  capped at the FEM's own peak memory from (2) -- the "same
#       GPU memory" comparison Timon asked for, NO's side
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
# BUG FOUND 2026-09-12: a hardcoded path here ('results/checkpoints/
# B1_neo_hookean/model_best.pt', which never existed on Drive) silently
# fell back to 'data_driven/B1_neo_hookean/model_best.pt' -- a COMPLETELY
# DIFFERENT model (train_data_driven.py's own data-driven-loss baseline
# from the round-5/6 comparison study). Fixed properly this time:
# resolve by CONTENT (sha256), verified against the zero-shot study's own
# already-trusted checkpoint fingerprint, not by guessing a path --
# see resolve_b1_checkpoint.py's own docstring for the full story.
from omar_pfem.resolve_b1_checkpoint import resolve_b1_neo_hookean_checkpoint
CKPT, _ckpt_fp = resolve_b1_neo_hookean_checkpoint(R)
print(f'Resolved checkpoint (verified by fingerprint): {CKPT}')
DATA = f'{R}/results/datasets/B1_neo_hookean/hyperelastic_training_data_q4.npz'
assert os.path.exists(DATA), f'dataset not found, update DATA: tried {DATA}'
print('Using checkpoint:', CKPT)
print('Using dataset:', DATA)

OUT_DIR = f'{R}/max_feasible_batch'
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
print('STEP 1/4: NO, own natural max feasible batch size')
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
print(f'STEP 4/4: NO, capped at the FEM\'s own peak memory ({fem_peak_gb:.2f} GB)')
print('=' * 70)
no_matched = run_no(f'{OUT_DIR}/no_max_batch_matched_to_fem.json', mem_budget_gb=fem_peak_gb)

print('\n' + '=' * 70)
print('SUMMARY -- maximum feasible batch size and throughput, N=21')
print('=' * 70)
for name, res in [('NO (own memory ceiling)', no_natural),
                  ('FEM (own memory ceiling)', fem_natural),
                  (f'FEM (capped at NO\'s {no_peak_gb:.2f} GB)', fem_matched),
                  (f'NO (capped at FEM\'s {fem_peak_gb:.2f} GB)', no_matched)]:
    last = res['rows'][-1]
    print(f'  {name}: max_feasible_bs={res["max_feasible_batch_size"]}, '
          f'peak_mem={last["peak_memory_mb"]:.1f} MB, '
          f'throughput={last["throughput_samples_per_s"]:.2f} samples/s '
          f'(bs=1 throughput: {res["rows"][0]["throughput_samples_per_s"]:.2f} samples/s)')
print('\nAll four JSON files saved under', OUT_DIR)

# ---- Figure ----------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import PRIMARY, SECONDARY

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

for res, label, color in [(no_natural, 'NO', PRIMARY), (fem_natural, 'FEM', SECONDARY)]:
    bs = [r['batch_size'] for r in res['rows']]
    thr = [r['throughput_samples_per_s'] for r in res['rows']]
    mem = [r['peak_memory_mb'] / 1024.0 for r in res['rows']]
    ax1.plot(bs, thr, 'o-', color=color, label=label)
    ax2.plot(bs, mem, 'o-', color=color, label=label)

ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Batch size')
ax1.set_ylabel('Throughput (samples/s)')
ax1.set_title('Throughput vs. batch size (own memory ceiling)')
ax1.grid(True, which='both', alpha=0.25)
ax1.legend()

ax2.set_xscale('log', base=2)
ax2.set_xlabel('Batch size')
ax2.set_ylabel('Peak GPU memory (GB)')
ax2.set_title('Peak memory vs. batch size')
ax2.grid(True, which='both', alpha=0.25)
ax2.legend()

fig.suptitle('Max feasible batch size, NO vs. GPU-FEM (N=21)')
fig.tight_layout()
FIG_PATH = f'{OUT_DIR}/fig_max_feasible_batch_size.png'
fig.savefig(FIG_PATH)
print('Saved figure:', FIG_PATH)
