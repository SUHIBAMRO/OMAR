# =====================================================================
#  CELL -- B3 clean, genuinely disjoint held-out validation set.
#
#  REAL BUG FOUND, 2026-09-26 (Omar's methodology review): the original
#  dataset.h5 (pfem_run/b3_dataset/) used seed=0, giving sample_seed=i
#  for i=0..99 -- i.e. seeds 0-99. But train_B3.py's sample_batch draws
#  seed = it*batch_size + b, which for it=1..12 at batch_size=8 sweeps
#  seeds 8-103 -- so 92 of the 100 "held-out" samples' exact (E,nu,phi)
#  were ALSO used as real training inputs, within the first 12
#  iterations of every training run (run 2 AND run 3 both). Confirmed by
#  direct computation, not assumed. This dataset uses seed=99999
#  (sample_seed range 999990-1000089) -- unreachable by any training run
#  so far or any reasonable future one (a 50,000-iteration run at
#  batch_size=8 only reaches seed ~400,007) -- so it is a genuinely
#  clean held-out set. Use THIS dataset for any accuracy claim going
#  forward, not the original dataset.h5.
#
#  Same pipeline as cell_b3_dataset_generation.py, verified already on
#  CPU in this session (100/100 succeeded at this exact seed, locally,
#  before this GPU run) -- this GPU run just reproduces it much faster.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

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

run([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista<0.49'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if (_mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.')
            or _mod_name == 'torchfem' or _mod_name.startswith('torchfem.')
            or _mod_name == 'pyvista' or _mod_name.startswith('pyvista.')):
        del sys.modules[_mod_name]

import numpy as np
import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

import gc
for _attr in ('last_traceback', 'last_value', 'last_type'):
    if hasattr(sys, _attr):
        delattr(sys, _attr)
gc.collect()
torch.cuda.empty_cache()

from omar_pfem.data.data_generate_B3_dataset import generate_dataset, DEFAULT_RESOLUTION

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
OUTPUT_DIR = f'{R}/b3_dataset_clean_holdout'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 100
SEED = 99999  # disjoint from every training seed -- see module docstring
Ntheta, Nr, Nz = DEFAULT_RESOLUTION
print(f'\nGenerating {NUM_SAMPLES} CLEAN held-out samples at seed={SEED} '
      f'(Ntheta,Nr,Nz)={DEFAULT_RESOLUTION} -> '
      f'{(Ntheta-1)*(Nr-1)*(Nz-1)} elements, saving to {OUTPUT_DIR}')

torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()
manifest = generate_dataset(
    NUM_SAMPLES, OUTPUT_DIR, Ntheta=Ntheta, Nr=Nr, Nz=Nz, seed=SEED, device=device,
    verbose_every=5,
)
elapsed_total = time.time() - t0

print(f'\n{"=" * 90}')
print(f'Dataset generation finished in {elapsed_total:.1f}s '
      f'({elapsed_total / max(len(manifest["successful_samples"]), 1):.2f}s/sample average).')
print(f'GPU peak memory during this run: '
      f'{torch.cuda.max_memory_allocated(device) / 1e6:.1f}MB allocated, '
      f'{torch.cuda.max_memory_reserved(device) / 1e6:.1f}MB reserved.')
print(f'Succeeded: {len(manifest["successful_samples"])}/{NUM_SAMPLES}, '
      f'Failed: {len(manifest["failed"])}')
if manifest['failed']:
    print('Failed sample indices and errors:')
    for idx, err in manifest['failed'].items():
        print(f'  [{idx}] {err}')

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(OUTPUT_DIR, kind='b3_dataset_clean_holdout',
                    args={'num_samples': NUM_SAMPLES, 'resolution': DEFAULT_RESOLUTION,
                          'seed': SEED, 'groove_depth': 0.20, 'groove_half_width': 0.15},
                    started_at=_started,
                    results={'succeeded': len(manifest['successful_samples']),
                             'failed': len(manifest['failed']),
                             'elapsed_total_s': elapsed_total},
                    outputs=[f'{OUTPUT_DIR}/dataset.h5', f'{OUTPUT_DIR}/manifest.json'],
                    notes="Genuinely disjoint B3 held-out set (seed=99999), replacing the "
                          "original dataset.h5 whose seeds (0-99) overlapped 92/100 with "
                          "real training inputs (seed=it*batch_size+b in train_B3.py). "
                          "Verified locally on CPU at this exact seed before this GPU run: "
                          "100/100 succeeded.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
