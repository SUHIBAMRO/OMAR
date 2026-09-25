# =====================================================================
#  CELL -- B3 (sharper groove) Transolver training: the FIRST real GPU
#  run of train_B3.py's Deep Energy Method training loop.
#
#  Verified locally (CPU) before this, not assumed: a full train() call
#  ran end to end with no shape errors, and a separate fixed-batch
#  overfitting test (200 iterations, same sampled batch every step)
#  showed the energy loss decrease monotonically from 130.7 to a stable
#  plateau around 2.25 -- real confirmation gradients flow correctly
#  through the whole pipeline. This cell is the first time the ACTUAL
#  training loop (a new random batch every iteration, not a fixed one)
#  runs at GPU scale, at the real (21,20,19)=6,840-element mesh.
#
#  Real CPU timing measured directly beforehand: ~20-22s/iteration at
#  batch_size=8 (steady state, after the first iteration's one-time
#  model/geometry setup cost). GPU should be substantially faster, but
#  this specific pipeline has never been timed on GPU before -- watch
#  the first several printed iterations closely for the real number
#  before assuming the rest of the 2000 iterations will finish quickly.
#
#  Because every iteration draws a NEW random material/load batch (by
#  design -- Deep Energy Method training needs no fixed dataset), the
#  printed loss is NOT expected to decrease monotonically iteration to
#  iteration the way it does on a fixed batch: different batches have
#  different absolute energy scales. Watch the TREND over many logged
#  rows, not single-step changes.
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
print(f'GPU memory at start: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, '
      f'{torch.cuda.memory_reserved()/1e9:.2f} GB reserved (should be ~0 either way -- '
      f'if not, the runtime was not actually restarted and still holds an earlier '
      f'crash alive; use Runtime > Restart session, not just re-running this cell)')

from omar_pfem.train_B3 import get_args, train, DEFAULT_RESOLUTION

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
OUTPUT_DIR = f'{R}/b3_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

args = get_args([
    '--n_iters', '2000',
    '--batch_size', '8',
    '--log_every', '20',
    '--ckpt_every', '200',
    '--output_dir', OUTPUT_DIR,
])
print(f'\nTraining at {DEFAULT_RESOLUTION} -> '
      f'{(DEFAULT_RESOLUTION[0]-1)*(DEFAULT_RESOLUTION[1]-1)*(DEFAULT_RESOLUTION[2]-1)} elements, '
      f'{args.n_iters} iterations, batch_size={args.batch_size}, checkpoints -> {OUTPUT_DIR}')
print('This is the FIRST real GPU run of the actual (non-fixed-batch) training loop -- '
      'watch the first several rows closely before assuming the rest will behave the same way.')

torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()
model = train(args, device)
elapsed_total = time.time() - t0

print(f'\n{"=" * 90}')
print(f'Training finished in {elapsed_total:.1f}s ({elapsed_total / args.n_iters:.3f}s/iteration average).')
print(f'GPU peak memory during this run: '
      f'{torch.cuda.max_memory_allocated(device) / 1e6:.1f}MB allocated, '
      f'{torch.cuda.max_memory_reserved(device) / 1e6:.1f}MB reserved.')

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(OUTPUT_DIR, kind='b3_transolver_training',
                    args={'n_iters': args.n_iters, 'batch_size': args.batch_size,
                          'resolution': DEFAULT_RESOLUTION, 'lr': args.lr},
                    started_at=_started,
                    results={'elapsed_total_s': elapsed_total,
                             's_per_iter': elapsed_total / args.n_iters},
                    outputs=[f'{OUTPUT_DIR}/checkpoint_{args.n_iters}.pt'],
                    notes="First real GPU run of train_B3.py's actual (non-fixed-batch) "
                          "Deep Energy Method training loop for the 3D Transolver. "
                          "Verified locally beforehand: a full train() call ran end to "
                          "end with no shape errors, and a fixed-batch overfitting test "
                          "showed the energy loss decrease monotonically and plateau -- "
                          "real confirmation the pipeline is correctly wired -- but this "
                          "is the first time actual training (new random batch every "
                          "iteration) has run at GPU scale.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
