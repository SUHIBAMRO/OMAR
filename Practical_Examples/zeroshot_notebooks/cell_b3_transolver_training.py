# =====================================================================
#  CELL -- B3 (sharper groove) Transolver training: FOURTH real GPU run --
#  same stable loop and 50,000-iteration budget as run 3, this time with
#  --normalize_inputs 1 added.
#
#  History in brief: run 1 diverged (a real instability, fixed via
#  OUTPUT_SCALE=0.02 -- see train_B3.py's own docstring). Run 2 (2000
#  iters) was numerically stable but evaluated poorly (combined rel L2
#  35.7%, uy 99.95%). Run 3 (50,000 iters, per this project's own real
#  B1/B2 precedent of 57,500-275,000 steps at their best checkpoints)
#  did not close the gap -- worse, `B3_Checkpoint_Sweep.ipynb` (real GPU
#  data, clean held-out set) showed uy swinging non-monotonically
#  between 79% and 111% across the 10 saved checkpoints, with the FINAL
#  checkpoint (50000) actually worse than an earlier one (35000) -- a
#  real training-instability signature, not a converging trend.
#
#  REAL FIX FOUND, 2026-09-26/27: a controlled A/B test (same fixed
#  8-sample pool, same seed, 4000 iterations each) compared raw E/nu/phi
#  inputs (E~1000, nu~0.45, phi~0.05 -- three orders of magnitude apart,
#  fed completely unnormalized, as every prior B3 run did) against
#  standardized inputs:
#
#    A (raw, baseline): combined rel L2 10.7%, uy 54.1%, loss oscillates
#    B (normalized):    combined rel L2  1.1%, uy  4.0%, loss smooth
#                        and monotonic, settling by iteration ~800
#
#  A ~10x drop in combined error and the complete disappearance of the
#  oscillation pattern -- the clearest, most decisive result of the
#  whole B3 investigation. Caveat stated plainly: that A/B test was a
#  memorization test on a fixed pool, not the real streaming/
#  generalization setting -- THIS run, evaluated afterward on the clean
#  held-out set (b3_dataset_clean_holdout/), is the test that actually
#  matters.
#
#  Saves to a SEPARATE output dir (b3_training_normalized, not run 3's
#  b3_training) so the new input_norm.json this run writes can never be
#  mistakenly applied to run 3's own unnormalized checkpoints by a
#  later evaluation call.
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
OUTPUT_DIR = f'{R}/b3_training_normalized'
os.makedirs(OUTPUT_DIR, exist_ok=True)

args = get_args([
    '--n_iters', '50000',
    '--batch_size', '8',
    '--log_every', '500',
    '--ckpt_every', '5000',
    '--output_dir', OUTPUT_DIR,
    '--normalize_inputs', '1',
])
print(f'\nTraining at {DEFAULT_RESOLUTION} -> '
      f'{(DEFAULT_RESOLUTION[0]-1)*(DEFAULT_RESOLUTION[1]-1)*(DEFAULT_RESOLUTION[2]-1)} elements, '
      f'{args.n_iters} iterations, batch_size={args.batch_size}, normalize_inputs={args.normalize_inputs}, '
      f'checkpoints -> {OUTPUT_DIR}')
print('This is the FOURTH real GPU run -- same 50,000-iteration budget as run 3, with '
      '--normalize_inputs 1 added. A controlled fixed-pool A/B test showed this drops '
      'combined rel L2 from 10.7% to 1.1% and removes the oscillation that made run 3\'s '
      'final checkpoint worse than an earlier one -- but that test was memorization, not '
      'generalization, so THIS run (evaluated on the clean held-out set afterward) is what '
      'actually confirms it. Expect roughly the same wall-clock as run 3 (~2 hours).')

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
                          'resolution': DEFAULT_RESOLUTION, 'lr': args.lr,
                          'normalize_inputs': args.normalize_inputs},
                    started_at=_started,
                    results={'elapsed_total_s': elapsed_total,
                             's_per_iter': elapsed_total / args.n_iters},
                    outputs=[f'{OUTPUT_DIR}/checkpoint_{args.n_iters}.pt',
                             f'{OUTPUT_DIR}/input_norm.json'],
                    notes="Fourth real GPU run of train_B3.py's Deep Energy Method training "
                          "loop -- same 50,000-iteration budget as run 3, with "
                          "--normalize_inputs 1 added (new code, verified locally first). "
                          "Run 3's checkpoint sweep (real GPU data, clean held-out set) "
                          "showed uy swinging non-monotonically between 79-111% across the "
                          "10 saved checkpoints, with the final one (50000) actually worse "
                          "than an earlier one (35000) -- a real training-instability "
                          "signature. A controlled fixed-pool A/B test (same seed, 4000 "
                          "iterations each) showed normalizing E/nu/phi (raw scales 1000, "
                          "0.45, 0.05 -- three orders of magnitude apart) drops combined "
                          "rel L2 from 10.7% to 1.1% and removes the oscillation entirely, "
                          "but that test was memorization on a fixed pool, not the real "
                          "streaming/generalization setting -- this run is the real test.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
