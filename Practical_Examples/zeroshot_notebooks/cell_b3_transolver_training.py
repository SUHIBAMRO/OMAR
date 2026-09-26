# =====================================================================
#  CELL -- B3 (sharper groove) Transolver training: THIRD real GPU run --
#  same stable training loop as the second run, just MUCH longer.
#
#  REAL INCIDENT, 2026-09-25 (first run): loss climbed from ~13,700 to a
#  peak over 163,000 -- a real optimization instability, root-caused to
#  the untrained network's raw output (~0.31) being 5-50x larger than
#  the true displacement scale (~0.01-0.06), pushing element deformation
#  gradients into the singular region of the Neo-Hookean energy near
#  det(F)->0. Fixed via `OUTPUT_SCALE = 0.02` in train_B3.py's
#  `apply_dirichlet_b3` -- see that module's docstring for the full
#  record. The second run (2000 iterations, with the fix) was stable
#  throughout (loss ~1.6-5.1, ending 1.56) -- confirmed the instability
#  was gone.
#
#  REAL FINDING, 2026-09-26 (second run's ACCURACY, not stability):
#  evaluating checkpoint_2000.pt against the 100 real FEM samples
#  (evaluate_B3.py / B3_Evaluate.ipynb) showed the loss curve being
#  healthy said nothing about accuracy -- relative L2 error was 32.0%
#  (ux), 99.95% (uy), 36.8% (uz), 35.7% (combined). Before assuming this
#  needed an architecture change, a controlled toy-scale diagnostic
#  tested whether the FROZEN OUTPUT_SCALE=0.02 was capping accuracy: a
#  10x larger frozen scale (0.2) and a LEARNABLE scale (init 0.02, in
#  the optimizer) were both tried at the same toy iteration budget.
#  Neither helped -- uy stayed stuck near 100% error in every variant,
#  and the larger/learnable scale actually made ux/uz slightly WORSE
#  (combined rel L2 0.306 at scale=0.02 vs 0.354 at scale=0.2 vs 0.371
#  learnable) -- so this hypothesis was FALSIFIED, not confirmed, by a
#  real controlled test, not assumed. OUTPUT_SCALE stays at 0.02
#  unchanged.
#
#  REAL PRECEDENT NUMBERS, from this project's own actually-recorded B1/B2
#  runs (point7a_results/*.json, not the argparse defaults): B1's three
#  cases reached their best checkpoint at 57,500 (mooney_rivlin), 65,000
#  (neo_hookean) and 70,000 (arruda_boyce) gradient steps, at ~5-10%
#  per-component error. B2's neo_hookean case reached 3.3% per-component
#  error only at 275,000 steps. B3's first two runs used just 2000 steps
#  -- nowhere close. Omar's call (2026-09-26): skip an intermediate
#  20,000-step checkpoint and go straight to 50,000 -- still short of
#  every real precedent above, but the largest single run worth
#  committing to before re-diagnosing, rather than spending two separate
#  GPU sessions to get there in stages.
#
#  The checkpoint from the FIRST run (before the OUTPUT_SCALE fix)
#  remains invalid/diverged. The SECOND run's checkpoint_2000.pt is
#  numerically valid (stable loss) but its accuracy is poor per the
#  finding above -- do not treat it as a final result.
#
#  Because every iteration still draws a NEW random material/load batch
#  (by design), the printed loss is still not expected to decrease
#  monotonically iteration to iteration the way it does on a fixed
#  batch -- watch the TREND over many logged rows.
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
    '--n_iters', '50000',
    '--batch_size', '8',
    '--log_every', '500',
    '--ckpt_every', '5000',
    '--output_dir', OUTPUT_DIR,
])
print(f'\nTraining at {DEFAULT_RESOLUTION} -> '
      f'{(DEFAULT_RESOLUTION[0]-1)*(DEFAULT_RESOLUTION[1]-1)*(DEFAULT_RESOLUTION[2]-1)} elements, '
      f'{args.n_iters} iterations, batch_size={args.batch_size}, checkpoints -> {OUTPUT_DIR}')
print('This is the THIRD real GPU run -- same stable loop as run 2, 25x longer. Testing '
      'whether more gradient steps closes the accuracy gap found by evaluating run 2\'s '
      'checkpoint, at a budget close to (though still short of) this project\'s own real '
      'B1/B2 precedent (57,500-275,000 steps at their best checkpoints). Expect roughly '
      '25x run 2\'s wall-clock time (run 2: 280s for 2000 iters -> ~2 hours).')

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
                    notes="Third real GPU run of train_B3.py's Deep Energy Method training "
                          "loop -- same stable loop as run 2 (OUTPUT_SCALE=0.02 fix in "
                          "place), 25x more iterations (50000 vs 2000, skipping an "
                          "intermediate 20000-step checkpoint per Omar's call to avoid two "
                          "separate GPU sessions). Run 2's checkpoint was numerically stable "
                          "but evaluated poorly against real FEM ground truth "
                          "(evaluate_B3.py: combined rel L2 35.7%, uy 99.95%). A controlled "
                          "toy-scale diagnostic falsified the hypothesis that OUTPUT_SCALE "
                          "itself was capping accuracy (a 10x larger and a learnable scale "
                          "both failed to help, and slightly hurt ux/uz), so this run "
                          "instead tests the much more likely cause found by direct "
                          "comparison with this project's own real B1/B2 training records "
                          "(point7a_results/*.json): B1's three cases reached their best "
                          "checkpoint at 57,500-70,000 steps (~5-10% error), B2's "
                          "neo_hookean case at 275,000 steps (3.3% error) -- B3's first two "
                          "runs used only 2000 steps.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
