# =====================================================================
#  CELL -- B3: real training run with LOCAL DEM-integration refinement
#  near the groove region (Timon's request, 2026-09-28/29, item 1 of
#  the "still remaining" list: "Local refined DEM training integration").
#
#  This is NOT a resolution change. The operator's own mesh/
#  discretization is completely UNCHANGED -- same 6,840 elements
#  (21x20x19), same nodes, same architecture, same normalization, same
#  material/load distributions, same 50,000-iteration budget as run 4.
#  The ONLY difference: elements touching the groove region have their
#  own energy-integral quadrature refined (n_sub=10, giving 566 region
#  points, the level this project's own convergence sweep already found
#  stable) instead of the standard 8-point rule -- decoupling the
#  operator's discretization from the DEM background-integration mesh,
#  exactly as Timon asked.
#
#  Why this should NOT need ~11 GPU-hours like the earlier (confounded)
#  resolution-increase pilot: the element/node COUNT never changes here
#  -- only a handful of elements near the groove (2/2184 at toy scale;
#  a similarly small fraction expected at production scale) get extra
#  internal quadrature points. Per-iteration cost should stay close to
#  run 4's own ~0.14s/iteration, not the 0.784s/iteration the full
#  43,400-element resolution increase required. This run uses the FULL
#  50,000-iteration budget (not a truncated pilot) since there is no
#  large per-iteration cost increase to budget around -- avoiding the
#  training-budget confound that made the earlier finer-resolution pilot
#  inconclusive.
#
#  Verified locally on CPU before any GPU run, in four real steps (not
#  assumed): (1) an IDENTITY check -- at n_sub=2 (matching the standard
#  rule's own order), the locally-refined energy EXACTLY reproduces the
#  standard total_potential_energy_B3, to ~1e-12, for ANY region-element
#  mask (even a deliberately oversized half-mesh mask); (2) a real
#  refinement check -- at n_sub=4/8, the region elements' contribution
#  changes only a small, sane amount (<0.1% of total energy), not a
#  blow-up; (3) a gradient/backward check -- finite, nonzero gradients
#  flow through the locally-refined energy via ordinary autograd, no
#  special handling needed; (4) a full real toy-scale end-to-end
#  training run (real FEM data, real training loop, real checkpoint),
#  then evaluated with this project's own convergence-sweep tool, no
#  errors anywhere in the pipeline.
#
#  Evaluation after training reuses evaluate_B3_region_local_refine.py's
#  own convergence sweep unmodified, so the resulting checkpoint's
#  region-stress accuracy is directly comparable to run 4's already-
#  established ~62.2-62.3% converged baseline.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import subprocess
import sys
import time

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
if os.path.isdir('/content/drive/MyDrive'):
    print('Drive already mounted at /content/drive.')
else:
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

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

R = '/content/drive/MyDrive/pfem_run'
RUN4_DIR = f'{R}/b3_training_normalized'
RUN5_DIR = f'{R}/b3_training_local_refine'
os.makedirs(RUN5_DIR, exist_ok=True)

from omar_pfem.train_B3 import get_args, train, DEFAULT_RESOLUTION
import shutil

# SAME normalization as run 4 -- reused unchanged, not recomputed, for a
# genuinely controlled comparison (only the region-integration
# refinement differs from run 4).
norm_src = f'{RUN4_DIR}/input_norm.json'
norm_dst = f'{RUN5_DIR}/input_norm.json'
for _attempt in range(3):
    if os.path.exists(norm_src):
        break
    print(f'  [attempt {_attempt + 1}/3] {norm_src} not visible yet (Drive sync) -- retrying')
    time.sleep(3)
assert os.path.exists(norm_src), f'run 4 input_norm.json not found: {norm_src}'
shutil.copy(norm_src, norm_dst)
print(f"Reused run 4's own input_norm.json unchanged (copied to {norm_dst}).")

args = get_args([
    '--n_iters', '50000',
    '--batch_size', '8',
    '--log_every', '500',
    '--ckpt_every', '5000',
    '--output_dir', RUN5_DIR,
    '--normalize_inputs', '1',
    '--local_refine_region', '1',
    '--local_refine_n_sub', '10',
])
n_elem = (DEFAULT_RESOLUTION[0] - 1) * (DEFAULT_RESOLUTION[1] - 1) * (DEFAULT_RESOLUTION[2] - 1)
print(f'\nTraining at {DEFAULT_RESOLUTION} -> {n_elem} elements (SAME as run 4 -- operator '
      f'discretization unchanged), 50,000 iterations, batch_size=8, normalize_inputs=1, '
      f'local_refine_region=1 (n_sub=10, 566 region points).')
print('Watch the s/iteration rate in the first few hundred iterations -- expected close to '
      "run 4's own ~0.14s/iteration (only a handful of elements get extra quadrature). "
      'If it is dramatically higher, interrupt and tell Claude the observed rate.')

torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
t0 = time.time()
model = train(args, device)
elapsed_total = time.time() - t0

print(f'\n{"=" * 90}')
print(f'Training finished in {elapsed_total:.1f}s ({elapsed_total / args.n_iters:.4f}s/iteration average).')
if torch.cuda.is_available():
    print(f'GPU peak memory: {torch.cuda.max_memory_allocated(device) / 1e6:.1f}MB allocated, '
          f'{torch.cuda.max_memory_reserved(device) / 1e6:.1f}MB reserved.')

print(f'\n=== Evaluating the new checkpoint with the same convergence sweep as run 4 ===')
CHECKPOINT = f'{RUN5_DIR}/checkpoint_{args.n_iters}.pt'
DATASET = f'{R}/b3_dataset_clean_holdout/dataset.h5'
OUT_JSON = f'{RUN5_DIR}/region_local_refine_convergence.json'

sys.argv = [
    'evaluate_B3_region_local_refine.py',
    '--checkpoint', CHECKPOINT,
    '--dataset', DATASET,
    '--out_json', OUT_JSON,
    '--n_sub_sweep', '2', '4', '6', '8', '10', '12',
]
from omar_pfem.evaluate_B3_region_local_refine import main as evaluate_main
evaluate_main()

print('\nCompare the converged pooled_frobenius_rel_error above against run 4\'s own '
      '~62.2-62.3% baseline (same convergence sweep, same held-out set).')
print('\nDone.')
