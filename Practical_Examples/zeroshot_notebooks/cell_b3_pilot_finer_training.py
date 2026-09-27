# =====================================================================
#  CELL -- B3: clean diagnostic pilot -- does training at a resolution
#  with enough Gauss points in the groove region (43,400 elements, 36
#  points, vs. 6 at production) measurably reduce region-stress error?
#
#  Why this exists: the real GPU finer-EVALUATION experiment showed the
#  TRUE region_avg_sigma_xx itself shifts by ~2x for most held-out
#  samples and by 20-79x (even flipping sign) for near-zero ones,
#  between 6,840 and 43,400 elements -- direct evidence B3's training
#  resolution does not give a reliable local-stress target. But the
#  network's own prediction barely moved between the two evaluation
#  resolutions of that SAME already-trained checkpoint, staying in a
#  fixed, large-magnitude, mostly-negative range -- so it is not yet
#  established whether training on a mesh with more resolution IN THE
#  REGION would actually help, or whether the real bottleneck is that a
#  small, sign-changing local quantity contributes negligibly to the
#  global energy objective regardless of mesh size (Omar's own framing,
#  2026-09-27: "the 6,840 mesh and the global-energy objective did not
#  give enough resolution for local stress gradients in the groove
#  region -- the cause could be resolution, could be the region's small
#  weight within the total energy, or both").
#
#  This is a CLEAN, MINIMAL diagnostic, not a production run: SAME
#  architecture, SAME normalization (the run-4 input_norm.json is
#  copied in and reused as-is, not recomputed), SAME material/load
#  distributions, SAME all other hyperparameters as run 4 -- the ONLY
#  thing that changes is mesh resolution (21,20,19 -> 41,36,32) and the
#  iteration BUDGET (a short pilot, not the full 50,000). Evaluated on a
#  genuinely INDEPENDENT FEM held-out set solved fresh AT THE SAME
#  43,400-element resolution (not the 6,840-element clean_holdout set,
#  and not an interpolated re-solve of it -- a real new dataset, disjoint
#  seed from anything the pilot's own on-the-fly training touches).
#
#  What decides the pilot, per Omar's own criterion (NOT the training
#  loss): does region_cauchy_field_rel drop clearly (e.g. from ~72% at
#  production resolution toward 20-30%) while displacement/energy/
#  reaction stay good? If yes -> resolution is a real lever, worth a
#  longer/multi-resolution run. If it stays 60-80% regardless -> the
#  bottleneck is likely the objective itself (a local-stress supervision
#  term, or a training-strategy change), not mesh size, and 8 hours
#  should not be spent on the same idea.
#
#  43,400 elements is NOT proposed as a final resolution for the paper
#  -- this project's own earlier FEM-only convergence study already
#  showed the real region-stress target (5-10%) needs much finer meshes,
#  near 480k elements (vs. the 950k-element reference). This pilot only
#  asks the narrower, cheaper diagnostic question above.
#
#  Every piece reused unmodified from already-verified code: dataset
#  generation (data_generate_B3_dataset.generate_dataset), training
#  (train_B3.train/get_args, resolution= kwarg), and evaluation
#  (evaluate_B3_qois.main, --resolution/--dataset are already fully
#  parametric) -- no new physics or metric code, only new orchestration,
#  verified locally at toy scale before this real run.
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import shutil
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
PILOT_DIR = f'{R}/b3_pilot_finer_43400'
PILOT_HOLDOUT_DIR = f'{R}/b3_dataset_pilot_holdout_43400'
os.makedirs(PILOT_DIR, exist_ok=True)

FINE_RESOLUTION = (41, 36, 32)   # 43,400 elements, 36 region Gauss points (vs. 6 at production)
N_ITERS_PILOT = 5000              # short budget, NOT the full 50,000 -- see cell docstring
N_HOLDOUT_SAMPLES = 20            # independent FEM samples, solved fresh at FINE_RESOLUTION
HOLDOUT_SEED = 99999              # same convention as the production clean_holdout set;
                                   # comfortably disjoint from the pilot's own on-the-fly
                                   # training seeds (0 .. n_iters*batch_size-1 = 0..39999)

n_elem = (FINE_RESOLUTION[0] - 1) * (FINE_RESOLUTION[1] - 1) * (FINE_RESOLUTION[2] - 1)
print(f'\nPilot resolution: {FINE_RESOLUTION} -> {n_elem:,} elements')
print(f'Pilot iteration budget: {N_ITERS_PILOT} (run 4 used 50,000 -- this is a short diagnostic, not a production run)')

# ---------------------------------------------------------------------
# Step 1: genuinely independent FEM held-out set, solved FRESH at the
# pilot's own resolution (not reused/interpolated from the 6,840-element
# clean_holdout set).
# ---------------------------------------------------------------------
from omar_pfem.data.data_generate_B3_dataset import generate_dataset

print(f'\n=== Step 1: independent FEM held-out set ({N_HOLDOUT_SAMPLES} samples @ {FINE_RESOLUTION}) ===')
t0 = time.time()
manifest = generate_dataset(
    num_samples=N_HOLDOUT_SAMPLES, output_dir=PILOT_HOLDOUT_DIR,
    Ntheta=FINE_RESOLUTION[0], Nr=FINE_RESOLUTION[1], Nz=FINE_RESOLUTION[2],
    seed=HOLDOUT_SEED, device=device)
print(f'Held-out set generation: {time.time() - t0:.1f}s, '
      f'{len(manifest["successful_samples"])}/{N_HOLDOUT_SAMPLES} succeeded')
assert len(manifest['successful_samples']) == N_HOLDOUT_SAMPLES, 'some held-out solves failed -- check manifest'

# ---------------------------------------------------------------------
# Step 2: short pilot training run, SAME hyperparameters/normalization
# as run 4, ONLY resolution and n_iters changed.
# ---------------------------------------------------------------------
from omar_pfem.train_B3 import get_args, train

norm_src = f'{RUN4_DIR}/input_norm.json'
norm_dst = f'{PILOT_DIR}/input_norm.json'
assert os.path.exists(norm_src), f'run 4 input_norm.json not found: {norm_src}'
shutil.copy(norm_src, norm_dst)
print(f'\nReused run 4\'s own input_norm.json unchanged (copied to {norm_dst}) -- '
      f'"SAME normalization" per Omar\'s own diagnostic requirement, not recomputed.')

args = get_args([
    '--n_iters', str(N_ITERS_PILOT),
    '--batch_size', '8',
    '--log_every', '100',
    '--ckpt_every', '1000',
    '--output_dir', PILOT_DIR,
    '--normalize_inputs', '1',
])
print(f'\n=== Step 2: pilot training ({N_ITERS_PILOT} iterations @ {FINE_RESOLUTION}) ===')
print('Watch the s/iteration rate in the first few hundred iterations below -- '
      'if it projects to an unreasonable total time, interrupt (Runtime > Interrupt '
      'execution) and tell Claude the observed rate so N_ITERS_PILOT can be adjusted.')
torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
t0 = time.time()
train(args, device, resolution=FINE_RESOLUTION)
elapsed_train = time.time() - t0
print(f'\nPilot training finished in {elapsed_train:.1f}s '
      f'({elapsed_train / N_ITERS_PILOT:.4f}s/iteration average).')
if torch.cuda.is_available():
    print(f'GPU peak memory: {torch.cuda.max_memory_allocated(device) / 1e6:.1f}MB allocated, '
          f'{torch.cuda.max_memory_reserved(device) / 1e6:.1f}MB reserved.')

# ---------------------------------------------------------------------
# Step 3: evaluate the pilot checkpoint against the independent
# fine-resolution held-out set -- reuses evaluate_B3_qois.py entirely
# unmodified (already fully resolution-parametric via --resolution/
# --dataset), so this reports displacement/energy/reaction/region-stress
# (including region_cauchy_field_rel) exactly like every other B3
# checkpoint evaluation in this project.
# ---------------------------------------------------------------------
print(f'\n=== Step 3: evaluate pilot checkpoint on the independent held-out set ===')
PILOT_CKPT = f'{PILOT_DIR}/checkpoint_{N_ITERS_PILOT}.pt'
PILOT_DATASET = f'{PILOT_HOLDOUT_DIR}/dataset.h5'
PILOT_OUT_JSON = f'{PILOT_DIR}/qois_pilot_{N_ITERS_PILOT}.json'

sys.argv = [
    'evaluate_B3_qois.py',
    '--checkpoint', PILOT_CKPT,
    '--dataset', PILOT_DATASET,
    '--out_json', PILOT_OUT_JSON,
    '--resolution', str(FINE_RESOLUTION[0]), str(FINE_RESOLUTION[1]), str(FINE_RESOLUTION[2]),
]
from omar_pfem.evaluate_B3_qois import main as evaluate_main
evaluate_main()

print('\n' + '=' * 90)
print('PILOT DECISION CRITERION (per Omar\'s own explicit framing, NOT the training loss):')
print('  Does region_cauchy_field_rel drop clearly from ~72% (production, 6,840 el) '
      'toward 20-30%, while displacement/energy/reaction stay good?')
print('  -> if yes: resolution is a real lever -- worth a longer/multi-resolution run.')
print('  -> if it stays ~60-80%% regardless: the bottleneck is likely the training '
      'objective itself (e.g. an explicit local-stress supervision term), not mesh size.')
print('Remember: 43,400 elements is a PILOT resolution only, not proposed as final -- '
      'the established FEM-only convergence study needs ~480k elements for the real '
      '5-10%% region-stress target.')
print('\nDone.')
