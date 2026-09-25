# =====================================================================
#  CELL -- B3 (sharper groove) Transolver training-dataset generation.
#  Geometry FIXED (groove_depth=0.20, half_width=0.15, r_grading=1.0,
#  Section 11.2's chosen production settings); per-sample: material
#  fields E(theta,r,z)/nu(theta,r,z) vary spatially via a 3D Gaussian
#  random field, and the rocking angle phi varies as a single scalar --
#  the exact scope decided with Prof. Rabczuk's 2026-09-25 confirmation
#  that "VINO" means this project's own Transolver (no architecture or
#  training-paradigm change needed).
#
#  Resolution (21,20,19)=6,840 elements: an existing, already-tested rung
#  from B3's own mesh-convergence ladder (confirmed there: ~4.86s/case on
#  a real A100, fixed material), matching B2's own dataset-generator
#  default (Ntheta=Nr=21) -- not a new, untested number.
#
#  This is the FIRST real GPU run of data_generate_B3_dataset.py. It was
#  verified locally on CPU before this (several seeds, an out-of-range
#  phi correctly failing, a real resume-logic bug found and fixed), but
#  never at GPU scale or at this sample count -- treat the first several
#  printed rows as the actual confirmation this pipeline works for real,
#  not just in principle.
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

# Same real, external pyvista>=0.49/Colab-IPython-7.34 incompatibility
# documented in the other B3 GPU cells -- pinning below 0.49 avoids it.
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

from omar_pfem.data.data_generate_B3_dataset import generate_dataset, DEFAULT_RESOLUTION

device = torch.device('cuda')

R = '/content/drive/MyDrive/pfem_run'
OUTPUT_DIR = f'{R}/b3_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 100  # matches B2's own dataset-generator default -- not a new number
Ntheta, Nr, Nz = DEFAULT_RESOLUTION
print(f'\nGenerating {NUM_SAMPLES} samples at (Ntheta,Nr,Nz)={DEFAULT_RESOLUTION} '
      f'-> {(Ntheta-1)*(Nr-1)*(Nz-1)} elements, saving to {OUTPUT_DIR}')
print('This is the FIRST real GPU run of this pipeline -- watch the first few rows '
      'closely before assuming the rest will behave the same way.')

torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()
manifest = generate_dataset(
    NUM_SAMPLES, OUTPUT_DIR, Ntheta=Ntheta, Nr=Nr, Nz=Nz, seed=0, device=device,
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
    write_manifest(OUTPUT_DIR, kind='b3_dataset_generation',
                    args={'num_samples': NUM_SAMPLES, 'resolution': DEFAULT_RESOLUTION,
                          'groove_depth': 0.20, 'groove_half_width': 0.15},
                    started_at=_started,
                    results={'succeeded': len(manifest['successful_samples']),
                             'failed': len(manifest['failed']),
                             'elapsed_total_s': elapsed_total},
                    outputs=[f'{OUTPUT_DIR}/dataset.h5', f'{OUTPUT_DIR}/manifest.json'],
                    notes="First real GPU run of the B3 Transolver training-dataset "
                          "generator: geometry fixed, E/nu vary spatially via a 3D GRF, "
                          "phi varies as a scalar. Pipeline verified on CPU beforehand "
                          "(several seeds, an out-of-range phi correctly failing, a real "
                          "resume-logic HDF5 bug found and fixed) but never before at "
                          "GPU scale or at this sample count.")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\nDone.')
