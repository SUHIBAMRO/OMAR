# =====================================================================
#  CELL -- B3: finer-resolution region-stress evaluation, to separate
#  an evaluation-resolution artifact from a genuine network accuracy
#  gap. See evaluate_B3_qois_finer.py's own module docstring for the
#  full derivation.
#
#  Context: region_cauchy_field_rel (the artifact-free full-tensor
#  field metric) came back mean=71.77%/median=68.45% on the real
#  checkpoint at PRODUCTION resolution (6,840 elements, only 6 Gauss
#  points in the region). That metric fixed HOW the two fields are
#  compared, but still compares them at only 6 points -- it does not
#  test whether 6 points is enough to SAMPLE the region in the first
#  place. This notebook re-solves a SUBSET of the same held-out samples
#  at a much finer mesh (36 Gauss points in the region at (41,36,32),
#  vs. 6 at production resolution), interpolating each sample's
#  already-generated material field onto the finer mesh (no new random
#  draw -- the SAME physical sample, just refined), and queries the
#  SAME trained network zero-shot at that finer mesh. If the finer
#  region_cauchy_field_rel is much lower than 71.77%, the production-
#  resolution number was largely an evaluation-sampling artifact. If it
#  stays similarly high, the gap is real and resolution-independent.
#
#  Verified locally on CPU first (toy scale, real solves/real network,
#  not stubs): (1) an identity check -- interpolating a field back onto
#  its own coarse grid reproduces it exactly; (2) a bounds check --
#  interpolating onto a finer grid keeps E/nu within their original
#  physical range; (3) a full toy end-to-end run exercising every real
#  code path (tiny dataset -> brief training -> finer re-solve ->
#  zero-shot query -> region_cauchy_field_rel), no errors.
#
#  COST IS NOT YET KNOWN ON REAL GPU DATA: each of the --n_samples
#  samples below requires a genuinely NEW nonlinear FEM solve at
#  43,400 elements (vs. 6,840 at production resolution) plus one
#  network forward pass. If this is taking unexpectedly long, interrupt
#  and re-run with a smaller --n_samples (even 3-5 samples already give
#  a real, if noisier, read on whether the gap shrinks).
# =====================================================================
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

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
# Idempotent: mount() raises "Mountpoint must not already contain files"
# if re-run in a still-alive runtime that already mounted Drive.
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

R = '/content/drive/MyDrive/pfem_run'
CHECKPOINT = f'{R}/b3_training_normalized/checkpoint_50000.pt'
DATASET = f'{R}/b3_dataset_clean_holdout/dataset.h5'
OUT_JSON = f'{R}/b3_training_normalized/qois_finer_resolution.json'

assert os.path.exists(CHECKPOINT), f'checkpoint not found: {CHECKPOINT}'
assert os.path.exists(DATASET), f'clean held-out dataset not found: {DATASET}'

sys.argv = [
    'evaluate_B3_qois_finer.py',
    '--checkpoint', CHECKPOINT,
    '--dataset', DATASET,
    '--out_json', OUT_JSON,
    '--fine_resolution', '41', '36', '32',
    '--n_samples', '10',
]
from omar_pfem.evaluate_B3_qois_finer import main
main()

print('\nDone.')
