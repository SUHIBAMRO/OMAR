# =====================================================================
#  CELL -- B3: local integration refinement for the region-stress QoI,
#  per Timon's explicit request (2026-09-28 reply, before any
#  resolution-changing retrain). See evaluate_B3_region_local_refine.py's
#  own module docstring for the full derivation.
#
#  Key point: this does NOT retrain, and does NOT change the operator's
#  own mesh/discretization (still checkpoint_50000.pt, still the same
#  6,840-element coarse mesh it was trained/evaluated on). It only
#  refines the QUADRATURE used to integrate/sample the groove region --
#  re-sampling the SAME already-known coarse-element displacement field
#  (true FEM, and the network's own coarse-mesh prediction) at many more
#  internal points per element (n_sub^3 instead of the default 8), via
#  torch-fem's own eval_shape_functions at arbitrary local coordinates.
#  No new FEM solve, no new network query -- this is pure, cheap
#  post-processing, so it should run in well under a minute, not hours.
#
#  Runs a CONVERGENCE SWEEP (n_sub = 2,4,6,8,10,12, i.e. roughly
#  6/36/124/292/566/994 region points at production resolution), per
#  Omar and Timon's own follow-up refinement of the plan (2026-09-29):
#  a single refinement level only answers "does it change," not "does it
#  stabilize" -- the sweep's own relative-change-between-levels is what
#  decides whether the region-stress gap is real (converges and stays
#  elevated) or a local-integration-count artifact (keeps changing).
#  Network inference runs ONCE and is reused across every sweep level.
#
#  Reports, at EACH sweep level (all requested explicitly by Timon):
#   - n_region_fine: how many quadrature points now sample the region.
#   - All SIX independent Cauchy stress components separately (their own
#     pooled RMS magnitude, median absolute magnitude, and pooled
#     relative error), so a naturally small component cannot silently
#     dominate a combined number.
#   - A SINGLE, dataset-wide Frobenius-norm relative error (one
#     normalization constant across all samples and points, not
#     per-sample), CORRECTLY double-counting the shear components
#     (sigma_xy, sigma_yz, sigma_xz) as a true symmetric-tensor
#     Frobenius norm requires -- ||sigma||_F^2 = xx^2+yy^2+zz^2 +
#     2*(xy^2+yz^2+xz^2).
#
#  Verified locally on CPU first, in four real steps (not assumed): (1)
#  a volume-integral check -- the fine quadrature reproduces each
#  element's own volume the coarse quadrature already gives, to ~1e-15
#  relative precision; (2) a consistency check -- at n_sub=2 (matching
#  the coarse rule's own order), this independently-written computation
#  agrees with the existing, already-verified region-stress code to
#  ~1e-13 relative precision on a real solved FEM field; (3) a DIRECT
#  check of the shear-doubling fix itself -- an earlier version of this
#  file summed the six components unweighted, which disagreed with the
#  existing full-tensor computation by up to 18% on a real field; the
#  corrected (Frobenius-weighted) version matches it to ~1e-14; (4) a
#  full toy-scale end-to-end run through this exact sweep-based CLI
#  path, no errors.
#
#  A real bug was caught during verification (before it reached this
#  cell): GROOVE_DEPTH/GROOVE_HALF_WIDTH were first imported from the
#  wrong module (mesh_convergence_B3's own STALE 0.05/0.15 defaults,
#  not the real production 0.20/0.15) -- caught by the volume check
#  failing (~16-40% off), not assumed correct.
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
OUT_JSON = f'{R}/b3_training_normalized/region_local_refine_convergence.json'

assert os.path.exists(CHECKPOINT), f'checkpoint not found: {CHECKPOINT}'
assert os.path.exists(DATASET), f'clean held-out dataset not found: {DATASET}'

sys.argv = [
    'evaluate_B3_region_local_refine.py',
    '--checkpoint', CHECKPOINT,
    '--dataset', DATASET,
    '--out_json', OUT_JSON,
    '--n_sub_sweep', '2', '4', '6', '8', '10', '12',
]
from omar_pfem.evaluate_B3_region_local_refine import main
main()

print('\nDone.')
