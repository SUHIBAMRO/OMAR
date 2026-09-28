# =====================================================================
#  CELL -- B3: real physical QoIs (total strain energy, reaction
#  force/moment, fixed-region Cauchy stress) for the trained checkpoint,
#  not just raw displacement error.
#
#  Why this exists: Timon named regional Cauchy stress, reaction
#  force/moment, and energy as the real success criteria for B3
#  (matching this project's own established B1/B2/mesh_convergence_B3
#  convention), not pointwise displacement L2 alone. checkpoint_50000.pt
#  (run 4, normalized inputs) reaches 1.88% combined displacement error
#  -- this checks whether that translates into comparably good
#  stress/reaction/energy accuracy.
#
#  No new FEM solve needed: every QoI is computed directly from a KNOWN
#  displacement field (the true FEM one, or the network's own
#  prediction) via the exact same total_potential_energy_B3 functional
#  training already uses. See evaluate_B3_qois.py's own module docstring
#  for the full derivation (reaction force = d(energy)/d(u) at a
#  Dirichlet node, by the same variational argument any FEM solver uses
#  internally; Cauchy stress from P=d(psi)/d(F) via autograd).
#
#  Verified locally on CPU first, in two steps: (1) the global
#  equilibrium-residual sanity check on a real solved FEM field came out
#  at machine precision (~1e-15), confirming the reaction-force
#  derivation is implemented correctly; (2) a real, if tiny, nonzero
#  region-Gauss-point-count case exercised the actual Cauchy-stress code
#  path (not just its NaN fallback) with no errors.
#
#  IMPORTANT CAVEAT, found while verifying: at B3's actual PRODUCTION
#  resolution (21,20,19 = 6,840 elements), only 6 Gauss points fall in
#  the fixed groove-neighborhood region -- far below the
#  MIN_RELIABLE_N_P99=20 threshold this project's own mesh_convergence_B3.py
#  already established, so region_p99_sigma_xx is correctly reported as
#  NaN ("not reliable") and region_avg_sigma_xx itself is a real but
#  statistically thin (n=6) sample. This directly confirms Omar's own
#  concern (raised 2026-09-26): training integrates the energy on a mesh
#  far coarser than what B3's own mesh-convergence study showed
#  region-stress needs (~480k elements) to converge to 5-10% accuracy.
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
OUT_JSON = f'{R}/b3_training_normalized/qois_50000.json'

assert os.path.exists(CHECKPOINT), f'checkpoint not found: {CHECKPOINT}'
assert os.path.exists(DATASET), f'clean held-out dataset not found: {DATASET}'

sys.argv = [
    'evaluate_B3_qois.py',
    '--checkpoint', CHECKPOINT,
    '--dataset', DATASET,
    '--out_json', OUT_JSON,
]
from omar_pfem.evaluate_B3_qois import main
main()

print('\nDone.')
