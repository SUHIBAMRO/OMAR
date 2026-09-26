# =====================================================================
#  CELL -- B3 Transolver checkpoint evaluation: the missing accuracy
#  check. Training (B3_Transolver_Training.ipynb) only minimizes the
#  Deep Energy Method loss (Pi = U) -- it never sees a single labeled
#  FEM displacement, so a healthy, stable loss curve proves the
#  optimization did not diverge, NOT that the learned field is close to
#  the true solution. This cell runs the trained checkpoint on the 100
#  real FEM samples (dataset.h5, generated separately, never used as a
#  training label) and computes the actual relative-L2 accuracy, plus
#  an inference-latency vs FEM-solve-time comparison -- matching B1/B2's
#  own evaluate_dataset_hyperelastic_Q4 / benchmark_inference_latency_Q4
#  methodology, extended to B3's 3 displacement components.
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
# checkpoint_2000.pt (run 2) evaluated poorly (combined rel L2 35.7%, uy
# 99.95% -- see PROJECT_STATUS.md 2026-09-26 entry). Run 3 retrains for
# 20000 iterations (10x) to test whether more gradient steps closes the
# gap, following a real controlled test that ruled out OUTPUT_SCALE
# itself as the cause. Pointing this at run 3's checkpoint.
CHECKPOINT = f'{R}/b3_training/checkpoint_20000.pt'
DATASET = f'{R}/b3_dataset/dataset.h5'
OUT_JSON = f'{R}/b3_training/eval_B3_20000.json'

assert os.path.exists(CHECKPOINT), f'checkpoint not found: {CHECKPOINT}'
assert os.path.exists(DATASET), f'dataset not found: {DATASET}'

sys.argv = [
    'evaluate_B3.py',
    '--checkpoint', CHECKPOINT,
    '--dataset', DATASET,
    '--out_json', OUT_JSON,
]
from omar_pfem.evaluate_B3 import main
main()

print('\nDone.')
