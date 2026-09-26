# =====================================================================
#  CELL -- B3: evaluate EVERY saved checkpoint (5k-50k from the third
#  training run) against the CLEAN held-out set, to find the real best
#  checkpoint instead of assuming the final one (50000) is best.
#
#  Why this exists: train_B3.py has no validation/checkpoint-selection
#  loop during training at all (unlike train_B1.py/train_B2.py's own
#  model_best.pt mechanism) -- Omar's methodology review, 2026-09-26,
#  flagged this directly, and pointed at this project's own real history
#  (a past B2 investigation where a "stuck at ~1.0" finding, chased for
#  weeks with sophisticated diagnostics, turned out to be a checkpoint-
#  selection bug, not a structural limitation) as a reason not to trust
#  the final checkpoint blindly. Since checkpoint_5000.pt through
#  checkpoint_50000.pt are all already saved (ckpt_every=5000 in the
#  third run), finding the real best one costs only inference time.
#
#  Uses the CLEAN held-out set (b3_dataset_clean_holdout/, seed=99999),
#  not the original dataset.h5 -- that one's seeds (0-99) overlap 92/100
#  with real training inputs (train_B3.py's sample_batch uses
#  seed=it*batch_size+b), so it is not a genuine held-out test.
#
#  Verified locally on CPU before this: a small toy checkpoint/dataset
#  ran through the exact same evaluate_B3_checkpoint_sweep.py end to end
#  with no errors.
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
CHECKPOINT_DIR = f'{R}/b3_training'
DATASET = f'{R}/b3_dataset_clean_holdout/dataset.h5'
OUT_JSON = f'{R}/b3_training/checkpoint_sweep.json'

assert os.path.exists(DATASET), (
    f'clean held-out dataset not found: {DATASET} -- run '
    f'B3_Dataset_Clean_Holdout.ipynb first')

sys.argv = [
    'evaluate_B3_checkpoint_sweep.py',
    '--checkpoint_dir', CHECKPOINT_DIR,
    '--iters', '5000', '10000', '15000', '20000', '25000',
    '30000', '35000', '40000', '45000', '50000',
    '--dataset', DATASET,
    '--out_json', OUT_JSON,
]
from omar_pfem.evaluate_B3_checkpoint_sweep import main
main()

print('\nDone.')
