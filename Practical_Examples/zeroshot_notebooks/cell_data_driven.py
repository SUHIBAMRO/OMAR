# =====================================================================
#  CELL — Data-driven operator, for comparison with the physics-informed one
#  (Timon round-5 point 7b; round-6 said to start with B1 x Neo-Hookean.)
#
#  CPU works but will be slow; a GPU runtime is strongly preferred.
#  Needs NO new FEM solves: the labels are already in the dataset the
#  physics-informed model trained on.
#  Saves a checkpoint and history to Drive at every evaluation, so a
#  disconnect loses at most `eval_every` steps of progress.
# =====================================================================
import os, subprocess, sys

def run(cmd):
    """Stream a child process's output into the notebook.

    subprocess.run() writes to the OS-level stdout descriptor, which Colab
    does not capture -- the child's output simply vanishes and a long run
    looks identical to a hung one. Popen with a line loop, plus `python -u`
    so the child does not buffer, puts it back.
    """
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
    run(['git', '-C', REPO, 'fetch', 'origin',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard',
                    'origin/claude/claude-code-question-d307wp'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE - this will be slow, prefer Runtime > Change runtime type > GPU')

R = '/content/drive/MyDrive/pfem_run'
OUT = f'{R}/data_driven'
os.makedirs(OUT, exist_ok=True)

# --- the case, and the settings that MUST match the physics-informed run ---
GEOMETRY  = 'B1'
MATERIAL  = 'neo_hookean'
DATA      = f'{R}/results/datasets/B1_neo_hookean/hyperelastic_training_data_q4.npz'
OPT_STEPS = 75000    # B1 x Neo-Hookean's own step count, from Table 7
BATCH     = 8        # the protocol's batch size (Table 4)
# --------------------------------------------------------------------------

assert os.path.exists(DATA), (
    f'dataset not found: {DATA}\n'
    'Check the path -- it must be the SAME .npz the physics-informed model '
    'trained on, or the comparison is not like-for-like.')
print('dataset:', DATA, f'({os.path.getsize(DATA)/1e6:.1f} MB)')

run([
    sys.executable, '-u', '-m', 'omar_pfem.train_data_driven',
    '--geometry', GEOMETRY, '--material', MATERIAL,
    '--path', DATA,
    '--ntrain', '800', '--ntest', '200',
    '--batch_size', str(BATCH),
    '--opt_steps', str(OPT_STEPS),
    '--loss', 'rel_l2',
    '--eval_every', '2000',
    '--out_dir', f'{OUT}/{GEOMETRY}_{MATERIAL}',
])

print(f'\nDone. Result: {OUT}/{GEOMETRY}_{MATERIAL}/'
      f'data_driven_{GEOMETRY}_{MATERIAL}.json')
print('Compare its best_val_rel_L2 against Table 5 (physics-informed, 0.0959 for')
print('this case) and note label_generation_cost_h, which the physics-informed')
print('model does not pay at all.')
