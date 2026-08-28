# =====================================================================
#  CELL — the missing cell of the point-7b 2x2 (Timon round 5, point 7b)
#
#  The comparison "physics-informed vs data-driven" currently has three of
#  its four cells measured:
#
#                        | Adam 2e-3 (the report's  | AdamW 1e-3 + OneCycle
#                        | own recipe)              | (the data-driven one)
#    --------------------+--------------------------+----------------------
#    physics-informed    | 0.0959                   | >>> THIS CELL <<<
#    data-driven         | 0.1307                   | 0.0826
#
#  Read down the first column and the physics-informed loss wins by 36%.
#  Read the diagonal and the data-driven one wins by 16%. Those disagree,
#  and they disagree because the diagonal changes the optimizer AND the loss
#  at the same time -- exactly the confound this project keeps running into.
#  Until the empty cell is filled, the honest statement is only the matched
#  one, and the two training PRINCIPLES cannot be ranked.
#
#  This cell trains the physics-informed model under the data-driven run's
#  own recipe -- AdamW lr=1e-3, weight_decay=1e-5, OneCycleLR, the same
#  75,000 optimizer steps -- so the 2x2 closes.
#
#  What the outcome means, decided in advance so it is not rationalised
#  afterwards:
#    * beats 0.0826  -> the physics-informed loss wins under BOTH recipes,
#                       and the principle is what matters.
#    * loses to it   -> the ranking depends on the optimizer, and the
#                       report must say so rather than pick a column.
#  Either way the data-driven model still pays 5.65 h of CPU for its 800 FEM
#  labels, which the physics-informed one does not pay at all.
#
#  GPU: yes. ~48 min, the same budget the other three cells took.
#  Resumable: writes into its own directory and resumes from checkpoints.
# =====================================================================
import os, subprocess, sys, json

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
BRANCH = 'claude/claude-code-question-d307wp'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH, 'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'checkout', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
assert torch.cuda.is_available(), (
    'No GPU. Runtime -> Change runtime type -> T4 (or better), then re-run. '
    'This is FP32 training, so any GPU will do -- but CPU would take hours.')
print('GPU:', torch.cuda.get_device_name(0))

R = '/content/drive/MyDrive/pfem_run'
OUT_DIR = f'{R}/point7b/pi_adamw_onecycle'
os.makedirs(OUT_DIR, exist_ok=True)

# It must be the SAME .npz as the other three boxes of the 2x2, or this cell
# measures a different dataset rather than the missing combination. The first
# path is where the other three actually loaded from; the rest are older
# layouts, and a recursive search is the last resort instead of a third guess.
import glob
NAME = 'hyperelastic_training_data_q4.npz'
CANDIDATES = [f'{R}/results/datasets/B1_neo_hookean/{NAME}',
              f'{R}/datasets/B1_neo_hookean/{NAME}',
              f'{R}/datasets_archive/B1_neo_hookean/{NAME}']
DATASET = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATASET is None:
    found = sorted(glob.glob(f'{R}/**/B1_neo_hookean/{NAME}', recursive=True))
    assert found, ('training dataset not found. Tried:\n  '
                   + '\n  '.join(CANDIDATES)
                   + f'\nand a recursive search of {R}, which matched nothing.')
    print('none of the expected paths exist; found by search:')
    for p in found:
        print('   ', p, f'({os.path.getsize(p) / 1e6:.1f} MB)')
    DATASET = found[0]
print('dataset:', DATASET, f'({os.path.getsize(DATASET) / 1e6:.1f} MB)')

# The recipe is copied from the data-driven run's own settings, recorded in
# point7b_results/comparison_B1_neo_hookean.json. It must match exactly, or
# this cell measures something other than the empty box.
OPT_STEPS = 75_000
BATCH = 8
STEPS_PER_EPOCH = 800 // BATCH          # 100
EPOCHS = OPT_STEPS // STEPS_PER_EPOCH   # 750
print(f'\nrecipe: AdamW lr=1e-3 wd=1e-5 + OneCycleLR over {OPT_STEPS:,} steps '
      f'({EPOCHS} epochs x {STEPS_PER_EPOCH} steps)')

run([sys.executable, '-u', '-m', 'omar_pfem.train_B1',
     '--path', DATASET, '--material', 'neo_hookean',
     '--ntrain', '800', '--ntest', '200',
     '--batch_size', str(BATCH),
     '--epochs', str(EPOCHS),
     '--optimizer', 'adamw_onecycle',
     '--lr', '1e-3', '--weight_decay', '1e-5',
     '--onecycle_total_steps', str(OPT_STEPS),
     '--validate_every', '25', '--save_every', '25',
     '--early_stop_patience', '0',      # run the full schedule: OneCycleLR's
                                        # last phase is where it converges,
                                        # so stopping early would cut the
                                        # recipe off mid-way and not measure it
     '--print_every', '999999',
     '--out_dir', OUT_DIR])

# ---------------------------------------------------------------- close it
meta = f'{OUT_DIR}/best_checkpoint_meta.json'
assert os.path.exists(meta), f'no best_checkpoint_meta.json in {OUT_DIR}'
best = json.load(open(meta))
pi_onecycle = best.get('best_val_error', best.get('val_error'))

prior = json.load(open('omar_pfem/point7b_results/comparison_B1_neo_hookean.json'))
pi_adam = prior['runs']['physics_informed']['best_val_rel_L2']
dd_adam = prior['runs']['data_driven_matched_optimizer']['best_val_rel_L2']
dd_onecycle = prior['runs']['data_driven_own_optimizer']['best_val_rel_L2']

print('\n' + '=' * 72)
print('POINT 7b — the 2x2, now complete (relative L2, lower is better)')
print('=' * 72)
print(f"{'':<22}{'Adam 2e-3':>16}{'AdamW+OneCycle':>18}")
print(f"{'physics-informed':<22}{pi_adam:>16.4f}{pi_onecycle:>18.4f}")
print(f"{'data-driven':<22}{dd_adam:>16.4f}{dd_onecycle:>18.4f}")

print('\nBy column — the comparison that isolates the LOSS:')
for name, a, b in (('Adam 2e-3', pi_adam, dd_adam),
                   ('AdamW+OneCycle', pi_onecycle, dd_onecycle)):
    w = 'physics-informed' if a < b else 'data-driven'
    print(f'  {name:<16}: {w} wins by {abs(a-b)/max(a,b)*100:.0f}%')

if pi_onecycle < dd_onecycle and pi_adam < dd_adam:
    print('\nThe physics-informed loss wins under BOTH optimizers. The')
    print('principle is what matters, and the earlier data-driven advantage')
    print('was the optimizer all along.')
elif pi_onecycle > dd_onecycle and pi_adam < dd_adam:
    print('\nThe ranking FLIPS with the optimizer. Neither principle wins')
    print('outright, and the report must say exactly that rather than pick')
    print('whichever column flatters the conclusion.')
else:
    print('\nRead the two columns above carefully before writing anything.')

print(f'\nAnd independently of all of it: the data-driven model needs 800 FEM')
print(f'solves = 5.65 h of CPU for its labels. The physics-informed one needs')
print(f'none. That gap does not depend on the optimizer.')
print(f'\nSave {OUT_DIR}/best_checkpoint_meta.json — it goes into '
      f'point7b_results/.')
