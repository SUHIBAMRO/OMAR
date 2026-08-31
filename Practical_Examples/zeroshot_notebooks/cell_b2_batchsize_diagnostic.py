# =====================================================================
#  DIAGNOSTIC — is batch size why B2 zero-shot will not train?
#
#  Where this stands. Report section 9.1 established that once B2's
#  boundary force is FEM-consistent, the training loss must be normalized
#  by the per-sample force scale (--loss_force_norm). That option was
#  missing from the zero-shot trainer, was added, and the retrain was run
#  with it on. It did not fix the problem:
#
#      case            best combined val   at epoch
#      B2 x NH                    0.9986         25
#      B2 x MR                    0.9752         25
#      B2 x AB                    1.0267        225
#
#  Best at essentially the FIRST validation, worse afterwards, early stop
#  fired. Eval errors 0.87-0.89 and flat in the mesh to the fourth
#  decimal. That is a model that never learned, not a model that learned
#  badly. So loss_force_norm was necessary and is not sufficient.
#
#  What differs from a recipe known to work. b2_accuracy_search.py's
#  "lossnorm" trial reached 9.11% on B2 x Neo-Hookean, and it calls
#  train_B2.py WITHOUT --batch_size, i.e. at train_B2's default of ONE.
#  The zero-shot trainer defaults to eight, and that is what these runs
#  used. B1 is not affected: train_B1 and the B1 zero-shot runs both use
#  eight, and the three B1 cases train fine.
#
#  This is a candidate, not a diagnosis, and the last candidate did not
#  hold. So this cell TESTS it rather than assuming it, on one case, at
#  MATCHED OPTIMIZER STEPS so the two arms are comparable:
#
#      arm A   batch 8, ~22,500 steps   -- reproduces what just happened
#      arm B   batch 1, ~22,500 steps   -- the recipe that reaches 9.11%
#
#  Equal steps, not equal epochs: at batch 1 an epoch is 800 steps rather
#  than 100, so equal epochs would give arm B eight times the training
#  and prove nothing about batch size.
#
#  ~15-20 min on an A100. Neither arm touches the real output directory.
# =====================================================================
import json
import os
import shutil
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
SRC = f'{R}/zeroshot_B2_neo_hookean'
WORKDIR = f'{R}/b2_batchsize_diagnostic'
STEPS = 22500

from google.colab import drive
drive.mount('/content/drive')

import torch


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH,
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])
run(['git', '-C', REPO, 'log', '--oneline', '-1'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
assert torch.cuda.is_available(), 'no GPU -- Runtime -> Change runtime type'
print('GPU:', torch.cuda.get_device_name(0))

cache = f'{SRC}/samples_cache.pt'
assert os.path.exists(cache), f'no sample cache at {cache}'

# 800 training samples over the two resolutions, so steps per epoch is
# 800/batch. Choose the epoch count per arm to land on the same step count.
N_TRAIN_TOTAL = 800
ARMS = []
for bs in (8, 1):
    spe = N_TRAIN_TOTAL // bs
    epochs = STEPS // spe
    # validate often enough to see the curve, but not so often that
    # validation dominates the wall clock
    val_every = max(1, epochs // 12)
    ARMS.append(dict(bs=bs, epochs=epochs, spe=spe, val_every=val_every,
                     steps=epochs * spe, d=f'{WORKDIR}/bs{bs}'))

print('\n' + '=' * 78)
print('THE TWO ARMS, matched on optimizer steps')
print('=' * 78)
for a in ARMS:
    print(f"  batch {a['bs']:<2}  {a['spe']:>4} steps/epoch x {a['epochs']:>4} "
          f"epochs = {a['steps']:,} steps   validate every {a['val_every']}")
assert len({a['steps'] for a in ARMS}) == 1, (
    'the arms do not have equal step counts -- fix before drawing any '
    'conclusion from them')

for a in ARMS:
    os.makedirs(a['d'], exist_ok=True)
    # Each arm needs the sample cache; copy rather than symlink so a
    # trainer that writes near it cannot touch the real directory.
    dst = f"{a['d']}/samples_cache.pt"
    if not os.path.exists(dst):
        print(f"\ncopying the sample cache into {a['d']} ...")
        shutil.copy2(cache, dst)

for a in ARMS:
    if os.path.exists(f"{a['d']}/metrics_history.json"):
        print(f"\n[batch {a['bs']}] already run -- skipping")
        continue
    print('\n' + '=' * 78)
    print(f"[batch {a['bs']}] {a['epochs']} epochs = {a['steps']:,} steps")
    print('=' * 78)
    run([sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot',
         'train', '--geometry', 'B2', '--material', 'neo_hookean',
         '--train_resolutions', '21,33',
         '--n_train_per_res', '400', '--n_val_per_res', '100',
         '--epochs', str(a['epochs']),
         '--validate_every', str(a['val_every']),
         '--batch_size', str(a['bs']),
         # early stopping off: both arms must run the full matched budget,
         # or the comparison is between different amounts of training
         '--early_stop_patience', '100000',
         '--out_dir', a['d']])

# ---- verdict ---------------------------------------------------------
print('\n' + '=' * 78)
print('RESULT — best combined val error at ~%s matched optimizer steps' % f'{STEPS:,}')
print('=' * 78)
best = {}
for a in ARMS:
    h = json.load(open(f"{a['d']}/metrics_history.json"))
    b = min(h, key=lambda e: e['combined_val_error'])
    best[a['bs']] = b['combined_val_error']
    print(f"  batch {a['bs']:<2}  best {b['combined_val_error']:.4f} at epoch "
          f"{b['epoch']} ({b['opt_steps']:,} steps)   "
          f"last {h[-1]['combined_val_error']:.4f}")

print(f"\n  for reference: the three B1 zero-shot cases, same trainer, batch 8,")
print(f"  reached 0.0658 to 0.0827 on this metric.")
print()
if best[1] < 0.5 * best[8]:
    print('BATCH SIZE IS THE CAUSE. Batch 1 is more than twice as good at the')
    print('same number of optimizer steps. The B2 zero-shot runs should be')
    print('redone at batch 1 -- which is what b2_accuracy_search used to reach')
    print('9.11%. Note it costs more wall clock per epoch, though not per step.')
elif best[1] < best[8]:
    print('Batch 1 helps but does not resolve it: better, yet not the order of')
    print('magnitude that separates 0.87 from the 0.07 B1 reaches. Something')
    print('else is also wrong, and this arm alone does not identify it.')
else:
    print('BATCH SIZE IS NOT THE CAUSE. Batch 1 is no better at matched steps,')
    print('so the difference from the 9.11% recipe lies elsewhere -- the next')
    print('candidates are the sample-construction path (the zero-shot cache is')
    print('built by build_sample_b2, the 9.11% run by data_generate_B2 plus')
    print('convert_B2_quad) and the Dirichlet node sets those two produce.')
print('=' * 78)
print('\nNeither arm wrote to', SRC)
print('Send this block over before any further B2 training.')
