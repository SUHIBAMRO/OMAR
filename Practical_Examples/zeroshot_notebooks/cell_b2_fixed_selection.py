# =====================================================================
#  B2 ZERO-SHOT, TRAINED TO THE END WITH A SELECTION THAT DOES NOT INVERT
#
#  WHY. The B2 zero-shot figure in the report was produced by a run whose
#  model_best.pt was chosen by a metric that ranks B2's checkpoints
#  BACKWARDS. Measured on all 100 val samples of each resolution, both
#  single-resolution arms:
#
#      arm     checkpoint    per_component    both_components
#      N=21    epoch  50        0.9622            0.7743
#      N=21    epoch 450        1.2255  (up)      0.6858  (DOWN)
#      N=33    epoch  50        1.0372            0.7043
#      N=33    epoch 450        1.1538  (up)      0.6822  (DOWN)
#
#  Early stopping obeyed the rising number, so every B2 run stopped at its
#  FIRST validation event while Pi was still falling (-2.48e-02 ->
#  -4.34e-02), the field was still getting smoother (2.32x -> 1.74x) and
#  the both-components error was still falling. B2 has never been trained
#  to convergence.
#
#  B1 was checked the same way and does NOT invert: all three cases agree
#  in both metrics that model_best.pt is the better checkpoint, and B1
#  genuinely degrades after it (Neo-Hookean 0.0444 -> 0.1761). So early
#  stopping did the right thing on B1 and the wrong thing on B2, and every
#  B1 number in the report stands.
#
#  WHAT THIS RUN IS FOR, stated plainly. It is NOT expected to rescue B2.
#  B2's best both-components error is 0.68 against B1's 0.044 -- the
#  selection defect cost it 0.77 -> 0.69, an eighth of the gap, not the
#  gap. What this buys is the ability to write "B2 was trained to
#  convergence under a selection criterion that orders checkpoints
#  correctly, and this is where it lands" instead of "B2's number came
#  from a run our own metric stopped early". If it lands near 0.68, that
#  is the honest B2 result and the report says so.
#
#  WHAT WAS VERIFIED BEFORE THIS CELL WAS WRITTEN, by running it:
#    * a real two-resolution training run end to end on the new code;
#    * both metrics printed each validation event, with the selected one
#      named;
#    * model_best.pt tracking the both-components number while
#      combined_val_error still records the per-component one at that same
#      checkpoint;
#    * resume from train_state_latest.pt continuing at the right epoch and
#      comparing against the right stored best;
#    * model_final.pt written on the EARLY-STOPPED path -- the for/else
#      bug that lost it on every early-stopped run;
#    * --selection_metric per_component reproducing the old behaviour;
#    * eval running and reporting both metrics per resolution.
#
#  NOTHING IS WRITTEN TO THE EXISTING RUN. Everything lands in a new
#  directory; the two caches are COPIED in.
#
#  COST, from the measured rate. The single-resolution arm did 450 epochs
#  of 400 samples in 12 m 8 s, i.e. 1.618 s/epoch. This run is two
#  resolutions, 800 samples an epoch, so about 3.24 s/epoch and 4,000
#  epochs is about 3 h 36 m on an A100 -- less if it plateaus, since
#  patience is still on, just on the metric that does not invert. It is
#  resumable at every validation event: re-running this cell after a
#  disconnect continues from the last one.
# =====================================================================
import json
import os
import shutil
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
SRC = f'{R}/zeroshot_B2_neo_hookean'          # read-only source of caches
DST = f'{R}/zeroshot_B2_neo_hookean_fixedsel'  # everything new goes here

EPOCHS = 4000
TEST_RES = '13,17,25,29,37,41,49'   # the seven unseen meshes the report uses
FINE_N = '101'
N_EVAL = '20'
BASELINE = 0.9986                   # the joint run's per_component number

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

os.makedirs(DST, exist_ok=True)

# ---- pre-flight, BEFORE 3.5 hours are spent -------------------------
print('\n' + '=' * 78)
print('PRE-FLIGHT')
print('=' * 78)
src_cache = f'{SRC}/samples_cache.pt'
src_fine = f'{SRC}/fine_ref_cache_N{FINE_N}.pt'
src_ckpt = f'{SRC}/model_best.pt'
assert os.path.exists(src_cache), (
    f'no sample cache at {src_cache}. Regenerating it is 7+ hours of FEM, so '
    f'this cell refuses to start without it.')

dst_cache = f'{DST}/samples_cache.pt'
if not os.path.exists(dst_cache):
    print(f'copying the sample cache -> {DST} (nothing writes to {SRC})')
    shutil.copy2(src_cache, dst_cache)
else:
    print('sample cache already in place')

# The eval caches its fine references in the directory of --out_json. Copying
# the existing one across means the 20 N=101 B2 solves are not repeated. If it
# is missing, say so LOUDLY now rather than after the training: this cell has
# no measured cost for a fresh N=101 B2 solve and will not invent one.
dst_fine = f'{DST}/fine_ref_cache_N{FINE_N}.pt'
if os.path.exists(dst_fine):
    print(f'fine-reference cache already in place '
          f'({len(torch.load(dst_fine, weights_only=False))} solves)')
elif os.path.exists(src_fine):
    n = len(torch.load(src_fine, map_location='cpu', weights_only=False))
    print(f'copying the N={FINE_N} fine-reference cache ({n} solves) -> {DST}')
    shutil.copy2(src_fine, dst_fine)
else:
    print(f'!! NO fine-reference cache at {src_fine}.')
    print(f'!! The eval stage will have to solve {N_EVAL} fresh N={FINE_N} B2')
    print('!! problems. There is no measured cost for that in this project, so')
    print('!! it is not estimated here -- the eval prints its own progress and')
    print('!! is resumable per resolution. The TRAINING below is unaffected.')

hist = f'{DST}/metrics_history.json'
if os.path.exists(hist):
    h = json.load(open(hist))
    print(f'\nRESUMING: {len(h)} validation events already recorded, last at '
          f'epoch {h[-1]["epoch"]}')
else:
    print('\nfresh run')

print(f'\n  train      N=21 and N=33 jointly, {EPOCHS} epochs, batch 8')
print(f'  selection  both_components  <- the metric that does not invert')
print(f'  baseline   {BASELINE} per_component, from the run whose selection did')
print(f'  cost       about 3 h 36 m on an A100 at the measured 3.24 s/epoch,')
print(f'             less if it plateaus. Resumable at every validation event.')

# ---- 1. the OLD checkpoint, scored on both metrics -------------------
# Cheap (no FEM: the coarse samples are built with solve_fem=False) and it
# gives the before/after on ONE axis. Written into DST, never into SRC.
old_eval = f'{DST}/zeroshot_eval_OLD_checkpoint.json'
WANT = [int(x) for x in TEST_RES.split(',')]


def rows_present(path):
    """Which resolutions a report file actually holds.

    NOT `os.path.exists`. A run killed part-way leaves a file with some of
    the resolutions in it, and treating the file's mere existence as "done"
    would drop the rest in silence -- the same mistake that had to be fixed
    in the Pareto cell. cmd_eval resumes per resolution and is fingerprint-
    guarded, so re-running it on a partial file costs only what is missing.
    """
    if not os.path.exists(path):
        return []
    try:
        return [r['N'] for r in json.load(open(path)).get('rows', [])]
    except Exception as e:
        print(f'  {path} is unreadable ({e.__class__.__name__}); '
              f'treating as not started')
        return []


have_old = rows_present(old_eval)
missing_old = [N for N in WANT if N not in have_old]
if not os.path.exists(src_ckpt):
    print(f'\n1/3  skipped: no checkpoint at {src_ckpt}')
elif not missing_old:
    print(f'\n1/3  already complete ({len(have_old)}/{len(WANT)} resolutions) '
          f'-- {old_eval}')
else:
    print('\n' + '=' * 78)
    print('1/3  the EXISTING B2 checkpoint, re-scored on both metrics')
    if have_old:
        print(f'     resuming: N={sorted(have_old)} done, N={missing_old} left')
    print('=' * 78)
    run([sys.executable, '-u', '-m',
         'omar_pfem.resolution_invariance_zeroshot', 'eval',
         '--geometry', 'B2', '--material', 'neo_hookean',
         '--checkpoint', src_ckpt,
         '--test_resolutions', TEST_RES, '--fine_N', FINE_N,
         '--n_eval_samples', N_EVAL, '--out_json', old_eval])

# ---- 2. train ---------------------------------------------------------
print('\n' + '=' * 78)
print(f'2/3  training, {EPOCHS} epochs, selecting on both_components')
print('=' * 78)
run([sys.executable, '-u', '-m',
     'omar_pfem.resolution_invariance_zeroshot', 'train',
     '--geometry', 'B2', '--material', 'neo_hookean',
     '--train_resolutions', '21,33',
     '--n_train_per_res', '400', '--n_val_per_res', '100',
     '--epochs', str(EPOCHS), '--validate_every', '50',
     '--batch_size', '8',
     '--early_stop_patience', '15',
     '--selection_metric', 'both_components',
     '--out_dir', DST])

# ---- 3. eval the new checkpoint --------------------------------------
new_eval = f'{DST}/zeroshot_eval.json'
print('\n' + '=' * 78)
print('3/3  zero-shot eval of the new checkpoint, seven unseen meshes')
print('=' * 78)
run([sys.executable, '-u', '-m',
     'omar_pfem.resolution_invariance_zeroshot', 'eval',
     '--geometry', 'B2', '--material', 'neo_hookean',
     '--checkpoint', f'{DST}/model_best.pt',
     '--test_resolutions', TEST_RES, '--fine_N', FINE_N,
     '--n_eval_samples', N_EVAL, '--out_json', new_eval])

# ---- the comparison ---------------------------------------------------
print('\n' + '=' * 78)
print('B2 ZERO-SHOT: OLD SELECTION AGAINST FIXED SELECTION')
print('=' * 78)
h = json.load(open(hist))
best = min(h, key=lambda e: e['both_components_val_error'])
print(f"  training stopped at epoch {h[-1]['epoch']}; best both_components "
      f"{best['both_components_val_error']:.4f} at epoch {best['epoch']} "
      f"({best['opt_steps']:,} steps)")
print(f"  its per_component number at that checkpoint: "
      f"{best['combined_val_error']:.4f}   (the old run's was {BASELINE})")

rows = {}
for tag, path in (('old', old_eval), ('new', new_eval)):
    if os.path.exists(path):
        rows[tag] = {r['N']: r for r in json.load(open(path))['rows']}

print(f"\n  {'N':>6}" + ''.join(f'{c:>18}' for c in
                                ('old per_comp', 'new per_comp',
                                 'old both', 'new both')))
for N in WANT:
    o, n = rows.get('old', {}).get(N), rows.get('new', {}).get(N)

    def f(row, key):
        return f"{row[key]:>18.4e}" if row and row.get(key) is not None \
            else f"{'-':>18}"

    print(f'  {N:>6}' + f(o, 'mean_rel_L2_vs_fine_reference')
          + f(n, 'mean_rel_L2_vs_fine_reference')
          + f(o, 'mean_combined_rel_L2_vs_fine_reference')
          + f(n, 'mean_combined_rel_L2_vs_fine_reference'))

print('\n  B1 reaches 0.050-0.106 per_component on these same seven meshes.')
print('\n  Read the two "both" columns against each other: that is the honest')
print('  before/after, since both were measured with the metric that orders')
print('  checkpoints correctly. The per_comp columns are what the report')
print('  currently quotes, kept so the tables stay comparable.')
print('=' * 78)
print(f'Everything new is in {DST}. {SRC} was only read.')
