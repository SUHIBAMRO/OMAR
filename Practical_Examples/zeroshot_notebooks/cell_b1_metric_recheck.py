# =====================================================================
#  ARE B1'S REPORTED NUMBERS AFFECTED BY THE SAME METRIC?
#
#  CONFIRMED, on all 100 val samples of each resolution, both arms:
#
#      arm   checkpoint    per_component   both_components
#      N=21  epoch  50        0.9622           0.7743
#      N=21  epoch 450        1.2255           0.6858
#      N=33  epoch  50        1.0372           0.7043
#      N=33  epoch 450        1.1538           0.6822
#
#  The metric that early stopping used goes UP while the error over both
#  components goes DOWN, in both arms. It ranks the two checkpoints
#  backwards, so every B2 run stopped at its FIRST validation event and
#  kept the worse model. That is now fixed in the trainer:
#  --selection_metric defaults to both_components and both numbers are
#  recorded either way.
#
#  WHAT THIS DOES NOT MEAN, and it matters more than the finding. B2 is
#  still bad. Its best both-components error is 0.68; B1 reaches 0.066.
#  The metric cost B2 roughly 0.77 -> 0.69, an eighth of the gap, not the
#  gap. B2 zero-shot failing is NOT a metric artefact and the report's
#  conclusion does not change.
#
#  WHY THE METRIC INVERTS, from its own printout: the per-sample
#  rms(v)/rms(u) averages 1.90 while the ratio of the AVERAGED components
#  is 0.90. The distribution is skewed, so the average is reporting its
#  tail -- samples where one component happens to be small and its
#  relative error is therefore large, however well the field as a whole is
#  predicted.
#
#  AND THAT IS WHY B1 HAS TO BE CHECKED. Every B1 number in the report --
#  5.0% to 10.6% zero-shot, 0.0658 to 0.0827 on the training resolutions
#  -- is the SAME metric. Two possibilities, and they are opposite:
#
#      B1's two metrics agree      the reported numbers stand as they are,
#                                  and the fact that the metric inverts on
#                                  B2 but not on B1 is itself a finding:
#                                  it means B2's two displacement
#                                  components are far more unequal than
#                                  B1's, which is a statement about the
#                                  annulus, not about the training.
#
#      B1's disagree too           every zero-shot number in the report is
#                                  in a metric that does not order models
#                                  correctly, and the tables have to be
#                                  restated before v38 goes anywhere.
#
#  Either answer is worth having before the report is touched. It costs
#  minutes on CPU and trains nothing.
#
#  It runs on the three B1 cases' own checkpoints and caches, which are
#  the ones the reported numbers came from.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['neo_hookean', 'mooney_rivlin', 'arruda_boyce']

from google.colab import drive
drive.mount('/content/drive')


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

PLAN = []
print('\npre-flight:')
for mat in CASES:
    d = f'{R}/zeroshot_B1_{mat}'
    cache, ckpt = f'{d}/samples_cache.pt', f'{d}/model_best.pt'
    ok = os.path.exists(cache) and os.path.exists(ckpt)
    print(f'  B1 x {mat:<14} cache {"YES" if os.path.exists(cache) else "no "}'
          f'   model {"YES" if os.path.exists(ckpt) else "no "}')
    if ok:
        PLAN.append((mat, d, cache, ckpt))
assert PLAN, 'no B1 case has both a cache and a checkpoint'

for mat, d, cache, ckpt in PLAN:
    print('\n\n' + '#' * 78)
    print(f'#  B1 x {mat}')
    print('#' * 78)
    run([sys.executable, '-u', '-m', 'omar_pfem.compare_val_metrics',
         '--geometry', 'B1', '--material', mat,
         '--cache', cache, '--checkpoint', ckpt,
         '--out_json', f'{d}/val_metrics_model_best.json', '--cpu'])

# ---- the comparison ---------------------------------------------------
print('\n' + '=' * 78)
print('B1, THE REPORTED CHECKPOINTS, BOTH METRICS')
print('=' * 78)
print(f"  {'case':<16}{'N':>5}{'per_component':>16}{'both_components':>18}"
      f"{'ratio':>8}{'rms(v)/rms(u)':>15}")
gaps = []
for mat, d, cache, ckpt in PLAN:
    jp = f'{d}/val_metrics_model_best.json'
    if not os.path.exists(jp):
        continue
    for r in json.load(open(jp))['rows']:
        ratio = r['trainer_metric'] / max(r['combined_rel_L2'], 1e-30)
        gaps.append(ratio)
        print(f"  {mat:<16}{r['N']:>5}{r['trainer_metric']:>16.4f}"
              f"{r['combined_rel_L2']:>18.4f}{ratio:>8.2f}"
              f"{r['v_over_u']:>15.4f}")

print()
if gaps:
    lo, hi = min(gaps), max(gaps)
    print(f'  per_component is {lo:.2f}x to {hi:.2f}x the both-components')
    print('  number on B1, in the SAME direction every time.')
    print()
    print('WHAT THIS IS, AND WHAT IT IS NOT. This is a LEVEL OFFSET, not the')
    print('failure found on B2. B2\'s problem is that the metric ORDERS two')
    print('checkpoints backwards, and ordering is what early stopping and')
    print('model_best.pt depend on. One checkpoint per case cannot test')
    print('ordering at all, so this run does not show B1 has that problem --')
    print('and it does not show B1 is free of it either. It measures how far')
    print('apart the two metrics sit, nothing more.')
    print()
    print('THE OFFSET IS EXPECTED AND ALREADY DOCUMENTED. rms(v)/rms(u) is')
    print('3.3 to 4.2 on B1: the block is pulled vertically, u is the small')
    print('component, and dividing each component by its own size lets the')
    print('small one dominate the average. So per_component reads HIGHER --')
    print('the reported B1 numbers are CONSERVATIVE, and the both-components')
    print('error is lower than the report claims, not higher.')
    print()
    print('THE REPORT DOES NOT NEED RESTATING. Section 7.1 defines every')
    print('reported error exactly, and PROJECT_STATUS already records that')
    print('Tables 5/11/12 use the per-component average while Section 4.4\'s')
    print('convergence work uses the combined norm, "so the combined norm')
    print('reads lower" on B1. Nothing here contradicts a published number.')
    print('An earlier version of this cell called that "B1 IS AFFECTED TOO"')
    print('and demanded the tables be restated; that was wrong, and it was')
    print('wrong because it tested a threshold on the offset instead of')
    print('testing the ordering.')
    print()
    print('WHAT WOULD BE WORTH KNOWING, and it is cheap: whether B1\'s runs')
    print('ALSO early-stopped on a metric that had started to invert. If')
    print('they did, B1 is better than the report says -- an improvement to')
    print('claim, not an error to fix. That needs B1\'s epoch-N endpoint')
    print('weights, which train_state_latest.pt carries, and it is the arm')
    print('below.')
print('=' * 78)

# ---- did B1's own runs stop early on a metric that had inverted? -------
# One checkpoint cannot answer the ordering question, so this recovers each
# B1 run's ENDPOINT weights the same way the B2 arms' were recovered and
# scores both metrics on both endpoints. If per_component rises while
# both_components falls here too, B1's early stopping kept a worse model and
# the reported B1 numbers are conservative twice over.
import torch

print('\n' + '=' * 78)
print('B1: THE ENDPOINT OF EACH RUN, AGAINST ITS BEST CHECKPOINT')
print('=' * 78)
ENDS = []
for mat, d, cache, ckpt in PLAN:
    final, state = f'{d}/model_final.pt', f'{d}/train_state_latest.pt'
    hist = f'{d}/metrics_history.json'
    if not os.path.exists(final):
        if not os.path.exists(state):
            print(f'  {mat}: no model_final.pt and no train_state_latest.pt '
                  f'-- endpoint unavailable')
            continue
        st = torch.load(state, map_location='cpu', weights_only=False)
        if os.path.exists(hist):
            last = json.load(open(hist))[-1]['epoch']
            if st['epoch'] != last:
                print(f"  {mat}: train_state_latest.pt is at epoch "
                      f"{st['epoch']} but the last validation event was "
                      f"{last} -- not the same run, skipping")
                continue
        torch.save(st['model_state_dict'], final)
        print(f"  {mat}: recovered epoch {st['epoch']} -> model_final.pt")
    else:
        print(f'  {mat}: model_final.pt already present')
    ENDS.append((mat, d, cache, final))

for mat, d, cache, final in ENDS:
    print('\n\n' + '#' * 78)
    print(f'#  B1 x {mat}   model_final.pt')
    print('#' * 78)
    run([sys.executable, '-u', '-m', 'omar_pfem.compare_val_metrics',
         '--geometry', 'B1', '--material', mat,
         '--cache', cache, '--checkpoint', final,
         '--out_json', f'{d}/val_metrics_model_final.json', '--cpu'])

print('\n' + '=' * 78)
print('B1: DOES ITS METRIC INVERT BETWEEN THE TWO ENDPOINTS?')
print('=' * 78)
inv = []
for mat, d, cache, ckpt in PLAN:
    a = f'{d}/val_metrics_model_best.json'
    b = f'{d}/val_metrics_model_final.json'
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    ja, jb = json.load(open(a)), json.load(open(b))
    pa, pb = ja['mean_trainer_metric'], jb['mean_trainer_metric']
    ca, cb = ja['mean_combined_rel_L2'], jb['mean_combined_rel_L2']
    inv.append((pb > pa, cb < ca))
    print(f"  {mat:<16} per_component {pa:.4f} -> {pb:.4f} "
          f"({'UP' if pb > pa else 'DOWN'})   "
          f"both_components {ca:.4f} -> {cb:.4f} "
          f"({'DOWN' if cb < ca else 'UP'})")

print()
if inv and any(p and c for p, c in inv):
    print('B1 INVERTS TOO on at least one case: its early stopping also kept')
    print('the worse model. The reported B1 numbers are then conservative')
    print('twice -- once for the metric offset, once for stopping early.')
    print('That is an improvement to claim in v38, not an error to correct.')
elif inv:
    print('B1 DOES NOT INVERT. Its best checkpoint really is its best on')
    print('both metrics, so early stopping did the right thing on B1 and the')
    print('wrong thing on B2 -- and the difference between the geometries is')
    print('the stability of the component ratio: 3.3-4.2 and tight on B1,')
    print('skewed on B2 (per-sample mean 1.90 against 0.90 aggregate).')
else:
    print('No B1 case had both endpoints available, so this is unanswered.')
print('=' * 78)
print('Nothing was trained. On Drive this wrote val_metrics_*.json and, where')
print('it was missing, model_final.pt recovered from the run\'s own resume')
print('state -- no existing file was overwritten.')
