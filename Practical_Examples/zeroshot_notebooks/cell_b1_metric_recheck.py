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
    print(f'  the two metrics differ by {lo:.2f}x to {hi:.2f}x on B1')
    print('  (on B2 they differ by 1.24x at epoch 50 and 1.79x at epoch 450,')
    print('   and in OPPOSITE directions between the two checkpoints)')
    print()
    if hi < 1.15:
        print('B1 IS ESSENTIALLY UNAFFECTED. Its two displacement components')
        print('are comparable in size, so normalising each by its own does')
        print('almost nothing, and every reported B1 number stands as it is.')
        print('That also explains the whole picture: the metric inverts on')
        print('the annulus and not on the block, which is a fact about the')
        print('geometry rather than about the training.')
    else:
        print('B1 IS AFFECTED TOO. The reported zero-shot numbers are in a')
        print('metric that differs materially from the both-components one on')
        print('this geometry as well, so the tables have to be restated in')
        print('both metrics before v38. Do not edit the report until this is')
        print('worked through -- the numbers, not just the wording, change.')
print('=' * 78)
print('Nothing was trained and nothing on Drive was overwritten except the')
print('val_metrics_*.json this run wrote.')
