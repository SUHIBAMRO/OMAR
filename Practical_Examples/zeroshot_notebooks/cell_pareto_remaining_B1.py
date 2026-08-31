# =====================================================================
#  STEP 4 — Accuracy/cost Pareto for B1 x Mooney-Rivlin and B1 x Arruda-Boyce
#
#  Point 2 of the round-5 review. It is already done for B1 x Neo-Hookean
#  (report Table 18). These are the two other cases whose zero-shot
#  checkpoints are valid, so the Pareto can be stated on three materials
#  instead of one.
#
#  COST, from the two runs that actually happened for B1 x Neo-Hookean:
#  1 h 54 m on one Colab runtime and 6 h 24 m on a slower one -- the SAME
#  configuration both times. Budget ~2 h per case and do not be surprised
#  by three times that. An earlier note in PROJECT_STATUS said "minutes";
#  that was wrong and is corrected.
#
#  Why it costs that: the Pareto draws its own problem instances (seeds
#  900_000 + i) and needs one fine FEM solve at N=101 per sample as the
#  shared reference. Those are not the zero-shot study's references and
#  cannot be borrowed from it -- different seeds, different problems.
#
#  GPU needed. Resumable: each case writes its own JSON, and re-running
#  skips a case whose JSON is already there.
# =====================================================================
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['mooney_rivlin', 'arruda_boyce']

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

assert torch.cuda.is_available(), (
    'no GPU. Runtime -> Change runtime type -> T4 or better. This one is FP64 '
    'FEM work on one side, so a T4 will be markedly slower than an A100.')
print('GPU:', torch.cuda.get_device_name(0))

# Pre-flight on BOTH cases before starting either, so a missing checkpoint
# costs a second rather than two hours.
PLAN = []
for mat in CASES:
    d = f'{R}/zeroshot_B1_{mat}'
    ckpt = f'{d}/model_best.pt'
    out_json = f'{d}/pareto_B1_{mat}.json'
    assert os.path.exists(ckpt), (
        f'checkpoint missing: {ckpt}\nThis case has no trained model, so its '
        'Pareto cannot run.')
    PLAN.append((mat, d, ckpt, out_json,
                 os.path.getsize(ckpt) / 1e6, os.path.exists(out_json)))

print('\nplan:')
for mat, d, ckpt, out_json, mb, done in PLAN:
    print('  %-14s checkpoint %.1f MB   %s'
          % (mat, mb, 'ALREADY DONE, will skip' if done else 'to run'))

for mat, d, ckpt, out_json, mb, done in PLAN:
    if done:
        print(f'\n[{mat}] {out_json} exists -- skipping. Delete it to force a '
              f're-run.')
        continue
    print('\n' + '=' * 70)
    print(f'[{mat}] starting -- expect roughly two hours, possibly more')
    print('=' * 70)
    run([sys.executable, '-u', '-m', 'omar_pfem.pareto_analysis',
         '--geometry', 'B1', '--material', mat,
         '--checkpoint', ckpt,
         # the same nine resolutions Table 18 uses, so the three cases are
         # read on one axis
         '--resolutions', '13,17,21,25,29,33,37,41,49',
         '--fine_N', '101',
         '--n_samples', '20',
         '--batch_size', '1',
         '--out_dir', d,
         '--out_json', out_json])
    print(f'\n[{mat}] wrote {out_json}')

print('\n' + '=' * 70)
print('All requested cases finished. Send Claude the contents of each')
print('pareto_B1_*.json -- or just the summary block each run prints -- and')
print('the three-material Pareto goes into the report.')
print('=' * 70)
