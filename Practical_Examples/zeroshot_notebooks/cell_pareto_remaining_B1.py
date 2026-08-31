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
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['mooney_rivlin', 'arruda_boyce']
# the same nine Table 18 uses, so the three cases are read on one axis
RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 49]

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
    # A case counts as done only when its JSON holds EVERY resolution. An
    # earlier version skipped on the file merely EXISTING, which would have
    # defeated the resolution-level resume added to pareto_analysis.py: a run
    # killed after N=13 leaves a one-row file, and "exists" would have read
    # that as finished and skipped the remaining eight resolutions in silence.
    have, stamped = [], False
    if os.path.exists(out_json):
        try:
            prev = json.load(open(out_json))
            have = [r['N'] for r in prev.get('rows', [])]
            # Rows written before the resume fix carry no checkpoint
            # fingerprint, so pareto_analysis.py cannot show they came from
            # THIS model and deliberately starts fresh. A complete file is
            # still skipped here and never touched; only a PARTIAL one is
            # redone from the first resolution.
            stamped = prev.get('checkpoint_fingerprint') is not None
        except Exception as e:
            print(f'[{mat}] {out_json} is unreadable ({e.__class__.__name__});'
                  f' treating as not started')
    missing = [N for N in RESOLUTIONS if N not in have]
    PLAN.append((mat, d, ckpt, out_json,
                 os.path.getsize(ckpt) / 1e6, have, missing, stamped))

print('\nplan:')
for mat, d, ckpt, out_json, mb, have, missing, stamped in PLAN:
    if not missing:
        state = 'COMPLETE (%d/%d), will skip' % (len(have), len(RESOLUTIONS))
    elif have and stamped:
        state = 'PARTIAL %d/%d -- will resume at N=%s' % (
            len(have), len(RESOLUTIONS), missing[0])
    elif have:
        state = ('PARTIAL %d/%d, but written before fingerprinting -- those '
                 'rows will be redone from N=%d' % (
                     len(have), len(RESOLUTIONS), RESOLUTIONS[0]))
    else:
        state = 'to run (%d resolutions)' % len(RESOLUTIONS)
    print('  %-14s checkpoint %.1f MB   %s' % (mat, mb, state))

for mat, d, ckpt, out_json, mb, have, missing, stamped in PLAN:
    if not missing:
        print(f'\n[{mat}] all {len(RESOLUTIONS)} resolutions present in '
              f'{out_json} -- skipping. Delete it to force a re-run.')
        continue
    if have and stamped:
        print(f'\n[{mat}] resuming: N={sorted(have)} already done, '
              f'N={missing} still to do.')
    elif have:
        print(f'\n[{mat}] N={sorted(have)} is on disk but predates '
              f'fingerprinting, so it cannot be shown to belong to this '
              f'checkpoint and will be recomputed. From here on every '
              f'resolution is stamped and a restart resumes.')
    print('\n' + '=' * 70)
    # The Mooney-Rivlin run measured 14 h 31 m, not the "roughly two hours"
    # an earlier version of this cell predicted. The estimate had been carried
    # over from B1 x Neo-Hookean without allowing for the 2.1-2.4x more
    # expensive autodiff assembly of the other two materials (Table 4a). Pure
    # CPU solve time is 6.6 h; the rest is Colab.
    print(f'[{mat}] starting -- 6.6 h of CPU solve time at minimum, and the '
          f'Mooney-Rivlin run took 14 h 31 m end to end')
    print(f'  N=49 alone is 1.8 h of it. The JSON is written after EVERY')
    print(f'  resolution and a restart now resumes, so a disconnect costs')
    print(f'  only the resolution in flight.')
    print('=' * 70)
    run([sys.executable, '-u', '-m', 'omar_pfem.pareto_analysis',
         '--geometry', 'B1', '--material', mat,
         '--checkpoint', ckpt,
         # the same nine resolutions Table 18 uses, so the three cases are
         # read on one axis
         '--resolutions', ','.join(str(N) for N in RESOLUTIONS),
         '--fine_N', '101',
         '--n_samples', '20',
         '--batch_size', '1',
         '--out_dir', d,
         '--out_json', out_json])
    n = len(json.load(open(out_json)).get('rows', []))
    print(f'\n[{mat}] wrote {out_json} -- {n}/{len(RESOLUTIONS)} resolutions')

print('\n' + '=' * 70)
print('All requested cases finished. Send Claude the contents of each')
print('pareto_B1_*.json -- or just the summary block each run prints -- and')
print('the three-material Pareto goes into the report.')
print('=' * 70)
