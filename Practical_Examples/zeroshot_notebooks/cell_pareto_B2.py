# =====================================================================
#  Accuracy/cost Pareto for the three B2 cases
#
#  Point 2 of the round-5 review, done for all three B1 materials
#  (report Table 18 + the two Pareto cases this directory's B1 cell adds).
#  This is the B2 side, on the same footing: same script
#  (omar_pfem.pareto_analysis), same nine resolutions, same protocol.
#
#  UNLIKE the B1 cell, this one does NOT require every case's checkpoint
#  to exist before it starts. As of 2026-09-01 B2 x Neo-Hookean's
#  fixed-selection checkpoint is done and this can run for it right now;
#  Mooney-Rivlin and Arruda-Boyce are still training under
#  cell_b2_fixed_selection_all.py. A case with no checkpoint yet is
#  reported and skipped, not treated as an error -- re-running this cell
#  later picks it up once its checkpoint exists.
#
#  COST is not measured for B2 yet. What is known: B2's native FEM
#  assembly cost is within 2% of B1's for the same material (report
#  section 4.2), so there is no reason to expect the B2 geometry itself to
#  change the picture -- but the material still matters exactly as it does
#  for B1: Mooney-Rivlin and Arruda-Boyce cost 2.1-2.4x more per sample to
#  assemble than Neo-Hookean (Table 4a, autodiff tangent vs analytic), and
#  B1's own Pareto cases showed that difference stretching a ~2h run into
#  14h31m. Read B1's numbers as the closest available guide, not a B2
#  measurement: Neo-Hookean 1h54m-6h24m on two different Colab runtimes,
#  Mooney-Rivlin 14h31m end to end. Arruda-Boyce is still being measured
#  (see cell_pareto_remaining_B1.py).
#
#  GPU needed. Resumable at every resolution within a case, and a case
#  whose JSON already holds every resolution is skipped entirely.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['neo_hookean', 'mooney_rivlin', 'arruda_boyce']
# the same nine resolutions Table 18 and the B1 Pareto cell use
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

# Pre-flight on all three cases before starting any. Unlike the B1 cell, a
# missing checkpoint is reported and the case is skipped rather than
# asserted -- two of the three are still training as this is written.
PLAN = []
for mat in CASES:
    d = f'{R}/zeroshot_B2_{mat}_fixedsel'
    ckpt = f'{d}/model_best.pt'
    out_json = f'{d}/pareto_B2_{mat}.json'
    if not os.path.exists(ckpt):
        PLAN.append((mat, d, ckpt, out_json, None, [], RESOLUTIONS, False,
                     'no checkpoint yet'))
        continue
    have, stamped = [], False
    if os.path.exists(out_json):
        try:
            prev = json.load(open(out_json))
            have = [r['N'] for r in prev.get('rows', [])]
            stamped = prev.get('checkpoint_fingerprint') is not None
        except Exception as e:
            print(f'[{mat}] {out_json} is unreadable ({e.__class__.__name__});'
                  f' treating as not started')
    missing = [N for N in RESOLUTIONS if N not in have]
    PLAN.append((mat, d, ckpt, out_json, os.path.getsize(ckpt) / 1e6, have,
                 missing, stamped, None))

print('\nplan:')
for mat, d, ckpt, out_json, mb, have, missing, stamped, reason in PLAN:
    if reason:
        state = reason
    elif not missing:
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
    mb_str = f'{mb:.1f} MB' if mb is not None else '--'
    print('  %-14s checkpoint %-8s %s' % (mat, mb_str, state))

for mat, d, ckpt, out_json, mb, have, missing, stamped, reason in PLAN:
    if reason:
        print(f'\n[{mat}] skipping -- {reason}. Re-run this cell once '
              f'cell_b2_fixed_selection_all.py finishes it.')
        continue
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
    print(f'[{mat}] starting. Cost is not measured for B2 yet -- see the '
          f'header of this cell for what B1 suggests to expect.')
    print(f'  The JSON is written after EVERY resolution and a restart now')
    print(f'  resumes, so a disconnect costs only the resolution in flight.')
    print('=' * 70)
    run([sys.executable, '-u', '-m', 'omar_pfem.pareto_analysis',
         '--geometry', 'B2', '--material', mat,
         '--checkpoint', ckpt,
         '--resolutions', ','.join(str(N) for N in RESOLUTIONS),
         '--fine_N', '101',
         '--n_samples', '20',
         '--batch_size', '1',
         '--out_dir', d,
         '--out_json', out_json])
    n = len(json.load(open(out_json)).get('rows', []))
    print(f'\n[{mat}] wrote {out_json} -- {n}/{len(RESOLUTIONS)} resolutions')

print('\n' + '=' * 70)
print('Every case that had a checkpoint is now either done or skipped.')
print('Send Claude the contents of each pareto_B2_*.json -- or just the')
print('summary block each run prints.')
print('=' * 70)
