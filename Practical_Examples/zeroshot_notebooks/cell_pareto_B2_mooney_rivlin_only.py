# =====================================================================
#  Accuracy/cost Pareto for B2 x Mooney-Rivlin ONLY
#
#  Why this file exists and is not just cell_pareto_B2.py: that cell loops
#  over all three materials in ONE session and, once Mooney-Rivlin's nine
#  resolutions are done, walks straight into Arruda-Boyce next -- no pause,
#  no prompt. If a SEPARATE session (cell_pareto_B2_arruda_boyce_only.py)
#  is already mid-sweep on Arruda-Boyce at that moment, both processes end
#  up writing the SAME pareto_B2_arruda_boyce.json: each one reads the file
#  once at its own start and holds its own copy in memory, so whichever
#  one writes last can silently overwrite resolutions the other already
#  saved. Restricting CASES to just Mooney-Rivlin here means this cell
#  finishes its own material and stops -- it never reaches Arruda-Boyce,
#  so it can run at the same time as the dedicated Arruda-Boyce cell with
#  zero risk to that file.
#
#  Point 2 of the round-5 review. Same script (omar_pfem.pareto_analysis),
#  same nine resolutions and protocol as every other Pareto cell in this
#  directory. Resuming this after a disconnect is exactly as safe as
#  before -- this file only changes WHICH cases run, not how progress is
#  saved or resumed.
#
#  GPU needed. Resumable at every resolution.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['mooney_rivlin']  # ONLY -- see header
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
        print(f'\n[{mat}] skipping -- {reason}.')
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
    print(f'[{mat}] starting.')
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
print('Mooney-Rivlin is now either done or skipped. This cell never touches')
print('Arruda-Boyce\'s files -- that material is handled entirely by')
print('cell_pareto_B2_arruda_boyce_only.py, running separately.')
print('Send Claude the contents of pareto_B2_mooney_rivlin.json -- or just')
print('the summary block this run printed.')
print('=' * 70)
