# =====================================================================
#  Accuracy/cost Pareto for B2 x Arruda-Boyce ONLY
#
#  Why this file exists and is not just cell_pareto_B2.py: that cell's
#  PRE-FLIGHT check runs once, at the top, over all three materials. A
#  session that started it before Arruda-Boyce's checkpoint existed will
#  never notice the checkpoint appearing later -- it already logged
#  "no checkpoint yet" for that material and moved on. Meanwhile
#  Neo-Hookean is done and Mooney-Rivlin may be mid-sweep in that other
#  session right now. Restricting CASES to just Arruda-Boyce here means
#  this cell can run in parallel, in its own Colab runtime, with zero risk
#  of touching either of the other two materials' files.
#
#  Point 2 of the round-5 review. Same script (omar_pfem.pareto_analysis),
#  same nine resolutions and protocol as every other Pareto cell in this
#  directory.
#
#  COST is not measured for B2 x Arruda-Boyce yet. B2 x Neo-Hookean's own
#  Pareto (pareto_B2_neo_hookean.json) is the closest available guide, not
#  a Arruda-Boyce-specific measurement: it took 7h11m end to end. B1's
#  three Arruda-Boyce numbers (Table 18b) suggest the material itself
#  costs 2.1-2.4x more per sample to assemble than Neo-Hookean, same as
#  for B1, so budget more than 7h11m rather than less.
#
#  GPU needed. Resumable at every resolution. If Neo-Hookean's or
#  Mooney-Rivlin's JSON already holds every resolution by the time this
#  runs, they are reported as complete and skipped -- untouched either way.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['arruda_boyce']  # ONLY -- see header
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
