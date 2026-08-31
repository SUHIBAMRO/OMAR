# =====================================================================
#  Q4 AND Q9 OVER THE MANUFACTURED FAMILY — closing Timon's point 9
#
#  His second email, on the MMS point:
#
#      "Ideally, we do it for a parametrised family of solutions which
#       will be a bit more time consuming."
#
#  Half of that is already done and half has never been touched:
#
#    * the OPERATOR is trained on a 64-member family, and its mean over
#      the 16 held-out test members is already recorded in each
#      mms_operator_B1_neo_hookean*.json as
#      `operator_mean_over_test_family`;
#
#    * Q4 and Q9 have NEVER been scored on the family. Tables 22, 23 and
#      24 are every one of them the single member alpha = 0.05, beta = 0.7.
#
#  So the report's three-way comparison is a single-member comparison, and
#  the family half of his request is missing on the FEM side only. This
#  fills it in. No new physics and no training -- mms_study.py already
#  takes --alpha and --beta, so this is that script in a loop, aggregated.
#
#  THE MEMBERS ARE THE OPERATOR'S OWN. They are drawn with the identical
#  call mms_operator.py makes -- sample_family(ntest, seed + 1) -- so the
#  FEM mean and the operator mean are over the SAME 16 problems and belong
#  in one table. Drawing a fresh set would give a mean over a different
#  family and the comparison would be quietly wrong.
#
#  AND THE REFERENCE MEMBER IS NOT ONE OF THEM. (0.05, 0.7) is not in the
#  drawn set -- checked. So this is a genuinely new measurement, not a
#  re-dressing of Tables 22-24, and the family mean is expected to differ
#  from the single-member numbers rather than reproduce them.
#
#  COST, from the measured N=17 run: Q4 14 s and Q9 73 s per member, so
#  about 23 min for that mesh. N=9 is far cheaper, N=33 roughly 4x more.
#  Budget about an hour and a half for all three meshes; it is bounded and
#  nothing here can run away. FP64 on CPU, so a GPU is not needed.
#
#  Resumable: a mesh whose JSON exists is skipped.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
OUT = '/content/drive/MyDrive/pfem_run/mms/family'
NS = '9,17,33'
NTEST = 16          # must match the operator runs' --ntest
SEED = 31000000     # must match the operator runs' --seed

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
os.makedirs(OUT, exist_ok=True)

OJ = f'{OUT}/mms_family_fem_B1_neo_hookean.json'
if os.path.exists(OJ):
    print(f'\n{OJ} exists -- skipping the sweep and reading it back.')
else:
    run([sys.executable, '-u', '-m', 'omar_pfem.mms_family_fem',
         '--material', 'neo_hookean', '--Ns', NS, '--orders', 'Q4,Q9',
         '--ntest', str(NTEST), '--seed', str(SEED),
         '--out_json', OJ, '--cpu'])

# ---- put the three columns side by side ------------------------------
print('\n' + '=' * 78)
print('THE THREE-WAY COMPARISON, OVER THE FAMILY')
print('=' * 78)
fam = json.load(open(OJ))

# the operator's family mean, from whichever operator runs are on Drive
OPJ = {
    9:  '/content/drive/MyDrive/pfem_run/mms/operator_rate/'
        'mms_operator_B1_neo_hookean_N9.json',
    17: '/content/drive/MyDrive/pfem_run/mms/'
        'mms_operator_B1_neo_hookean.json',
    33: '/content/drive/MyDrive/pfem_run/mms/operator_rate/'
        'mms_operator_B1_neo_hookean_N33.json',
}
KEYS = ('L2_rel', 'H1_semi_rel', 'stress_rel_L2', 'energy_rel')

for row in fam['rows']:
    N = row['N']
    print(f'\nN = {N}')
    print(f"  {'method':<28}" + ''.join(f'{k:>14}' for k in KEYS))
    for order in ('Q4', 'Q9'):
        if order in row:
            print(f'  {order + " (family mean)":<28}'
                  + ''.join(f"{row[order][k]['mean']:>14.4e}" for k in KEYS))
    path = OPJ.get(N)
    if path and os.path.exists(path):
        op = json.load(open(path))
        fm = op.get('operator_mean_over_test_family')
        sm = op.get('operator_on_the_reference_member')
        if fm:
            print(f'  {"operator (family mean)":<28}'
                  + ''.join(f'{fm[k]:>14.4e}' for k in KEYS))
        if sm:
            print(f'  {"operator (single member)":<28}'
                  + ''.join(f'{sm[k]:>14.4e}' for k in KEYS)
                  + '   <- what Table 24 prints')
        if fm and 'Q4' in row:
            print('  operator/Q4 on the family: '
                  + ', '.join(f"{k.split('_')[0]} "
                              f"{fm[k] / row['Q4'][k]['mean']:.2f}x"
                              for k in KEYS))
    else:
        print(f'  operator: no run JSON at {path} -- column missing')
    if 'Q4' in row:
        print('  spread across the family (stdev/mean), Q4: '
              + ', '.join(f"{k.split('_')[0]} "
                          f"{row['Q4'][k]['stdev'] / row['Q4'][k]['mean']:.3f}"
                          for k in KEYS))

print('\n' + '=' * 78)
print('WHAT TO LOOK FOR')
print('=' * 78)
print('1. Does the family mean differ much from the single member Tables')
print('   22-24 print? If it barely moves, the single-member tables were')
print('   already representative and that is worth stating. If it moves a')
print('   lot, the report should quote the family.')
print('2. Does operator/Q4 on the FAMILY match the 2.42x that Table 24')
print('   reports on the single member? A large gap would mean the operator')
print('   was tuned, in effect, to the member it is scored on.')
print('3. The SPREAD. A small FEM spread beside a large operator spread')
print('   would say the operator is inconsistent across the family even')
print('   where its mean looks fine -- and that belongs in the report.')
print('\nSend the whole block over. It closes the family half of point 9.')
