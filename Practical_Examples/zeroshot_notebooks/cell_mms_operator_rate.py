# =====================================================================
#  OPTIONAL — give the MMS operator a convergence rate of its own
#
#  Section 8.11 currently states this as a limitation, in its own words:
#  "The operator was trained and scored at one mesh only, so it has no
#  convergence rate of its own and cannot appear in Table 23; producing
#  one means a training run per refinement, not a solve per refinement."
#
#  This does those training runs. N=17 is already measured (8.2 min on an
#  A100, operator/Q4 = 2.42x in L2). Adding N=9 and N=33 gives three
#  points, which is what Table 23's rates are fitted on for Q4 and Q9.
#
#  Then the operator can be read the same way the two solvers are: does
#  its error fall with refinement at all, and if so at what rate? There
#  is no theory to check it against -- unlike Q4 and Q9, whose expected
#  rates Table 23 already confirms -- so whatever comes out is a
#  measurement, not a verification. It could be flat. That would itself
#  be worth reporting: an operator whose error does not improve with the
#  mesh is a different object from a discretization that does.
#
#  COST, extrapolated from the measured N=17 run by degrees of freedom:
#    N=9   162 DOF   ~2-3 min training
#    N=33  2,178 DOF  ~30 min training
#  plus the Q4 and Q9 FEM references each run solves for itself, which at
#  N=33 are the slower part. Budget about an hour, and it is bounded --
#  nothing here can run away.
#
#  GPU preferred (FP32 training); the FEM references are FP64 on CPU.
#  Resumable: a resolution whose JSON exists is skipped.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
OUT = '/content/drive/MyDrive/pfem_run/mms/operator_rate'
RESOLUTIONS = [9, 33]          # 17 is already done and is read back below
DONE_N17 = '/content/drive/MyDrive/pfem_run/mms/mms_operator_B1_neo_hookean.json'

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
os.makedirs(OUT, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else 'NONE (this will be slow)')

# The functional check first, exactly as the N=17 production run did it.
# It is cheap and it is what makes the numbers mean anything: the Q4
# solution must be the exact minimizer of the Pi the network minimizes.
print('\n' + '=' * 70)
print('verifying the energy functional against the FEM solver')
print('=' * 70)
run([sys.executable, '-u', '-m', 'omar_pfem.test_mms_operator'])

for N in RESOLUTIONS:
    oj = f'{OUT}/mms_operator_B1_neo_hookean_N{N}.json'
    if os.path.exists(oj):
        print(f'\n[N={N}] {oj} exists -- skipping.')
        continue
    print('\n' + '=' * 70)
    print(f'[N={N}] training the operator on the manufactured family')
    print('=' * 70)
    # Identical to the N=17 production run in every respect except N, so
    # the three points differ only in the mesh.
    run([sys.executable, '-u', '-m', 'omar_pfem.mms_operator',
         '--material', 'neo_hookean', '--N', str(N),
         '--ntrain', '64', '--ntest', '16',
         '--epochs', '2000', '--batch_size', '8', '--validate_every', '25',
         '--out_dir', f'{OUT}/N{N}', '--out_json', oj])

# ---- read all three back and fit the rate ---------------------------
print('\n' + '=' * 78)
print('THE OPERATOR ACROSS THREE MESHES')
print('=' * 78)
rows = []
for N in sorted(RESOLUTIONS + [17]):
    path = DONE_N17 if N == 17 else f'{OUT}/mms_operator_B1_neo_hookean_N{N}.json'
    if not os.path.exists(path):
        print(f'  N={N}: {path} not found -- skipped')
        continue
    d = json.load(open(path))
    op = d['operator_on_the_reference_member']
    q4 = d['fem_reference_same_mesh']['Q4']
    rows.append(dict(N=N, dof=d['n_dof'], op=op['L2_rel'], q4=q4['L2_rel'],
                     oph1=op['H1_semi_rel'], q4h1=q4['H1_semi_rel']))

print('%-5s %8s %12s %12s %10s %12s %12s %10s'
      % ('N', 'DOF', 'operator L2', 'Q4 L2', 'op/Q4', 'operator H1',
         'Q4 H1', 'op/Q4'))
for r in rows:
    print('%-5d %8d %12.4e %12.4e %9.2fx %12.4e %12.4e %9.2fx'
          % (r['N'], r['dof'], r['op'], r['q4'], r['op'] / r['q4'],
             r['oph1'], r['q4h1'], r['oph1'] / r['q4h1']))

if len(rows) >= 3:
    import math
    def rate(key):
        # least squares on log(error) against log(h), h ~ 1/(N-1)
        xs = [math.log(1.0 / (r['N'] - 1)) for r in rows]
        ys = [math.log(r[key]) for r in rows]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den

    print('\nfitted convergence rates in h (least squares on three points):')
    print('  operator L2   %.2f' % rate('op'))
    print('  Q4       L2   %.2f   (theory 2, and Table 23 measures 1.98)'
          % rate('q4'))
    print('  operator H1   %.2f' % rate('oph1'))
    print('  Q4       H1   %.2f   (theory 1, and Table 23 measures 1.00)'
          % rate('q4h1'))
    ratios = [r['op'] / r['q4'] for r in rows]
    print(f'\noperator/Q4 in L2 across the three meshes: '
          + ', '.join(f'{x:.2f}x' for x in ratios))
    if max(ratios) / min(ratios) < 1.3:
        print('  Roughly constant, which would mean the operator inherits Q4\'s')
        print('  rate and sits a fixed factor above it.')
    elif ratios[-1] > ratios[0]:
        print('  GROWING with refinement: the operator falls further behind the')
        print('  finer the mesh, so it does NOT inherit Q4\'s rate. That is the')
        print('  more interesting outcome and the more important to report.')
    else:
        print('  Shrinking with refinement: the operator closes on Q4 as the')
        print('  mesh refines.')
    print('\nThe Q4 rates above are the control: they should land near Table')
    print('23\'s 1.98 and 1.00. If they do not, this run is not comparable to')
    print('that table and the operator rates should not be quoted either.')
else:
    print('\nFewer than three resolutions present -- no rate fitted.')

print('\nJSONs under', OUT)
print('Send the block above; section 8.11 loses its "no convergence rate"')
print('limitation, or gains a measured reason why the operator has none.')
