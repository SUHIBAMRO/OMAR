# =====================================================================
#  STEP 5 — What does the GPU-FEM solve cost when CG is allowed to converge?
#
#  The point-8 sweep ran with cg_max_iter=2000 and CG never converged at
#  N >= 401: it hit that cap on EVERY Newton step. Report Table 20's times
#  at those sizes are therefore the cost of a truncated budget.
#
#  Section 8.5 now predicts the converged cost from two separately
#  measured pieces -- CG needs 5.011 x N iterations per Newton solve (from
#  the three resolutions that DID converge) and one iteration costs about
#  76 ns per degree of freedom at the large end. This cell tests that
#  prediction instead of leaving it a model.
#
#  Two resolutions only, chosen because they are the cheapest that are
#  actually in the truncated regime:
#
#    N=501  predicted ~2,510 CG iters/Newton, ~34 min
#    N=701  predicted ~3,513 CG iters/Newton, ~88 min
#
#  Those come from the measured 40.4 and 74.8 ms per CG iteration and an
#  assumed 20 Newton steps. The Newton count is the part that could move:
#  a converged CG should NEED fewer Newton steps than the truncated run
#  did (30 at N=701), so if anything the estimate is pessimistic there.
#
#  cg_max_iter is set to 8000 -- above the ~3,513 predicted for N=701 with
#  room to spare, so a prediction that is too low shows up as a longer run
#  rather than as another silent truncation. If CG STILL hits 8000 that is
#  itself the finding, and the cell says so.
#
#  NEEDS A GPU, and an A100 if you want the estimates above to hold; the
#  original sweep was measured on one. Resumable per resolution.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
OUT_DIR = f'{R}/gpu_fem_cg_check'
OUT_JSON = f'{OUT_DIR}/gpu_fem_cg_converged_B1_neo_hookean.json'
RESOLUTIONS = '501,701'
CG_MAX_ITER = 8000

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
os.makedirs(OUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), 'no GPU -- Runtime -> Change runtime type'
gpu = torch.cuda.get_device_name(0)
print('GPU:', gpu)
if 'A100' not in gpu:
    print('NOTE: the original sweep was measured on an A100. On', gpu,
          'the wall clocks will not be comparable to Table 20 -- the '
          'ITERATION COUNTS still will be, and they are the point of this '
          'run.')

# The prediction being tested, stated before the run so it cannot be
# adjusted afterwards to fit.
KBAR = 5.011
print('\n' + '=' * 70)
print('PREDICTION UNDER TEST (from section 8.5, fixed before this run)')
print('=' * 70)
for N in (501, 701):
    print(f'  N={N}: CG should converge in about {KBAR * N:,.0f} iterations '
          f'per Newton solve')
print(f'  and the Newton count should be at or below the truncated run\'s '
      f'(20 at N=501, 30 at N=701)')
print('=' * 70)

# Same settings as the sweep in every respect except the CG budget, so the
# only thing that changed is the one thing under test.
run([sys.executable, '-u', '-m', 'omar_pfem.gpu_fem_scaling_sweep',
     '--geometry', 'B1', '--material', 'neo_hookean', '--order', 'Q4',
     '--resolutions', RESOLUTIONS,
     '--nsteps', '10',
     '--newton_max', '30',
     '--newton_tol', '1e-7',
     '--cg_tol', '1e-6',
     '--cg_max_iter', str(CG_MAX_ITER),
     '--out_json', OUT_JSON,
     '--checkpoint_dir', OUT_DIR])

# ---- read the counters back and score the prediction ----------------
d = json.load(open(OUT_JSON))
print('\n' + '=' * 78)
print('RESULT')
print('=' * 78)
print('%-6s %10s %8s %10s %8s %14s %12s'
      % ('N', 'DOF', 'newton', 'cg iters', 'failed', 'cg/newton', 'solve'))
still_capped = []
for r in sorted(d['rows'], key=lambda r: r['N']):
    s = r['stats']
    per = s['cg_iters_total'] / max(1, s['newton_iters_total'])
    print('%-6d %10d %8d %10d %8d %14.1f %10.1f min'
          % (r['N'], r['n_dof'], s['newton_iters_total'], s['cg_iters_total'],
             s['cg_failures'], per, r['solve_s'] / 60))
    if s['cg_failures'] > 0:
        still_capped.append(r['N'])

print()
if still_capped:
    print(f'CG STILL hit the {CG_MAX_ITER:,} cap at N={still_capped}. The '
          f'prediction of ~{KBAR} x N is then too low, and the real '
          f'requirement is larger than this run can show. That is a result, '
          f'not a failure -- send it over and section 8.5 gets the correction '
          f'rather than the confirmation.')
else:
    print('CG converged at every resolution, so these are the first converged')
    print('timings for this solver at these sizes. Against the prediction:')
    for r in sorted(d['rows'], key=lambda r: r['N']):
        s = r['stats']
        per = s['cg_iters_total'] / max(1, s['newton_iters_total'])
        pred = KBAR * r['N']
        print('  N=%-5d predicted %7.0f, measured %7.1f  (%+.1f%%)'
              % (r['N'], pred, per, (per / pred - 1) * 100))
    print()
    print('And against Table 20\'s truncated times for the same sizes')
    print('(1,616 s at N=501, 4,487 s at N=701 on an A100):')
    OLD = {501: 1616.06, 701: 4487.19}
    for r in sorted(d['rows'], key=lambda r: r['N']):
        if r['N'] in OLD:
            print('  N=%-5d truncated %7.0f s, converged %7.0f s  (%+.0f%%)'
                  % (r['N'], OLD[r['N']], r['solve_s'],
                     (r['solve_s'] / OLD[r['N']] - 1) * 100))
    print()
    print('That last comparison is only meaningful on an A100. It answers the')
    print('question section 8.5 currently leaves open: whether allowing CG to')
    print('converge makes the solve more expensive or less, given it also')
    print('removes Newton steps.')

print('\nJSON at', OUT_JSON)
print('Send it, or the block above, and section 8.5 stops being a model.')
