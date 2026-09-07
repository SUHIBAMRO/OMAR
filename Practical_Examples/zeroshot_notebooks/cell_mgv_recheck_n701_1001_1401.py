# =====================================================================
#  CELL -- Real-GPU test of the geometric multigrid V-cycle preconditioner
#  (item #4) at ALL THREE remaining target resolutions, N=701, N=1001,
#  N=1401, in one run. Read this before starting.
#
#  THE STORY SO FAR: N=401 was tested first and, after fixing a real
#  performance bug (a serial dense-factor build at the coarsest level),
#  converged cleanly: cg_failures 20 -> 0, wall-clock ~43.6 min (was ~87
#  min before that fix, ~27 min for the old, non-converging plain-Jacobi
#  run). That cleared this item's own stated bar for proceeding.
#
#  A SECOND, more serious issue was found and fixed before this cell was
#  ever written -- checked directly, not assumed to generalize from
#  N=401: coarsen_N requires (N-1) even at every halving, so how deep a
#  given N can coarsen depends on how many factors of 2 divide (N-1),
#  not on N's own size. N=401 happened to coarsen 4 times down to N=26
#  (~1,300 free DOF, cheap to solve exactly). N=701 and N=1401 both only
#  coarsen down to N=176 (~62,000 free DOF); N=1001 to N=126 (~32,000)
#  -- an exact dense LU solve at that scale is not merely slower, it is
#  computationally infeasible (O(n^3) time; a rough estimate put a
#  single factorization at tens of hours, and a solve needs about 20 of
#  them). Fixed by adding a size threshold to the coarsest-level solve
#  in multigrid_precond.py's build_mg_precond_apply: below it, the same
#  exact dense LU N=401 used; above it (all three resolutions this cell
#  targets), an APPROXIMATE coarse solve instead -- a capped, cheap
#  matrix-free CG call with Jacobi preconditioning, a standard multigrid
#  variant, not driven to full convergence. Validated on CPU that this
#  fallback logic is correct (reproduces the same solution as the exact
#  dense path on small test cases) BEFORE ever pointing it at these
#  three real, much larger coarsest levels -- see
#  multigrid_precond.py's build_mg_precond_apply docstring and
#  PROJECT_STATUS.md's item #4 for the full story.
#
#  WHAT IS GENUINELY UNKNOWN, stated honestly: none of N=701/1001/1401
#  has ever run this approximate coarse-solve path on real hardware.
#  Rough linear-in-DOF extrapolations from N=401's confirmed 2615.8s
#  give ~2.2h / ~4.5h / ~8.9h respectively (~15h combined) -- but that
#  assumes multigrid's own iteration count stays flat with N, unproven
#  at this scale, and N=701/1401's shallower hierarchies (3-4 levels vs.
#  N=401's 5) plus the inexact coarse solve could both push the real
#  numbers higher. This cell exists to get real data points, not to
#  confirm a guess -- read each resolution's own printed numbers as it
#  finishes.
#
#  WHAT THIS CELL DOES: runs high_dof_convergence_study ONCE across
#  --resolutions 51,101,201,401,701,1001,1401 with --precond_kind mgv,
#  so all three new resolutions solve in a single process, in order
#  (701 -> 1001 -> 1401). N=51/101/201/401 and the fine reference reuse
#  their already-good checkpoints (N=401's is now the REAL converged
#  mgv result) and resume in seconds.
#
#  RESUMABILITY IF COLAB DISCONNECTS MID-RUN: a naive "delete this
#  checkpoint, then solve" script (as N=401's own recheck cell used) is
#  fine for a single resolution finishing in under an hour, but wrong to
#  reuse verbatim for a ~15h combined run -- if Colab disconnects during,
#  say, N=1001, simply re-running that kind of cell from scratch would
#  DELETE N=1001's own already-in-progress checkpoint again before
#  resuming, throwing away real progress. Fixed here with a marker file
#  per target N on Drive (mgv_reset_done_N{n}.marker): the forced
#  checkpoint reset for a given N only ever happens the FIRST time this
#  cell runs for that N; every subsequent re-run (including after a
#  disconnect) skips the reset and lets solve_matrix_free's own
#  per-Newton-iteration checkpoint resume exactly where it left off --
#  and any resolution that already finished in a prior run of this cell
#  resumes in seconds, just like N=51/101/201/401 always have.
# =====================================================================
import os, subprocess, sys, time

def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
                    'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout',
                    'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard',
                    'origin/claude/claude-code-question-d307wp'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE - Runtime > Change runtime type > GPU strongly recommended, '
           'this study is expensive even on GPU')

R = '/content/drive/MyDrive/pfem_run'
CHECKPOINT_DIR = '/content/drive/MyDrive/pfem_ckpt'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GEOMETRY = 'B1'
MATERIAL = 'neo_hookean'
ORDER = 'Q4'

# Force each of N=701/1001/1401's checkpoint to a genuine fresh mgv solve,
# but only the FIRST time this cell ever runs for that N -- see the
# resumability note above for why a plain unconditional delete would be
# wrong to reuse for a run this long. N=51/101/201/401 are never touched.
FORCE_FRESH_NS = [701, 1001, 1401]
for n in FORCE_FRESH_NS:
    marker = os.path.join(CHECKPOINT_DIR, f'mgv_reset_done_N{n}.marker')
    if os.path.exists(marker):
        print(f'[resume] N={n} was already reset by an earlier run of this '
              f'cell -- leaving its checkpoint alone so it resumes '
              f'(or reuses its finished result) instead of restarting.')
        continue
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh mgv solve at N={n}')
    with open(marker, 'w') as f:
        f.write('reset done\n')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_mgv_N701_1001_1401.json'

RESOLUTIONS = '51,101,201,401,701,1001,1401'
FINE_N = 2236

t0 = time.time()
run([
    sys.executable, '-u', '-m', 'omar_pfem.high_dof_convergence_study',
    '--geometry', GEOMETRY, '--material', MATERIAL,
    '--resolutions', RESOLUTIONS, '--fine_N', str(FINE_N),
    '--orders', ORDER,
    '--precond_kind', 'mgv',
    '--checkpoint_dir', CHECKPOINT_DIR,
    '--out_json', OUT_JSON,
    '--cg_progress_every', '500',
    '--cg_checkpoint_every', '2000',
])
elapsed = time.time() - t0

print(f'\nDone in {elapsed/3600:.2f} h. Results: {OUT_JSON}')
print('\nWHAT TO CHECK, for EACH of N=701/1001/1401:')
print('  1. cg_failures -- was 30/40/80 respectively with both plain Jacobi '
      'and block2x2. Did mgv reduce each, ideally to 0?')
print('  2. wall_clock_s -- rough pre-run estimates were ~2.2h/4.5h/8.9h; '
      'compare each against its own estimate, and against each other (N=701 '
      'and N=1401 share the same coarsest-level size, N=176, so their '
      'per-DOF cost should be comparable).')
print('  3. Look for "[mgv] coarsest-level solve: approximate CG fallback" '
      'in the log above, once per resolution -- confirms the new size-based '
      'threshold triggered as expected for all three.')
print('  4. peak_stress_rel_err / peak_stress_ref -- should look sane and '
      'consistent with N=401\'s own 31.99%.')
print('  If cg_failures is 0 across all three, item #4 is DONE -- update '
      'PROJECT_STATUS.md with the final N=51..1401 comparison.')
