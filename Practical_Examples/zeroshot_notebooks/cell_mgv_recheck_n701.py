# =====================================================================
#  CELL -- Real-GPU test of the geometric multigrid V-cycle preconditioner
#  (item #4) at N=701, the second resolution that both plain Jacobi and
#  block2x2 failed to converge at. Read this before running.
#
#  THE STORY SO FAR: N=401 was tested first with this same preconditioner
#  and, after fixing a real performance bug (a serial dense-factor build
#  at the coarsest level), converged cleanly: cg_failures 20 -> 0, wall
#  clock ~43.6 min (was ~87 min before that fix, ~27 min for the old,
#  non-converging plain-Jacobi run). That cleared this item's own stated
#  bar for proceeding ("only continue if this shows a genuine reduction
#  in cg_failures").
#
#  A SECOND issue was found and fixed before this cell was ever written,
#  by checking the actual facts rather than assuming N=401's fix
#  generalizes: coarsen_N requires (N-1) even at every halving, so how
#  deep a given N can coarsen depends on how many factors of 2 divide
#  (N-1), not on N's own size. N=401 happened to coarsen 4 times down to
#  N=26 (~1,300 free DOF, cheap to solve exactly). N=701 only coarsens
#  TWICE, bottoming out at N=176 (~62,000 free DOF) -- an exact dense
#  solve there would be computationally infeasible (O(n^3) time, O(n^2)
#  memory; a rough estimate put a single factorization at tens of hours,
#  and a solve needs about 20 of them). Fixed by adding a size threshold
#  to the coarsest-level solve: below it, the same exact dense LU as
#  before; above it (N=701's case), an APPROXIMATE coarse solve instead
#  -- a capped, cheap matrix-free CG call with Jacobi preconditioning,
#  a standard multigrid variant, not driven to full convergence.
#  Validated on CPU that this fallback logic is correct (reproduces the
#  same solution as the exact dense path on small test cases) BEFORE
#  this cell was ever pointed at N=701's real, much-larger coarsest
#  level -- see multigrid_precond.py's build_mg_precond_apply docstring
#  and PROJECT_STATUS.md's item #4 for the full story.
#
#  WHAT IS GENUINELY UNKNOWN, stated honestly: N=701 has never run this
#  approximate coarse-solve path on real hardware. A rough linear-in-DOF
#  extrapolation from N=401's confirmed 2615.8s puts N=701 (3.06x the
#  DOF) around ~2.2 h -- but that assumes multigrid's own iteration
#  count stays flat with N, which N=701's shallower hierarchy (3 levels
#  vs. N=401's 5) and inexact coarse solve could break. This cell exists
#  to get a REAL data point, not to confirm a guess.
#
#  WHAT THIS CELL DOES: forces a genuine fresh --precond_kind mgv solve
#  at ONLY N=701 by deleting its checkpoint first. N=51/101/201/401 and
#  the fine reference all reuse their existing good checkpoints (N=401's
#  is now the REAL converged mgv result, not a stale plain-Jacobi one)
#  and resume in seconds.
#
#  WHAT TO CHECK when this finishes:
#    1. cg_failures at N=701 -- was 30 with both plain Jacobi and
#       block2x2. Did mgv reduce it, ideally to 0?
#    2. wall_clock_s at N=701 -- compare against the ~2.2 h rough
#       estimate above. A large overshoot would mean the approximate
#       coarse solve needs tuning (more/fewer CG iterations, a different
#       threshold) before trusting N=1001/N=1401's own estimates.
#    3. The printed "[mgv] coarsest-level solve: ..." line -- confirm it
#       says "approximate CG fallback", not "exact dense LU" (would mean
#       the threshold logic did not trigger as expected for this N).
#    4. peak_stress_rel_err / peak_stress_ref -- should look like a sane
#       physical value, consistent with N=401's own 31.99%.
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

# Force a genuine fresh solve for exactly N=701 -- its existing checkpoint
# predates this whole preconditioner effort. N=51/101/201/401 and the
# fine reference are deliberately left untouched and will resume in
# seconds from their already-good checkpoints (N=401's is now the real
# converged mgv result).
FORCE_FRESH_NS = [701]
for n in FORCE_FRESH_NS:
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh mgv solve at N={n}')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_mgv_N701.json'

RESOLUTIONS = '51,101,201,401,701'
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
print('\nWHAT TO CHECK:')
print('  1. cg_failures at N=701 -- was 30 with both plain Jacobi and '
      'block2x2. Did mgv reduce it, ideally to 0?')
print('  2. wall_clock_s at N=701 -- rough pre-run estimate was ~2.2 h '
      '(linear-in-DOF extrapolation from N=401, an assumption this run '
      'either confirms or breaks).')
print('  3. Look for "[mgv] coarsest-level solve: approximate CG fallback" '
      'in the log above -- confirms the new size-based threshold triggered '
      'as expected for this N (its coarsest level, N=176, has ~62,000 free '
      'DOF, far above the exact-solve threshold).')
print('  4. peak_stress_rel_err / peak_stress_ref -- should look sane, '
      'consistent with N=401\'s own 31.99%.')
print('  Report the real numbers either way -- this result also informs '
      'whether N=1001/N=1401\'s own time estimates need revising before '
      'running them.')
