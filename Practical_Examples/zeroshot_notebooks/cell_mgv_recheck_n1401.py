# =====================================================================
#  CELL -- Real-GPU test of the geometric multigrid V-cycle preconditioner
#  (item #4) at N=1401 -- the last and most expensive resolution this
#  item exists to fix. Read this before running.
#
#  RUN THIS LAST, AFTER N=701 AND N=1001 HAVE BOTH FINISHED AND LOOKED
#  GOOD. N=1401's coarsest hierarchy level (N=176, ~62,000 free DOF) is
#  IDENTICAL in size to N=701's own coarsest level (both chains pass
#  through N=176: 1401=2^3*175+1 coarsens to 701,351,176 while 701 itself
#  coarsens to 351,176) -- so N=701's real result is the single best
#  predictor available for how N=1401's approximate coarse-solve fallback
#  will behave, more so than any straight-line extrapolation from N=401.
#
#  ROUGH TIME ESTIMATE (2026-09-07, explicitly uncertain): a linear-in-DOF
#  extrapolation from N=401's confirmed 2615.8s (n_dof=321,602) to
#  N=1401's n_dof~3,925,602 (12.21x) gives ~8.9 h -- the largest and least
#  certain of the three estimates, since it compounds both the largest
#  DOF ratio and the same coarse-solve-fallback uncertainty N=701/1001
#  carry. If N=701's real wall-clock already overshot its own ~2.2 h
#  estimate by some factor, scale this number up by roughly the same
#  factor before trusting it -- do not run N=1401 blind on the original
#  estimate if N=701 already showed it was optimistic.
#
#  WHAT THIS CELL DOES: forces a genuine fresh --precond_kind mgv solve
#  at ONLY N=1401 by deleting its checkpoint first. N=51/101/201/401/701/
#  1001 and the fine reference all reuse their existing good checkpoints.
#
#  WHAT TO CHECK when this finishes:
#    1. cg_failures at N=1401 -- was 80 with the original plain-Jacobi
#       run (the worst of all four target resolutions). Did mgv reduce
#       it, ideally to 0?
#    2. wall_clock_s at N=1401 -- compare against the ~8.9 h rough
#       estimate AND against N=701's real result (same coarsest-level
#       size, N=176, so their per-V-cycle coarse-solve cost should be
#       comparable; the difference should mostly reflect the larger fine
#       mesh).
#    3. The printed "[mgv] coarsest-level solve: ..." line -- confirm it
#       says "approximate CG fallback".
#    4. peak_stress_rel_err / peak_stress_ref -- should look sane,
#       consistent with N=401/N=701/N=1001.
#  Once this finishes, item #4's actual deliverable (N=1001 and N=1401
#  reaching full CG convergence) is done -- update PROJECT_STATUS.md
#  with the final numbers across all four resolutions.
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

# Force a genuine fresh solve for exactly N=1401 -- its existing
# checkpoint predates this whole preconditioner effort. N=51/101/201/401/
# 701/1001 are deliberately left untouched and reuse their existing
# checkpoints.
FORCE_FRESH_NS = [1401]
for n in FORCE_FRESH_NS:
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh mgv solve at N={n}')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_mgv_N1401.json'

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
print('\nWHAT TO CHECK:')
print('  1. cg_failures at N=1401 -- was 80 with the original plain-Jacobi '
      'run. Did mgv reduce it, ideally to 0?')
print('  2. wall_clock_s at N=1401 -- rough pre-run estimate was ~8.9 h; '
      'compare also against N=701\'s real result (same coarsest-level '
      'size, N=176).')
print('  3. Look for "[mgv] coarsest-level solve: approximate CG fallback" '
      'in the log above.')
print('  4. peak_stress_rel_err / peak_stress_ref -- should look sane, '
      'consistent with N=401/N=701/N=1001.')
print('  If cg_failures is 0 here too, item #4 is DONE -- update '
      'PROJECT_STATUS.md with the final N=51..1401 comparison.')
