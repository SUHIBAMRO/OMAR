# =====================================================================
#  CELL -- Real-GPU test of the geometric multigrid V-cycle preconditioner
#  (item #4) at N=1001. Read this before running.
#
#  RUN THIS AFTER N=701 (Round6_MGV_Recheck_N701.ipynb) HAS FINISHED AND
#  LOOKED GOOD -- N=701 is the first real test of the new approximate
#  coarse-solve fallback this fix needed (see that notebook's own header
#  for the full story: N=1001/1001/1401's coarsest hierarchy level is far
#  too large, ~32,000-62,000 free DOF, for the exact dense solve that
#  worked fine at N=401's ~1,300). If N=701 showed cg_failures did not
#  drop, or wall-clock wildly exceeded its own rough estimate, the coarse
#  fallback likely needs tuning (its iteration cap, its size threshold)
#  before spending more GPU time here.
#
#  N=1001's own hierarchy: coarsen_N chain [1001,501,251,126], bottoming
#  out at N=126 (~32,000 free DOF) -- actually SMALLER than N=701/1401's
#  own N=176 (~62,000), since 1000=2^3*125 carries one more factor of 2
#  than 700=2^2*175. So N=1001's approximate coarse solve is a bit
#  cheaper per V-cycle than N=701's own, even though N=1001's fine mesh
#  is larger.
#
#  ROUGH TIME ESTIMATE (2026-09-07, explicitly uncertain): a linear-in-DOF
#  extrapolation from N=401's confirmed 2615.8s (n_dof=321,602) to
#  N=1001's n_dof~2,004,002 (6.23x) gives ~4.5 h -- but this assumes
#  multigrid's own iteration count stays flat with N, unproven at this
#  scale, and does not yet reflect whatever N=701 measures for real.
#  Revise this estimate using N=701's actual result before trusting it.
#
#  WHAT THIS CELL DOES: forces a genuine fresh --precond_kind mgv solve
#  at ONLY N=1001 by deleting its checkpoint first. N=51/101/201/401/701
#  and the fine reference all reuse their existing good checkpoints.
#
#  WHAT TO CHECK when this finishes:
#    1. cg_failures at N=1001 -- was 40 with the original plain-Jacobi
#       run. Did mgv reduce it, ideally to 0?
#    2. wall_clock_s at N=1001 -- compare against both the ~4.5 h rough
#       estimate above AND against N=701's own real result (its per-DOF
#       cost should be in a similar ballpark, not wildly different,
#       since the coarse-solve mechanism is the same design).
#    3. The printed "[mgv] coarsest-level solve: ..." line -- confirm it
#       says "approximate CG fallback" (N=1001's coarsest level, N=126,
#       has ~32,000 free DOF, above the exact-solve threshold).
#    4. peak_stress_rel_err / peak_stress_ref -- should look sane,
#       consistent with N=401/N=701's own values.
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

# Force a genuine fresh solve for exactly N=1001 -- its existing
# checkpoint predates this whole preconditioner effort. N=51/101/201/401
# and N=701 (if already run) are deliberately left untouched and reuse
# their existing checkpoints.
FORCE_FRESH_NS = [1001]
for n in FORCE_FRESH_NS:
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh mgv solve at N={n}')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_mgv_N1001.json'

RESOLUTIONS = '51,101,201,401,701,1001'
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
print('  1. cg_failures at N=1001 -- was 40 with the original plain-Jacobi '
      'run. Did mgv reduce it, ideally to 0?')
print('  2. wall_clock_s at N=1001 -- rough pre-run estimate was ~4.5 h; '
      'compare also against N=701\'s own real per-DOF cost.')
print('  3. Look for "[mgv] coarsest-level solve: approximate CG fallback" '
      'in the log above.')
print('  4. peak_stress_rel_err / peak_stress_ref -- should look sane, '
      'consistent with N=401/N=701.')
