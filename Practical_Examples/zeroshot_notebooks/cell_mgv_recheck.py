# =====================================================================
#  CELL -- Real-GPU test of the geometric multigrid V-cycle preconditioner
#  (item #4, second attempt) on the cheapest resolution that block2x2
#  failed at. Read this before running.
#
#  THE STORY SO FAR: block2x2 (a 2x2 per-node block-Jacobi preconditioner)
#  was tried first and measured to NOT help -- a real GPU re-run showed
#  IDENTICAL cg_failures at N=401 (20) and N=701 (30) to plain scalar
#  Jacobi, with essentially unchanged wall-clock. Rather than accept that
#  as a dead end or just "note the limitation," a genuine geometric
#  multigrid V-cycle was built instead (multigrid_precond.py) -- the
#  textbook-correct fix for CG stagnation on a structured elasticity
#  grid, and one that stays fully matrix-free (never forms K), unlike
#  the other option considered (incomplete Cholesky, which needs the
#  actual sparse matrix and would have undercut this whole solver's
#  reason for existing).
#
#  VALIDATED BEFORE THIS CELL EVER RAN (validate_multigrid_precond.py,
#  CPU only, small meshes): the V-cycle-preconditioned solve reproduces
#  the SAME converged displacement as the existing plain-Jacobi solve on
#  BOTH B1 and B2 (including B2's own partially-fixed-DOF nodes, exactly
#  where block2x2 needed its own special-case handling), to 1e-12
#  relative difference -- and its iteration-count advantage over plain
#  Jacobi GREW with problem size rather than staying flat: 3.0x fewer
#  iterations at N=9, 5.2x at N=33, 9.7x at N=65. That growing advantage
#  is the actual signature of correctly-functioning multigrid (a
#  preconditioner whose benefit stays constant or shrinks with N is not
#  really fixing the underlying conditioning problem), which is why this
#  is worth spending real GPU time on where block2x2 was not.
#
#  WHAT THIS CELL DOES: forces a genuine fresh --precond_kind mgv solve
#  at ONLY N=401 -- the cheapest of the four resolutions that hit the CG
#  iteration cap in the original run (N=401: 20 failures, ~27 min;
#  N=701: 30, ~75 min; N=1001: 40; N=1401: 80) -- by deleting its
#  checkpoint first. N=51/101/201 and the N=2236 fine reference reuse
#  their existing good checkpoints and resume in seconds. This is
#  deliberately the SAME cost-conscious staging the block2x2 recheck
#  used: answer "does this actually help?" at the cheapest failing
#  resolution before spending time on 701/1001/1401.
#
#  WHAT TO CHECK when this finishes -- printed as an explicit checklist
#  at the end, but stated here too:
#    1. cg_failures at N=401 -- did mgv reduce it from 20 (ideally to 0)?
#    2. wall_clock_s at N=401 -- mgv's own CPU validation showed HIGHER
#       per-iteration cost than plain Jacobi (more work per V-cycle:
#       several matvecs across multiple mesh levels plus a coarse-level
#       factorization, versus Jacobi's one elementwise division), so a
#       fair reading needs BOTH numbers together: fewer cg_failures with
#       a similar or better wall-clock is a clean win; fewer failures
#       but much slower is a real trade-off to weigh, not a clean win.
#    3. peak_stress_ref -- should still be the fixed value from the
#       already-verified bug fix (identical across every row in a full
#       sweep); this cell only runs one resolution so there is nothing
#       to compare it ACROSS yet, just confirm it looks like a sane
#       physical value.
#  Only proceed to N=701/1001/1401 if N=401 here shows a genuine
#  reduction in cg_failures, not just a plausible-looking number.
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

# Force a genuine fresh solve for exactly N=401 -- its existing checkpoint
# predates this whole preconditioner effort and was solved with plain
# Jacobi, already hitting the CG cap 20 times. N=51/101/201 and the
# N=2236 fine reference are deliberately left untouched and will resume
# in seconds from their already-good checkpoints.
FORCE_FRESH_NS = [401]
for n in FORCE_FRESH_NS:
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh mgv solve at N={n}')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_mgv_N401.json'

RESOLUTIONS = '51,101,201,401'
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
print('  1. cg_failures at N=401 -- the ORIGINAL run (plain Jacobi) had 20; '
      'the block2x2 re-run (2026-09-07) ALSO had 20, no improvement. Did mgv '
      'reduce this, ideally to 0?')
print('  2. wall_clock_s at N=401 -- mgv is expected to cost MORE per CG '
      'iteration than Jacobi (each V-cycle does several matvecs across '
      'multiple mesh levels plus a coarse-level factorization); judge the '
      'trade-off using BOTH cg_failures and wall_clock_s together, not '
      'wall-clock alone.')
print('  3. peak_stress_rel_err / peak_stress_ref -- should look like a sane '
      'physical value, consistent with the already-verified bug fix.')
print('  Only proceed to N=701 (and beyond) if this shows a genuine '
      'reduction in cg_failures -- report the real number either way, not '
      'a smoothed-over one.')
