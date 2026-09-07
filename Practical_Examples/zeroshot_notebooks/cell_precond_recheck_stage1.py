# =====================================================================
#  CELL -- Stage 1 of the item-4 (preconditioner) + item-3 (peak-stress
#  bug fix) combined re-run. Read this before running.
#
#  TWO independent fixes landed since the first high-DOF/peak-stress run
#  (2026-09-06/07):
#    1. compute_peak_stress_error() had a real bug: the "reference" peak
#       stress value was max(|P|) over the COARSE mesh's own Gauss
#       points, a point set that gets denser (and so gives a bigger max)
#       as the coarse resolution N grows -- confirmed directly in the old
#       data, where peak_stress_ref itself drifted from 13.9 (N=51) to
#       39.3 (N=1401). Fixed: the fine reference's own true peak-stress
#       point is now found ONCE (find_fine_peak_stress) and every coarse
#       resolution is compared against that SAME fixed point. Verified on
#       CPU: an identity check (coarse==fine) gives EXACTLY zero error,
#       and peak_stress_ref is now identical across different N in a
#       sweep. This part just needs re-running through this fixed code --
#       it does not need GPU time by itself, since it is a post-processing
#       step applied to already-solved fields.
#    2. A new opt-in 2x2 block-Jacobi preconditioner (--precond_kind
#       block2x2) exists as an alternative to the scalar Jacobi every
#       previous number used. The ORIGINAL run had CG hit its 2000-
#       iteration cap without reaching cg_tol at N=401 (20 failures), 701
#       (30), 1001 (40), 1401 (80) -- this is what item 4 exists to fix.
#
#  THE CATCH: a checkpoint marked "done" (a full Newton solve already
#  completed) makes solve_one() return that SAME old solution immediately,
#  regardless of --precond_kind -- switching the preconditioner flag alone
#  changes NOTHING unless the checkpoint for that resolution is deleted
#  first, forcing a genuine fresh solve. This cell deletes ONLY the
#  checkpoints for N=401 and N=701 (and any leftover mid-CG .cg_state
#  file for them) before running -- the fine reference (N=2236) and
#  N=51/101/201 keep their existing, already-good checkpoints and resume
#  in seconds, since nothing about them needs re-testing.
#
#  WHY STAGE 1 AND NOT ALL 7 AT ONCE: N=1001 and N=1401 have never
#  actually been solved fresh in this whole project -- every previous run
#  free-rode on a checkpoint from before this session, so their true
#  from-scratch cost with EITHER preconditioner is unknown, and at
#  ~2-4 million DOF each it could be a genuinely long solve. Forcing
#  fresh solves at all four failing resolutions in one shot risks many
#  hours before finding out whether block2x2 even helps. This stage only
#  re-solves the two CHEAPER failing resolutions (401, 701 -- known
#  costs from the original run: ~27 min and ~75 min respectively with the
#  OLD preconditioner) to answer that question first. If the CG failure
#  counts at 401/701 do not improve here, there is no reason to spend
#  hours more on 1001/1401 without further preconditioner work --
#  re-think before running the Stage 2 notebook.
#
#  Expected new compute this stage: roughly the ~27+75 = ~100 min the
#  original run needed for these two resolutions (block2x2 may be
#  faster or slower per-iteration; iteration COUNT is the thing to
#  actually compare in the output, not assumed savings).
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

# Force a genuine fresh solve for exactly these two resolutions -- delete
# their checkpoint (and any stray mid-CG state file) so solve_one() cannot
# short-circuit to the OLD Jacobi-preconditioned answer. N=51/101/201 and
# the N=2236 fine reference are deliberately left untouched.
FORCE_FRESH_NS = [401, 701]
for n in FORCE_FRESH_NS:
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh block2x2 solve at N={n}')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_block2x2_stage1.json'

RESOLUTIONS = '51,101,201,401,701'
FINE_N = 2236

t0 = time.time()
run([
    sys.executable, '-u', '-m', 'omar_pfem.high_dof_convergence_study',
    '--geometry', GEOMETRY, '--material', MATERIAL,
    '--resolutions', RESOLUTIONS, '--fine_N', str(FINE_N),
    '--orders', ORDER,
    '--precond_kind', 'block2x2',
    '--checkpoint_dir', CHECKPOINT_DIR,
    '--out_json', OUT_JSON,
    '--cg_progress_every', '500',
    '--cg_checkpoint_every', '2000',
])
elapsed = time.time() - t0

print(f'\nDone in {elapsed/3600:.2f} h. Results: {OUT_JSON}')
print('\nWHAT TO CHECK before running Stage 2:')
print('  1. cg_failures at N=401 and N=701 -- did block2x2 reduce them '
      'from the original 20 and 30, ideally to 0?')
print('  2. peak_stress_ref -- should now be IDENTICAL across every row '
      'in this file (the bug fix). If it still varies row to row, STOP '
      'and report this -- the fix did not take effect.')
print('  3. peak_stress_rel_err -- does it now move in a sane direction '
      'with N, or is it still noisy? Some noise is still expected/normal '
      'for a pointwise QoI even when everything is correct -- what '
      'matters is whether it looks like the earlier small-scale CPU '
      'checks (a believable positive rate, e.g. ~0.5-1.0), not perfectly '
      'monotonic.')
print('  Only run the Stage 2 notebook (N=1001, N=1401 -- the expensive, '
      'never-fresh-solved resolutions) if (1) and (2) look right here.')
