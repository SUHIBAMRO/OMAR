# =====================================================================
#  CELL -- Stage 2 of the item-4 (preconditioner) + item-3 (peak-stress
#  bug fix) combined re-run. ONLY run this after Stage 1
#  (Round6_Precond_Recheck_Stage1.ipynb) has finished AND its printed
#  checklist looked right (cg_failures reduced at N=401/701, peak_stress_ref
#  identical across rows). If Stage 1 did not help, re-think the
#  preconditioner before spending the time this stage needs.
#
#  This is the expensive, higher-risk half: N=1001 (~2,004,002 DOF) and
#  N=1401 (~3,925,602 DOF) have NEVER been solved fresh anywhere in this
#  project's history -- every previous run (including Table 6a's original
#  numbers) resumed from a checkpoint made before this session even
#  started, so their true from-scratch cost with EITHER preconditioner is
#  genuinely unknown. Expect this to be the single most expensive part of
#  the whole re-check; there is no reliable time estimate to give here
#  precisely because it has never been timed from scratch before.
#
#  Same mechanism as Stage 1: a checkpoint marked "done" short-circuits
#  solve_one() regardless of --precond_kind, so N=1001 and N=1401's
#  checkpoints (originally from BEFORE this session, solved with the old
#  Jacobi preconditioner and already hitting the CG cap 40/80 times) are
#  deleted here to force genuine fresh block2x2 solves. N=51/101/201 and
#  the fine reference keep resuming instantly as before; N=401/701 resume
#  from Stage 1's fresh block2x2 solve (no need to redo them).
#
#  Running all 7 resolutions again (not just the 2 new ones) means this
#  produces ONE final, directly comparable 7-row table in one file,
#  instead of needing a manual merge of two partial JSONs.
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

# Force a genuine fresh solve for exactly these two resolutions -- their
# existing checkpoints predate this whole session and were solved with the
# old scalar Jacobi preconditioner, already hitting the CG cap. Everything
# else (fine reference, N=51/101/201, and N=401/701 if Stage 1 already ran)
# is deliberately left untouched and will resume in seconds.
FORCE_FRESH_NS = [1001, 1401]
for n in FORCE_FRESH_NS:
    for suffix in ('.pt', '.pt.cg_state', '.pt.tmp'):
        p = os.path.join(CHECKPOINT_DIR, f'coarse_{GEOMETRY}_{MATERIAL}_{ORDER}_N{n}{suffix}')
        if os.path.exists(p):
            os.remove(p)
            print(f'[reset] deleted {p} to force a fresh block2x2 solve at N={n}')

OUT_JSON = f'{R}/high_dof_stress_qoi_B1_neo_hookean_block2x2_final.json'

RESOLUTIONS = '51,101,201,401,701,1001,1401'
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
print('\nThis is the final, directly-comparable 7-row table: fixed '
      'peak-stress QoI throughout, block2x2 preconditioner at every '
      'resolution that previously had CG failures (401/701/1001/1401). '
      'Check cg_failures across all 7 rows -- any remaining failures at '
      '1001/1401 mean block2x2 alone was not enough at that scale and '
      'should be reported honestly, not smoothed over, same as before.')
