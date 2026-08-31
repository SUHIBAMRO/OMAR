# =====================================================================
#  STEP 1 — B2 x Neo-Hookean: diagnose the load, then repair if needed
#
#  B2 x Mooney-Rivlin and B2 x Arruda-Boyce were repaired on 2026-08-29:
#  their cached nodal force was overstated by a MESH-DEPENDENT factor,
#  about 13x at N=21 and 21x at N=33. B2 x Neo-Hookean was NOT in that
#  run, yet its eval report shows the same signature -- a relative error
#  of 8.09 that barely moves with the mesh.
#
#  This cell does NOT assume the same cause. It runs the repair in
#  --dry_run first and looks at what comes back, because the two
#  outcomes need different next steps:
#
#    * overstatement > 1 -> same bug. Repair, delete the models trained
#      on the bad load, retrain.
#    * overstatement = 1 -> the cache was ALREADY correct, and the 8.09
#      has a DIFFERENT cause. Retraining would then be chasing the wrong
#      thing, and the cell says so instead of proceeding.
#
#  Nothing is written and no model is deleted unless the dry run finds a
#  real overstatement.
#
#  No GPU needed. Seconds to a couple of minutes.
# =====================================================================
import os
import re
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
CASE = 'B2_neo_hookean'
OUT = f'/content/drive/MyDrive/pfem_run/zeroshot_{CASE}'

from google.colab import drive
drive.mount('/content/drive')


def sh(cmd, **kw):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, text=True, **kw)


# ---- code -----------------------------------------------------------
if not os.path.isdir(REPO):
    sh(['git', 'clone', '-b', BRANCH,
        'https://github.com/SUHIBAMRO/OMAR.git', REPO], check=True)
else:
    sh(['git', '-C', REPO, 'fetch', '--quiet', 'origin', BRANCH], check=True)
    sh(['git', '-C', REPO, 'reset', '--hard', '--quiet', f'origin/{BRANCH}'],
       check=True)
sha = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
print('code at:', sha)

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)

assert os.path.isdir(OUT), (
    f'{OUT} does not exist.\nCheck the directory name for this case on Drive '
    'before going further.')
import glob
caches = sorted(glob.glob(f'{OUT}/samples_cache*.pt'))
print('\ncaches found:', [os.path.basename(c) for c in caches] or 'NONE')
assert caches, (
    'no sample cache in this directory, so there is nothing to diagnose. '
    'This case needs its FEM data generated first.')

# ---- 1. diagnose, writing nothing -----------------------------------
print('\n' + '=' * 68)
print('1. DRY RUN -- reading the cache, writing nothing')
print('=' * 68)
r = subprocess.run([sys.executable, '-m', 'omar_pfem.repair_b2_sample_cache',
                    '--out_dir', OUT, '--dry_run'],
                   capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise SystemExit('the dry run failed -- send this output to Claude before '
                     'changing anything')

m = re.search(r'overstatement removed: ([\d.]+)x to ([\d.]+)x', r.stdout)
assert m, ('could not read the overstatement line out of the dry run. Send '
           'the output to Claude rather than guessing.')
lo, hi = float(m.group(1)), float(m.group(2))
mesh_ok = 'PASS' in r.stdout

print('\n' + '=' * 68)
print(f'overstatement in this cache: {lo:.2f}x to {hi:.2f}x')
print(f'mesh-independence of the CORRECTED load: '
      f'{"PASS" if mesh_ok else "FAIL"}')
print('=' * 68)

# ---- 2. branch on what was found ------------------------------------
NEEDS_REPAIR = hi > 1.01

if not NEEDS_REPAIR:
    print("""
STOP HERE. The cache is already correct -- the recomputed load matches
what is stored, to within a percent, at every sample.

That matters, because it means the 8.09 relative error in this case's
eval report does NOT come from the load bug that hit the other two B2
cases. Retraining now would be treating the wrong cause.

Nothing was written and no model was deleted.

Send this output to Claude. The next thing to look at is whether that
eval report predates the B2 force fix recorded in PROJECT_STATUS, in
which case the model is stale rather than wrong, and a re-eval alone
settles it.""")
else:
    print(f"""
Same bug as the other two B2 cases: the stored load is up to {hi:.1f}x
too large. Repairing now.
""")
    r2 = subprocess.run([sys.executable, '-m',
                         'omar_pfem.repair_b2_sample_cache', '--out_dir', OUT],
                        capture_output=True, text=True)
    print(r2.stdout[-6000:])
    if r2.returncode != 0:
        print(r2.stderr[-3000:])
        raise SystemExit('the repair failed -- send this to Claude')
    assert 'PASS' in r2.stdout, (
        'the repair ran but the mesh-independence check did NOT pass. Do not '
        'retrain on this cache; send the output to Claude.')

    # Only now is it right to remove the models: they were fitted to the
    # load this cache no longer holds. Training resumes from
    # train_state_latest.pt, so leaving it would rebuild on that state.
    print('\n' + '=' * 68)
    print('deleting the models trained on the wrong load')
    print('=' * 68)
    removed = []
    for f in ('model_best.pt', 'model_final.pt', 'train_state_latest.pt',
              'metrics_history.json', 'EARLY_STOPPED'):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            os.remove(p)
            removed.append(f)
    print('removed:', ', '.join(removed) if removed else 'nothing was there')
    print("""
The stale eval report is deliberately NOT deleted -- the eval stage
fingerprints its checkpoint and refuses rows from a different model, so
it cannot silently mix old and new. It will be overwritten when the
re-eval runs.

Done. Next: retrain this case, then re-evaluate it.""")
