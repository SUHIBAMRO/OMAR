# =====================================================================
#  STEPS 2+3 — Retrain and re-evaluate the three B2 zero-shot cases
#
#  Why all three need it, established 2026-08-29:
#
#    B2 x Mooney-Rivlin, B2 x Arruda-Boyce -- their caches carried a
#      mesh-dependent load overstatement (13x at N=21, 21x at N=33). The
#      caches were repaired and the models trained on the bad load were
#      deleted.
#
#    B2 x Neo-Hookean -- its cache turned out to be ALREADY correct, but
#      its directory holds no model at all. The file times tell the whole
#      story: the eval report is stamped 19:47:50 and the cache 19:48:14
#      on 2026-08-27, twenty-four seconds later. So an earlier repair ran
#      right after that eval, fixed the cache, and deleted the model. Its
#      8.09 error was a model trained on the bad load being scored against
#      freshly built correct references -- which is exactly the mismatch
#      that produces a large error flat in N. No mystery left, and the fix
#      is the same as the other two.
#
#  THE EVAL IS MUCH CHEAPER THAN IT LOOKS. The B1 evals took about eight
#  hours each, almost all of it solving twenty N=101 references. Those
#  references are cached per case in fine_ref_cache_N101.pt, and they are
#  NOT affected by the load bug: the eval builds each fine sample fresh
#  and the FEM solver assembles its own consistent force internally, so a
#  cached reference is as valid now as it was. Where the cache is present
#  and complete, the eval reduces to the operator's own inference.
#
#  This cell prints, per case, whether that cache is there BEFORE running
#  anything, so the real cost is visible up front rather than discovered
#  eight hours in.
#
#  GPU needed. Resumable at every stage; a finished stage is skipped.
# =====================================================================
import json
import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'
CASES = ['neo_hookean', 'mooney_rivlin', 'arruda_boyce']
TEST_RES = '13,17,25,29,37,41,49'      # the same list the three B1 cases use
EVAL_JSON = 'zeroshot_eval_coarse_and_fine.json'

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

assert torch.cuda.is_available(), 'no GPU -- Runtime -> Change runtime type'
print('GPU:', torch.cuda.get_device_name(0))

# ---- pre-flight on all three, before starting any -------------------
print('\n' + '=' * 78)
print('PRE-FLIGHT')
print('=' * 78)
PLAN = []
for mat in CASES:
    d = f'{R}/zeroshot_B2_{mat}'
    assert os.path.isdir(d), f'missing directory: {d}'
    cache = os.path.join(d, 'samples_cache.pt')
    assert os.path.exists(cache), (
        f'{mat}: no samples_cache.pt. This case still needs its FEM data '
        f'generated, which is hours, and this cell does not do that.')
    ckpt = os.path.join(d, 'model_best.pt')
    fine = os.path.join(d, f'fine_ref_cache_N101.pt')
    out = os.path.join(d, EVAL_JSON)
    n_fine = None
    if os.path.exists(fine):
        try:
            n_fine = len(torch.load(fine, weights_only=False))
        except Exception as e:
            n_fine = f'unreadable ({e.__class__.__name__})'
    PLAN.append(dict(mat=mat, d=d, ckpt=ckpt, out=out, n_fine=n_fine,
                     trained=os.path.exists(ckpt), evaled=os.path.exists(out)))
    print(f'  {mat:<14} cache {os.path.getsize(cache) / 1e6:7.1f} MB   '
          f'model {"YES" if os.path.exists(ckpt) else "no "}   '
          f'eval {"YES" if os.path.exists(out) else "no "}   '
          f'fine refs cached: {n_fine if n_fine is not None else "NONE"}')

print("""
Reading the last column: 20 cached fine references means the eval stage
is cheap. NONE means it must solve twenty N=101 problems first, which is
where the eight hours went the first time.""")

# The stale eval reports are renamed, not deleted -- they are evidence of
# what the bad load did, and leaving them under their usual name is how
# someone quotes 8.09 as a result six months from now.
print('\n' + '=' * 78)
print('SETTING THE STALE EVAL REPORTS ASIDE')
print('=' * 78)
for p in PLAN:
    stale = os.path.join(p['d'], 'zeroshot_eval_report.json')
    if os.path.exists(stale):
        newname = os.path.join(
            p['d'], 'zeroshot_eval_report.STALE_model_trained_on_bad_load.json')
        os.replace(stale, newname)
        print(f'  {p["mat"]:<14} renamed -> {os.path.basename(newname)}')
    else:
        print(f'  {p["mat"]:<14} nothing to rename')

# ---- stage 1: train --------------------------------------------------
for p in PLAN:
    if p['trained']:
        print(f'\n[{p["mat"]}] model_best.pt already present -- skipping '
              f'training. Delete it to force a retrain.')
        continue
    print('\n' + '=' * 78)
    print(f'[{p["mat"]}] TRAINING -- roughly 45 min to 1 h 40 m depending on '
          f'the GPU')
    print('=' * 78)
    # Exactly the protocol the three B1 cases used, so B2 rows are
    # comparable to B1 rows. The sample cache is already on disk, so no
    # FEM generation happens here.
    run([sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot',
         'train', '--geometry', 'B2', '--material', p['mat'],
         '--train_resolutions', '21,33',
         '--n_train_per_res', '400', '--n_val_per_res', '100',
         '--epochs', '2000', '--validate_every', '25', '--batch_size', '8',
         '--out_dir', p['d']])
    assert os.path.exists(p['ckpt']), (
        f'{p["mat"]}: training finished but produced no model_best.pt')

# ---- stage 2: evaluate ----------------------------------------------
for p in PLAN:
    if p['evaled']:
        print(f'\n[{p["mat"]}] {EVAL_JSON} already present -- skipping eval.')
        continue
    print('\n' + '=' * 78)
    print(f'[{p["mat"]}] EVALUATING on {TEST_RES}')
    print('=' * 78)
    run([sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot',
         'eval', '--geometry', 'B2', '--material', p['mat'],
         '--checkpoint', p['ckpt'],
         '--test_resolutions', TEST_RES,
         '--fine_N', '101', '--n_eval_samples', '20',
         '--out_json', p['out']])

# ---- summary ---------------------------------------------------------
print('\n' + '=' * 78)
print('RESULT -- B2, mean relative L2 against the N=101 reference')
print('=' * 78)
print('%-6s %14s %14s %14s' % ('N', 'Neo-Hookean', 'Mooney-Riv', 'Arruda-B'))
tables = {}
for p in PLAN:
    if os.path.exists(p['out']):
        tables[p['mat']] = {r['N']: r['mean_rel_L2_vs_fine_reference']
                            for r in json.load(open(p['out']))['rows']}
for N in [int(x) for x in TEST_RES.split(',')]:
    cells = []
    for mat in CASES:
        v = tables.get(mat, {}).get(N)
        cells.append('%14.4f' % v if v is not None else '%14s' % '-')
    print('%-6d%s' % (N, ''.join(cells)))

bad = {m: max(t.values()) for m, t in tables.items() if max(t.values()) > 1.0}
print()
if bad:
    print('STILL BROKEN:', bad)
    print('A relative error above 1.0 means the prediction is further from the')
    print('truth than predicting zero would be. Do NOT report these. Send the')
    print('output over -- retraining did not fix it and the cause is elsewhere.')
else:
    print('All finished cases are in a sane range. For comparison the three B1')
    print('cases span 0.050 to 0.106. Send these numbers and the B2 rows go')
    print('into Table 12 beside the B1 ones.')
print('=' * 78)
