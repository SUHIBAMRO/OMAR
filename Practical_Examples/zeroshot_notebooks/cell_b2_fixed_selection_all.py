# =====================================================================
#  THE OTHER TWO B2 CASES, WITH THE SELECTION FIXED
#
#  WHY. Mooney-Rivlin and Arruda-Boyce were trained by the same trainer,
#  with the same defect, and stopped the same way:
#
#      case              best epoch   stopped at   best val (old metric)
#      neo_hookean            25          225            0.9986
#      mooney_rivlin          25          225            0.9752
#      arruda_boyce          225          425            1.0267
#
#  Every one of them peaked at its FIRST or second validation event. That
#  is not three materials failing; it is one early-stopping criterion
#  firing three times. On B2 the per-component metric
#  `0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v))` RISES while the model
#  improves, so patience ran out immediately.
#
#  Neo-Hookean has now been rerun with `--selection_metric
#  both_components` and reached **0.0598** per_component at epoch 950,
#  against its old 0.9986 and against B1 x Neo-Hookean's 0.0657 on the
#  identical metric. These two are the same experiment on the other two
#  materials.
#
#  ⚠️ RUN THIS ONLY AFTER THE NEO-HOOKEAN RUN HAS FINISHED and its
#  zero-shot eval is in. If that one turns out not to hold up, these two
#  are seven hours of GPU spent on a premise that did not survive. The
#  cell checks for the finished Neo-Hookean eval and refuses to start
#  without it, rather than trusting anyone to remember.
#
#  COST, from the measured rate on this exact configuration: 3.387
#  s/epoch for two resolutions at 800 samples an epoch, so 4,000 epochs
#  is about **3 h 46 m per case** and both together about **7 h 32 m** on
#  an A100 -- less where early stopping fires on the metric that now
#  orders correctly. Resumable at every validation event; re-running this
#  cell after a disconnect continues from the last one, and a case whose
#  eval is already complete is skipped.
#
#  NOTHING IS WRITTEN TO THE EXISTING RUNS. Each case gets a new
#  directory and the two caches are COPIED in, so the sample cache (7+
#  hours of FEM apiece) and the N=101 fine references are never
#  regenerated.
# =====================================================================
import json
import os
import shutil
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'
R = '/content/drive/MyDrive/pfem_run'

CASES = ['mooney_rivlin', 'arruda_boyce']
EPOCHS = 4000
TEST_RES = '13,17,25,29,37,41,49'
FINE_N = '101'
N_EVAL = '20'
# what each case scored under the broken selection, for the before/after
OLD_VAL = {'mooney_rivlin': 0.9752, 'arruda_boyce': 1.0267}
# the gate: Neo-Hookean's fixed-selection eval must exist and be complete
GATE = f'{R}/zeroshot_B2_neo_hookean_fixedsel/zeroshot_eval.json'

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

WANT = [int(x) for x in TEST_RES.split(',')]


def rows_present(path):
    """Which resolutions a report file holds -- not whether it exists."""
    if not os.path.exists(path):
        return []
    try:
        return [r['N'] for r in json.load(open(path)).get('rows', [])]
    except Exception as e:
        print(f'  {path} is unreadable ({e.__class__.__name__})')
        return []


# ---- the gate ---------------------------------------------------------
print('\n' + '=' * 78)
print('GATE: has the Neo-Hookean fixed-selection run finished?')
print('=' * 78)
gate_rows = rows_present(GATE)
if len(gate_rows) < len(WANT):
    print(f'  {GATE}')
    print(f'  holds {len(gate_rows)}/{len(WANT)} resolutions.')
    raise SystemExit(
        '\nSTOPPING. The Neo-Hookean case is the evidence that the fixed\n'
        'selection works; these two cases are seven and a half hours of GPU\n'
        'spent on that premise. Let it finish, read its eval, and only then\n'
        'run this. Nothing was started and nothing was written.')
g = json.load(open(GATE))
print(f'  complete ({len(gate_rows)}/{len(WANT)} resolutions). Its numbers:')
print(f"  {'N':>6}{'per_component':>16}{'both_components':>18}")
for r in g['rows']:
    c = r.get('mean_combined_rel_L2_vs_fine_reference')
    print(f"  {r['N']:>6}{r['mean_rel_L2_vs_fine_reference']:>16.4e}"
          + (f'{c:>18.4e}' if c is not None else f"{'-':>18}"))
print('\n  Read that against B1: 0.050 to 0.106 per_component on these same')
print('  seven meshes. If the Neo-Hookean row above is nowhere near it, stop')
print('  and say so rather than spending the next seven hours.')

# ---- pre-flight on BOTH cases before starting either ------------------
print('\n' + '=' * 78)
print('PRE-FLIGHT')
print('=' * 78)
PLAN = []
for mat in CASES:
    src = f'{R}/zeroshot_B2_{mat}'
    dst = f'{R}/zeroshot_B2_{mat}_fixedsel'
    cache, fine = f'{src}/samples_cache.pt', f'{src}/fine_ref_cache_N{FINE_N}.pt'
    ok = os.path.exists(cache)
    have = rows_present(f'{dst}/zeroshot_eval.json')
    state = ('COMPLETE, will skip' if len(have) >= len(WANT)
             else 'to run' if not os.path.exists(f'{dst}/metrics_history.json')
             else 'RESUMING')
    print(f'  B2 x {mat:<14} cache {"YES" if ok else "NO "}'
          f'   fine-ref {"YES" if os.path.exists(fine) else "no "}   {state}')
    assert ok, (f'no sample cache at {cache}. Regenerating it is 7+ hours of '
                f'FEM, so this cell refuses to start without it.')
    PLAN.append((mat, src, dst, cache, fine, have))

for mat, src, dst, cache, fine, have in PLAN:
    if len(have) >= len(WANT):
        print(f'\n[{mat}] already complete -- skipping')
        continue

    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f'{dst}/samples_cache.pt'):
        print(f'\n[{mat}] copying the sample cache -> {dst} '
              f'(nothing writes to {src})')
        shutil.copy2(cache, f'{dst}/samples_cache.pt')
    dst_fine = f'{dst}/fine_ref_cache_N{FINE_N}.pt'
    if not os.path.exists(dst_fine):
        if os.path.exists(fine):
            n = len(torch.load(fine, map_location='cpu', weights_only=False))
            print(f'[{mat}] copying the N={FINE_N} fine-reference cache '
                  f'({n} solves)')
            shutil.copy2(fine, dst_fine)
        else:
            print(f'[{mat}] !! no fine-reference cache at {fine}; the eval '
                  f'will solve {N_EVAL} fresh N={FINE_N} problems. There is no '
                  f'measured cost for that here, so it is not estimated -- the '
                  f'eval prints its own progress and resumes per resolution.')

    print('\n' + '=' * 78)
    print(f'[{mat}] training, {EPOCHS} epochs, selecting on both_components')
    print(f'  old best under the broken selection: {OLD_VAL[mat]}')
    print(f'  about 3 h 46 m at the measured 3.387 s/epoch, less if it')
    print(f'  plateaus. Resumable at every validation event.')
    print('=' * 78)
    run([sys.executable, '-u', '-m',
         'omar_pfem.resolution_invariance_zeroshot', 'train',
         '--geometry', 'B2', '--material', mat,
         '--train_resolutions', '21,33',
         '--n_train_per_res', '400', '--n_val_per_res', '100',
         '--epochs', str(EPOCHS), '--validate_every', '50',
         '--batch_size', '8',
         '--early_stop_patience', '15',
         '--selection_metric', 'both_components',
         '--out_dir', dst])

    print('\n' + '=' * 78)
    print(f'[{mat}] zero-shot eval, seven unseen meshes')
    print('=' * 78)
    run([sys.executable, '-u', '-m',
         'omar_pfem.resolution_invariance_zeroshot', 'eval',
         '--geometry', 'B2', '--material', mat,
         '--checkpoint', f'{dst}/model_best.pt',
         '--test_resolutions', TEST_RES, '--fine_N', FINE_N,
         '--n_eval_samples', N_EVAL, '--out_json', f'{dst}/zeroshot_eval.json'])

# ---- all three B2 cases, before and after ----------------------------
print('\n' + '=' * 78)
print('ALL THREE B2 CASES, BROKEN SELECTION AGAINST FIXED')
print('=' * 78)
print(f"  {'case':<16}{'old val':>10}{'new val':>10}"
      f"{'new eval per_comp':>20}{'new eval both':>16}")
for mat, old in [('neo_hookean', 0.9986)] + list(OLD_VAL.items()):
    d = f'{R}/zeroshot_B2_{mat}_fixedsel'
    h, ev = f'{d}/metrics_history.json', f'{d}/zeroshot_eval.json'
    new_val = '-'
    if os.path.exists(h):
        hist = json.load(open(h))
        key = ('both_components_val_error' if 'both_components_val_error'
               in hist[0] else 'combined_val_error')
        b = min(hist, key=lambda e: e[key])
        new_val = f"{b['combined_val_error']:.4f}"
    pc = bc = '-'
    if os.path.exists(ev):
        rows = json.load(open(ev)).get('rows', [])
        if rows:
            pc = (f"{min(r['mean_rel_L2_vs_fine_reference'] for r in rows):.4f}"
                  f"-{max(r['mean_rel_L2_vs_fine_reference'] for r in rows):.4f}")
            cs = [r.get('mean_combined_rel_L2_vs_fine_reference') for r in rows]
            cs = [c for c in cs if c is not None]
            if cs:
                bc = f'{min(cs):.4f}-{max(cs):.4f}'
    print(f'  {mat:<16}{old:>10.4f}{new_val:>10}{pc:>20}{bc:>16}')

print('\n  B1, the same seven meshes: 0.050 to 0.106 per_component.')
print('  B2 x Neo-Hookean under the broken selection: 0.871 to 0.873.')
print('=' * 78)
print('Nothing was written to any existing run directory.')
