# =====================================================================
#  THE OTHER TWO B2 CASES, WITH THE SELECTION FIXED
#
#  WHY. Mooney-Rivlin and Arruda-Boyce were trained by the same trainer,
#  with the same defect, and stopped the same way:
#
#      case              best epoch   of budget   opt steps   best val
#      neo_hookean            25         2000        2,500      0.9986
#      mooney_rivlin          25         2000        2,500      0.9752
#      arruda_boyce          225         2000       22,500      1.0267
#
#  Every one of them peaked at its FIRST or second validation event. That
#  is not three materials failing; it is one early-stopping criterion
#  firing three times. On B2 the per-component metric
#  `0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v))` RISES while the model
#  improves, so patience ran out immediately.
#
#  Neo-Hookean has since been rerun with `--selection_metric
#  both_components`. It needed **275,000 optimiser steps** (epoch 2,750 of
#  4,000, 3 h 18 m) to reach its best. Mooney-Rivlin was given 2,500 of
#  those steps and Arruda-Boyce 22,500 -- 0.9% and 8%. They were not
#  measured; they were stopped. This cell measures them.
#
#  THE NUMBERS ARE NOT TYPED IN HERE. Everything quoted below is read at
#  run time from the committed JSONs -- `B2_zeroshot_fixedselection.json`
#  for the Neo-Hookean rerun, `zeroshot_B1_*.json` for the B1 span -- so a
#  stale pasted copy of this cell cannot print a number the repo has since
#  corrected. A pasted cell body does NOT refresh when the repo does; that
#  has misled this project three times. Run it through
#  `Round6_RUN_THIS.ipynb`, which execs the current file from the clone.
#
#  ⚠️ RUN THIS ONLY AFTER THE NEO-HOOKEAN RUN HAS FINISHED and its
#  zero-shot eval is in. The cell checks for that eval and refuses to
#  start without it, rather than trusting anyone to remember.
#
#  COST, from the measured rate on this exact configuration: 3.387
#  s/epoch for two resolutions at 800 samples an epoch, so 4,000 epochs
#  is about **3 h 46 m per case** and both together about **7 h 32 m** on
#  an A100 -- less where early stopping fires on the metric that now
#  orders correctly (Neo-Hookean stopped itself at 3,500). The eval that
#  follows each is seconds, not hours: its twenty N=101 references are
#  already cached per case, which is why both old evals produced all
#  seven rows. Resumable at every validation event; re-running this cell
#  after a disconnect continues from the last one, and a case whose eval
#  is already complete is skipped.
#
#  NOTHING IS WRITTEN TO THE EXISTING RUNS. Each case gets a new
#  directory and the two caches are COPIED in, so the sample cache (7+
#  hours of FEM apiece) and the N=101 fine references are never
#  regenerated.
#
#  WHAT THIS CELL CANNOT TELL YOU IN ADVANCE. Whether these two land near
#  Neo-Hookean's numbers. One corrected case is one case. The defect is
#  shared and documented, which is a reason to expect improvement and not
#  a measurement of it -- and a prediction made here before the
#  Neo-Hookean rerun ("expect ~0.68") was wrong by a factor of twenty,
#  because it extrapolated from arms that had themselves been cut short
#  by the defect under test. No target is printed below for that reason.
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
# what each case scored under the broken selection, and at which epoch
OLD = {'mooney_rivlin': (0.9752, 25), 'arruda_boyce': (1.0267, 225)}
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
PF = f'{WORK}/omar_pfem'


def rows_present(path):
    """Which resolutions a report file holds -- not whether it exists."""
    if not os.path.exists(path):
        return []
    try:
        return [r['N'] for r in json.load(open(path)).get('rows', [])]
    except Exception as e:
        print(f'  {path} is unreadable ({e.__class__.__name__})')
        return []


# ---- what the repo says about the case that has already been fixed ----
# Read, not typed. If these files are missing the cell still runs; it just
# cannot print the context, and says so instead of inventing it.
NH = None
try:
    NH = json.load(open(f'{PF}/point7a_results/B2_zeroshot_fixedselection.json'))
except Exception as e:
    print(f'[context] B2_zeroshot_fixedselection.json unreadable ({e.__class__.__name__})')
B1_SPAN = None
try:
    vals = [r['mean_rel_L2_vs_fine_reference']
            for m in ('neo_hookean', 'mooney_rivlin', 'arruda_boyce')
            for r in json.load(
                open(f'{PF}/point7a_results/zeroshot_B1_{m}.json'))['rows']]
    B1_SPAN = (min(vals), max(vals))
except Exception as e:
    print(f'[context] the B1 zero-shot JSONs are unreadable ({e.__class__.__name__})')

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

if NH is not None:
    tr = NH['training']
    print(f'\n  Neo-Hookean, from the committed record:')
    print(f'    validation  {tr["superseded_run_best_per_component_val_error"]}'
          f' -> {tr["per_component_val_error_at_that_checkpoint"]:.4f}'
          f' per_component, {tr["best_both_components_val_error"]:.4f} both')
    print(f'    best at epoch {tr["best_epoch"]:,}'
          f' ({tr["opt_steps_at_best"]:,} steps), stopped at'
          f' {tr["final_epoch"]:,}')
    # cross-check the committed record against the file on Drive, since the
    # gate above is the live artefact and this is the transcription of it
    rec = {r['N']: r['mean_rel_L2_vs_fine_reference'] for r in NH['rows']}
    live = {r['N']: r['mean_rel_L2_vs_fine_reference'] for r in g['rows']}
    bad = [N for N in rec if N in live and abs(rec[N] - live[N]) > 5e-5]
    print(f'    the repo record and the file on Drive agree at all'
          f' {len(rec)} meshes' if not bad else
          f'    !! repo record and Drive DISAGREE at N={bad} -- read both'
          f' before trusting either')
if B1_SPAN is not None:
    print(f'\n  B1 on these same seven meshes: {B1_SPAN[0]:.4f} to'
          f' {B1_SPAN[1]:.4f} per_component (three materials).')
print('\n  If the Neo-Hookean row above is nowhere near that, stop and say so')
print('  rather than spending the next seven hours.')

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

    # COPY ALL THE SAMPLE CACHES, NOT JUST THE COMBINED ONE, AND VERIFY.
    #
    # An earlier version of this cell copied `samples_cache.pt` alone. On the
    # first Mooney-Rivlin attempt the trainer then printed "Generating
    # training/validation samples ... real FEM solves" and started re-solving
    # from zero -- `os.path.exists` on the freshly copied 55 MB file returned
    # False in the subprocess, and there was nothing else for it to fall back
    # on. The trainer's own resume path looks for the PER-RESOLUTION caches
    # `samples_cache_N{N}.pt`, which exist in the source directory and were
    # simply never copied. Copying them makes that fallback work.
    #
    # The cost of not doing this is measured, not guessed: the generator's own
    # docstring records 7.3 hours for N=21 alone, and N=33 is larger.
    #
    # Every copy is then checked by size against its source, because a copy
    # that silently did not land is exactly the failure being fixed. The
    # samples themselves are seeded deterministically (10_000*N + i), so a
    # regeneration would be bit-identical -- the loss is hours, never
    # correctness.
    wanted = [('samples_cache.pt', cache)] + [
        (f'samples_cache_N{N}.pt', f'{src}/samples_cache_N{N}.pt')
        for N in (21, 33)]
    for name, s in wanted:
        d = f'{dst}/{name}'
        if not os.path.exists(s):
            print(f'[{mat}] {name}: not in {src} -- nothing to copy')
            continue
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            print(f'[{mat}] {name}: already present, {os.path.getsize(d)/1e6:.1f} MB')
            continue
        print(f'[{mat}] copying {name} ({os.path.getsize(s)/1e6:.1f} MB) '
              f'-> {dst}  (nothing writes to {src})')
        shutil.copy2(s, d)
        got, want = os.path.getsize(d), os.path.getsize(s)
        assert got == want, (
            f'{d} is {got} bytes against the source\'s {want}. The copy did '
            f'not land. Re-run this cell; do NOT let training start, because '
            f'it would silently re-solve the FEM instead of reading this.')
        print(f'[{mat}]   verified {got/1e6:.1f} MB')
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

    old_val, old_epoch = OLD[mat]
    print('\n' + '=' * 78)
    print(f'[{mat}] training, {EPOCHS} epochs, selecting on both_components')
    print(f'  under the broken selection this case peaked at epoch {old_epoch}'
          f' with {old_val}')
    print(f'  about 3 h 46 m at the measured 3.387 s/epoch, less if it')
    print(f'  early-stops. Resumable at every validation event.')
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
print(f"  {'case':<16}{'old val':>10}{'old ep':>8}{'new val':>10}{'new ep':>8}"
      f"{'new eval per_comp':>20}{'new eval both':>16}")
ALL = [('neo_hookean', 0.9986, 25)] + [(m,) + OLD[m] for m in CASES]
for mat, old, old_ep in ALL:
    d = f'{R}/zeroshot_B2_{mat}_fixedsel'
    h, ev = f'{d}/metrics_history.json', f'{d}/zeroshot_eval.json'
    new_val, new_ep = '-', '-'
    if os.path.exists(h):
        try:
            hist = json.load(open(h))
            key = ('both_components_val_error' if 'both_components_val_error'
                   in hist[0] else 'combined_val_error')
            b = min(hist, key=lambda e: e[key])
            new_val = f"{b['combined_val_error']:.4f}"
            new_ep = str(b.get('epoch', '-'))
        except Exception as e:
            new_val = f'({e.__class__.__name__})'
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
    print(f'  {mat:<16}{old:>10.4f}{old_ep:>8}{new_val:>10}{new_ep:>8}'
          f'{pc:>20}{bc:>16}')

if B1_SPAN is not None:
    print(f'\n  B1, the same seven meshes: {B1_SPAN[0]:.4f} to {B1_SPAN[1]:.4f}'
          f' per_component.')
print('  B2 x Neo-Hookean under the broken selection was flat at 0.871-0.873;'
      ' that')
print('  flatness was insensitivity, not invariance -- a model whose output')
print('  barely responds to its input gives nearly the same error on every')
print('  mesh. Read the SPREAD of each new row, not only its minimum.')
print('=' * 78)
print('Nothing was written to any existing run directory.')
print('\nSend the whole block back. Report v38 quotes no B2 number for these')
print('two materials, and section 10 names them as the only outstanding item.')
