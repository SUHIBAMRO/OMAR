# =====================================================================
#  CELL — DD-NO trained on coarse vs. fine FEM labels (Timon round-8,
#  point 3, sharpened): if a data-driven operator learns from FEM data
#  generated on a coarse mesh vs. a finer one, how does its accuracy and
#  zero-shot generalization across resolutions change? This targets an
#  asymmetry the physics-informed operator cannot have (it never trains
#  on FEM labels at all), unlike the existing single-resolution
#  data-driven comparison of Table 21/21a.
#
#  B1 x Neo-Hookean, matching every other point7b/Table-21 comparison in
#  this report. Two label-generation resolutions:
#    COARSE_N = 13  (169 nodes)
#    FINE_N   = 33  (1,089 nodes) -- one of the physics-informed model's
#                     own two joint-training resolutions, so its Table 12
#                     zero-shot numbers are a natural reference point
#  Both DD-NO checkpoints are then evaluated zero-shot at the SAME seven
#  test resolutions Table 12 uses (13, 17, 25, 29, 37, 41, 49) against
#  the same N=101 reference, so the result sits directly next to Table
#  12's physics-informed numbers for B1 x Neo-Hookean.
#
#  COST IS NOT MEASURED AT THESE RESOLUTIONS -- do not trust a guessed
#  runtime. Table 4a only measured native FEM cost at N=21 (25.4 s/
#  sample); N=13 should be cheaper and N=33 more expensive, but by how
#  much has not been measured here. Stage 0 below generates a small
#  calibration batch (20 samples) at each resolution FIRST and prints
#  the actual measured per-sample cost, so the full-size run in Stage 1
#  can be sized from real numbers instead of a guess. Read that output
#  before letting Stage 1 run unattended.
#
#  Sample count and optimizer budget MATCH Table 21 exactly: 800 train /
#  200 test, 75,000 optimizer steps. This is deliberate, not a leftover
#  default -- both COARSE and FINE must be trained under the SAME budget
#  Table 21 uses. Giving fine a bigger budget than coarse (e.g. running
#  fine at 800/75,000 while leaving coarse at a smaller pilot budget)
#  would confound label-resolution with training budget: any accuracy
#  gap could then come from fine simply being trained longer/on more
#  data, not from its finer label mesh. Both resolutions here get the
#  identical 800/200/75,000 budget, so the only thing that differs
#  between the two runs is COARSE_N vs. FINE_N.
#
#  Self-contained: mounts Drive, clones/updates the repo. Resumable:
#  every stage (generate/convert/train/eval, per resolution) is skipped
#  if its output already exists on Drive -- and every stage past
#  calibration is now keyed by the budget itself (see BUDGET_TAG below),
#  not just by resolution, so a directory left over from an earlier run
#  at a DIFFERENT N_TRAIN/N_TEST/OPT_STEPS can never be silently reused
#  as if it were this run. (This is not hypothetical: an earlier run of
#  this exact cell, before the budget was raised to 800/200/75,000,
#  left pilot-scale 200/50/20,000 directories on Drive under the old
#  unkeyed paths -- re-running the cell after the fix printed "[skip]
#  already done" for every stage and silently reused those pilot
#  results instead of training at the new budget. If you have Drive
#  output from before this fix, it sits under the OLD paths without a
#  budget tag and is simply ignored now -- safe to delete once you no
#  longer need it, but harmless to leave in place either way.)
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
      else 'NONE - prefer Runtime > Change runtime type > GPU for training')

R = '/content/drive/MyDrive/pfem_run'
OUT = f'{R}/dd_no_coarse_vs_fine'
os.makedirs(OUT, exist_ok=True)

GEOMETRY = 'B1'
MATERIAL = 'neo_hookean'
RESOLUTIONS = {'coarse': 13, 'fine': 33}
CALIBRATION_SAMPLES = 20
N_TRAIN = 800   # matches Table 21 exactly, both coarse AND fine
N_TEST = 200    # matches Table 21 exactly, both coarse AND fine
OPT_STEPS = 75_000   # matches Table 7/21's own budget, both coarse AND
                     # fine -- see the module docstring above for why
                     # this must be identical across the two resolutions
BATCH = 8
TEST_RESOLUTIONS = '13,17,25,29,37,41,49'   # Table 12's own seven
FINE_REF_N = 101

# Every stage past calibration is keyed by this budget tag, NOT just by
# tag/N. A resumability check that only asked "does this directory
# exist?" would silently reuse a stale directory from an earlier run at
# a DIFFERENT budget (e.g. an old 200/50/20,000 pilot run sharing the
# same {tag}_N{N}_raw path as this 800/200/75,000 run) and report
# "[skip] already done" without ever training at the intended budget.
# That happened once already on this project's Drive -- see
# PROJECT_STATUS.md. Baking the budget into the path makes a changed
# budget always start a fresh directory instead of silently adopting
# whatever happens to already sit at the old path.
BUDGET_TAG = f'{N_TRAIN}tr_{N_TEST}te_{OPT_STEPS}steps'


def stage0_calibrate(tag, N):
    calib_dir = f'{OUT}/{tag}_N{N}_calib'
    marker = f'{calib_dir}/DONE'
    if os.path.exists(marker):
        print(f'[skip] {tag} (N={N}) calibration already done: {calib_dir}')
        return
    os.makedirs(calib_dir, exist_ok=True)
    t0 = time.time()
    run([
        sys.executable, '-u', '-m', 'omar_pfem.data.data_generate_B1',
        '--num_samples', str(CALIBRATION_SAMPLES),
        '--Nx', str(N), '--Ny', str(N),
        '--material', MATERIAL,
        '--out_dir', calib_dir,
        '--n_workers', '4',
    ])
    elapsed = time.time() - t0
    per_sample = elapsed / CALIBRATION_SAMPLES
    print(f'\n{tag} (N={N}): {elapsed:.1f} s for {CALIBRATION_SAMPLES} '
          f'samples = {per_sample:.2f} s/sample (measured, not guessed).')
    print(f'Full run at N_TRAIN+N_TEST={N_TRAIN+N_TEST} samples would take '
          f'roughly {(N_TRAIN+N_TEST)*per_sample/60:.1f} minutes at this rate.')
    open(marker, 'w').write(f'{per_sample}\n')


def stage1_generate(tag, N):
    raw_dir = f'{OUT}/{tag}_N{N}_{BUDGET_TAG}_raw'
    if os.path.exists(f'{raw_dir}/split_indices.json'):
        print(f'[skip] {tag} (N={N}) full dataset already generated: {raw_dir}')
        return raw_dir
    run([
        sys.executable, '-u', '-m', 'omar_pfem.data.data_generate_B1',
        '--num_samples', str(N_TRAIN + N_TEST),
        '--Nx', str(N), '--Ny', str(N),
        '--material', MATERIAL,
        '--out_dir', raw_dir,
        '--n_workers', '4',
    ])
    return raw_dir


def stage2_convert(tag, N, raw_dir):
    q4_dir = f'{OUT}/{tag}_N{N}_{BUDGET_TAG}_q4'
    npz = f'{q4_dir}/hyperelastic_training_data_q4.npz'
    if os.path.exists(npz):
        print(f'[skip] {tag} (N={N}) already converted: {npz}')
        return npz
    run([
        sys.executable, '-u', '-m', 'omar_pfem.data.convert_B1_quad',
        '--h5_dir', raw_dir, '--out_dir', q4_dir,
    ])
    assert os.path.exists(npz), f'conversion did not produce {npz}'
    return npz


def stage3_train(tag, N, npz):
    train_dir = f'{OUT}/{tag}_N{N}_{BUDGET_TAG}_train'
    ckpt = f'{train_dir}/model_best.pt'
    if os.path.exists(ckpt):
        print(f'[skip] {tag} (N={N}) already trained: {ckpt}')
        return ckpt
    run([
        sys.executable, '-u', '-m', 'omar_pfem.train_data_driven',
        '--geometry', GEOMETRY, '--material', MATERIAL,
        '--path', npz,
        '--ntrain', str(N_TRAIN), '--ntest', str(N_TEST),
        '--batch_size', str(BATCH),
        '--opt_steps', str(OPT_STEPS),
        '--loss', 'rel_l2',
        '--eval_every', '2000',
        '--out_dir', train_dir,
    ])
    assert os.path.exists(ckpt), f'training did not produce {ckpt}'
    return ckpt


def stage4_zeroshot_eval(tag, N, ckpt):
    out_json = f'{OUT}/{tag}_N{N}_{BUDGET_TAG}_zeroshot.json'
    if os.path.exists(out_json):
        print(f'[skip] {tag} (N={N}) already evaluated: {out_json}')
        return out_json
    run([
        sys.executable, '-u', '-m', 'omar_pfem.resolution_invariance_zeroshot',
        'eval',
        '--geometry', GEOMETRY, '--material', MATERIAL,
        '--checkpoint', ckpt,
        '--test_resolutions', TEST_RESOLUTIONS,
        '--fine_N', str(FINE_REF_N),
        '--n_eval_samples', '20',
        '--out_json', out_json,
    ])
    return out_json


for tag, N in RESOLUTIONS.items():
    print(f'\n{"="*70}\n{tag.upper()} (N={N})\n{"="*70}')
    stage0_calibrate(tag, N)
    raw_dir = stage1_generate(tag, N)
    npz = stage2_convert(tag, N, raw_dir)
    ckpt = stage3_train(tag, N, npz)
    result = stage4_zeroshot_eval(tag, N, ckpt)
    print(f'{tag} (N={N}) done: {result}')

print(f'\nAll done. Compare coarse_N13_{BUDGET_TAG}_zeroshot.json against '
      f'fine_N33_{BUDGET_TAG}_zeroshot.json, and both against Table 12\'s '
      f'physics-informed B1 x Neo-Hookean column, at the same seven test '
      f'resolutions.')
