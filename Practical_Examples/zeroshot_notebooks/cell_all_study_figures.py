# =====================================================================
#  CELL -- Item #12: field-panel-grid figures for EVERY study that has a
#  saved checkpoint/converged solve on Drive, in the style Omar pointed
#  to from Timon's own group's papers (VINO Figs. 7a/8d/9d; the PFEM/
#  NOWS screenshots he shared) -- one panel per case, arranged in a row,
#  showing the actual field, not a summary number. Tables themselves are
#  left completely untouched, matching those papers' own practice of
#  keeping the numeric table AND adding a companion figure, not
#  replacing one with the other.
#
#  NO RE-SOLVING AND NO RE-TRAINING HAPPEN HERE for any cell that reuses
#  an already-converged checkpoint. Sections A and D load an already-
#  converged/trained checkpoint and just plot it. Sections B and C run a
#  handful of CHEAP small-N forward passes (and, for the OOD cell, small
#  fresh FEM solves at N=21 -- seconds each, not the large multi-million
#  DOF solves elsewhere in this project) through an ALREADY-TRAINED
#  model, reusing that study's own validated pipeline
#  (resolution_invariance_zeroshot.py / ood_progressive.py) directly, so
#  correctness matches the real numbers already in Tables 12/19/25/26.
#
#  Each section is independent -- if one checkpoint path below is wrong
#  for your actual Drive layout, fix that section's own CHECKPOINT
#  constant and re-run just that cell; the others are unaffected.
# =====================================================================
import os, sys, subprocess

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

DRIVE_RUN = '/content/drive/MyDrive/pfem_run'
DRIVE_CKPT = '/content/drive/MyDrive/pfem_ckpt'
OUT_DIR = f'{DRIVE_RUN}/figures'

results = {}

# --- Section A: FEM convergence field grid (Table 6a/20), B1 Neo-Hookean.
# No new solving -- reuses item #4's own converged checkpoints.
print('\n=== A: FEM convergence field grid ===')
from omar_pfem.field_snapshot_grid import make_resolution_grid_figure
try:
    results['A_fem_convergence'] = make_resolution_grid_figure(
        [51, 101, 201, 401, 701, 1001, 1401], DRIVE_CKPT,
        os.path.join(OUT_DIR, 'fig_B1_resolution_grid.png'))
except Exception as e:
    print('SECTION A FAILED:', e)

# --- Section B: zero-shot resolution field grid (Table 12), B1 Neo-Hookean.
# ONE trained checkpoint, evaluated at several resolutions, no retraining.
print('\n=== B: zero-shot resolution field grid ===')
from omar_pfem.study_field_grids import make_zeroshot_resolution_grid
ZEROSHOT_CKPT = f'{DRIVE_RUN}/zeroshot_B1_neo_hookean/model_best.pt'
try:
    results['B_zeroshot'] = make_zeroshot_resolution_grid(
        ZEROSHOT_CKPT, 'B1', 'neo_hookean', [13, 17, 25, 29, 37, 41, 49],
        seed=20_000_000, out_path=os.path.join(OUT_DIR, 'fig_B1_zeroshot_grid.png'))
except Exception as e:
    print('SECTION B FAILED (check ZEROSHOT_CKPT path above):', e)

# --- Section C: OOD shift field grid (Tables 19/25), B2 (all 3 materials
# have a confirmed checkpoint path from their own already-committed
# ood_progressive_*.json files).
print('\n=== C: OOD shift field grids ===')
from omar_pfem.study_field_grids import make_ood_shift_grid
OOD_CHECKPOINTS = {
    'neo_hookean': f'{DRIVE_RUN}/B2_accuracy_search/lossnorm/train/model_best.pt',
    'mooney_rivlin': f'{DRIVE_RUN}/B2_accuracy_search_mooney_rivlin/lossnorm/train/model_best.pt',
    'arruda_boyce': f'{DRIVE_RUN}/B2_accuracy_search_arruda_boyce/lossnorm/train/model_best.pt',
}
for material, ckpt in OOD_CHECKPOINTS.items():
    try:
        results[f'C_ood_{material}'] = make_ood_shift_grid(
            ckpt, 'B2', material, factor='material', shift_sigmas=[0, 1.5, 3.0],
            out_path=os.path.join(OUT_DIR, f'fig_B2_{material}_ood_grid.png'))
    except Exception as e:
        print(f'SECTION C ({material}) FAILED:', e)

# --- Section D: DD-NO coarse-vs-fine field grid (Table 26), B1 Neo-Hookean.
print('\n=== D: DD-NO coarse-vs-fine field grid ===')
from omar_pfem.study_field_grids import make_dd_no_coarse_vs_fine_grid
COARSE_CKPT = f'{DRIVE_RUN}/dd_no_coarse_vs_fine/coarse_N13_800tr_200te_75000steps_train/model_best.pt'
FINE_CKPT = f'{DRIVE_RUN}/dd_no_coarse_vs_fine/fine_N33_800tr_200te_75000steps_train/model_best.pt'
try:
    results['D_dd_no'] = make_dd_no_coarse_vs_fine_grid(
        COARSE_CKPT, FINE_CKPT, [13, 25, 49],
        out_path=os.path.join(OUT_DIR, 'fig_B1_dd_no_coarse_vs_fine_grid.png'))
except Exception as e:
    print('SECTION D FAILED:', e)

print('\n' + '=' * 70)
print('DONE. Figures saved to Drive:')
for k, v in results.items():
    print(f'  {k}: {v}')
print(f'\nAll under {OUT_DIR} -- fetch them from Drive to bring them back.')
print('Any section marked FAILED above needs its checkpoint path checked')
print('against your actual Drive layout before re-running just that section.')
