# =====================================================================
#  CELL — Progressive OOD shift for the five remaining cases
#  (Timon round-8, point 2: "OOD is well addressed and for the paper we
#  should do that for the other cases." B1 x Neo-Hookean is already done
#  -- see ood_progressive_B1_neo_hookean.json / Tables 19, 19a.)
#
#  Same protocol as cell_ood_progressive.py (which did B1 x Neo-Hookean),
#  run in sequence for the other five geometry x material combinations,
#  using each case's own Table-5/7/11 checkpoint (NOT the zero-shot
#  fixed-selection checkpoints Table 12/18 use -- those are a different,
#  jointly-trained-at-two-resolutions model family).
#
#  Self-contained: mounts Drive, clones/updates the repo, runs each case.
#  Resumable at TWO levels: this cell skips a whole case if its output
#  JSON already exists on Drive, and within a case the underlying
#  omar_pfem.ood_progressive module writes each (factor, shift) cell as
#  it finishes and skips it on a re-run -- a disconnect anywhere costs at
#  most one cell, not the whole case.
#
#  Cost: cell_ood_progressive.py budgeted 1.5-4 h for one case
#  (n_samples=10, 19 cells of CPU FEM solves at N=21). Five cases is
#  therefore roughly 7.5-20 h of wall-clock if run start to finish in one
#  sitting -- expect to need several Colab sessions; that is exactly what
#  the resumability above is for.
# =====================================================================
import os, subprocess, sys

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

R = '/content/drive/MyDrive/pfem_run'
OUT = f'{R}/ood_progressive'
os.makedirs(OUT, exist_ok=True)

# Each case's own Table-5/7/11 checkpoint -- verified against
# point5_results/physical_quantities_B2_*.json's own "checkpoint"
# provenance field, which the report says shares checkpoints with
# Tables 5, 7 and 11.
CASES = [
    ('B1', 'mooney_rivlin', f'{R}/results/B1_mooney_rivlin/model_best.pt'),
    ('B1', 'arruda_boyce', f'{R}/results/B1_arruda_boyce/model_best.pt'),
    ('B2', 'neo_hookean',
     f'{R}/B2_accuracy_search/lossnorm/train/model_best.pt'),
    ('B2', 'mooney_rivlin',
     f'{R}/B2_accuracy_search_mooney_rivlin/lossnorm/train/model_best.pt'),
    ('B2', 'arruda_boyce',
     f'{R}/B2_accuracy_search_arruda_boyce/lossnorm/train/model_best.pt'),
]

for geometry, material, ckpt in CASES:
    out_json = f'{OUT}/ood_progressive_{geometry}_{material}.json'
    if os.path.exists(out_json):
        print(f'\n[skip] {geometry} x {material}: {out_json} already exists')
        continue
    if not os.path.exists(ckpt):
        print(f'\n[skip] {geometry} x {material}: checkpoint not found at '
              f'{ckpt} -- fix the path above and re-run')
        continue
    print(f'\n=== {geometry} x {material} ===')
    print('checkpoint:', ckpt, f'({os.path.getsize(ckpt)/1e6:.1f} MB)')
    run([
        sys.executable, '-u', '-m', 'omar_pfem.ood_progressive',
        '--geometry', geometry, '--material', material,
        '--checkpoint', ckpt,
        '--N', '21',
        '--shifts', '0,0.5,1.0,1.5,2.0,2.5,3.0',
        '--factors', 'material,loading,both',
        '--n_samples', '10',
        '--out_json', out_json,
    ])
    print('Done:', out_json)

print('\nAll cases attempted. Re-run this cell if any were skipped due to '
      'a missing checkpoint once the path is fixed, or to resume after a '
      'disconnect.')
