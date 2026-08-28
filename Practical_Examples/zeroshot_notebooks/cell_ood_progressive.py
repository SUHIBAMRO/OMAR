# =====================================================================
#  CELL — Progressive OOD shift (Timon round-6, point 1)
#  Self-contained: mounts Drive, clones/updates the repo, runs the sweep.
#  Resumable: every (factor, shift) cell is written to Drive as it finishes
#  and skipped on a re-run, so a disconnect costs at most one cell.
#  Run it again after any disconnect — it picks up where it stopped.
# =====================================================================
import os, subprocess, sys

def run(cmd):
    """Stream a child process's output into the notebook.

    subprocess.run() writes to the OS-level stdout descriptor, which Colab
    does not capture -- the child's output simply vanishes and a long run
    looks identical to a hung one. Popen with a line loop, plus `python -u`
    so the child does not buffer, puts it back.
    """
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

# --- pick the case here -----------------------------------------------
GEOMETRY = 'B1'
MATERIAL = 'neo_hookean'
CKPT = f'{R}/results/B1_neo_hookean/model_best.pt'   # Table 5's own checkpoint
# For B2 use the corrected loss-normalized runs instead, e.g.
#   f'{R}/B2_accuracy_search/lossnorm/train/model_best.pt'
# ----------------------------------------------------------------------

assert os.path.exists(CKPT), f'checkpoint not found: {CKPT}'
print('checkpoint:', CKPT, f'({os.path.getsize(CKPT)/1e6:.1f} MB)')

# 3 factors x 6 non-zero shifts + 1 shared baseline = 19 cells.
# Each cell is n_samples CPU FEM solves at N=21, roughly 9-25 s each
# depending on the runtime, so budget about 1.5-4 h for n_samples=10.
run([
    sys.executable, '-u', '-m', 'omar_pfem.ood_progressive',
    '--geometry', GEOMETRY, '--material', MATERIAL,
    '--checkpoint', CKPT,
    '--N', '21',
    '--shifts', '0,0.5,1.0,1.5,2.0,2.5,3.0',
    '--factors', 'material,loading,both',
    '--n_samples', '10',
    '--out_json', f'{OUT}/ood_progressive_{GEOMETRY}_{MATERIAL}.json',
])

print('\nDone. Result:', f'{OUT}/ood_progressive_{GEOMETRY}_{MATERIAL}.json')
