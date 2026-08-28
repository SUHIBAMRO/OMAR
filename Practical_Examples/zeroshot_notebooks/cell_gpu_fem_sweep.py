# =====================================================================
#  CELL — GPU-native FEM scaling sweep + cost breakdown
#  (Timon round-5 point 8, extended by round-6: smaller intermediate sizes
#   and an assembly-versus-solver cost breakdown.)
#
#  NEEDS A GPU RUNTIME. Runtime > Change runtime type > T4/A100.
#  Resumable: each resolution is appended to the output JSON as it finishes
#  and skipped on a re-run, and each solve checkpoints internally, so a
#  disconnect costs at most one in-progress solve. Just run the cell again.
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

import torch
assert torch.cuda.is_available(), \
    'No GPU. Runtime > Change runtime type > GPU, then re-run this cell.'
print('GPU:', torch.cuda.get_device_name(0))

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
OUT = f'{R}/gpu_fem_scaling'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/checkpoints', exist_ok=True)

GEOMETRY = 'B1'
MATERIAL = 'neo_hookean'

# 0.02M -> 3.93M DOF. The four small ones take minutes between them and are
# what make the us/DOF trend a curve instead of four points at one end; the
# last two are hours each. Drop 1401 from the list if time is short -- the
# JSON keeps whatever finished.
run([
    sys.executable, '-u', '-m', 'omar_pfem.gpu_fem_scaling_sweep',
    '--geometry', GEOMETRY, '--material', MATERIAL,
    '--resolutions', '101,201,301,401,501,701,1001,1401',
    '--out_json', f'{OUT}/gpu_fem_scaling_{GEOMETRY}_{MATERIAL}.json',
    '--checkpoint_dir', f'{OUT}/checkpoints',
])

print('\nDone. Result:', f'{OUT}/gpu_fem_scaling_{GEOMETRY}_{MATERIAL}.json')
print('Each row now carries cost_breakdown_pct (residual / preconditioner / CG)')
print('and accounted_frac_of_solve — see the note in matrix_free_solver.py on')
print('why CG time is not cleanly separable from assembly in a matrix-free solver.')
