# =====================================================================
#  RESUME cell -- run this in a NEW cell in the SAME Colab kernel right
#  after cell_build_amgx.py's build succeeded (libamgxsh.so already
#  exists on disk at /content/AMGX/build/libamgxsh.so -- no need to
#  rebuild). This finishes the setup that assumed torchfem was already
#  installed (a real oversight: this notebook is standalone and does
#  not share a kernel with B8_GPU_MeshConvergence.ipynb by default) and
#  then runs the exact 105,456-element case that failed with Jacobi.
# =====================================================================
import os
import subprocess
import sys

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'


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
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod in list(sys.modules):
    if _mod == 'torchfem' or _mod.startswith('torchfem.') or _mod == 'omar_pfem' or _mod.startswith('omar_pfem.'):
        del sys.modules[_mod]

AMGX_SO = '/content/AMGX/build/libamgxsh.so'
assert os.path.isfile(AMGX_SO), f'{AMGX_SO} not found -- the build cell may not have finished'
os.environ['AMGX_DLL'] = AMGX_SO

from torchfem.sparse import available_backends
print(f'available_backends: {available_backends}')
assert 'amgx' in available_backends, (
    f'AMGX_DLL is set to {AMGX_SO} but torchfem still does not report "amgx" -- '
    f'available_backends={available_backends}')

print('\nAmgX confirmed available. Testing the exact failing case (53,27) = '
      '105,456 elements on CUDA with preconditioner="amg" -- same geometry, '
      'materials, region, precision, load path, and tolerances as the run '
      'that failed with Jacobi.')

import time
from omar_pfem.data.mesh_convergence_B8 import solve_case

device = torch.device('cuda')
t0 = time.time()
r = solve_case(53, 27, device=device, preconditioner='amg', verbose=True)
elapsed = time.time() - t0
print(f'\nAMG/CUDA/AmgX result: n_elements={r["n_elements"]:,}  total_time={elapsed:.2f}s')
print(f'force_rel_residual={r["force_rel_residual"]:.2e}')
print(f'region_avg_sigma_xx={r["region_avg_sigma_xx"]:.4f} '
      f'(true_max={r["region_true_max_sigma_xx"]:.4f}, diagnostic only)')
