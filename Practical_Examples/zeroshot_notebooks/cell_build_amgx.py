# =====================================================================
#  EXPERIMENTAL -- build NVIDIA AmgX from source in this Colab runtime,
#  so torch-fem's CUDA AMG preconditioner path becomes available, then
#  retest the exact 105,456-element B8-final case that failed with
#  Jacobi (real ConvergenceError: CG did not reach 1e-08 within the
#  iteration limit; Newton-Raphson did not converge in increment 4
#  after 10 cutbacks).
#
#  AmgX ships no pip wheel -- this is a real from-source build, not
#  guaranteed to succeed on the first try. This cell reports the real
#  output of every step so a failure can be diagnosed from what it
#  actually prints, not guessed at. Expected build time: on the order
#  of 15-40 minutes (compiling a large CUDA/C++ codebase), separate
#  from and in addition to however long the actual solve takes
#  afterward.
#
#  Standalone and idempotent -- does its own repo clone/pip install (no
#  dependency on another notebook having run first in the same
#  kernel), pins pyvista<0.49 (a live, external PyPI break: 0.49+
#  unconditionally imports IPython.core.guarded_eval, which only
#  exists from IPython>=8.8, but Colab ships 7.34) and clears any
#  already-cached pyvista/torchfem entries from sys.modules (a failed
#  import earlier in the SAME kernel leaves a broken cached module that
#  a fresh pip install on disk does not retroactively fix). If a
#  previous run already built libamgxsh.so on this VM's disk, this
#  skips straight to using it instead of rebuilding -- safe to just
#  re-run this same cell after a Runtime > Restart or a mid-way
#  failure. Requires a GPU runtime (checks torch.cuda.is_available()).
# =====================================================================
import os
import subprocess
import sys

import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'

from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'


def _run_setup(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


if not os.path.isdir(REPO):
    _run_setup(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
                'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    _run_setup(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    _run_setup(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    _run_setup(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

# Real, external, currently-active PyPI issue (2026-09-23): pyvista
# 0.49+ unconditionally imports IPython.core.guarded_eval, a module
# that only exists from IPython>=8.8 -- Colab ships IPython 7.34, so a
# fresh `pip install torch-fem` (which pulls in whatever pyvista is
# "latest" at install time, unpinned) can suddenly start failing with
# ModuleNotFoundError as soon as PyPI's latest pyvista crosses 0.49 --
# confirmed directly: the AmgX build itself succeeded cleanly, but the
# following `from torchfem.sparse import ...` failed with exactly this
# error. Pinning below 0.49 avoids the broken import path entirely.
_run_setup([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista<0.49'])
_run_setup([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod in list(sys.modules):
    if (_mod == 'torchfem' or _mod.startswith('torchfem.')
            or _mod == 'omar_pfem' or _mod.startswith('omar_pfem.')
            or _mod == 'pyvista' or _mod.startswith('pyvista.')):
        del sys.modules[_mod]


def run(cmd, cwd=None, check=True):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return p.returncode


# Real CUDA compute capability of this GPU, not assumed -- an A100 is
# sm_80, but this reads it directly so the same cell works unmodified
# on whatever GPU Colab actually assigns.
cc_major, cc_minor = torch.cuda.get_device_capability(0)
cuda_arch = f'{cc_major}{cc_minor}'
print(f'GPU: {torch.cuda.get_device_name(0)}  compute capability: {cc_major}.{cc_minor} '
      f'(CUDA_ARCH={cuda_arch})')

run(['nvcc', '--version'])

AMGX_SRC = '/content/AMGX'

# Idempotent: if a previous run in this same VM/disk already built
# libamgxsh.so, reuse it instead of repeating a 15-40 minute build.
# This matters because Colab's disk (unlike its Python process memory)
# can survive a Runtime > Restart -- so a restart taken to clear a
# stale pyvista import (see below) does not throw away real build work.
found = subprocess.run(
    ['find', AMGX_SRC, '-iname', 'libamgxsh.so'],
    capture_output=True, text=True,
).stdout.strip().splitlines() if os.path.isdir(AMGX_SRC) else []

if found:
    amgx_so = found[0]
    print(f'Found an existing build: {amgx_so} -- skipping the build, reusing it.')
else:
    if not os.path.isdir(AMGX_SRC):
        run(['git', 'clone', '--depth', '1', '--recursive',
             'https://github.com/NVIDIA/AMGX.git', AMGX_SRC])

    BUILD_DIR = f'{AMGX_SRC}/build'
    os.makedirs(BUILD_DIR, exist_ok=True)

    cmake_cmd = [
        'cmake', '..',
        '-DCMAKE_C_COMPILER=gcc', '-DCMAKE_CXX_COMPILER=g++',
        '-DCMAKE_BUILD_TYPE=Release', f'-DCUDA_ARCH={cuda_arch}',
    ]
    rc = run(cmake_cmd, cwd=BUILD_DIR, check=False)
    if rc != 0:
        raise RuntimeError(f'cmake configure failed with exit code {rc} -- see output above')

    rc = run(['make', '-j', str(os.cpu_count())], cwd=BUILD_DIR, check=False)
    if rc != 0:
        print('\nStandard build failed -- retrying with explicit OpenMP link flags '
              '(a documented AMGX build issue: the linker sometimes does not pull '
              'in libgomp automatically, github.com/NVIDIA/AMGX issue #214).')
        run(['cmake', '..',
             '-DCMAKE_C_COMPILER=gcc', '-DCMAKE_CXX_COMPILER=g++',
             '-DCMAKE_BUILD_TYPE=Release', f'-DCUDA_ARCH={cuda_arch}',
             '-DCMAKE_EXE_LINKER_FLAGS=-lgomp',
             '-DCMAKE_SHARED_LINKER_FLAGS=-lgomp'],
            cwd=BUILD_DIR)
        run(['make', '-j', str(os.cpu_count())], cwd=BUILD_DIR)

    # Find whatever the build actually produced -- don't assume the exact
    # path, search for it, since AMGX's own CMake layout can place build
    # artifacts in different subdirectories across versions.
    found = subprocess.run(
        ['find', AMGX_SRC, '-iname', 'libamgxsh.so'],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    if not found:
        raise RuntimeError(
            'Build finished but libamgxsh.so was not found anywhere under '
            f'{AMGX_SRC} -- check the make output above for what libraries it '
            'actually produced.')
    amgx_so = found[0]
    print(f'\nFound: {amgx_so}')

os.environ['AMGX_DLL'] = amgx_so

# Re-import torchfem's sparse module fresh so it re-checks AMGX_DLL and
# available_backends now that the library exists. Clearing pyvista too:
# a failed pyvista import earlier in this SAME kernel run would leave a
# partially-initialized module cached in sys.modules, and re-importing
# torchfem would just hit that same cached failure again.
for _mod in list(sys.modules):
    if (_mod == 'torchfem' or _mod.startswith('torchfem.')
            or _mod == 'pyvista' or _mod.startswith('pyvista.')):
        del sys.modules[_mod]

from torchfem.sparse import available_backends
print(f'available_backends after build: {available_backends}')
if 'amgx' not in available_backends:
    raise RuntimeError(
        'libamgxsh.so was built and AMGX_DLL is set, but torchfem still does '
        'not report "amgx" in available_backends -- something in the binding '
        'itself (ctypes symbol lookup) is not matching this build; report the '
        'exact available_backends value above.')

print('\nAmgX is now available to torch-fem. Testing the exact failing case '
      '(53,27) = 105,456 elements on CUDA with preconditioner="amg" -- same '
      'geometry, materials, region, precision, load path, and tolerances as '
      'the run that failed with Jacobi.')

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
print('\nIf this reaches "converged | 10 increments" above without a '
      'ConvergenceError, AmgX + AMG genuinely fixes the conditioning problem '
      'at full GPU speed -- real result, not a weakened target.')
