# =====================================================================
#  CELL -- real, medium-scale test of B8-final's NEW rigid-shim model
#  (rigid_shim_solver.py), before any full 10^5-10^6-element production
#  study. Real motivation: the deformable-steel model failed to
#  converge at 105,456 elements with BOTH Jacobi and real GPU AmgX/AMG
#  (proved this was the extreme steel/rubber stiffness ratio itself,
#  not a solver-choice problem). The new rigid-shim model (each shim an
#  exact rigid body, motion determined by equilibrium) was validated at
#  15,600 elements: displacement L2 error 2.09%, region-Cauchy field
#  error 1.63%, reaction force difference 1.36% vs the deformable
#  model -- all comfortably inside the 5-10% QoI band, and 6x FASTER
#  (49.2s vs 292.8s at that same resolution).
#
#  This cell does NOT jump straight to the full production range. The
#  new model's reduced linear system uses a DIRECT sparse solve (SciPy)
#  rather than an iterative one -- a real, untested scaling question at
#  this larger size (direct solves can become slow/memory-heavy as
#  problem size grows, a DIFFERENT failure mode than the ill-
#  conditioning that broke the deformable model, and not yet checked).
#  A real caution from the same review that requested this model: rubber
#  is also near-incompressible, and displacement-based HEX8 elements can
#  suffer their own volumetric-locking-related conditioning problems,
#  independent of the (now-removed) shim stiffness contrast -- if THIS
#  run itself struggles, that is the next real thing to investigate, not
#  a reason to alter the real material parameters.
#
#  Ladder: (37,19)=50,544 / (45,23)=75,504 / (53,27)=105,456 elements --
#  the last one is the EXACT resolution that failed for the deformable
#  model, for a direct, meaningful comparison.
# =====================================================================
import os
import subprocess
import sys

_started_msg = 'Starting rigid-shim medium-scale test'
print(_started_msg, flush=True)

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


try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO = '/content/OMAR' if IN_COLAB else os.getcwd()
if IN_COLAB:
    if not os.path.isdir(REPO):
        run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
             'https://github.com/SUHIBAMRO/OMAR.git', REPO])
    else:
        run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
        run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
        run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

    # Real, external, currently-active PyPI issue (2026-09-23): pyvista
    # 0.49+ unconditionally imports IPython.core.guarded_eval, which
    # only exists from IPython>=8.8 -- Colab ships IPython 7.34.
    run([sys.executable, '-m', 'pip', 'install', '-q', 'pyvista<0.49'])
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

    WORK = f'{REPO}/Practical_Examples'
    os.chdir(WORK)
    sys.path.insert(0, WORK)

    for _mod in list(sys.modules):
        if (_mod == 'torchfem' or _mod.startswith('torchfem.')
                or _mod == 'omar_pfem' or _mod.startswith('omar_pfem.')
                or _mod == 'pyvista' or _mod.startswith('pyvista.')):
            del sys.modules[_mod]

device = torch.device('cuda') if (IN_COLAB and torch.cuda.is_available()) else torch.device('cpu')
print(f'device = {device}' + (f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

from omar_pfem.data.rigid_shim_solver import solve_case as solve_rigid_shim

RESOLUTIONS = [(37, 19), (45, 23), (53, 27)]  # 50,544 / 75,504 / 105,456 elements

rows = []
for Ntheta, Nr in RESOLUTIONS:
    print(f'\n{"=" * 90}\nSolving rigid-shim model at ({Ntheta},{Nr})...', flush=True)
    r = solve_rigid_shim(Ntheta, Nr, device=device, verbose=True)
    rows.append(r)
    print(f"\n  n_elements={r['n_elements']:,}  time={r['elapsed_s']:.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}")
    print(f"  max_disp={r['max_disp']:.4f}mm  total_strain_energy={r['total_strain_energy']:.4f}")
    print(f"  reaction_force_bottom={r['reaction_force_bottom']}")

print(f'\n{"=" * 90}\nSummary across the ladder:')
for (Ntheta, Nr), r in zip(RESOLUTIONS, rows):
    print(f"  ({Ntheta},{Nr})  n_elem={r['n_elements']:>9,}  time={r['elapsed_s']:>8.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}  max_disp={r['max_disp']:.4f}mm")

print('\nDone. If (53,27)=105,456 elements converged cleanly and in reasonable '
      'time here (the exact resolution that failed for the deformable-steel '
      'model), the rigid-shim model is a real candidate for the full '
      '10^5-10^6-element production study next -- report the real numbers '
      'above either way.')
