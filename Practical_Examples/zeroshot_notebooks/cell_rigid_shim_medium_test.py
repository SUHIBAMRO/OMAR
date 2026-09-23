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
#  UPDATE, 2026-09-23, real evidence from THIS ladder's own live GPU run:
#  the reduced system's DIRECT sparse solve (SciPy `spsolve`) -- fast at
#  50,544 elements (555.27s total, matching an earlier CPU-only check) --
#  became highly disproportionately slow at 75,504 elements (confirmed
#  alive, not hung, via `top`: the python process sat at 100% CPU for
#  20+ minutes on a single Newton iteration that should cost seconds).
#  Added `linear_solver='cg'` to `rigid_shim_solver.solve_case` (Jacobi-
#  preconditioned CG, WITH per-iteration progress printing so a slow
#  solve is never mistaken for a silent hang again).
#
#  SECOND UPDATE, same day, real GPU evidence again: 'cg' is CORRECT at
#  50,544 elements -- bit-identical to 'direct' (max_disp rel diff
#  1.6e-15) and 4.3x FASTER (126.66s vs 546.11s) -- but genuinely BREAKS
#  DOWN at 75,504 elements, and WORSE than 'direct's mere slowness: CG's
#  own iteration count exploded (4,275 -> 6,673 -> 13,129 iterations
#  across successive Newton iterations of increment 1 alone), the
#  Newton residual DIVERGED instead of converging (1.19e5 -> 2.27e6 ->
#  1.23e9), and the run crashed with `AssertionError: non-positive
#  diagonal in K_reduced` -- the tangent stiffness itself became invalid,
#  almost certainly because a bad/inaccurate CG step (plain Jacobi
#  preconditioning is known to be weak when a system mixes very
#  different physical DOF scales, here free rubber translations vs.
#  each shim's 3-dof rigid block) pushed the deformation state somewhere
#  physically invalid. This is exactly the standing caution already
#  written into this module: rubber is also near-incompressible, and
#  displacement-based HEX8 elements can suffer their own volumetric-
#  locking-related conditioning problems, independent of the (now-
#  removed) shim stiffness contrast -- plain-Jacobi CG is not yet a safe
#  choice at this scale, so 'direct' is used again for (45,23)/(53,27)
#  below (slow but numerically robust, unlike CG's outright divergence
#  here) while a stronger preconditioner for CG is investigated
#  separately, offline, before being tried again at this scale.
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

print(f'\n{"=" * 90}\nSTEP 1: live GPU validation of the new "cg" linear_solver against the '
      f'already-confirmed-good "direct" one, at (37,19)=50,544 elements (a real, '
      f'previously-solved resolution -- not a guess).', flush=True)
r_direct = solve_rigid_shim(37, 19, device=device, verbose=True, linear_solver='direct')
print(f"\n  [direct] n_elements={r_direct['n_elements']:,}  time={r_direct['elapsed_s']:.2f}s  "
      f"force_rel_residual={r_direct['force_rel_residual']:.2e}  max_disp={r_direct['max_disp']:.4f}mm  "
      f"total_strain_energy={r_direct['total_strain_energy']:.4f}")

r_cg = solve_rigid_shim(37, 19, device=device, verbose=True, linear_solver='cg')
print(f"\n  [cg] n_elements={r_cg['n_elements']:,}  time={r_cg['elapsed_s']:.2f}s  "
      f"force_rel_residual={r_cg['force_rel_residual']:.2e}  max_disp={r_cg['max_disp']:.4f}mm  "
      f"total_strain_energy={r_cg['total_strain_energy']:.4f}")

disp_rel_diff = abs(r_direct['max_disp'] - r_cg['max_disp']) / abs(r_direct['max_disp'])
energy_rel_diff = abs(r_direct['total_strain_energy'] - r_cg['total_strain_energy']) / abs(r_direct['total_strain_energy'])
print(f"\n  direct vs cg: max_disp rel diff={disp_rel_diff:.2e}  "
      f"total_strain_energy rel diff={energy_rel_diff:.2e}  "
      f"(should be ~1e-10 or smaller -- confirms 'cg' solves the SAME physics, "
      f"not an approximation)")
print(f"  time: direct={r_direct['elapsed_s']:.2f}s vs cg={r_cg['elapsed_s']:.2f}s")

# 'direct' is used for the rest of the ladder: 'cg' with plain Jacobi
# preconditioning is not yet safe at this scale (see the SECOND UPDATE
# above -- it diverged and crashed, not just slow). Slower but robust.
rows = [r_direct]
errors = []

for Ntheta, Nr in RESOLUTIONS[1:]:
    print(f'\n{"=" * 90}\nSolving rigid-shim model at ({Ntheta},{Nr}) with linear_solver=\'direct\' '
          f'(\'cg\' with plain Jacobi is known to diverge at this scale -- see the module docstring; '
          f'\'direct\' is slower but numerically robust)...', flush=True)
    try:
        r = solve_rigid_shim(Ntheta, Nr, device=device, verbose=True, linear_solver='direct')
    except Exception as e:
        print(f"\n  FAILED at ({Ntheta},{Nr}): {type(e).__name__}: {e}", flush=True)
        errors.append((Ntheta, Nr, str(e)))
        continue
    rows.append(r)
    print(f"\n  n_elements={r['n_elements']:,}  time={r['elapsed_s']:.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}")
    print(f"  max_disp={r['max_disp']:.4f}mm  total_strain_energy={r['total_strain_energy']:.4f}")
    print(f"  reaction_force_bottom={r['reaction_force_bottom']}")

print(f'\n{"=" * 90}\nSummary across the ladder (row 1 has both direct/cg numbers above; '
      f'the rest solved with linear_solver=\'direct\'):')
for r in rows:
    print(f"  ({r['Ntheta']},{r['Nr']})  n_elem={r['n_elements']:>9,}  time={r['elapsed_s']:>8.2f}s  "
          f"force_rel_residual={r['force_rel_residual']:.2e}  max_disp={r['max_disp']:.4f}mm")
if errors:
    print(f'\n  {len(errors)} resolution(s) FAILED (real, reported honestly, not hidden):')
    for Ntheta, Nr, msg in errors:
        print(f"    ({Ntheta},{Nr}): {msg}")

print('\nDone. If (53,27)=105,456 elements converged cleanly and in reasonable '
      'time here (the exact resolution that failed for the deformable-steel '
      'model), the rigid-shim model is a real candidate for the full '
      '10^5-10^6-element production study next -- report the real numbers '
      'above either way.')
