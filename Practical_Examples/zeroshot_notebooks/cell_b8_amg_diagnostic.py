# =====================================================================
#  DIAGNOSTIC CELL -- paste this into a NEW cell in the SAME Colab
#  runtime as B8_GPU_MeshConvergence.ipynb, AFTER its main cell has
#  already run at least once (so the repo is cloned and
#  omar_pfem/torchfem are already importable). Do not change geometry,
#  materials, QoI region, precision, load path, or tolerances -- this
#  tests ONLY the linear-solver preconditioner on the EXACT same
#  105,456-element case that failed with a real ConvergenceError
#  (CG did not reach 1e-08 within iteration limit; Newton-Raphson did
#  not converge in increment 4 after 10 cutbacks).
#
#  torchfem's CUDA AMG path requires AmgX, a library with no pip wheel
#  (must be built from source, confirmed unavailable here). The CPU
#  path uses pyamg (pure Python/SciPy, already installed with
#  torch-fem's own dependencies) automatically whenever preconditioner
#  is left as its own default on a CPU device -- this is the real,
#  immediately-testable alternative: same equations, same stol=1e-8,
#  same 30-iteration Newton cap, same 10-cutback budget, only the
#  linear solve's preconditioner and device (CPU instead of GPU, since
#  pyamg has no CUDA implementation) differ.
# =====================================================================
import time
from omar_pfem.data.mesh_convergence_B8 import solve_case

print('Testing the exact failing case (53,27) = 105,456 elements with AMG '
      'preconditioner on CPU (pyamg) -- same geometry, materials, region, '
      'precision, load path, and tolerances as the GPU/Jacobi run that failed.')
t0 = time.time()
r = solve_case(53, 27, device='cpu', preconditioner='amg', verbose=True)
elapsed = time.time() - t0
print(f'\nAMG/CPU result: n_elements={r["n_elements"]:,}  total_time={elapsed:.2f}s')
print(f'force_rel_residual={r["force_rel_residual"]:.2e}')
print(f'region_avg_sigma_xx={r["region_avg_sigma_xx"]:.4f} '
      f'(true_max={r["region_true_max_sigma_xx"]:.4f}, diagnostic only)')
print(f'max_strain_shim={r["max_strain_shim"]:.6e}  '
      f'max_strain_rubber={r["max_strain_rubber"]:.6e}')
print('\nIf this reaches "converged | 10 increments" above without a '
      'ConvergenceError, AMG genuinely fixes the conditioning problem that '
      'Jacobi could not, at this exact resolution -- real result, not a '
      'weakened target.')
