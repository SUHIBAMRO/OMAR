"""Item G scoping ("faster iterative solvers: AMG-Krylov with multi-GPU/
domain decomposition", Timon's 2026-10-11 email): before designing any
new multi-GPU/domain-decomposition infrastructure (a large, genuinely new
engineering effort), check what's already available for the FIRST half
of the ask -- algebraic multigrid (AMG) -- since torch-fem turns out to
already wire this up natively.

Real finding from reading torchfem's own source
(site-packages/torchfem/sparse.py): `Solid.solve(..., preconditioner=)`
already supports "amg" (via `pyamg.smoothed_aggregation_solver`, already
installed in this environment -- `pyamg==5.3.0`, confirmed via `pip show`)
on CPU with ZERO new dependencies, and "amg" on CUDA routes to NVIDIA
AmgX, which is NOT available here (ships no prebuilt wheels, needs
building from source with AMGX_DLL set -- real infra work, not done).
Multi-GPU/domain decomposition is not implemented anywhere in torch-fem
or this project at all -- a substantial, genuinely new task, well beyond
swapping a preconditioner flag.

This script answers the one thing testable RIGHT NOW with no new
dependencies and no GPU: does torch-fem's existing CPU AMG preconditioner
actually help on this project's own real B3 problem, measured, not
assumed -- the standard this project always holds speedup/solver claims
to.
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3_dataset import (
    DEFAULT_RESOLUTION, sample_material_and_load, solve_one_sample,
)

RESOLUTIONS = [(13, 12, 11), (17, 16, 15), (21, 20, 19), (29, 26, 27)]  # 1,320 / 3,360 / 6,840 / 18,200 elements

for Ntheta, Nr, Nz in RESOLUTIONS:
    n_elem = (Ntheta - 1) * (Nr - 1) * (Nz - 1)
    E_node, nu_node, phi, _, _ = sample_material_and_load(Ntheta, Nr, Nz, seed=0)

    for precond in ("jacobi", "amg"):
        t0 = time.time()
        result = solve_one_sample(
            Ntheta, Nr, Nz, E_node, nu_node, phi,
            device=torch.device("cpu"), verbose=False, preconditioner=precond,
        )
        elapsed = time.time() - t0
        print(f"n_elem={n_elem:>6}  preconditioner={precond:<8}  "
              f"wall_clock={elapsed:7.3f}s  "
              f"force_rel_residual={result['force_rel_residual']:.3e}  "
              f"max_disp={result['max_disp']:.4f}")
    print()
