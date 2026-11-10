"""Direct, small-scale, CPU-only sanity check: does a genuinely locally-
refined (beta>0) mesh actually solve correctly (Newton converges, no
NaN/Inf, physically sensible displacement), not just "build without
crashing"? Run BEFORE trusting this mesh generator with any real GPU
time, matching this project's own standing discipline.

Usage: python -m omar_pfem.data.test_z_refine_solve_sanity
"""
import numpy as np
import torch

from omar_pfem.data.data_generate_B3_local_z_refine import generate_grid_hex8_bushing_z_refined
from omar_pfem.data.data_generate_B3 import boundary_node_sets, rigid_rotation_displacement
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.20, 0.15


def solve_with_mesh(nodes, elements, phi=0.05, E=1000.0, nu=0.45, n_increments=11, dtype=torch.float64):
    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    params = torch.tensor([mu, lam], dtype=dtype).unsqueeze(0).expand(n_elem, -1).contiguous()

    nodes_t = torch.tensor(nodes, dtype=dtype)
    elements_t = torch.tensor(elements, dtype=torch.long)
    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
    with torch.device("cpu"):
        model = Solid(nodes_t, elements_t, material)

    model.forces = torch.zeros(n_nodes, 3, dtype=dtype)
    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool)
    displacements = torch.zeros(n_nodes, 3, dtype=dtype)
    constraints[outer, :] = True
    ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, phi)
    constraints[inner, :] = True
    displacements[inner, 0] = torch.tensor(ux, dtype=dtype)
    displacements[inner, 1] = torch.tensor(uy, dtype=dtype)
    displacements[inner, 2] = torch.tensor(uz, dtype=dtype)
    constraints[sym, 1] = True
    model.constraints = constraints
    model.displacements = displacements

    increments = torch.linspace(0.0, 1.0, n_increments, dtype=dtype)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("cpu"), torch.no_grad():
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", verbose=False,
                aggregate_integration_points=False)
    finally:
        torch.set_default_dtype(old_dtype)

    u_np = u.cpu().numpy()
    ok = np.isfinite(u_np).all()
    f_np = f.cpu().numpy()
    force_rel_residual = float(np.linalg.norm(f_np.sum(axis=0)) /
                                max(np.linalg.norm(f_np[outer].sum(axis=0)), 1e-8))
    return ok, float(np.abs(u_np).max()), force_rel_residual


if __name__ == "__main__":
    Ntheta, Nr, Nz = 13, 12, 19  # small, CPU-feasible; Nz kept at a real value so z-refine has room to work
    for beta in (0.0, 2.0, 4.0, 6.0):
        nodes, elements = generate_grid_hex8_bushing_z_refined(
            R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH,
            r_grading=1.0, z_beta=beta)
        ok, max_disp, force_rel_residual = solve_with_mesh(nodes, elements)
        print(f"beta={beta:>4.1f}: n_elem={elements.shape[0]:>6}  finite={ok}  "
              f"max|u|={max_disp:.4f}  force_rel_residual={force_rel_residual:.2e}  "
              f"{'PASS' if ok and force_rel_residual < 1e-6 else 'FAIL'}")
