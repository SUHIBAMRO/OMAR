"""CPU-only smoke test for the CURRENT B3 design (continuously-bonded
core with a smooth groove feature + true rigid-rotation kinematics --
see data_generate_B3.py's own docstring for the full design history and
rationale). Solve one small mesh with torch-fem's real 3D hyperelastic
solid element (Solid + Hyperelastic3D + neo_hookean_psi_3d, unchanged
from torchfem_comparison.py) before spending any GPU time or building
the full data-generation pipeline.

Checks: Newton converges with a real, non-homogeneous Dirichlet BC (the
rotating core); the core's own prescribed rotation is respected exactly
at every bonded node (not just at two ends -- the core is now bonded at
EVERY z); the outer housing stays fixed; the deformation is genuinely
3D (nonzero u_y, u_z in the interior); and -- new for this design --
the core's own u_z is nonzero (confirming the TRUE rotation kinematics
are in effect, not the old z-motion-free linear approximation)."""
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
    groove_radius_of_curvature)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d


def main():
    dtype = torch.float64
    device = torch.device("cpu")

    R_in0, R_out, Lz = 0.5, 1.0, 1.0
    groove_depth, groove_half_width = 0.05, 0.15
    Ntheta, Nr, Nz = 13, 6, 15
    nodes, elements = generate_grid_hex8_bushing(
        R_in0, R_out, Lz, Ntheta, Nr, Nz, groove_depth, groove_half_width)
    inner, outer, sym = boundary_node_sets(nodes, R_in0, R_out, Lz, groove_depth, groove_half_width)
    print(f"nodes={nodes.shape[0]}, elements={elements.shape[0]}, "
          f"inner_core(bonded, ALL z)={inner.sum()}, outer_housing={outer.sum()}")
    print(f"groove radius of curvature: {groove_radius_of_curvature(groove_depth, groove_half_width):.4f}")

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)

    E, nu = 1000.0, 0.45
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    params = torch.tensor([mu, lam], dtype=dtype, device=device)
    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
    with torch.device(device):
        model = Solid(nodes_t, elements_t, material)

    n_nodes = nodes.shape[0]
    model.forces = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

    constraints[outer, :] = True  # fixed housing

    phi = 0.05  # rotation angle, radians (~2.9 degrees) -- small but finite
    ux, uy, uz = rigid_rotation_displacement(nodes[inner], Lz, phi)
    constraints[inner, :] = True
    displacements[inner, 0] = torch.tensor(ux, dtype=dtype, device=device)
    displacements[inner, 1] = torch.tensor(uy, dtype=dtype, device=device)
    displacements[inner, 2] = torch.tensor(uz, dtype=dtype, device=device)

    constraints[sym, 1] = True  # symmetry plane, u_y=0

    model.constraints = constraints
    model.displacements = displacements

    increments = torch.linspace(0.0, 1.0, 11, dtype=dtype, device=device)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", nlgeom=True, verbose=True)
    finally:
        torch.set_default_dtype(old_default_dtype)

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all(), "NaN/Inf in the solution"
    ux_sol, uy_sol, uz_sol = u_np[:, 0], u_np[:, 1], u_np[:, 2]

    # BC respected exactly at every bonded core node (all z now, not just
    # two ends).
    expected_ux, expected_uy, expected_uz = rigid_rotation_displacement(nodes[inner], Lz, phi)
    assert np.allclose(ux_sol[inner], expected_ux, atol=1e-6)
    assert np.allclose(uy_sol[inner], expected_uy, atol=1e-6)
    assert np.allclose(uz_sol[inner], expected_uz, atol=1e-6)
    print(f"core BC respected: max |ux err|={np.abs(ux_sol[inner]-expected_ux).max():.2e}, "
          f"max |uz err|={np.abs(uz_sol[inner]-expected_uz).max():.2e}")
    assert np.abs(uz_sol[inner]).max() > 1e-4, (
        "core's own uz should be meaningfully nonzero -- confirms TRUE rotation "
        "kinematics are active, not the old z-motion-free linear approximation")

    assert np.abs(ux_sol[outer]).max() < 1e-10, "outer housing should stay exactly fixed"
    assert np.abs(uy_sol[outer]).max() < 1e-10
    assert np.abs(uz_sol[outer]).max() < 1e-10

    interior = ~(inner | outer | sym)
    print(f"interior uy range: [{uy_sol[interior].min():.3e}, {uy_sol[interior].max():.3e}]")
    print(f"interior uz range: [{uz_sol[interior].min():.3e}, {uz_sol[interior].max():.3e}]")
    assert uy_sol[interior].max() - uy_sol[interior].min() > 1e-9
    assert uz_sol[interior].max() - uz_sol[interior].min() > 1e-9

    print("\nSMOKE TEST PASSED: continuously-bonded groove bushing with true "
          "rigid-rotation kinematics solves cleanly, respects its own BC "
          "everywhere on the core, and gives a genuinely non-degenerate 3D "
          "deformation field.")


if __name__ == "__main__":
    main()
