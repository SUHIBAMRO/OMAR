"""CPU-only smoke test for the CORRECTED B3 design (rocking rubber
bushing, see data_generate_B3.py's own docstring for the full
rationale): solve one small mesh with torch-fem's real 3D hyperelastic
solid element (Solid + Hyperelastic3D + neo_hookean_psi_3d, unchanged
from torchfem_comparison.py) before spending any GPU time or building
the full data-generation pipeline.

Checks: Newton converges with a real, non-homogeneous Dirichlet BC (the
rocking core, not just a force -- this is the first time this project
applies a nonzero prescribed-displacement BC via torch-fem, not only
Neumann forces, so this is itself a real capability check); the inner
core's own applied displacement is respected at both ends (z=0: -delta0,
z=Lz: +delta0); the outer housing stays fixed; the deformation genuinely
varies with z, not just following the core linearly (i.e. real material
response, not a rigid rotation of everything)."""
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rocking_displacement_x)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d


def main():
    dtype = torch.float64
    device = torch.device("cpu")

    R_in, R_out, Lz = 0.5, 1.0, 1.0
    Ntheta, Nr, Nz = 13, 6, 7
    nodes, elements = generate_grid_hex8_bushing(R_in, R_out, Lz, Ntheta, Nr, Nz)
    inner, outer, sym = boundary_node_sets(nodes, R_in, R_out)
    print(f"nodes={nodes.shape[0]}, elements={elements.shape[0]}")

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)

    E, nu = 1000.0, 0.45  # rubber-like, same convention as B1/B2
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

    # Rigid outer housing: fixed in all 3 components.
    constraints[outer, :] = True

    # Rigid inner core: rocks in x only (its own rigid-body radial/z
    # motion is zero -- a pure tilt), free... but the core is RIGID, so
    # its own y and z displacement at the bonded surface must also be
    # prescribed (zero, since the core does not translate in y or z, and
    # does not expand/contract -- it is rigid). Constrain all 3
    # components on the inner surface: x per the rocking profile, y=z=0.
    delta0 = 0.03 * (R_out - R_in)  # small tilt amplitude, a fraction of the gap width
    constraints[inner, :] = True
    displacements[inner, 0] = torch.tensor(
        rocking_displacement_x(nodes[inner], Lz, delta0), dtype=dtype, device=device)
    # displacements[inner, 1] and [..., 2] stay 0 (no y/z rigid motion)

    # Symmetry (y=0 plane, both theta=0 and theta=pi rows): u_y = 0.
    # (Already implied by inner/outer at those rows since y=0 there too,
    # but the INTERIOR nodes at theta=0/pi, r strictly between R_in and
    # R_out, are not otherwise constrained and need this explicitly.)
    constraints[sym, 1] = True

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

    ux, uy, uz = u_np[:, 0], u_np[:, 1], u_np[:, 2]

    # The prescribed BC must be respected exactly (up to solver tolerance)
    # at the constrained nodes.
    z0_inner = inner & (np.abs(nodes[:, 2]) < 1e-9)
    zL_inner = inner & (np.abs(nodes[:, 2] - Lz) < 1e-9)
    print(f"inner core ux at z=0: mean={ux[z0_inner].mean():.6e} (expect ~{-delta0:.6e})")
    print(f"inner core ux at z=Lz: mean={ux[zL_inner].mean():.6e} (expect ~{+delta0:.6e})")
    assert abs(ux[z0_inner].mean() - (-delta0)) < 1e-6
    assert abs(ux[zL_inner].mean() - delta0) < 1e-6
    assert np.abs(ux[outer]).max() < 1e-10, "outer housing should stay exactly fixed"
    assert np.abs(uy[outer]).max() < 1e-10
    assert np.abs(uz[outer]).max() < 1e-10

    # Genuinely 3D check: uy and uz should be non-degenerate (not
    # identically zero) somewhere in the INTERIOR of the domain -- a bug
    # that accidentally decoupled z-slices (e.g. a mesh/BC error making
    # this behave like independent 2D rings) would still show pure x
    # motion with uy, uz basically zero everywhere except by symmetry.
    interior = ~(inner | outer | sym)
    print(f"interior uy range: [{uy[interior].min():.3e}, {uy[interior].max():.3e}]")
    print(f"interior uz range: [{uz[interior].min():.3e}, {uz[interior].max():.3e}]")
    assert uy[interior].max() - uy[interior].min() > 1e-9
    assert uz[interior].max() - uz[interior].min() > 1e-9

    print("\nSMOKE TEST PASSED: rocking rubber-mount bushing solves cleanly, "
          "respects its own prescribed BC, and gives a genuinely non-degenerate "
          "3D deformation field.")


if __name__ == "__main__":
    main()
