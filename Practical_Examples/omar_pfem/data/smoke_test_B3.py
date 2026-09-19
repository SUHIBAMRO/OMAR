"""CPU-only smoke test: solve one tiny B3 mesh with torch-fem's real 3D
hyperelastic solid element (Solid + Hyperelastic3D + neo_hookean_psi_3d,
reused unchanged from torchfem_comparison.py) before spending any GPU
time or building the full data-generation pipeline. Matches this
project's own standing discipline (verify small/cheap before scaling).

Checks: Newton converges; no NaN/Inf; the loaded face moves in +x (the
direction of the applied traction); the free faces (y=Ly, hole, z=0/Lz)
show a real Poisson-type contraction (u_y<0 near the loaded region,
u_z != 0 away from the mid-plane) -- i.e. a genuinely 3D, physically
sane deformation, not a degenerate/planar one.
"""
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import generate_grid_hex8_plate_with_hole, boundary_node_sets
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d


def main():
    dtype = torch.float64
    device = torch.device("cpu")

    Lx = Ly = 1.0
    Lz = 0.3
    R = 0.2
    Ntheta, Nr, Nz = 9, 7, 4
    nodes, elements = generate_grid_hex8_plate_with_hole(Lx, Ly, Lz, R, Ntheta, Nr, Nz)
    sym_y0, sym_x0, loaded, free_y, hole = boundary_node_sets(nodes, Lx, Ly)
    print(f"nodes={nodes.shape[0]}, elements={elements.shape[0]}")

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)

    # Single material, Neo-Hookean, same (mu, lam) convention as B1/B2's
    # own solver (data/materials.py) -- E=1000, nu=0.45 (rubber-like,
    # nearly incompressible but well within this project's own already-
    # validated Poisson-ratio range for Neo-Hookean).
    E, nu = 1000.0, 0.45
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    params = torch.tensor([mu, lam], dtype=dtype, device=device)  # one material for all elements (uniform field)

    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
    with torch.device(device):
        model = Solid(nodes_t, elements_t, material)

    n_nodes = nodes.shape[0]
    forces = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    # Small uniform +x force on the loaded face, split evenly across its
    # own nodes -- a cheap stand-in for a real traction field, sufficient
    # for THIS smoke test (real data generation will use a proper
    # consistent nodal-force assembly, matching B1/B2's own convention).
    total_fx = 1.0
    loaded_idx = np.where(loaded)[0]
    forces[loaded_idx, 0] = total_fx / len(loaded_idx)
    model.forces = forces

    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    constraints[sym_y0, 1] = True  # u_y = 0 on y=0
    constraints[sym_x0, 0] = True  # u_x = 0 on x=0
    # Pin z-rigid-body motion: fix u_z on the z=0 plane only (does not
    # touch the physics elsewhere -- z=0 is already a free surface in
    # traction, this just removes the rigid-body translation along z).
    z0 = np.abs(nodes[:, 2]) < 1e-9
    constraints[z0, 2] = True
    model.constraints = constraints
    model.displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

    increments = torch.linspace(0.0, 1.0, 11, dtype=dtype, device=device)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            u, *_ = model.solve(increments=increments, max_iter=30, rtol=1e-8, atol=1e-8,
                                 stol=1e-8, method="cg", preconditioner="jacobi",
                                 nlgeom=True, verbose=True)
    finally:
        torch.set_default_dtype(old_default_dtype)

    u_np = u[-1].cpu().numpy() if u.dim() == 3 else u.cpu().numpy()
    print("u shape:", u_np.shape)
    assert np.isfinite(u_np).all(), "NaN/Inf in the solution -- solve did not really converge cleanly"

    ux, uy, uz = u_np[:, 0], u_np[:, 1], u_np[:, 2]
    print(f"max ux={ux.max():.6e} (loaded face mean ux={ux[loaded_idx].mean():.6e})")
    print(f"min uy={uy.min():.6e} (should be <0 somewhere -- Poisson contraction)")
    print(f"uz range=[{uz.min():.6e}, {uz.max():.6e}] (should be non-degenerate, genuinely 3D)")

    assert ux[loaded_idx].mean() > 0, "loaded face should move in +x"
    assert uy.min() < -1e-12, "expected some Poisson-type y-contraction under +x tension"
    assert (uz.max() - uz.min()) > 1e-12, "uz should not be uniformly zero -- solution should be genuinely 3D"
    print("\nSMOKE TEST PASSED: real 3D hyperelastic solve converges and gives a sane, non-degenerate result.")


if __name__ == "__main__":
    main()
