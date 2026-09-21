"""CPU-only smoke test for the tire-sector candidate (see
data_generate_tire.py's own docstring for the full design rationale and
disclosed simplifications). Solve one small mesh with torch-fem's real
3D hyperelastic solid element (Solid + Hyperelastic3D + neo_hookean_
psi_3d, unchanged from torchfem_comparison.py, exactly as B3 uses it)
before spending any GPU time.

Checks: Newton converges with a real pressure load (torch-fem's own
`integrate_surface_load`, a genuinely new load TYPE for this project --
B3 used prescribed displacement, this uses a real distributed pressure
load); the rim stays exactly fixed; the tread deflects inward under the
pressure (a basic physical sanity check -- the contact patch should
move in the direction of the applied pressure, i.e. inward, toward the
wheel axis); the deformation is genuinely non-degenerate in all three
directions (confirms the torus-sector geometry is really 3D, not
collapsed)."""
import numpy as np
import torch

from omar_pfem.data.data_generate_tire import generate_grid_hex8_tire_sector, boundary_node_sets
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d


def main():
    dtype = torch.float64
    device = torch.device("cpu")

    R_bead, R_tread0, R_big = 0.5, 1.0, 3.0
    Phi_max = np.pi / 6
    groove_depth, groove_half_theta = 0.05, 0.15
    contact_half_theta = 0.3
    Ntheta, Nr, Nphi = 13, 6, 9
    nodes, elements = generate_grid_hex8_tire_sector(
        R_bead, R_tread0, R_big, Phi_max, Ntheta, Nr, Nphi, groove_depth, groove_half_theta)
    rim, contact_patch = boundary_node_sets(
        nodes, R_bead, R_tread0, R_big, groove_depth, groove_half_theta, contact_half_theta)
    print(f"nodes={nodes.shape[0]}, elements={elements.shape[0]}, "
          f"rim={rim.sum()}, contact_patch nodes={contact_patch.sum()}")

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
    contact_mask = torch.tensor(contact_patch, device=device)
    pressure = -5.0  # negative = inward (compresses the tread), a modest magnitude
    # torch-fem's own _scatter/index_add_ picks up whatever the GLOBAL
    # default dtype is for its internal zero-tensor allocation -- same
    # class of bug already documented in torchfem_comparison.py (near_
    # null_space/skew hardcoding float32 unless the global default is
    # overridden first). Wrap this call the same way .solve() is wrapped
    # below.
    _old_dt = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        forces = model.integrate_surface_load(contact_mask, pressure)
    finally:
        torch.set_default_dtype(_old_dt)
    model.forces = forces

    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    constraints[rim, :] = True  # rim bonded to a FIXED rigid rim
    model.constraints = constraints
    model.displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

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

    assert np.abs(u_np[rim]).max() < 1e-10, "rim should stay exactly fixed"

    # Contact-patch nodes should move INWARD (toward the wheel axis,
    # i.e. their own radial-from-wheel-axis distance should decrease)
    # under the inward pressure.
    radial0 = np.sqrt(nodes[contact_patch, 0] ** 2 + nodes[contact_patch, 2] ** 2)
    new_pos = nodes[contact_patch] + u_np[contact_patch]
    radial1 = np.sqrt(new_pos[:, 0] ** 2 + new_pos[:, 2] ** 2)
    print(f"contact patch mean radial displacement: {(radial1 - radial0).mean():.6e} (expect < 0)")
    assert (radial1 - radial0).mean() < 0, "contact patch should move inward under an inward pressure"

    interior = ~(rim | contact_patch)
    disp_mag = np.linalg.norm(u_np[interior], axis=1)
    print(f"interior |u| range: [{disp_mag.min():.3e}, {disp_mag.max():.3e}]")
    assert disp_mag.max() > 1e-6, "interior deformation should be non-degenerate"
    # genuinely 3D check: displacement should have a real component along
    # all three global axes somewhere in the interior (not collapsed to
    # a 2D-equivalent motion)
    for axis, name in enumerate("xyz"):
        rng = u_np[interior, axis].max() - u_np[interior, axis].min()
        print(f"  interior u_{name} range: {rng:.3e}")
        assert rng > 1e-8, f"u_{name} suspiciously flat -- geometry may have collapsed a dimension"

    print("\nSMOKE TEST PASSED: tire-sector torus geometry with a real pressure "
          "load solves cleanly, respects its own BC, deflects in the physically "
          "expected direction, and gives a genuinely non-degenerate 3D "
          "deformation field.")


if __name__ == "__main__":
    main()
