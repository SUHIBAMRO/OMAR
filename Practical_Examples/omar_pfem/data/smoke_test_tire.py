"""CPU-only smoke test for the tire-sector candidate (see
data_generate_tire.py's own docstring for the full design rationale and
disclosed simplifications). Solve one small mesh with torch-fem's real
3D hyperelastic solid element (Solid + Hyperelastic3D + neo_hookean_
psi_3d, unchanged from torchfem_comparison.py, exactly as B3 uses it)
before spending any GPU time.

REVISED 2026-09-21 (matches data_generate_tire.py's own review-point
fixes): uses the new (rim, full_tread, tread_load_region) 3-mask
signature (renamed from the old (rim, contact_patch) 2-mask one), and
now applies TWO superposed surface loads under a single incremental
ramp, per review point 3 -- an internal inflation pressure (positive,
OUTWARD) over the entire tread, plus an additional localized tread load
(negative, INWARD) restricted to `tread_load_region` -- rather than a
single inward-only pressure over just the local window. Both loads are
summed into one `model.forces` tensor and solved together (a disclosed
simplification: not two truly sequential load stages -- see the module
docstring in data_generate_tire.py, point 3).

Checks: Newton converges with the real superposed pressure loads
(torch-fem's own `integrate_surface_load`, a genuinely new load TYPE
for this project -- B3 used prescribed displacement, this uses real
distributed pressure loads); the rim stays exactly fixed; the REST of
the tread (inflation only, no local load) bulges OUTWARD as expected
for a pressurized tire; the tread LOAD region (inflation + local
inward load, net inward since the local load dominates in magnitude)
moves INWARD; the deformation is genuinely non-degenerate in all three
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
    tread_load_half_theta = 0.3
    Ntheta, Nr, Nphi = 13, 6, 9
    nodes, elements = generate_grid_hex8_tire_sector(
        R_bead, R_tread0, R_big, Phi_max, Ntheta, Nr, Nphi, groove_depth, groove_half_theta)
    rim, full_tread, tread_load_region = boundary_node_sets(
        nodes, R_bead, R_tread0, R_big, groove_depth, groove_half_theta, tread_load_half_theta)
    print(f"nodes={nodes.shape[0]}, elements={elements.shape[0]}, "
          f"rim={rim.sum()}, full_tread={full_tread.sum()}, "
          f"tread_load_region nodes={tread_load_region.sum()}")

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
    full_tread_mask = torch.tensor(full_tread, device=device)
    load_mask = torch.tensor(tread_load_region, device=device)
    INFLATION_PRESSURE = 2.0  # positive = outward, internal inflation, over the WHOLE tread
    LOCAL_LOAD_PRESSURE = -5.0  # negative = inward, additional localized tread load
    # torch-fem's own _scatter/index_add_ picks up whatever the GLOBAL
    # default dtype is for its internal zero-tensor allocation -- same
    # class of bug already documented in torchfem_comparison.py (near_
    # null_space/skew hardcoding float32 unless the global default is
    # overridden first). Wrap both calls the same way .solve() is
    # wrapped below.
    _old_dt = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        forces_inflation = model.integrate_surface_load(full_tread_mask, INFLATION_PRESSURE)
        forces_local_load = model.integrate_surface_load(load_mask, LOCAL_LOAD_PRESSURE)
    finally:
        torch.set_default_dtype(_old_dt)
    model.forces = forces_inflation + forces_local_load

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

    def radial_from_wheel_axis(pos):
        return np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)

    # The REST of the tread (inflation only, net outward pressure) should
    # bulge OUTWARD.
    rest_of_tread = full_tread & ~tread_load_region
    radial0_rest = radial_from_wheel_axis(nodes[rest_of_tread])
    radial1_rest = radial_from_wheel_axis(nodes[rest_of_tread] + u_np[rest_of_tread])
    d_rest = (radial1_rest - radial0_rest).mean()
    print(f"rest-of-tread (inflation only) mean radial displacement: {d_rest:.6e} (expect > 0)")
    assert d_rest > 0, "inflation-only tread should bulge outward"

    # The LOAD region (inflation + a larger-magnitude inward local load)
    # should move net INWARD.
    radial0_load = radial_from_wheel_axis(nodes[tread_load_region])
    radial1_load = radial_from_wheel_axis(nodes[tread_load_region] + u_np[tread_load_region])
    d_load = (radial1_load - radial0_load).mean()
    print(f"tread load region mean radial displacement: {d_load:.6e} (expect < 0)")
    assert d_load < 0, "tread load region should move net inward (local load dominates inflation)"

    interior = ~(rim | full_tread)
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

    print("\nSMOKE TEST PASSED: tire-sector torus geometry with superposed "
          "inflation + localized tread-load pressure solves cleanly, respects "
          "its own BC, deflects in the physically expected directions (bulges "
          "under pure inflation, net-inward under the added local load), and "
          "gives a genuinely non-degenerate 3D deformation field.")


if __name__ == "__main__":
    main()
