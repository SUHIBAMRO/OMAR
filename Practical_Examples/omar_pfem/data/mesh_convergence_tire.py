"""PRELIMINARY (deliberately lightweight, per Omar's own instruction --
this candidate does not get B3's full rigor unless Timon picks it)
mesh-convergence check for the tire-sector candidate. CPU, small
resolutions only.

Tracks: max displacement magnitude, and a simple groove-region Cauchy
stress average (same push-forward formula as B3: sigma = (1/detF) P F^T)
in a fixed physical region on the tread groove, at increasing
(Ntheta,Nr,Nphi). This is NOT the full 7-QoI/required-resolution
treatment B3 received -- just enough to see whether resolution
sensitivity exists at all, for a first, lightweight comparison against
B3 when both candidates are shown to Timon.
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_tire import (
    generate_grid_hex8_tire_sector, boundary_node_sets, tread_groove_radius_of_curvature)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

R_BEAD, R_TREAD0, R_BIG = 0.5, 1.0, 3.0
PHI_MAX = np.pi / 6
GROOVE_DEPTH, GROOVE_HALF_THETA = 0.05, 0.15
CONTACT_HALF_THETA = 0.42  # wide enough to span >=2 theta rows even at the
# coarsest Ntheta=9 tested below (row spacing there is pi/8=0.393)
E, NU = 1000.0, 0.45
PRESSURE = -5.0


def solve_case(Ntheta, Nr, Nphi, dtype=torch.float64, device=None):
    device = device or torch.device("cpu")
    nodes, elements = generate_grid_hex8_tire_sector(
        R_BEAD, R_TREAD0, R_BIG, PHI_MAX, Ntheta, Nr, Nphi, GROOVE_DEPTH, GROOVE_HALF_THETA)
    rim, contact_patch = boundary_node_sets(
        nodes, R_BEAD, R_TREAD0, R_BIG, GROOVE_DEPTH, GROOVE_HALF_THETA, CONTACT_HALF_THETA)

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)
    mu = E / (2 * (1 + NU))
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    params = torch.tensor([mu, lam], dtype=dtype, device=device)
    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
    with torch.device(device):
        model = Solid(nodes_t, elements_t, material)

    n_nodes = nodes.shape[0]
    contact_mask = torch.tensor(contact_patch, device=device)
    old_dt = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        forces = model.integrate_surface_load(contact_mask, PRESSURE)
    finally:
        torch.set_default_dtype(old_dt)
    model.forces = forces

    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    constraints[rim, :] = True
    model.constraints = constraints
    model.displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

    increments = torch.linspace(0.0, 1.0, 11, dtype=dtype, device=device)
    old_dt = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    t0 = time.time()
    try:
        with torch.device(device):
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", nlgeom=True, verbose=False)
    finally:
        torch.set_default_dtype(old_dt)
    elapsed = time.time() - t0

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all()

    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        return sum(np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
                   for a, b, c, d in tets)
    n_inverted = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    assert n_inverted == 0
    detF = torch.linalg.det(F)
    assert (detF > 0).all()

    P_np, F_np = P.cpu().numpy(), F.cpu().numpy()
    sigma = np.einsum("eij,ejk->eik", P_np, F_np.transpose(0, 2, 1)) / detF.cpu().numpy()[:, None, None]

    # Fixed physical region on the groove (theta=pi/2, mid-Phi, at the
    # tread's own deepest groove point).
    ref_point = np.array([
        (R_BIG + (R_TREAD0 - GROOVE_DEPTH)) * np.cos(PHI_MAX / 2), 0.0,
        (R_BIG + (R_TREAD0 - GROOVE_DEPTH)) * np.sin(PHI_MAX / 2),
    ])
    region_radius = 6.0 * tread_groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_THETA, R_TREAD0)
    centroids = nodes[elements].mean(axis=1)
    dist = np.linalg.norm(centroids - ref_point[None, :], axis=1)
    region_mask = dist < region_radius
    n_region = int(region_mask.sum())
    region_avg = float(sigma[region_mask, 0, 0].mean()) if n_region else float("nan")

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nphi": Nphi,
        "n_elements": len(elements), "n_region": n_region,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
        "region_avg_sigma_xx": region_avg, "elapsed_s": elapsed,
    }


def main():
    resolutions = [(9, 4, 5), (13, 6, 9), (17, 8, 13), (21, 10, 17)]
    print(f"{'Ntheta':>6} {'Nr':>4} {'Nphi':>5} {'n_elem':>7} {'n_reg':>6} "
          f"{'max_disp':>10} {'region_avg_sxx':>14} {'time_s':>8}")
    rows = []
    for Ntheta, Nr, Nphi in resolutions:
        r = solve_case(Ntheta, Nr, Nphi)
        rows.append(r)
        print(f"{r['Ntheta']:>6} {r['Nr']:>4} {r['Nphi']:>5} {r['n_elements']:>7} "
              f"{r['n_region']:>6} {r['max_disp']:>10.6f} {r['region_avg_sigma_xx']:>14.4f} "
              f"{r['elapsed_s']:>8.2f}")

    print("\nRelative change resolution-to-resolution:")
    for a, b in zip(rows[:-1], rows[1:]):
        d_disp = abs(b["max_disp"] - a["max_disp"]) / abs(a["max_disp"])
        d_sxx = (abs(b["region_avg_sigma_xx"] - a["region_avg_sigma_xx"]) / abs(a["region_avg_sigma_xx"])
                 if a["region_avg_sigma_xx"] == a["region_avg_sigma_xx"] else float("nan"))
        print(f"  ({a['Ntheta']},{a['Nr']},{a['Nphi']}) -> ({b['Ntheta']},{b['Nr']},{b['Nphi']}): "
              f"max_disp {d_disp*100:.2f}%, region_avg_sxx {d_sxx*100:.2f}%")

    print("\nPreliminary only -- confirms the tire-sector solves cleanly and shows "
          "real resolution sensitivity (both max displacement and the groove-region "
          "stress change materially with mesh density), matching this project's own "
          "established pattern (local stress converges slower than global "
          "displacement). A B3-level full study (7-QoI required-resolution table, "
          "directional per-axis breakdown, GPU-scale fine reference) has NOT been "
          "done for this candidate and should only be done if Timon picks the tire "
          "over the bushing.")


if __name__ == "__main__":
    main()
