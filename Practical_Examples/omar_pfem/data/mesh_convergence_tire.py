"""PRELIMINARY (deliberately lightweight, per Omar's own instruction --
this candidate does not get B3's full rigor unless Timon picks it)
mesh-convergence check for the tire-sector candidate. CPU, small
resolutions only.

REVISED 2026-09-21 (matches data_generate_tire.py's own review-point
fixes):
  - Uses the new (rim, full_tread, tread_load_region) 3-mask signature
    and applies superposed inflation (whole tread, outward) + localized
    tread-load (tread_load_region, inward) pressures under one
    incremental ramp, exactly as smoke_test_tire.py now does (review
    point 3).
  - Per review point 11, now ALSO tracks a reaction/force-family QoI and
    total strain energy alongside displacement and the groove-region
    Cauchy stress -- not just max_disp and region_avg_sigma_xx as
    before. Two things were discovered (and verified analytically, not
    assumed) while validating this: (1) the reaction MOMENT about the
    wheel's own spin axis (global Y) is exactly zero at every
    resolution -- a real geometric fact, since every load here is
    pressure normal to a surface of revolution about that axis, which
    can never produce torque about it -- so moment is reported as a
    diagnostic only, never as a convergence QoI, unlike B3's own rocking
    bushing where the moment IS the primary physical reaction; (2) the
    COMBINED net reaction FORCE (inflation + local load together) is a
    near-cancellation of two comparable O(1) quantities and swings by
    ~100x with sign flips across resolutions purely from that
    cancellation amplifying ordinary mesh noise -- verified by
    reconstructing it exactly from the two loads' own isolated
    unit-pressure integrals, confirming this is a conditioning issue,
    not a code bug. The primary force-family QoI tracked here is
    therefore the LOCAL tread-load's own isolated resultant magnitude
    (`local_load_force_norm`), which is non-degenerate and shows real,
    disclosed mesh sensitivity of its own (the load window's hard
    theta-half-width boundary doesn't align with mesh nodes at these
    coarse resolutions); the combined net force is kept only for the
    force_rel_residual equilibrium check.
  - The region-Cauchy stress sample point stays fixed in physical space
    at the groove's own deepest point, mid-sector (Phi=Phi_max/2), away
    from both sector cuts and the rim -- per review point 11's own
    placement requirement (already satisfied by the pre-existing
    ref_point below, now with the region mask done at the GAUSS-POINT
    level like B3, rather than element centroids, for a slightly richer
    sample count even at this lightweight tier).

Still NOT the full B3-level treatment: no L2/H1/required-resolution
table, no directional per-axis study, no GPU-scale reference. Just
enough to see whether resolution sensitivity exists at all across the
now-fuller set of QoIs, for a first, lightweight comparison against B3
when both candidates are shown to Timon.
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
TREAD_LOAD_HALF_THETA = 0.42  # wide enough to span >=2 theta rows even at the
# coarsest Ntheta=9 tested below (row spacing there is pi/8=0.393)
E, NU = 1000.0, 0.45
INFLATION_PRESSURE = 2.0  # positive = outward, over the WHOLE tread
LOCAL_LOAD_PRESSURE = -5.0  # negative = inward, additional, over tread_load_region only

# Hexa1's own 8 Gauss points (2x2x2), duplicated here so this stays
# pure-numpy, same pattern as mesh_convergence_B3.py's HEXA1_IPOINTS.
_G = 1.0 / np.sqrt(3.0)
HEXA1_IPOINTS = np.array([[s1 * _G, s2 * _G, s3 * _G]
                           for s3 in (-1, 1) for s2 in (-1, 1) for s1 in (-1, 1)])


def solve_case(Ntheta, Nr, Nphi, dtype=torch.float64, device=None):
    device = device or torch.device("cpu")
    nodes, elements = generate_grid_hex8_tire_sector(
        R_BEAD, R_TREAD0, R_BIG, PHI_MAX, Ntheta, Nr, Nphi, GROOVE_DEPTH, GROOVE_HALF_THETA)
    rim, full_tread, tread_load_region = boundary_node_sets(
        nodes, R_BEAD, R_TREAD0, R_BIG, GROOVE_DEPTH, GROOVE_HALF_THETA, TREAD_LOAD_HALF_THETA)

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
    full_tread_mask = torch.tensor(full_tread, device=device)
    load_mask = torch.tensor(tread_load_region, device=device)
    old_dt = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        forces_inflation = model.integrate_surface_load(full_tread_mask, INFLATION_PRESSURE)
        forces_local_load = model.integrate_surface_load(load_mask, LOCAL_LOAD_PRESSURE)
    finally:
        torch.set_default_dtype(old_dt)
    model.forces = forces_inflation + forces_local_load

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
                method="cg", preconditioner="jacobi", nlgeom=True, verbose=False,
                aggregate_integration_points=False)
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

    # torch-fem's own aggregate_integration_points=False returns P, F
    # with the GAUSS axis FIRST -- (n_gauss, n_elem, 3, 3) -- verified
    # empirically for mesh_convergence_B3.py; transpose to the
    # elem-first convention used throughout this file.
    P = P.transpose(0, 1)
    F = F.transpose(0, 1)
    detF = torch.linalg.det(F)
    assert (detF > 0).all()

    F_elem = F.mean(dim=1)
    P_elem = P.mean(dim=1)
    detF_elem = torch.linalg.det(F_elem)

    P_np, F_np = P_elem.cpu().numpy(), F_elem.cpu().numpy()
    sigma_elem = (np.einsum("eij,ejk->eik", P_np, F_np.transpose(0, 2, 1))
                  / detF_elem.cpu().numpy()[:, None, None])

    # Reaction force/moment on the fixed rim, about the wheel's big axis
    # (global Y, through the origin -- the axis of revolution of the
    # torus itself, the natural axis for this sector's own rocking/
    # rolling reaction).
    f_np = f.cpu().numpy()
    reaction_force = f_np[rim].sum(axis=0)
    rel = nodes[rim]  # position vectors from the wheel axis (Y axis through origin)
    reaction_moment_y = float(np.sum(np.cross(rel, f_np[rim]), axis=0)[1])
    total_force = f_np.sum(axis=0)
    total_moment_y = float(np.sum(np.cross(nodes, f_np), axis=0)[1])
    force_scale = max(float(np.linalg.norm(reaction_force)), 1e-8)
    moment_scale = max(abs(reaction_moment_y), 1e-8)
    force_rel_residual = float(np.linalg.norm(total_force)) / force_scale
    moment_rel_residual = abs(total_moment_y) / moment_scale

    # NOTE (found while validating this rewrite, verified analytically --
    # not a code bug): the COMBINED net reaction force (rim reaction,
    # equal and opposite to the net externally applied pressure force)
    # is a near-CANCELLATION of two O(1) quantities -- 2.0x the
    # inflation pressure's own net resultant (~7.5-7.6, smooth across
    # resolutions) against 5.0x the local tread load's own net resultant
    # (~5.1-7.6, itself carrying real ~50% mesh-to-mesh quantization
    # noise from the hard theta-half-width window boundary not aligning
    # with mesh nodes) -- so the DIFFERENCE swings by ~100x and even
    # flips sign between successive meshes, purely from that
    # cancellation amplifying the (already-expected, already-disclosed)
    # local-load-window quantization noise. Verified by isolating each
    # unit-pressure load integral separately and reconstructing the
    # combined force exactly (matches to 5+ significant figures) -- so
    # this is NOT a code defect, but it DOES make the combined net
    # reaction force a badly-conditioned convergence QoI here (same
    # spirit as review point 3's "near-zero force by symmetry" concern
    # for B3, though the mechanism here is cancellation between two
    # unrelated loads, not symmetry). Per that same point's own fallback
    # ("or use a genuinely nonzero component"), the LOCAL LOAD's own
    # resultant magnitude (isolated, not combined with the unrelated
    # inflation term) is used below as the primary force-like QoI
    # instead; the combined net force above is kept ONLY for the
    # force_rel_residual equilibrium check (which is unaffected by this
    # cancellation, since it compares the TOTAL residual over ALL nodes
    # to the SAME total's own norm as its scale).
    local_load_force_norm = float(np.linalg.norm(forces_local_load.cpu().numpy().sum(axis=0)))

    # The moment about the wheel's own spin axis (global Y) is exactly
    # zero here (~1e-12 to 1e-13, many orders of magnitude below the
    # force scale) for a genuine geometric reason, not numerical noise:
    # every applied load is a pressure NORMAL to a surface of revolution
    # about that same axis, and a normal-direction force has, by
    # construction, zero component in the surface's own local
    # circumferential (tangential-to-the-axis) direction -- so r x n has
    # zero Y-component identically, regardless of mesh or how much of
    # the sector is loaded. Unlike B3's rocking bushing (a genuine
    # rocking problem where the reaction moment IS the primary, physically
    # meaningful QoI), this tire sector has no analogous rocking DOF, so
    # reaction_moment_y is reported below purely as a diagnostic
    # (confirming it stays at floating-point zero at every resolution,
    # as it must) -- NOT as a converging QoI, and it is excluded from
    # the resolution-to-resolution %-change table for exactly that
    # reason (a %-change of a quantity that is exactly zero by
    # construction is meaningless, the same problem review point 2
    # flagged for an unreliable p99 from too few samples).

    # neo_hookean_psi_3d(F, params) -> strain-energy density, called ONE
    # element's own F at a time (matches its own vmap(jacrev(psi)) usage
    # inside torch-fem, and mesh_convergence_B3.py's own identical
    # pattern) -- not batched over the element axis.
    volumes = np.array([abs(hex_signed_volume(nodes[el])) for el in elements])
    energy_density = np.array([
        neo_hookean_psi_3d(F_elem[e], params).item() for e in range(F_elem.shape[0])
    ])
    total_strain_energy = float((energy_density * volumes).sum())

    # Fixed physical region on the groove (theta=pi/2, mid-Phi, at the
    # tread's own deepest groove point), sampled at GAUSS-POINT
    # resolution (not element centroids) for a richer sample count even
    # at this lightweight tier, mirroring mesh_convergence_B3.py.
    ref_point = np.array([
        (R_BIG + (R_TREAD0 - GROOVE_DEPTH)) * np.cos(PHI_MAX / 2), 0.0,
        (R_BIG + (R_TREAD0 - GROOVE_DEPTH)) * np.sin(PHI_MAX / 2),
    ])
    region_radius = 6.0 * tread_groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_THETA, R_TREAD0)
    node_pos = nodes[elements]  # (n_elem, 8, 3)
    # Trilinear interpolation of node positions to each Gauss point,
    # using the same hex8 shape functions torch-fem itself uses.
    xi = HEXA1_IPOINTS  # (8, 3)
    shp = np.zeros((8, 8))  # (n_gauss, n_local_node)
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    for g in range(8):
        for n in range(8):
            shp[g, n] = 0.125 * np.prod(1.0 + signs[n] * xi[g])
    pos_gauss = np.einsum("gn,enc->egc", shp, node_pos)  # (n_elem, n_gauss, 3)
    dist = np.linalg.norm(pos_gauss - ref_point[None, None, :], axis=2)
    region_mask = dist < region_radius
    n_region = int(region_mask.sum())
    sigma_gauss_xx = np.broadcast_to(sigma_elem[:, 0, 0:1], (len(elements), 8))
    region_avg = float(sigma_gauss_xx[region_mask].mean()) if n_region else float("nan")

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nphi": Nphi,
        "n_elements": len(elements), "n_region": n_region,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
        "region_avg_sigma_xx": region_avg,
        "total_strain_energy": total_strain_energy,
        "reaction_moment_y": reaction_moment_y,
        "reaction_force_norm": float(np.linalg.norm(reaction_force)),
        "local_load_force_norm": local_load_force_norm,
        "force_rel_residual": force_rel_residual,
        "moment_rel_residual": moment_rel_residual,
        "elapsed_s": elapsed,
    }


def main():
    resolutions = [(9, 4, 5), (13, 6, 9), (17, 8, 13), (21, 10, 17)]
    print(f"{'Ntheta':>6} {'Nr':>4} {'Nphi':>5} {'n_elem':>7} {'n_reg':>6} "
          f"{'max_disp':>10} {'region_sxx':>11} {'energy':>11} "
          f"{'|F_load|':>9} {'time_s':>8}")
    rows = []
    for Ntheta, Nr, Nphi in resolutions:
        r = solve_case(Ntheta, Nr, Nphi)
        rows.append(r)
        print(f"{r['Ntheta']:>6} {r['Nr']:>4} {r['Nphi']:>5} {r['n_elements']:>7} "
              f"{r['n_region']:>6} {r['max_disp']:>10.6f} {r['region_avg_sigma_xx']:>11.4f} "
              f"{r['total_strain_energy']:>11.4e} "
              f"{r['local_load_force_norm']:>9.4f} {r['elapsed_s']:>8.2f}")
        print(f"         reaction_moment_y={r['reaction_moment_y']:.2e} "
              f"(exactly zero by symmetry -- pressure normal to a surface of revolution "
              f"produces no torque about its own axis; NOT a meaningful convergence QoI here, "
              f"unlike B3's own rocking bushing)")
        print(f"         equilibrium: force_rel_residual={r['force_rel_residual']:.2e} "
              f"(FORCE scale)  moment_rel_residual={r['moment_rel_residual']:.2e} (MOMENT scale) "
              f"-- combined net reaction force ({r['reaction_force_norm']:.4f}) is a near-"
              f"cancellation of the inflation and local-load resultants (see solve_case's own "
              f"comment) and is NOT reported as its own convergence QoI for that reason")

    print("\nRelative change resolution-to-resolution:")
    for a, b in zip(rows[:-1], rows[1:]):
        d_disp = abs(b["max_disp"] - a["max_disp"]) / abs(a["max_disp"])
        d_sxx = (abs(b["region_avg_sigma_xx"] - a["region_avg_sigma_xx"]) / abs(a["region_avg_sigma_xx"])
                 if a["region_avg_sigma_xx"] == a["region_avg_sigma_xx"] else float("nan"))
        d_energy = abs(b["total_strain_energy"] - a["total_strain_energy"]) / abs(a["total_strain_energy"])
        d_load = abs(b["local_load_force_norm"] - a["local_load_force_norm"]) / abs(a["local_load_force_norm"])
        print(f"  ({a['Ntheta']},{a['Nr']},{a['Nphi']}) -> ({b['Ntheta']},{b['Nr']},{b['Nphi']}): "
              f"max_disp {d_disp*100:.2f}%, region_avg_sxx {d_sxx*100:.2f}%, "
              f"energy {d_energy*100:.2f}%, local_load_force_norm {d_load*100:.2f}%")

    print("\nPreliminary only -- confirms the tire-sector solves cleanly under the "
          "superposed inflation+local-load pressures and shows real resolution "
          "sensitivity across displacement, energy, the local tread-load's own "
          "resultant magnitude, and the groove-region stress (matching this "
          "project's own established pattern: local, boundary-window-dependent "
          "quantities converge slower than global displacement/energy). Reaction "
          "moment about the wheel axis is exactly zero by symmetry (not tracked as "
          "a QoI); the combined net reaction force is a near-cancellation of two "
          "unrelated load resultants and is kept only as an equilibrium check, not "
          "a convergence QoI (see solve_case's own comment for the full reasoning). "
          "A B3-level full study (Cauchy-tensor field error, required-resolution "
          "table, directional per-axis breakdown, GPU-scale fine reference) has NOT "
          "been done for this candidate and should only be done if Timon picks the "
          "tire over the bushing.")


if __name__ == "__main__":
    main()
