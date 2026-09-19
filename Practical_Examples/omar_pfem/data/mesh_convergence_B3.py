"""Mesh-convergence check for B3 (the rocking rubber-mount bushing WITH
the corrective fillet -- see data_generate_B3.py's own docstring for the
full design rationale and history). Run on CPU at small/cheap
resolutions first, per this project's own standing discipline, BEFORE
any data generation or training: Omar's own explicit instruction was
"do not start training until the mesh-convergence results show that
genuinely finer 3D resolution is required."

QoI: region-averaged Cauchy stress (sigma_xx) in a FIXED PHYSICAL region
centered on the fillet surface itself (not a fixed mesh index) -- the
one explicit, disclosed, finite-radius geometric feature this case's
own stress concentration is now attributed to, after the fillet fix.
Cauchy stress is derived from the (P, F) pair `.solve()` already returns
(flux=1st PK stress P, grad=deformation gradient F, both aggregated per
element) via sigma = (1/det F) * P @ F^T -- the SAME push-forward
formula this project's 2D code already uses (high_dof_convergence_
study.py's own _cauchy_stress_at), now on genuine 3x3 tensors.
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rocking_displacement_x)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d


def solve_case(Ntheta, Nr, Nz, R_in=0.5, R_out=1.0, Lz=1.0, r_fillet=0.1,
               E=1000.0, nu=0.45, delta0=None, dtype=torch.float64, device=None,
               verbose=False):
    device = device or torch.device("cpu")
    if delta0 is None:
        delta0 = 0.03 * (R_out - R_in)

    nodes, elements = generate_grid_hex8_bushing(R_in, R_out, Lz, Ntheta, Nr, Nz, r_fillet)
    inner, outer, sym = boundary_node_sets(nodes, R_in, R_out, Lz, r_fillet)

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)

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
    constraints[outer, :] = True
    constraints[inner, :] = True
    displacements[inner, 0] = torch.tensor(
        rocking_displacement_x(nodes[inner], Lz, delta0), dtype=dtype, device=device)
    constraints[sym, 1] = True
    model.constraints = constraints
    model.displacements = displacements

    increments = torch.linspace(0.0, 1.0, 11, dtype=dtype, device=device)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    t0 = time.time()
    try:
        with torch.device(device):
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", nlgeom=True, verbose=verbose)
    finally:
        torch.set_default_dtype(old_default_dtype)
    elapsed = time.time() - t0

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all(), "NaN/Inf in solution"

    detF = torch.linalg.det(F)
    sigma = torch.einsum("eij,ejk->eik", P, F.transpose(-1, -2)) / detF[:, None, None]
    sigma_np = sigma.cpu().numpy()

    # Fixed PHYSICAL region: a small ball centered on a representative
    # point on the fillet surface itself (not a mesh index) -- roughly
    # the point of highest curvature along the bottom fillet arc, near
    # theta=0 (the rocking/mirror-symmetry line, y~0, x>0), at 45 degrees
    # around the quarter-circle fillet profile.
    z_ref = r_fillet * (1 - np.cos(np.pi / 4))
    r_ref = (R_in + r_fillet) - np.sqrt(max(r_fillet ** 2 - (z_ref - r_fillet) ** 2, 0.0))
    ref_point = np.array([r_ref, 0.0, z_ref])
    region_radius = 2.0 * r_fillet  # fixed physical size, independent of mesh resolution
    # (large enough that even the coarsest resolution tested has at least
    # one element centroid inside it -- checked directly: 0.15 at the
    # coarsest (Ntheta=9,Nr=4,Nz=7) mesh, so 2*r_fillet=0.2 comfortably covers it)

    centroids = nodes[elements].mean(axis=1)
    dist = np.linalg.norm(centroids - ref_point[None, :], axis=1)
    region_mask = dist < region_radius
    n_region = int(region_mask.sum())

    sigma_xx_region = sigma_np[region_mask, 0, 0]
    region_avg_sigma_xx = float(sigma_xx_region.mean()) if n_region else float("nan")

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz,
        "n_nodes": n_nodes, "n_elements": len(elements), "n_region": n_region,
        "region_avg_sigma_xx": region_avg_sigma_xx,
        "max_ux": float(u_np[:, 0].max()),
        "elapsed_s": elapsed,
    }


def main():
    resolutions = [
        (9, 4, 7), (13, 6, 11), (17, 8, 15), (21, 10, 19), (25, 12, 23),
    ]
    print(f"{'Ntheta':>6} {'Nr':>4} {'Nz':>4} {'n_elem':>7} {'n_reg':>6} "
          f"{'avg_sigma_xx':>13} {'time_s':>8}")
    rows = []
    for Ntheta, Nr, Nz in resolutions:
        r = solve_case(Ntheta, Nr, Nz)
        rows.append(r)
        print(f"{r['Ntheta']:>6} {r['Nr']:>4} {r['Nz']:>4} {r['n_elements']:>7} "
              f"{r['n_region']:>6} {r['region_avg_sigma_xx']:>13.3f} {r['elapsed_s']:>8.2f}")

    print("\nRelative change in region_avg_sigma_xx between successive resolutions:")
    for a, b in zip(rows[:-1], rows[1:]):
        rel = abs(b["region_avg_sigma_xx"] - a["region_avg_sigma_xx"]) / abs(a["region_avg_sigma_xx"])
        print(f"  ({a['Ntheta']},{a['Nr']},{a['Nz']}) -> ({b['Ntheta']},{b['Nr']},{b['Nz']}): "
              f"{rel * 100:.2f}% change")

    print("\nIf this keeps changing meaningfully (not yet plateaued) across the "
          "resolutions tested, the fillet-region stress genuinely needs finer 3D "
          "mesh to resolve -- checked empirically here, not assumed. Per Omar's "
          "own explicit instruction: do not proceed to data generation/training "
          "until this shows real resolution-sensitivity.")


if __name__ == "__main__":
    main()
