"""A careful external review's sharpest point on the Q20 (quadratic-
element) groove-stress check: showing that each mesh's own REGION
max/average happens to land in a similar range is NOT the same as
showing the underlying STRESS FIELD has converged at the SAME physical
locations -- the single extremal Gauss point could be a different
point each time, especially since Q20's own Gauss-point layout differs
from Q8's. This script closes that gap directly: it extracts the
stress along the SAME inner-wall (theta~0) contour used by the
original (linear-element, 839,040-vs-950,400) contour-plot script,
now across the three already-tested quadratic resolutions
(2,376 / 5,760 / 10,200 elements), interpolates each onto a common
z-grid, and reports the actual pointwise relative difference between
resolutions -- a genuine field-convergence check, not a convergence
check of a single summary statistic.

Reuses build_quadratic_mesh_with_curved_surfaces (same curved-surface-
snapped Q20 mesh already validated) and _element_ijk from
mesh_convergence_B3 (same (j==0)&(i==0) inner-wall/first-angular-ring
selection as the original linear contour script -- valid unchanged for
the Q20 mesh since linear_to_quadratic preserves element COUNT and
ORDER, only adding nodes, confirmed directly when that mesh was built).

Usage: python -m omar_pfem.data.plot_quadratic_contour_convergence --out_dir <dir>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import rigid_rotation_displacement, groove_radius_of_curvature
from omar_pfem.data.mesh_convergence_B3 import _element_ijk
from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    build_quadratic_mesh_with_curved_surfaces, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH, E, NU, PHI)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

STRESS_COMPONENTS = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]


def quadratic_inner_wall_contour_sigma(Ntheta, Nr, Nz, r_grading=1.0, dtype=torch.float64, device=None):
    device = device or torch.device("cpu")
    nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces(
        Ntheta, Nr, Nz, r_grading=r_grading)

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
    model.forces = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    constraints[outer, :] = True
    ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, PHI)
    constraints[inner, :] = True
    displacements[inner, 0] = torch.tensor(ux, dtype=dtype, device=device)
    displacements[inner, 1] = torch.tensor(uy, dtype=dtype, device=device)
    displacements[inner, 2] = torch.tensor(uz, dtype=dtype, device=device)
    constraints[sym, 1] = True
    model.constraints = constraints
    model.displacements = displacements

    increments = torch.linspace(0.0, 1.0, 11, dtype=dtype, device=device)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device), torch.no_grad():
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", verbose=False,
                aggregate_integration_points=False)
    finally:
        torch.set_default_dtype(old_default_dtype)

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all(), "NaN/Inf in solution"
    P = P.transpose(0, 1)
    F = F.transpose(0, 1)
    detF_gauss = torch.linalg.det(F)
    assert (detF_gauss > 0).all(), "non-positive det(F) -- element inversion"

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            N_gauss, _, _ = model.eval_shape_functions(model.etype.ipoints)
    finally:
        torch.set_default_dtype(old_default_dtype)

    P_np, F_np = P.cpu().numpy(), F.cpu().numpy()
    detF_np = detF_gauss.cpu().numpy()
    sigma_gauss = (np.einsum("egij,egjk->egik", P_np, F_np.transpose(0, 1, 3, 2))
                   / detF_np[..., None, None])  # (n_elem, n_gauss, 3, 3)

    nodes_per_elem = nodes_t[elements_t]
    pos_g = torch.einsum("gn,enD->egD", N_gauss, nodes_per_elem).cpu().numpy()  # (n_elem, n_gauss, 3)
    z_g = pos_g[..., 2]

    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    contour_elem_mask = (j == 0) & (i == 0)
    sigma_contour = sigma_gauss[contour_elem_mask]  # (n_contour_elem, n_gauss, 3, 3)
    z_contour = z_g[contour_elem_mask]

    z_flat = z_contour.reshape(-1)
    order = np.argsort(z_flat)
    z_sorted = z_flat[order]
    out = {}
    for name, a, b in STRESS_COMPONENTS:
        out[name] = sigma_contour[:, :, a, b].reshape(-1)[order]
    return z_sorted, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[13, 12, 19, 17, 16, 25, 21, 18, 31],
                     help="flattened list of Ntheta,Nr,Nz triples")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    triples = [tuple(args.resolutions[i:i + 3]) for i in range(0, len(args.resolutions), 3)]
    results = {}
    for Ntheta, Nr, Nz in triples:
        print(f"solving Ntheta={Ntheta} Nr={Nr} Nz={Nz} ...", flush=True)
        z, sigma = quadratic_inner_wall_contour_sigma(Ntheta, Nr, Nz)
        n_elem = (Ntheta - 1) * (Nr - 1) * (Nz - 1)
        results[n_elem] = (z, sigma)
        print(f"  done, n_elem={n_elem}, n_contour_gauss={len(z)}", flush=True)

    zc = LZ / 2.0
    band_lo, band_hi = zc - GROOVE_HALF_WIDTH, zc + GROOVE_HALF_WIDTH
    rho = groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    region_radius = 2.0 * rho

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    n_elems_sorted = sorted(results.keys())
    for ax, (name, _, _) in zip(axes.ravel(), STRESS_COMPONENTS):
        for n_elem in n_elems_sorted:
            z, sigma = results[n_elem]
            ax.plot(z, sigma[name], "o-", ms=2, label=f"{n_elem:,} elements (Q20)", alpha=0.75)
        ax.axvline(zc, color="k", linestyle=":", linewidth=1)
        ax.axvline(band_lo, color="r", linestyle="--", linewidth=1)
        ax.axvline(band_hi, color="r", linestyle="--", linewidth=1)
        ax.axvspan(zc - region_radius, zc + region_radius, color="gray", alpha=0.15)
        ax.set_title(f"sigma_{name}")
        ax.set_xlabel("z")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("B3 inner-wall (theta~0) stress along z, Q20 elements at 3 resolutions -- "
                 "field-level convergence check (not just region max/avg)")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "quadratic_contour_convergence.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    print("\n=== Pointwise field convergence: finest two resolutions, interpolated onto a common grid ===")
    n_coarse, n_fine = n_elems_sorted[-2], n_elems_sorted[-1]
    z_coarse, sigma_coarse = results[n_coarse]
    z_fine, sigma_fine = results[n_fine]
    for name, _, _ in STRESS_COMPONENTS:
        interp = np.interp(z_fine, z_coarse, sigma_coarse[name])
        denom = np.abs(sigma_fine[name]).max() + 1e-12
        rel = np.abs(interp - sigma_fine[name]) / denom
        worst_z = z_fine[np.argmax(rel)]
        # also report the relative error restricted to the existing region-mask band,
        # the physically relevant zone for the groove-stress QoI itself
        in_region = np.abs(z_fine - zc) < region_radius
        region_rel = rel[in_region] if in_region.any() else np.array([np.nan])
        print(f"  sigma_{name}: whole-contour max pointwise rel. diff = {rel.max()*100:.1f}% at z={worst_z:.4f}  "
              f"| within existing region mask: max={np.nanmax(region_rel)*100:.1f}% "
              f"mean={np.nanmean(region_rel)*100:.1f}%")


if __name__ == "__main__":
    main()
