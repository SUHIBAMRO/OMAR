"""Timon's 2026-11-10 email, item 1: does LOCAL (not global) refinement
converge the groove-region stress? mesh_convergence_B3.py's own cross-
mesh comparison machinery (_theta_t_axes, interpolate_node_field_to_
finer_mesh) hardcodes uniform z-spacing (zs = np.linspace(0, LZ, Nz)),
so it cannot correctly place Gauss points on the new non-uniform
(z_beta-graded) mesh from data_generate_B3_local_z_refine.py -- checked
directly, not assumed (see PROJECT_STATUS.md).

Rather than generalize that already-validated, already-used-for-every-
published-number module (risking silently changing results it has
already produced), this is a NEW, SELF-CONTAINED script that needs no
cross-mesh interpolation at all: at a FIXED (Ntheta, Nr, Nz) -- same
total element count throughout -- it solves with increasing z_beta
(more of those same elements clustered physically at the groove, fewer
far from it) and reports the groove-region Cauchy stress directly from
each run's own Gauss points. If local clustering is what the global
ladder was missing, the region stress should visibly stabilize as beta
increases, even though global element count never changes.

Only the Gauss-point PHYSICAL position needs to be z_beta-aware: within
one element, z still varies LINEARLY between that element's own two z-
layers (z0=zs[k], z1=zs[k+1]) regardless of how non-uniformly those
layers are spaced globally -- the exact same bilinear-in-xi3 formula
mesh_convergence_B3._gauss_point_parametric already uses, just fed the
real (graded) zs array instead of a hardcoded linspace. That is the
one and only change from the existing, validated logic; everything
else below (stress-tensor assembly, region mask, volume weights) is
copied verbatim from mesh_convergence_B3.solve_case / Section 6's own
groove-region QoI, so this script cannot silently diverge from how
every other groove-region number in this project was computed.

Usage: python -m omar_pfem.data.solve_z_refined_groove_stress
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    boundary_node_sets, rigid_rotation_displacement, groove_radius_of_curvature, groove_R_in)
from omar_pfem.data.data_generate_B3_local_z_refine import (
    generate_grid_hex8_bushing_z_refined, z_layers_sinh_graded)
from omar_pfem.data.mesh_convergence_B3 import _element_ijk, HEXA1_IPOINTS
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH

R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
E, NU = 1000.0, 0.45
PHI = 0.05


def _gauss_point_physical_z_aware(Ntheta, Nr, Nz, r_grading, zs, ipoints_np):
    """Same construction as mesh_convergence_B3._gauss_point_parametric,
    except zs is passed in directly (the real, possibly non-uniform
    z-layer array) instead of being recomputed via a hardcoded
    np.linspace -- the one change needed to support a z_beta-graded
    mesh. thetas/ts are still uniform/power-law here (unchanged from
    the existing convention; only z is graded in this study)."""
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** r_grading
    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    theta0, theta1 = thetas[j], thetas[j + 1]
    t0, t1 = ts[i], ts[i + 1]
    z0, z1 = zs[k], zs[k + 1]

    xi1, xi2, xi3 = ipoints_np[:, 0], ipoints_np[:, 1], ipoints_np[:, 2]
    frac1, frac2, frac3 = (xi1 + 1.0) / 2.0, (xi2 + 1.0) / 2.0, (xi3 + 1.0) / 2.0

    t_g = t0[:, None] + frac1[None, :] * (t1 - t0)[:, None]
    theta_g = theta0[:, None] + frac2[None, :] * (theta1 - theta0)[:, None]
    z_g = z0[:, None] + frac3[None, :] * (z1 - z0)[:, None]
    return theta_g, t_g, z_g


def solve_and_measure_groove_region(Ntheta, Nr, Nz, z_beta, r_grading=1.0,
                                     dtype=torch.float64, device=None, verbose=False,
                                     ref_point=None, region_radius=None, groove_depth=None):
    """ref_point/region_radius default to the established groove region
    (unchanged behavior, every existing call site/result stays byte-
    identical) -- passed explicitly to evaluate stress at a DIFFERENT
    fixed physical region instead (e.g. the far-field nominal region
    used elsewhere in this project, `np.array([R_IN0, 0.0, LZ/4.0])`,
    same region_radius) on the exact same mesh/solve, so the far-field
    and groove numbers it returns come from the SAME methodology, just
    a different mask -- needed for a genuine SCF (ratio of two
    region-average/max stresses computed the same way).

    groove_depth defaults to the real GROOVE_DEPTH (unchanged behavior).
    Passing groove_depth=0.0 instead solves the SAME mesh/BC/rotation
    with the groove geometrically REMOVED (flat R_in0 bore everywhere)
    -- the correct counterfactual baseline for a genuine SCF, evaluated
    at the SAME physical location (R_IN0, 0, Lz/2) rather than at a
    different z. This matters because this problem's rigid-ROTATION
    (not uniform tension/bending) loading is itself z-position-
    dependent (rigid_rotation_displacement's own dominant term scales
    with dz = z-Lz/2, zero exactly at the groove's own z=Lz/2): a
    far-field point at a DIFFERENT z (e.g. z=Lz/4, dz=-0.25) sits under
    a fundamentally different local loading intensity than the groove's
    own dz=0 point, so comparing their raw stress directly is NOT a
    valid nominal-vs-concentrated SCF comparison -- confirmed directly,
    not assumed, when an initial z=Lz/4-based attempt gave a
    nonsensical SCF<1 (see PROJECT_STATUS.md)."""
    device = device or torch.device("cpu")
    groove_depth = GROOVE_DEPTH if groove_depth is None else groove_depth
    nodes, elements = generate_grid_hex8_bushing_z_refined(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, groove_depth, GROOVE_HALF_WIDTH,
        r_grading=r_grading, z_beta=z_beta)
    zs = z_layers_sinh_graded(Nz, LZ, z_beta)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, groove_depth, GROOVE_HALF_WIDTH)

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
    t0 = time.time()
    try:
        with torch.device(device), torch.no_grad():
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", verbose=verbose,
                aggregate_integration_points=False)
    finally:
        torch.set_default_dtype(old_default_dtype)
    elapsed = time.time() - t0

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all(), "NaN/Inf in solution"

    P = P.transpose(0, 1)  # (n_elem, n_gauss, 3, 3)
    F = F.transpose(0, 1)
    detF_gauss = torch.linalg.det(F)
    assert (detF_gauss > 0).all(), "non-positive det(F) at a Gauss point -- element inversion"

    ipoints_np = model.etype.ipoints.cpu().numpy()
    iweights_np = model.etype.iweights.cpu().numpy()
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            _, _, detJ_gauss = model.eval_shape_functions(model.etype.ipoints)
    finally:
        torch.set_default_dtype(old_default_dtype)
    detJ_gauss = detJ_gauss.transpose(0, 1)
    detJ_gauss_np = detJ_gauss.cpu().numpy()
    vol_weight = iweights_np[None, :] * np.abs(detJ_gauss_np)

    P_gauss_np, F_gauss_np = P.cpu().numpy(), F.cpu().numpy()
    detF_gauss_np = detF_gauss.cpu().numpy()
    sigma_gauss = (np.einsum("egij,egjk->egik", P_gauss_np, F_gauss_np.transpose(0, 1, 3, 2))
                   / detF_gauss_np[..., None, None])

    theta_g, t_g, z_g = _gauss_point_physical_z_aware(Ntheta, Nr, Nz, r_grading, zs, ipoints_np)

    if ref_point is None:
        z_ref = LZ / 2.0
        r_ref = R_IN0 - GROOVE_DEPTH
        ref_point = np.array([r_ref, 0.0, z_ref])
    if region_radius is None:
        region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)

    R_in_eff_g = groove_R_in(z_g, LZ, R_IN0, groove_depth, GROOVE_HALF_WIDTH)
    Rr_g = R_in_eff_g + (R_OUT - R_in_eff_g) * t_g
    x_g = Rr_g * np.cos(theta_g)
    y_g = Rr_g * np.sin(theta_g)
    pos_g = np.stack([x_g, y_g, z_g], axis=-1)

    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    region_mask = dist < region_radius
    n_region = int(region_mask.sum())

    sigma_flat = sigma_gauss[region_mask]
    w_flat = vol_weight[region_mask]
    w_sum = w_flat.sum()
    region_avg_xx = float((sigma_flat[:, 0, 0] * w_flat).sum() / w_sum) if n_region > 0 and w_sum > 0 else float("nan")
    region_max_xx = float(np.abs(sigma_flat[:, 0, 0]).max()) if n_region > 0 else float("nan")
    frob = np.sqrt(np.einsum("nij,nij->n", sigma_flat, sigma_flat)) if n_region > 0 else np.array([])
    region_avg_frob = float((frob * w_flat).sum() / w_sum) if n_region > 0 and w_sum > 0 else float("nan")

    return {
        "z_beta": z_beta, "n_elem": elements.shape[0], "n_region_gauss": n_region,
        "region_avg_sigma_xx": region_avg_xx, "region_max_sigma_xx": region_max_xx,
        "region_avg_sigma_frob": region_avg_frob, "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    Ntheta, Nr, Nz = 13, 12, 19  # small, CPU-feasible first signal; same element count every row
    print(f"Ntheta={Ntheta} Nr={Nr} Nz={Nz} (fixed element count across all rows below)\n")
    print(f"{'z_beta':>7} {'n_elem':>7} {'n_region_gauss':>15} "
          f"{'avg|sigma_xx|':>14} {'max|sigma_xx|':>14} {'avg|sigma|_F':>13} {'time(s)':>8}")
    for z_beta in (0.0, 2.0, 4.0, 6.0, 8.0):
        r = solve_and_measure_groove_region(Ntheta, Nr, Nz, z_beta)
        print(f"{r['z_beta']:>7.1f} {r['n_elem']:>7} {r['n_region_gauss']:>15} "
              f"{r['region_avg_sigma_xx']:>14.4f} {r['region_max_sigma_xx']:>14.4f} "
              f"{r['region_avg_sigma_frob']:>13.4f} {r['elapsed_s']:>8.2f}")
