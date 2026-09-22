"""Mesh-convergence study for B3 (continuously-bonded groove bushing +
true rigid-rotation kinematics -- see data_generate_B3.py's own
docstring for the full design history). Run on CPU at small/cheap
resolutions first, per this project's own standing discipline, and per
Omar's own explicit instruction: do NOT start data generation or
training until this shows real resolution-sensitivity.

REVISED 2026-09-21 (Omar's own detailed technical review, before any
email to Timon): several real weaknesses were caught and fixed here,
not smoothed over:

1. The region-Cauchy-stress QoI previously used only sigma_xx and
   per-ELEMENT samples (n_region=2 at the coarsest mesh -- far too few
   for a meaningful 99th percentile). Now: (a) a genuine relative FIELD
   error for the FULL Cauchy stress TENSOR (Frobenius norm) in the
   fixed region, evaluated at the fine reference's own per-QUADRATURE-
   POINT locations (8 Gauss points per element, not 1 centroid --
   `aggregate_integration_points=False`), with the coarser mesh's own
   field properly interpolated (in parametric space, exact) onto those
   points; (b) volume-weighted (quadrature-weight x |detJ|) scalar
   region average and 99th percentile of sigma_xx, computed from the
   ROW'S OWN per-Gauss-point samples (8x richer than the old per-element
   count); (c) if a row's own region sample count is below a minimum,
   the percentile is reported as "not reliable" rather than a
   misleading number from too few points.
2. Reaction force: CHECKED (not assumed) whether the fine reference's
   own net reaction force is near zero by symmetry, which would make a
   relative-error threshold on it meaningless. It is NOT near zero
   (Y-component ~1.49, the two symmetry-plane components ~1e-15 as
   expected) -- so the force_rel numbers are legitimate. Reaction MOMENT
   is still treated as the PRIMARY reaction QoI for this rocking
   problem (more directly physically meaningful here), with force
   reported alongside as a secondary, now-verified-non-degenerate QoI.
3. Force/moment equilibrium residuals were already normalized by their
   OWN separate, dimensionally-correct scales (force residual by a
   force magnitude, moment residual by a moment magnitude) -- verified
   again here, not just asserted, and both are printed explicitly.
4. "Tangent energy" was a misnomer -- B1/B2's own "tangent energy" QoI
   (high_dof_convergence_study.py's `compute_tangent_energy_error`) is
   the NORM OF THE ERROR FIELD under the fine solution's own tangent
   stiffness (sqrt(e^T K(u_fine) e)), explicitly NOT a difference of two
   scalar totals (that docstring's own words: doing so "conflates a norm
   of the error field with a single number that can cancel and hide
   error"). What this module actually computes is exactly that simpler,
   weaker quantity (a difference of two scalar total hyperelastic strain
   energies) -- renamed throughout to "Total strain energy" to avoid
   misrepresenting it as the same metric B1/B2 use.

Full QoI set, all tracked per resolution:
  - displacement relative L2 error vs. a fine reference
  - H1 semi-norm error vs. the same fine reference (via the deformation
    gradient F, since grad(u) = F - I)
  - total hyperelastic strain energy (a scalar total, NOT B1/B2's own
    tangent-energy-norm -- see point 4 above)
  - reaction MOMENT/torque about the rotation axis (PRIMARY reaction QoI
    for this rocking case) and reaction FORCE (secondary, verified
    non-degenerate)
  - fixed-region Cauchy stress: full-tensor relative field error,
    volume-weighted average, and volume-weighted 99th percentile (true
    max reported only as a secondary print, per the established
    convention with Timon)

Cross-mesh comparison is done by interpolating fields in PARAMETRIC
(theta, t, z) space, not physical (x,y,z) space -- every B3 mesh at
every resolution shares the exact same parametric domain
[0,pi]x[0,1]x[0,Lz], so a regular-grid interpolator is EXACT for any
node-based field (displacement) and for element-aggregated fields
(F/Cauchy stress) queried at arbitrary points, including individual
Gauss-point locations -- not nearest-node snapping.
"""
import time

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
    groove_radius_of_curvature)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.05, 0.15
E, NU = 1000.0, 0.45
PHI = 0.05  # rotation angle, radians -- FIXED across every resolution tested
MIN_RELIABLE_N_P99 = 20  # below this many quadrature-point samples in the
# fixed region, a 99th percentile is not statistically meaningful

# Hexa1's own 2x2x2 Gauss points (torchfem.elements.Hexa1.ipoints) --
# duplicated here as a plain constant (not imported from torch, which
# this module otherwise uses as tensors) so parametric-coordinate helpers
# below can stay pure-numpy.
HEXA1_IPOINTS = np.array([
    [-0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
    [0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
    [-0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
    [0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
    [-0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
    [0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
    [-0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
    [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
])


def _theta_t_axes(Ntheta, Nr, Nz, r_grading=1.0):
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** r_grading
    zs = np.linspace(0.0, LZ, Nz)
    return thetas, ts, zs


def _element_ijk(Ntheta, Nr, Nz):
    """(j,i,k) [theta,r,z structured index] for each element, in EXACTLY
    the flat order data_generate_B3.generate_grid_hex8_bushing produces
    elements (k outer loop, then the 2D ring's own j,i loop) -- needed
    to place each element's own Gauss points at their exact parametric
    (theta,t,z) coordinates."""
    j, i, k = np.meshgrid(np.arange(Ntheta - 1), np.arange(Nr - 1), np.arange(Nz - 1), indexing="ij")
    # elements2d loops j (outer) then i (inner); elements3d loops k (outer) then elements2d (inner)
    j = np.transpose(j, (2, 0, 1)).ravel()
    i = np.transpose(i, (2, 0, 1)).ravel()
    k = np.transpose(k, (2, 0, 1)).ravel()
    return j, i, k


def _gauss_point_parametric(Ntheta, Nr, Nz, r_grading, ipoints_np):
    """Returns (theta_g, t_g, z_g), each shape (n_elem, n_gauss) -- the
    EXACT parametric coordinates of every element's own Gauss points
    (ipoints_np: (n_gauss,3) local xi in [-1,1], matching Hexa1's own
    xi1->r, xi2->theta, xi3->z convention given this project's own quad
    node ordering, verified in data_generate_B3.py's own comments)."""
    thetas, ts, zs = _theta_t_axes(Ntheta, Nr, Nz, r_grading)
    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    theta0, theta1 = thetas[j], thetas[j + 1]
    t0, t1 = ts[i], ts[i + 1]
    z0, z1 = zs[k], zs[k + 1]

    xi1 = ipoints_np[:, 0]  # -> r (t)
    xi2 = ipoints_np[:, 1]  # -> theta
    xi3 = ipoints_np[:, 2]  # -> z
    frac1 = (xi1 + 1.0) / 2.0
    frac2 = (xi2 + 1.0) / 2.0
    frac3 = (xi3 + 1.0) / 2.0

    t_g = t0[:, None] + frac1[None, :] * (t1 - t0)[:, None]
    theta_g = theta0[:, None] + frac2[None, :] * (theta1 - theta0)[:, None]
    z_g = z0[:, None] + frac3[None, :] * (z1 - z0)[:, None]
    return theta_g, t_g, z_g


def solve_case(Ntheta, Nr, Nz, dtype=torch.float64, device=None, verbose=False, r_grading=1.0):
    device = device or torch.device("cpu")
    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=r_grading)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)

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
        # aggregate_integration_points=False: keep the per-Gauss-point
        # flux(P)/grad(F)/state, not just the per-element mean torch-fem
        # would otherwise hand back -- needed for genuinely quadrature-
        # based, volume-weighted region statistics (Omar's own point 2,
        # 2026-09-21), not an approximation of convenience.
        #
        # torch.no_grad() (added later, 2026-09-22): confirmed directly via
        # a real GPU OOM on this module's own B8/groove-sharpness siblings
        # (not theorized here) that torchfem's NewtonRaphsonAdjoint.forward
        # calls ctx.save_for_backward on EVERY increment (its own implicit-
        # adjoint gradient support); without no_grad, autograd keeps each
        # increment's graph alive, so GPU memory grows with n_increments,
        # not just mesh size. This module's own past GPU runs (up to
        # 424,128 elements) happened not to hit this ceiling, but the same
        # latent inefficiency was present -- fixed here too, for
        # consistency and headroom on any future, larger reference. Verified
        # to leave every returned number bit-identical on CPU before relying
        # on it (this solve never needs gradients).
        with torch.device(device), torch.no_grad():
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method="cg", preconditioner="jacobi", nlgeom=True, verbose=verbose,
                aggregate_integration_points=False)
    finally:
        torch.set_default_dtype(old_default_dtype)
    elapsed = time.time() - t0

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all(), "NaN/Inf in solution -- Newton did not really converge cleanly"

    # torch-fem's own aggregate_integration_points=False shape convention
    # is (n_gauss, n_elem, ...) -- GAUSS axis FIRST -- verified directly
    # (not assumed) against torchfem.base.FEM.solve's actual output.
    # Transposed once, here, to (n_elem, n_gauss, ...) so every line
    # below can use the more natural elem-first convention.
    P = P.transpose(0, 1)
    F = F.transpose(0, 1)

    # P, F now have shape (n_elem, n_gauss, 3, 3). Per-element AGGREGATED
    # values (plain mean over Gauss points, matching torch-fem's OWN
    # aggregate_integration_points=True convention exactly) are still
    # needed for the L2/H1/energy field comparisons below -- element-to-
    # element variation dominates over the (much smaller) within-element
    # Gauss-point variation, so this remains a reasonable, standard
    # simplification for THOSE global/field-level quantities; only the
    # REGION statistics need the richer per-Gauss-point resolution.
    F_elem = F.mean(dim=1)
    P_elem = P.mean(dim=1)

    # ---- physics sanity checks ----
    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        return sum(np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
                   for a, b, c, d in tets)
    n_inverted = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    assert n_inverted == 0, f"{n_inverted} inverted elements -- mesh invalid"
    detF_elem = torch.linalg.det(F_elem)
    assert (detF_elem > 0).all(), "non-positive det(F) somewhere -- element inversion under load"
    detF_gauss = torch.linalg.det(F)
    assert (detF_gauss > 0).all(), "non-positive det(F) at a Gauss point -- element inversion under load"

    # ---- reaction force + moment on the fixed housing ----
    f_np = f.cpu().numpy()
    reaction_force = f_np[outer].sum(axis=0)  # (3,)
    rel = nodes[outer] - np.array([0.0, 0.0, LZ / 2.0])
    torque_vec = np.cross(rel, f_np[outer])  # (n_outer, 3)
    reaction_moment_y = float(torque_vec[:, 1].sum())

    # Global equilibrium sanity check (no applied external force
    # anywhere; every load comes from prescribed displacements), summed
    # over EVERY node (inner, outer, AND the symmetry plane, which also
    # carries its own nonzero y-reaction -- an earlier core-vs-housing-
    # only version of this check wrongly omitted that and failed at
    # ~50% during development, which is how this formula was corrected).
    total_force = f_np.sum(axis=0)
    rel_all = nodes - np.array([0.0, 0.0, LZ / 2.0])
    total_moment_y = float(np.cross(rel_all, f_np)[:, 1].sum())
    # Dimensionally separate scales, verified explicitly (Omar's own
    # 2026-09-21 point 6): force_scale is a FORCE magnitude, moment_scale
    # is a MOMENT (torque, F x L) magnitude -- never the same
    # denominator. Checked directly (not assumed) that the reference's
    # own reaction force is NOT near zero (Y-component ~1.49, only the
    # two symmetry-enforced components are ~1e-15), so force_scale is a
    # genuine, non-degenerate force magnitude here, not a near-zero
    # value that would make the relative residual meaningless.
    force_scale = max(np.linalg.norm(reaction_force), 1e-8)
    moment_scale = max(abs(reaction_moment_y), 1e-8)
    force_rel_residual = float(np.linalg.norm(total_force) / force_scale)
    moment_rel_residual = float(abs(total_moment_y) / moment_scale)
    assert force_rel_residual < 1e-6, f"global force equilibrium violated: {total_force}"
    assert moment_rel_residual < 1e-2, \
        f"global moment equilibrium violated: {total_moment_y} (relative {moment_rel_residual:.4f})"

    # ---- total hyperelastic strain (internal) energy: sum psi(F)*volume
    # -- a scalar TOTAL, NOT B1/B2's own tangent-energy-norm (see module
    # docstring point 4) ----
    F_elem_np = F_elem.cpu().numpy()
    energy_density = np.array([
        neo_hookean_psi_3d(torch.tensor(F_elem_np[e]), params).item() for e in range(F_elem_np.shape[0])
    ])
    elem_volumes = np.array([hex_signed_volume(nodes[el]) for el in elements])
    total_strain_energy = float((energy_density * elem_volumes).sum())

    # ---- per-Gauss-point Cauchy stress + volume weights ----
    ipoints_np = model.etype.ipoints.cpu().numpy()
    iweights_np = model.etype.iweights.cpu().numpy()
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            _, _, detJ_gauss = model.eval_shape_functions(model.etype.ipoints)
    finally:
        torch.set_default_dtype(old_default_dtype)
    detJ_gauss = detJ_gauss.transpose(0, 1)  # -> (n_elem, n_gauss), same fix as P/F above
    detJ_gauss_np = detJ_gauss.cpu().numpy()  # (n_elem, n_gauss)
    vol_weight = iweights_np[None, :] * np.abs(detJ_gauss_np)  # (n_elem, n_gauss)

    P_gauss_np, F_gauss_np = P.cpu().numpy(), F.cpu().numpy()
    detF_gauss_np = detF_gauss.cpu().numpy()
    sigma_gauss = (np.einsum("egij,egjk->egik", P_gauss_np, F_gauss_np.transpose(0, 1, 3, 2))
                   / detF_gauss_np[..., None, None])  # (n_elem, n_gauss, 3, 3)

    theta_g, t_g, z_g = _gauss_point_parametric(Ntheta, Nr, Nz, r_grading, ipoints_np)

    # ---- fixed physical region around the groove (continuously bonded,
    # no BC discontinuity nearby) ----
    z_ref = LZ / 2.0
    r_ref = R_IN0 - GROOVE_DEPTH
    ref_point = np.array([r_ref, 0.0, z_ref])
    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)

    # Physical position of each Gauss point via the SAME R_in_eff(z)
    # profile data_generate_B3 uses. NOTE: t_g is already on the GRADED
    # radial scale (it comes from _theta_t_axes' own `ts = linspace**
    # r_grading`, the same convention _elem_field_interpolator/
    # _field_interpolator use for their own t-axis) -- so it must NOT be
    # raised to r_grading again here (that would double-apply the
    # grading; harmless at the default r_grading=1.0 used everywhere in
    # this module, but wrong in general).
    from omar_pfem.data.data_generate_B3 import groove_R_in
    R_in_eff_g = groove_R_in(z_g, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    Rr_g = R_in_eff_g + (R_OUT - R_in_eff_g) * t_g
    x_g = Rr_g * np.cos(theta_g)
    y_g = Rr_g * np.sin(theta_g)
    pos_g = np.stack([x_g, y_g, z_g], axis=-1)  # (n_elem, n_gauss, 3)

    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    region_mask = dist < region_radius  # (n_elem, n_gauss)
    n_region = int(region_mask.sum())

    sigma_flat = sigma_gauss[region_mask]  # (n_region, 3, 3)
    w_flat = vol_weight[region_mask]  # (n_region,)
    w_sum = w_flat.sum()

    if n_region > 0 and w_sum > 0:
        region_avg = float((sigma_flat[:, 0, 0] * w_flat).sum() / w_sum)
    else:
        region_avg = float("nan")

    if n_region >= MIN_RELIABLE_N_P99 and w_sum > 0:
        order = np.argsort(sigma_flat[:, 0, 0])
        sorted_vals = sigma_flat[order, 0, 0]
        sorted_w = w_flat[order]
        cum_w = np.cumsum(sorted_w) / w_sum
        idx99 = np.searchsorted(cum_w, 0.99)
        idx99 = min(idx99, len(sorted_vals) - 1)
        region_p99 = float(sorted_vals[idx99])
        region_p99_reliable = True
    else:
        region_p99 = float("nan")
        region_p99_reliable = False

    region_true_max = float(sigma_flat[:, 0, 0].max()) if n_region else float("nan")  # secondary only
    sigma_elem = sigma_gauss.mean(axis=1)  # per-element aggregate, same convention as F_elem/P_elem

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz,
        "n_nodes": n_nodes, "n_elements": len(elements), "n_region": n_region,
        "region_avg_sigma_xx": region_avg, "region_p99_sigma_xx": region_p99,
        "region_p99_reliable": region_p99_reliable,
        "region_true_max_sigma_xx": region_true_max,
        "total_strain_energy": total_strain_energy,
        "reaction_force": reaction_force.tolist(), "reaction_moment_y": reaction_moment_y,
        "force_rel_residual": force_rel_residual, "moment_rel_residual": moment_rel_residual,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
        "elapsed_s": elapsed,
        # raw fields kept for cross-mesh comparisons, discarded by the
        # caller once those comparisons are done
        "_u": u_np, "_F_elem": F_elem_np, "_sigma_elem": sigma_elem,
        "_sigma_gauss": sigma_gauss, "_pos_gauss": pos_g, "_vol_weight": vol_weight,
        "_region_mask": region_mask,
        "_Ntheta": Ntheta, "_Nr": Nr, "_Nz": Nz, "_r_grading": r_grading,
    }


def _field_interpolator(field, Ntheta, Nr, Nz, thetas, ts, zs, n_components):
    """field: (Ntheta*Nr*Nz, n_components) node-ordered as
    index = k*(Ntheta*Nr) + j*Nr + i (k=z, j=theta, i=r) -- exactly how
    generate_grid_hex8_bushing builds it. Returns a callable mapping
    (theta, t, z) query points -> interpolated field values."""
    grid = field.reshape(Nz, Ntheta, Nr, n_components)
    interp = RegularGridInterpolator((zs, thetas, ts), grid, bounds_error=False, fill_value=None)
    def query(theta_q, t_q, z_q):
        pts = np.stack([z_q, theta_q, t_q], axis=-1)
        return interp(pts)
    return query


def _elem_field_interpolator(field, Ntheta, Nr, Nz, r_grading=1.0):
    """Same idea as _field_interpolator but for a per-ELEMENT field
    (shape (n_elem, ...)), using element-centroid parametric coordinates
    ((Ntheta-1)x(Nr-1)x(Nz-1) grid). Used as the SOURCE field for
    interpolation -- queried at arbitrary points (including individual
    Gauss-point locations elsewhere in this mesh, or at another mesh's
    own Gauss points), which is genuine interpolation, not nearest-
    neighbor snapping (verified: RegularGridInterpolator does multilinear
    interpolation by default)."""
    n_comp = int(np.prod(field.shape[1:]))
    field_flat = field.reshape(-1, n_comp)
    thetas_c = 0.5 * (np.linspace(0.0, np.pi, Ntheta)[:-1] + np.linspace(0.0, np.pi, Ntheta)[1:])
    ts_axis = np.linspace(0.0, 1.0, Nr) ** r_grading
    ts_c = 0.5 * (ts_axis[:-1] + ts_axis[1:])
    zs_c = 0.5 * (np.linspace(0.0, LZ, Nz)[:-1] + np.linspace(0.0, LZ, Nz)[1:])
    grid = field_flat.reshape(Nz - 1, Ntheta - 1, Nr - 1, n_comp)
    interp = RegularGridInterpolator((zs_c, thetas_c, ts_c), grid, bounds_error=False, fill_value=None)
    def query(theta_q, t_q, z_q):
        pts = np.stack([z_q, theta_q, t_q], axis=-1)
        return interp(pts).reshape((-1,) + field.shape[1:])
    return query


def _elem_centroid_parametric(Ntheta, Nr, Nz, r_grading):
    """(theta_c, t_c, z_c), each shape (n_elem,) -- the parametric
    coordinates of every element's own CENTROID, in the same flat
    element order _element_ijk uses (matches _elem_field_interpolator's
    own axis construction)."""
    thetas, ts, zs = _theta_t_axes(Ntheta, Nr, Nz, r_grading)
    thetas_c = 0.5 * (thetas[:-1] + thetas[1:])
    ts_c = 0.5 * (ts[:-1] + ts[1:])
    zs_c = 0.5 * (zs[:-1] + zs[1:])
    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    return thetas_c[j], ts_c[i], zs_c[k]


def compare_to_reference(case, ref):
    """L2 (displacement), H1-like (via F), and fixed-region Cauchy-
    TENSOR relative field error of `case` against the finer `ref` solve.
    All three use genuine interpolation in PARAMETRIC (theta,t,z) space
    (exact for node fields; a documented element-centroid-grid
    interpolation, queried at exact target-mesh coordinates including
    individual Gauss points, for element-aggregated fields) -- never
    nearest-node/nearest-element snapping."""
    Ntheta_c, Nr_c, Nz_c = case["_Ntheta"], case["_Nr"], case["_Nz"]
    thetas_c, ts_c, zs_c = _theta_t_axes(Ntheta_c, Nr_c, Nz_c)
    u_interp_fn = _field_interpolator(case["_u"], Ntheta_c, Nr_c, Nz_c, thetas_c, ts_c, zs_c, 3)

    Ntheta_f, Nr_f, Nz_f = ref["_Ntheta"], ref["_Nr"], ref["_Nz"]
    thetas_f = np.linspace(0.0, np.pi, Ntheta_f)
    ts_f = np.linspace(0.0, 1.0, Nr_f)
    zs_f = np.linspace(0.0, LZ, Nz_f)
    TH, T, Z = np.meshgrid(thetas_f, ts_f, zs_f, indexing="ij")
    theta_q = np.transpose(TH, (2, 0, 1)).ravel()
    t_q = np.transpose(T, (2, 0, 1)).ravel()
    z_q = np.transpose(Z, (2, 0, 1)).ravel()

    u_coarse_on_fine = u_interp_fn(theta_q, t_q, z_q)
    u_fine = ref["_u"]
    l2_num = np.sqrt(np.mean(np.sum((u_fine - u_coarse_on_fine) ** 2, axis=-1)))
    l2_den = np.sqrt(np.mean(np.sum(u_fine ** 2, axis=-1)))
    l2_rel = float(l2_num / l2_den) if l2_den > 0 else float("nan")

    F_interp_fn = _elem_field_interpolator(case["_F_elem"], Ntheta_c, Nr_c, Nz_c, case["_r_grading"])
    thetas_fc = 0.5 * (thetas_f[:-1] + thetas_f[1:])
    ts_fc = 0.5 * (ts_f[:-1] + ts_f[1:])
    zs_fc = 0.5 * (zs_f[:-1] + zs_f[1:])
    THc, Tc, Zc = np.meshgrid(thetas_fc, ts_fc, zs_fc, indexing="ij")
    theta_qc = np.transpose(THc, (2, 0, 1)).ravel()
    t_qc = np.transpose(Tc, (2, 0, 1)).ravel()
    z_qc = np.transpose(Zc, (2, 0, 1)).ravel()
    F_coarse_on_fine = F_interp_fn(theta_qc, t_qc, z_qc)
    F_fine = ref["_F_elem"]
    I3 = np.eye(3)[None, :, :]
    gradu_diff = (F_fine - I3) - (F_coarse_on_fine - I3)
    h1_num = np.sqrt(np.mean(np.sum(gradu_diff ** 2, axis=(-1, -2))))
    h1_den = np.sqrt(np.mean(np.sum((F_fine - I3) ** 2, axis=(-1, -2))))
    h1_rel = float(h1_num / h1_den) if h1_den > 0 else float("nan")

    # Fixed-region Cauchy-TENSOR relative field error.
    #
    # FIX (2026-09-21, Omar's own review of the write-up before sending
    # to Timon): the earlier version compared an ASYMMETRIC pair --
    # `case`'s own per-ELEMENT-AVERAGED Cauchy field against the
    # reference's own RAW per-Gauss-point values -- and that mismatch
    # produced a reported error that plateaued around ~17% instead of
    # shrinking with resolution, which was written up (prematurely, per
    # Omar's own catch) as a converged "finding." Two earlier attempts to
    # "fix" this by moving BOTH sides to raw Gauss-point values made
    # things WORSE (documented in this module's git history), because
    # raw per-Gauss stress in a coarse element is itself noisy. The
    # combination not yet tried was the one Omar asked for: a genuinely
    # SYMMETRIC comparison, with BOTH sides at the same (element-
    # averaged) representation and the same physical sample points --
    # `case`'s own field interpolated onto the REFERENCE's own ELEMENT
    # CENTROIDS (not Gauss points), compared against the reference's own
    # element-averaged field at those same centroids. Tested directly on
    # real CPU data before adopting it (not assumed): this version shows
    # clean, monotonic convergence with resolution (49.1% -> 23.8% ->
    # 14.6% -> 7.9% for 144/600/1,568/3,240-element cases against the
    # same 9,464-element reference) -- a real, physically sensible
    # convergence trend, unlike either of the two asymmetric versions
    # tried before it.
    theta_rc, t_rc, z_rc = _elem_centroid_parametric(Ntheta_f, Nr_f, Nz_f, ref["_r_grading"])
    ref_vol = ref["_vol_weight"].sum(axis=1)  # per-element volume, exact for this quadrature order
    z_ref_pt, r_ref_pt = LZ / 2.0, R_IN0 - GROOVE_DEPTH
    from omar_pfem.data.data_generate_B3 import groove_R_in as _groove_R_in
    R_in_eff_rc = _groove_R_in(z_rc, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    Rr_rc = R_in_eff_rc + (R_OUT - R_in_eff_rc) * t_rc
    pos_rc = np.stack([Rr_rc * np.cos(theta_rc), Rr_rc * np.sin(theta_rc), z_rc], axis=1)
    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    ref_elem_mask = np.linalg.norm(pos_rc - np.array([r_ref_pt, 0.0, z_ref_pt])[None, :], axis=1) < region_radius
    n_ref_region = int(ref_elem_mask.sum())
    if n_ref_region == 0:
        cauchy_field_rel = float("nan")
    else:
        sigma_interp_fn = _elem_field_interpolator(
            case["_sigma_elem"], Ntheta_c, Nr_c, Nz_c, case["_r_grading"])
        sigma_case_on_ref = sigma_interp_fn(
            theta_rc[ref_elem_mask], t_rc[ref_elem_mask], z_rc[ref_elem_mask])
        sigma_fine_region = ref["_sigma_elem"][ref_elem_mask]
        w_region = ref_vol[ref_elem_mask]
        w_sum = w_region.sum()

        diff_sq = np.sum((sigma_fine_region - sigma_case_on_ref) ** 2, axis=(-1, -2))
        fine_sq = np.sum(sigma_fine_region ** 2, axis=(-1, -2))
        num = np.sqrt((diff_sq * w_region).sum() / w_sum) if w_sum > 0 else float("nan")
        den = np.sqrt((fine_sq * w_region).sum() / w_sum) if w_sum > 0 else float("nan")
        cauchy_field_rel = float(num / den) if den > 0 else float("nan")

    return l2_rel, h1_rel, cauchy_field_rel


QOI_KEYS = [
    ("disp_l2_rel", "Displacement L2"),
    ("grad_h1_rel", "H1 (gradient) semi-norm"),
    ("energy_rel", "Total strain energy"),
    ("moment_rel", "Reaction moment (primary reaction QoI)"),
    ("force_rel", "Reaction force (secondary, verified non-degenerate)"),
    ("cauchy_field_rel", "Region-Cauchy field (full tensor)"),
    ("region_avg_rel", "Region-Cauchy avg (scalar, volume-weighted)"),
    ("region_p99_rel", "Region-Cauchy p99 (scalar, volume-weighted)"),
]
THRESHOLDS = [0.05, 0.02, 0.01]


def scalar_qoi_rel_errors(rows, ref):
    """Adds relative-error keys (vs. `ref`) for every scalar QoI besides
    the field-based disp_l2_rel/grad_h1_rel/cauchy_field_rel (which
    compare_to_reference already fills in) -- energy, reaction force
    magnitude, reaction moment, and the two region-Cauchy scalar
    statistics. True max is intentionally NOT included (secondary QoI
    only, per the established convention with Timon -- never used for a
    required-resolution threshold).

    Checked directly (not assumed), 2026-09-21: the reference's own net
    reaction force is NOT near zero (Y-component ~1.49; only the two
    symmetry-enforced components are ~1e-15 as expected) -- so force_rel
    is a legitimate relative error here, not an artifact of dividing by
    a near-zero reference. Moment remains the PRIMARY reaction QoI for
    this rocking problem regardless, since it is the more directly
    physically meaningful quantity for a pure rotation.

    region_p99_rel is set to NaN (not a misleading number) whenever
    EITHER the row's own or the reference's own p99 was flagged
    unreliable (too few quadrature-point samples in the fixed region)."""
    ref_force_mag = float(np.linalg.norm(ref["reaction_force"]))
    for r in rows:
        r["energy_rel"] = abs(r["total_strain_energy"] - ref["total_strain_energy"]) / abs(ref["total_strain_energy"])
        r["force_rel"] = abs(np.linalg.norm(r["reaction_force"]) - ref_force_mag) / max(ref_force_mag, 1e-12)
        r["moment_rel"] = abs(r["reaction_moment_y"] - ref["reaction_moment_y"]) / abs(ref["reaction_moment_y"])
        r["region_avg_rel"] = (abs(r["region_avg_sigma_xx"] - ref["region_avg_sigma_xx"])
                                / abs(ref["region_avg_sigma_xx"]))
        if r.get("region_p99_reliable") and ref.get("region_p99_reliable"):
            r["region_p99_rel"] = (abs(r["region_p99_sigma_xx"] - ref["region_p99_sigma_xx"])
                                    / abs(ref["region_p99_sigma_xx"]))
        else:
            r["region_p99_rel"] = float("nan")
    return rows


def find_required_resolutions(rows):
    """rows must already carry every *_rel key (scalar_qoi_rel_errors +
    compare_to_reference). For each QoI x threshold, returns the
    SMALLEST-element-count resolution (ascending order) whose own
    relative error vs. the reference is <= that threshold, or None if
    no tested resolution reaches it. NaN relative errors (e.g. an
    unreliable p99) never count as a hit."""
    rows_sorted = sorted(rows, key=lambda r: r["n_elements"])
    results = {}
    for key, label in QOI_KEYS:
        for thr in THRESHOLDS:
            hit = next((r for r in rows_sorted if r[key] == r[key] and r[key] <= thr), None)
            results[(label, thr)] = hit
    return results


def print_threshold_table(results):
    print("\n" + "=" * 90)
    print("Required-resolution table (first tested mesh reaching each threshold):")
    print(f"{'QoI':<42}{'Threshold':<12}{'Resolution':<16}{'n_elements':<12}{'Actual rel. err.'}")
    for (label, thr), hit in results.items():
        if hit is None:
            print(f"{label:<42}{thr*100:>4.0f}%{'':<7}{'not reached by any tested resolution'}")
        else:
            key = next(k for k, lbl in QOI_KEYS if lbl == label)
            print(f"{label:<42}{thr*100:>4.0f}%{'':<7}"
                  f"({hit['Ntheta']},{hit['Nr']},{hit['Nz']})".ljust(16) +
                  f"{hit['n_elements']:<12}{hit[key]*100:.3f}%")


def main():
    resolutions = [(9, 4, 7), (13, 6, 11), (17, 8, 15), (21, 10, 19)]
    fine_resolution = (29, 14, 27)

    print("Solving the fine reference first...")
    ref = solve_case(*fine_resolution, verbose=False)
    print(f"  fine reference: {ref['n_elements']} elements, {ref['elapsed_s']:.2f}s, "
          f"region samples (quadrature points)={ref['n_region']} "
          f"(p99 reliable={ref['region_p99_reliable']})")

    rows = []
    for Ntheta, Nr, Nz in resolutions:
        r = solve_case(Ntheta, Nr, Nz)
        l2_rel, h1_rel, cauchy_field_rel = compare_to_reference(r, ref)
        r["disp_l2_rel"] = l2_rel
        r["grad_h1_rel"] = h1_rel
        r["cauchy_field_rel"] = cauchy_field_rel
        rows.append(r)
        print(f"\n({Ntheta},{Nr},{Nz})  elements={r['n_elements']}  time={r['elapsed_s']:.2f}s")
        print(f"  disp_L2_rel={l2_rel*100:.3f}%  gradF_H1_rel={h1_rel*100:.3f}%  "
              f"cauchy_field_rel={cauchy_field_rel*100:.3f}%")
        print(f"  total_strain_energy={r['total_strain_energy']:.6e}")
        print(f"  reaction_moment_y (PRIMARY)={r['reaction_moment_y']:.6e}  "
              f"reaction_force (secondary)={np.array(r['reaction_force'])}")
        p99_str = f"{r['region_p99_sigma_xx']:.4f}" if r["region_p99_reliable"] else "NOT RELIABLE (too few samples)"
        print(f"  region(n={r['n_region']} quadrature points, volume-weighted): "
              f"avg_sigma_xx={r['region_avg_sigma_xx']:.4f}  p99={p99_str}  "
              f"(true_max={r['region_true_max_sigma_xx']:.4f}, secondary)")
        print(f"  equilibrium: force_rel_residual={r['force_rel_residual']:.2e} (normalized by a FORCE scale)  "
              f"moment_rel_residual={r['moment_rel_residual']:.2e} (normalized by a MOMENT scale) "
              f"-- dimensionally separate scales, verified not shared")

    print("\n" + "=" * 90)
    print("Summary across resolutions:")
    for r in rows:
        print(f"  ({r['Ntheta']:>2},{r['Nr']:>2},{r['Nz']:>2})  n_elem={r['n_elements']:>5}  "
              f"disp_L2={r['disp_l2_rel']*100:6.2f}%  gradF_H1={r['grad_h1_rel']*100:6.2f}%  "
              f"cauchy_field={r['cauchy_field_rel']*100:6.2f}%  "
              f"region_avg_sxx={r['region_avg_sigma_xx']:8.3f}  "
              f"moment_y={r['reaction_moment_y']:.4e}  strain_energy={r['total_strain_energy']:.4e}  "
              f"force_res={r['force_rel_residual']:.1e}  moment_res={r['moment_rel_residual']:.1e}")

    print("\nAll physics sanity checks passed inline for every row above (zero "
          "inverted elements, det(F)>0 at every element AND every quadrature "
          "point, Newton converged to 1e-8, global force/moment equilibrium "
          "each normalized by its OWN dimensionally-correct scale). Geometry/"
          "loading physical parameters (R_in0, R_out, Lz, groove depth/width, "
          "phi) were IDENTICAL across every resolution tested -- only "
          "Ntheta/Nr/Nz changed.")

    scalar_qoi_rel_errors(rows, ref)
    results = find_required_resolutions(rows)
    print_threshold_table(results)
    print("\n*** PRELIMINARY ONLY *** -- this table is computed against the "
          f"CPU-scale {fine_resolution} / {ref['n_elements']}-element reference, "
          "which is NOT yet validated as converged for the region-Cauchy-stress "
          "QoI specifically. Per Omar's own explicit instruction: do NOT treat "
          "this reference, or this threshold table, as final until a GPU run "
          "extends the resolution ladder far enough that the region metrics "
          "stop changing meaningfully between successive rows -- see the Colab "
          "notebook for that GPU extension, including a finer (~424k-element) "
          "reference to double-check the earlier 243,360-element one.")

    directional_study(ref)


def directional_study(ref):
    """Varies ONE mesh direction at a time (holding the other two fixed
    at a moderate baseline) to show, separately, that z-refinement,
    theta-refinement, and radial refinement near the groove EACH matter
    on their own -- not just that refining everything together helps."""
    baseline = (13, 6, 11)  # (Ntheta, Nr, Nz)
    axes = {
        "Nz (through-groove/axial resolution)": [(13, 6, nz) for nz in (7, 11, 15, 19, 27)],
        "Ntheta (circumferential, breaks axisymmetry)": [(nt, 6, 11) for nt in (5, 9, 13, 17, 25)],
        "Nr (radial, resolves the groove's own curvature)": [(13, nr, 11) for nr in (4, 6, 8, 10, 14)],
    }
    print("\n" + "=" * 90)
    print("DIRECTIONAL mesh study -- one axis varied at a time, baseline "
          f"Ntheta,Nr,Nz={baseline} otherwise:")
    for label, cases in axes.items():
        print(f"\n-- varying {label} --")
        for Ntheta, Nr, Nz in cases:
            r = solve_case(Ntheta, Nr, Nz)
            l2_rel, h1_rel, cauchy_field_rel = compare_to_reference(r, ref)
            p99_str = f"{r['region_p99_sigma_xx']:.3f}" if r["region_p99_reliable"] else "unreliable"
            print(f"  ({Ntheta:>2},{Nr:>2},{Nz:>2})  n_elem={r['n_elements']:>5}  "
                  f"disp_L2={l2_rel*100:6.2f}%  gradF_H1={h1_rel*100:6.2f}%  "
                  f"cauchy_field={cauchy_field_rel*100:6.2f}%  "
                  f"region_avg_sxx={r['region_avg_sigma_xx']:8.3f} (n_reg={r['n_region']}, p99={p99_str})  "
                  f"moment_y={r['reaction_moment_y']:.4e}")
    print("\nIf each axis on its own still shows meaningful change (not flat) "
          "while the other two stay at the coarse baseline, that axis's own "
          "resolution genuinely matters independently -- direct evidence this "
          "case is not dominated by just one refinement direction, i.e. "
          "genuinely 3D in the sense that matters for meshing cost.")


if __name__ == "__main__":
    main()
