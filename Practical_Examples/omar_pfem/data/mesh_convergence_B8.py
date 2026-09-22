"""Mesh-convergence study for B8 (3D laminated annular elastomeric seismic
bearing -- Option B). Per Omar's explicit correction (2026-09-22): a
two-resolution smoke test is NOT a real result -- this module is the real
study: a genuine resolution ladder against a fine reference, with the
SAME symmetric, volume-weighted, quadrature-based region-Cauchy-field
methodology already validated for B3 (mesh_convergence_B3.py), adapted to
this geometry's own (non-uniform, banded) z-axis and its own reference
point (the free edge of an internal rubber-shim interface, not a groove).

Physics/BCs/material: see data_generate_B8.py's own docstring for the
full design and the real API-checked reasoning behind modeling shims as
per-element vectorized Neo-Hookean parameters rather than an unsupported
rigid multi-point constraint or torch-fem's shell-only Laminate class.

Run on CPU first, per this project's own standing discipline (do not
start GPU work until CPU shows real resolution-sensitivity) -- same
discipline B3's own mesh-convergence study followed.
"""
import time

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from omar_pfem.data.data_generate_B8 import (
    generate_grid_hex8_laminated_bearing, boundary_node_sets,
    rigid_top_plate_displacement, build_vectorized_neo_hookean_params,
    build_z_axis, first_internal_shim_mid_z,
    R_IN, R_OUT, N_RUBBER_LAYERS, T_RUBBER, T_SHIM)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d
from omar_pfem.data.mesh_convergence_B3 import _element_ijk  # generic index math, geometry-agnostic

COMPRESSION_FRAC = 0.02   # nominal compressive strain on the bearing height
SHEAR_X = 0.05            # lateral (shear) displacement of the top plate
PHI_Y = 0.0               # optional rigid rocking of the top plate (0 = pure shear+compression)
MIN_RELIABLE_N_P99 = 20

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


def _theta_t_z_axes(Ntheta, Nr, n_rubber_layers, nz_per_rubber, nz_per_shim, r_grading=1.0):
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** r_grading
    zs, layer_is_shim = build_z_axis(n_rubber_layers, nz_per_rubber, nz_per_shim, T_RUBBER, T_SHIM)
    return thetas, ts, zs, layer_is_shim


def _gauss_point_parametric(Ntheta, Nr, zs, r_grading, ipoints_np):
    Nz = len(zs)
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


def _elem_centroid_parametric(Ntheta, Nr, zs, r_grading):
    Nz = len(zs)
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** r_grading
    thetas_c = 0.5 * (thetas[:-1] + thetas[1:])
    ts_c = 0.5 * (ts[:-1] + ts[1:])
    zs_c = 0.5 * (zs[:-1] + zs[1:])
    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    return thetas_c[j], ts_c[i], zs_c[k]


def _field_interpolator(field, Ntheta, Nr, zs, n_components):
    Nz = len(zs)
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr)  # node field interpolation uses raw t (matches node generation order)
    grid = field.reshape(Nz, Ntheta, Nr, n_components)
    interp = RegularGridInterpolator((zs, thetas, ts), grid, bounds_error=False, fill_value=None)
    def query(theta_q, t_q, z_q):
        pts = np.stack([z_q, theta_q, t_q], axis=-1)
        return interp(pts)
    return query


def _elem_field_interpolator(field, Ntheta, Nr, zs, r_grading=1.0):
    n_comp = int(np.prod(field.shape[1:]))
    field_flat = field.reshape(-1, n_comp)
    thetas_c = 0.5 * (np.linspace(0.0, np.pi, Ntheta)[:-1] + np.linspace(0.0, np.pi, Ntheta)[1:])
    ts_axis = np.linspace(0.0, 1.0, Nr) ** r_grading
    ts_c = 0.5 * (ts_axis[:-1] + ts_axis[1:])
    zs_c = 0.5 * (zs[:-1] + zs[1:])
    grid = field_flat.reshape(len(zs) - 1, Ntheta - 1, Nr - 1, n_comp)
    interp = RegularGridInterpolator((zs_c, thetas_c, ts_c), grid, bounds_error=False, fill_value=None)
    def query(theta_q, t_q, z_q):
        pts = np.stack([z_q, theta_q, t_q], axis=-1)
        return interp(pts).reshape((-1,) + field.shape[1:])
    return query


def solve_case(Ntheta, Nr, n_rubber_layers=N_RUBBER_LAYERS, nz_per_rubber=3, nz_per_shim=2,
               r_grading=1.0, n_increments=11, dtype=torch.float64, device=None,
               verbose=False, method="cg", preconditioner="jacobi"):
    device = device or torch.device("cpu")
    nodes, elements, is_shim, Lz = generate_grid_hex8_laminated_bearing(
        R_IN, R_OUT, Ntheta, Nr, n_rubber_layers=n_rubber_layers,
        nz_per_rubber=nz_per_rubber, nz_per_shim=nz_per_shim, r_grading=r_grading)
    bottom, top, outer, inner, sym = boundary_node_sets(nodes, R_IN, R_OUT, Lz)

    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        return sum(np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
                   for a, b, c, d in tets)
    n_inverted = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    assert n_inverted == 0, f"{n_inverted} inverted elements -- mesh invalid"

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)
    params_np = build_vectorized_neo_hookean_params(is_shim)
    params_t = torch.tensor(params_np, dtype=dtype, device=device)
    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params_t)
    assert material.is_vectorized, "expected a per-element vectorized material"
    with torch.device(device):
        model = Solid(nodes_t, elements_t, material)

    n_nodes = nodes.shape[0]
    model.forces = torch.zeros(n_nodes, 3, dtype=dtype, device=device)

    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    constraints[bottom, :] = True
    ux, uy, uz = rigid_top_plate_displacement(
        nodes[top], Lz, compression=COMPRESSION_FRAC * Lz, shear_x=SHEAR_X, phi_y=PHI_Y)
    constraints[top, :] = True
    displacements[top, 0] = torch.tensor(ux, dtype=dtype, device=device)
    displacements[top, 1] = torch.tensor(uy, dtype=dtype, device=device)
    displacements[top, 2] = torch.tensor(uz, dtype=dtype, device=device)
    constraints[sym, 1] = True
    model.constraints = constraints
    model.displacements = displacements

    increments = torch.linspace(0.0, 1.0, n_increments, dtype=dtype, device=device)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    t0 = time.time()
    try:
        with torch.device(device):
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method=method, preconditioner=preconditioner, nlgeom=True, verbose=verbose,
                aggregate_integration_points=False)
    finally:
        torch.set_default_dtype(old_default_dtype)
    elapsed = time.time() - t0

    u_np = u.cpu().numpy()
    assert np.isfinite(u_np).all(), "NaN/Inf in solution -- Newton did not really converge cleanly"

    P = P.transpose(0, 1)
    F = F.transpose(0, 1)
    F_elem = F.mean(dim=1)
    detF_elem = torch.linalg.det(F_elem)
    assert (detF_elem > 0).all(), "non-positive det(F) somewhere -- element inversion under load"
    detF_gauss = torch.linalg.det(F)
    assert (detF_gauss > 0).all(), "non-positive det(F) at a Gauss point -- element inversion under load"

    f_np = f.cpu().numpy()
    reaction_force_bottom = f_np[bottom].sum(axis=0)
    reaction_force_top = f_np[top].sum(axis=0)
    total_force = f_np.sum(axis=0)
    force_scale = max(np.linalg.norm(reaction_force_bottom), 1e-8)
    force_rel_residual = float(np.linalg.norm(total_force) / force_scale)
    assert force_rel_residual < 1e-6, f"global force equilibrium violated: {total_force}"

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
    sigma_elem = sigma_gauss.mean(axis=1)

    zs, layer_is_shim = build_z_axis(n_rubber_layers, nz_per_rubber, nz_per_shim, T_RUBBER, T_SHIM)
    theta_g, t_g, z_g = _gauss_point_parametric(Ntheta, Nr, zs, r_grading, ipoints_np)
    Rr_g = R_IN + (R_OUT - R_IN) * t_g
    x_g = Rr_g * np.cos(theta_g)
    y_g = Rr_g * np.sin(theta_g)
    pos_g = np.stack([x_g, y_g, z_g], axis=-1)

    z_ref = first_internal_shim_mid_z(n_rubber_layers, nz_per_rubber, nz_per_shim, T_RUBBER, T_SHIM)
    ref_point = np.array([R_OUT, 0.0, z_ref])
    # 4*T_SHIM (not 2x, unlike B3's own 2*rho convention): confirmed
    # directly that this reference point sits AT two domain edges at once
    # (r=R_out, theta=0), where Gauss points are always offset from the
    # boundary by a fixed ~0.21 fraction of the local element size in
    # EACH of those two directions simultaneously -- a geometric sampling
    # constraint, not a resolution problem, so 2x needed a widened but
    # still shim-thickness-scaled region rather than a resolution fix.
    region_radius = 6.0 * T_SHIM
    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    region_mask = dist < region_radius
    n_region = int(region_mask.sum())

    sigma_flat = sigma_gauss[region_mask]
    w_flat = vol_weight[region_mask]
    w_sum = w_flat.sum()
    if n_region > 0 and w_sum > 0:
        region_avg = float((sigma_flat[:, 0, 0] * w_flat).sum() / w_sum)
    else:
        region_avg = float("nan")
    region_true_max = float(np.abs(sigma_flat[:, 0, 0]).max()) if n_region else float("nan")

    return {
        "Ntheta": Ntheta, "Nr": Nr, "n_elements": len(elements), "n_nodes": n_nodes,
        "n_shim_elements": int(is_shim.sum()), "Lz": Lz, "n_region": n_region,
        "reaction_force_bottom": reaction_force_bottom.tolist(),
        "reaction_force_top": reaction_force_top.tolist(),
        "force_rel_residual": force_rel_residual,
        "region_avg_sigma_xx": region_avg, "region_true_max_sigma_xx": region_true_max,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
        "elapsed_s": elapsed,
        "_u": u_np, "_F_elem": F_elem.cpu().numpy(), "_sigma_elem": sigma_elem,
        "_vol_weight": vol_weight,
        "_Ntheta": Ntheta, "_Nr": Nr, "_r_grading": r_grading, "_zs": zs,
        "_n_rubber_layers": n_rubber_layers, "_nz_per_rubber": nz_per_rubber,
        "_nz_per_shim": nz_per_shim,
    }


def compare_to_reference(case, ref):
    """Displacement L2 (node field, exact interpolation) and fixed-region
    Cauchy-TENSOR relative field error at the internal shim's outer free
    edge, using the SAME symmetric element-centroid-to-element-centroid
    methodology validated for B3 (case's field interpolated onto ref's own
    element centroids, compared against ref's own element-averaged field
    at those same centroids, volume-weighted) -- not the asymmetric
    Gauss-vs-element comparison that was found and fixed as a real bug in
    B3's own study."""
    Ntheta_c, Nr_c, zs_c = case["_Ntheta"], case["_Nr"], case["_zs"]
    thetas_c = np.linspace(0.0, np.pi, Ntheta_c)
    ts_c = np.linspace(0.0, 1.0, Nr_c)
    u_interp_fn = _field_interpolator(case["_u"], Ntheta_c, Nr_c, zs_c, 3)

    Ntheta_f, Nr_f, zs_f = ref["_Ntheta"], ref["_Nr"], ref["_zs"]
    thetas_f = np.linspace(0.0, np.pi, Ntheta_f)
    ts_f = np.linspace(0.0, 1.0, Nr_f)
    TH, T, Z = np.meshgrid(thetas_f, ts_f, zs_f, indexing="ij")
    theta_q = np.transpose(TH, (2, 0, 1)).ravel()
    t_q = np.transpose(T, (2, 0, 1)).ravel()
    z_q = np.transpose(Z, (2, 0, 1)).ravel()

    u_coarse_on_fine = u_interp_fn(theta_q, t_q, z_q)
    u_fine = ref["_u"]
    l2_num = np.sqrt(np.mean(np.sum((u_fine - u_coarse_on_fine) ** 2, axis=-1)))
    l2_den = np.sqrt(np.mean(np.sum(u_fine ** 2, axis=-1)))
    l2_rel = float(l2_num / l2_den) if l2_den > 0 else float("nan")

    theta_rc, t_rc, z_rc = _elem_centroid_parametric(Ntheta_f, Nr_f, zs_f, ref["_r_grading"])
    ref_vol = ref["_vol_weight"].sum(axis=1)
    z_ref = first_internal_shim_mid_z(ref["_n_rubber_layers"], ref["_nz_per_rubber"],
                                       ref["_nz_per_shim"], T_RUBBER, T_SHIM)
    Rr_rc = R_IN + (R_OUT - R_IN) * t_rc
    pos_rc = np.stack([Rr_rc * np.cos(theta_rc), Rr_rc * np.sin(theta_rc), z_rc], axis=1)
    region_radius = 6.0 * T_SHIM  # matches solve_case's own region_radius exactly
    ref_elem_mask = np.linalg.norm(pos_rc - np.array([R_OUT, 0.0, z_ref])[None, :], axis=1) < region_radius
    n_ref_region = int(ref_elem_mask.sum())
    if n_ref_region == 0:
        cauchy_field_rel = float("nan")
    else:
        sigma_interp_fn = _elem_field_interpolator(case["_sigma_elem"], Ntheta_c, Nr_c, zs_c, case["_r_grading"])
        sigma_case_on_ref = sigma_interp_fn(theta_rc[ref_elem_mask], t_rc[ref_elem_mask], z_rc[ref_elem_mask])
        sigma_fine_region = ref["_sigma_elem"][ref_elem_mask]
        w_region = ref_vol[ref_elem_mask]
        w_sum = w_region.sum()
        diff_sq = np.sum((sigma_fine_region - sigma_case_on_ref) ** 2, axis=(-1, -2))
        fine_sq = np.sum(sigma_fine_region ** 2, axis=(-1, -2))
        num = np.sqrt((diff_sq * w_region).sum() / w_sum) if w_sum > 0 else float("nan")
        den = np.sqrt((fine_sq * w_region).sum() / w_sum) if w_sum > 0 else float("nan")
        cauchy_field_rel = float(num / den) if den > 0 else float("nan")

    return l2_rel, cauchy_field_rel, n_ref_region


def main():
    print("B8 (laminated annular seismic bearing) -- REAL mesh-convergence "
          "study (not a smoke test): a genuine resolution ladder against a "
          "fine reference, same symmetric region-Cauchy-field methodology "
          "already validated for B3, at the free edge of an internal "
          "rubber-shim interface.\n")

    fine_resolution = (25, 13)
    print(f"Solving fine reference {fine_resolution} ...")
    ref = solve_case(*fine_resolution, r_grading=1.0)
    print(f"  fine ref: n_elements={ref['n_elements']}  time={ref['elapsed_s']:.1f}s  "
          f"n_region={ref['n_region']}  region_avg_sxx={ref['region_avg_sigma_xx']:.4f}  "
          f"true_max={ref['region_true_max_sigma_xx']:.4f}")

    resolutions = [(9, 5), (13, 7), (17, 9), (21, 11)]
    rows = []
    for Ntheta, Nr in resolutions:
        r = solve_case(Ntheta, Nr, r_grading=1.0)
        l2_rel, cauchy_field_rel, n_ref_region = compare_to_reference(r, ref)
        r["disp_l2_rel"] = l2_rel
        r["cauchy_field_rel"] = cauchy_field_rel
        rows.append(r)
        cf_str = f"{cauchy_field_rel*100:.3f}%" if cauchy_field_rel == cauchy_field_rel else \
            f"NaN (ref n_region={n_ref_region})"
        print(f"  ({Ntheta:>3},{Nr:>3})  n_elem={r['n_elements']:>6}  time={r['elapsed_s']:6.1f}s  "
              f"n_region={r['n_region']:>4}  disp_L2={l2_rel*100:6.2f}%  cauchy_field={cf_str}  "
              f"true_max_sxx={r['region_true_max_sigma_xx']:>9.3f}")

    print("\n" + "=" * 90)
    print("SUMMARY (real CPU resolution ladder, B8 laminated seismic bearing):")
    for r in rows:
        print(f"  n_elem={r['n_elements']:>6}  disp_L2={r['disp_l2_rel']*100:6.2f}%  "
              f"cauchy_field={r['cauchy_field_rel']*100:6.2f}%  n_region={r['n_region']:>4}")
    print("\n*** PRELIMINARY ONLY *** -- CPU-scale reference "
          f"({ref['n_elements']} elements), not yet GPU-confirmed converged, "
          "same standing discipline as B3's own study: do not treat this as "
          "final until extended on GPU if/when this design is confirmed "
          "with the advisor.")


if __name__ == "__main__":
    main()
