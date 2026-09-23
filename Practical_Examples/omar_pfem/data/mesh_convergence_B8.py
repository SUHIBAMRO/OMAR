"""Mesh-convergence study for B8-FINAL (3D laminated annular elastomeric
seismic bearing -- Option B), rebuilt 2026-09-23 from a real published
bearing design. See data_generate_B8.py's own docstring for the full
geometry/material rebuild (real dimensions and materials from Kalantari
& Rofooei 2010, real linear-elastic steel via a per-element Neo-
Hookean/St. Venant-Kirchhoff dispatch). This module's own real fix:

STRESS-QOI REGION, REVISED: the prototype (mesh_convergence_B8_
prototype.py, archived unmodified) centered its region at the shim's
own mid-height, spanning BOTH rubber and steel, and its cross-mesh
comparison interpolated a field ACROSS that material discontinuity --
not a clean QoI definition. Fixed here: the reference point is the
rubber/shim-1 INTERFACE itself (z = T_RUBBER, r = R_out, theta = 0 --
the outer free edge of the first rubber layer, closest to the fixed
foundation), and BOTH the region mask (for solve_case's own scalar
region_avg) AND the cross-mesh comparison interpolator (in
compare_to_reference) are explicitly restricted to elements strictly
inside rubber layer 1 (element z-index range from data_generate_B8.
first_rubber_layer_z_range) -- never a shim element, and never an
interpolation that spans the two materials' different constitutive
response. true_max stays diagnostic-only throughout, same convention
as the rest of this project.

Uses the SAME symmetric, volume-weighted, quadrature-based region-
Cauchy-field methodology already validated for B3 (mesh_convergence_
B3.py), scoped to rubber-only elements as above.

Run on CPU first (small sanity checks), per this project's own standing
discipline -- same discipline B3's own mesh-convergence study followed.
GPU work is explicitly NOT started from this module yet (per
instruction, 2026-09-23): fix the model first, sanity-check it on CPU,
THEN plan the GPU ladder.
"""
import time

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from omar_pfem.data.data_generate_B8 import (
    generate_grid_hex8_laminated_bearing, boundary_node_sets,
    rigid_top_plate_displacement, build_vectorized_material_params,
    build_z_axis, first_rubber_layer_z_range,
    R_IN, R_OUT, N_RUBBER_LAYERS, T_RUBBER, T_SHIM)
from omar_pfem.torchfem_comparison import neo_hookean_or_stvk_psi_3d
from omar_pfem.data.mesh_convergence_B3 import _element_ijk  # generic index math, geometry-agnostic

COMPRESSION_FRAC = 0.02   # nominal compressive strain on the bearing height
SHEAR_X = 0.05 * T_RUBBER * N_RUBBER_LAYERS  # lateral (shear) displacement of the top plate, scaled to the real bearing height
PHI_Y = 0.0               # optional rigid rocking of the top plate (0 = pure shear+compression)
MIN_RELIABLE_N_P99 = 20
# Region radius: physical scale tied to shim thickness (a real, fixed
# geometric quantity of this design, T_SHIM=3.0mm), same principle as
# B3's own "region scaled to the local feature size" convention -- not
# an arbitrary number. 2xT_SHIM gave n_region=0 at every CPU-feasible
# resolution tested (confirmed directly): the reference point sits at
# r=R_out AND theta=0 simultaneously (two domain edges at once), where
# Gauss/centroid samples are always offset from the boundary by a fixed
# fraction of local element size in both directions -- the SAME
# geometric sampling constraint already found and fixed the same way in
# the B8-prototype. 6xT_SHIM (matching the prototype's own confirmed
# value) gives real, non-degenerate sampling (8-38 quadrature points
# across the tested resolutions) while staying strictly inside rubber
# layer 1 (enforced separately by the explicit rubber-layer-1 element
# mask, not by this radius).
REGION_RADIUS = 6.0 * T_SHIM

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
    return theta_g, t_g, z_g, k


def _elem_centroid_parametric(Ntheta, Nr, zs, r_grading):
    Nz = len(zs)
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** r_grading
    thetas_c = 0.5 * (thetas[:-1] + thetas[1:])
    ts_c = 0.5 * (ts[:-1] + ts[1:])
    zs_c = 0.5 * (zs[:-1] + zs[1:])
    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    return thetas_c[j], ts_c[i], zs_c[k], k


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


def _elem_field_interpolator_rubber1(field, Ntheta, Nr, zs, k0, k1, r_grading=1.0):
    """SAME idea as a structured element-centroid interpolator, but
    scoped to ONLY the z-element-layer range [k0,k1) belonging to rubber
    layer 1 (data_generate_B8.first_rubber_layer_z_range) -- built from a
    field already reshaped to (Nz-1, Ntheta-1, Nr-1, ...) and sliced to
    that z-range BEFORE building the interpolator, so it structurally
    cannot return a value influenced by any shim element, regardless of
    query point."""
    n_comp = int(np.prod(field.shape[1:]))
    field_flat = field.reshape(-1, n_comp)
    thetas_c = 0.5 * (np.linspace(0.0, np.pi, Ntheta)[:-1] + np.linspace(0.0, np.pi, Ntheta)[1:])
    ts_axis = np.linspace(0.0, 1.0, Nr) ** r_grading
    ts_c = 0.5 * (ts_axis[:-1] + ts_axis[1:])
    zs_c_full = 0.5 * (zs[:-1] + zs[1:])
    grid_full = field_flat.reshape(len(zs) - 1, Ntheta - 1, Nr - 1, n_comp)
    grid_rubber1 = grid_full[k0:k1]
    zs_c_rubber1 = zs_c_full[k0:k1]
    interp = RegularGridInterpolator((zs_c_rubber1, thetas_c, ts_c), grid_rubber1,
                                      bounds_error=False, fill_value=None)
    def query(theta_q, t_q, z_q):
        pts = np.stack([z_q, theta_q, t_q], axis=-1)
        return interp(pts).reshape((-1,) + field.shape[1:])
    return query


def solve_case(Ntheta, Nr, n_rubber_layers=N_RUBBER_LAYERS, nz_per_rubber=2, nz_per_shim=2,
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
    params_np = build_vectorized_material_params(is_shim)
    params_t = torch.tensor(params_np, dtype=dtype, device=device)
    material = Hyperelastic3D(psi=neo_hookean_or_stvk_psi_3d, params=params_t)
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
        # torch.no_grad(): confirmed directly (real GPU OOM in the
        # prototype's own study, not theorized) that torchfem's
        # NewtonRaphsonAdjoint.forward calls ctx.save_for_backward on
        # EVERY increment (its own implicit-adjoint gradient support);
        # without no_grad, autograd retains that whole per-increment
        # graph, so GPU memory grows with n_increments, not just mesh
        # size. This solve never needs gradients, so no_grad is the
        # correct fix -- verified bit-identical results before/after.
        with torch.device(device), torch.no_grad():
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

    # Real, requested sanity check: shim elements should show SMALL
    # strain (confirming the St. Venant-Kirchhoff/linear-elastic
    # assumption is self-consistent, not just asserted) even though they
    # may undergo large ROTATION as part of the overall shear+rocking
    # deformation. Green-Lagrange strain E = 0.5*(F^T F - I); its
    # largest component magnitude over all shim elements is the relevant
    # diagnostic (small strain, e.g. << 1%, is expected given steel's
    # huge stiffness relative to the applied load).
    F_elem_np = F_elem.cpu().numpy()
    E_elem = 0.5 * (np.einsum("eji,ejk->eik", F_elem_np, F_elem_np) - np.eye(3)[None])
    max_strain_shim = float(np.abs(E_elem[is_shim]).max()) if is_shim.any() else float("nan")
    max_strain_rubber = float(np.abs(E_elem[~is_shim]).max())

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
    z0, z1, k0, k1 = first_rubber_layer_z_range(n_rubber_layers, nz_per_rubber, nz_per_shim, T_RUBBER, T_SHIM)
    theta_g, t_g, z_g, k_g = _gauss_point_parametric(Ntheta, Nr, zs, r_grading, ipoints_np)
    Rr_g = R_IN + (R_OUT - R_IN) * t_g
    x_g = Rr_g * np.cos(theta_g)
    y_g = Rr_g * np.sin(theta_g)
    pos_g = np.stack([x_g, y_g, z_g], axis=-1)

    # Region: the rubber/shim-1 INTERFACE (z1), at the outer free edge
    # (r=R_out), on the symmetry plane (theta=0) -- REVISED from the
    # prototype's shim-mid-height point. Restricted to rubber layer 1's
    # OWN element z-index range [k0,k1) as an explicit, structural
    # guarantee that no shim element/Gauss point is ever included,
    # regardless of the physical region_radius chosen.
    ref_point = np.array([R_OUT, 0.0, z1])
    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    in_rubber1 = (k_g >= k0) & (k_g < k1)
    region_mask = (dist < REGION_RADIUS) & in_rubber1[:, None]
    n_region = int(region_mask.sum())
    assert not is_shim[np.where(region_mask.any(axis=1))[0]].any(), \
        "region mask includes a shim element -- QoI region definition is broken"

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
        "max_strain_shim": max_strain_shim, "max_strain_rubber": max_strain_rubber,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
        "elapsed_s": elapsed,
        "_u": u_np, "_F_elem": F_elem_np, "_sigma_elem": sigma_elem,
        "_vol_weight": vol_weight, "_is_shim": is_shim,
        "_Ntheta": Ntheta, "_Nr": Nr, "_r_grading": r_grading, "_zs": zs,
        "_n_rubber_layers": n_rubber_layers, "_nz_per_rubber": nz_per_rubber,
        "_nz_per_shim": nz_per_shim,
    }


def compare_to_reference(case, ref):
    """Displacement L2 (node field, exact interpolation, whole domain --
    both materials, since displacement itself is continuous across the
    interface by construction) and fixed-region Cauchy-TENSOR relative
    field error AT THE RUBBER/SHIM-1 INTERFACE'S OUTER FREE EDGE, using
    the SAME symmetric element-centroid-to-element-centroid methodology
    validated for B3, but with BOTH sides' interpolators explicitly
    scoped to rubber-layer-1 elements only (_elem_field_interpolator_
    rubber1) -- never a shim element, never a cross-material
    interpolation."""
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

    z0_c, z1_c, k0_c, k1_c = first_rubber_layer_z_range(
        case["_n_rubber_layers"], case["_nz_per_rubber"], case["_nz_per_shim"], T_RUBBER, T_SHIM)
    z0_f, z1_f, k0_f, k1_f = first_rubber_layer_z_range(
        ref["_n_rubber_layers"], ref["_nz_per_rubber"], ref["_nz_per_shim"], T_RUBBER, T_SHIM)

    theta_rc, t_rc, z_rc, k_rc = _elem_centroid_parametric(Ntheta_f, Nr_f, zs_f, ref["_r_grading"])
    ref_vol = ref["_vol_weight"].sum(axis=1)
    Rr_rc = R_IN + (R_OUT - R_IN) * t_rc
    pos_rc = np.stack([Rr_rc * np.cos(theta_rc), Rr_rc * np.sin(theta_rc), z_rc], axis=1)
    in_rubber1_f = (k_rc >= k0_f) & (k_rc < k1_f)
    ref_elem_mask = (np.linalg.norm(pos_rc - np.array([R_OUT, 0.0, z1_f])[None, :], axis=1) < REGION_RADIUS) \
        & in_rubber1_f
    n_ref_region = int(ref_elem_mask.sum())
    if n_ref_region:
        assert not ref["_is_shim"][ref_elem_mask].any(), \
            "region mask includes a shim element in the reference mesh -- QoI region definition is broken"
    if n_ref_region == 0:
        cauchy_field_rel = float("nan")
    else:
        sigma_interp_fn = _elem_field_interpolator_rubber1(
            case["_sigma_elem"], Ntheta_c, Nr_c, zs_c, k0_c, k1_c, case["_r_grading"])
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
    print("B8-FINAL (laminated annular seismic bearing, real published-source "
          "geometry + real materials) -- CPU SANITY CHECK, per explicit "
          "instruction (2026-09-23): small resolutions only, verifying mesh "
          "validity, det(F)>0, equilibrium, shim strain (small-strain "
          "self-consistency), reactions, and the revised rubber-only stress "
          "QoI region -- NOT a mesh-convergence study or GPU work yet.\n")

    r = solve_case(9, 5, verbose=True)
    print(f"n_elements={r['n_elements']} (n_shim_elements={r['n_shim_elements']}) "
          f"n_nodes={r['n_nodes']} Lz={r['Lz']:.2f}mm time={r['elapsed_s']:.2f}s")
    print(f"reaction_force_bottom={np.array(r['reaction_force_bottom'])}")
    print(f"reaction_force_top={np.array(r['reaction_force_top'])}")
    print(f"force_rel_residual={r['force_rel_residual']:.2e} (global equilibrium sanity check)")
    print(f"max_disp={r['max_disp']:.4f}mm")
    print(f"max_strain (Green-Lagrange, abs): shim={r['max_strain_shim']:.6e}  "
          f"rubber={r['max_strain_rubber']:.6e}  "
          f"(shim strain should be SMALL -- confirms the linear-elastic/"
          f"St.Venant-Kirchhoff assumption is self-consistent)")
    print(f"region (rubber layer 1 only, n={r['n_region']} quadrature points): "
          f"avg_sigma_xx={r['region_avg_sigma_xx']:.4f} MPa  "
          f"(true_max={r['region_true_max_sigma_xx']:.4f} MPa, diagnostic only)")
    print("\nAll physics sanity checks passed inline (zero inverted elements, "
          "det(F)>0 at every element AND every quadrature point, Newton "
          "converged to 1e-8, global force equilibrium normalized by its own "
          "force scale, region mask asserted shim-free).")


if __name__ == "__main__":
    main()
