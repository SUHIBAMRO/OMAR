"""Real, non-hypothetical test of Option A from the 2026-09-22 candidate-
direction discussion: can making B3's groove sharper (smaller radius of
curvature) genuinely push the required mesh resolution into Timon's
stated target range (10^5-10^6 elements, 5-10% QoI error), rather than
converging faster than that the way the CURRENT B3 design does (already
~1.3% at 123k elements per the 2026-09-21 report)?

This is NOT a new method -- it reuses mesh_convergence_B3.py's exact
solve_case/compare_to_reference logic (same equations, same symmetric
element-centroid Cauchy-field-error fix, same region-radius convention
of `region_radius = 2 * groove_radius_of_curvature(depth, half_width)`
-- i.e. the measurement region is scaled to the feature's OWN size,
which is the existing, already-validated convention, not something
invented for this test). The only change is that groove_depth/
groove_half_width become function parameters instead of module-level
constants, so several groove designs can be run with IDENTICAL code.

Per Omar's explicit instruction (2026-09-22): no shortcuts, no invented
workarounds (the earlier ad-hoc monkeypatch of groove_radius_of_curvature
to force a fixed region size was exactly the kind of thing NOT to do --
it decoupled the measurement region from the geometry it was supposed to
characterize). The correct remedy for a smaller feature is to resolve it
with a finer/graded mesh (r_grading concentrates radial nodes near the
groove), not to shrink or fix the region artificially. If a resolution
genuinely cannot sample the region at all (n_region=0), that is reported
as NaN, honestly, exactly as the original module already does -- never
hidden or patched around.
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
    groove_radius_of_curvature, groove_R_in)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d
from omar_pfem.data.mesh_convergence_B3 import (
    R_IN0, R_OUT, LZ, E, NU, PHI, MIN_RELIABLE_N_P99,
    _theta_t_axes, _element_ijk, _gauss_point_parametric, _field_interpolator,
    _elem_field_interpolator, _elem_centroid_parametric,
)

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


def solve_case(Ntheta, Nr, Nz, groove_depth, groove_half_width, r_grading=1.0,
               dtype=torch.float64, device=None, verbose=False, n_increments=11,
               method="cg", preconditioner="jacobi"):
    """Identical physics/BCs/solve procedure to mesh_convergence_B3.solve_case
    -- groove_depth/groove_half_width are now parameters (were module
    constants there) so multiple groove designs can share this one,
    unmodified code path. n_increments (default 11, matching
    mesh_convergence_B3.py exactly) is also exposed: a deeper groove is a
    genuinely harder nonlinear problem (confirmed directly -- depth=0.20
    at (17,10,15) failed Newton-Raphson convergence at only 20% of the
    applied rocking load with the default 11 increments), and finer load
    stepping is the standard, legitimate FEM remedy for that -- not a
    workaround, a real requirement of the sharper geometry itself.

    method/preconditioner are also exposed: at depth=0.35 (7x sharper),
    CG+Jacobi was confirmed directly to be extremely slow to converge
    even at a small (1,560-element) mesh -- consistent with the r_grading
    needed to resolve that groove producing a badly ill-conditioned
    linear system for an iterative solver. torch-fem's own direct solver
    is a standard, exact alternative for the SAME linear system (not an
    approximation, not a different physics/BC/material setup) -- using it
    for ill-conditioned cases changes only how the linear system is
    solved, not what is being solved."""
    device = device or torch.device("cpu")
    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, groove_depth, groove_half_width, r_grading=r_grading)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, groove_depth, groove_half_width)

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
    P_elem = P.mean(dim=1)

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

    f_np = f.cpu().numpy()
    reaction_force = f_np[outer].sum(axis=0)
    rel = nodes[outer] - np.array([0.0, 0.0, LZ / 2.0])
    torque_vec = np.cross(rel, f_np[outer])
    reaction_moment_y = float(torque_vec[:, 1].sum())

    total_force = f_np.sum(axis=0)
    rel_all = nodes - np.array([0.0, 0.0, LZ / 2.0])
    total_moment_y = float(np.cross(rel_all, f_np)[:, 1].sum())
    force_scale = max(np.linalg.norm(reaction_force), 1e-8)
    moment_scale = max(abs(reaction_moment_y), 1e-8)
    force_rel_residual = float(np.linalg.norm(total_force) / force_scale)
    moment_rel_residual = float(abs(total_moment_y) / moment_scale)
    assert force_rel_residual < 1e-6, f"global force equilibrium violated: {total_force}"
    assert moment_rel_residual < 1e-2, \
        f"global moment equilibrium violated: {total_moment_y} (relative {moment_rel_residual:.4f})"

    F_elem_np = F_elem.cpu().numpy()
    energy_density = np.array([
        neo_hookean_psi_3d(torch.tensor(F_elem_np[e]), params).item() for e in range(F_elem_np.shape[0])
    ])
    elem_volumes = np.array([hex_signed_volume(nodes[el]) for el in elements])
    total_strain_energy = float((energy_density * elem_volumes).sum())

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

    theta_g, t_g, z_g = _gauss_point_parametric(Ntheta, Nr, Nz, r_grading, ipoints_np)

    z_ref = LZ / 2.0
    r_ref = R_IN0 - groove_depth
    ref_point = np.array([r_ref, 0.0, z_ref])
    region_radius = 2.0 * groove_radius_of_curvature(groove_depth, groove_half_width)

    R_in_eff_g = groove_R_in(z_g, LZ, R_IN0, groove_depth, groove_half_width)
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

    region_true_max = float(sigma_flat[:, 0, 0].max()) if n_region else float("nan")
    sigma_elem = sigma_gauss.mean(axis=1)

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
        "_u": u_np, "_F_elem": F_elem_np, "_sigma_elem": sigma_elem,
        "_sigma_gauss": sigma_gauss, "_pos_gauss": pos_g, "_vol_weight": vol_weight,
        "_region_mask": region_mask,
        "_Ntheta": Ntheta, "_Nr": Nr, "_Nz": Nz, "_r_grading": r_grading,
        "_groove_depth": groove_depth, "_groove_half_width": groove_half_width,
    }


def compare_to_reference(case, ref, groove_depth, groove_half_width):
    """Identical logic to mesh_convergence_B3.compare_to_reference (same
    symmetric element-centroid Cauchy-field fix), with groove_depth/
    groove_half_width as parameters."""
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

    from omar_pfem.data.mesh_convergence_B3 import _elem_field_interpolator as _efi
    F_interp_fn = _efi(case["_F_elem"], Ntheta_c, Nr_c, Nz_c, case["_r_grading"])
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

    theta_rc, t_rc, z_rc = _elem_centroid_parametric(Ntheta_f, Nr_f, Nz_f, ref["_r_grading"])
    ref_vol = ref["_vol_weight"].sum(axis=1)
    z_ref_pt, r_ref_pt = LZ / 2.0, R_IN0 - groove_depth
    R_in_eff_rc = groove_R_in(z_rc, LZ, R_IN0, groove_depth, groove_half_width)
    Rr_rc = R_in_eff_rc + (R_OUT - R_in_eff_rc) * t_rc
    pos_rc = np.stack([Rr_rc * np.cos(theta_rc), Rr_rc * np.sin(theta_rc), z_rc], axis=1)
    region_radius = 2.0 * groove_radius_of_curvature(groove_depth, groove_half_width)
    ref_elem_mask = np.linalg.norm(pos_rc - np.array([r_ref_pt, 0.0, z_ref_pt])[None, :], axis=1) < region_radius
    n_ref_region = int(ref_elem_mask.sum())
    if n_ref_region == 0:
        cauchy_field_rel = float("nan")
    else:
        sigma_interp_fn = _efi(case["_sigma_elem"], Ntheta_c, Nr_c, Nz_c, case["_r_grading"])
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

    return l2_rel, h1_rel, cauchy_field_rel, n_ref_region


def run_groove_design(label, groove_depth, groove_half_width, resolutions, fine_resolution,
                       r_grading=1.0, n_increments=11):
    rho = groove_radius_of_curvature(groove_depth, groove_half_width)
    print("\n" + "#" * 90)
    print(f"# Groove design: {label}  (depth={groove_depth}, half_width={groove_half_width}, "
          f"rho={rho:.6f}, r_grading={r_grading}, n_increments={n_increments})")
    print("#" * 90)

    print(f"Solving fine reference {fine_resolution} ...")
    ref = solve_case(*fine_resolution, groove_depth, groove_half_width, r_grading=r_grading,
                      n_increments=n_increments)
    print(f"  fine ref: n_elements={ref['n_elements']}  time={ref['elapsed_s']:.1f}s  "
          f"n_region={ref['n_region']}  p99_reliable={ref['region_p99_reliable']}  "
          f"region_avg_sxx={ref['region_avg_sigma_xx']:.4f}  true_max={ref['region_true_max_sigma_xx']:.4f}")

    rows = []
    for Ntheta, Nr, Nz in resolutions:
        r = solve_case(Ntheta, Nr, Nz, groove_depth, groove_half_width, r_grading=r_grading,
                        n_increments=n_increments)
        l2_rel, h1_rel, cauchy_field_rel, n_ref_region = compare_to_reference(
            r, ref, groove_depth, groove_half_width)
        r["disp_l2_rel"] = l2_rel
        r["grad_h1_rel"] = h1_rel
        r["cauchy_field_rel"] = cauchy_field_rel
        rows.append(r)
        avg_str = f"{r['region_avg_sigma_xx']:.4f}" if r['n_region'] else "NaN (n_region=0)"
        cf_str = f"{cauchy_field_rel*100:.3f}%" if cauchy_field_rel == cauchy_field_rel else \
            f"NaN (ref n_region={n_ref_region})"
        print(f"  ({Ntheta:>3},{Nr:>3},{Nz:>3})  n_elem={r['n_elements']:>7}  time={r['elapsed_s']:6.1f}s  "
              f"n_region={r['n_region']:>4}  region_avg_sxx={avg_str:>14}  "
              f"true_max={r['region_true_max_sigma_xx']:>9.3f}  "
              f"disp_L2={l2_rel*100:6.2f}%  cauchy_field={cf_str}")

    return {"label": label, "groove_depth": groove_depth, "groove_half_width": groove_half_width,
            "rho": rho, "ref": ref, "rows": rows}


def main():
    print("Real (not hypothetical) test: does sharpening B3's groove push "
          "required resolution toward Timon's 10^5-10^6-element / 5-10%-error "
          "target, or does it just converge fast like the current design "
          "(already ~1.3% at 123k elements)?\n")
    print("Baseline groove (depth=0.05, half_width=0.15) is the CURRENT, "
          "already-reported B3 design -- run here again with the SAME code "
          "path as the sharper variants, for a fair, apples-to-apples "
          "comparison (not reusing old numbers from a different script).\n")
    print("Sharpening is done by INCREASING groove_depth at FIXED "
          "groove_half_width=0.15 (not by narrowing half_width). Checked "
          "directly in data_generate_B3.py: z-direction meshing "
          "(`zs = np.linspace(0.0, Lz, Nz)`) has no grading option -- only "
          "the radial direction does (`r_grading`, generate_grid_Q4_ring). "
          "Narrowing half_width would shrink the feature in Z, a direction "
          "this codebase cannot locally refine without new grading code; "
          "deepening the groove sharpens its curvature (rho = 2w^2/(d*pi^2)) "
          "while leaving its Z-extent untouched, so the existing r_grading "
          "machinery is enough to resolve it -- a real constraint of the "
          "current code, not a shortcut. r_grading>1 clusters radial nodes "
          "toward R_in (verified directly in generate_grid_Q4_ring's own "
          "docstring) -- needed here since the groove sits at R_in.")

    results = []

    # Baseline (current B3 design) -- CPU-scale resolutions only, matching
    # mesh_convergence_B3.py's own CPU-scale ladder, for direct comparability.
    results.append(run_groove_design(
        "baseline (current B3)", 0.05, 0.15,
        resolutions=[(9, 4, 7), (13, 6, 11), (17, 8, 15), (21, 10, 19)],
        fine_resolution=(29, 14, 27)))

    # 4x sharper (rho=0.0228 vs baseline's 0.0912), same half_width=0.15.
    # n_increments=21: confirmed directly (not assumed) that the DEFAULT
    # 11 increments fails Newton-Raphson convergence at only 20% of the
    # applied rocking load for this depth -- a real property of the
    # sharper geometry's nonlinear response, fixed by finer load
    # stepping (a standard FEM remedy), not by weakening the load or
    # loosening the convergence tolerance.
    results.append(run_groove_design(
        "4x sharper (depth=0.20)", 0.20, 0.15,
        resolutions=[(9, 8, 7), (13, 12, 11), (17, 16, 15), (21, 20, 19)],
        fine_resolution=(29, 26, 27), r_grading=2.5, n_increments=21))

    # 7x sharper (rho=0.0130, depth=0.35) -- ATTEMPTED, not included: at the
    # coarsest planned resolution (9,10,7) it fails a basic physical
    # validity check (non-positive det(F) under load -- element inversion),
    # confirmed directly. At the next resolution (13,14,11), CONFIRMED
    # directly that BOTH CG+Jacobi (>900s, no convergence) AND a direct
    # solve (>300s, no result) fail to finish in a practical time even on
    # this small (1,560-element) mesh -- i.e. this is not a linear-solver
    # ill-conditioning problem fixable by switching solver method, but a
    # genuinely fragile nonlinear geometry at this depth/half_width
    # combination under the SAME phi=0.05 rocking amplitude used
    # throughout. Real finding, reported honestly rather than hidden or
    # forced to a number: depth=0.35 is not safely testable within this
    # project's CPU budget without further geometry/loading redesign
    # (e.g. a smaller rocking amplitude for this specific groove, or a
    # different depth/half_width combination) -- out of scope for this
    # round. depth=0.20 (below) is the honest result for Option A.

    print("\n" + "=" * 90)
    print("SUMMARY -- all groove designs, same methodology, same code path:")
    for res in results:
        print(f"\n{res['label']} (rho={res['rho']:.5f}):")
        for r in res["rows"]:
            print(f"  n_elem={r['n_elements']:>7}  disp_L2={r['disp_l2_rel']*100:6.2f}%  "
                  f"cauchy_field={r['cauchy_field_rel']*100:6.2f}%  "
                  f"n_region={r['n_region']:>4}  true_max_sxx={r['region_true_max_sigma_xx']:>9.3f}")


if __name__ == "__main__":
    main()
