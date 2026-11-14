"""Omar's explicit instruction (2026-11-14): the just-discovered
non-monotonic max|sigma_xx| under GLOBAL uniform quadratic refinement
(0.7372 at 23,520 elem -> 0.7064 at 33,120 -> 0.8911 at 45,000) means
GCI/Richardson extrapolation cannot be trusted on that statistic. This
project's own earlier finding (Item 1a, LINEAR elements) was that LOCAL
z-clustering at the groove -- not global refinement -- is what actually
governs peak-stress convergence there. This module is the QUADRATIC
(Hexa2) counterpart of that same idea, at a FIXED (Ntheta, Nr, Nz) --
same total element count throughout -- solving with increasing z_beta
(more of those same elements clustered physically at the groove).

Built by mirroring solve_quadratic_hex_groove_stress.py's own
build_quadratic_mesh_with_curved_surfaces / solve_and_measure_groove_
region_quadratic almost verbatim (same curved-surface midside-node
snapping, same region-stress QoI machinery), with the ONE change being
the mesh builder call: generate_grid_hex8_bushing_z_refined (sinh-
graded z-layers) instead of generate_grid_hex8_bushing (uniform
linspace z-layers) -- the same non-destructive, additive-script pattern
already used for the linear z_beta study (solve_z_refined_groove_
stress.py), so neither already-validated module is touched.

z_beta=0.0 reproduces the uniform-z quadratic mesh exactly (inherited
from z_layers_sinh_graded's own verified beta=0 behavior).

Usage: python -m omar_pfem.data.solve_quadratic_hex_groove_stress_z_refined
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    boundary_node_sets, rigid_rotation_displacement,
    groove_radius_of_curvature, groove_R_in)
from omar_pfem.data.data_generate_B3_local_z_refine import generate_grid_hex8_bushing_z_refined
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH

R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
E, NU = 1000.0, 0.45
PHI = 0.05


def build_quadratic_mesh_with_curved_surfaces_z_refined(Ntheta, Nr, Nz, z_beta,
                                                          r_grading=1.0, groove_depth=None):
    """Identical to solve_quadratic_hex_groove_stress.
    build_quadratic_mesh_with_curved_surfaces, except the underlying
    hex8 mesh comes from generate_grid_hex8_bushing_z_refined (sinh-
    graded z-layers) instead of generate_grid_hex8_bushing (uniform
    z-layers) -- the one and only change. The quadratic-upgrade +
    curved-surface midside-node snapping logic is untouched and does
    not care whether z-spacing is uniform or graded (it only looks at
    each new node's own (theta, z) and its two parent corners)."""
    from torchfem.elements import Hexa1, linear_to_quadratic

    groove_depth = GROOVE_DEPTH if groove_depth is None else groove_depth
    nodes8, elements8 = generate_grid_hex8_bushing_z_refined(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, groove_depth, GROOVE_HALF_WIDTH,
        r_grading=r_grading, z_beta=z_beta)
    inner8, outer8, sym8 = boundary_node_sets(nodes8, R_IN0, R_OUT, LZ, groove_depth, GROOVE_HALF_WIDTH)
    n_nodes8 = nodes8.shape[0]

    nodes8_t = torch.tensor(nodes8, dtype=torch.float64)
    elements8_t = torch.tensor(elements8, dtype=torch.long)
    nodes20_t, elements20_t = linear_to_quadratic(nodes8_t, elements8_t)
    nodes20 = nodes20_t.numpy().copy()
    elements20 = elements20_t.numpy()

    edges = Hexa1.edges.numpy()
    mid_global = elements20[:, 8:20]
    parent_a = elements8[:, edges[:, 0]]
    parent_b = elements8[:, edges[:, 1]]
    flat_mid, flat_a, flat_b = mid_global.ravel(), parent_a.ravel(), parent_b.ravel()
    uniq_mid, first_idx = np.unique(flat_mid, return_index=True)
    assert np.array_equal(uniq_mid, np.arange(n_nodes8, n_nodes8 + len(uniq_mid))), \
        "new node indices not contiguous from n_nodes8 as expected -- torch-fem's own convention changed"
    parent_pairs = np.stack([flat_a[first_idx], flat_b[first_idx]], axis=1)

    is_new_inner = inner8[parent_pairs[:, 0]] & inner8[parent_pairs[:, 1]]
    is_new_outer = outer8[parent_pairs[:, 0]] & outer8[parent_pairs[:, 1]]

    new_slice = slice(n_nodes8, None)
    xy = nodes20[new_slice, 0:2]
    theta_mid = np.arctan2(xy[:, 1], xy[:, 0])
    z_mid = nodes20[new_slice, 2]

    r_inner_exact = groove_R_in(z_mid, LZ, R_IN0, groove_depth, GROOVE_HALF_WIDTH)
    nodes20[new_slice, 0][is_new_inner] = (r_inner_exact * np.cos(theta_mid))[is_new_inner]
    nodes20[new_slice, 1][is_new_inner] = (r_inner_exact * np.sin(theta_mid))[is_new_inner]
    nodes20[new_slice, 0][is_new_outer] = (R_OUT * np.cos(theta_mid))[is_new_outer]
    nodes20[new_slice, 1][is_new_outer] = (R_OUT * np.sin(theta_mid))[is_new_outer]

    inner20, outer20, sym20 = boundary_node_sets(
        nodes20, R_IN0, R_OUT, LZ, groove_depth, GROOVE_HALF_WIDTH)
    n_new_inner_expected = int(is_new_inner.sum())
    n_new_outer_expected = int(is_new_outer.sum())
    assert int(inner20[new_slice].sum()) == n_new_inner_expected, \
        "snapping did not make every intended new inner node pass the exact boundary check"
    assert int(outer20[new_slice].sum()) == n_new_outer_expected, \
        "snapping did not make every intended new outer node pass the exact boundary check"
    assert np.abs(nodes20[new_slice, 1][sym20[new_slice] == False]).min() > 0 or sym8.sum() == 0

    return nodes20, elements20, inner20, outer20, sym20


def solve_and_measure_groove_region_quadratic_z_refined(
        Ntheta, Nr, Nz, z_beta, r_grading=1.0, dtype=torch.float64, device=None, verbose=False,
        groove_depth=None, ref_point=None, region_radius=None):
    """Identical to solve_quadratic_hex_groove_stress.
    solve_and_measure_groove_region_quadratic, except the mesh comes
    from build_quadratic_mesh_with_curved_surfaces_z_refined (z_beta-
    graded) instead of the uniform-z builder."""
    device = device or torch.device("cpu")
    groove_depth_eff = GROOVE_DEPTH if groove_depth is None else groove_depth
    nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces_z_refined(
        Ntheta, Nr, Nz, z_beta, r_grading=r_grading, groove_depth=groove_depth_eff)

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
    assert model.etype.__name__ == "Hexa2", f"expected Hexa2, got {model.etype.__name__}"

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

    P = P.transpose(0, 1)
    F = F.transpose(0, 1)
    detF_gauss = torch.linalg.det(F)
    assert (detF_gauss > 0).all(), "non-positive det(F) at a Gauss point -- element inversion"

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            N_gauss, _, detJ_gauss = model.eval_shape_functions(model.etype.ipoints)
    finally:
        torch.set_default_dtype(old_default_dtype)
    iweights_np = model.etype.iweights.cpu().numpy()
    detJ_gauss = detJ_gauss.transpose(0, 1)
    detJ_gauss_np = detJ_gauss.cpu().numpy()
    vol_weight = iweights_np[None, :] * np.abs(detJ_gauss_np)

    P_gauss_np, F_gauss_np = P.cpu().numpy(), F.cpu().numpy()
    detF_gauss_np = detF_gauss.cpu().numpy()
    sigma_gauss = (np.einsum("egij,egjk->egik", P_gauss_np, F_gauss_np.transpose(0, 1, 3, 2))
                   / detF_gauss_np[..., None, None])

    nodes_per_elem = nodes_t[elements_t]
    pos_g = torch.einsum("gn,enD->egD", N_gauss, nodes_per_elem).cpu().numpy()

    if ref_point is None:
        z_ref = LZ / 2.0
        r_ref = R_IN0 - groove_depth_eff
        ref_point = np.array([r_ref, 0.0, z_ref])
    if region_radius is None:
        region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)

    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    region_mask = dist < region_radius
    n_region = int(region_mask.sum())

    sigma_flat = sigma_gauss[region_mask]
    w_flat = vol_weight[region_mask]
    w_sum = w_flat.sum()

    COMPONENTS = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
    per_component = {}
    for name, i, j in COMPONENTS:
        vals = sigma_flat[:, i, j]
        avg = float((vals * w_flat).sum() / w_sum) if n_region > 0 and w_sum > 0 else float("nan")
        mx = float(np.abs(vals).max()) if n_region > 0 else float("nan")
        per_component[name] = {"avg": avg, "max": mx}

    frob = np.sqrt(np.einsum("nij,nij->n", sigma_flat, sigma_flat)) if n_region > 0 else np.array([])
    region_avg_frob = float((frob * w_flat).sum() / w_sum) if n_region > 0 and w_sum > 0 else float("nan")

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz, "z_beta": z_beta,
        "n_elem": elements.shape[0], "n_nodes": n_nodes,
        "n_region_gauss": n_region,
        "region_avg_sigma_xx": per_component["xx"]["avg"],
        "region_max_sigma_xx": per_component["xx"]["max"],
        "region_avg_sigma_frob": region_avg_frob,
        "per_component": per_component,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    for beta in (0.0, 1.0, 2.0):
        r = solve_and_measure_groove_region_quadratic_z_refined(21, 18, 31, beta)
        print(f"z_beta={beta:4.1f}  n_elem={r['n_elem']}  "
              f"avg|sigma_xx|={r['region_avg_sigma_xx']:.4f}  max|sigma_xx|={r['region_max_sigma_xx']:.4f}  "
              f"elapsed={r['elapsed_s']:.1f}s")
