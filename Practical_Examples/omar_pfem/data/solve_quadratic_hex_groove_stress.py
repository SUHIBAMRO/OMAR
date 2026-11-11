"""Timon's 2026-11-10 item 1c ("maybe higher-order elements"): does
switching from the project's standard 8-node trilinear hex (Hexa1) to
a 20-node quadratic serendipity hex (Hexa2) -- genuine p-refinement,
already built into torch-fem, no custom element code needed -- help
converge the groove-region stress, at the SAME element count that
global h-refinement (the original mesh-convergence ladder) never
converged?

torch-fem ships `linear_to_quadratic(nodes, elements)` to upgrade an
existing hex8 mesh to hex20 by adding edge-midpoint nodes. Used naively,
this places every midside node at the STRAIGHT-LINE average of its two
parent corners -- correct for a flat face, but WRONG for this mesh's
curved inner surface (r=R_in0 cylinder) and outer surface (r=R_out
cylinder): a straight chord between two points on a circle sits
strictly inside the true circle, and near the groove the surface is
additionally non-linear in z. Checked directly (not assumed): with
torch-fem's own node ordering (`linear_to_quadratic` appends new nodes
in exactly `Hexa1.edges` order, confirmed against a toy single-element
mesh), this means `boundary_node_sets`' tight (1e-9) tolerance would
silently FAIL to recognize most inner/outer-surface midside nodes as
boundary nodes at all -- an easy way to end up with an unconstrained
surface almost everywhere, including right at the groove, if not
caught. Fixed here by tracing each new midside node back to its two
parent corner nodes (via `Hexa1.edges` + the original hex8
connectivity) and, for any new node whose BOTH parents are on the
inner or outer surface, SNAPPING it exactly onto that surface (exact
r=groove_R_in(z) or r=R_OUT at the node's own angle) instead of
trusting the straight-line average -- the standard practice for
curved-boundary quadratic elements. Symmetry-plane (y=0) nodes need no
such fix (both parents already have y=0 exactly, so does their
average, checked directly).

Usage: python -m omar_pfem.data.solve_quadratic_hex_groove_stress
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
    groove_radius_of_curvature, groove_R_in)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.05, 0.15
E, NU = 1000.0, 0.45
PHI = 0.05


def build_quadratic_mesh_with_curved_surfaces(Ntheta, Nr, Nz, r_grading=1.0):
    from torchfem.elements import Hexa1, linear_to_quadratic

    nodes8, elements8 = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=r_grading)
    inner8, outer8, sym8 = boundary_node_sets(nodes8, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    n_nodes8 = nodes8.shape[0]

    nodes8_t = torch.tensor(nodes8, dtype=torch.float64)
    elements8_t = torch.tensor(elements8, dtype=torch.long)
    nodes20_t, elements20_t = linear_to_quadratic(nodes8_t, elements8_t)
    nodes20 = nodes20_t.numpy().copy()
    elements20 = elements20_t.numpy()

    edges = Hexa1.edges.numpy()  # (12, 2), LOCAL corner indices per edge
    mid_global = elements20[:, 8:20]          # (n_elem, 12) global midside indices
    parent_a = elements8[:, edges[:, 0]]       # (n_elem, 12) global parent-corner indices
    parent_b = elements8[:, edges[:, 1]]
    flat_mid, flat_a, flat_b = mid_global.ravel(), parent_a.ravel(), parent_b.ravel()
    uniq_mid, first_idx = np.unique(flat_mid, return_index=True)
    assert np.array_equal(uniq_mid, np.arange(n_nodes8, n_nodes8 + len(uniq_mid))), \
        "new node indices not contiguous from n_nodes8 as expected -- torch-fem's own convention changed"
    parent_pairs = np.stack([flat_a[first_idx], flat_b[first_idx]], axis=1)  # (n_new, 2)

    is_new_inner = inner8[parent_pairs[:, 0]] & inner8[parent_pairs[:, 1]]
    is_new_outer = outer8[parent_pairs[:, 0]] & outer8[parent_pairs[:, 1]]

    new_slice = slice(n_nodes8, None)
    xy = nodes20[new_slice, 0:2]
    theta_mid = np.arctan2(xy[:, 1], xy[:, 0])
    z_mid = nodes20[new_slice, 2]

    r_inner_exact = groove_R_in(z_mid, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    nodes20[new_slice, 0][is_new_inner] = (r_inner_exact * np.cos(theta_mid))[is_new_inner]
    nodes20[new_slice, 1][is_new_inner] = (r_inner_exact * np.sin(theta_mid))[is_new_inner]
    nodes20[new_slice, 0][is_new_outer] = (R_OUT * np.cos(theta_mid))[is_new_outer]
    nodes20[new_slice, 1][is_new_outer] = (R_OUT * np.sin(theta_mid))[is_new_outer]

    inner20, outer20, sym20 = boundary_node_sets(
        nodes20, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    n_new_inner_expected = int(is_new_inner.sum())
    n_new_outer_expected = int(is_new_outer.sum())
    assert int(inner20[new_slice].sum()) == n_new_inner_expected, \
        "snapping did not make every intended new inner node pass the exact boundary check"
    assert int(outer20[new_slice].sum()) == n_new_outer_expected, \
        "snapping did not make every intended new outer node pass the exact boundary check"
    # symmetry-plane midside nodes need no snapping: both parents have y=0 exactly,
    # so their straight-line average does too -- checked, not assumed.
    assert np.abs(nodes20[new_slice, 1][sym20[new_slice] == False]).min() > 0 or sym8.sum() == 0

    return nodes20, elements20, inner20, outer20, sym20


def solve_and_measure_groove_region_quadratic(Ntheta, Nr, Nz, r_grading=1.0,
                                               dtype=torch.float64, device=None, verbose=False):
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

    P = P.transpose(0, 1)  # (n_elem, n_gauss, 3, 3)
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
    detJ_gauss = detJ_gauss.transpose(0, 1)  # (n_elem, n_gauss)
    detJ_gauss_np = detJ_gauss.cpu().numpy()
    vol_weight = iweights_np[None, :] * np.abs(detJ_gauss_np)

    P_gauss_np, F_gauss_np = P.cpu().numpy(), F.cpu().numpy()
    detF_gauss_np = detF_gauss.cpu().numpy()
    sigma_gauss = (np.einsum("egij,egjk->egik", P_gauss_np, F_gauss_np.transpose(0, 1, 3, 2))
                   / detF_gauss_np[..., None, None])

    nodes_per_elem = nodes_t[elements_t]  # (n_elem, 20, 3)
    pos_g = torch.einsum("gn,enD->egD", N_gauss, nodes_per_elem).cpu().numpy()  # (n_elem, n_gauss, 3)

    z_ref = LZ / 2.0
    r_ref = R_IN0 - GROOVE_DEPTH
    ref_point = np.array([r_ref, 0.0, z_ref])
    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)

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
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz, "n_elem": elements.shape[0], "n_nodes": n_nodes,
        "n_region_gauss": n_region, "region_avg_sigma_xx": region_avg_xx,
        "region_max_sigma_xx": region_max_xx, "region_avg_sigma_frob": region_avg_frob,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    print(f"{'Ntheta':>6} {'Nr':>4} {'Nz':>4} {'n_elem':>7} {'n_nodes':>8} "
          f"{'n_region_gauss':>15} {'avg|sigma_xx|':>14} {'max|sigma_xx|':>14} "
          f"{'avg|sigma|_F':>13} {'time(s)':>8}")
    for Ntheta, Nr, Nz in [(13, 12, 19), (17, 16, 25)]:
        r = solve_and_measure_groove_region_quadratic(Ntheta, Nr, Nz)
        print(f"{r['Ntheta']:>6} {r['Nr']:>4} {r['Nz']:>4} {r['n_elem']:>7} {r['n_nodes']:>8} "
              f"{r['n_region_gauss']:>15} {r['region_avg_sigma_xx']:>14.4f} "
              f"{r['region_max_sigma_xx']:>14.4f} {r['region_avg_sigma_frob']:>13.4f} {r['elapsed_s']:>8.2f}")
