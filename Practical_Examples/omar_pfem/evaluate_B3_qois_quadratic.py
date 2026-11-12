"""Professor Rabczuk's 2026-11-12 email: evaluate PI and the DD
variants against the now-converged QUADRATIC (Q20) groove reference,
report the quadratic meshes' DOF, and include the SCF. This module
builds the Q20-equivalent of evaluate_B3_qois_finer.py's own
`generate_fine_reference`, reusing as much of the existing, validated
machinery as possible rather than reimplementing it:

- `compute_region_sigma_full`, `compute_region_sigma_xx`,
  `compute_energy`, `compute_reaction`, `compute_H1_seminorm_rel`
  (omar_pfem.evaluate_B3_qois) are already fully generic over element
  type -- they only read geom["elements"]/["B"]/["detJ"]/["iweights"]
  and index u by geom["elements"], never hardcoding 8 nodes/element
  anywhere (confirmed directly by reading their source before relying
  on this). So a Q20 `geom` dict with the same keys works UNCHANGED.
- `evaluate_u_only_checkpoint_against_fine_reference` (same reasons)
  is reused completely unmodified -- it queries the network directly at
  geom_fine["nodes"] (the operator is coordinate-based, not tied to a
  fixed mesh), so no interpolation of the NETWORK's own prediction onto
  the Q20 mesh is needed at all.

What genuinely needs new code: (1) a Q20 `build_fixed_geometry`
equivalent (nodes/elements from the already-validated, curved-surface-
snapped build_quadratic_mesh_with_curved_surfaces, plus theta_node/
t_node recovered by inverting the EXACT physical mapping -- r =
R_in_eff(z) + (R_out-R_in_eff(z))*t, theta = atan2(y,x) -- rather than
assuming linear_to_quadratic's own node ordering follows any parametric
formula, which it does not for the new midside nodes); (2) a Q20
region mask (same ref_point/radius, physical Gauss positions via the
model's own shape functions); (3) a Q20 solve_one_sample (heterogeneous
per-node E/nu, same averaging-over-element-nodes construction
solve_one_sample already uses, which is itself already node-count-
agnostic); (4) interpolating each held-out sample's E_node/nu_node from
the training mesh onto the Q20 mesh's nodes, reusing
mesh_convergence_B3._field_interpolator directly at custom (theta_node,
t_node, z) query points instead of its usual structured-grid query.
"""
import os

import h5py
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import rigid_rotation_displacement, groove_R_in, groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import (
    GROOVE_DEPTH, GROOVE_HALF_WIDTH, R_GRADING, DEFAULT_RESOLUTION)
from omar_pfem.data.mesh_convergence_B3 import R_IN0, R_OUT, LZ, _theta_t_axes, _field_interpolator
from omar_pfem.data.solve_quadratic_hex_groove_stress import build_quadratic_mesh_with_curved_surfaces
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

MIN_RELIABLE_N_P99 = 20


def _invert_physical_to_parametric(nodes_np):
    """theta = atan2(y,x) (exact, any node); t = (r - R_in_eff(z)) /
    (R_out - R_in_eff(z)) (exact, since every node -- corner or
    midside -- sits at its own TRUE physical radius after the curved-
    surface snapping already done in build_quadratic_mesh_with_curved_
    surfaces; inner/outer surface nodes land at EXACTLY t=0/1 by
    construction, not approximately)."""
    x, y, z = nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2]
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    r_in_eff = groove_R_in(z, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    t = (r - r_in_eff) / (R_OUT - r_in_eff)
    return theta, np.clip(t, 0.0, 1.0)


def build_fixed_geometry_quadratic(Ntheta, Nr, Nz, device, dtype=torch.float64, r_grading=R_GRADING):
    nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces(
        Ntheta, Nr, Nz, r_grading=r_grading)
    theta_node, t_node = _invert_physical_to_parametric(nodes)

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        from torchfem import Solid
        from torchfem.materials import Hyperelastic3D

        nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
        elements_t = torch.tensor(elements, dtype=torch.long, device=device)
        dummy_params = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=dummy_params)
        with torch.device(device):
            model = Solid(nodes_t, elements_t, material)
        assert model.etype.__name__ == "Hexa2", f"expected Hexa2, got {model.etype.__name__}"
        N, B, detJ = model.eval_shape_functions(model.etype.ipoints)
        iweights = model.etype.iweights.to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(old_default_dtype)

    return {
        "nodes": nodes_t, "elements": elements_t,
        "inner": torch.tensor(inner, device=device), "outer": torch.tensor(outer, device=device),
        "sym": torch.tensor(sym, device=device),
        "theta_node": torch.tensor(theta_node, dtype=dtype, device=device),
        "t_node": torch.tensor(t_node, dtype=dtype, device=device),
        "N": N, "B": B, "detJ": detJ, "iweights": iweights,
        "n_nodes": nodes.shape[0], "n_elements": elements.shape[0],
        "dof": nodes.shape[0] * 3,
    }


def build_region_mask_quadratic(geom, ref_point=None, region_radius=None):
    nodes_t = geom["nodes"]
    N_gauss = geom["N"]
    elements_t = geom["elements"]
    nodes_per_elem = nodes_t[elements_t]
    pos_g = torch.einsum("gn,enD->egD", N_gauss, nodes_per_elem).cpu().numpy()

    if ref_point is None:
        ref_point = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
    if region_radius is None:
        region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    mask_np = dist < region_radius
    return torch.tensor(mask_np, device=nodes_t.device), int(mask_np.sum())


def solve_one_sample_quadratic(Ntheta, Nr, Nz, E_node, nu_node, phi, r_grading=R_GRADING,
                                n_increments=11, dtype=torch.float64, device=None, verbose=False):
    device = device or torch.device("cpu")
    nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces(
        Ntheta, Nr, Nz, r_grading=r_grading)

    assert E_node.shape[0] == nodes.shape[0], "E_node must be sampled at this exact Q20 mesh's own nodes"
    assert nu_node.shape[0] == nodes.shape[0], "nu_node must be sampled at this exact Q20 mesh's own nodes"
    E_elem = E_node[elements].mean(axis=1)
    nu_elem = nu_node[elements].mean(axis=1)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    params = torch.tensor(np.stack([mu_elem, lam_elem], axis=1), dtype=dtype, device=device)

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)
    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
    assert material.is_vectorized, "per-element params did not vectorize as expected"
    with torch.device(device):
        model = Solid(nodes_t, elements_t, material)

    n_nodes = nodes.shape[0]
    model.forces = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=device)
    displacements = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    constraints[outer, :] = True
    ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, phi)
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
    import time
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
    f_np = f.cpu().numpy()
    reaction_force = f_np[outer].sum(axis=0)
    total_force = f_np.sum(axis=0)
    force_rel_residual = float(np.linalg.norm(total_force) / max(np.linalg.norm(reaction_force), 1e-8))

    return {"u": u_np, "force_rel_residual": force_rel_residual, "elapsed_s": elapsed}


def generate_fine_reference_quadratic(dataset_path, fine_resolution, n_samples, device, dtype=torch.float64,
                                       verbose=True, cache_path=None, n_increments=11,
                                       r_grading=R_GRADING):
    Ntheta_f, Nr_f, Nz_f = fine_resolution
    Ntheta_c, Nr_c, Nz_c = DEFAULT_RESOLUTION

    geom_fine = build_fixed_geometry_quadratic(Ntheta_f, Nr_f, Nz_f, device, dtype=dtype, r_grading=r_grading)
    region_mask_fine_torch, n_region_fine = build_region_mask_quadratic(geom_fine)
    if verbose:
        print(f"Q20 fine geometry: {geom_fine['n_elements']} elements, {geom_fine['n_nodes']} nodes, "
              f"DOF={geom_fine['dof']}")
        print(f"  fine-mesh fixed region: {n_region_fine} Gauss points "
              f"(reliable needs >= {MIN_RELIABLE_N_P99})")

    with h5py.File(dataset_path, "r") as h5:
        E_node_all = h5["E_node"][:]
        nu_node_all = h5["nu_node"][:]
        phi_all = h5["phi"][:]
    n_available = E_node_all.shape[0]
    n_samples = min(n_samples, n_available)

    thetas_c, ts_c, zs_c = _theta_t_axes(Ntheta_c, Nr_c, Nz_c, r_grading)
    theta_q = geom_fine["theta_node"].cpu().numpy()
    t_q = geom_fine["t_node"].cpu().numpy()
    z_q = geom_fine["nodes"][:, 2].cpu().numpy()

    samples = []
    done_indices = set()
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location=device, weights_only=False)
        if cached.get("fine_resolution") == [Ntheta_f, Nr_f, Nz_f]:
            samples = cached["samples"]
            done_indices = {s["sample_index"] for s in samples}
            if verbose:
                print(f"  resuming from {cache_path}: {len(done_indices)} sample(s) already solved")

    for idx in range(n_samples):
        if idx in done_indices:
            continue
        E_c = E_node_all[idx].astype(np.float64)
        nu_c = nu_node_all[idx].astype(np.float64)
        phi = float(phi_all[idx])

        interp_E = _field_interpolator(E_c.reshape(-1, 1), Ntheta_c, Nr_c, Nz_c, thetas_c, ts_c, zs_c, 1)
        interp_nu = _field_interpolator(nu_c.reshape(-1, 1), Ntheta_c, Nr_c, Nz_c, thetas_c, ts_c, zs_c, 1)
        E_f = interp_E(theta_q, t_q, z_q).reshape(-1)
        nu_f = interp_nu(theta_q, t_q, z_q).reshape(-1)

        r = solve_one_sample_quadratic(Ntheta_f, Nr_f, Nz_f, E_f, nu_f, phi, r_grading=r_grading,
                                        device=device, dtype=dtype, n_increments=n_increments)
        u_true_f = torch.tensor(r["u"], dtype=dtype, device=device).unsqueeze(0)
        E_f_t = torch.tensor(E_f, dtype=dtype, device=device).unsqueeze(0)
        nu_f_t = torch.tensor(nu_f, dtype=dtype, device=device).unsqueeze(0)
        phi_t = torch.tensor([phi], dtype=dtype, device=device)

        samples.append({
            "sample_index": idx, "phi": phi,
            "E_f_t": E_f_t, "nu_f_t": nu_f_t, "phi_t": phi_t, "u_true_f": u_true_f,
            "force_rel_residual": r["force_rel_residual"], "elapsed_s": r["elapsed_s"],
        })
        samples.sort(key=lambda s: s["sample_index"])
        if verbose:
            print(f"  [{idx + 1}/{n_samples}] Q20 fine solve elapsed_s={r['elapsed_s']:.2f}s "
                  f"force_rel_residual={r['force_rel_residual']:.2e}", flush=True)
        if cache_path:
            tmp_path = cache_path + ".tmp"
            torch.save({"fine_resolution": [Ntheta_f, Nr_f, Nz_f], "samples": samples}, tmp_path)
            os.replace(tmp_path, cache_path)

    return {"geom_fine": geom_fine, "region_mask_fine_torch": region_mask_fine_torch,
            "fine_resolution": [Ntheta_f, Nr_f, Nz_f], "coarse_resolution": [Ntheta_c, Nr_c, Nz_c],
            "n_region_fine": n_region_fine, "samples": samples}
