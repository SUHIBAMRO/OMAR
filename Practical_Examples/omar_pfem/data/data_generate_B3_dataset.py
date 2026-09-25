"""B3 dataset generation for the Transolver/VINO training set (2026-09-25
scope decision, confirmed against Prof. Rabczuk's own reply that "VINO"
means this project's own Transolver, not a different architecture -- so
no training-paradigm change was needed, and this file follows exactly
the same per-sample randomization philosophy already used by
data_generate_B1.py/data_generate_B2.py, extended from 2D to 3D):

- Geometry is FIXED across every sample, at Section 11.2's final chosen
  production settings (groove_depth=0.20, groove_half_width=0.15,
  r_grading=1.0) -- matching B1/B2's own convention of a fixed geometry
  per dataset.
- The rubber's material fields, Young's modulus E(theta,r,z) and
  Poisson's ratio nu(theta,r,z), vary spatially per sample, generated as
  3D Gaussian random fields (grf.generate_gaussian_random_field_3d).
- The loading varies through the rigid core's rocking angle, phi, a
  single scalar per sample (not a spatial field, since the applied load
  here is a rigid-body rotation amplitude, not a distributed pressure).

Per-node fields are generated DIRECTLY on the mesh's own structured
(theta, r, z) parametric grid (Ntheta, Nr, Nz) and mapped onto the node
array using generate_grid_hex8_bushing's own exact node-ordering
convention (verified against mesh_convergence_B3.py's own
_theta_t_axes/_element_ijk: node index = k*(Ntheta*Nr) + j*Nr + i, where
j/i/k are the theta/r/z structured indices) -- no interpolation from a
separate "fine" GRF grid is needed, unlike B1/B2, since this dataset
uses one fixed mesh resolution throughout (the same convention B2 itself
uses: geometry and mesh resolution are both fixed per dataset).

torch-fem's own Hyperelastic3D material class natively supports a
per-element params tensor (confirmed directly against the installed
torch-fem 0.11.0 source: `is_vectorized = self.params.dim() > 1`, no
custom solver code needed) -- each element's local (mu, lambda) is
derived from the mean of its own 8 nodes' sampled E/nu.

The exact E/nu/phi sampling ranges (E_mean=1000.0/E_std=200.0 matching
B2's own defaults; nu_mean=0.45/nu_std=0.02 clipped to (0.40, 0.49) to
respect B3's existing incompressible-rubber choice and avoid volumetric
locking; phi_mean=0.05/phi_std=0.02, matching PHI=0.05's own value from
the mesh-convergence study as the mean, with a spread mirroring B2's
~40% relative load variation) are the defaults proposed in this
session's earlier discussion -- NOT independently re-verified against
new physical evidence, and should be treated as a starting point, not a
locked-in choice, until checked against real solved samples at
production scale.
"""
import json
import os
import time

import h5py
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import (
    generate_grid_hex8_bushing, boundary_node_sets, rigid_rotation_displacement,
)
from omar_pfem.data.grf import generate_gaussian_random_field_3d
from omar_pfem.data.mesh_convergence_B3 import R_IN0, R_OUT, LZ
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.20, 0.15  # FIXED, matching Section 11.2's chosen production geometry
R_GRADING = 1.0


def sample_material_and_load(
    Ntheta, Nr, Nz, seed,
    E_mean=1000.0, E_std=200.0,
    nu_mean=0.45, nu_std=0.02, nu_clip=(0.40, 0.49),
    phi_mean=0.05, phi_std=0.02, phi_clip=(0.01, 0.15),
    correlation_length=0.5,
):
    """Samples one instance of the per-node E/nu fields and the scalar
    rocking angle phi, directly on the mesh's own (Ntheta, Nr, Nz)
    structured grid -- see module docstring for the exact node-ordering
    mapping this relies on."""
    E_field, *_ = generate_gaussian_random_field_3d(
        np.pi, 1.0, LZ, Ntheta, Nr, Nz, mean=E_mean, std=E_std,
        correlation_length=correlation_length, seed=seed)
    nu_field_raw, *_ = generate_gaussian_random_field_3d(
        np.pi, 1.0, LZ, Ntheta, Nr, Nz, mean=nu_mean, std=nu_std,
        correlation_length=correlation_length,
        seed=(seed + 1000 if seed is not None else None))
    nu_field = np.clip(nu_field_raw, nu_clip[0], nu_clip[1])

    rng = np.random.RandomState(seed + 2000 if seed is not None else None)
    phi = float(np.clip(rng.normal(phi_mean, phi_std), phi_clip[0], phi_clip[1]))

    # field[j, i, k] (theta, r, z) -> node index k*(Ntheta*Nr) + j*Nr + i,
    # i.e. transpose to (k, j, i) then ravel in C order.
    E_node = E_field.transpose(2, 0, 1).ravel()
    nu_node = nu_field.transpose(2, 0, 1).ravel()
    return E_node, nu_node, phi, E_field, nu_field


def solve_one_sample(
    Ntheta, Nr, Nz, E_node, nu_node, phi,
    groove_depth=GROOVE_DEPTH, groove_half_width=GROOVE_HALF_WIDTH, r_grading=R_GRADING,
    n_increments=11, dtype=torch.float64, device=None, verbose=False,
    method="cg", preconditioner="jacobi",
):
    """Solves one B3 case with spatially-varying material and a sampled
    rocking angle, via torch-fem's native per-element material support
    (no custom solver code)."""
    device = device or torch.device("cpu")
    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, groove_depth, groove_half_width, r_grading=r_grading)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, groove_depth, groove_half_width)

    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D

    assert E_node.shape[0] == nodes.shape[0], "E_node must be sampled at this exact (Ntheta,Nr,Nz)"
    assert nu_node.shape[0] == nodes.shape[0], "nu_node must be sampled at this exact (Ntheta,Nr,Nz)"

    E_elem = E_node[elements].mean(axis=1)
    nu_elem = nu_node[elements].mean(axis=1)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    params = torch.tensor(np.stack([mu_elem, lam_elem], axis=1), dtype=dtype, device=device)

    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)
    material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
    assert material.is_vectorized, "per-element params did not vectorize the material as expected"
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
    t0 = time.time()
    try:
        with torch.device(device), torch.no_grad():
            u, f, P, F, state = model.solve(
                increments=increments, max_iter=30, rtol=1e-8, atol=1e-8, stol=1e-8,
                method=method, preconditioner=preconditioner, nlgeom=True, verbose=verbose,
                aggregate_integration_points=False)
    finally:
        torch.set_default_dtype(old_default_dtype)
    elapsed = time.time() - t0

    u_np = u.cpu().numpy()
    if not np.isfinite(u_np).all():
        raise RuntimeError("NaN/Inf in solution -- Newton did not really converge cleanly")

    f_np = f.cpu().numpy()
    total_force = f_np.sum(axis=0)
    reaction_force = f_np[outer].sum(axis=0)
    force_scale = max(np.linalg.norm(reaction_force), 1e-8)
    force_rel_residual = float(np.linalg.norm(total_force) / force_scale)

    return {
        "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz,
        "n_nodes": n_nodes, "n_elements": len(elements),
        "nodes": nodes, "elements": elements,
        "u": u_np, "phi": phi,
        "E_node": E_node, "nu_node": nu_node,
        "force_rel_residual": force_rel_residual,
        "elapsed_s": elapsed,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
    }


def generate_dataset(
    num_samples, output_dir, Ntheta, Nr, Nz, seed=0, device=None,
    dist_kwargs=None, verbose_every=10,
):
    """Serial dataset generation with a resumable JSON manifest (simpler
    than B2's multiprocessing pipeline -- worth adding later once this is
    actually run at production scale; correctness-first for now, matching
    the standing project discipline of not trusting untested code paths
    with expensive GPU time)."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    h5_path = os.path.join(output_dir, "dataset.h5")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["num_samples"] == num_samples, \
            f"manifest num_samples mismatch ({manifest['num_samples']} != {num_samples})"
        assert manifest["Ntheta"] == Ntheta and manifest["Nr"] == Nr and manifest["Nz"] == Nz
        done = set(manifest["successful_samples"])
        failed = dict(manifest["failed"])
    else:
        manifest = {"num_samples": num_samples, "Ntheta": Ntheta, "Nr": Nr, "Nz": Nz,
                    "successful_samples": [], "failed": {}}
        done = set()
        failed = {}

    with h5py.File(h5_path, "a") as h5:
        n_nodes_ref = h5["E_node"].shape[1] if "E_node" in h5 else None
        for i in range(num_samples):
            if i in done:
                continue
            sample_seed = seed * 10000 + i
            E_node, nu_node, phi, _, _ = sample_material_and_load(
                Ntheta, Nr, Nz, sample_seed, **(dist_kwargs or {}))
            try:
                r = solve_one_sample(Ntheta, Nr, Nz, E_node, nu_node, phi, device=device)
            except Exception as e:
                failed[str(i)] = str(e)
                if (i + 1) % verbose_every == 0 or i == num_samples - 1:
                    print(f"  [{i + 1}/{num_samples}] FAILED: {e}")
                continue

            if n_nodes_ref is None:
                n_nodes_ref = r["n_nodes"]
                h5.create_dataset("displacements", (num_samples, n_nodes_ref, 3), dtype="f4")
                h5.create_dataset("E_node", (num_samples, n_nodes_ref), dtype="f4")
                h5.create_dataset("nu_node", (num_samples, n_nodes_ref), dtype="f4")
                h5.create_dataset("phi", (num_samples,), dtype="f4")
                h5.create_dataset("force_rel_residual", (num_samples,), dtype="f4")
                h5.create_dataset("elapsed_s", (num_samples,), dtype="f4")

            h5["displacements"][i] = r["u"].astype("f4")
            h5["E_node"][i] = r["E_node"].astype("f4")
            h5["nu_node"][i] = r["nu_node"].astype("f4")
            h5["phi"][i] = r["phi"]
            h5["force_rel_residual"][i] = r["force_rel_residual"]
            h5["elapsed_s"][i] = r["elapsed_s"]
            done.add(i)

            if (i + 1) % verbose_every == 0 or i == num_samples - 1:
                print(f"  [{i + 1}/{num_samples}] elements={r['n_elements']:,} "
                      f"time={r['elapsed_s']:.2f}s force_rel_residual={r['force_rel_residual']:.2e} "
                      f"max_disp={r['max_disp']:.4f}")

            manifest["successful_samples"] = sorted(done)
            manifest["failed"] = failed
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    print(f"\nDone: {len(done)}/{num_samples} succeeded, {len(failed)} failed.")
    return manifest
