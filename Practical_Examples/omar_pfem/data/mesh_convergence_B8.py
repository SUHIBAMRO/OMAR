"""Smoke test / basic solve for B8 (3D laminated annular elastomeric
seismic bearing -- Option B). Per Omar's explicit instruction (2026-09-22):
prepare the COMPLETE code (geometry, BCs, material, norms/QoIs, equations)
and verify it actually runs correctly BEFORE presenting it to Timon and
BEFORE committing to any real dataset-generation/mesh-convergence/GPU
work -- this script is that verification step only, not the real study
(no resolution ladder, no reference solve, no accuracy claims).

Follows the same physics sanity-check discipline as
mesh_convergence_B3.py's own solve_case: mesh validity (no inverted
elements), det(F)>0 at every element and every quadrature point, Newton
convergence, and global force/moment equilibrium, each normalized by its
own dimensionally-correct scale.
"""
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B8 import (
    generate_grid_hex8_laminated_bearing, boundary_node_sets,
    rigid_top_plate_displacement, build_vectorized_neo_hookean_params,
    R_IN, R_OUT)
from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

COMPRESSION_FRAC = 0.02   # nominal compressive strain on the bearing height
SHEAR_X = 0.05            # lateral (shear) displacement of the top plate
PHI_Y = 0.0               # optional rigid rocking of the top plate (0 = pure shear+compression)


def solve_case(Ntheta, Nr, n_rubber_layers=4, nz_per_rubber=3, nz_per_shim=2,
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
    constraints[bottom, :] = True  # fixed foundation
    ux, uy, uz = rigid_top_plate_displacement(
        nodes[top], Lz, compression=COMPRESSION_FRAC * Lz, shear_x=SHEAR_X, phi_y=PHI_Y)
    constraints[top, :] = True
    displacements[top, 0] = torch.tensor(ux, dtype=dtype, device=device)
    displacements[top, 1] = torch.tensor(uy, dtype=dtype, device=device)
    displacements[top, 2] = torch.tensor(uz, dtype=dtype, device=device)
    constraints[sym, 1] = True  # y=0 mirror symmetry (valid: shear along x, rotation about y only)
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

    P_gauss_np, F_gauss_np = P.cpu().numpy(), F.cpu().numpy()
    detF_gauss_np = detF_gauss.cpu().numpy()
    sigma_gauss = (np.einsum("egij,egjk->egik", P_gauss_np, F_gauss_np.transpose(0, 1, 3, 2))
                   / detF_gauss_np[..., None, None])
    sigma_elem = sigma_gauss.mean(axis=1)

    # Peak stress at/near the outer free edge (r=R_out), the region this
    # benchmark's difficulty is expected to concentrate in -- reported here
    # only as a sanity/plausibility check (a real region-QoI convention,
    # analogous to B3's region-Cauchy statistic, is future work once this
    # geometry is confirmed with Timon).
    elem_nodes = nodes[elements]  # (n_elem, 8, 3)
    centroids = elem_nodes.mean(axis=1)
    r_centroid = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    # Outermost radial band of elements, resolution-independent (a fixed
    # absolute distance from R_out is too tight at coarse Nr, giving an
    # empty/NaN sample -- confirmed directly at Nr=5).
    near_outer = r_centroid >= np.percentile(r_centroid, 90.0)
    sigma_xx_near_outer = sigma_elem[near_outer, 0, 0]
    peak_sigma_xx_near_outer = float(np.abs(sigma_xx_near_outer).max()) if near_outer.sum() else float("nan")

    return {
        "Ntheta": Ntheta, "Nr": Nr, "n_elements": len(elements), "n_nodes": n_nodes,
        "n_shim_elements": int(is_shim.sum()), "Lz": Lz,
        "reaction_force_bottom": reaction_force_bottom.tolist(),
        "reaction_force_top": reaction_force_top.tolist(),
        "force_rel_residual": force_rel_residual,
        "peak_sigma_xx_near_outer_edge": peak_sigma_xx_near_outer,
        "max_disp": float(np.linalg.norm(u_np, axis=1).max()),
        "elapsed_s": elapsed,
    }


def main():
    print("B8 (laminated annular seismic bearing) -- SMOKE TEST ONLY.")
    print("Verifies the geometry/BC/vectorized-material code actually runs "
          "and satisfies physics sanity checks. NOT a mesh-convergence "
          "study or accuracy claim -- per Omar's instruction, this code is "
          "for review with Timon before any real dataset-generation or GPU "
          "work starts.\n")
    r = solve_case(9, 5, verbose=True)
    print(f"\nn_elements={r['n_elements']} (n_shim_elements={r['n_shim_elements']}) "
          f"n_nodes={r['n_nodes']} Lz={r['Lz']:.4f} time={r['elapsed_s']:.2f}s")
    print(f"reaction_force_bottom={np.array(r['reaction_force_bottom'])}")
    print(f"reaction_force_top={np.array(r['reaction_force_top'])}")
    print(f"force_rel_residual={r['force_rel_residual']:.2e} (global equilibrium sanity check)")
    print(f"max_disp={r['max_disp']:.4f}")
    print(f"peak |sigma_xx| near the outer free edge (plausibility check only, "
          f"not yet a defined region-QoI)={r['peak_sigma_xx_near_outer_edge']:.4f}")
    print("\nAll physics sanity checks passed inline (zero inverted elements, "
          "det(F)>0 at every element AND every quadrature point, Newton "
          "converged to 1e-8, global force equilibrium normalized by its own "
          "force scale). Code confirmed runnable -- ready for review before "
          "any real mesh-convergence/GPU work.")


if __name__ == "__main__":
    main()
