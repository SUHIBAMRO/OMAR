"""
FEM ground-truth generator for the B2 benchmark (quarter ring, R_in=1,
R_out=2, internal pressure on the inner arc, symmetry BCs on the two
straight edges), for the PFEM/Transolver pipeline.

Unlike B1 (a straight axis-swap of PFEM-main's beam example), B2's curved
geometry and boundary conditions have no PFEM-main precedent and are
implemented from scratch here, reusing only the geometry-agnostic pieces
(Q4 shape functions, element internal-force/tangent routine, Dirichlet
elimination -- omar_pfem/data/fem_core.py; GRF samplers --
omar_pfem/data/grf.py; material PK1/tangent registry --
omar_pfem/data/materials.py).

Geometry: nodes are generated on a structured (theta, r) grid with
theta in [0, pi/2] (fast index i, playing the role B1's x played) and r in
[R_in, R_out] (slow index j, playing the role B1's y played), mapped to
Cartesian via (x, y) = (r*cos(theta), r*sin(theta)). Element connectivity
is the same index formula as a flat Q4 grid -- the isoparametric element
routine only ever sees each element's 4 corner reference coordinates, so a
bilinear Q4 built from points that happen to lie on circular arcs is just
as valid an element as one built from a flat grid (each element's own edges
are still straight chords between its corner nodes).

Boundary conditions:
  - Inner arc (r=R_in, j=0): internal pressure, analogous to B1's top-edge
    traction but with a spatially-varying LOCAL OUTWARD NORMAL (computed per
    boundary edge from that edge's own chord, not the exact circular normal
    -- the standard, force-consistent way to load a curved boundary
    discretized by straight-edged elements) instead of a fixed (0,1)
    direction.
  - theta=0 edge (i=0, the segment lying on y=0): symmetry BC u_y=0.
  - theta=pi/2 edge (i=Ntheta-1, the segment lying on x=0): symmetry BC
    u_x=0.
  - Outer arc (r=R_out): traction-free (no explicit BC needed).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
import h5py
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from omar_pfem.data.materials import get_material_fns
from omar_pfem.data.fem_core import shape_Q4, element_K_and_fint_TL, apply_dirichlet
from omar_pfem.data.grf import generate_gaussian_random_field_2d, generate_gaussian_random_field_1d


# -------------------------
# Grid generator: structured Q4 on a quarter-ring.
#
# Node/element index convention: r is the FAST index (i), theta is the SLOW
# index (j) -- i.e. n1->n2 steps in r, n1->n4 steps in theta. This (not the
# more "natural"-looking theta-fast/r-slow layout) is what's required for a
# positive reference Jacobian: the local orthonormal frame (r_hat, theta_hat)
# is right-handed/CCW (like (x, y) in a flat grid), so stepping through r
# first and theta second reproduces that same CCW winding for each element's
# corner nodes. (theta-fast/r-slow gives the reversed, CW pair (theta_hat,
# r_hat) and a negative det(J0) at every element.)
# -------------------------
def generate_grid_Q4_ring(R_in, R_out, Ntheta, Nr, theta_max=np.pi / 2):
    thetas = np.linspace(0.0, theta_max, Ntheta)
    rs = np.linspace(R_in, R_out, Nr)
    R, TH = np.meshgrid(rs, thetas, indexing="xy")
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T

    elements = []
    for j in range(Ntheta - 1):
        for i in range(Nr - 1):
            n1 = j * Nr + i
            n2 = n1 + 1
            n3 = (j + 1) * Nr + i + 1
            n4 = (j + 1) * Nr + i
            elements.append([n1, n2, n3, n4])
    return nodes, np.array(elements, dtype=int)


# -------------------------
# Internal pressure on the INNER arc (r=R_in). Since r is the FAST local
# index (see generate_grid_Q4_ring), the inner arc is the xi=-1 edge of the
# Q4 reference element (local nodes n1, n4), not eta=-1: n1->n2 steps in r
# (a radial segment), while n4->n1 (xi=-1, eta from +1 to -1) steps in
# theta at fixed (innermost) r -- that's the arc. Traction direction is the
# per-edge chord's own outward normal (not the exact circular normal),
# consistent with how a bilinear Q4 mesh discretizes a curved boundary as a
# polygon -- this is the standard force-consistent way to load such a
# boundary and converges to the exact radial traction as the mesh is
# refined in theta.
# -------------------------
def assemble_traction_inner_curved(nodes, elements, R_in, p_interp):
    n_nodes = nodes.shape[0]
    Fext = np.zeros(2 * n_nodes, dtype=float)

    g = 1.0 / np.sqrt(3.0)
    gps = [-g, g]
    ws = [1.0, 1.0]

    tol = 1e-9
    for e in elements:
        Xe = nodes[e]
        r1 = np.linalg.norm(Xe[0])
        r4 = np.linalg.norm(Xe[3])
        if abs(r1 - R_in) < tol and abs(r4 - R_in) < tol:
            X1, X4 = Xe[0], Xe[3]
            edge_vec = X4 - X1
            edge_len = np.linalg.norm(edge_vec)
            tangent_hat = edge_vec / edge_len

            # rotate tangent by -90 deg; orient outward (away from center)
            # by checking against the edge midpoint's own position vector
            normal = np.array([tangent_hat[1], -tangent_hat[0]])
            midpoint = 0.5 * (X1 + X4)
            if np.dot(normal, midpoint) < 0:
                normal = -normal

            # On inner edge of Q4: xi=-1, eta in [-1,1]
            for (eta, w1d) in zip(gps, ws):
                xi = -1.0
                N, _ = shape_Q4(xi, eta)

                pos = N @ Xe
                theta_pos = np.arctan2(pos[1], pos[0])

                p_value = p_interp(np.array([[theta_pos, R_in]]))[0]
                t = p_value * normal

                Jedge = edge_len / 2.0
                for a in range(4):
                    fa = N[a] * t * (w1d * Jedge)
                    node_idx = e[a]
                    Fext[2 * node_idx:2 * node_idx + 2] += fa
    return Fext


# -------------------------
# Nonlinear solver: inner-arc pressure loaded, symmetry BCs on the two
# straight edges (u_y=0 at theta=0, u_x=0 at theta=pi/2), outer arc free
# -------------------------
def solve_hyperelastic_TL_ring(nodes, elements, E_grid, nu_grid, p_grid, R_in,
                                nsteps=10, newton_max=25, tol=1e-8,
                                material="neo_hookean", profile=None):
    """profile: optional dict: if given, accumulates wall-clock timing broken
    down into 'assembly' (the per-element loop + sparse matrix formation +
    Dirichlet elimination) and 'solve' (spsolve) buckets, plus iteration
    bookkeeping, with no other change in behavior. See fem_cost_breakdown.py."""
    import time as _time
    PK1_and_tangent_fn, E_nu_to_params_fn = get_material_fns(material)

    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes
    u = np.zeros(ndof, dtype=float)

    tolx = 1e-9
    theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]       # y=0 edge
    thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]  # x=0 edge
    fixed_dofs = []
    for n in theta0_nodes:
        fixed_dofs.append(2 * n + 1)       # u_y = 0
    for n in thetahalfpi_nodes:
        fixed_dofs.append(2 * n)           # u_x = 0

    Fext_full = assemble_traction_inner_curved(nodes, elements, R_in, p_grid)

    def _material_fn(X_gp):
        r_gp = np.linalg.norm(X_gp)
        theta_gp = np.arctan2(X_gp[1], X_gp[0])
        E_val = E_grid(np.array([[theta_gp, r_gp]]))[0]
        nu_val = nu_grid(np.array([[theta_gp, r_gp]]))[0]
        return E_nu_to_params_fn(E_val, nu_val)

    if profile is not None:
        profile.setdefault("t_assembly_s", 0.0)
        profile.setdefault("t_solve_s", 0.0)
        profile.setdefault("n_newton_iters_total", 0)
        profile.setdefault("n_load_steps", nsteps)
        profile.setdefault("newton_tol", tol)
        profile.setdefault("newton_max", newton_max)
        profile.setdefault("dtype", str(u.dtype))
        profile.setdefault("linear_solver", "scipy.sparse.linalg.spsolve (direct, sparse LU)")

    for step in range(1, nsteps + 1):
        alpha = step / nsteps
        Fext = alpha * Fext_full

        for it in range(1, newton_max + 1):
            t0 = _time.perf_counter() if profile is not None else None

            rows, cols, vals = [], [], []
            fint = np.zeros(ndof, dtype=float)

            for e in elements:
                Xe = nodes[e]
                ue = u.reshape(n_nodes, 2)[e].reshape(-1)

                ke, fe_int = element_K_and_fint_TL(
                    Xe, ue, None, PK1_and_tangent_fn, material_fn=_material_fn)

                edofs = []
                for a in e:
                    edofs += [2 * a, 2 * a + 1]
                for i_local, I in enumerate(edofs):
                    fint[I] += fe_int[i_local]
                    for j_local, J in enumerate(edofs):
                        rows.append(I)
                        cols.append(J)
                        vals.append(ke[i_local, j_local])

            K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
            R = fint - Fext

            free, Kff, Rf = apply_dirichlet(K, R, fixed_dofs)

            if profile is not None:
                profile["t_assembly_s"] += _time.perf_counter() - t0
                profile["n_newton_iters_total"] += 1

            res_norm = np.linalg.norm(Rf)
            if res_norm < tol:
                print(f"[step {step}/{nsteps}] converged in {it-1} iters, ||R||={res_norm:.3e}")
                break

            t1 = _time.perf_counter() if profile is not None else None
            du_free = spsolve(Kff, -Rf)
            if profile is not None:
                profile["t_solve_s"] += _time.perf_counter() - t1
            u[free] += du_free

            if it == newton_max:
                print(f"[step {step}/{nsteps}] NOT converged, last ||R||={res_norm:.3e}")

    return u.reshape(n_nodes, 2)


# -------------------------
# Generate one random sample: material fields (over theta,r) + inner-arc
# pressure profile (varies with theta)
# -------------------------
def generate_random_sample_ring(
    R_in, R_out, Ntheta, Nr, theta_max=np.pi / 2, seed=None,
    E_mean=1000.0, E_std=200.0,
    nu_mean=0.3, nu_std=0.05, nu_clip=(0.2, 0.4),
    p_mean=5.0, p_std=2.0,
):
    """See data_generate_B1.py's generate_random_sample_spatial for the
    rationale: the (E_mean, E_std, nu_mean, nu_std, nu_clip, p_mean, p_std)
    defaults are B2's in-distribution values; an OOD dataset is generated
    by calling this same function with different values (see main() below),
    not separate sampling code."""
    if seed is not None:
        np.random.seed(seed)

    dr = R_out - R_in

    E_field, th_fine, r_fine_shifted = generate_gaussian_random_field_2d(
        theta_max, dr, Ntheta, Nr,
        mean=E_mean, std=E_std,
        correlation_length=0.5 * theta_max,
        seed=seed
    )
    r_fine = R_in + r_fine_shifted

    nu_field_raw, _, _ = generate_gaussian_random_field_2d(
        theta_max, dr, Ntheta, Nr,
        mean=nu_mean, std=nu_std,
        correlation_length=0.5 * theta_max,
        seed=seed + 1000 if seed is not None else None
    )
    nu_field = np.clip(nu_field_raw, nu_clip[0], nu_clip[1])

    p_1d, th_p = generate_gaussian_random_field_1d(
        theta_max, Ntheta,
        mean=p_mean, std=p_std,
        correlation_length=0.5 * theta_max,
        seed=seed + 2000 if seed is not None else None
    )

    # Only the inner boundary (r = R_in) carries a nonzero pressure,
    # varying with theta
    p_field_2d = np.zeros((Ntheta, Nr), dtype=np.float64)
    p_field_2d[:, 0] = p_1d  # innermost r index

    E_interp = RegularGridInterpolator((th_fine, r_fine), E_field,
                                       method='linear', bounds_error=False, fill_value=None)
    nu_interp = RegularGridInterpolator((th_fine, r_fine), nu_field,
                                        method='linear', bounds_error=False, fill_value=None)
    p_interp = RegularGridInterpolator((th_fine, r_fine), p_field_2d,
                                       method='linear', bounds_error=False, fill_value=None)

    return E_interp, nu_interp, p_interp, E_field, nu_field, p_field_2d


# -------------------------
# Per-sample worker (module-level so it's picklable for multiprocessing --
# each sample is fully independent given its own seed)
# -------------------------
def _generate_one_sample(i, seed, nodes, elements, R_in, R_out, Ntheta, Nr, material, inner_nodes,
                          dist_kwargs=None):
    try:
        E_interp, nu_interp, p_interp, E_field, nu_field, p_field = generate_random_sample_ring(
            R_in, R_out, Ntheta, Nr, seed=seed, **(dist_kwargs or {})
        )

        u = solve_hyperelastic_TL_ring(
            nodes, elements, E_interp, nu_interp, p_interp, R_in,
            nsteps=10, newton_max=30, tol=1e-7, material=material
        )

        mu_nodes = E_field / (2.0 * (1.0 + nu_field))
        u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        avg_strain_energy = 0.5 * np.mean(mu_nodes.ravel()) * np.mean(u_mag**2)

        node_theta = np.arctan2(nodes[:, 1], nodes[:, 0])
        node_r = np.linalg.norm(nodes, axis=1)
        node_polar = np.stack([node_theta, node_r], axis=1)
        E_node_vals = E_interp(node_polar)
        nu_node_vals = nu_interp(node_polar)

        inner_polar = node_polar[inner_nodes]
        p_inner_vals = p_interp(inner_polar)

        return i, True, dict(
            E_field=E_field.astype(np.float32),
            nu_field=nu_field.astype(np.float32),
            p_field=p_field.astype(np.float32),
            displacement=u.astype(np.float32),
            E_node_vals=E_node_vals.astype(np.float32),
            nu_node_vals=nu_node_vals.astype(np.float32),
            p_inner_vals=p_inner_vals.astype(np.float32),
            avg_strain_energy=np.float32(avg_strain_energy),
            max_disp=float(np.max(u_mag)),
        ), None
    except Exception as e:
        return i, False, None, str(e)


# -------------------------
# Generate dataset for physics-informed training
# -------------------------
def generate_dataset_for_physics(num_samples=100, output_dir="physics_training_data_B2", seed=None,
                                 R_in=1.0, R_out=2.0, Ntheta=21, Nr=21, material="neo_hookean", n_workers=1,
                                 dist_kwargs=None):
    os.makedirs(output_dir, exist_ok=True)

    nodes, elements = generate_grid_Q4_ring(R_in, R_out, Ntheta, Nr)
    n_nodes = nodes.shape[0]

    mesh_info = {
        'nodes': nodes, 'elements': elements,
        'R_in': R_in, 'R_out': R_out, 'Ntheta': Ntheta, 'Nr': Nr
    }
    np.savez(os.path.join(output_dir, 'mesh_info.npz'), **mesh_info)

    tol = 1e-9
    theta0_nodes = np.where(np.abs(nodes[:, 1]) < tol)[0]
    thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tol)[0]
    inner_nodes = np.where(np.abs(np.linalg.norm(nodes, axis=1) - R_in) < tol)[0]
    outer_nodes = np.where(np.abs(np.linalg.norm(nodes, axis=1) - R_out) < tol)[0]

    boundary_info = {
        'theta0_nodes': theta0_nodes, 'thetahalfpi_nodes': thetahalfpi_nodes,
        'inner_nodes': inner_nodes, 'outer_nodes': outer_nodes
    }
    np.savez(os.path.join(output_dir, 'boundary_info.npz'), **boundary_info)

    with h5py.File(os.path.join(output_dir, 'hyperelastic_dataset_physics.h5'), 'w') as f:
        E_fields = f.create_dataset("E_fields", (num_samples, Ntheta, Nr),
                                    dtype=np.float32, compression="gzip")
        nu_fields = f.create_dataset("nu_fields", (num_samples, Ntheta, Nr),
                                     dtype=np.float32, compression="gzip")
        p_fields = f.create_dataset("p_fields", (num_samples, Ntheta, Nr),
                                    dtype=np.float32, compression="gzip")
        displacements = f.create_dataset("displacements", (num_samples, n_nodes, 2),
                                         dtype=np.float32, compression="gzip")
        E_nodes = f.create_dataset("E_nodes", (num_samples, n_nodes),
                                   dtype=np.float32, compression="gzip")
        nu_nodes = f.create_dataset("nu_nodes", (num_samples, n_nodes),
                                    dtype=np.float32, compression="gzip")
        inner_pressure = f.create_dataset("inner_pressure", (num_samples, len(inner_nodes)),
                                          dtype=np.float32, compression="gzip")
        strain_energy = f.create_dataset("strain_energy", (num_samples,),
                                         dtype=np.float32, compression="gzip")

        successful_samples = 0
        failed_samples = []

        def _store_success(i, res):
            E_fields[i] = res["E_field"]
            nu_fields[i] = res["nu_field"]
            p_fields[i] = res["p_field"]
            displacements[i] = res["displacement"]
            E_nodes[i] = res["E_node_vals"]
            nu_nodes[i] = res["nu_node_vals"]
            inner_pressure[i] = res["p_inner_vals"]
            strain_energy[i] = res["avg_strain_energy"]
            print(f"  Sample {i+1}: max displacement={res['max_disp']:.3e}")

        def _store_failure(i, err):
            print(f"  Error in sample {i+1}: {err}")
            failed_samples.append(i)
            E_fields[i] = np.nan
            nu_fields[i] = np.nan
            p_fields[i] = np.nan
            displacements[i] = np.nan
            E_nodes[i] = np.nan
            nu_nodes[i] = np.nan
            inner_pressure[i] = np.nan
            strain_energy[i] = np.nan

        if n_workers is not None and n_workers > 1:
            print(f"Generating {num_samples} samples using {n_workers} worker processes...")
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_generate_one_sample, i, seed*10000 + i, nodes, elements,
                              R_in, R_out, Ntheta, Nr, material, inner_nodes, dist_kwargs): i
                    for i in range(num_samples)
                }
                done_count = 0
                for fut in as_completed(futures):
                    i, ok, res, err = fut.result()
                    done_count += 1
                    if done_count % 10 == 0 or done_count == num_samples:
                        print(f"Completed {done_count}/{num_samples} samples")
                    if ok:
                        _store_success(i, res)
                        successful_samples += 1
                    else:
                        _store_failure(i, err)
        else:
            for i in range(num_samples):
                print(f"Generating sample {i+1}/{num_samples}")
                _, ok, res, err = _generate_one_sample(
                    i, seed*10000 + i, nodes, elements, R_in, R_out, Ntheta, Nr, material, inner_nodes,
                    dist_kwargs
                )
                if ok:
                    _store_success(i, res)
                    successful_samples += 1
                else:
                    _store_failure(i, err)

        f.attrs['num_samples'] = num_samples
        f.attrs['successful_samples'] = successful_samples
        f.attrs['failed_samples'] = str(failed_samples)
        f.attrs['R_in'] = R_in
        f.attrs['R_out'] = R_out
        f.attrs['Ntheta'] = Ntheta
        f.attrs['Nr'] = Nr
        f.attrs['n_nodes'] = n_nodes

    print(f"\nDataset generated with {successful_samples} successful samples out of {num_samples}")

    train_ratio, val_ratio = 0.7, 0.15
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)

    split_indices = {
        'train': indices[:n_train].tolist(),
        'val': indices[n_train:n_train+n_val].tolist(),
        'test': indices[n_train+n_val:].tolist()
    }

    with open(os.path.join(output_dir, 'split_indices.json'), 'w') as f:
        json.dump(split_indices, f, indent=2)

    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={num_samples - n_train - n_val}")

    return nodes, elements, split_indices


def main():
    parser = argparse.ArgumentParser(
        "PFEM/Transolver Q4 FEM ground-truth generator for B2 (quarter ring, internal pressure)"
    )
    parser.add_argument("--num_index", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--R_in", type=float, default=1.0)
    parser.add_argument("--R_out", type=float, default=2.0)
    parser.add_argument("--Ntheta", type=int, default=21)
    parser.add_argument("--Nr", type=int, default=21)
    parser.add_argument("--material", type=str, default="neo_hookean",
                        choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override the default physics_training_data_B2_<material>_<num_index> dir name")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel worker processes for sample generation (each sample is independent)")
    # Out-of-distribution (OOD) overrides -- see data_generate_B1.py's main()
    # for the rationale; defaults are B2's in-distribution values.
    parser.add_argument("--E_mean", type=float, default=1000.0)
    parser.add_argument("--E_std", type=float, default=200.0)
    parser.add_argument("--nu_mean", type=float, default=0.3)
    parser.add_argument("--nu_std", type=float, default=0.05)
    parser.add_argument("--nu_clip_min", type=float, default=0.2)
    parser.add_argument("--nu_clip_max", type=float, default=0.4)
    parser.add_argument("--p_mean", type=float, default=5.0)
    parser.add_argument("--p_std", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = args.out_dir or f"physics_training_data_B2_{args.material}_{args.num_index}"

    dist_kwargs = dict(
        E_mean=args.E_mean, E_std=args.E_std,
        nu_mean=args.nu_mean, nu_std=args.nu_std,
        nu_clip=(args.nu_clip_min, args.nu_clip_max),
        p_mean=args.p_mean, p_std=args.p_std,
    )
    is_ood = dist_kwargs != dict(
        E_mean=1000.0, E_std=200.0, nu_mean=0.3, nu_std=0.05,
        nu_clip=(0.2, 0.4), p_mean=5.0, p_std=2.0,
    )
    print(f"Generating B2 dataset ({args.material}) for physics-informed training"
          f"{' [OUT-OF-DISTRIBUTION: ' + str(dist_kwargs) + ']' if is_ood else ''}...")

    generate_dataset_for_physics(
        num_samples=args.num_samples, output_dir=output_dir, seed=args.num_index,
        R_in=args.R_in, R_out=args.R_out, Ntheta=args.Ntheta, Nr=args.Nr,
        material=args.material, n_workers=args.n_workers, dist_kwargs=dist_kwargs,
    )
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
