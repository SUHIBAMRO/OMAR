"""
FEM ground-truth generator for the B1 benchmark (unit square, top-edge
traction, fixed bottom edge), for the PFEM/Transolver pipeline.

Adapted from PFEM-main's data/hyper/data_generate_beam.py (Yizheng Wang et
al., "Pretrain finite element method", JMPS 2026) -- same Q4 mesh, same
Total-Lagrangian Newton-Raphson solver structure, same GRF-based random
material/traction sampling. Two changes from the original beam example:
  - Geometry/BC axis swapped: the beam fixes the LEFT edge (x=0) and loads
    the RIGHT edge (x=Lx) with a y-varying traction; B1 fixes the BOTTOM
    edge (y=0) and loads the TOP edge (y=Ly) with an x-varying traction.
  - Material selectable via --material (neo_hookean / mooney_rivlin /
    arruda_boyce) through omar_pfem.data.materials's shared PK1/tangent
    registry, instead of being hardcoded to Neo-Hookean.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
import h5py
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from omar_pfem.data.materials import get_material_fns
from omar_pfem.data.fem_core import shape_Q4, element_K_and_fint_TL, apply_dirichlet
from omar_pfem.data.grf import generate_gaussian_random_field_2d, generate_gaussian_random_field_1d
import argparse


# -------------------------
# Grid generator: structured Q4 -- unchanged, generic
# (node layout: n1=bottom-left, n2=bottom-right, n3=top-right, n4=top-left)
# -------------------------
def generate_grid_Q4(Lx, Ly, Nx, Ny):
    xs = np.linspace(0.0, Lx, Nx)
    ys = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.vstack([X.ravel(), Y.ravel()]).T

    elements = []
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            n1 = j * Nx + i
            n2 = n1 + 1
            n3 = (j + 1) * Nx + i + 1
            n4 = (j + 1) * Nx + i
            elements.append([n1, n2, n3, n4])
    return nodes, np.array(elements, dtype=int)


# -------------------------
# External traction on the TOP boundary (y=Ly), spatially varying in x
# (B1's counterpart of the beam's assemble_traction_right_spatial, which
# loads the right edge x=Lx with a traction varying in y; here the loaded
# edge is the top edge y=Ly, local nodes (n3, n4) i.e. eta=+1, and the
# traction varies with the physical x-coordinate along that edge)
# -------------------------
def assemble_traction_top_spatial(nodes, elements, Ly, ty_interp):
    n_nodes = nodes.shape[0]
    Fext = np.zeros(2 * n_nodes, dtype=float)

    g = 1.0 / np.sqrt(3.0)
    gps = [-g, g]
    ws = [1.0, 1.0]

    tol = 1e-12
    for e in elements:
        Xe = nodes[e]
        # local top edge nodes are (n3, n4) -> indices 2 and 3
        y3, y4 = Xe[2, 1], Xe[3, 1]
        if abs(y3 - Ly) < tol and abs(y4 - Ly) < tol:
            X3, X4 = Xe[2], Xe[3]
            edge_len = np.linalg.norm(X4 - X3)

            # On top edge of Q4: eta=+1, xi in [-1,1]
            for (xi, w1d) in zip(gps, ws):
                eta = 1.0
                N, _ = shape_Q4(xi, eta)

                x_pos = 0.0
                for a in range(4):
                    x_pos += N[a] * Xe[a, 0]

                ty_value = ty_interp(np.array([[x_pos, Ly]]))[0]
                t = np.array([0.0, ty_value], dtype=float)

                Jedge = edge_len / 2.0
                for a in range(4):
                    fa = N[a] * t * (w1d * Jedge)
                    node_idx = e[a]
                    Fext[2*node_idx:2*node_idx+2] += fa
    return Fext


# -------------------------
# Nonlinear solver, bottom fixed / top loaded
# -------------------------
def solve_hyperelastic_TL_spatial(nodes, elements, E_grid, nu_grid, ty_grid, Ly,
                                  nsteps=10, newton_max=25, tol=1e-8,
                                  material="neo_hookean"):
    PK1_and_tangent_fn, E_nu_to_params_fn = get_material_fns(material)

    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes
    u = np.zeros(ndof, dtype=float)

    # Dirichlet: bottom edge fixed (u=v=0)
    tolx = 1e-12
    bottom_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    fixed_dofs = []
    for n in bottom_nodes:
        fixed_dofs += [2*n, 2*n+1]

    Fext_full = assemble_traction_top_spatial(nodes, elements, Ly, ty_grid)

    for step in range(1, nsteps + 1):
        alpha = step / nsteps
        Fext = alpha * Fext_full

        for it in range(1, newton_max + 1):
            rows, cols, vals = [], [], []
            fint = np.zeros(ndof, dtype=float)

            for e in elements:
                Xe = nodes[e]
                ue = u.reshape(n_nodes, 2)[e].reshape(-1)

                elem_center = np.mean(Xe, axis=0)
                E_val = E_grid(elem_center[None, :])[0]
                nu_val = nu_grid(elem_center[None, :])[0]

                mat_params = E_nu_to_params_fn(E_val, nu_val)

                ke, fe_int = element_K_and_fint_TL(Xe, ue, mat_params, PK1_and_tangent_fn)

                edofs = []
                for a in e:
                    edofs += [2*a, 2*a+1]
                for i_local, I in enumerate(edofs):
                    fint[I] += fe_int[i_local]
                    for j_local, J in enumerate(edofs):
                        rows.append(I)
                        cols.append(J)
                        vals.append(ke[i_local, j_local])

            K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
            R = fint - Fext

            free, Kff, Rf = apply_dirichlet(K, R, fixed_dofs)
            res_norm = np.linalg.norm(Rf)
            if res_norm < tol:
                print(f"[step {step}/{nsteps}] converged in {it-1} iters, ||R||={res_norm:.3e}")
                break

            du_free = spsolve(Kff, -Rf)
            u[free] += du_free

            if it == newton_max:
                print(f"[step {step}/{nsteps}] NOT converged, last ||R||={res_norm:.3e}")

    return u.reshape(n_nodes, 2)


# -------------------------
# Generate one random sample: material fields + top-edge traction profile
# (varies with x, unlike the beam's right-edge traction which varies with y)
# -------------------------
def generate_random_sample_spatial(
    Lx, Ly, Nx, Ny, seed=None,
    E_mean=1000.0, E_std=200.0,
    nu_mean=0.3, nu_std=0.05, nu_clip=(0.2, 0.4),
    ty_mean=-5.0, ty_std=2.0,
):
    """The (E_mean, E_std, nu_mean, nu_std, nu_clip, ty_mean, ty_std)
    defaults are the in-distribution values used for every training/test
    sample throughout this study. An out-of-distribution (OOD) dataset --
    per the advisor's request to test generalization beyond the training
    distribution -- is generated by calling this same function with
    different values for these arguments (see --E_mean/--E_std/etc. in
    main() below), not by writing separate sampling code, so the OOD and
    in-distribution samples are guaranteed identical in every respect
    (mesh, solver, correlation length, spatial structure) except the
    distribution actually being asked about."""
    if seed is not None:
        np.random.seed(seed)

    E_field, x_fine, y_fine = generate_gaussian_random_field_2d(
        Lx, Ly, Nx, Ny,
        mean=E_mean, std=E_std,
        correlation_length=0.5*Lx,
        seed=seed
    )

    nu_field_raw, _, _ = generate_gaussian_random_field_2d(
        Lx, Ly, Nx, Ny,
        mean=nu_mean, std=nu_std,
        correlation_length=0.5*Lx,
        seed=seed+1000 if seed is not None else None
    )
    nu_field = np.clip(nu_field_raw, nu_clip[0], nu_clip[1])

    ty_1d, x_ty = generate_gaussian_random_field_1d(
        Lx, Nx,
        mean=ty_mean, std=ty_std,
        correlation_length=0.5*Lx,
        seed=seed+2000 if seed is not None else None
    )

    # Only the top boundary (y = Ly) carries a nonzero traction, varying with x
    ty_field_2d = np.zeros((Nx, Ny), dtype=np.float64)
    ty_field_2d[:, -1] = ty_1d  # topmost y index

    E_interp = RegularGridInterpolator((x_fine, y_fine), E_field,
                                       method='linear', bounds_error=False, fill_value=None)
    nu_interp = RegularGridInterpolator((x_fine, y_fine), nu_field,
                                        method='linear', bounds_error=False, fill_value=None)
    ty_interp = RegularGridInterpolator((x_fine, y_fine), ty_field_2d,
                                       method='linear', bounds_error=False, fill_value=None)

    return E_interp, nu_interp, ty_interp, E_field, nu_field, ty_field_2d


# -------------------------
# Per-sample worker (module-level so it's picklable for multiprocessing --
# each sample is fully independent given its own seed, so sample generation
# parallelizes trivially across CPU cores)
# -------------------------
def _generate_one_sample(i, seed, nodes, elements, Lx, Ly, Nx, Ny, material, top_nodes,
                          dist_kwargs=None):
    try:
        E_interp, nu_interp, ty_interp, E_field, nu_field, ty_field = generate_random_sample_spatial(
            Lx, Ly, Nx, Ny, seed=seed, **(dist_kwargs or {})
        )

        u = solve_hyperelastic_TL_spatial(
            nodes, elements, E_interp, nu_interp, ty_interp, Ly,
            nsteps=10, newton_max=30, tol=1e-7, material=material
        )

        mu_nodes = E_field / (2.0 * (1.0 + nu_field))
        u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        avg_strain_energy = 0.5 * np.mean(mu_nodes.ravel()) * np.mean(u_mag**2)

        E_node_vals = E_interp(nodes)
        nu_node_vals = nu_interp(nodes)

        top_node_coords = nodes[top_nodes]
        ty_top_vals = ty_interp(top_node_coords)

        return i, True, dict(
            E_field=E_field.astype(np.float32),
            nu_field=nu_field.astype(np.float32),
            ty_field=ty_field.astype(np.float32),
            displacement=u.astype(np.float32),
            E_node_vals=E_node_vals.astype(np.float32),
            nu_node_vals=nu_node_vals.astype(np.float32),
            ty_top_vals=ty_top_vals.astype(np.float32),
            avg_strain_energy=np.float32(avg_strain_energy),
            max_disp=float(np.max(u_mag)),
        ), None
    except Exception as e:
        return i, False, None, str(e)


# -------------------------
# Generate dataset for physics-informed training
# -------------------------
def generate_dataset_for_physics(num_samples=100, output_dir="physics_training_data_B1", seed=None,
                                 Lx=1.0, Ly=1.0, Nx=21, Ny=21, material="neo_hookean", n_workers=1,
                                 dist_kwargs=None):
    os.makedirs(output_dir, exist_ok=True)

    nodes, elements = generate_grid_Q4(Lx, Ly, Nx, Ny)
    n_nodes = nodes.shape[0]

    mesh_info = {
        'nodes': nodes, 'elements': elements,
        'Lx': Lx, 'Ly': Ly, 'Nx': Nx, 'Ny': Ny
    }
    np.savez(os.path.join(output_dir, 'mesh_info.npz'), **mesh_info)

    tol = 1e-12
    left_nodes = np.where(nodes[:, 0] < tol)[0]
    right_nodes = np.where(np.abs(nodes[:, 0] - Lx) < tol)[0]
    bottom_nodes = np.where(nodes[:, 1] < tol)[0]
    top_nodes = np.where(np.abs(nodes[:, 1] - Ly) < tol)[0]

    boundary_info = {
        'left_nodes': left_nodes, 'right_nodes': right_nodes,
        'bottom_nodes': bottom_nodes, 'top_nodes': top_nodes
    }
    np.savez(os.path.join(output_dir, 'boundary_info.npz'), **boundary_info)

    with h5py.File(os.path.join(output_dir, 'hyperelastic_dataset_physics.h5'), 'w') as f:
        num_coarse_x, num_coarse_y = Nx, Ny

        E_fields = f.create_dataset("E_fields", (num_samples, num_coarse_x, num_coarse_y),
                                    dtype=np.float32, compression="gzip")
        nu_fields = f.create_dataset("nu_fields", (num_samples, num_coarse_x, num_coarse_y),
                                     dtype=np.float32, compression="gzip")
        ty_fields = f.create_dataset("ty_fields", (num_samples, num_coarse_x, num_coarse_y),
                                     dtype=np.float32, compression="gzip")
        displacements = f.create_dataset("displacements", (num_samples, n_nodes, 2),
                                         dtype=np.float32, compression="gzip")
        E_nodes = f.create_dataset("E_nodes", (num_samples, n_nodes),
                                   dtype=np.float32, compression="gzip")
        nu_nodes = f.create_dataset("nu_nodes", (num_samples, n_nodes),
                                    dtype=np.float32, compression="gzip")
        bottom_bc = f.create_dataset("bottom_bc", (num_samples, len(bottom_nodes), 2),
                                     dtype=np.float32, compression="gzip")
        top_traction = f.create_dataset("top_traction", (num_samples, len(top_nodes)),
                                        dtype=np.float32, compression="gzip")
        strain_energy = f.create_dataset("strain_energy", (num_samples,),
                                         dtype=np.float32, compression="gzip")

        successful_samples = 0
        failed_samples = []

        def _store_success(i, res):
            E_fields[i] = res["E_field"]
            nu_fields[i] = res["nu_field"]
            ty_fields[i] = res["ty_field"]
            displacements[i] = res["displacement"]
            E_nodes[i] = res["E_node_vals"]
            nu_nodes[i] = res["nu_node_vals"]
            bottom_bc[i] = np.zeros((len(bottom_nodes), 2), dtype=np.float32)
            top_traction[i] = res["ty_top_vals"]
            strain_energy[i] = res["avg_strain_energy"]
            print(f"  Sample {i+1}: max displacement={res['max_disp']:.3e}")

        def _store_failure(i, err):
            print(f"  Error in sample {i+1}: {err}")
            failed_samples.append(i)
            E_fields[i] = np.nan
            nu_fields[i] = np.nan
            ty_fields[i] = np.nan
            displacements[i] = np.nan
            E_nodes[i] = np.nan
            nu_nodes[i] = np.nan
            bottom_bc[i] = np.nan
            top_traction[i] = np.nan
            strain_energy[i] = np.nan

        if n_workers is not None and n_workers > 1:
            print(f"Generating {num_samples} samples using {n_workers} worker processes...")
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_generate_one_sample, i, seed*10000 + i, nodes, elements,
                              Lx, Ly, Nx, Ny, material, top_nodes, dist_kwargs): i
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
                    i, seed*10000 + i, nodes, elements, Lx, Ly, Nx, Ny, material, top_nodes, dist_kwargs
                )
                if ok:
                    _store_success(i, res)
                    successful_samples += 1
                else:
                    _store_failure(i, err)

        f.attrs['num_samples'] = num_samples
        f.attrs['successful_samples'] = successful_samples
        f.attrs['failed_samples'] = str(failed_samples)
        f.attrs['Lx'] = Lx
        f.attrs['Ly'] = Ly
        f.attrs['Nx'] = Nx
        f.attrs['Ny'] = Ny
        f.attrs['num_coarse_x'] = num_coarse_x
        f.attrs['num_coarse_y'] = num_coarse_y
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
        "PFEM/Transolver Q4 FEM ground-truth generator for B1 (unit square, top-edge traction, fixed bottom)"
    )
    parser.add_argument("--num_index", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--Nx", type=int, default=21)
    parser.add_argument("--Ny", type=int, default=21)
    parser.add_argument("--material", type=str, default="neo_hookean",
                        choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override the default physics_training_data_B1_<material>_<num_index> dir name")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel worker processes for sample generation (each sample is independent)")
    # Out-of-distribution (OOD) overrides -- default to the in-distribution
    # values used everywhere else in this study (see generate_random_sample_spatial).
    # Per the advisor's request to test generalization beyond the training
    # distribution, an OOD test set is generated by overriding one or more
    # of these to a range the network never saw during training, e.g.
    # --E_mean 2000 (stiffer material than any training sample) or
    # --ty_mean -10 (double the in-distribution load magnitude).
    parser.add_argument("--E_mean", type=float, default=1000.0)
    parser.add_argument("--E_std", type=float, default=200.0)
    parser.add_argument("--nu_mean", type=float, default=0.3)
    parser.add_argument("--nu_std", type=float, default=0.05)
    parser.add_argument("--nu_clip_min", type=float, default=0.2)
    parser.add_argument("--nu_clip_max", type=float, default=0.4)
    parser.add_argument("--ty_mean", type=float, default=-5.0)
    parser.add_argument("--ty_std", type=float, default=2.0)
    args = parser.parse_args()

    dist_kwargs = dict(
        E_mean=args.E_mean, E_std=args.E_std,
        nu_mean=args.nu_mean, nu_std=args.nu_std,
        nu_clip=(args.nu_clip_min, args.nu_clip_max),
        ty_mean=args.ty_mean, ty_std=args.ty_std,
    )
    is_ood = dist_kwargs != dict(
        E_mean=1000.0, E_std=200.0, nu_mean=0.3, nu_std=0.05,
        nu_clip=(0.2, 0.4), ty_mean=-5.0, ty_std=2.0,
    )

    print(f"Generating B1 dataset ({args.material}) for physics-informed training"
          f"{' [OUT-OF-DISTRIBUTION: ' + str(dist_kwargs) + ']' if is_ood else ''}...")
    output_dir = args.out_dir or f"physics_training_data_B1_{args.material}_{args.num_index}"
    generate_dataset_for_physics(
        num_samples=args.num_samples,
        output_dir=output_dir, seed=args.num_index,
        Lx=1.0, Ly=1.0, Nx=args.Nx, Ny=args.Ny, material=args.material,
        n_workers=args.n_workers, dist_kwargs=dist_kwargs,
    )
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
