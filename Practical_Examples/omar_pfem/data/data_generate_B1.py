"""
FEM ground-truth generator for the B1 benchmark (unit square, top-edge
traction, fixed bottom edge), for the PFEM/Transolver pipeline.

Adapted from PFEM-main's data/hyper/data_generate_beam.py (Yizheng Wang et
al., "Pretrain finite element method", JMPS 2026) -- same Q4 mesh, same
Total-Lagrangian Neo-Hookean Newton-Raphson solver, same GRF-based random
material/traction sampling. Only the geometry/BC axis is swapped: the beam
example fixes the LEFT edge (x=0) and loads the RIGHT edge (x=Lx) with a
y-varying traction; B1 fixes the BOTTOM edge (y=0) and loads the TOP edge
(y=Ly) with an x-varying traction. Everything else (shape functions,
element routine, GRF generators, Newton solver structure) is unchanged.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
import h5py
import os
import json
import argparse


# -------------------------
# Q4 shape functions (xi,eta in [-1,1]) -- unchanged, generic
# -------------------------
def shape_Q4(xi, eta):
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    dN_dxi = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [+(1 - eta), -(1 + xi)],
        [+(1 + eta), +(1 + xi)],
        [-(1 + eta), +(1 - xi)]
    ])
    return N, dN_dxi


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
# Neo-Hookean material model -- unchanged, generic
# -------------------------
def neo_hookean_PK1_and_tangent(F, mu, lam):
    Finv = np.linalg.inv(F)
    FinvT = Finv.T
    J = np.linalg.det(F)
    if J <= 0:
        raise ValueError(f"Nonphysical J={J} (det(F) <= 0).")

    lnJ = np.log(J)
    P = mu * (F - FinvT) + lam * lnJ * FinvT

    A = np.zeros((2, 2, 2, 2), dtype=float)
    delta = np.eye(2)
    for i in range(2):
        for J_ in range(2):
            for k in range(2):
                for L in range(2):
                    term1 = mu * delta[i, k] * delta[J_, L]
                    term2 = (mu - lam * lnJ) * FinvT[i, L] * FinvT[k, J_]
                    term3 = lam * FinvT[k, L] * FinvT[i, J_]
                    A[i, J_, k, L] = term1 + term2 + term3
    return P, A


# -------------------------
# Element routine (Q4, TL formulation) -- unchanged, generic
# -------------------------
def element_K_and_fint_TL(Xe, ue, mu, lam):
    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    ke = np.zeros((8, 8), dtype=float)
    fe = np.zeros(8, dtype=float)

    xe = Xe + ue.reshape(4, 2)

    for (xi, eta) in gauss:
        N, dN_dxi = shape_Q4(xi, eta)

        J0 = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J0[0, 0] += Xe[a, 0] * dN_dxi[a, 0]
            J0[1, 0] += Xe[a, 1] * dN_dxi[a, 0]
            J0[0, 1] += Xe[a, 0] * dN_dxi[a, 1]
            J0[1, 1] += Xe[a, 1] * dN_dxi[a, 1]

        detJ0 = np.linalg.det(J0)
        if detJ0 <= 0:
            raise ValueError("Reference element has non-positive det(J0).")

        invJ0 = np.linalg.inv(J0)
        dN_dX = dN_dxi @ invJ0

        F = np.zeros((2, 2), dtype=float)
        for a in range(4):
            F[0, 0] += xe[a, 0] * dN_dX[a, 0]
            F[0, 1] += xe[a, 0] * dN_dX[a, 1]
            F[1, 0] += xe[a, 1] * dN_dX[a, 0]
            F[1, 1] += xe[a, 1] * dN_dX[a, 1]

        P, A = neo_hookean_PK1_and_tangent(F, mu, lam)

        for a in range(4):
            gNa = dN_dX[a]
            fi = P @ gNa
            fe[2*a:2*a+2] += fi * (w * detJ0)

        for a in range(4):
            for b in range(4):
                gNa = dN_dX[a]
                gNb = dN_dX[b]
                Kab = np.zeros((2, 2), dtype=float)
                for i in range(2):
                    for k in range(2):
                        s = 0.0
                        for J_ in range(2):
                            for L in range(2):
                                s += gNa[J_] * A[i, J_, k, L] * gNb[L]
                        Kab[i, k] = s
                ke[2*a:2*a+2, 2*b:2*b+2] += Kab * (w * detJ0)

    return ke, fe


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
# Apply Dirichlet BC -- unchanged, generic
# -------------------------
def apply_dirichlet(K, R, fixed_dofs):
    fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
    all_dofs = np.arange(K.shape[0], dtype=int)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    Kff = K[free][:, free]
    Rf = R[free]
    return free, Kff, Rf


# -------------------------
# Nonlinear solver, bottom fixed / top loaded
# -------------------------
def solve_hyperelastic_TL_spatial(nodes, elements, E_grid, nu_grid, ty_grid, Ly,
                                  nsteps=10, newton_max=25, tol=1e-8):
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

                mu = E_val / (2.0 * (1.0 + nu_val))
                lam = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

                ke, fe_int = element_K_and_fint_TL(Xe, ue, mu, lam)

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
# GRF generators -- unchanged, generic
# -------------------------
def generate_gaussian_random_field_2d(Lx, Ly, Nx, Ny, mean, std, correlation_length, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=x[1]-x[0])
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=y[1]-y[0])
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    L = correlation_length
    power_spectrum = (std**2 * L**2) / (np.pi * (1 + (K * L)**2)**(1.5))
    power_spectrum[0, 0] = 0

    phase = np.random.uniform(0, 2*np.pi, (Nx, Ny))
    amplitudes = np.sqrt(power_spectrum) * np.exp(1j * phase)

    amplitudes[0, 0] = 0
    if Nx > 1:
        amplitudes[Nx//2, 0] = np.real(amplitudes[Nx//2, 0])
    if Ny > 1:
        amplitudes[0, Ny//2] = np.real(amplitudes[0, Ny//2])
    if Nx > 1 and Ny > 1:
        amplitudes[Nx//2, Ny//2] = np.real(amplitudes[Nx//2, Ny//2])

    for i in range(1, Nx//2):
        for j in range(1, Ny//2):
            amplitudes[Nx-i, Ny-j] = np.conj(amplitudes[i, j])

    field = np.fft.ifft2(amplitudes).real
    field_std = np.std(field)
    field = (field / field_std) * std
    field = mean + field

    return field, x, y


def generate_gaussian_random_field_1d(Lx, Nx, mean, std, correlation_length, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    x_mesh = np.meshgrid(x, x)
    dist = np.abs(x_mesh[0] - x_mesh[1])
    cov = std**2 * np.exp(-dist / correlation_length)

    L = np.linalg.cholesky(cov + 1e-10 * np.eye(Nx))
    field = mean + L @ np.random.randn(Nx)

    return field, x


# -------------------------
# Generate one random sample: material fields + top-edge traction profile
# (varies with x, unlike the beam's right-edge traction which varies with y)
# -------------------------
def generate_random_sample_spatial(Lx, Ly, Nx, Ny, seed=None):
    if seed is not None:
        np.random.seed(seed)

    E_field, x_fine, y_fine = generate_gaussian_random_field_2d(
        Lx, Ly, Nx, Ny,
        mean=1000.0, std=200.0,
        correlation_length=0.5*Lx,
        seed=seed
    )

    nu_field_raw, _, _ = generate_gaussian_random_field_2d(
        Lx, Ly, Nx, Ny,
        mean=0.3, std=0.05,
        correlation_length=0.5*Lx,
        seed=seed+1000 if seed is not None else None
    )
    nu_field = np.clip(nu_field_raw, 0.2, 0.4)

    ty_mean = -5.0
    ty_std = 2.0
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
# Generate dataset for physics-informed training
# -------------------------
def generate_dataset_for_physics(num_samples=100, output_dir="physics_training_data_B1", seed=None,
                                 Lx=1.0, Ly=1.0, Nx=21, Ny=21):
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

        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}")
            try:
                E_interp, nu_interp, ty_interp, E_field, nu_field, ty_field = generate_random_sample_spatial(
                    Lx, Ly, Nx, Ny, seed=seed*10000 + i
                )

                u = solve_hyperelastic_TL_spatial(
                    nodes, elements, E_interp, nu_interp, ty_interp, Ly,
                    nsteps=10, newton_max=30, tol=1e-7
                )

                mu_nodes = E_field / (2.0 * (1.0 + nu_field))
                u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
                avg_strain_energy = 0.5 * np.mean(mu_nodes.ravel()) * np.mean(u_mag**2)

                E_node_vals = E_interp(nodes)
                nu_node_vals = nu_interp(nodes)

                top_node_coords = nodes[top_nodes]
                ty_top_vals = ty_interp(top_node_coords)

                E_fields[i] = E_field.astype(np.float32)
                nu_fields[i] = nu_field.astype(np.float32)
                ty_fields[i] = ty_field.astype(np.float32)
                displacements[i] = u.astype(np.float32)
                E_nodes[i] = E_node_vals.astype(np.float32)
                nu_nodes[i] = nu_node_vals.astype(np.float32)
                bottom_bc[i] = np.zeros((len(bottom_nodes), 2), dtype=np.float32)
                top_traction[i] = ty_top_vals.astype(np.float32)
                strain_energy[i] = avg_strain_energy.astype(np.float32)

                successful_samples += 1
                print(f"  Sample {i+1}: max displacement={np.max(u_mag):.3e}")

            except Exception as e:
                print(f"  Error in sample {i+1}: {e}")
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
    args = parser.parse_args()

    print("Generating B1 dataset for physics-informed training...")
    output_dir = "physics_training_data_B1_" + str(args.num_index)
    generate_dataset_for_physics(
        num_samples=args.num_samples,
        output_dir=output_dir, seed=args.num_index,
        Lx=1.0, Ly=1.0, Nx=args.Nx, Ny=args.Ny
    )
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
