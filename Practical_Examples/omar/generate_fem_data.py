"""
FEM ground-truth data generation for the B1 and B2 benchmarks using FEniCSx
(dolfinx). Run with the 'fenicsx' conda environment
(mamba create -n fenicsx -c conda-forge fenics-dolfinx mpich numpy scipy),
NOT the jax/flax environment used for training -- dolfinx and jax are kept
separate on purpose.

Fixes the earlier prototype's traction bug: the full spatially-varying GRF
traction / pressure profile is applied at each boundary node (via a
dolfinx Function interpolated from the sampled profile), not just its mean.
The GRF sampler is the same RBF-kernel Gaussian process used by the
original repo's FGMBeam_solver (utils/pde_solvers.py), reusing the
traction_mean / traction_var / traction_scale config keys.

B1 (unit square): traction on top edge (y=1) varies with x, bottom edge
    (y=0) fixed.
B2 (quarter ring, R_in=1, R_out=2): pressure on inner edge (r=R_in) varies
    with theta, symmetry BCs (u_y=0 at theta=0, u_x=0 at theta=pi/2) on the
    two straight edges. The mesh is built as a unit-square topology whose
    node coordinates are remapped from (r, theta) parameter space to
    physical (x, y) = (r cos theta, r sin theta) -- no gmsh needed.

Output .npz format (matches the original Hyperelasticity_dataset convention):
    B1: traction [N, numPtsU], disp2D [N, numPtsV, numPtsU, 2]
    B2: traction [N, numPtsV], disp2D [N, numPtsV, numPtsU, 2]  (grid is in
        (theta, r) order: rows = theta, columns = r)
"""
import os
import sys
import time
import argparse

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, Function
from dolfinx.fem.petsc import NonlinearProblem
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omar.config import B1_NH, B1_MR, B1_AB, B2_NH, B2_MR, B2_AB


# ======================================================================
#  GRF sampler -- same RBF-kernel Gaussian process as
#  utils/pde_solvers.py's FGMBeam_solver.GRF
# ======================================================================
def rbf_kernel(x1, x2, length_scale):
    diffs = np.expand_dims(x1 / length_scale, 1) - np.expand_dims(x2 / length_scale, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return np.exp(-0.5 * r2)


def grf_sample(n_pts, extent, mean, variance, length_scale, rng):
    jitter = 1e-10
    x = np.linspace(0, extent, n_pts)[:, None]
    k = rbf_kernel(x, x, length_scale)
    chol = np.linalg.cholesky(k + jitter * np.eye(n_pts))
    sample = variance * chol.dot(rng.standard_normal(n_pts)) + mean
    return x.flatten(), sample


# ======================================================================
#  Strain energy densities -- same formulas as omar/losses.py
#  (ArrudaBoyce2D uses the 2D-consistent 2^k reference, not 3^k)
# ======================================================================
def strain_energy(material, u, beam):
    identity = ufl.Identity(2)
    defgrad = identity + ufl.grad(u)
    c_tensor = defgrad.T * defgrad
    ic = ufl.tr(c_tensor)
    jac = ufl.det(defgrad)

    if material == 'neohookean':
        e_mod, nu = beam["E"], beam["nu"]
        mu = e_mod / (2 * (1 + nu))
        lam = e_mod * nu / ((1 + nu) * (1 - 2 * nu))
        return 0.5 * mu * (ic - 2) - mu * ufl.ln(jac) + 0.5 * lam * ufl.ln(jac) ** 2

    if material == 'mooneyrivlin':
        c1, c2, c = beam["param_c1"], beam["param_c2"], beam["param_c"]
        d = 2 * (c1 + 2 * c2)
        iic = 0.5 * (ic ** 2 - ufl.tr(c_tensor * c_tensor))
        return c * (jac - 1) ** 2 - d * ufl.ln(jac) + c1 * (ic - 2) + c2 * (iic - 1)

    if material == 'arrudaboyce':
        mu_ab, n_ab, kappa_ab = beam["param_mu_ab"], beam["param_N_ab"], beam["param_kappa_ab"]
        alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
        psi = 0.0
        for k, a in enumerate(alpha, start=1):
            psi = psi + mu_ab * a / n_ab ** (k - 1) * (ic ** k - 2 ** k)
        return psi + kappa_ab / 2 * (jac - 1) ** 2

    raise ValueError(f"Unknown material: {material}")


def newton_solve(psi, u, v, bcs):
    residual = ufl.derivative(psi, u, v)
    problem = NonlinearProblem(
        residual, u, bcs=bcs, petsc_options_prefix="omar_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    problem.solve()
    snes = problem.solver
    converged = snes.getConvergedReason() > 0
    n_it = snes.getIterationNumber()
    return converged, n_it


# ======================================================================
#  B1 -- unit square, top-edge traction, fixed bottom edge
# ======================================================================
def solve_b1(material, beam, nx, ny, x_pts, t_vals):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx - 1, ny - 1)
    v_space = functionspace(domain, ('Lagrange', 1, (2,)))
    u = Function(v_space)
    v_test = ufl.TestFunction(v_space)

    top_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 1.0))
    facet_tags = mesh.meshtags(domain, 1, top_facets, np.full(len(top_facets), 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    t_scalar = functionspace(domain, ('Lagrange', 1))
    t_func = Function(t_scalar)
    t_func.interpolate(lambda x: np.interp(x[0], x_pts, t_vals))
    t_vec = ufl.as_vector([0.0, t_func])

    bottom_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0))
    bc_dofs = fem.locate_dofs_topological(v_space, 1, bottom_facets)
    bc = fem.dirichletbc(np.zeros(2), bc_dofs, v_space)

    psi = strain_energy(material, u, beam) * ufl.dx - ufl.dot(t_vec, u) * ds(1)
    converged, n_it = newton_solve(psi, u, v_test, [bc])

    bs = v_space.dofmap.index_map_bs
    n_local = v_space.dofmap.index_map.size_local
    u_arr = u.x.array[:n_local * bs].reshape(n_local, bs)
    coords = v_space.tabulate_dof_coordinates()[:n_local, :2]

    xi, yi = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    x_grid, y_grid = np.meshgrid(xi, yi)
    disp = np.stack([
        griddata(coords, u_arr[:, 0], (x_grid, y_grid), method='linear', fill_value=0.0),
        griddata(coords, u_arr[:, 1], (x_grid, y_grid), method='linear', fill_value=0.0),
    ], axis=-1)
    return converged, n_it, disp


# ======================================================================
#  B2 -- quarter ring (R_in=1, R_out=2), mesh built in (r, theta)
#  parameter space then remapped to physical (x, y) coordinates
# ======================================================================
def solve_b2(material, beam, nx, ny, theta_pts, p_vals, r_in, r_out):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx - 1, ny - 1)

    # remap node coordinates: (u, v) in [0, 1]^2 -> (r, theta) -> (x, y)
    coords = domain.geometry.x
    r = coords[:, 0] * (r_out - r_in) + r_in
    theta = coords[:, 1] * (np.pi / 2.0)
    coords[:, 0] = r * np.cos(theta)
    coords[:, 1] = r * np.sin(theta)

    v_space = functionspace(domain, ('Lagrange', 1, (2,)))
    u = Function(v_space)
    v_test = ufl.TestFunction(v_space)

    inner_facets = mesh.locate_entities_boundary(
        domain, 1, lambda x: np.isclose(np.hypot(x[0], x[1]), r_in))
    facet_tags = mesh.meshtags(domain, 1, inner_facets, np.full(len(inner_facets), 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    t_func = Function(v_space)

    def t_expr(x):
        th = np.arctan2(x[1], x[0])
        p = np.interp(th, theta_pts, p_vals)
        return np.vstack([p * np.cos(th), p * np.sin(th)])

    t_func.interpolate(t_expr)

    # symmetry BCs: u_y = 0 on theta = 0 (y = 0 edge), u_x = 0 on theta = pi/2 (x = 0 edge)
    theta0_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0))
    thetapi2_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0))
    dofs_uy = fem.locate_dofs_topological(v_space.sub(1), 1, theta0_facets)
    dofs_ux = fem.locate_dofs_topological(v_space.sub(0), 1, thetapi2_facets)
    bc_uy = fem.dirichletbc(0.0, dofs_uy, v_space.sub(1))
    bc_ux = fem.dirichletbc(0.0, dofs_ux, v_space.sub(0))

    psi = strain_energy(material, u, beam) * ufl.dx - ufl.dot(t_func, u) * ds(1)
    converged, n_it = newton_solve(psi, u, v_test, [bc_uy, bc_ux])

    bs = v_space.dofmap.index_map_bs
    n_local = v_space.dofmap.index_map.size_local
    u_arr = u.x.array[:n_local * bs].reshape(n_local, bs)
    phys_coords = v_space.tabulate_dof_coordinates()[:n_local, :2]

    # target grid: rows = theta in [0, pi/2], columns = r in [R_in, R_out]
    theta_grid_1d = np.linspace(0, np.pi / 2, ny)
    r_grid_1d = np.linspace(r_in, r_out, nx)
    theta_grid, r_grid = np.meshgrid(theta_grid_1d, r_grid_1d, indexing='ij')
    x_target = r_grid * np.cos(theta_grid)
    y_target = r_grid * np.sin(theta_grid)

    disp = np.stack([
        griddata(phys_coords, u_arr[:, 0], (x_target, y_target), method='linear', fill_value=0.0),
        griddata(phys_coords, u_arr[:, 1], (x_target, y_target), method='linear', fill_value=0.0),
    ], axis=-1)
    return converged, n_it, disp


# ======================================================================
#  Driver
# ======================================================================
CASES = {
    "B1_NH": ("B1", B1_NH), "B1_MR": ("B1", B1_MR), "B1_AB": ("B1", B1_AB),
    "B2_NH": ("B2", B2_NH), "B2_MR": ("B2", B2_MR), "B2_AB": ("B2", B2_AB),
}


def generate(case_name, n_samples=None, seed_offset=0):
    geometry, cfg = CASES[case_name]
    beam = cfg["beam"]
    material = cfg["FEM_data"]["energy_type"]
    nx, ny = beam["numPtsU"], beam["numPtsV"]
    n = n_samples if n_samples is not None else cfg["fno"]["n_data"]

    os.makedirs(cfg["dir"], exist_ok=True)
    path = cfg["path"] if n_samples is None else cfg["path"].replace(".npz", f"_n{n}_pilot.npz")

    tractions, disps = [], []
    n_fail = 0
    t0 = time.time()
    for i in range(n):
        rng = np.random.default_rng(seed_offset + i)
        if geometry == "B1":
            x_pts, vals = grf_sample(
                nx, beam["length"], beam["traction_mean"],
                beam["traction_var"] * abs(beam["traction_mean"]),
                beam["traction_scale"] * beam["length"], rng)
            converged, n_it, disp = solve_b1(material, beam, nx, ny, x_pts, vals)
        else:
            theta_pts, vals = grf_sample(
                ny, np.pi / 2, beam["pressure"],
                beam["traction_var"] * abs(beam["pressure"]),
                beam["traction_scale"] * (np.pi / 2), rng)
            converged, n_it, disp = solve_b2(
                material, beam, nx, ny, theta_pts, vals, beam["R_inner"], beam["R_outer"])

        if not converged or np.any(np.isnan(disp)):
            n_fail += 1
            print(f"  [{case_name}] sample {i}: FAILED TO CONVERGE ({n_it} iterations)")
            disp = np.zeros((ny, nx, 2))

        tractions.append(vals)
        disps.append(disp)
        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  [{case_name}] {i + 1}/{n}  ({elapsed:.1f}s, {elapsed / (i + 1):.2f}s/sample, "
                  f"{n_fail} failed)")

    tractions = np.array(tractions)
    disps = np.array(disps)
    np.savez(path, traction=tractions, disp2D=disps)
    print(f"Saved {path}: traction={tractions.shape}, disp2D={disps.shape}, {n_fail}/{n} failed")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASES.keys()),
                       help="subset of case names to generate, default: all 6")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="override sample count (default: 550, from config)")
    args = parser.parse_args()

    for case in args.cases:
        print(f"\n===== Generating {case} =====")
        generate(case, n_samples=args.n_samples)
