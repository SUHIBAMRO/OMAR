"""
Validates matrix_free_solver.py's Q4 mode against the existing, already-
trusted CPU reference solver (data_generate_B1.py / B2.py's direct sparse
solve) at small N -- exactly the same discipline gpu_fem_solver.py's own
validate_gpu_fem_solver.py established for the batched GPU solver, applied
here to the matrix-free solver before it is trusted for anything larger.

Agreement is NOT expected to reach machine precision the way the dense
batched GPU solver did against the CPU reference: this solver's linear
system is solved iteratively (CG) to a finite tolerance (--cg_tol) rather
than by direct factorization, and Newton's own outer tolerance
(--newton_tol) is also finite. Agreement should instead be commensurate
with (roughly) the tighter of these two tolerances.

Usage:
  python -m omar_pfem.validate_matrix_free_solver --geometry B1 --material neo_hookean --N 11
"""
import argparse
import numpy as np
import torch

from omar_pfem.matrix_free_solver import solve_matrix_free


def main():
    parser = argparse.ArgumentParser("Validate the matrix-free Q4 solver against the CPU reference")
    parser.add_argument("--geometry", type=str, default="B1", choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--N", type=int, default=11)
    parser.add_argument("--cg_tol", type=float, default=1e-8)
    parser.add_argument("--newton_tol", type=float, default=1e-8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    if args.geometry == "B1":
        from omar_pfem.data.data_generate_B1 import (
            generate_grid_Q4, generate_random_sample_spatial, solve_hyperelastic_TL_spatial,
            assemble_traction_top_spatial,
        )
        from omar_pfem.gpu_fem_solver import precompute_element_params_B1 as precompute_params

        Lx = Ly = 1.0
        nodes, elements = generate_grid_Q4(Lx, Ly, args.N, args.N)
        E_interp, nu_interp, ty_interp, *_ = generate_random_sample_spatial(Lx, Ly, args.N, args.N, seed=42)

        print("Solving with the existing CPU reference solver (direct sparse solve)...")
        u_ref = solve_hyperelastic_TL_spatial(nodes, elements, E_interp, nu_interp, ty_interp, Ly,
                                               nsteps=10, newton_max=30, tol=1e-7, material=args.material)

        tolx = 1e-12
        bottom_nodes = np.where(nodes[:, 1] < tolx)[0]
        fixed_dofs = np.concatenate([2 * bottom_nodes, 2 * bottom_nodes + 1])
        ndof = 2 * len(nodes)
        free_dofs_np = np.setdiff1d(np.arange(ndof), fixed_dofs)
        fext_full_np = assemble_traction_top_spatial(nodes, elements, Ly, ty_interp)

        elem_params_np = precompute_params(nodes, elements, E_interp, nu_interp, args.material)
    else:
        from omar_pfem.data.data_generate_B2 import (
            generate_grid_Q4_ring, generate_random_sample_ring, solve_hyperelastic_TL_ring,
            assemble_traction_inner_curved,
        )
        from omar_pfem.gpu_fem_solver import precompute_element_params_B2 as precompute_params

        R_in, R_out = 1.0, 2.0
        nodes, elements = generate_grid_Q4_ring(R_in, R_out, args.N, args.N)
        E_interp, nu_interp, p_interp, *_ = generate_random_sample_ring(R_in, R_out, args.N, args.N, seed=42)

        print("Solving with the existing CPU reference solver (direct sparse solve)...")
        u_ref = solve_hyperelastic_TL_ring(nodes, elements, E_interp, nu_interp, p_interp, R_in,
                                            nsteps=10, newton_max=30, tol=1e-7, material=args.material)

        tolx = 1e-9
        theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
        thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]
        fixed_dofs = np.concatenate([2 * theta0_nodes + 1, 2 * thetahalfpi_nodes])
        ndof = 2 * len(nodes)
        free_dofs_np = np.setdiff1d(np.arange(ndof), fixed_dofs)
        fext_full_np = assemble_traction_inner_curved(nodes, elements, R_in, p_interp)

        elem_params_np = precompute_params(nodes, elements, E_interp, nu_interp, args.material)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_dofs_t = torch.tensor(free_dofs_np, dtype=torch.long, device=device)
    elem_params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in elem_params_np)
    fext_free_t = torch.tensor(fext_full_np[free_dofs_np], dtype=dtype, device=device)

    print(f"\nSolving with the NEW matrix-free Newton-CG solver (Q4, {device})...")
    u_free, stats = solve_matrix_free(
        xy_t, quad_t, free_dofs_t, elem_params_t, fext_free_t, n_free=len(free_dofs_np),
        material=args.material, order="Q4", nsteps=10, newton_max=30,
        newton_tol=args.newton_tol, cg_tol=args.cg_tol, device=device, dtype=dtype)

    u_full_mf = np.zeros(ndof)
    u_full_mf[free_dofs_np] = u_free.cpu().numpy()
    u_mf = u_full_mf.reshape(len(nodes), 2)

    diff = u_mf - u_ref
    max_abs_diff = np.max(np.abs(diff))
    max_ref = np.max(np.abs(u_ref)) + 1e-30
    mean_rel_err = np.mean(np.abs(diff)) / (np.mean(np.abs(u_ref)) + 1e-30)
    max_rel_err = max_abs_diff / max_ref

    print(f"\n{'='*70}\nVALIDATION RESULT (N={args.N}, {args.geometry} x {args.material})\n{'='*70}")
    print(f"max|u_ref| = {max_ref:.6e}")
    print(f"max abs. diff (matrix-free vs. CPU reference) = {max_abs_diff:.6e}")
    print(f"max relative diff = {max_rel_err:.6e}")
    print(f"mean relative diff = {mean_rel_err:.6e}")
    print(f"Newton iterations (total across all load steps) = {stats['newton_iters_total']}")
    print(f"CG iterations (total) = {stats['cg_iters_total']}, CG failures = {stats['cg_failures']}")
    verdict = "PASS" if max_rel_err < 1e-4 else "FAIL"
    print(f"\nVerdict: {verdict} (threshold: max relative diff < 1e-4, commensurate with "
          f"cg_tol={args.cg_tol:.0e} and newton_tol={args.newton_tol:.0e})")


if __name__ == "__main__":
    main()
