"""
One-off correctness check (not part of the study's normal pipeline):
solves the SAME problem instance (same mesh, same E/nu/load fields) with
(a) the existing, validated CPU Newton-Raphson solver
(data_generate_B{1,2}.solve_hyperelastic_TL_{spatial,ring}) and (b) the
new GPU-style autodiff Newton-Raphson solver (gpu_fem_solver.py), and
reports how closely they agree. Run this before trusting any timing
numbers from gpu_fem_solver.py -- an implementation that runs without
error but converges to the wrong equilibrium would make any speed
comparison meaningless.

Usage: python -m omar_pfem.validate_gpu_fem_solver --geometry B1 [--N 11] [--cpu]
       python -m omar_pfem.validate_gpu_fem_solver --geometry B2 [--N 11] [--cpu]
"""
import argparse
import numpy as np
import torch

from omar_pfem.gpu_fem_solver import (
    precompute_element_params_B1, precompute_element_params_B2,
    solve_batch_gpu_newton_B1, solve_batch_gpu_newton_B2,
    assemble_fext_B1_numpy, assemble_fext_B2_numpy,
)


def run_b1(args, device):
    from omar_pfem.data.data_generate_B1 import (
        generate_grid_Q4, generate_random_sample_spatial, solve_hyperelastic_TL_spatial,
    )
    Lx = Ly = 1.0
    nodes, elements = generate_grid_Q4(Lx, Ly, args.N, args.N)
    tol = 1e-12
    bottom_nodes = np.where(nodes[:, 1] < tol)[0]

    E_interp, nu_interp, ty_interp, *_ = generate_random_sample_spatial(Lx, Ly, args.N, args.N, seed=args.seed)

    print(f"\n--- (a) CPU reference solver (solve_hyperelastic_TL_spatial), N={args.N} ---")
    u_cpu = solve_hyperelastic_TL_spatial(
        nodes, elements, E_interp, nu_interp, ty_interp, Ly,
        nsteps=10, newton_max=30, tol=1e-7, material=args.material,
    )

    print(f"\n--- (b) GPU-style autodiff Newton solver, N={args.N} ---")
    dtype = torch.float64
    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    bottom_nodes_t = torch.tensor(bottom_nodes, dtype=torch.long, device=device)

    elem_params_np = precompute_element_params_B1(nodes, elements, E_interp, nu_interp, args.material)
    elem_params_t = tuple(torch.tensor(p, dtype=dtype, device=device).unsqueeze(0) for p in elem_params_np)

    fext_full_np = assemble_fext_B1_numpy(nodes, elements, Ly, ty_interp)
    fext_full_t = torch.tensor(fext_full_np, dtype=dtype, device=device).unsqueeze(0)

    u_gpu_batch = solve_batch_gpu_newton_B1(
        xy_t, quad_t, bottom_nodes_t, elem_params_t, fext_full_t,
        material=args.material, nsteps=10, newton_max=30, tol=1e-7, device=device,
    )
    return u_cpu, u_gpu_batch[0].detach().cpu().numpy()


def run_b2(args, device):
    from omar_pfem.data.data_generate_B2 import (
        generate_grid_Q4_ring, generate_random_sample_ring, solve_hyperelastic_TL_ring,
    )
    R_in, R_out = 1.0, 2.0
    nodes, elements = generate_grid_Q4_ring(R_in, R_out, args.N, args.N)
    tolx = 1e-9
    theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]

    E_interp, nu_interp, p_interp, *_ = generate_random_sample_ring(R_in, R_out, args.N, args.N, seed=args.seed)

    print(f"\n--- (a) CPU reference solver (solve_hyperelastic_TL_ring), N={args.N} ---")
    u_cpu = solve_hyperelastic_TL_ring(
        nodes, elements, E_interp, nu_interp, p_interp, R_in,
        nsteps=10, newton_max=30, tol=1e-7, material=args.material,
    )

    print(f"\n--- (b) GPU-style autodiff Newton solver, N={args.N} ---")
    dtype = torch.float64
    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    theta0_t = torch.tensor(theta0_nodes, dtype=torch.long, device=device)
    thetahalfpi_t = torch.tensor(thetahalfpi_nodes, dtype=torch.long, device=device)

    elem_params_np = precompute_element_params_B2(nodes, elements, E_interp, nu_interp, args.material)
    elem_params_t = tuple(torch.tensor(p, dtype=dtype, device=device).unsqueeze(0) for p in elem_params_np)

    fext_full_np = assemble_fext_B2_numpy(nodes, elements, R_in, p_interp)
    fext_full_t = torch.tensor(fext_full_np, dtype=dtype, device=device).unsqueeze(0)

    u_gpu_batch = solve_batch_gpu_newton_B2(
        xy_t, quad_t, theta0_t, thetahalfpi_t, elem_params_t, fext_full_t,
        material=args.material, nsteps=10, newton_max=30, tol=1e-7, device=device,
    )
    return u_cpu, u_gpu_batch[0].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--N", type=int, default=11)
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    if args.geometry == "B1":
        u_cpu, u_gpu = run_b1(args, device)
    else:
        u_cpu, u_gpu = run_b2(args, device)

    diff = u_gpu - u_cpu
    abs_err = np.sqrt(np.sum(diff ** 2, axis=1))
    ref_norm = np.sqrt(np.sum(u_cpu ** 2, axis=1)) + 1e-12
    rel_err = abs_err / ref_norm

    print("\n" + "=" * 80)
    print("CORRECTNESS CHECK: GPU-style solver vs. validated CPU reference solver")
    print("=" * 80)
    print(f"max |u_cpu|:                 {np.abs(u_cpu).max():.6e}")
    print(f"max |u_gpu - u_cpu| (abs):   {abs_err.max():.6e}")
    print(f"mean node-wise relative err: {rel_err.mean():.6e}")
    print(f"max  node-wise relative err: {rel_err.max():.6e}")
    print("=" * 80)
    if rel_err.mean() < 1e-6:
        print("PASS: the two solvers agree to within machine precision -- same equilibrium.")
    else:
        print("FAIL: the two solvers disagree more than expected -- do not trust "
              "gpu_fem_solver.py's timing numbers until this is resolved.")


if __name__ == "__main__":
    main()
