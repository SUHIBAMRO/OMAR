"""Validates multigrid_precond.py before it is ever pointed at the real
N=401/701 problem it exists to fix -- same discipline every other solver
change in this codebase has followed (validate_matrix_free_solver.py,
validate_gpu_fem_solver.py): a plausible-looking convergence-rate
improvement is not proof of correctness, an independent check against a
trusted reference is.

Three checks, in increasing order of what they'd catch:
  1. Prolongation P is a partition of unity (every row sums to 1) and
     restriction is exactly its 1/4-scaled transpose -- a pure linear-
     algebra check on the interpolation operators themselves, independent
     of the FEM problem entirely. If this fails, nothing downstream can be
     trusted no matter how plausible the CG numbers look.
  2. The full V-cycle-preconditioned solve reproduces the SAME converged
     displacement field as the existing, already-validated plain-Jacobi
     solve, on tiny B1 and B2 meshes (both geometries, since B2's mixed
     partial-fixed-DOF nodes are exactly where the earlier block2x2
     preconditioner needed its own special-case handling -- multigrid's
     restriction/prolongation must not silently corrupt those nodes
     either). Agreement should be commensurate with cg_tol, not exact,
     since both are iterative solves to a finite tolerance.
  3. Only once (1) and (2) both pass: a CPU-feasible moderate-N comparison
     of cg_iters between "jacobi" and "mgv", to see whether there is even
     a DIRECTIONAL benefit before spending any GPU time on the real
     N=401/701 problem -- exactly the same cost-conscious staging this
     project used before the block2x2 GPU re-run.

Usage:
  python -m omar_pfem.validate_multigrid_precond
"""
import numpy as np
import torch

from omar_pfem.multigrid_precond import (
    build_prolongation_matrix, coarsen_N, build_mg_hierarchy)
from omar_pfem.matrix_free_solver import solve_matrix_free
from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs


def check_interpolation_operators():
    print("=== 1. Prolongation/restriction operator checks ===")
    ok = True
    for Nc in (5, 9, 13):
        Nf = 2 * (Nc - 1) + 1
        P = build_prolongation_matrix(Nc, Nf)
        row_sums = np.asarray(P.sum(axis=1)).reshape(-1)
        max_dev = np.abs(row_sums - 1.0).max()
        print(f"  Nc={Nc} Nf={Nf}: max |row sum - 1| = {max_dev:.3e}")
        ok &= max_dev < 1e-14

        R = (P.T * 0.25).tocsr()
        # A constant coarse field should prolong-then-restrict to itself
        # scaled by... check R @ (P @ ones) == ones exactly on this
        # structured grid's INTERIOR (boundary rows of P^T/4 have fewer
        # terms so do not sum to 1 -- that is correct and expected, not
        # checked here).
        ones_c = np.ones(Nc * Nc)
        rt = R @ (P @ ones_c)
        # interior coarse nodes (not on the Nc x Nc grid's own boundary)
        interior_mask = np.zeros(Nc * Nc, dtype=bool)
        for J in range(1, Nc - 1):
            for I in range(1, Nc - 1):
                interior_mask[J * Nc + I] = True
        if interior_mask.any():
            dev = np.abs(rt[interior_mask] - ones_c[interior_mask]).max()
            print(f"    R(P(1)) vs 1 on interior coarse nodes: max diff = {dev:.3e}")
            ok &= dev < 1e-12
    print("  " + ("PASS" if ok else "FAIL"))
    return ok


def check_solution_matches(geometry, material, N, device, dtype):
    print(f"\n=== 2. V-cycle solve vs plain-Jacobi solve, {geometry} x {material}, N={N} ===")
    Ns = [N]
    while True:
        nc = coarsen_N(Ns[-1])
        if nc is None or nc < 5 or len(Ns) >= 4:
            break
        Ns.append(nc)
    print(f"  hierarchy N: {Ns}")

    mesh_tuples = []
    fine_fext = None
    for n in Ns:
        nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
            geometry, "Q4", n, material, device, dtype)
        mesh_tuples.append((n, nodes, elements, free_dofs, elem_params))
        if fine_fext is None:
            fine_fext = fext_full

    _, fine_nodes, fine_elements, fine_free, fine_params = mesh_tuples[0]
    xy_t = torch.tensor(fine_nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(fine_elements, dtype=torch.long, device=device)
    free_t = torch.tensor(fine_free, dtype=torch.long, device=device)
    params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in fine_params)
    fext_free_t = torch.tensor(fine_fext[fine_free], dtype=dtype, device=device)

    kwargs = dict(material=material, order="Q4", nsteps=5, newton_max=40,
                  newton_tol=1e-10, cg_tol=1e-10, cg_max_iter=3000,
                  use_jacobi=True, device=device, dtype=dtype, verbose=False)

    u_jacobi, stats_jacobi = solve_matrix_free(
        xy_t, quad_t, free_t, params_t, fext_free_t, n_free=len(fine_free),
        precond_kind="jacobi", **kwargs)

    mg_hierarchy = build_mg_hierarchy(mesh_tuples, dtype, device)
    u_mgv, stats_mgv = solve_matrix_free(
        xy_t, quad_t, free_t, params_t, fext_free_t, n_free=len(fine_free),
        precond_kind="mgv", mg_hierarchy=mg_hierarchy, **kwargs)

    diff = (u_jacobi - u_mgv).norm().item()
    ref_norm = u_jacobi.norm().item() + 1e-30
    rel_diff = diff / ref_norm
    print(f"  jacobi: cg_iters_total={stats_jacobi['cg_iters_total']}, "
          f"cg_failures={stats_jacobi['cg_failures']}")
    print(f"  mgv:    cg_iters_total={stats_mgv['cg_iters_total']}, "
          f"cg_failures={stats_mgv['cg_failures']}")
    print(f"  relative diff between the two solutions: {rel_diff:.3e}")
    ok = rel_diff < 1e-6 and stats_mgv["cg_failures"] == 0
    print("  " + ("PASS" if ok else "FAIL"))
    return ok


def check_iteration_count(geometry, material, N, device, dtype):
    print(f"\n=== 3. CG iteration count, jacobi vs mgv, {geometry} x {material}, N={N} ===")
    Ns = [N]
    while True:
        nc = coarsen_N(Ns[-1])
        if nc is None or nc < 13 or len(Ns) >= 5:
            break
        Ns.append(nc)
    print(f"  hierarchy N: {Ns}")

    mesh_tuples = []
    fine_fext = None
    for n in Ns:
        nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
            geometry, "Q4", n, material, device, dtype)
        mesh_tuples.append((n, nodes, elements, free_dofs, elem_params))
        if fine_fext is None:
            fine_fext = fext_full

    _, fine_nodes, fine_elements, fine_free, fine_params = mesh_tuples[0]
    xy_t = torch.tensor(fine_nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(fine_elements, dtype=torch.long, device=device)
    free_t = torch.tensor(fine_free, dtype=torch.long, device=device)
    params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in fine_params)
    fext_free_t = torch.tensor(fine_fext[fine_free], dtype=dtype, device=device)

    kwargs = dict(material=material, order="Q4", nsteps=10, newton_max=30,
                  newton_tol=1e-8, cg_tol=1e-8, cg_max_iter=3000,
                  use_jacobi=True, device=device, dtype=dtype, verbose=False)

    import time
    t0 = time.time()
    u_jacobi, stats_jacobi = solve_matrix_free(
        xy_t, quad_t, free_t, params_t, fext_free_t, n_free=len(fine_free),
        precond_kind="jacobi", **kwargs)
    t_jacobi = time.time() - t0

    mg_hierarchy = build_mg_hierarchy(mesh_tuples, dtype, device)
    t0 = time.time()
    u_mgv, stats_mgv = solve_matrix_free(
        xy_t, quad_t, free_t, params_t, fext_free_t, n_free=len(fine_free),
        precond_kind="mgv", mg_hierarchy=mg_hierarchy, **kwargs)
    t_mgv = time.time() - t0

    print(f"  jacobi: cg_iters_total={stats_jacobi['cg_iters_total']}, "
          f"cg_failures={stats_jacobi['cg_failures']}, wall={t_jacobi:.1f}s")
    print(f"  mgv:    cg_iters_total={stats_mgv['cg_iters_total']}, "
          f"cg_failures={stats_mgv['cg_failures']}, wall={t_mgv:.1f}s")


if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float64

    ok1 = check_interpolation_operators()
    ok2a = check_solution_matches("B1", "neo_hookean", 17, device, dtype)
    ok2b = check_solution_matches("B2", "neo_hookean", 17, device, dtype)

    if ok1 and ok2a and ok2b:
        print("\nAll correctness checks PASSED -- proceeding to the iteration-count comparison.")
        check_iteration_count("B1", "neo_hookean", 65, device, dtype)
    else:
        print("\nCorrectness checks FAILED -- not running the iteration-count "
              "comparison until these are fixed.")
