"""
Fast GPU ground-truth generator for the zero-shot study's own B1 parametric
test problems (Timon round-10, item 1: "the NO accuracy at N=1401 itself,
including these QoIs has not been checked").

Why this exists: build_sample_b1's own solve_fem=True path
(data_generate_B1.solve_hyperelastic_TL_spatial) is a CPU, per-element
Python-loop assembly + scipy sparse solve, run for 10 load steps of up to
30 Newton iterations each -- the reason
cell_no_inference_vs_torchfem_N1401.py deliberately built its N=1401 sample
with solve_fem=False and estimated the real solve "would cost hours at this
size" if attempted.

high_dof_convergence_study.py already has a GPU-vectorized, order-agnostic
path for the exact same B1 geometry/BCs (bottom clamped, top traction):
gpu_fem_solver.precompute_element_params_B1's own docstring states it
"Reproduces data_generate_B1.solve_hyperelastic_TL_spatial's own per-element
(E, nu) evaluation exactly", and assemble_traction_top_generic's docstring
states "identical physics and quadrature convention for Q4" as
data_generate_B1.assemble_traction_top_spatial. Both take any field
callable of (nodes), not specifically AnalyticFieldB1 (the field used
elsewhere in that file for FEM-vs-FEM mesh-convergence studies) -- so
substituting ParametricFieldB1 (the SAME field generator build_sample_b1
itself uses for the NO's own train/test samples) and feeding the result
into solve_matrix_free (the DEFAULT, already-reported GPU solver, NOT the
experimental assembled+direct one from Points 8/9) should solve the exact
same physical problem build_sample_b1 solves, just via a fast path instead
of the slow one -- solve_matrix_free's own defaults (nsteps=10,
newton_max=30, newton_tol=1e-7) already match build_sample_b1's own call to
solve_hyperelastic_TL_spatial exactly, so no override is needed for that
part.

MUST be verified against solve_hyperelastic_TL_spatial's own real output
before this is ever trusted at N=1401 -- that is what _correctness_check
does, at small N where the slow CPU solver is still cheap enough to run
here as the ground truth for the ground truth.
"""
import numpy as np
import torch

from omar_pfem.data.parametric_field import ParametricFieldB1
from omar_pfem.gpu_fem_solver import precompute_element_params_B1
from omar_pfem.high_dof_convergence_study import assemble_traction_top_generic
from omar_pfem.matrix_free_solver import solve_matrix_free


def solve_b1_fast_gpu(N, seed, material, device, dtype, Lx=1.0, Ly=1.0, order="Q4",
                       **solve_kwargs):
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4

    E_fn = ParametricFieldB1("E", seed)
    nu_fn = ParametricFieldB1("nu", seed)
    ty_fn = ParametricFieldB1("ty", seed)

    nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
    tolx = 1e-12
    bottom_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    fixed_dofs = np.concatenate([2 * bottom_nodes, 2 * bottom_nodes + 1])
    ndof = 2 * len(nodes)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    fext_full = assemble_traction_top_generic(nodes, elements, Ly, ty_fn, order)
    elem_params_np = precompute_element_params_B1(nodes, elements, E_fn, nu_fn, material)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_dofs_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
    elem_params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in elem_params_np)
    fext_free_t = torch.tensor(fext_full[free_dofs], dtype=dtype, device=device)

    u_free, stats = solve_matrix_free(
        xy_t, quad_t, free_dofs_t, elem_params_t, fext_free_t, len(free_dofs),
        material=material, order=order, device=device, dtype=dtype,
        **solve_kwargs,
    )

    u_full = np.zeros(ndof, dtype=np.float64)
    u_full[free_dofs] = u_free.detach().cpu().numpy().astype(np.float64)
    return u_full, nodes, elements, stats


def _correctness_check(N=11, material="neo_hookean", seed=0, verbose=False):
    """CPU-only. Compares solve_b1_fast_gpu's own output against the slow
    reference build_sample_b1 itself uses (data_generate_B1.
    solve_hyperelastic_TL_spatial) at a small N. Must PASS (relative
    difference at the same level as this project's other matrix-free-vs-
    reference checks, ~1e-6 or tighter) before this is ever pointed at
    N=1401."""
    import time

    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, solve_hyperelastic_TL_spatial

    device = torch.device("cpu")
    dtype = torch.float64

    E_fn = ParametricFieldB1("E", seed)
    nu_fn = ParametricFieldB1("nu", seed)
    ty_fn = ParametricFieldB1("ty", seed)
    nodes, elements = generate_grid_Q4(1.0, 1.0, N, N)

    t0 = time.time()
    u_ref = solve_hyperelastic_TL_spatial(nodes, elements, E_fn, nu_fn, ty_fn, 1.0,
                                           nsteps=10, newton_max=30, tol=1e-7,
                                           material=material)
    t_ref = time.time() - t0

    t0 = time.time()
    u_fast, nodes2, elements2, stats = solve_b1_fast_gpu(
        N, seed, material, device, dtype, verbose=verbose)
    t_fast = time.time() - t0

    assert np.allclose(nodes, nodes2) and np.array_equal(elements, elements2), \
        "mesh mismatch -- solve_b1_fast_gpu is not building the same grid as build_sample_b1"

    u_ref_flat = np.asarray(u_ref).reshape(-1)
    rel_diff = np.linalg.norm(u_fast - u_ref_flat) / np.linalg.norm(u_ref_flat)
    print(f"N={N}: reference (slow CPU) {t_ref:.2f}s, fast path (on CPU here) {t_fast:.2f}s, "
          f"relative displacement difference {rel_diff:.3e}")
    return rel_diff


if __name__ == "__main__":
    import sys

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    _correctness_check(N)
