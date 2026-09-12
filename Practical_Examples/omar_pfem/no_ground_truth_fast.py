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
into a fast GPU solver should solve the exact same physical problem
build_sample_b1 solves, just via a fast path instead of the slow one.

**Solver backend: assembled_direct_solver.solve_assembled_direct, NOT
solve_matrix_free.** The first version of this module used solve_matrix_free
(the DEFAULT solver reported everywhere else in this project), reasoning
that touching the experimental assembled+direct solver from Points 8/9
should be avoided per the standing "ask before treating it as more than an
experiment" rule. That was wrong in a way that would have cost ~7.5 hours
of real GPU time: solve_matrix_free's own already-measured N=1401 cost is
27,257.4s (the 204-306x-slower-than-torch-fem finding that motivated
building the assembled+direct solver in the first place), not something
this module's speed claim can survive. Switched to solve_assembled_direct
once Omar explicitly confirmed this is fine PURELY as an internal
ground-truth calculation tool (its own accuracy has already been verified
bit-for-bit identical to solve_matrix_free/torch-fem in every check done so
far -- the standing rule is about not presenting it as a finalized RESULT
without review, not about whether its output can be trusted as a correct
FEM solution, which is separately and thoroughly established).

One real wrinkle checked before trusting this, not assumed: solve_matrix_free
and solve_hyperelastic_TL_spatial both use 10-step incremental LOAD STEPPING
(nsteps=10) to help Newton's own convergence, while solve_assembled_direct
does a SINGLE full-load Newton solve with no load-stepping and no
warm-start-from-previous-step option. Verified directly (not assumed) that
this does not matter for THIS problem: single-shot solve_assembled_direct
converges to the correct answer at N=11 (relative difference 8.41e-11 vs.
the slow reference) and N=21 (8.44e-11) -- both at the same
bit-for-bit-identical level as every other check in this project, and
faster even on CPU (N=21: 0.28s vs. the slow reference's 17.7s) before this
was ever pointed at N=1401 or a GPU.

MUST be verified against solve_hyperelastic_TL_spatial's own real output
before this is ever trusted at N=1401 -- that is what _correctness_check
does, at small N where the slow CPU solver is still cheap enough to run
here as the ground truth for the ground truth.
"""
import numpy as np
import torch

from omar_pfem.assembled_direct_solver import solve_assembled_direct
from omar_pfem.data.parametric_field import ParametricFieldB1
from omar_pfem.gpu_fem_solver import precompute_element_params_B1
from omar_pfem.high_dof_convergence_study import assemble_traction_top_generic


def solve_b1_fast_gpu(N, seed, material, device, dtype, Lx=1.0, Ly=1.0, order="Q4",
                       nsteps=1, **solve_kwargs):
    """nsteps=1 (default, unchanged from every result already published with
    this function, including the N=11/N=21 correctness checks below): a
    single full-load Newton solve, exactly as before.

    nsteps>1: genuine incremental load-stepping built around
    solve_assembled_direct's new u0_init parameter -- solve at
    (step/nsteps)*fext_full, feed that converged displacement back in as
    the next step's warm start, exactly mirroring solve_matrix_free's/
    solve_hyperelastic_TL_spatial's own nsteps=10 convention (their own
    absolute tol/max_iter defaults are UNCHANGED here; this only adds the
    load-stepping they already have and solve_assembled_direct's own
    single-shot call does not). Added 2026-09-12 after tightening tol/
    max_iter alone left the ground truth genuinely stalled and not
    converged at N=1401 (relative residual 2.612e-03, identical whether
    tol/max_iter were the function's own defaults or tightened by 100x/2x)
    -- i.e. the problem was never "not enough iterations" at the full
    load, it was starting Newton at the full load from zero at all."""
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
    mu, lam = precompute_element_params_B1(nodes, elements, E_fn, nu_fn, material)

    # solve_assembled_direct calls torch.set_default_dtype(dtype) internally and never
    # restores it -- harmless in isolation, but a real bug for any caller (like the
    # N=1401 accuracy pipeline) that runs an fp32 NO forward pass in the SAME process
    # afterward: new tensors created without an explicit dtype elsewhere (e.g. inside
    # the model's own input-normalization helper) silently become float64, causing a
    # "mat1 and mat2 must have the same dtype" crash in the model's own first Linear
    # layer. Save/restore the global default here rather than patching the shared
    # solver file, since this is the only call site that combines the two.
    _prev_default_dtype = torch.get_default_dtype()
    try:
        if nsteps <= 1:
            u_full_t = solve_assembled_direct(
                nodes, elements, free_dofs, fext_full, mu, lam,
                dtype=dtype, material=material, order=order, device=device,
                **solve_kwargs,
            )
        else:
            u0 = None
            for step in range(1, nsteps + 1):
                alpha = step / nsteps
                u_full_t = solve_assembled_direct(
                    nodes, elements, free_dofs, alpha * fext_full, mu, lam,
                    dtype=dtype, material=material, order=order, device=device,
                    u0_init=u0, **solve_kwargs,
                )
                u0 = (u_full_t.reshape(-1) if torch.is_tensor(u_full_t)
                      else np.asarray(u_full_t).reshape(-1))
    finally:
        torch.set_default_dtype(_prev_default_dtype)

    if torch.is_tensor(u_full_t):
        u_full = u_full_t.detach().cpu().numpy()
    else:
        u_full = np.asarray(u_full_t)
    u_full = u_full.astype(np.float64).reshape(-1)
    return u_full, nodes, elements, None


def check_convergence(nodes, elements, free_dofs, fext_full, mu, lam, u_full,
                       material, order, device, dtype):
    """Independent, post-hoc convergence check for solve_assembled_direct's
    own output -- reimplements the SAME residual solve_assembled_direct's
    own internal Newton loop drives to zero (grad(energy) - f_ext on the
    free DOFs), from OUTSIDE that function, since solve_assembled_direct
    hardcodes verbose=False internally and exposes no residual/convergence
    info to its caller. Added after a real N=1401 accuracy run produced
    physically implausible numbers (e.g. peak PK1 stress ~3.4e6x the
    ground truth's own value) that look much more like "the ground truth
    itself never converged at this size" than "the operator is merely
    inaccurate here" -- this checks that hypothesis directly instead of
    assuming either explanation."""
    from omar_pfem.matrix_free_solver import element_energy_order_agnostic, precompute_shape_data
    from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch

    n_nodes = nodes.shape[0]
    n_dof = 2 * n_nodes
    fixed_set = set(np.setdiff1d(np.arange(n_dof), free_dofs).tolist())
    free_mask_dof = torch.tensor([i not in fixed_set for i in range(n_dof)], device=device)

    xy = torch.tensor(nodes, dtype=dtype, device=device)
    quad = torch.tensor(elements, dtype=torch.long, device=device)
    energy_density_fn, _ = get_material_fns_torch(material)
    shape_data = precompute_shape_data(order, device, dtype)
    elem_params = (torch.as_tensor(mu, dtype=dtype, device=device),
                   torch.as_tensor(lam, dtype=dtype, device=device))
    f_ext_flat = torch.tensor(fext_full, dtype=dtype, device=device)
    u_flat = torch.tensor(u_full, dtype=dtype, device=device)

    def energy_fn(u):
        uv = u.reshape(n_nodes, 2)
        return element_energy_order_agnostic(xy, quad, uv, elem_params, energy_density_fn,
                                              shape_data, dtype)

    grad_full = torch.func.grad(energy_fn)(u_flat)
    res = grad_full - f_ext_flat
    res_free = res[free_mask_dof]
    f_ext_free_norm = torch.linalg.norm(f_ext_flat[free_mask_dof])
    res_norm = torch.linalg.norm(res_free)
    rel_res = (res_norm / f_ext_free_norm.clamp_min(1e-30)).item()

    return {
        "residual_norm": res_norm.item(),
        "f_ext_free_norm": f_ext_free_norm.item(),
        "relative_residual": rel_res,
        "u_max_abs": u_flat.abs().max().item(),
        "converged_likely": rel_res < 1e-4,
    }


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
        N, seed, material, device, dtype)
    t_fast = time.time() - t0

    assert np.allclose(nodes, nodes2) and np.array_equal(elements, elements2), \
        "mesh mismatch -- solve_b1_fast_gpu is not building the same grid as build_sample_b1"

    u_ref_flat = np.asarray(u_ref).reshape(-1)
    rel_diff = np.linalg.norm(u_fast - u_ref_flat) / np.linalg.norm(u_ref_flat)
    print(f"N={N}: reference (slow CPU) {t_ref:.2f}s, fast path (on CPU here) {t_fast:.2f}s, "
          f"relative displacement difference {rel_diff:.3e}")
    return {
        "N": N, "material": material, "seed": seed,
        "t_reference_slow_cpu_s": t_ref, "t_fast_path_on_cpu_s": t_fast,
        "relative_displacement_difference": rel_diff,
    }


if __name__ == "__main__":
    import json
    import sys

    Ns = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [11, 21]
    results = [_correctness_check(N) for N in Ns]
    out_path = "omar_pfem/no_ground_truth_fast_correctness.json"
    with open(out_path, "w") as f:
        json.dump({"checks": results}, f, indent=2)
    print("Saved:", out_path)
