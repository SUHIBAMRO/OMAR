"""EXPERIMENTAL (2026-09-11), NOT YET RUN ON GPU, NOT A FINALIZED RESULT.

Omar's own explicit request: torch-fem and TensorMesh's memory-heavy
explicit-matrix-assembly + direct-solve approach already works correctly
and fast at every resolution this project has tested "ours" own
matrix-free solver at (up to N=1401, using ~69GB of an 80GB A100 --
comfortably inside the limit, never crashing). Since staying matrix-free
is a choice made to avoid an out-of-memory failure that, at these exact
sizes, never actually happens for the other two solvers, Omar asked
directly: "is there something we could try?" -- build "ours" own solver
the SAME way (assemble the global sparse tangent explicitly, factorize
it directly) and see whether it becomes competitive in speed too,
instead of assuming matrix-free is the only option.

WHY THIS DOES NOT NEED TENSORMESH AT ALL, EVEN THOUGH THE ASSEMBLY CODE
WAS FIRST WRITTEN FOR THE TENSORMESH COMPARISON:
tensormesh_comparison.py's own build_sparse_jac_fn already builds OUR OWN
element energy's Hessian (matrix_free_solver.py's own
_local_element_energy under torch.func.vmap+hessian, materials_torch.py's
own energy density) -- it never touches TensorMesh's ElementAssembler for
the physics, only (in solve_tensormesh) for obtaining a correctly-shaped
SparseTensor object to call .nonlinear_solve on. torch_sla.SparseTensor
can be built directly from a (values, row, col, shape) COO triple with no
TensorMesh involvement at all (confirmed by reading torch_sla's own
installed sparse_tensor/core.py __init__ directly: it just stores the
four arguments, no mesh/assembler object required) -- so this module
builds that dummy A itself, dropping the TensorMesh dependency entirely
for what is otherwise "our own" solver, just assembled instead of
matrix-free.

RESIDUAL CONVENTION: FULL-DOF-with-masking (torch.where(free_mask_dof,
res, u_flat)), matching solve_tensormesh's own convention and
build_sparse_jac_fn's own fixed-row-identity construction exactly --
NOT solve_matrix_free's free-DOF-only convention (index_copy into a zero
vector). The two conventions are not interchangeable: build_sparse_jac_fn
was written for the former and reused here unmodified.

STANDING CONSTRAINT (see PROJECT_STATUS.md's own 2026-09-10 reminder,
which applies equally to this idea): this is the same category of change
as the cached-Hessian speedup -- a new way to make "ours" own core solver
faster. It must be verified on the GPU test notebook first, and then run
past Timon, before being finalized, applied broadly, or presented as an
official project result. This module only adds an opt-in alternative
solve path; it changes no existing default and no already-published
number.
"""
import numpy as np
import torch
from torch_sla import SparseTensor

from omar_pfem.tensormesh_comparison import build_sparse_jac_fn


def _newton_cudss_reuse_analysis(residual_fn, jac_fn, u0, f_ext, tol, atol, max_iter,
                                  line_search=True, verbose=False):
    """Custom Newton loop, bypassing torch_sla's own SparseTensor.
    nonlinear_solve entirely for the linear-solve step -- added 2026-09-11
    after a real, GPU-measured finding (omar_pfem/profile_cudss_analysis_
    reuse.py): reading torch_sla's own installed source
    (NonlinearSolveFunction.forward) showed it calls spsolve(...) fresh on
    EVERY Newton iteration, and its own cuDSS backend does a brand-new
    handle + ANALYSIS + FACTORIZATION + SOLVE + destroy every single call.
    ANALYSIS (fill-reducing reordering) depends ONLY on the Jacobian's
    sparsity PATTERN, which build_sparse_jac_fn's own row/col template
    never changes within one Newton solve -- confirmed on CPU (identical
    CSR structure across different displacement fields) AND on real GPU
    hardware (Omar's own A100 run, N=401): ANALYSIS was 95.7% of one full
    solve's own time, and reusing it gave the SAME answer as redoing it
    (relative difference ~1e-14, floating-point noise) at up to ~3x
    speedup on just 3 calls -- growing towards roughly 22x per iteration
    once ANALYSIS's one-time cost is amortized over a full Newton solve's
    worth of iterations, not just 3.

    Mirrors torch_sla's own Newton+Armijo-line-search logic exactly
    (NonlinearSolveFunction.forward: same convergence test, same
    backtracking rule, same du = J^-1 @ (-F) sign convention) so behavior
    is otherwise identical to calling A.nonlinear_solve(...) -- the ONLY
    difference is the linear-solve step reuses cuDSS's own ANALYSIS
    across iterations instead of redoing it. CUDA/cuDSS only (no CPU
    fallback -- cuDSS itself has none); asserts the sparsity pattern
    really does stay fixed at every iteration rather than assuming it,
    and raises rather than silently returning a wrong answer if it ever
    doesn't.

    Returns (u, stats) where stats has n_newton_iters and phase-time
    totals (analysis/factorization/solve), for verifying the real
    end-to-end speedup this buys, not just the isolated-call number
    profile_cudss_analysis_reuse.py already measured."""
    import time as _time

    import nvmath.bindings.cudss as cudss
    from torch_sla.backends.nvmath_backend import CUDA_R_32I, _DTYPE_MAP

    def F_eval(uu):
        return residual_fn(uu, None, f_ext)

    u = u0.clone()
    F = F_eval(u)
    f0 = float(torch.linalg.vector_norm(F))
    converged = False
    stats = {"n_newton_iters": 0, "t_analysis_s": 0.0, "t_factorization_s": 0.0, "t_solve_s": 0.0}

    handle = config = data = A_desc = None
    cval_buf = crow0 = ccol0 = None
    value_type = None

    try:
        for it in range(max_iter):
            fnorm = float(torch.linalg.vector_norm(F))
            if verbose:
                print(f"  Newton iter {it}: ||F|| = {fnorm:.3e}")
            if fnorm < atol or (it > 0 and fnorm < tol * f0):
                converged = True
                break
            stats["n_newton_iters"] += 1

            val, row, col, shape = jac_fn(u, None)
            m, n = shape
            indices = torch.stack([row, col], dim=0)
            A_coo = torch.sparse_coo_tensor(indices, val, (m, n)).coalesce()
            A_csr = A_coo.to_sparse_csr()
            crow = A_csr.crow_indices().int()
            ccol = A_csr.col_indices().int()
            cval = A_csr.values()

            if handle is None:
                crow0, ccol0 = crow, ccol
                cval_buf = cval.clone()
                value_type = _DTYPE_MAP[cval_buf.dtype]
                handle = cudss.create()
                cudss.set_stream(handle, torch.cuda.current_stream().cuda_stream)
                config = cudss.config_create()
                data = cudss.data_create(handle)
                A_desc = cudss.matrix_create_csr(
                    m, n, cval_buf.numel(), crow0.data_ptr(), 0, ccol0.data_ptr(),
                    cval_buf.data_ptr(), CUDA_R_32I, value_type,
                    cudss.MatrixType.GENERAL.value, cudss.MatrixViewType.FULL.value,
                    cudss.IndexBase.ZERO.value)
                do_analysis = True
            else:
                if not (torch.equal(crow, crow0) and torch.equal(ccol, ccol0)):
                    raise RuntimeError(
                        "Jacobian sparsity pattern changed mid-solve -- the "
                        "analysis-reuse assumption does not hold here; aborting "
                        "rather than silently returning a wrong answer.")
                cval_buf.copy_(cval)
                do_analysis = False

            neg_F = (-F).unsqueeze(0).contiguous()
            du_col = torch.zeros_like(neg_F)
            b_desc = cudss.matrix_create_dn(m, 1, m, neg_F.data_ptr(), value_type,
                                             cudss.Layout.COL_MAJOR.value)
            x_desc = cudss.matrix_create_dn(m, 1, m, du_col.data_ptr(), value_type,
                                             cudss.Layout.COL_MAJOR.value)

            torch.cuda.synchronize(); t0 = _time.time()
            if do_analysis:
                cudss.execute(handle, cudss.Phase.ANALYSIS.value, config, data, A_desc, x_desc,
                               b_desc)
            torch.cuda.synchronize(); t1 = _time.time()
            cudss.execute(handle, cudss.Phase.FACTORIZATION.value, config, data, A_desc, x_desc,
                           b_desc)
            torch.cuda.synchronize(); t2 = _time.time()
            cudss.execute(handle, cudss.Phase.SOLVE.value, config, data, A_desc, x_desc, b_desc)
            torch.cuda.synchronize(); t3 = _time.time()
            stats["t_analysis_s"] += t1 - t0
            stats["t_factorization_s"] += t2 - t1
            stats["t_solve_s"] += t3 - t2

            du = du_col.squeeze(0).clone()
            cudss.matrix_destroy(x_desc)
            cudss.matrix_destroy(b_desc)

            if line_search:
                alpha = 1.0
                for _ in range(30):
                    Fn = F_eval(u + alpha * du)
                    if float(torch.linalg.vector_norm(Fn)) <= (1 - 1e-4 * alpha) * fnorm:
                        break
                    alpha *= 0.5
            else:
                alpha = 1.0
            u = u + alpha * du
            F = F_eval(u)
        else:
            fnorm = float(torch.linalg.vector_norm(F))

        if not converged and float(torch.linalg.vector_norm(F)) >= max(atol, tol * f0):
            import warnings
            warnings.warn(
                f"Newton (cuDSS analysis-reuse) did not converge in {max_iter} iters: "
                f"||F||={float(torch.linalg.vector_norm(F)):.3e}")
    finally:
        if A_desc is not None:
            cudss.matrix_destroy(A_desc)
        if data is not None:
            cudss.data_destroy(handle, data)
        if config is not None:
            cudss.config_destroy(config)
        if handle is not None:
            cudss.destroy(handle)

    return u, stats


def solve_assembled_direct(nodes, elements, free_dofs, fext_full, mu, lam, dtype=torch.float64,
                            tol=1e-8, material="neo_hookean", order="Q4", device=None,
                            linear_solver=None, max_iter=30, reuse_analysis=False,
                            return_stats=False):
    """Newton + a real direct solver (cuDSS on CUDA, matching Timon's own
    "Newton-type solve with a direct solver" requirement, and the same
    linear_solver policy solve_tensormesh already uses: 'auto' silently
    drops to an iterative-only backend above torch_sla's own
    CUDA_ITERATIVE_THRESHOLD=2M DOF regardless of cuDSS availability, so
    'cudss' is forced explicitly on CUDA instead of accepting that
    untested), applied to OUR OWN residual/energy
    (matrix_free_solver.py's own element_energy_order_agnostic), not
    TensorMesh's model.energy.

    Returns the full nodal displacement field, shape (n_nodes, 2),
    matching solve_matrix_free's/solve_tensormesh's own return
    convention (solve_matrix_free itself returns free-DOF-only u_free,
    so compare against a full-DOF reconstruction of that -- see this
    module's own _correctness_check).

    reuse_analysis=False (default, unchanged from every result already
    published with this function): calls torch_sla's own
    A.nonlinear_solve, which redoes cuDSS's ANALYSIS+FACTORIZATION+SOLVE
    from scratch every Newton iteration. reuse_analysis=True (opt-in,
    2026-09-11): uses _newton_cudss_reuse_analysis instead -- a custom
    Newton loop that computes cuDSS's own ANALYSIS phase ONCE (it depends
    only on the Jacobian's sparsity pattern, which never changes within
    one Newton solve) and reuses it via an in-place value-buffer update
    every subsequent iteration, redoing only FACTORIZATION+SOLVE. Found
    by reading torch_sla's own source directly, then verified for real on
    an A100 (profile_cudss_analysis_reuse.py): ANALYSIS was 95.7% of one
    full solve's own time at N=401, and reuse gave the identical answer
    (~1e-14 relative difference) to redoing it. CUDA-only (cuDSS has no
    CPU path) -- reuse_analysis=True raises if device is not CUDA.

    return_stats=True additionally returns the Newton-loop stats dict
    from _newton_cudss_reuse_analysis (n_newton_iters, per-phase time
    totals) when reuse_analysis=True; ignored (returns None as the second
    element) when reuse_analysis=False, since torch_sla's own
    nonlinear_solve does not expose that breakdown."""
    from omar_pfem.matrix_free_solver import element_energy_order_agnostic, precompute_shape_data
    from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch

    device = device or torch.device("cpu")
    if linear_solver is None:
        linear_solver = "cudss" if device.type == "cuda" else "auto"
    torch.set_default_dtype(dtype)

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

    def energy_fn(u_flat):
        uv = u_flat.reshape(n_nodes, 2)
        return element_energy_order_agnostic(xy, quad, uv, elem_params, energy_density_fn,
                                              shape_data, dtype)

    def residual(u_flat, _A, f_ext):
        grad_full = torch.func.grad(energy_fn)(u_flat)
        res = grad_full - f_ext
        return torch.where(free_mask_dof, res, u_flat)

    jac_fn = build_sparse_jac_fn(nodes, elements, mu, lam, free_mask_dof, material, order, device,
                                  dtype)

    u0 = torch.zeros(n_dof, dtype=dtype, device=device)
    stats = None
    if reuse_analysis:
        if device.type != "cuda":
            raise RuntimeError("reuse_analysis=True requires CUDA (cuDSS has no CPU path).")
        u, stats = _newton_cudss_reuse_analysis(residual, jac_fn, u0, f_ext_flat, tol=tol,
                                                 atol=1e-12, max_iter=max_iter, line_search=True,
                                                 verbose=False)
    else:
        # Dummy SparseTensor A: nonlinear_solve passes it automatically as
        # the second positional arg to residual/jac_fn, but neither uses
        # it -- the real tangent comes from jac_fn's own closure over
        # "our own" energy. An identity matrix of the right shape is
        # enough; built directly via torch_sla.SparseTensor's own COO
        # constructor, no TensorMesh Mesh or ElementAssembler needed.
        ident = torch.arange(n_dof, device=device)
        A = SparseTensor(torch.ones(n_dof, dtype=dtype, device=device), ident, ident,
                          (n_dof, n_dof))
        u = A.nonlinear_solve(residual, u0, f_ext_flat, jac_fn=jac_fn, method="newton",
                               verbose=False, max_iter=max_iter, tol=tol, linear_method="lu",
                               linear_solver=linear_solver)

    u_np = u.reshape(n_nodes, 2).detach().cpu().numpy()
    if return_stats:
        return u_np, stats
    return u_np


def _correctness_check(N=11):
    """Compares against solve_matrix_free's OWN result (not TensorMesh's),
    on CPU, at a small N -- the same discipline used for every other
    change in this project: never trust a speed claim before correctness
    is verified against an already-established reference. solve_matrix_free
    returns free-DOF-only displacements in a different (index_copy)
    convention than this module's own full-DOF-with-masking one, so both
    are reconstructed to the same full (n_nodes, 2) field before
    comparing."""
    print(f"=== correctness check: solve_assembled_direct vs. solve_matrix_free, "
          f"B1 x Neo-Hookean, N={N} ===")
    from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs
    from omar_pfem.matrix_free_solver import solve_matrix_free

    device = torch.device("cpu")
    dtype = torch.float64
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        "B1", "Q4", N, "neo_hookean", device, dtype)
    mu, lam = elem_params
    n_nodes = nodes.shape[0]

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_dofs_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
    elem_params_t = (torch.tensor(mu, dtype=dtype, device=device),
                     torch.tensor(lam, dtype=dtype, device=device))
    fext_free_t = torch.tensor(fext_full[free_dofs], dtype=dtype, device=device)

    import time
    t0 = time.time()
    u_free_mf, stats_mf = solve_matrix_free(
        xy_t, quad_t, free_dofs_t, elem_params_t, fext_free_t, len(free_dofs),
        material="neo_hookean", order="Q4", nsteps=10, device=device, dtype=dtype, verbose=False)
    t_mf = time.time() - t0
    u_full_mf = torch.zeros(2 * n_nodes, dtype=dtype)
    u_full_mf[free_dofs_t] = u_free_mf
    u_full_mf = u_full_mf.reshape(n_nodes, 2).numpy()
    print(f"  matrix_free:       wall_clock={t_mf:.2f}s, "
          f"cg_iters_total={stats_mf['cg_iters_total']}")

    t0 = time.time()
    u_assembled = solve_assembled_direct(nodes, elements, free_dofs, fext_full, mu, lam,
                                          dtype=dtype, material="neo_hookean", order="Q4",
                                          device=device)
    t_ad = time.time() - t0
    print(f"  assembled_direct:  wall_clock={t_ad:.2f}s")

    diff = np.linalg.norm(u_full_mf.reshape(-1) - u_assembled.reshape(-1))
    ref = np.linalg.norm(u_full_mf) + 1e-30
    rel_diff = diff / ref
    print(f"  relative displacement-field difference: {rel_diff:.3e}")
    ok = rel_diff < 1e-6
    print("  " + ("PASS" if ok else "FAIL"))
    return ok, rel_diff


def _correctness_check_reuse_analysis(N=11, device=None):
    """GPU-only (cuDSS has no CPU path): compares reuse_analysis=True
    against reuse_analysis=False (the already CPU-verified default path)
    at a small N, and reports the real end-to-end speedup and Newton
    iteration count -- not just the isolated 3-call number
    profile_cudss_analysis_reuse.py already measured. Must be run on a
    real GPU; this module's own CPU correctness check cannot cover this
    path at all."""
    print(f"=== correctness check: reuse_analysis=True vs. reuse_analysis=False, "
          f"B1 x Neo-Hookean, N={N} ===")
    from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs

    device = device or torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError("This check needs CUDA -- cuDSS has no CPU path.")
    dtype = torch.float64
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        "B1", "Q4", N, "neo_hookean", device, dtype)
    mu, lam = elem_params

    import time
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    u_baseline = solve_assembled_direct(nodes, elements, free_dofs, fext_full, mu, lam,
                                         dtype=dtype, material="neo_hookean", order="Q4",
                                         device=device, reuse_analysis=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_baseline = time.time() - t0
    print(f"  reuse_analysis=False: wall_clock={t_baseline:.3f}s")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    u_reuse, stats = solve_assembled_direct(nodes, elements, free_dofs, fext_full, mu, lam,
                                             dtype=dtype, material="neo_hookean", order="Q4",
                                             device=device, reuse_analysis=True,
                                             return_stats=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_reuse = time.time() - t0
    print(f"  reuse_analysis=True:  wall_clock={t_reuse:.3f}s, "
          f"n_newton_iters={stats['n_newton_iters']}, "
          f"t_analysis={stats['t_analysis_s']:.4f}s, "
          f"t_factorization={stats['t_factorization_s']:.4f}s, "
          f"t_solve={stats['t_solve_s']:.4f}s")

    diff = np.linalg.norm(u_baseline.reshape(-1) - u_reuse.reshape(-1))
    ref = np.linalg.norm(u_baseline) + 1e-30
    rel_diff = diff / ref
    print(f"  relative displacement-field difference: {rel_diff:.3e}")
    ok = rel_diff < 1e-6
    print("  " + ("PASS" if ok else "FAIL -- DO NOT trust the speedup below if this is FAIL"))
    if ok:
        print(f"  speedup (reuse_analysis=True vs False): {t_baseline / t_reuse:.2f}x")
    return ok, rel_diff


def run_assembled_direct_convergence_study(resolutions, out_json, geometry="B1",
                                            material="neo_hookean", order="Q4", fine_N=2236,
                                            checkpoint_dir=None, device=None, tol=1e-8,
                                            dtype=torch.float64, reuse_analysis=False):
    """Mirrors tensormesh_comparison.py's own run_tensormesh_convergence_study
    exactly (same fine reference, same accuracy metrics, same resumable-JSON
    pattern), so its output is directly comparable row-for-row against the
    already-committed torch-fem and TensorMesh production sweeps. Adds a
    peak-GPU-memory column per row (torch.cuda.reset_peak_memory_stats /
    max_memory_allocated), the number this whole experiment exists to
    produce: whether "ours", assembled the same way as the other two, uses
    comparably little/much memory and comparable/better wall-clock time at
    the SAME resolutions those two already solved successfully.

    reuse_analysis: passed straight through to solve_assembled_direct
    (default False, the already-published behavior). True uses the
    2026-09-11 cuDSS analysis-reuse Newton loop instead -- see that
    function's own docstring; verify with _correctness_check_reuse_
    analysis at a small N BEFORE trusting a production run with this set,
    since it changes the linear-solve code path entirely (bypasses
    torch_sla's own nonlinear_solve)."""
    import json
    import os

    from omar_pfem.high_dof_convergence_study import (
        build_mesh_and_bcs, compute_l2_h1_errors, fit_convergence_rate, solve_one)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('device:', device)

    fine_ckpt = (os.path.join(checkpoint_dir, f"fine_{geometry}_{material}_{order}_N{fine_N}.pt")
                 if checkpoint_dir else None)
    print(f'Loading/resuming fine reference N={fine_N} (checkpoint={fine_ckpt})...')
    fine = solve_one(geometry, order, fine_N, material, device, torch.float64,
                      cg_tol=1e-8, newton_tol=1e-8, checkpoint_path=fine_ckpt)
    print(f'  fine reference ready: n_dof={fine["n_dof"]}, wall_clock_s={fine["wall_clock_s"]:.1f}')

    done = {}
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            done = {r["N"]: r for r in json.load(f).get("rows", [])}
    rows = list(done.values())

    for N in resolutions:
        if N in done:
            print(f'  N={N} already in {out_json}, skipping')
            continue
        nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
            geometry, order, N, material, device, dtype)
        mu, lam = elem_params

        import time
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        u_ad = solve_assembled_direct(nodes, elements, free_dofs, fext_full, mu, lam, dtype=dtype,
                                       tol=tol, material=material, order=order, device=device,
                                       reuse_analysis=reuse_analysis)
        elapsed = time.time() - t0
        peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6
                       if device.type == "cuda" else None)

        coarse = {"nodes": nodes, "elements": elements, "N": N,
                  "u": u_ad.reshape(len(nodes), 2)}
        errs = compute_l2_h1_errors(coarse, fine, order, geometry)

        row = {"N": N, "n_dof": int(2 * nodes.shape[0]), "tol": tol,
               "assembled_direct_wall_clock_s": elapsed,
               "assembled_direct_peak_mem_mb": peak_mem_mb,
               "l2_rel": errs["l2_rel"], "h1_semi_rel": errs["h1_semi_rel"]}
        mem_s = f'{peak_mem_mb:.1f}MB' if peak_mem_mb is not None else '(n/a, not CUDA)'
        print(f'  N={N}: wall_clock={elapsed:.2f}s peak_mem={mem_s} '
              f'l2_rel={errs["l2_rel"]:.3e} h1_semi_rel={errs["h1_semi_rel"]:.3e}')
        rows.append(row)
        rows.sort(key=lambda r: r["N"])
        if out_json:
            with open(out_json, "w") as f:
                json.dump({"geometry": geometry, "material": material, "order": order,
                           "fine_N": fine_N, "device": str(device), "tol": tol,
                           "rows": rows}, f, indent=2)

    sub_fine_rows = [r for r in rows if r["N"] < fine_N]
    hs = [1.0 / (r["N"] - 1) for r in sub_fine_rows]
    if len(hs) >= 2:
        rate_l2, _ = fit_convergence_rate(hs, [r["l2_rel"] for r in sub_fine_rows])
        rate_h1, _ = fit_convergence_rate(hs, [r["h1_semi_rel"] for r in sub_fine_rows])
        print(f'\nFitted convergence rate (assembled_direct, {order}): L2 p={rate_l2:.3f} '
              f'(expected 2), H1 p={rate_h1:.3f} (expected 1)')
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "convergence":
        # python -m omar_pfem.assembled_direct_solver convergence <Ns> <out_json> <ckpt_dir> <fine_N> [reuse]
        Ns = [int(n) for n in sys.argv[2].split(",")]
        out_json = sys.argv[3]
        ckpt_dir = sys.argv[4] if len(sys.argv) > 4 else None
        fine_N = int(sys.argv[5]) if len(sys.argv) > 5 else 2236
        reuse = len(sys.argv) > 6 and sys.argv[6] == "reuse"
        run_assembled_direct_convergence_study(Ns, out_json, checkpoint_dir=ckpt_dir, fine_N=fine_N,
                                                reuse_analysis=reuse)
    elif len(sys.argv) > 1 and sys.argv[1] == "reuse_check":
        # python -m omar_pfem.assembled_direct_solver reuse_check <N>  (GPU only)
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 11
        ok, _rel_diff = _correctness_check_reuse_analysis(N)
        sys.exit(0 if ok else 1)
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
        ok, _rel_diff = _correctness_check(N)
        sys.exit(0 if ok else 1)
