"""DIAGNOSTIC ONLY (2026-09-11) -- does NOT change solve_assembled_direct
or any other solver in this project. Answers one question before any
real engineering effort is spent: is cuDSS's own ANALYSIS phase (the
fill-reducing reordering step) a big enough fraction of one solve's cost
to be worth reusing across Newton iterations?

WHY THIS IS WORTH CHECKING AT ALL, confirmed by reading torch_sla's own
installed source directly (not assumed): `NonlinearSolveFunction.
forward` (torch_sla/sparse_tensor/autograd.py) calls `spsolve(...)`
fresh on EVERY Newton iteration, and torch_sla's own cuDSS backend
(`backends/nvmath_backend.py`'s `nvmath_solve`) does a brand-new
`cudss.create()` -> ANALYSIS -> FACTORIZATION -> SOLVE -> `cudss.
destroy()` every single call, with zero reuse across calls. ANALYSIS
(the reordering/elimination-tree computation) depends ONLY on the
matrix's own sparsity PATTERN (which (row, col) entries are nonzero),
NOT on the numeric values -- and `build_sparse_jac_fn`'s own row/col
template is static across every Newton iteration of one solve (only the
VALUES change each call, per its own docstring). So the current code
re-pays the ANALYSIS cost on every single Newton iteration even though
the pattern never changes within one solve -- a real, source-confirmed
inefficiency, not a guess. What's NOT yet known: how big a fraction of
one solve's total time ANALYSIS actually is (could be small), and
whether cuDSS's own API even supports "reuse ANALYSIS, refactorize with
new values on the SAME matrix handle" the way this script assumes.

THIS SCRIPT CHECKS BOTH, on a real Jacobian pattern from this project's
own build_sparse_jac_fn, with a correctness check (not just a timing
number) before trusting anything: Method B (reuse) must give the SAME
solution as Method A (the current, always-redo-everything behavior) or
this whole idea is invalid, not just slower than hoped.

CUDA-only (cuDSS has no CPU path) -- cannot be run or verified in this
development environment; must be run on Omar's own GPU."""
import sys
import time

import numpy as np
import torch


def build_three_jacobians(N=401, geometry="B1", material="neo_hookean", order="Q4",
                           device=None, dtype=torch.float64):
    """Three real Jacobians with the IDENTICAL sparsity pattern (same
    mesh/connectivity -- build_sparse_jac_fn's own row/col template never
    changes) but DIFFERENT numeric values (three different displacement
    fields) -- exactly the situation across consecutive Newton
    iterations, without needing a full nonlinear solve first (this
    script only cares about the LINEAR-solve phase's own cost)."""
    from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs
    from omar_pfem.tensormesh_comparison import build_sparse_jac_fn

    device = device or torch.device("cuda")
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        geometry, order, N, material, device, dtype)
    mu, lam = elem_params
    n_dof = 2 * nodes.shape[0]
    fixed_set = set(np.setdiff1d(np.arange(n_dof), free_dofs).tolist())
    free_mask_dof = torch.tensor([i not in fixed_set for i in range(n_dof)], device=device)

    jac_fn = build_sparse_jac_fn(nodes, elements, mu, lam, free_mask_dof, material, order,
                                  device, dtype)

    torch.manual_seed(0)
    scale = 1e-4
    u0 = torch.zeros(n_dof, dtype=dtype, device=device)
    u1 = scale * torch.randn(n_dof, dtype=dtype, device=device)
    u2 = 2 * scale * torch.randn(n_dof, dtype=dtype, device=device)

    jacs, rhs = [], []
    for u in (u0, u1, u2):
        val, row, col, shape = jac_fn(u, None)
        jacs.append((val, row, col, shape))
        rhs.append(torch.randn(n_dof, dtype=dtype, device=device))
    return jacs, rhs


def _to_csr(val, row, col, shape):
    m, n = shape
    indices = torch.stack([row, col], dim=0)
    A_coo = torch.sparse_coo_tensor(indices, val, (m, n)).coalesce()
    A_csr = A_coo.to_sparse_csr()
    return A_csr.crow_indices().int(), A_csr.col_indices().int(), A_csr.values()


def solve_baseline_full_each_time(jacs, rhs):
    """Current torch_sla behavior: fresh handle + ANALYSIS + FACTORIZATION
    + SOLVE, every single call. Times each phase separately so ANALYSIS's
    own share of one full solve is visible even before reuse is
    considered."""
    import nvmath.bindings.cudss as cudss
    from torch_sla.backends.nvmath_backend import CUDA_R_32I, _DTYPE_MAP

    xs = []
    phase_times = {"analysis": 0.0, "factorization": 0.0, "solve": 0.0}
    total_t0 = time.time()
    for (val, row, col, shape), b in zip(jacs, rhs):
        m, n = shape
        crow, ccol, cval = _to_csr(val, row, col, shape)
        nnz = cval.numel()
        value_type = _DTYPE_MAP[cval.dtype]

        b_col = b.unsqueeze(0).contiguous()
        x_col = torch.zeros_like(b_col)

        handle = cudss.create()
        cudss.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        A_desc = cudss.matrix_create_csr(
            m, n, nnz, crow.data_ptr(), 0, ccol.data_ptr(), cval.data_ptr(),
            CUDA_R_32I, value_type, cudss.MatrixType.GENERAL.value,
            cudss.MatrixViewType.FULL.value, cudss.IndexBase.ZERO.value)
        b_desc = cudss.matrix_create_dn(m, 1, m, b_col.data_ptr(), value_type,
                                         cudss.Layout.COL_MAJOR.value)
        x_desc = cudss.matrix_create_dn(m, 1, m, x_col.data_ptr(), value_type,
                                         cudss.Layout.COL_MAJOR.value)
        config = cudss.config_create()
        data = cudss.data_create(handle)

        torch.cuda.synchronize(); t0 = time.time()
        cudss.execute(handle, cudss.Phase.ANALYSIS.value, config, data, A_desc, x_desc, b_desc)
        torch.cuda.synchronize(); t1 = time.time()
        cudss.execute(handle, cudss.Phase.FACTORIZATION.value, config, data, A_desc, x_desc, b_desc)
        torch.cuda.synchronize(); t2 = time.time()
        cudss.execute(handle, cudss.Phase.SOLVE.value, config, data, A_desc, x_desc, b_desc)
        torch.cuda.synchronize(); t3 = time.time()
        phase_times["analysis"] += t1 - t0
        phase_times["factorization"] += t2 - t1
        phase_times["solve"] += t3 - t2

        xs.append(x_col.squeeze(0).clone())

        cudss.data_destroy(handle, data)
        cudss.config_destroy(config)
        cudss.matrix_destroy(x_desc)
        cudss.matrix_destroy(b_desc)
        cudss.matrix_destroy(A_desc)
        cudss.destroy(handle)
    total_time = time.time() - total_t0
    return xs, phase_times, total_time


def solve_reuse_analysis(jacs, rhs):
    """HYPOTHESIS UNDER TEST, not assumed correct: build ONE A_desc from
    the first matrix's CSR buffers, run ANALYSIS once, then for every
    matrix (including the first) -- copy that matrix's own CSR VALUES
    into the SAME underlying value buffer A_desc already points to
    (crow/ccol never change, only cval's contents), and run only
    FACTORIZATION+SOLVE. This assumes (a) all three matrices really do
    share the identical (crow, ccol) structure -- asserted explicitly
    below, not just assumed -- and (b) cuDSS's own API honors updated
    values through the same descriptor without a fresh ANALYSIS. (b) is
    exactly what this diagnostic exists to find out; the correctness
    check against Method A's own output is the real arbiter, not this
    docstring."""
    import nvmath.bindings.cudss as cudss
    from torch_sla.backends.nvmath_backend import CUDA_R_32I, _DTYPE_MAP

    csrs = [_to_csr(*jac) for jac in jacs]
    crow0, ccol0, cval0 = csrs[0]
    for i, (crow_i, ccol_i, _) in enumerate(csrs[1:], start=1):
        if not (torch.equal(crow_i, crow0) and torch.equal(ccol_i, ccol0)):
            raise RuntimeError(
                f"matrix {i}'s CSR structure differs from matrix 0's -- the "
                f"'same pattern, different values' assumption this diagnostic "
                f"depends on does NOT hold; stop here, do not trust anything below.")

    m, n = jacs[0][3]
    nnz = cval0.numel()
    value_type = _DTYPE_MAP[cval0.dtype]
    cval_buf = cval0.clone()  # the ONE buffer A_desc will point at throughout

    xs = []
    phase_times = {"analysis": 0.0, "factorization": 0.0, "solve": 0.0}
    handle = cudss.create()
    cudss.set_stream(handle, torch.cuda.current_stream().cuda_stream)
    config = cudss.config_create()
    data = cudss.data_create(handle)
    A_desc = cudss.matrix_create_csr(
        m, n, nnz, crow0.data_ptr(), 0, ccol0.data_ptr(), cval_buf.data_ptr(),
        CUDA_R_32I, value_type, cudss.MatrixType.GENERAL.value,
        cudss.MatrixViewType.FULL.value, cudss.IndexBase.ZERO.value)

    total_t0 = time.time()
    for i, ((_, _, cval_i), b) in enumerate(zip(csrs, rhs)):
        cval_buf.copy_(cval_i)  # refresh values in place; A_desc's pointer is unchanged

        b_col = b.unsqueeze(0).contiguous()
        x_col = torch.zeros_like(b_col)
        b_desc = cudss.matrix_create_dn(m, 1, m, b_col.data_ptr(), value_type,
                                         cudss.Layout.COL_MAJOR.value)
        x_desc = cudss.matrix_create_dn(m, 1, m, x_col.data_ptr(), value_type,
                                         cudss.Layout.COL_MAJOR.value)

        torch.cuda.synchronize(); t0 = time.time()
        if i == 0:
            cudss.execute(handle, cudss.Phase.ANALYSIS.value, config, data, A_desc, x_desc, b_desc)
        torch.cuda.synchronize(); t1 = time.time()
        cudss.execute(handle, cudss.Phase.FACTORIZATION.value, config, data, A_desc, x_desc, b_desc)
        torch.cuda.synchronize(); t2 = time.time()
        cudss.execute(handle, cudss.Phase.SOLVE.value, config, data, A_desc, x_desc, b_desc)
        torch.cuda.synchronize(); t3 = time.time()
        if i == 0:
            phase_times["analysis"] += t1 - t0
        phase_times["factorization"] += t2 - t1
        phase_times["solve"] += t3 - t2

        xs.append(x_col.squeeze(0).clone())
        cudss.matrix_destroy(x_desc)
        cudss.matrix_destroy(b_desc)
    total_time = time.time() - total_t0

    cudss.matrix_destroy(A_desc)
    cudss.data_destroy(handle, data)
    cudss.config_destroy(config)
    cudss.destroy(handle)
    return xs, phase_times, total_time


def run(N=401):
    device = torch.device("cuda")
    print(f"Building 3 real Jacobians at N={N} (B1 x Neo-Hookean, Q4, same pattern, "
          f"different values -- simulating 3 Newton iterations)...")
    jacs, rhs = build_three_jacobians(N=N, device=device)
    n_dof = jacs[0][3][0]
    print(f"  n_dof={n_dof}")

    print("\n=== Method A: baseline (fresh handle+ANALYSIS+FACTORIZATION+SOLVE every call) ===")
    xs_a, phases_a, total_a = solve_baseline_full_each_time(jacs, rhs)
    for k, v in phases_a.items():
        print(f"  total {k} time (3 calls): {v:.4f}s  (avg/call: {v/3:.4f}s)")
    per_call_total = sum(phases_a.values()) / 3
    print(f"  ANALYSIS as a fraction of one full solve: "
          f"{(phases_a['analysis']/3) / per_call_total * 100:.1f}%")
    print(f"  TOTAL (3 full solves): {total_a:.4f}s")

    print("\n=== Method B: reuse ANALYSIS from the first matrix for all 3 ===")
    try:
        xs_b, phases_b, total_b = solve_reuse_analysis(jacs, rhs)
    except RuntimeError as e:
        print(f"  ABORTED: {e}")
        return
    for k, v in phases_b.items():
        print(f"  total {k} time: {v:.4f}s")
    print(f"  TOTAL (1 analysis + 3 factor+solve): {total_b:.4f}s")

    print("\n=== Correctness: does reusing ANALYSIS give the SAME answer as Method A? ===")
    max_rel_diff = 0.0
    for i, (xa, xb) in enumerate(zip(xs_a, xs_b)):
        diff = torch.linalg.vector_norm(xa - xb) / (torch.linalg.vector_norm(xa) + 1e-30)
        max_rel_diff = max(max_rel_diff, float(diff))
        print(f"  matrix {i}: relative difference = {float(diff):.3e}")
    ok = max_rel_diff < 1e-6

    print("\n=== Summary ===")
    print(f"  baseline total (3 full solves):        {total_a:.4f}s")
    print(f"  reuse-analysis total (1 analysis + 3):  {total_b:.4f}s")
    if ok:
        print(f"  CORRECTNESS: PASS -- reuse gives the same answer.")
        print(f"  speedup on this 3-call sample: {total_a / total_b:.2f}x")
        print(f"  (a full Newton solve has many more iterations than 3 -- this ratio should")
        print(f"  approach (per-call total) / (factorization+solve only) as iteration count grows,")
        print(f"  since ANALYSIS's one-time cost gets amortized over more calls.)")
    else:
        print(f"  CORRECTNESS: FAIL (max relative difference {max_rel_diff:.3e}) -- "
              f"DO NOT trust the speedup number, and do not build the full optimization on this "
              f"approach without first understanding why reuse produced a different answer.")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 401
    run(N)
