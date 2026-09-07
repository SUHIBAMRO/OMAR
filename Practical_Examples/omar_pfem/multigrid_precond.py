"""Geometric multigrid V-cycle, used as a CG preconditioner for the
matrix-free Newton-CG solver (matrix_free_solver.py), built because the
2x2 block-Jacobi preconditioner tried first (item #4) measurably failed:
a real GPU re-run showed IDENTICAL cg_failures at N=401/701 (20/30) to
plain scalar Jacobi -- see PROJECT_STATUS.md's items #3/#4 and the
committed `highdof_stress_qoi_results/..._block2x2_stage1.json`.

WHY GEOMETRIC MULTIGRID, NOT INCOMPLETE CHOLESKY (the other option
discussed): IC0 needs the actual sparse tangent matrix to factor, which
conflicts with this whole solver's reason for existing -- it is
matrix-free specifically so it can reach millions of DOF without ever
forming K (see matrix_free_solver.py's own docstring). Geometric
multigrid needs no matrix either: only the ability to apply K's action
(already exists, matrix_free_hvp) on a hierarchy of coarser meshes of
the SAME domain. Both B1 (unit square) and B2 (quarter ring) meshes in
this codebase are built as structured (i, j) grids with i FAST, j SLOW
node ordering (generate_grid_Q4 for B1; generate_grid_Q4_ring / the
polar-mapped Q9 grid for B2 -- confirmed directly in
high_dof_convergence_study.py's build_mesh_and_bcs and its own comments)
-- exactly the structure classical geometric multigrid is built for, and
what makes a single geometry-agnostic implementation here possible: this
module only ever operates on abstract (i, j) index-space grids, never on
physical (x, y) coordinates, so the same prolongation/restriction code
serves both geometries.

WHAT THIS MODULE DOES NOT DO: it does not decide the mesh hierarchy or
sample the material at each level -- that stays the caller's job
(high_dof_convergence_study.py's build_mesh_and_bcs, called once per
level at solve setup, reusing that already-validated mesh/BC/material
code exactly, not reimplementing it), so this file can import
matrix_free_solver.py's primitives without creating a circular import
(matrix_free_solver.py itself stays geometry-agnostic and does not
import this file's internals beyond the optional precond_kind="mgv"
dispatch it adds).

Design (standard symmetric V-cycle, so the preconditioner stays
compatible with CG's SPD assumption):
  pre-smooth (damped Jacobi) -> restrict residual -> recurse on the
  coarser level -> prolong the correction -> post-smooth.
The coarsest level is solved EXACTLY via a dense LU factorization built
once per Newton iteration (its own free-DOF count is small by
construction, a few hundred at most) and reused for every V-cycle call
within that Newton iteration's CG solve -- an earlier version re-ran a
full CG solve from scratch at the coarsest level on every V-cycle call
instead, which is correct but far too slow (a V-cycle is invoked once
per FINE-level CG iteration, so redoing an iterative coarse solve that
often dominated the total cost; caught by a tiny smoke test hanging,
fixed before any real-scale timing was trusted).

VALIDATION STATUS, stated explicitly per this project's own discipline
of not trusting a plausible-looking number: this file has been checked
by build_mg_hierarchy_meshes/build_mg_precond_apply's own unit checks
in validate_multigrid_precond.py (prolongation is a partition of unity,
restriction is its exact 1/4-scaled transpose, and the full V-cycle
matches solve_matrix_free's plain-CG solution on tiny B1 and B2 meshes
to CG tolerance) BEFORE it was ever pointed at the real N=401/701
problem this module exists to fix -- see that script for the actual
numbers, and PROJECT_STATUS.md for whether it was found to help.
"""
import numpy as np
import scipy.sparse as sp
import torch

from omar_pfem.matrix_free_solver import (
    _make_energy_fn, matrix_free_hvp, compute_jacobi_diagonal)
from torch.func import grad, vmap


def coarsen_N(N):
    """One step of structured 2x coarsening: Nf = 2*(Nc-1)+1 inverted.
    Returns None if N-1 is odd (cannot coarsen further while keeping the
    domain's corner nodes exactly, the property that makes every coarser
    level's boundary a literal subset of the finer level's)."""
    if (N - 1) % 2 != 0:
        return None
    return (N - 1) // 2 + 1


def build_prolongation_matrix(Nc, Nf):
    """Bilinear-interpolation prolongation P, shape (Nf*Nf, Nc*Nc), from a
    structured Nc x Nc index-space grid to Nf x Nf where Nf = 2*(Nc-1)+1,
    assuming row-major node ordering index = j*N + i (i fast, j slow) at
    BOTH levels -- the convention every mesh generator in this codebase
    already shares (see this module's own docstring). P acts on ONE scalar
    field at a time; the SAME P is used for both displacement components
    (x, y) unchanged, since interpolation only depends on index-space
    position, never on which physical component is being moved.

    Every row sums to exactly 1 (a fine node's value is always a convex
    combination of its 1, 2, or 4 surrounding coarse nodes) -- checked
    directly in validate_multigrid_precond.py, since a row that doesn't
    sum to 1 would mean a constant field fails to prolong to itself
    exactly, a correctness bug a convergence-rate check alone would not
    obviously reveal."""
    assert Nf == 2 * (Nc - 1) + 1, f"Nf={Nf} is not 2*(Nc-1)+1 for Nc={Nc}"
    rows, cols, vals = [], [], []

    def cidx(I, J):
        return J * Nc + I

    for J in range(Nf):
        cj, fj = divmod(J, 2)
        for I in range(Nf):
            ci, fi = divmod(I, 2)
            fidx = J * Nf + I
            if fi == 0 and fj == 0:
                rows.append(fidx); cols.append(cidx(ci, cj)); vals.append(1.0)
            elif fi == 1 and fj == 0:
                rows += [fidx, fidx]
                cols += [cidx(ci, cj), cidx(ci + 1, cj)]
                vals += [0.5, 0.5]
            elif fi == 0 and fj == 1:
                rows += [fidx, fidx]
                cols += [cidx(ci, cj), cidx(ci, cj + 1)]
                vals += [0.5, 0.5]
            else:
                rows += [fidx] * 4
                cols += [cidx(ci, cj), cidx(ci + 1, cj), cidx(ci, cj + 1), cidx(ci + 1, cj + 1)]
                vals += [0.25] * 4
    P = sp.csr_matrix((vals, (rows, cols)), shape=(Nf * Nf, Nc * Nc))
    return P


def injection_indices(Nc, Nf):
    """For each coarse node (I,J), the index of the COINCIDING fine node
    (2I,2J) -- used to restrict the CURRENT nonlinear displacement field
    (the point the tangent operator is linearized at), which must be exact
    physical injection, not the averaged P^T restriction used for
    residuals/RHS vectors below (those are different objects: one is a
    primal field sampled at a point, the other is a dual/residual quantity
    that should be redistributed, not merely subsampled)."""
    assert Nf == 2 * (Nc - 1) + 1
    J, I = np.meshgrid(np.arange(Nc), np.arange(Nc), indexing="ij")
    return (2 * J * Nf + 2 * I).reshape(-1)


class MGLevel:
    """One level of the hierarchy. `xy_t`/`quad_t`/`free_dofs_t`/
    `elem_params_t` come directly from build_mesh_and_bcs at this level's
    own N, exactly as solve_matrix_free's caller already builds them for
    the finest level -- this class does not re-derive the mesh, BCs, or
    material sampling, only stores what's already been built and adds the
    interpolation operator to the next coarser level."""

    def __init__(self, N, xy_t, quad_t, free_dofs_t, elem_params_t, dtype, device):
        self.N = N
        self.xy_t = xy_t
        self.quad_t = quad_t
        self.free_dofs_t = free_dofs_t
        self.elem_params_t = elem_params_t
        self.n_nodes = xy_t.shape[0]
        self.n_free = free_dofs_t.shape[0]
        self.dtype = dtype
        self.device = device
        # Filled in by build_mg_hierarchy once the next-coarser level exists.
        self.P_to_finer = None       # torch sparse (n_nodes_finer, n_nodes) -- unused at finest level
        self.R_from_finer = None     # torch sparse (n_nodes, n_nodes_finer) -- unused at finest level
        self.inject_from_finer = None  # LongTensor (n_nodes,) index into the finer level's nodes


def _scipy_to_torch_sparse(A, dtype, device):
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=A.shape, dtype=dtype, device=device).coalesce()


def build_mg_hierarchy(mesh_tuples, dtype, device):
    """mesh_tuples: list of (N, nodes_np, elements_np, free_dofs_np,
    elem_params_np) from finest to coarsest, each produced by the
    caller's OWN build_mesh_and_bcs at successively coarser N (this
    function does not decide the N sequence or build meshes itself --
    see this module's docstring for why). Returns a list of MGLevel,
    finest first, with P/R/injection operators linking each pair of
    adjacent levels."""
    levels = []
    for N, nodes, elements, free_dofs, elem_params in mesh_tuples:
        xy_t = torch.tensor(nodes, dtype=dtype, device=device)
        quad_t = torch.tensor(elements, dtype=torch.long, device=device)
        free_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
        params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in elem_params)
        levels.append(MGLevel(N, xy_t, quad_t, free_t, params_t, dtype, device))

    for k in range(len(levels) - 1):
        Nf, Nc = levels[k].N, levels[k + 1].N
        P = build_prolongation_matrix(Nc, Nf)
        R = (P.T * 0.25).tocsr()   # standard 2D variational scaling, R = P^T / 4
        # NOTE: do not touch levels[k].R_from_finer here -- it defaults to
        # None in MGLevel.__init__ and, for k > 0, was already correctly
        # set to a real sparse tensor by the PREVIOUS iteration (when this
        # same level was levels[k+1]). A line here that reset it to None
        # unconditionally was a real bug: harmless with exactly 2 levels
        # (k only ever 0, and level 0 never needs its own R_from_finer),
        # but silently wiped level 1's operator the moment a 3rd level
        # existed -- caught only because a 3-level N=65 hierarchy test was
        # run, not by the 2-level N=9 smoke tests, which is exactly why
        # both were worth running rather than stopping at the first PASS.
        levels[k + 1].P_to_finer = _scipy_to_torch_sparse(P, dtype, device)
        levels[k + 1].R_from_finer = _scipy_to_torch_sparse(R, dtype, device)
        levels[k + 1].inject_from_finer = torch.tensor(
            injection_indices(Nc, Nf), dtype=torch.long, device=device)
    return levels


def _apply_scalar_op(op, x_scalar):
    """op: torch sparse (out, in). x_scalar: (in,) dense. -> (out,) dense."""
    return torch.sparse.mm(op, x_scalar.unsqueeze(-1)).squeeze(-1)


def _restrict_full_vector(v_full, level_coarser):
    """v_full: (n_nodes_fine*2,) full-numbering vector on the FINER level.
    Returns the coarser level's own full-numbering vector, by applying
    R = P^T/4 to the x- and y-components independently."""
    n_fine = level_coarser.R_from_finer.shape[1]
    v2 = v_full.reshape(n_fine, 2)
    rx = _apply_scalar_op(level_coarser.R_from_finer, v2[:, 0])
    ry = _apply_scalar_op(level_coarser.R_from_finer, v2[:, 1])
    return torch.stack([rx, ry], dim=1).reshape(-1)


def _prolong_full_vector(v_full_coarse, level_coarser):
    """v_full_coarse: (n_nodes_coarse*2,) on the COARSER level. Returns the
    finer level's full-numbering vector via P."""
    n_coarse = level_coarser.P_to_finer.shape[1]
    v2 = v_full_coarse.reshape(n_coarse, 2)
    px = _apply_scalar_op(level_coarser.P_to_finer, v2[:, 0])
    py = _apply_scalar_op(level_coarser.P_to_finer, v2[:, 1])
    return torch.stack([px, py], dim=1).reshape(-1)


def _embed_free(v_free, free_dofs, ndof, dtype, device):
    out = torch.zeros(ndof, dtype=dtype, device=device)
    out[free_dofs] = v_free
    return out


def _inject_displacement(u_full_fine, level_coarser):
    """Restricts the CURRENT nonlinear displacement field (used to build
    the coarser level's own tangent operator at a matching linearization
    point) by direct injection -- physical subsampling at coinciding
    nodes, not the averaged R used for residual vectors above. See
    injection_indices' docstring for why these must be different
    operators."""
    idx = level_coarser.inject_from_finer
    u2 = u_full_fine.reshape(-1, 2)
    return u2[idx].reshape(-1)


def _level_matvec_and_diag(level, u_full_level, material, order):
    """Builds this level's own tangent-operator matvec (matrix_free_hvp,
    exactly the same construction solve_matrix_free uses for the finest
    level) and Jacobi diagonal (for the smoother), both linearized at
    u_full_level -- the displacement injected down to this level."""
    n_nodes = level.n_nodes
    energy_fn = _make_energy_fn(level.xy_t, level.quad_t, n_nodes, material, order, level.dtype)
    residual_fn = grad(energy_fn, argnums=0)

    def matvec(v_free):
        u_free = u_full_level[level.free_dofs_t]
        return matrix_free_hvp(residual_fn, u_free, v_free, level.elem_params_t, level.free_dofs_t)

    diag = compute_jacobi_diagonal(level.xy_t, level.quad_t, u_full_level, level.elem_params_t,
                                    material, order, level.dtype, level.free_dofs_t)
    return matvec, diag


def _build_dense_factor(matvec, n_free, dtype, device, chunk_size=500):
    """Materializes the (coarsest-level-only) tangent operator as a dense
    matrix by applying matvec to batches of unit basis vectors via vmap,
    then LU-factors it ONCE. This is the fix for an earlier, much slower
    design that re-ran conjugate_gradient from scratch at the coarsest
    level on EVERY V-cycle call -- and a V-cycle is called once per
    FINE-level CG ITERATION, so an unconverged or slowly-converging coarse
    CG inside it made the whole preconditioner far more expensive than
    plain Jacobi instead of cheaper (caught directly: a tiny N=9/coarsest-
    N=5 smoke test hung for over a minute before that fix). The coarsest
    level's operator does not change during a single Newton iteration's CG
    solve (only the smoother/matvec's LINEARIZATION point does, which is
    fixed for the whole Newton iteration), so factoring it once and
    reusing the factorization for every V-cycle call in that CG solve is
    both correct and the obviously cheaper design.

    Second fix, on top of the first: a SERIAL Python loop calling matvec
    (jvp of a reverse-mode grad) once per basis vector was still the real
    bottleneck on GPU once mg_max_levels stopped coarsening before the
    coarsest level's n_free got small (e.g. ~5000 free DOFs at N=51) --
    real GPU timing showed each CG solve costing ~82x more than the old
    non-converging plain-Jacobi run, even though CG now genuinely
    converged. Root cause: thousands of tiny sequential autodiff calls
    each pay GPU kernel-launch/dispatch overhead that dominates their
    actual (cheap) FLOP cost. Batching matvec over chunks of basis vectors
    with vmap (the same chunk-and-accumulate pattern compute_jacobi_diagonal
    already uses for the same reason) turns those thousands of serial
    dispatches into a handful of batched ones -- same exact matrix, just
    built without paying per-column overhead thousands of times over.
    chunk_size bounds peak memory the same way it does there; it changes
    nothing mathematically. CONFIRMED on real GPU data 2026-09-07: after
    this fix plus raising mg_max_levels, a full N=401 re-run hit
    cg_failures=0 (was 20) at wall_clock_s=2615.8 (~43.6 min, vs. an ~87
    min projection before this fix and ~27 min for the old, non-converging
    run) -- see PROJECT_STATUS.md's item #4 for the full comparison."""
    basis = torch.eye(n_free, dtype=dtype, device=device)
    batched_matvec = vmap(matvec)
    rows = []
    for start in range(0, n_free, chunk_size):
        end = min(start + chunk_size, n_free)
        rows.append(batched_matvec(basis[start:end]))
    # rows[i] holds matvec(e_j) as its i-th ROW for each basis vector e_j in
    # that chunk -- i.e. row j = K @ e_j = column j of K, so concatenating
    # rows gives K^T, not K.
    K = torch.cat(rows, dim=0).T
    K = 0.5 * (K + K.T)  # matvec is symmetric in exact arithmetic; symmetrize away FP noise
    return torch.linalg.lu_factor(K)


def build_mg_precond_apply(levels, u_full_fine, material, order, nu1=2, nu2=2, omega=0.6):
    """Builds the v -> M^-1 v closure for one Newton iteration's CG solve,
    a standard symmetric V-cycle: damped-Jacobi pre-smooth, restrict the
    residual, recurse on the coarser level, prolong the correction back,
    damped-Jacobi post-smooth. The coarsest level is solved EXACTLY via a
    dense LU factorization built once per Newton iteration (see
    _build_dense_factor's docstring for why a from-scratch CG re-solve at
    the coarsest level, tried first, was a real performance bug, not a
    correctness one) -- its own free-DOF count is small by construction (a
    few hundred at most, since coarsening stops well before that in
    high_dof_convergence_study.py's own hierarchy choice), so this is
    cheap relative to the fine-level CG this whole thing preconditions.

    u_full_fine: the CURRENT full-numbering displacement at the finest
    level (dtype/device matching `levels[0]`) -- rebuilt every Newton
    iteration by the caller, exactly like compute_jacobi_diagonal/
    compute_block_jacobi already are, since the tangent operator changes
    with the nonlinear state."""
    dtype, device = levels[0].dtype, levels[0].device
    n_levels = len(levels)

    u_full_per_level = [u_full_fine]
    for k in range(1, n_levels):
        u_full_per_level.append(_inject_displacement(u_full_per_level[k - 1], levels[k]))

    matvecs, diags = [], []
    for k in range(n_levels):
        mv, dg = _level_matvec_and_diag(levels[k], u_full_per_level[k], material, order)
        matvecs.append(mv)
        diags.append(dg)

    coarse_lu = _build_dense_factor(matvecs[-1], levels[-1].n_free, dtype, device)

    def damped_jacobi(level_idx, r_free, x0, iters):
        x = x0
        mv, dg = matvecs[level_idx], diags[level_idx]
        for _ in range(iters):
            res = r_free - mv(x)
            x = x + omega * (res / dg)
        return x

    def vcycle(level_idx, r_free):
        if level_idx == n_levels - 1:
            LU, piv = coarse_lu
            x = torch.linalg.lu_solve(LU, piv, r_free.unsqueeze(-1)).squeeze(-1)
            return x

        lvl = levels[level_idx]
        lvl_c = levels[level_idx + 1]
        x0 = torch.zeros_like(r_free)
        x = damped_jacobi(level_idx, r_free, x0, nu1)

        res_free = r_free - matvecs[level_idx](x)
        res_full = _embed_free(res_free, lvl.free_dofs_t, 2 * lvl.n_nodes, dtype, device)
        res_full_c = _restrict_full_vector(res_full, lvl_c)
        res_free_c = res_full_c[lvl_c.free_dofs_t]

        e_free_c = vcycle(level_idx + 1, res_free_c)

        e_full_c = _embed_free(e_free_c, lvl_c.free_dofs_t, 2 * lvl_c.n_nodes, dtype, device)
        e_full = _prolong_full_vector(e_full_c, lvl_c)
        e_free = e_full[lvl.free_dofs_t]
        x = x + e_free

        x = damped_jacobi(level_idx, r_free, x, nu2)
        return x

    def apply(v_free):
        return vcycle(0, v_free)

    return apply
