"""
Matrix-free Newton-Conjugate-Gradient Total-Lagrangian FEM solver, built
specifically to reach the advisor's requested ~10 million DOF mesh-
convergence reference. Every other solver in this codebase (the CPU
reference in data/data_generate_B1.py / B2.py, and the GPU-native batched
solver in gpu_fem_solver.py) explicitly FORMS the tangent stiffness matrix
-- a sparse matrix on CPU, a dense per-sample matrix on GPU -- and neither
approach scales to millions of DOF: a direct sparse factorization's fill-in
blows up memory in 2D well before 10M unknowns, and a dense matrix at that
size is simply impossible to store (10M^2 entries).

This solver instead NEVER forms the tangent matrix K. It only ever needs
Hessian-VECTOR products K @ v, computed via forward-over-reverse automatic
differentiation (torch.func.jvp composed with torch.func.grad) -- exactly
the same trick used everywhere else in this codebase to avoid hand-deriving
tangents (see gpu_fem_solver.py's own docstring), taken one step further:
here we differentiate through the RESIDUAL's action on a direction, rather
than materializing the full second derivative. Memory cost is O(DOF) (a
handful of vectors), not O(DOF^2) or O(nnz fill-in), which is what makes
pushing to millions of DOF possible on a single GPU at all.

The resulting linear system at each Newton step is then solved with
(unpreconditioned) Conjugate Gradient using only these Hv products --
appropriate because the Hessian of a scalar energy functional is symmetric
by construction, and positive definite in a neighborhood of a stable
equilibrium for a physically stable hyperelastic material. No
preconditioner is implemented in this first version (a Jacobi/diagonal
preconditioner would need each node's own diagonal stiffness contribution,
which is cheap to get from LOCAL per-element Hessians but is left as
future work); if CG fails to converge within --cg_max_iter, this is
reported explicitly rather than silently returning a wrong answer.

Supports both Q4 (bilinear) and Q9 (biquadratic, data/q9_element.py)
elements through one shared, order-agnostic energy assembly, so the same
solver produces the polynomial-order comparison the advisor asked for.

Correctness: validated against the existing CPU reference solver at small
N (see validate_matrix_free_solver.py) before this is trusted for anything
larger -- exactly the same discipline gpu_fem_solver.py's own
validate_gpu_fem_solver.py already established for the batched GPU solver.
"""
import os
import time
import numpy as np
import torch
from torch.func import grad, jvp, hessian, vmap

from omar_pfem.data.fem_core import shape_Q4
from omar_pfem.data.q9_element import shape_Q9, gauss_3x3
from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch


def _gauss_2x2():
    g = 1.0 / np.sqrt(3.0)
    return [(-g, -g, 1.0), (g, -g, 1.0), (g, g, 1.0), (-g, g, 1.0)]


def _shape_fn_and_quadrature(order):
    if order == "Q4":
        return shape_Q4, _gauss_2x2()
    elif order == "Q9":
        return shape_Q9, gauss_3x3()
    raise ValueError(order)


def precompute_shape_data(order, device, dtype):
    """Precomputes (N, dN_dxi) at every Gauss point once, as constant
    tensors reused for every element/sample -- these depend only on the
    reference element, never on nodal coordinates or material."""
    shape_fn, gauss = _shape_fn_and_quadrature(order)
    Ns, dNs, ws = [], [], []
    for (xi, eta, w) in gauss:
        N, dN_dxi = shape_fn(xi, eta)
        Ns.append(torch.tensor(N, dtype=dtype, device=device))
        dNs.append(torch.tensor(dN_dxi, dtype=dtype, device=device))
        ws.append(w)
    return Ns, dNs, ws


def element_energy_order_agnostic(xy, quad, uv, elem_params, energy_density_fn, shape_data, dtype):
    """Total internal energy U(u) = sum over elements and Gauss points of
    psi(F) dV0, for EITHER Q4 or Q9 (order determined entirely by quad's
    number of columns and shape_data's length -- 4 Gauss points for Q4, 9
    for Q9). Vectorized over all elements at once via the quad index
    tensor, exactly mirroring gpu_fem_solver._element_energy_Q4's
    approach, generalized from a hardcoded 4 local nodes to however many
    quad.shape[1] actually is."""
    Ns, dNs, ws = shape_data
    Xe = xy[quad]   # (Q, n_local, 2)
    ue = uv[quad]   # (Q, n_local, 2)

    U = torch.zeros((), device=xy.device, dtype=dtype)
    for N, dN_dxi, w in zip(Ns, dNs, ws):
        J0 = torch.einsum("qai,aj->qij", Xe, dN_dxi)
        detJ0 = J0[:, 0, 0] * J0[:, 1, 1] - J0[:, 0, 1] * J0[:, 1, 0]
        detJ0 = torch.clamp(detJ0, min=1e-12)

        invJ0 = torch.zeros_like(J0)
        invJ0[:, 0, 0] = J0[:, 1, 1] / detJ0
        invJ0[:, 1, 1] = J0[:, 0, 0] / detJ0
        invJ0[:, 0, 1] = -J0[:, 0, 1] / detJ0
        invJ0[:, 1, 0] = -J0[:, 1, 0] / detJ0

        dN_dX = torch.einsum("aj,qjk->qak", dN_dxi, invJ0)
        grad_u = torch.einsum("qai,qaj->qij", ue, dN_dX)

        I = torch.eye(2, device=xy.device, dtype=dtype).reshape(1, 2, 2).expand(Xe.shape[0], 2, 2)
        F = I + grad_u

        psi = energy_density_fn(F, *elem_params, dtype=dtype)
        U = U + torch.sum(psi * detJ0 * w)
    return U


def _local_element_energy(ue_flat, Xe, elem_param_tuple, energy_density_fn, shape_data, dtype):
    """Internal energy of ONE element, as a function of its own flattened
    nodal displacement (2*n_local,) -- small enough (8 or 18 entries) that
    its own Hessian is cheap, unlike the global tangent this whole module
    exists to avoid forming."""
    n_local = Xe.shape[0]
    ue = ue_flat.reshape(n_local, 2)
    Ns, dNs, ws = shape_data
    U = torch.zeros((), device=Xe.device, dtype=dtype)
    for N, dN_dxi, w in zip(Ns, dNs, ws):
        J0 = torch.einsum("ai,aj->ij", Xe, dN_dxi)
        detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
        detJ0 = torch.clamp(detJ0, min=1e-12)
        invJ0 = torch.stack([torch.stack([J0[1, 1] / detJ0, -J0[0, 1] / detJ0]),
                              torch.stack([-J0[1, 0] / detJ0, J0[0, 0] / detJ0])])
        dN_dX = dN_dxi @ invJ0
        grad_u = torch.einsum("ai,aj->ij", ue, dN_dX)
        F = torch.eye(2, device=Xe.device, dtype=dtype) + grad_u
        # energy_density_fn expects a leading batch dim (F[:, 0, 0]-style
        # indexing) -- add and remove a size-1 one, this is a single element.
        params_1 = tuple(p.reshape(1) for p in elem_param_tuple)
        psi = energy_density_fn(F.unsqueeze(0), *params_1, dtype=dtype)[0]
        U = U + psi * detJ0 * w
    return U


def compute_jacobi_diagonal(xy, quad, uv_full, elem_params, material, order, dtype, free_dofs):
    """Global diagonal of the tangent stiffness at the CURRENT displacement
    uv_full, computed WITHOUT ever forming the global (or even any element's
    full local) matrix explicitly beyond its own small diagonal: each
    element's local Hessian (cheap: 8x8 for Q4, 18x18 for Q9) is computed
    via vmap+hessian across all elements at once, its diagonal extracted,
    and scatter-added into the global diagonal by connectivity. This is the
    exact diagonal of K (sum of every element's own diagonal contribution
    at each shared node), not an approximation -- the standard, cheap way
    to build a Jacobi preconditioner for a matrix-free solver."""
    n_nodes = xy.shape[0]
    ndof = 2 * n_nodes
    energy_density_fn, _ = get_material_fns_torch(material)
    shape_data = precompute_shape_data(order, xy.device, dtype)

    Xe_all = xy[quad]                       # (Q, n_local, 2)
    ue_all = uv_full.reshape(n_nodes, 2)[quad].reshape(Xe_all.shape[0], -1)  # (Q, 2*n_local)

    local_hess_fn = hessian(_local_element_energy, argnums=0)
    batched_hess = vmap(local_hess_fn, in_dims=(0, 0, 0, None, None, None))
    H_local = batched_hess(ue_all, Xe_all, elem_params, energy_density_fn, shape_data, dtype)  # (Q, 2nl, 2nl)
    diag_local = torch.diagonal(H_local, dim1=-2, dim2=-1)  # (Q, 2*n_local)

    n_local = quad.shape[1]
    dof_idx = torch.stack([2 * quad, 2 * quad + 1], dim=-1).reshape(quad.shape[0], -1)  # (Q, 2*n_local)

    global_diag = torch.zeros(ndof, device=xy.device, dtype=dtype)
    global_diag.scatter_add_(0, dof_idx.reshape(-1), diag_local.reshape(-1))

    diag_free = global_diag[free_dofs]
    # Guard against a (rare, near-singular) near-zero diagonal entry, which
    # would blow up the preconditioner rather than merely slow it down.
    diag_free = torch.where(diag_free.abs() > 1e-10, diag_free, torch.ones_like(diag_free))
    return diag_free


def _make_energy_fn(xy, quad, n_nodes, material, order, dtype):
    energy_density_fn, _ = get_material_fns_torch(material)
    shape_data = precompute_shape_data(order, xy.device, dtype)
    ndof = 2 * n_nodes

    def energy_fn(u_free, elem_params, free_dofs):
        u_full = torch.zeros(ndof, dtype=dtype, device=u_free.device)
        u_full = u_full.index_copy(0, free_dofs, u_free)
        uv = u_full.reshape(n_nodes, 2)
        return element_energy_order_agnostic(xy, quad, uv, elem_params, energy_density_fn, shape_data, dtype)

    return energy_fn


def matrix_free_hvp(residual_fn, u, v, elem_params, free_dofs):
    """K @ v at point u, without ever forming K: forward-mode differentiate
    the RESIDUAL (already a reverse-mode gradient of the energy) along
    direction v. This is the one step that makes memory cost O(DOF)
    instead of O(DOF^2)."""
    _, Hv = jvp(lambda uu: residual_fn(uu, elem_params, free_dofs), (u,), (v,))
    return Hv


def conjugate_gradient(matvec, b, x0, tol, max_iter, precond_diag=None,
                        progress_every=None, progress_prefix=""):
    """CG for symmetric (K = Hessian of a scalar energy) systems, with an
    optional Jacobi (diagonal) preconditioner: precond_diag, if given, is
    the diagonal of K (see compute_jacobi_diagonal) and M^-1 is simply
    elementwise division by it -- the cheapest preconditioner available,
    but on the heterogeneous-material problems in this study it captures
    exactly the effect (large E in one region making that region's
    equations locally much stiffer) that otherwise drives up CG's
    iteration count. Returns (x, n_iter, final_residual_norm, converged).

    progress_every: if set, print a heartbeat line (iteration count,
    current relative residual, elapsed time, iters/s) every this many
    iterations -- unconditional on `verbose` in the caller, because at
    ~10M DOF a single CG call can itself run for hours, and with no
    heartbeat there is no way to tell 'still working' from 'stuck' from
    the outside. Cheap: only a norm and a print every progress_every
    iterations, not every iteration."""
    def apply_M_inv(v):
        return v if precond_diag is None else v / precond_diag

    t_start = time.time()
    x = x0.clone()
    r = b - matvec(x)
    z = apply_M_inv(r)
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = torch.linalg.norm(b) + 1e-30
    if torch.linalg.norm(r) / b_norm < tol:
        return x, 0, float(torch.linalg.norm(r)), True
    for it in range(1, max_iter + 1):
        Ap = matvec(p)
        alpha = rz_old / (torch.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rel_res = torch.linalg.norm(r) / b_norm
        if progress_every and it % progress_every == 0:
            elapsed = time.time() - t_start
            print(f"    {progress_prefix}[CG heartbeat] iter {it}/{max_iter}, "
                  f"rel_res={float(rel_res):.3e} (target {tol:.1e}), "
                  f"elapsed={elapsed:.0f}s, {it/max(elapsed, 1e-9):.1f} it/s")
        if rel_res < tol:
            return x, it, float(torch.linalg.norm(r)), True
        z = apply_M_inv(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / (rz_old + 1e-30)) * p
        rz_old = rz_new
    return x, max_iter, float(torch.linalg.norm(r)), False


def solve_matrix_free(xy, quad, free_dofs, elem_params, fext_free_full, n_free,
                       material="neo_hookean", order="Q4", nsteps=10, newton_max=30,
                       newton_tol=1e-7, cg_tol=1e-6, cg_max_iter=2000, use_jacobi=True,
                       device=None, dtype=torch.float64, verbose=True, checkpoint_path=None,
                       cg_progress_every=None):
    """Single-sample (no batch dimension) matrix-free Newton-CG solve.
    elem_params: tuple of (n_elements,) tensors. use_jacobi: recompute the
    exact diagonal of K at the start of every Newton iteration (see
    compute_jacobi_diagonal) and use it to precondition CG -- cheap
    relative to the CG iterations it is meant to reduce, since it only
    needs small per-element Hessians, never the global one. Returns
    (u_free, stats).

    checkpoint_path: if given, save (u_free, stats, next step) to this file
    after every completed load step, and resume from it if it already
    exists -- a multi-hour solve at ~10M DOF has no other way to survive a
    Colab disconnect or an accidental interrupt short of restarting from
    load step 1. Checkpointing at load-step granularity (not mid-Newton or
    mid-CG) keeps this simple; a step is at most a few minutes of work at
    the scales this solver targets, so re-doing one lost step is cheap
    relative to the whole solve."""
    device = device or xy.device
    n_nodes = xy.shape[0]
    ndof = 2 * n_nodes
    energy_fn = _make_energy_fn(xy, quad, n_nodes, material, order, dtype)
    residual_fn = grad(energy_fn, argnums=0)

    u_free = torch.zeros(n_free, dtype=dtype, device=device)
    stats = {"newton_iters_total": 0, "cg_iters_total": 0, "cg_failures": 0, "load_steps": nsteps}
    start_step = 1

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        u_free = ckpt["u_free"].to(device=device, dtype=dtype)
        stats = ckpt["stats"]
        start_step = ckpt["next_step"]
        print(f"  [checkpoint] resuming from {checkpoint_path}: load steps 1-{start_step - 1} "
              f"already done, continuing from step {start_step}/{nsteps}")
        if start_step > nsteps:
            return u_free, stats

    for step in range(start_step, nsteps + 1):
        alpha = step / nsteps
        fext_free = alpha * fext_free_full
        for it in range(1, newton_max + 1):
            R = residual_fn(u_free, elem_params, free_dofs) - fext_free
            res_norm = torch.linalg.norm(R)
            if res_norm < newton_tol:
                break
            stats["newton_iters_total"] += 1
            if cg_progress_every:
                print(f"  [step {step}/{nsteps}] Newton iter {it}: ||R||={float(res_norm):.3e} "
                      f"(target {newton_tol:.1e}) -- starting CG solve...")
            t_cg = time.time()

            def matvec(v):
                return matrix_free_hvp(residual_fn, u_free, v, elem_params, free_dofs)

            precond_diag = None
            if use_jacobi:
                u_full = torch.zeros(ndof, dtype=dtype, device=device).index_copy(0, free_dofs, u_free)
                precond_diag = compute_jacobi_diagonal(xy, quad, u_full, elem_params, material, order,
                                                        dtype, free_dofs)

            delta, cg_iters, cg_res, cg_converged = conjugate_gradient(
                matvec, -R, torch.zeros_like(u_free), cg_tol, cg_max_iter, precond_diag=precond_diag,
                progress_every=cg_progress_every, progress_prefix=f"step {step}/{nsteps} newton {it}: ")
            stats["cg_iters_total"] += cg_iters
            if cg_progress_every:
                print(f"    CG done: {cg_iters} iters in {time.time() - t_cg:.0f}s, "
                      f"converged={cg_converged}, final residual={cg_res:.3e}")
            if not cg_converged:
                stats["cg_failures"] += 1
                if verbose or cg_progress_every:
                    print(f"  [WARNING] CG did not converge at step {step}, Newton iter {it} "
                          f"(residual {cg_res:.3e} after {cg_iters} iters)")
            u_free = u_free + delta

            if (verbose or cg_progress_every) and it == newton_max:
                print(f"  [step {step}/{nsteps}] Newton did NOT converge, ||R||={res_norm:.3e}")
        if verbose:
            print(f"  [step {step}/{nsteps}] converged, ||R||={res_norm:.3e} "
                  f"(cumulative CG iters so far: {stats['cg_iters_total']})")

        if checkpoint_path is not None:
            tmp_path = checkpoint_path + ".tmp"
            torch.save({"u_free": u_free.cpu(), "stats": stats, "next_step": step + 1}, tmp_path)
            os.replace(tmp_path, checkpoint_path)  # atomic -- never leaves a half-written checkpoint
            print(f"  [checkpoint] saved after step {step}/{nsteps} -> {checkpoint_path}")

    return u_free, stats
