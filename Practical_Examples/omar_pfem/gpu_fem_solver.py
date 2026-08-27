"""
GPU-native Total-Lagrangian Newton-Raphson FEM solver, per the advisor's
request: "the comparison is based on a CPU implementation of the FEM
solver ... add a comparison with a GPU-native FEM implementation so that
computational efficiency is evaluated on comparable hardware."

Design: rather than re-implementing element assembly by hand in PyTorch
(duplicating and risking diverging from the validated CPU solver in
omar_pfem/data/data_generate_B{1,2}.py), this solver reuses:
  - The EXACT same reference force assembly (assemble_traction_top_spatial
    for B1, assemble_traction_inner_curved for B2) that produces the CPU
    solver's own external force vector Fext -- computed once with NumPy
    (it does not depend on the unknown displacement, so there is nothing
    to gain from doing it on GPU), then moved to a torch tensor.
  - The EXACT same per-element material-parameter evaluation the CPU
    solver uses (see precompute_element_params_B1/B2 below): E and nu are
    evaluated at each element's centroid via the SAME interpolator calls
    the CPU solver makes, THEN converted to (mu, lambda) etc. This is
    deliberately NOT the training-side compute_hyperelastic_energy_Q4
    convention (which converts (E, nu) -> material params per NODE first,
    then averages the converted params per element) -- for a nonlinear
    conversion, averaging-then-converting and converting-then-averaging
    are genuinely different formulas, and for B2 in particular the CPU
    convention (Cartesian element centroid -> polar (theta, r) -> lookup)
    has no simpler nodal-average shortcut at all, since Cartesian
    averaging and polar averaging do not commute under B2's curved
    coordinate map. Precomputing exactly what the CPU solver computes,
    once, in NumPy, and feeding it into the GPU solve as a constant
    per-element tensor sidesteps this ambiguity entirely rather than
    trying to reproduce it approximately inside PyTorch.

Newton-Raphson then uses PyTorch autodiff instead of a hand-derived
tangent: the residual R(u) = dU/du - Fext is the gradient of
Pi(u) = U(u) - Fext.u, and the tangent K(u) = d^2U/du^2 is its Hessian
(the Fext term is linear in u, so it contributes nothing to the Hessian).
torch.func.hessian + torch.func.vmap compute this exactly (not
numerically), batched across many samples solved simultaneously on the
GPU in one pass -- the natural GPU-native analogue of the CPU solver's
one-sample-at-a-time loop.

Correctness (see validate_gpu_fem_solver.py): with this precomputed-params
design, this solver's converged displacement field agrees with the CPU
reference solver to machine precision (~1e-13 relative), for both B1 and
B2, multiple mesh resolutions, and multiple materials -- not just "close",
the same equilibrium up to floating-point round-off.

This module intentionally does NOT reproduce the soft-Dirichlet ramp used
in the *training* objective (total_potential_energy_Q4_hyperelastic): that
ramp exists so the network's unconstrained output is projected toward a
BC-satisfying field for gradient-descent training. A genuine equilibrium
solve instead splits the true unknowns into free vs. fixed DOFs, exactly
like the CPU reference solver.
"""
import numpy as np
import torch
from torch.func import grad, hessian, vmap

from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch
from omar_pfem.data.materials import get_material_fns as get_material_fns_numpy
# the reference solver's own shape functions, so the Gauss-point locations
# this module looks the material up at are the same points, not a re-derivation
from omar_pfem.data.fem_core import shape_Q4
from omar_pfem.train_B1 import shape_Q4_torch


# ============================================================
# Per-element material parameter precomputation (NumPy, CPU-matching)
# ============================================================

def precompute_element_params_B1(nodes, elements, E_interp, nu_interp, material):
    """Reproduces data_generate_B1.solve_hyperelastic_TL_spatial's own
    per-element (E, nu) evaluation exactly: E/nu are looked up at each
    element's Cartesian centroid, then converted to material parameters.
    Returns a tuple of (n_elements,) float64 arrays (e.g. (mu, lam) for
    Neo-Hookean)."""
    _, E_nu_to_params_fn = get_material_fns_numpy(material)
    param_lists = None
    for e in elements:
        Xe = nodes[e]
        elem_center = Xe.mean(axis=0)
        E_val = E_interp(elem_center[None, :])[0]
        nu_val = nu_interp(elem_center[None, :])[0]
        params = E_nu_to_params_fn(E_val, nu_val)
        if param_lists is None:
            param_lists = [[] for _ in params]
        for i, p in enumerate(params):
            param_lists[i].append(float(p))
    return tuple(np.array(pl, dtype=np.float64) for pl in param_lists)


def precompute_element_params_B2(nodes, elements, E_interp, nu_interp, material):
    """Reproduces data_generate_B2.solve_hyperelastic_TL_ring's own material
    evaluation exactly: (E, nu) are looked up at EACH of the four Gauss
    points' own physical locations (N @ Xe), converted to polar (theta, r)
    before the lookup.

    Not at the element centroid. B2's solver was changed to per-Gauss-point
    sampling on 2026-08-10 (commit af7e67c) because its material field
    varies over a curved domain and a single centroid sample misses changes
    an element can span -- fem_core.element_K_and_fint_TL's docstring calls
    that "a genuine discretization error distinct from mesh refinement".
    This function was not updated to follow, so the two solvers quietly
    stopped solving the same problem and validate_gpu_fem_solver.py went
    from PASS to FAIL (4.81e-5 absolute, 1.15e-3 mean relative at N=11)
    without anyone re-running it. Returns (n_elements, 4) arrays, one value
    per element per Gauss point, in the same Gauss order fem_core uses.

    B1 is deliberately left at the centroid: its own CPU solver samples
    there too (data_generate_B1.solve_hyperelastic_TL_spatial), on a
    rectangular mesh where the 2026-08-10 commit confirmed it makes no
    difference. Matching the reference is the requirement, not uniformity
    between the two geometries."""
    _, E_nu_to_params_fn = get_material_fns_numpy(material)
    # Order-aware: high_dof_convergence_study.py calls this for Q9 meshes too,
    # where an element has 9 nodes and the rule is 3x3, not 4 and 2x2. Picking
    # the rule from the mesh keeps one function honest for both instead of
    # silently mis-indexing a Q9 element with Q4 shape functions.
    n_local = np.asarray(elements).shape[1]
    if n_local == 4:
        shape_fn = shape_Q4
        g = 1.0 / np.sqrt(3.0)
        gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    elif n_local == 9:
        from omar_pfem.data.q9_element import shape_Q9, gauss_3x3
        shape_fn = shape_Q9
        gauss = [(xi, eta) for (xi, eta, _w) in gauss_3x3()]
    else:
        raise ValueError(f"unsupported element with {n_local} local nodes")
    n_gp = len(gauss)

    param_lists = None
    for e in elements:
        Xe = nodes[e]
        for (xi, eta) in gauss:
            N, _ = shape_fn(xi, eta)
            X_gp = N @ Xe
            r_gp = np.linalg.norm(X_gp)
            theta_gp = np.arctan2(X_gp[1], X_gp[0])
            E_val = E_interp(np.array([[theta_gp, r_gp]]))[0]
            nu_val = nu_interp(np.array([[theta_gp, r_gp]]))[0]
            params = E_nu_to_params_fn(E_val, nu_val)
            if param_lists is None:
                param_lists = [[] for _ in params]
            for i, pv in enumerate(params):
                param_lists[i].append(float(pv))
    n_el = len(elements)
    return tuple(np.array(pl, dtype=np.float64).reshape(n_el, n_gp) for pl in param_lists)


# ============================================================
# Differentiable internal energy (shared Q4/2x2-Gauss assembly)
# ============================================================

def _element_energy_Q4(xy, quad, uv, elem_params, energy_density_fn, dtype):
    """Internal energy U(u) via full 2x2 Gauss quadrature, given
    PRECOMPUTED material parameters -- see precompute_element_params_B1/B2.

    Each entry of elem_params is either (n_elements,), one value used at all
    four Gauss points, or (n_elements, 4), one value per Gauss point. Both
    are needed: B1's reference solver samples at the element centroid and B2's
    samples at each Gauss point, and this module's job is to match whichever
    reference it is being compared against. uv is (n_nodes, 2) for a single
    sample (no batch dim; vmap adds it)."""
    Xe = xy[quad]     # (Q,4,2)
    ue = uv[quad]     # (Q,4,2)

    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    U = torch.zeros((), device=xy.device, dtype=dtype)
    for g_idx, (xi, eta) in enumerate(gps):
        xi_t = torch.tensor(float(xi), device=xy.device, dtype=dtype)
        eta_t = torch.tensor(float(eta), device=xy.device, dtype=dtype)
        _, dN_dxi = shape_Q4_torch(xi_t, eta_t, xy.device, dtype)

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
        F = I + grad_u  # (Q,2,2)

        gp_params = tuple(p[:, g_idx] if p.dim() == 2 else p for p in elem_params)
        psi = energy_density_fn(F, *gp_params, dtype=dtype)  # (Q,)
        U = U + torch.sum(psi * detJ0 * w)

    return U


def _make_energy_fn(xy, quad, n_nodes, material, dtype):
    """energy_fn(u_free, elem_params, free_dofs) -> scalar U for ONE
    sample, suitable for torch.func.grad/hessian + vmap. elem_params is a
    tuple of (n_elements,) tensors that VARY per sample (different random
    material field); free_dofs is shared (in_dims=None under vmap)."""
    energy_density_fn, _ = get_material_fns_torch(material)
    ndof = 2 * n_nodes

    def energy_fn(u_free, elem_params, free_dofs):
        u_full = torch.zeros(ndof, dtype=dtype, device=u_free.device)
        u_full = u_full.index_copy(0, free_dofs, u_free)
        uv = u_full.reshape(n_nodes, 2)
        return _element_energy_Q4(xy, quad, uv, elem_params, energy_density_fn, dtype)

    return energy_fn


def _newton_solve(energy_fn, free_dofs, elem_params_batch, fext_free_full,
                   n_free, B, nsteps, newton_max, tol, device, dtype, profile=None):
    """profile: optional dict: if given, accumulates wall-clock timing broken
    down into 'assembly' (residual + tangent/Hessian evaluation via
    torch.func autodiff -- the GPU-native analogue of forming fint/K) and
    'solve' (the dense torch.linalg.solve per Newton step) buckets, each
    bracketed by torch.cuda.synchronize() when on GPU so CUDA's asynchronous
    dispatch cannot leak into the wrong bucket or be missed entirely. See
    fem_cost_breakdown.py."""
    import time as _time
    residual_fn = grad(energy_fn, argnums=0)
    tangent_fn = hessian(energy_fn, argnums=0)
    batched_residual = vmap(residual_fn, in_dims=(0, 0, None))
    batched_tangent = vmap(tangent_fn, in_dims=(0, 0, None))

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if profile is not None:
        profile.setdefault("t_assembly_s", 0.0)   # residual + tangent (Hessian) autodiff evaluation
        profile.setdefault("t_solve_s", 0.0)      # torch.linalg.solve (dense, per Newton step)
        profile.setdefault("n_newton_iters_total", 0)
        profile.setdefault("n_load_steps", nsteps)
        profile.setdefault("newton_tol", tol)
        profile.setdefault("newton_max", newton_max)
        profile.setdefault("dtype", str(dtype))
        profile.setdefault("linear_solver", "torch.linalg.solve (dense, per-sample batched via vmap)")

    u_free = torch.zeros(B, n_free, dtype=dtype, device=device)
    for step in range(1, nsteps + 1):
        alpha = step / nsteps
        fext_free = alpha * fext_free_full
        for it in range(1, newton_max + 1):
            if profile is not None:
                _sync(); t0 = _time.perf_counter()
            dU_du = batched_residual(u_free, elem_params_batch, free_dofs)
            R = dU_du - fext_free
            res_norm = torch.linalg.norm(R, dim=1)
            converged = torch.all(res_norm < tol)
            if profile is not None:
                _sync(); profile["t_assembly_s"] += _time.perf_counter() - t0
                profile["n_newton_iters_total"] += 1
            if converged:
                break
            if profile is not None:
                _sync(); t0 = _time.perf_counter()
            K = batched_tangent(u_free, elem_params_batch, free_dofs)
            if profile is not None:
                _sync(); profile["t_assembly_s"] += _time.perf_counter() - t0
                _sync(); t0 = _time.perf_counter()
            delta = torch.linalg.solve(K, -R.unsqueeze(-1)).squeeze(-1)
            if profile is not None:
                _sync(); profile["t_solve_s"] += _time.perf_counter() - t0
            u_free = u_free + delta
    return u_free


def solve_batch_gpu_newton_B1(
    xy, quad, bottom_nodes, elem_params_batch, fext_full_batch,
    material="neo_hookean", nsteps=10, newton_max=30, tol=1e-7,
    device=None, dtype=torch.float64, profile=None,
):
    """Batched GPU Newton-Raphson solve for B1 (bottom edge fixed, both
    DOFs). elem_params_batch: tuple of (B, n_elements) tensors (per-sample
    precomputed material params, see precompute_element_params_B1).
    fext_full_batch: (B, ndof). Returns (B, n_nodes, 2). profile: optional
    dict, see _newton_solve."""
    device = device or xy.device
    n_nodes = xy.shape[0]
    ndof = 2 * n_nodes
    B = fext_full_batch.shape[0]

    bottom_dofs = torch.cat([2 * bottom_nodes, 2 * bottom_nodes + 1])
    all_dofs = torch.arange(ndof, device=device)
    free_mask = torch.ones(ndof, dtype=torch.bool, device=device)
    free_mask[bottom_dofs] = False
    free_dofs = all_dofs[free_mask]
    n_free = free_dofs.shape[0]

    energy_fn = _make_energy_fn(xy, quad, n_nodes, material, dtype)
    fext_free_full = fext_full_batch[:, free_dofs]

    u_free = _newton_solve(energy_fn, free_dofs, elem_params_batch, fext_free_full,
                            n_free, B, nsteps, newton_max, tol, device, dtype, profile=profile)

    u_full = torch.zeros(B, ndof, dtype=dtype, device=device)
    u_full[:, free_dofs] = u_free
    return u_full.reshape(B, n_nodes, 2)


def solve_batch_gpu_newton_B2(
    xy, quad, theta0_nodes, thetahalfpi_nodes, elem_params_batch, fext_full_batch,
    material="neo_hookean", nsteps=10, newton_max=30, tol=1e-7,
    device=None, dtype=torch.float64, profile=None,
):
    """Batched GPU Newton-Raphson solve for B2 (mixed-component symmetry
    BCs: u_y=0 at theta0_nodes, u_x=0 at thetahalfpi_nodes -- see
    data_generate_B2.solve_hyperelastic_TL_ring). elem_params_batch: tuple
    of (B, n_elements) tensors (see precompute_element_params_B2).
    fext_full_batch: (B, ndof). Returns (B, n_nodes, 2). profile: optional
    dict, see _newton_solve."""
    device = device or xy.device
    n_nodes = xy.shape[0]
    ndof = 2 * n_nodes
    B = fext_full_batch.shape[0]

    fixed_dofs = torch.cat([2 * theta0_nodes + 1, 2 * thetahalfpi_nodes])
    all_dofs = torch.arange(ndof, device=device)
    free_mask = torch.ones(ndof, dtype=torch.bool, device=device)
    free_mask[fixed_dofs] = False
    free_dofs = all_dofs[free_mask]
    n_free = free_dofs.shape[0]

    energy_fn = _make_energy_fn(xy, quad, n_nodes, material, dtype)
    fext_free_full = fext_full_batch[:, free_dofs]

    u_free = _newton_solve(energy_fn, free_dofs, elem_params_batch, fext_free_full,
                            n_free, B, nsteps, newton_max, tol, device, dtype, profile=profile)

    u_full = torch.zeros(B, ndof, dtype=dtype, device=device)
    u_full[:, free_dofs] = u_free
    return u_full.reshape(B, n_nodes, 2)


def assemble_fext_B1_numpy(nodes, elements, Ly, ty_interp):
    """Thin wrapper around the CPU reference solver's own consistent nodal
    force assembly, so this module's external-force vector is guaranteed
    identical (not just similar) to the one the CPU solver equilibrates
    against."""
    from omar_pfem.data.data_generate_B1 import assemble_traction_top_spatial
    return assemble_traction_top_spatial(nodes, elements, Ly, ty_interp)


def assemble_fext_B2_numpy(nodes, elements, R_in, p_interp):
    """Thin wrapper around the CPU reference solver's own consistent nodal
    pressure-force assembly (see assemble_fext_B1_numpy)."""
    from omar_pfem.data.data_generate_B2 import assemble_traction_inner_curved
    return assemble_traction_inner_curved(nodes, elements, R_in, p_interp)
