"""
Mesh (h-refinement) convergence study using the matrix-free Newton-CG
solver, computing L2 norm, H1 semi-norm, and energy-norm errors (plus
observed convergence rates) against a common fine-mesh reference solution
-- and comparing Q4 (bilinear) against Q9 (biquadratic) at the same
corner-node resolutions -- per the advisor's fourth-round request.

Norm definitions (all against the SAME fixed analytic (E, nu, load) field,
reusing mesh_convergence.py's own AnalyticFieldB1/B2 for the same reason it
uses them: a genuine closed-form function of (x, y) is required so the
identical physical problem is solved at every mesh size):

  - L2 norm:  ||e||_L2 = sqrt( integral_Omega |e|^2 dOmega ), e = u_h -
    u_ref, evaluated via Gauss quadrature on the COARSE mesh's own
    elements; u_ref (known only at the fine reference mesh's own nodes) is
    evaluated at each coarse Gauss point EXACTLY, via direct point location
    into the fine mesh's own element (index arithmetic on its known
    structured grid, no search) followed by that element's own shape
    functions (see evaluate_fe_field_and_gradient) -- not a generic
    re-triangulation/interpolation of its nodes.
  - H1 semi-norm: ||e||_H1 = sqrt( integral_Omega |grad(e)|^2 dOmega ).
    grad(u_h) at each coarse Gauss point comes directly from the coarse
    element's own shape-function gradients (exact for that element).
    grad(u_ref) at the same points comes from the SAME point-located fine
    element's own shape-function gradients (exact for a piecewise-
    bilinear/biquadratic FE field) -- an earlier version of this module
    instead finite-differenced a generic Delaunay-triangulation
    interpolant of the fine mesh's nodes, which produced a gradient with a
    fixed noise floor that did NOT shrink as the coarse mesh was refined
    (a piecewise-linear interpolant's true gradient is only piecewise-
    constant; finite-differencing across its triangle boundaries measures
    interpolation artifacts, not the fine solution's own discretization
    error) -- this showed up empirically as a near-zero or even negative
    observed H1 convergence rate despite L2 and the energy norm both
    converging correctly, which is what prompted this fix.
  - Energy norm: relative error in total strain energy,
    |U(u_h) - U(u_ref)| / |U(u_ref)|, both computed via the same Gauss-
    quadrature energy assembly (matrix_free_solver.element_energy_order_agnostic)
    -- a global scalar that can look converged even where a pointwise
    L2/H1 error has not, and vice versa.

Convergence rate: for two consecutive resolutions with errors e1 (coarser,
mesh size h1) and e2 (finer, h2), p = log(e1/e2) / log(h1/h2), plus a
single log-log least-squares fit across every resolution for a more robust
overall estimate. h is taken proportional to 1/(N-1) (N = corner nodes per
side), i.e. h1/h2 = (N2-1)/(N1-1).

Usage:
  python -m omar_pfem.high_dof_convergence_study \
      --geometry B1 --material neo_hookean \
      --resolutions 6,11,16,21,31,41 --fine_N 81 --orders Q4,Q9 \
      --out_json high_dof_convergence_B1_neo_hookean.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from omar_pfem.mesh_convergence import AnalyticFieldB1, AnalyticFieldB2
from omar_pfem.matrix_free_solver import (
    solve_matrix_free, element_energy_order_agnostic, precompute_shape_data, _gauss_2x2,
)
from omar_pfem.data.q9_element import generate_grid_Q9, gauss_3x3, shape_Q9, _NODE_XI_ETA, _lagrange1d
from omar_pfem.data.fem_core import shape_Q4
from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch


def shape_Q4_batched(xi, eta):
    """Vectorized shape_Q4: xi, eta are (Q,) arrays. Returns N (Q,4),
    dN_dxi (Q,4,2) -- avoids a Python-level loop over Gauss/query points,
    which otherwise dominates wall-clock time once thousands of points are
    evaluated (every coarse-mesh Gauss point, times several Newton
    corrections each, for the reference-field evaluation)."""
    N = 0.25 * np.stack([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                          (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], axis=-1)
    z = np.zeros_like(xi)
    dN_dxi = 0.25 * np.stack([
        np.stack([-(1 - eta), -(1 - xi)], axis=-1),
        np.stack([(1 - eta), -(1 + xi)], axis=-1),
        np.stack([(1 + eta), (1 + xi)], axis=-1),
        np.stack([-(1 + eta), (1 - xi)], axis=-1),
    ], axis=1)
    return N, dN_dxi


def shape_Q9_batched(xi, eta):
    """Vectorized shape_Q9 (see shape_Q4_batched's rationale). xi, eta:
    (Q,) arrays. Returns N (Q,9), dN_dxi (Q,9,2)."""
    Q = xi.shape[0]
    N = np.zeros((Q, 9))
    dN_dxi = np.zeros((Q, 9, 2))
    for k, (nx, ny) in enumerate(_NODE_XI_ETA):
        Lx, dLx = _lagrange1d(xi, nx)
        Ly, dLy = _lagrange1d(eta, ny)
        N[:, k] = Lx * Ly
        dN_dxi[:, k, 0] = dLx * Ly
        dN_dxi[:, k, 1] = Lx * dLy
    return N, dN_dxi


def _edge_quadrature(order):
    """1D Gauss points/weights along a reference edge: 2-point for Q4
    (matches data_generate_B1/B2.py's own edge quadrature exactly), 3-point
    for Q9 -- consistent with its 3x3 interior rule, and needed because a
    Q9 edge's own shape functions are themselves quadratic in the edge
    coordinate, not linear."""
    if order == "Q4":
        g = 1.0 / np.sqrt(3.0)
        return [(-g, 1.0), (g, 1.0)]
    g = np.sqrt(3.0 / 5.0)
    return [(-g, 5.0 / 9.0), (0.0, 8.0 / 9.0), (g, 5.0 / 9.0)]


def _shape_fn(order):
    return shape_Q4 if order == "Q4" else shape_Q9


def assemble_traction_top_generic(nodes, elements, Ly, ty_interp, order, tol=1e-9):
    """Order-agnostic replacement for data_generate_B1.assemble_traction_top_spatial:
    identical physics and quadrature convention for Q4 (verified to reduce
    to exactly the same local-node pair), generalized to also correctly
    load a Q9 element's own edge midside node -- the ORIGINAL Q4-only
    function silently ignores that node entirely if reused on a Q9 mesh,
    which is a real (not just cosmetic) bug: it would under-integrate the
    top-edge traction on every Q9 mesh."""
    n_nodes = nodes.shape[0]
    Fext = np.zeros(2 * n_nodes, dtype=float)
    shape_fn = _shape_fn(order)
    gps = _edge_quadrature(order)
    # Corner indices spanning the top edge (eta=+1), used only to check
    # whether THIS element is a boundary element and to get its edge
    # length -- same corner pair as Q4 for both orders, since Q9's corner
    # ordering matches Q4's by construction (see q9_element.py).
    for e in elements:
        Xe = nodes[e]
        y_top_a, y_top_b = Xe[2, 1], Xe[3, 1]
        if abs(y_top_a - Ly) < tol and abs(y_top_b - Ly) < tol:
            edge_len = np.linalg.norm(Xe[3] - Xe[2])
            for (xi, w1d) in gps:
                N, _ = shape_fn(xi, 1.0)
                x_pos = float(N @ Xe[:, 0])
                ty_value = ty_interp(np.array([[x_pos, Ly]]))[0]
                t = np.array([0.0, ty_value], dtype=float)
                Jedge = edge_len / 2.0
                for a in range(len(N)):
                    Fext[2 * e[a]:2 * e[a] + 2] += N[a] * t * (w1d * Jedge)
    return Fext


def assemble_traction_inner_generic(nodes, elements, R_in, p_interp, order, tol=1e-9):
    """Order-agnostic replacement for data_generate_B2.assemble_traction_inner_curved,
    same rationale as assemble_traction_top_generic above."""
    n_nodes = nodes.shape[0]
    Fext = np.zeros(2 * n_nodes, dtype=float)
    shape_fn = _shape_fn(order)
    gps = _edge_quadrature(order)
    for e in elements:
        Xe = nodes[e]
        r1 = np.linalg.norm(Xe[0])
        r4 = np.linalg.norm(Xe[3])
        if abs(r1 - R_in) < tol and abs(r4 - R_in) < tol:
            X1, X4 = Xe[0], Xe[3]
            edge_vec = X4 - X1
            edge_len = np.linalg.norm(edge_vec)
            tangent_hat = edge_vec / edge_len
            normal = np.array([tangent_hat[1], -tangent_hat[0]])
            midpoint = 0.5 * (X1 + X4)
            if np.dot(normal, midpoint) < 0:
                normal = -normal
            for (eta, w1d) in gps:
                N, _ = shape_fn(-1.0, eta)
                pos = N @ Xe
                theta_pos = np.arctan2(pos[1], pos[0])
                p_value = p_interp(np.array([[theta_pos, R_in]]))[0]
                t = p_value * normal
                Jedge = edge_len / 2.0
                for a in range(len(N)):
                    Fext[2 * e[a]:2 * e[a] + 2] += N[a] * t * (w1d * Jedge)
    return Fext


def build_mesh_and_bcs(geometry, order, N, material, device, dtype):
    if geometry == "B1":
        Lx = Ly = 1.0
        E_fn, nu_fn, ty_fn = AnalyticFieldB1("E"), AnalyticFieldB1("nu"), AnalyticFieldB1("ty")
        if order == "Q4":
            from omar_pfem.data.data_generate_B1 import generate_grid_Q4
            nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
        else:
            nodes, elements = generate_grid_Q9(Lx, Ly, N, N)
        tolx = 1e-9
        bottom_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
        fixed_dofs = np.concatenate([2 * bottom_nodes, 2 * bottom_nodes + 1])
        fext_full = assemble_traction_top_generic(nodes, elements, Ly, ty_fn, order)
        from omar_pfem.gpu_fem_solver import precompute_element_params_B1 as precompute_params
        elem_params = precompute_params(nodes, elements, E_fn, nu_fn, material)
    else:
        R_in, R_out = 1.0, 2.0
        E_fn, nu_fn, p_fn = AnalyticFieldB2("E"), AnalyticFieldB2("nu"), AnalyticFieldB2("p")
        if order == "Q4":
            from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring
            nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
        else:
            # Polar-mapped Q9 mesh: build the Q9 reference grid in (r, theta)
            # -- r FAST, theta SLOW, matching generate_grid_Q4_ring's own
            # convention exactly (see that function's docstring) so both
            # orders share one element-traversal convention for the same
            # geometry -- then map to Cartesian.
            theta_max = np.pi / 2
            nodes_polar, elements = generate_grid_Q9(R_out - R_in, theta_max, N, N)
            r = nodes_polar[:, 0] + R_in
            theta = nodes_polar[:, 1]
            nodes = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        tolx = 1e-9
        theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
        thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]
        fixed_dofs = np.concatenate([2 * theta0_nodes + 1, 2 * thetahalfpi_nodes])
        fext_full = assemble_traction_inner_generic(nodes, elements, R_in, p_fn, order)
        from omar_pfem.gpu_fem_solver import precompute_element_params_B2 as precompute_params
        elem_params = precompute_params(nodes, elements, E_fn, nu_fn, material)

    ndof = 2 * len(nodes)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    return nodes, elements, free_dofs, fext_full, elem_params


def solve_one(geometry, order, N, material, device, dtype, cg_tol, newton_tol,
              use_jacobi=True, cg_max_iter=2000, verbose=False, checkpoint_path=None):
    nodes, elements, free_dofs, fext_full, elem_params_np = build_mesh_and_bcs(
        geometry, order, N, material, device, dtype)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_dofs_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
    elem_params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in elem_params_np)
    fext_free_t = torch.tensor(fext_full[free_dofs], dtype=dtype, device=device)

    t0 = time.time()
    u_free, stats = solve_matrix_free(
        xy_t, quad_t, free_dofs_t, elem_params_t, fext_free_t, n_free=len(free_dofs),
        material=material, order=order, nsteps=10, newton_max=30,
        newton_tol=newton_tol, cg_tol=cg_tol, cg_max_iter=cg_max_iter, use_jacobi=use_jacobi,
        device=device, dtype=dtype, verbose=verbose, checkpoint_path=checkpoint_path)
    wall_s = time.time() - t0

    ndof = 2 * len(nodes)
    u_full = torch.zeros(ndof, dtype=dtype, device=device)
    u_full = u_full.index_copy(0, free_dofs_t, u_free)

    shape_data = precompute_shape_data(order, device, dtype)
    energy_density_fn, _ = get_material_fns_torch(material)
    U = element_energy_order_agnostic(xy_t, quad_t, u_full.reshape(len(nodes), 2),
                                       elem_params_t, energy_density_fn, shape_data, dtype)

    return {
        "nodes": nodes, "elements": elements, "u": u_full.reshape(len(nodes), 2).cpu().numpy(),
        "N": N, "n_dof": ndof, "n_free_dof": len(free_dofs), "strain_energy": float(U.item()),
        "wall_clock_s": wall_s, "stats": stats,
    }


def gauss_points_and_weights_physical(nodes, elements, order):
    """Physical (x,y) coordinates, detJ, and weight of every Gauss point of
    every element in a mesh -- used both to integrate the L2/H1 errors and
    to evaluate u_h / grad(u_h) there via the coarse mesh's OWN shape
    functions (exact, since these are that mesh's own quadrature points).
    Vectorized over elements (each of the few Gauss points is the same
    reference location for every element, so N/dN_dxi there are computed
    ONCE and broadcast, not recomputed per element)."""
    gauss = _gauss_2x2() if order == "Q4" else gauss_3x3()
    shape_fn_batched = shape_Q4_batched if order == "Q4" else shape_Q9_batched
    n_elements = len(elements)
    Xe_all = nodes[elements]  # (n_elements, n_local, 2)

    pts, detJs, ws, Ns, dN_dXs, elem_idx = [], [], [], [], [], []
    for (xi, eta, w) in gauss:
        xi_arr = np.full(n_elements, xi)
        eta_arr = np.full(n_elements, eta)
        N, dN_dxi = shape_fn_batched(xi_arr, eta_arr)              # (n_elements, n_local), (n_elements, n_local, 2)
        J0 = np.einsum("qai,qaj->qij", Xe_all, dN_dxi)             # (n_elements, 2, 2)
        detJ0 = J0[:, 0, 0] * J0[:, 1, 1] - J0[:, 0, 1] * J0[:, 1, 0]
        invJ0 = np.zeros_like(J0)
        invJ0[:, 0, 0] = J0[:, 1, 1] / detJ0
        invJ0[:, 1, 1] = J0[:, 0, 0] / detJ0
        invJ0[:, 0, 1] = -J0[:, 0, 1] / detJ0
        invJ0[:, 1, 0] = -J0[:, 1, 0] / detJ0
        dN_dX = np.einsum("qaj,qjk->qak", dN_dxi, invJ0)           # (n_elements, n_local, 2)
        phys = np.einsum("qa,qad->qd", N, Xe_all)                  # (n_elements, 2)

        pts.append(phys); detJs.append(detJ0); ws.append(np.full(n_elements, w))
        Ns.append(N); dN_dXs.append(dN_dX); elem_idx.append(np.arange(n_elements))

    return (np.concatenate(pts), np.concatenate(detJs), np.concatenate(ws),
            np.concatenate(Ns), np.concatenate(dN_dXs), np.concatenate(elem_idx))


def _physical_to_normalized(pts, geometry, R_in=1.0, R_out=2.0, theta_max=np.pi / 2, Lx=1.0, Ly=1.0):
    """Maps physical (x, y) points to the mesh generators' own [0,1]x[0,1]
    structured reference coordinates: direct for B1 (Cartesian), via
    (r, theta) for B2 -- r is the FAST index (x_norm), theta the SLOW index
    (y_norm), matching generate_grid_Q4_ring's own convention (and, after
    the fix above, generate_grid_Q9's B2 call)."""
    if geometry == "B1":
        return pts[:, 0] / Lx, pts[:, 1] / Ly
    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    return (r - R_in) / (R_out - R_in), theta / theta_max


def evaluate_fe_field_and_gradient(query_pts, fine, order, geometry, **geom_kwargs):
    """Exact FE evaluation of a fine structured mesh's own displacement
    field AND its gradient at arbitrary physical query points, via direct
    point location (index arithmetic on the known structured grid, no
    search) into the fine mesh's own element, followed by that element's
    own shape functions/gradients -- replacing an earlier version that
    finite-differenced a generic Delaunay-triangulation interpolant, which
    produced a gradient with a fixed noise floor that did not shrink as
    the COARSE mesh was refined (a piecewise-linear interpolant's true
    gradient is only piecewise-CONSTANT; finite-differencing across its
    triangle boundaries injects error that is NOT the fine solution's own
    discretization error). This version is both more correct (uses the
    fine mesh's own, already-known shape functions exactly, not a
    re-triangulation of its nodes) and faster (index arithmetic, not a
    Delaunay build + per-point interpolator calls)."""
    nodes_f, elements_f, u_f, N_fine = fine["nodes"], fine["elements"], fine["u"], fine["N"]
    nx_elem = N_fine - 1
    ny_elem = N_fine - 1

    x_norm, y_norm = _physical_to_normalized(query_pts, geometry, **geom_kwargs)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)

    ie = np.clip(np.floor(x_norm * nx_elem).astype(int), 0, nx_elem - 1)
    je = np.clip(np.floor(y_norm * ny_elem).astype(int), 0, ny_elem - 1)
    dx, dy = 1.0 / nx_elem, 1.0 / ny_elem
    xi = 2.0 * (x_norm - ie * dx) / dx - 1.0
    eta = 2.0 * (y_norm - je * dy) / dy - 1.0
    xi = np.clip(xi, -1.0, 1.0)
    eta = np.clip(eta, -1.0, 1.0)

    # WHICH fine element contains the point is exact (cell boundaries are
    # constant-r / constant-theta lines for B2, constant-x/y for B1, so
    # locating (ie, je) from (x_norm, y_norm) alone is always correct).
    # But (xi, eta) WITHIN that element is only a good INITIAL GUESS here,
    # not exact for B2: linearly interpolating in (r, theta) is NOT the
    # same map as the element's own isoparametric map N(xi,eta) @ Xe, which
    # is bilinear/biquadratic in CARTESIAN (x, y) -- these two coincide
    # exactly for B1 (whose mesh already IS linear in Cartesian coordinates)
    # but not for B2's polar-mapped, curved-looking quadrilaterals. A few
    # Newton corrections on the true isoparametric inverse map fix this
    # for both geometries (and are a costless no-op for B1, where the
    # initial guess already satisfies the map to machine precision).
    elem_flat_idx = je * nx_elem + ie          # matches every mesh generator's own (je-outer, ie-inner) build order
    elem_nodes = elements_f[elem_flat_idx]     # (Q, n_local)
    Xe = nodes_f[elem_nodes]                   # (Q, n_local, 2): this element's OWN corner/edge/center physical coords
    ue = u_f[elem_nodes]                       # (Q, n_local, 2): this element's OWN nodal displacements

    shape_fn_batched = shape_Q4_batched if order == "Q4" else shape_Q9_batched

    for _ in range(4):
        N_all, dN_dxi_all = shape_fn_batched(xi, eta)
        phys_guess = np.einsum("qa,qad->qd", N_all, Xe)
        residual = phys_guess - query_pts                       # (Q, 2)
        J0_newton = np.einsum("qai,qaj->qij", Xe, dN_dxi_all)    # d(phys)/d(xi,eta), (Q,2,2)
        det = J0_newton[:, 0, 0] * J0_newton[:, 1, 1] - J0_newton[:, 0, 1] * J0_newton[:, 1, 0]
        inv00, inv11 = J0_newton[:, 1, 1] / det, J0_newton[:, 0, 0] / det
        inv01, inv10 = -J0_newton[:, 0, 1] / det, -J0_newton[:, 1, 0] / det
        d_xi = inv00 * residual[:, 0] + inv01 * residual[:, 1]
        d_eta = inv10 * residual[:, 0] + inv11 * residual[:, 1]
        xi = np.clip(xi - d_xi, -1.0, 1.0)
        eta = np.clip(eta - d_eta, -1.0, 1.0)

    N_all, dN_dxi_all = shape_fn_batched(xi, eta)
    u_at_pts = np.einsum("qa,qad->qd", N_all, ue)

    J0 = np.einsum("qai,qaj->qij", Xe, dN_dxi_all)          # (Q,2,2)
    detJ0 = J0[:, 0, 0] * J0[:, 1, 1] - J0[:, 0, 1] * J0[:, 1, 0]
    invJ0 = np.zeros_like(J0)
    invJ0[:, 0, 0] = J0[:, 1, 1] / detJ0
    invJ0[:, 1, 1] = J0[:, 0, 0] / detJ0
    invJ0[:, 0, 1] = -J0[:, 0, 1] / detJ0
    invJ0[:, 1, 0] = -J0[:, 1, 0] / detJ0
    dN_dX = np.einsum("qaj,qjk->qak", dN_dxi_all, invJ0)     # (Q, n_local, 2)
    grad_u = np.einsum("qad,qak->qdk", ue, dN_dX)            # (Q,2,2) = du_d/dx_k

    return u_at_pts, grad_u


def compute_l2_h1_errors(coarse, fine, order, geometry, **geom_kwargs):
    nodes_c, elements_c, u_c = coarse["nodes"], coarse["elements"], coarse["u"]

    pts, detJs, ws, Ns, dN_dXs, elem_idx = gauss_points_and_weights_physical(nodes_c, elements_c, order)

    u_ref_at_gp, grad_u_ref = evaluate_fe_field_and_gradient(pts, fine, order, geometry, **geom_kwargs)
    # u_h at each Gauss point: N . u_c over that Gauss point's own element
    u_c_per_elem = u_c[elements_c]  # (n_elements, n_local, 2)
    u_h_at_gp = np.einsum("qa,qad->qd", Ns, u_c_per_elem[elem_idx])

    e = u_h_at_gp - u_ref_at_gp
    l2_sq_integrand = np.sum(e ** 2, axis=1) * detJs * ws
    l2_norm = np.sqrt(np.sum(l2_sq_integrand))

    grad_u_h = np.einsum("qad,qak->qdk", u_c_per_elem[elem_idx], dN_dXs)  # (Q, 2, 2) = du_i/dx_j

    grad_e = grad_u_h - grad_u_ref
    h1_sq_integrand = np.sum(grad_e ** 2, axis=(1, 2)) * detJs * ws
    h1_seminorm = np.sqrt(np.sum(h1_sq_integrand))

    ref_norm_l2 = np.sqrt(np.sum(np.sum(u_ref_at_gp ** 2, axis=1) * detJs * ws)) + 1e-30
    ref_norm_h1 = np.sqrt(np.sum(np.sum(grad_u_ref ** 2, axis=(1, 2)) * detJs * ws)) + 1e-30

    return {
        "l2_abs": float(l2_norm), "l2_rel": float(l2_norm / ref_norm_l2),
        "h1_semi_abs": float(h1_seminorm), "h1_semi_rel": float(h1_seminorm / ref_norm_h1),
    }


def fit_convergence_rate(hs, errors):
    """Single log-log least-squares fit p across all (h, error) pairs where
    error > 0, plus each consecutive pair's own p for comparison."""
    hs, errors = np.array(hs), np.array(errors)
    mask = errors > 0
    if mask.sum() < 2:
        return None, []
    log_h, log_e = np.log(hs[mask]), np.log(errors[mask])
    p_fit = float(np.polyfit(log_h, log_e, 1)[0])
    pairwise = []
    for i in range(1, len(hs)):
        if errors[i] > 0 and errors[i - 1] > 0:
            p = np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])
            pairwise.append(float(p))
    return p_fit, pairwise


def main():
    parser = argparse.ArgumentParser("High-DOF mesh convergence study (matrix-free Newton-CG, Q4 vs Q9)")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--resolutions", type=str, default="6,11,16,21,31,41")
    parser.add_argument("--fine_N", type=int, default=81,
                         help="Common fine-mesh reference resolution (corner nodes per side)")
    parser.add_argument("--orders", type=str, default="Q4,Q9")
    parser.add_argument("--cg_tol", type=float, default=1e-8)
    parser.add_argument("--newton_tol", type=float, default=1e-8)
    parser.add_argument("--cg_max_iter", type=int, default=2000)
    parser.add_argument("--no_jacobi", action="store_true",
                         help="Disable the Jacobi preconditioner (plain CG) -- for comparing "
                              "iteration counts/wall-clock with vs. without it on your own hardware")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                         help="Directory to checkpoint the FINE reference solve into (one file per "
                              "order), resumed automatically if present. Point this at a Google Drive "
                              "path (e.g. /content/drive/MyDrive/pfem_ckpt) for a multi-hour run, since "
                              "/content alone does not survive a full Colab runtime reset.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    resolutions = [int(n) for n in args.resolutions.split(",") if n.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    report = {"geometry": args.geometry, "material": args.material, "fine_N": args.fine_N,
              "device": str(device), "orders": {}}

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    for order in orders:
        print(f"\n{'='*90}\nORDER = {order}\n{'='*90}")
        print(f"Solving common fine reference at N={args.fine_N}...")
        ckpt_path = (os.path.join(args.checkpoint_dir,
                                   f"fine_{args.geometry}_{args.material}_{order}_N{args.fine_N}.pt")
                     if args.checkpoint_dir else None)
        fine = solve_one(args.geometry, order, args.fine_N, args.material, device, dtype,
                          args.cg_tol, args.newton_tol, use_jacobi=not args.no_jacobi,
                          cg_max_iter=args.cg_max_iter, verbose=False, checkpoint_path=ckpt_path)
        print(f"  Fine reference: n_dof={fine['n_dof']}, strain_energy={fine['strain_energy']:.6e}, "
              f"wall_clock={fine['wall_clock_s']:.1f}s, "
              f"Newton iters={fine['stats']['newton_iters_total']}, "
              f"CG iters={fine['stats']['cg_iters_total']}, "
              f"CG failures={fine['stats']['cg_failures']}")

        rows = []
        for N in resolutions:
            if N >= args.fine_N:
                continue
            if args.fine_N / N < 3.0:
                print(f"  WARNING: fine_N={args.fine_N} is only {args.fine_N / N:.1f}x N={N}. "
                      f"The reference mesh's own discretization error is not negligible next to "
                      f"this resolution's error, which will flatten (underestimate) the measured "
                      f"convergence rate -- especially for H1 and for Q9. Use fine_N >= 4x this N "
                      f"(or drop this N from --resolutions) for a trustworthy rate.")
            print(f"\nSolving N={N} ({order})...")
            coarse = solve_one(args.geometry, order, N, args.material, device, dtype,
                                args.cg_tol, args.newton_tol, use_jacobi=not args.no_jacobi,
                                cg_max_iter=args.cg_max_iter, verbose=False)
            geom_kwargs = ({"Lx": 1.0, "Ly": 1.0} if args.geometry == "B1"
                            else {"R_in": 1.0, "R_out": 2.0, "theta_max": np.pi / 2})
            errs = compute_l2_h1_errors(coarse, fine, order, args.geometry, **geom_kwargs)
            energy_rel_err = abs(coarse["strain_energy"] - fine["strain_energy"]) / (abs(fine["strain_energy"]) + 1e-30)
            row = {
                "N": N, "n_dof": coarse["n_dof"], "wall_clock_s": coarse["wall_clock_s"],
                "newton_iters": coarse["stats"]["newton_iters_total"],
                "cg_iters": coarse["stats"]["cg_iters_total"],
                "cg_failures": coarse["stats"]["cg_failures"],
                "l2_rel_error": errs["l2_rel"], "h1_semi_rel_error": errs["h1_semi_rel"],
                "energy_rel_error": float(energy_rel_err),
            }
            rows.append(row)
            if row["cg_failures"] > 0:
                print(f"  WARNING: {row['cg_failures']} CG failure(s) at N={N} -- the linear "
                      f"solve did not hit cg_tol within cg_max_iter on at least one Newton "
                      f"iteration, leaving extra, non-discretization error in this row's solution.")
            print(f"  N={N}: n_dof={row['n_dof']}, newton_iters={row['newton_iters']}, "
                  f"cg_iters={row['cg_iters']}, L2_rel={row['l2_rel_error']:.4e}, "
                  f"H1_rel={row['h1_semi_rel_error']:.4e}, energy_rel={row['energy_rel_error']:.4e}, "
                  f"wall_clock={row['wall_clock_s']:.1f}s")

        hs = [1.0 / (N - 1) for N in resolutions if N < args.fine_N]
        rate_l2, pairwise_l2 = fit_convergence_rate(hs, [r["l2_rel_error"] for r in rows])
        rate_h1, pairwise_h1 = fit_convergence_rate(hs, [r["h1_semi_rel_error"] for r in rows])
        rate_energy, pairwise_energy = fit_convergence_rate(hs, [r["energy_rel_error"] for r in rows])

        print(f"\n{order} convergence rates (log-log LEAST-SQUARES fit across ALL "
              f"{len(rows)} resolutions -- this is the number to report, not any single "
              f"pairwise ratio below):")
        print(f"  L2:     p = {rate_l2}   (pairwise: {[round(p, 2) for p in pairwise_l2]})")
        print(f"  H1:     p = {rate_h1}   (pairwise: {[round(p, 2) for p in pairwise_h1]})")
        print(f"  Energy: p = {rate_energy}   (pairwise: {[round(p, 2) for p in pairwise_energy]})")
        if any(len(pw) >= 2 and (max(pw) - min(pw)) > 1.0 for pw in [pairwise_l2, pairwise_h1, pairwise_energy]):
            print(f"  NOTE: individual pairwise rates above vary a lot (even going negative) at "
                  f"coarse/pre-asymptotic resolutions -- this is expected FEM behavior for a "
                  f"smoothly-varying (not mesh-aligned) exact field, NOT a bug. It happens because "
                  f"how well 2 particular mesh resolutions happen to align with the field's own "
                  f"oscillations varies point to point. The multi-point LEAST-SQUARES fit above "
                  f"averages this out and is the statistically meaningful number; more resolution "
                  f"points (not fewer) make it more robust.")

        report["orders"][order] = {
            "fine_reference": {"N": args.fine_N, "n_dof": fine["n_dof"],
                                "strain_energy": fine["strain_energy"],
                                "wall_clock_s": fine["wall_clock_s"]},
            "rows": rows,
            "convergence_rates": {
                "l2_fit": rate_l2, "l2_pairwise": pairwise_l2,
                "h1_semi_fit": rate_h1, "h1_semi_pairwise": pairwise_h1,
                "energy_fit": rate_energy, "energy_pairwise": pairwise_energy,
            },
        }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to {args.out_json}")


if __name__ == "__main__":
    main()
