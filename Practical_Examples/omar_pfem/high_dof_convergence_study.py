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
    elements; u_ref (known only at the fine reference mesh's nodes) is
    evaluated at each coarse Gauss point via linear interpolation
    (scipy.interpolate.LinearNDInterpolator) over the fine mesh's nodes.
  - H1 semi-norm: ||e||_H1 = sqrt( integral_Omega |grad(e)|^2 dOmega ).
    grad(u_h) at each coarse Gauss point comes directly from the coarse
    element's own shape-function gradients (exact for that element).
    grad(u_ref) at the same points is obtained by CENTERED FINITE
    DIFFERENCING of the same linear interpolator used for the L2 norm
    (step size --fd_h) -- an approximation (interpolant gradients are only
    piecewise-constant otherwise), stated explicitly here and in the
    output, rather than a fully mesh-consistent FE gradient recovery on
    the fine mesh (which would require point-location within specific
    fine elements).
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
import time

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator

from omar_pfem.mesh_convergence import AnalyticFieldB1, AnalyticFieldB2
from omar_pfem.matrix_free_solver import (
    solve_matrix_free, element_energy_order_agnostic, precompute_shape_data, _gauss_2x2,
)
from omar_pfem.data.q9_element import generate_grid_Q9, gauss_3x3, shape_Q9
from omar_pfem.data.fem_core import shape_Q4
from omar_pfem.materials_torch import get_material_fns as get_material_fns_torch


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
            # Polar-mapped Q9 mesh: build the Q9 reference grid in (theta, r)
            # then map to Cartesian, mirroring data_generate_B2.generate_grid_Q4_ring's
            # own polar-to-Cartesian construction.
            theta_max = np.pi / 2
            nodes_polar, elements = generate_grid_Q9(theta_max, R_out - R_in, N, N)
            theta = nodes_polar[:, 0]
            r = nodes_polar[:, 1] + R_in
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
              use_jacobi=True, cg_max_iter=2000, verbose=False):
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
        device=device, dtype=dtype, verbose=verbose)
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
        "n_dof": ndof, "n_free_dof": len(free_dofs), "strain_energy": float(U.item()),
        "wall_clock_s": wall_s, "stats": stats,
    }


def gauss_points_and_weights_physical(nodes, elements, order):
    """Physical (x,y) coordinates, detJ, and weight of every Gauss point of
    every element in a mesh -- used both to integrate the L2/H1 errors and
    to evaluate u_h / grad(u_h) there via the coarse mesh's OWN shape
    functions (exact, since these are that mesh's own quadrature points)."""
    shape_fn = shape_Q4 if order == "Q4" else __import__(
        "omar_pfem.data.q9_element", fromlist=["shape_Q9"]).shape_Q9
    gauss = _gauss_2x2() if order == "Q4" else gauss_3x3()
    Xe_all = nodes[elements]  # (Q, n_local, 2)
    pts, detJs, ws, Ns, dN_dXs, elem_idx = [], [], [], [], [], []
    for qi in range(len(elements)):
        Xe = Xe_all[qi]
        for (xi, eta, w) in gauss:
            N, dN_dxi = shape_fn(xi, eta)
            J0 = dN_dxi.T @ Xe
            detJ0 = np.linalg.det(J0)
            invJ0 = np.linalg.inv(J0)
            dN_dX = dN_dxi @ invJ0
            phys = N @ Xe
            pts.append(phys); detJs.append(detJ0); ws.append(w)
            Ns.append(N); dN_dXs.append(dN_dX); elem_idx.append(qi)
    return (np.array(pts), np.array(detJs), np.array(ws),
            np.array(Ns), np.array(dN_dXs), np.array(elem_idx))


def compute_l2_h1_errors(coarse, fine, order, fd_h=1e-5):
    nodes_c, elements_c, u_c = coarse["nodes"], coarse["elements"], coarse["u"]
    nodes_f, u_f = fine["nodes"], fine["u"]

    interp_u = LinearNDInterpolator(nodes_f, u_f[:, 0])
    interp_v = LinearNDInterpolator(nodes_f, u_f[:, 1])

    pts, detJs, ws, Ns, dN_dXs, elem_idx = gauss_points_and_weights_physical(nodes_c, elements_c, order)

    u_ref_at_gp = np.stack([interp_u(pts), interp_v(pts)], axis=1)
    # u_h at each Gauss point: N . u_c over that Gauss point's own element
    u_c_per_elem = u_c[elements_c]  # (n_elements, n_local, 2)
    u_h_at_gp = np.einsum("qa,qad->qd", Ns, u_c_per_elem[elem_idx])

    e = u_h_at_gp - u_ref_at_gp
    l2_sq_integrand = np.sum(e ** 2, axis=1) * detJs * ws
    l2_norm = np.sqrt(np.sum(l2_sq_integrand))

    grad_u_h = np.einsum("qad,qak->qdk", u_c_per_elem[elem_idx], dN_dXs)  # (Q, 2, 2) = du_i/dx_j

    def fd_grad(interp, pts):
        gx = (interp(pts + np.array([fd_h, 0])) - interp(pts - np.array([fd_h, 0]))) / (2 * fd_h)
        gy = (interp(pts + np.array([0, fd_h])) - interp(pts - np.array([0, fd_h]))) / (2 * fd_h)
        return np.stack([gx, gy], axis=1)

    grad_u_ref = np.stack([fd_grad(interp_u, pts), fd_grad(interp_v, pts)], axis=1)  # (Q,2,2)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    resolutions = [int(n) for n in args.resolutions.split(",") if n.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    report = {"geometry": args.geometry, "material": args.material, "fine_N": args.fine_N,
              "device": str(device), "orders": {}}

    for order in orders:
        print(f"\n{'='*90}\nORDER = {order}\n{'='*90}")
        print(f"Solving common fine reference at N={args.fine_N}...")
        fine = solve_one(args.geometry, order, args.fine_N, args.material, device, dtype,
                          args.cg_tol, args.newton_tol, use_jacobi=not args.no_jacobi,
                          cg_max_iter=args.cg_max_iter, verbose=False)
        print(f"  Fine reference: n_dof={fine['n_dof']}, strain_energy={fine['strain_energy']:.6e}, "
              f"wall_clock={fine['wall_clock_s']:.1f}s, "
              f"Newton iters={fine['stats']['newton_iters_total']}, "
              f"CG iters={fine['stats']['cg_iters_total']}, "
              f"CG failures={fine['stats']['cg_failures']}")

        rows = []
        for N in resolutions:
            if N >= args.fine_N:
                continue
            print(f"\nSolving N={N} ({order})...")
            coarse = solve_one(args.geometry, order, N, args.material, device, dtype,
                                args.cg_tol, args.newton_tol, use_jacobi=not args.no_jacobi,
                                cg_max_iter=args.cg_max_iter, verbose=False)
            errs = compute_l2_h1_errors(coarse, fine, order)
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
            print(f"  N={N}: n_dof={row['n_dof']}, L2_rel={row['l2_rel_error']:.4e}, "
                  f"H1_rel={row['h1_semi_rel_error']:.4e}, energy_rel={row['energy_rel_error']:.4e}, "
                  f"wall_clock={row['wall_clock_s']:.1f}s")

        hs = [1.0 / (N - 1) for N in resolutions if N < args.fine_N]
        rate_l2, pairwise_l2 = fit_convergence_rate(hs, [r["l2_rel_error"] for r in rows])
        rate_h1, pairwise_h1 = fit_convergence_rate(hs, [r["h1_semi_rel_error"] for r in rows])
        rate_energy, pairwise_energy = fit_convergence_rate(hs, [r["energy_rel_error"] for r in rows])

        print(f"\n{order} convergence rates (log-log fit across all resolutions):")
        print(f"  L2:     p = {rate_l2}")
        print(f"  H1:     p = {rate_h1}")
        print(f"  Energy: p = {rate_energy}")

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
