"""Code verification of torchfem's 20-node hexahedron (Q20, the B3 element)
by the method of manufactured solutions, compressible neo-Hookean, finite
strain, total Lagrangian.

Manufactured displacement on the unit cube (amplitude a = 0.05, so
|grad u| ~ 0.16):
    u_x = a sin(pi x) cos(pi y/2) sin(pi z/2)
    u_y = a sin(pi x/2) sin(pi y) cos(pi z/2)
    u_z = a cos(pi x/2) sin(pi y/2) sin(pi z)
Body force b = -Div P(F(u)), with P = dpsi/dF of the SAME energy B3 uses
(torchfem_comparison.neo_hookean_psi_3d), evaluated by automatic
differentiation and integrated with the full 3x3x3 rule; every boundary
node carries the exact displacement. Errors against the exact solution are
integrated with the 3x3x3 rule on the reference configuration.

Theory for a degree-2 element: ||u - u_h||_L2 = O(h^3), |u - u_h|_H1 = O(h^2).
Both the reduced 2x2x2 rule (torchfem's default, used for every B3 result)
and the full 3x3x3 rule are run.

    python -m omar_pfem.verification_agent.mms_q20 [--n 2,4,8,16] [--nu 0.3,0.45]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
A, E = 0.05, 1000.0
PI = np.pi


def u_exact(X):
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return torch.stack([
        A * torch.sin(PI * x) * torch.cos(PI * y / 2) * torch.sin(PI * z / 2),
        A * torch.sin(PI * x / 2) * torch.sin(PI * y) * torch.cos(PI * z / 2),
        A * torch.cos(PI * x / 2) * torch.sin(PI * y / 2) * torch.sin(PI * z)], dim=-1)


def _grad_u(X):
    """(n,3,3) exact displacement gradient du_i/dX_J."""
    return torch.func.vmap(torch.func.jacrev(u_exact))(X)


def _body_force(X, params, chunk=4096):
    """(n,3) b = -Div_X P(F(X)), by automatic differentiation."""
    from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

    def P_of_X(x):
        F = torch.eye(3, dtype=x.dtype) + torch.func.jacrev(u_exact)(x)
        return torch.func.jacrev(neo_hookean_psi_3d)(F, params)

    def b_of_X(x):
        dP = torch.func.jacrev(P_of_X)(x)          # (3,3,3): dP_iJ/dX_K
        return -torch.einsum("iJJ->i", dP)

    f = torch.func.vmap(b_of_X)
    return torch.cat([f(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def cube_q20(n, dtype=torch.float64):
    """Structured n x n x n mesh of the unit cube, Hexa1 then torchfem's
    linear_to_quadratic (straight edges, midside nodes at edge midpoints)."""
    from torchfem.elements import linear_to_quadratic
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    nodes = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
    idx = lambda i, j, k: i + (n + 1) * j + (n + 1) ** 2 * k
    el = [[idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
           idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
          for k in range(n) for j in range(n) for i in range(n)]
    n20, e20 = linear_to_quadratic(torch.tensor(nodes, dtype=dtype),
                                   torch.tensor(el, dtype=torch.long))
    return n20, e20


def solve(n, nu, quadrature, increments=5):
    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    from omar_pfem.torchfem_comparison import neo_hookean_psi_3d
    from omar_pfem.verification_agent.b3_global import _full_integration_solid
    dtype = torch.float64
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        nodes, elements = cube_q20(n, dtype)
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        params = torch.tensor([mu, lam], dtype=dtype)
        Full = _full_integration_solid()
        mat = lambda: Hyperelastic3D(psi=neo_hookean_psi_3d, params=params)
        quad = Full(nodes, elements, mat())             # 27-point rule for b and errors
        model = (Full if quadrature == "full" else Solid)(nodes, elements, mat())

        # consistent nodal body force, 3x3x3 rule
        Nq, Bq, detJ = quad.eval_shape_functions(quad.etype.ipoints)     # (g,20), (g,e,3,20), (g,e)
        w = quad.etype.iweights
        Xg = torch.einsum("gn,enD->egD", Nq, nodes[elements])            # (e,g,3)
        b = _body_force(Xg.reshape(-1, 3), params).reshape(Xg.shape)
        fe = torch.einsum("g,ge,gn,egi->eni", w, detJ, Nq, b)             # (e,20,3)
        forces = torch.zeros_like(nodes)
        forces.index_add_(0, elements.reshape(-1), fe.reshape(-1, 3))

        on_bnd = ((nodes < 1e-12) | (nodes > 1 - 1e-12)).any(dim=1)
        model.forces = forces
        model.constraints = torch.zeros_like(nodes, dtype=torch.bool)
        model.constraints[on_bnd, :] = True
        model.displacements = torch.zeros_like(nodes)
        model.displacements[on_bnd] = u_exact(nodes[on_bnd])
        t0 = time.time()
        with torch.no_grad():
            u = model.solve(increments=torch.linspace(0.0, 1.0, increments + 1), max_iter=30,
                            rtol=1e-10, atol=1e-10, method="direct", verbose=False)[0]
        elapsed = time.time() - t0

        # errors, 3x3x3 rule on the reference configuration
        uh = torch.einsum("gn,eni->egi", Nq, u[elements])
        guh = torch.einsum("geJn,eni->egiJ", Bq, u[elements])
        ue, gue = u_exact(Xg), _grad_u(Xg.reshape(-1, 3)).reshape(*Xg.shape[:2], 3, 3)
        wd = (w[:, None] * detJ).T                                         # (e,g)
        integ = lambda v: float(torch.sqrt((wd * v).sum()))
        l2 = integ(((uh - ue) ** 2).sum(-1)) / integ((ue ** 2).sum(-1))
        h1 = integ(((guh - gue) ** 2).sum((-1, -2))) / integ((gue ** 2).sum((-1, -2)))
        detF = torch.linalg.det(torch.eye(3) + guh)
        return {"n": n, "h": 1.0 / n, "nu": nu, "quadrature": quadrature,
                "n_elem": int(elements.shape[0]), "n_dof": int(3 * nodes.shape[0]),
                "L2_rel": l2, "H1_rel": h1, "min_detF": float(detF.min()),
                "elapsed_s": elapsed}
    finally:
        torch.set_default_dtype(old)


def rates(rows, key):
    return [float(np.log(a[key] / b[key]) / np.log(a["h"] / b["h"]))
            for a, b in zip(rows, rows[1:])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="2,4,8,16")
    ap.add_argument("--nu", default="0.3,0.45")
    ap.add_argument("--quadrature", default="reduced,full")
    args = ap.parse_args(argv)
    out = os.path.join(HERE, "mms_q20_result.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["n"], r["nu"], r["quadrature"]) for r in res}
    for nu in [float(v) for v in args.nu.split(",")]:
        for quadrature in args.quadrature.split(","):
            for n in [int(v) for v in args.n.split(",")]:
                if (n, nu, quadrature) in done:
                    continue
                r = solve(n, nu, quadrature)
                res.append(r)
                json.dump(res, open(out, "w"), indent=1)
                print(f"nu={nu} {quadrature:<7} n={n:>2} ({r['n_elem']} el, {r['n_dof']} dof, "
                      f"{r['elapsed_s']:.0f}s): L2 {r['L2_rel']:.3e}  H1 {r['H1_rel']:.3e}  "
                      f"min detF {r['min_detF']:.3f}", flush=True)
            rows = sorted((r for r in res if r["nu"] == nu and r["quadrature"] == quadrature),
                          key=lambda r: r["n"])
            if len(rows) > 1:
                print(f"nu={nu} {quadrature}: L2 rates {[round(p, 2) for p in rates(rows, 'L2_rel')]}"
                      f" (theory 3), H1 rates {[round(p, 2) for p in rates(rows, 'H1_rel')]}"
                      f" (theory 2)", flush=True)


if __name__ == "__main__":
    main()
