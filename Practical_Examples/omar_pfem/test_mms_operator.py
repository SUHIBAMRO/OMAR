"""Is the operator's energy functional the same problem the FEM solver solves?

This is the check that matters for mms_operator.py. The network is trained by
minimizing Pi = U - W over the Q4 space; the Q4 finite-element solution is BY
DEFINITION the minimizer of that same functional over that same space. So:

    Pi(u_FEM) must be the lowest value of Pi anywhere in the Q4 space.

If W carried a wrong factor -- train_B1 divides its traction work by
len(top_edges), and copying that here would have been the obvious mistake --
then Pi would be minimized somewhere else, the network would train happily
toward the wrong field, and nothing in the training curve would look amiss.
This test would fail.

    python -m omar_pfem.test_mms_operator
"""
import numpy as np
import torch

from omar_pfem.materials_torch import get_material_fns
from omar_pfem.mms_study import (
    build_mesh, assemble_body_force, boundary_nodes, solve_mms, u_exact,
    DEFAULT_ALPHA, DEFAULT_BETA)
from omar_pfem.mms_operator import energy_loss, dirichlet_mask

N, MAT = 9, "neo_hookean"
DT = torch.float64


def setup():
    nodes, elements = build_mesh("Q4", N)
    _, conv = get_material_fns(MAT)
    params = conv(torch.tensor(1000.0, dtype=DT), torch.tensor(0.3, dtype=DT),
                  mode="plane_strain")
    n_el = len(elements)
    mu_e = torch.full((n_el,), float(params[0]), dtype=DT)
    lam_e = torch.full((n_el,), float(params[1]), dtype=DT)
    f = assemble_body_force(nodes, elements, "Q4", mu_e, lam_e, MAT,
                            DEFAULT_ALPHA, DEFAULT_BETA, DT).reshape(len(nodes), 2)
    xy = torch.tensor(nodes, dtype=DT)
    quad = torch.tensor(elements, dtype=torch.long)
    pn = tuple(torch.full((1, len(nodes)), float(p), dtype=DT) for p in params)
    fn, _ = get_material_fns(MAT)
    return nodes, elements, xy, quad, f, pn, fn, mu_e, lam_e


def Pi_of(xy, quad, uv, f, pn, fn):
    Pi, U, W = energy_loss(xy, quad, uv[None], f[None], pn, fn, DT)
    return float(Pi[0]), float(U[0]), float(W[0])


def main():
    nodes, elements, xy, quad, f, pn, fn, mu_e, lam_e = setup()
    mask = dirichlet_mask(nodes, DT)

    err, u_fem, _, _ = solve_mms("Q4", N, MAT, DEFAULT_ALPHA, DEFAULT_BETA,
                                 torch.device("cpu"))
    u_fem = torch.tensor(u_fem, dtype=DT)
    Pi_fem, U_fem, W_fem = Pi_of(xy, quad, u_fem, f, pn, fn)
    print(f"Pi(u_FEM)              = {Pi_fem:.12f}   (U {U_fem:.6f}, W {W_fem:.6f})")

    # the FEM solution must satisfy the same Dirichlet condition the operator
    # is constrained to, or they are not minimizing over the same space
    bnd = boundary_nodes(nodes)
    assert np.abs(u_fem.numpy()[bnd]).max() < 1e-14, \
        "the FEM solution is not zero on the boundary"
    print("PASS  the FEM solution lies in the operator's constrained space")

    # 1) the interpolant of the exact solution must not beat it
    u_star = u_exact(xy, DEFAULT_ALPHA, DEFAULT_BETA) * mask[:, None]
    Pi_star, _, _ = Pi_of(xy, quad, u_star, f, pn, fn)
    print(f"Pi(interp u*)          = {Pi_star:.12f}   "
          f"(excess {Pi_star - Pi_fem:+.3e})")
    assert Pi_star > Pi_fem, \
        "the interpolant of u* has LOWER energy than the FEM solution -- " \
        "the FEM solution is not the minimizer of this functional"
    print("PASS  u* interpolant does not beat the FEM solution")

    # 2) random admissible perturbations, both signs, several magnitudes
    torch.manual_seed(0)
    worst = None
    for eps in (1e-3, 1e-4, 1e-5):
        for trial in range(6):
            d = torch.randn_like(u_fem) * mask[:, None]
            d = d / d.norm() * u_fem.norm()
            for s in (+1.0, -1.0):
                Pi_p, _, _ = Pi_of(xy, quad, u_fem + s * eps * d, f, pn, fn)
                excess = Pi_p - Pi_fem
                worst = excess if worst is None else min(worst, excess)
                assert excess > 0, (
                    f"a perturbation LOWERED the energy (eps={eps}, sign={s}, "
                    f"excess={excess:.3e}) -- Pi is not minimized at u_FEM")
    print(f"PASS  36 admissible perturbations all raise Pi "
          f"(smallest excess {worst:.3e})")

    # 3) the quadratic signature: excess should scale as eps^2 about a
    #    minimum, which distinguishes a true stationary point from a merely
    #    small gradient
    d = torch.randn_like(u_fem) * mask[:, None]
    d = d / d.norm() * u_fem.norm()
    e1 = Pi_of(xy, quad, u_fem + 1e-4 * d, f, pn, fn)[0] - Pi_fem
    e2 = Pi_of(xy, quad, u_fem + 2e-4 * d, f, pn, fn)[0] - Pi_fem
    print(f"excess at eps and 2eps: {e1:.3e}, {e2:.3e}  ratio {e2 / e1:.3f} "
          f"(4.0 expected at a minimum)")
    assert 3.5 < e2 / e1 < 4.5, \
        f"the energy excess does not grow quadratically (ratio {e2/e1:.2f}); " \
        f"u_FEM is not a stationary point of this Pi"
    print("PASS  the excess grows quadratically -- u_FEM is a true minimum")

    # 4) the wrong-factor trap, made explicit. train_B1 divides its traction
    #    work by len(top_edges); copying that convention for a volumetric
    #    term would have been the natural mistake. Scan a scale factor along
    #    u_FEM: with the correct W the best scale is 1, and with the wrong
    #    one it is not, because a weaker load has a smaller minimizer. This
    #    also proves the checks above can actually fail.
    def best_scale(f_used):
        ss = np.linspace(0.05, 1.6, 63)
        vals = [Pi_of(xy, quad, u_fem * float(s), f_used, pn, fn)[0] for s in ss]
        return float(ss[int(np.argmin(vals))])

    s_ok = best_scale(f)
    n_top = int(np.sqrt(len(nodes))) - 1
    s_bad = best_scale(f / n_top)
    print(f"\nbest scale along u_FEM: correct W -> {s_ok:.3f}, "
          f"W wrongly divided by {n_top} -> {s_bad:.3f}")
    assert abs(s_ok - 1.0) < 0.03, \
        f"with the correct W the energy is minimized at scale {s_ok:.3f}, not 1"
    assert s_bad < 0.9, \
        "dividing W by len(top_edges) did NOT move the minimizer, so this " \
        "test cannot detect a wrongly scaled work term and proves nothing"
    print("PASS  the correct W puts the minimum at scale 1, and the wrong one "
          "visibly moves it -- the checks above can fail")

    print("\nthe operator minimizes exactly the functional the FEM solver solves")


if __name__ == "__main__":
    main()
