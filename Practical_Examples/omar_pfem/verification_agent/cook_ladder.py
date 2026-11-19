"""Cook's membrane: a HELD-OUT benchmark WITH a corner singularity.

Tapered panel, vertices (0,0), (48,44), (48,60), (0,44); left edge x = 0
clamped; right edge x = 48 loaded by a uniform dead shear traction t_y =
TAU (total force 16 TAU) in 10 increments; top and bottom edges free.
Plane-strain compressible neo-Hookean (the B1/B2 energy), E = 1000,
nu = 0.3. Q4 elements, 2x2 Gauss.

Solver: scikit-fem with the residual/tangent of independent_solver.py
(which reproduces the project's B1 and B2 solvers to ~1e-13 on the same
discrete problems). The project's own 2D solvers are written for the B1/B2
geometries only.

Corners: the clamped left edge meets the free bottom edge at (0,0) (interior
angle 47.5 deg) and the free top edge at (0,44) (108.4 deg). Williams
eigenproblem (singularity.py, clamped/free, plane strain, nu = 0.3): no
singular exponent below 1 at 47.5 deg; lam = 0.6442 at 108.4 deg. The
right-hand corners are free/free with angles < 180 deg (regular).

Levels: N x N elements on the parametric square, N = 4 .. 256 (h = 1/N).
Strategies at N = 64: grading toward the clamped edge and both left
corners, s -> s - a s (1 - s), t -> t - a sin(2 pi t) / (2 pi), a = 0.5, 0.8
(corner elements 2x and 5x smaller, same element count).

    python -m omar_pfem.verification_agent.cook_ladder
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse.linalg as spla
import skfem as fem

from omar_pfem.verification_agent.independent_solver import (_F, _det, a_form, energy_form,
                                                            lame, pk1, r_form)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cook_ladder.json")
E_MOD, NU, TAU = 1000.0, 0.3, 5.0
TIP = (48.0, 60.0)
REGION = ((24.0, 40.0), 4.0)          # sharp disc
KERNEL_ELL = 3.0                      # Gaussian weight exp(-(d/l)^2), same centre


def cook_corner_exponent(nu=NU):
    from omar_pfem.verification_agent.singularity import smallest_exponent
    lams = [smallest_exponent(a, "clamped", "free", nu, "strain")
            for a in (np.pi / 2 - np.arctan(44 / 48), np.pi / 2 + np.arctan(16 / 48))]
    return min(l for l in lams if l is not None)


def mesh(N, a=0.0):
    m = fem.MeshQuad.init_tensor(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))

    def st(p):
        s = p[0] - a * p[0] * (1 - p[0])
        t = p[1] - a * np.sin(2 * np.pi * p[1]) / (2 * np.pi)
        return s, t
    return m.morphed(lambda p: 48.0 * st(p)[0],
                     lambda p: (1 - st(p)[1]) * 44.0 * st(p)[0]
                     + st(p)[1] * (44.0 + 16.0 * st(p)[0]))


def solve_level(N, a=0.0, nsteps=10, tol=1e-9, newton_max=30):
    m = mesh(N, a)
    ib = fem.Basis(m, fem.ElementVector(fem.ElementQuad1()), intorder=2)
    mu, lam = lame(np.full((m.t.shape[1], ib.X.shape[1]), E_MOD), NU)
    right = fem.FacetBasis(m, fem.ElementVector(fem.ElementQuad1()),
                           facets=m.facets_satisfying(lambda x: np.isclose(x[0], 48.0)),
                           intorder=2)
    fext = fem.asm(fem.LinearForm(lambda v, w: TAU * v.value[1]), right)
    fixed = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    u = ib.zeros()
    t0, hist = time.time(), []
    for step in range(1, nsteps + 1):
        f = step / nsteps * fext
        prev = np.inf
        for it in range(newton_max):
            w = dict(u=ib.interpolate(u), mu=mu, lam=lam)
            R = fem.asm(r_form, ib, **w) - f
            Rn = np.linalg.norm(np.delete(R, fixed))
            if it > 0 and (Rn < tol or (Rn < 1e-7 and Rn > 0.5 * prev)):
                break
            prev = Rn
            K = fem.asm(a_form, ib, **w)
            u = u + fem.solve(*fem.condense(K, -R, D=fixed), solver=spla.spsolve)
        else:
            raise RuntimeError(f"Newton did not converge at step {step} (|R| = {Rn:.2e})")
        hist.append((it, float(Rn)))
    sec = time.time() - t0
    w = dict(u=ib.interpolate(u), mu=mu, lam=lam)
    F = _F(w["u"].grad)
    Pn = np.sqrt((pk1(F, mu, lam) ** 2).sum(axis=(0, 1)))
    x, wq = ib.global_coordinates().value, ib.dx
    (cx, cy), r = REGION
    d = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2)
    k = wq * np.exp(-(d / KERNEL_ELL) ** 2)
    U = u[ib.nodal_dofs].T
    j = int(np.argmin((m.p[0] - TIP[0]) ** 2 + (m.p[1] - TIP[1]) ** 2))
    assert np.hypot(*(m.p[:, j] - np.array(TIP))) < 1e-9
    ipk = np.unravel_index(np.argmax(Pn), Pn.shape)
    return {"N": N, "grading": a, "h": 1.0 / N, "n_elem": int(m.t.shape[1]),
            "n_dof": int(ib.N), "wall_clock_s": sec, "newton": hist,
            "min_detF": float(_det(F).min()),
            "min_elem_size": float(np.sqrt(ib.dx.sum(1).min())),   # sqrt of min area
            "u_tip": float(U[j, 0]), "v_tip": float(U[j, 1]),
            "max_disp_mag": float(np.sqrt((U ** 2).sum(1)).max()),
            "strain_energy": float(fem.asm(energy_form, ib, **w)),
            "interior_avg_PK1_frob": float((Pn * wq)[d < r].sum() / wq[d < r].sum()),
            "interior_gk_PK1_frob": float((Pn * k).sum() / k.sum()),
            "interior_region_gauss_points": int((d < r).sum()),
            "peak_PK1_frob": float(Pn.max()),
            "peak_location": [float(x[0][ipk]), float(x[1][ipk])]}


def run_ladder(Ns, a=0.0):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["N"], r["grading"]) for r in rows}
    for N in Ns:
        if (N, a) in done:
            continue
        row = solve_level(N, a)
        rows.append(row)
        rows.sort(key=lambda r: (r["grading"], r["N"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"N={N:<4} a={a} ({row['wall_clock_s']:.1f}s) " + "  ".join(
            f"{k}={v:.8g}" for k, v in row.items() if isinstance(v, float)
            and k not in ("h", "grading", "wall_clock_s")), flush=True)
    return sorted((r for r in rows if r["N"] in Ns and r["grading"] == a), key=lambda r: r["N"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="4,8,16,32,64,128,256")
    ap.add_argument("--strategies", default="0.5,0.8")
    args = ap.parse_args(argv)
    run_ladder([int(n) for n in args.Ns.split(",")])
    for a in [float(v) for v in args.strategies.split(",") if v]:
        run_ladder([64], a)


if __name__ == "__main__":
    main()
