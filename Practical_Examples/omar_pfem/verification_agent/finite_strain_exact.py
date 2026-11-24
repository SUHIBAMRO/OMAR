"""Finite-strain exact solutions: coverage of the certified interval beyond linear
elasticity (2026-11-24).

Every exact solution so far was linear elastic. The project is about hyperelasticity, so
this module manufactures plane-strain compressible neo-Hookean problems with a known
solution: a displacement field is chosen, the body force that makes it an equilibrium
solution is derived symbolically (b = -Div P(F)), and the exact field is imposed on the
whole boundary. The field has a steep, smooth bump, the kind of local feature the B3 groove
has, with a width s and an amplitude chosen for a given peak strain:

    u_x = a y (1 - y) x (1 - x) * B,   u_y = a x (1 - x) y (1 - y) * B * 0.5,
    B = 1 + 4 exp(-((x - 0.5)^2 + (y - 0.5)^2) / s^2)

Material: W = mu/2 (tr C - 3 - 2 ln J) + lam/2 (ln J)^2, plane strain (F_33 = 1),
E = 1000, nu = 0.3 or 0.45. Cases: s = 0.3, 0.15, 0.08 (milder to steeper) x amplitude for a
peak Green-Lagrange strain near 5% or 20% x nu: 12 cases. Unit square, Q2 (9-node)
elements, n = 4, 8, 16, 32, 64 per side, Newton with 4 load steps (the body force and the
boundary data scaled together), tangent dP/dF = mu d_ik d_jl + (mu - lam ln J) Fit_il Fit_kj
+ lam Fit_ij Fit_kl.

Quantities: strain_energy (integral of W), u_probe = u_x at (0.5, 0.75), gk_avg_von_mises
(Cauchy von Mises averaged with exp(-(d/0.1)^2) about the centre). Exact values by 2D
Gauss-Legendre quadrature of the exact field (smooth integrands).

    python -m omar_pfem.verification_agent.finite_strain_exact exact
    python -m omar_pfem.verification_agent.finite_strain_exact solve
"""
import json
import os
import sys
import time

import numpy as np
import skfem as fem
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
EXACT = os.path.join(HERE, "finite_strain_exact_values.json")
LADDER = os.path.join(HERE, "finite_strain_exact_ladder.json")
E = 1000.0
LEVELS = (4, 8, 16, 32, 64)
NSTEPS = 4
ELL = 0.1
PROBE = (0.5, 0.75)
QOIS = ("strain_energy", "u_probe", "gk_avg_von_mises")


def case_list():
    out = []
    for nu in (0.3, 0.45):
        for s in (0.3, 0.15, 0.08):
            for strain in (0.05, 0.2):
                out.append({"id": f"s{int(s * 100):02d}_e{int(strain * 100):02d}_nu{int(nu * 100)}",
                            "s": s, "strain": strain, "nu": nu})
    return out


class Case:
    def __init__(self, s, strain, nu, **_):
        self.s, self.nu = s, nu
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        x, y, a = sp.symbols("x y a", real=True)
        B = 1 + 4 * sp.exp(-((x - sp.Rational(1, 2)) ** 2 + (y - sp.Rational(1, 2)) ** 2) / s ** 2)
        ux = a * y * (1 - y) * x * (1 - x) * B
        uy = a * x * (1 - x) * y * (1 - y) * B / 2
        F = sp.Matrix([[1 + sp.diff(ux, x), sp.diff(ux, y)], [sp.diff(uy, x), 1 + sp.diff(uy, y)]])
        J = F.det()
        Fit = F.inv().T
        P = self.mu * (F - Fit) + self.lam * sp.log(J) * Fit
        bx = -(sp.diff(P[0, 0], x) + sp.diff(P[0, 1], y))
        by = -(sp.diff(P[1, 0], x) + sp.diff(P[1, 1], y))
        C = F.T * F
        W = self.mu / 2 * (C.trace() + 1 - 3 - 2 * sp.log(J)) + self.lam / 2 * sp.log(J) ** 2
        sig = P * F.T / J                                    # in-plane Cauchy
        s33 = self.lam * sp.log(J) / J                       # F_33 = 1
        vm = sp.sqrt(sp.Rational(1, 2) * ((sig[0, 0] - sig[1, 1]) ** 2 + (sig[1, 1] - s33) ** 2
                                          + (s33 - sig[0, 0]) ** 2) + 3 * sig[0, 1] ** 2)
        Egl = (C - sp.eye(2)) / 2
        emax = sp.sqrt((Egl[0, 0] - Egl[1, 1]) ** 2 / 4 + Egl[0, 1] ** 2) + (Egl[0, 0] + Egl[1, 1]) / 2
        lf = lambda e: sp.lambdify((x, y, a), e, "numpy")
        self._u = lf(sp.Matrix([ux, uy]))
        self._b = lf(sp.Matrix([bx, by]))
        self._W, self._vm, self._emax = lf(W), lf(vm), lf(emax)
        g = np.linspace(0, 1, 201)
        X, Y = np.meshgrid(g, g)
        lo, hi = 0.0, 50.0                                   # amplitude for the target strain
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.max(self._emax(X, Y, mid)) > strain:
                hi = mid
            else:
                lo = mid
        self.a = 0.5 * (lo + hi)

    def u(self, x, y, frac=1.0):
        v = self._u(x, y, self.a * frac)
        return np.asarray(v[0, 0]) * np.ones_like(x), np.asarray(v[1, 0]) * np.ones_like(x)

    def body(self, x, y, frac=1.0):
        v = self._b(x, y, self.a * frac)
        return np.asarray(v[0, 0]) * np.ones_like(x), np.asarray(v[1, 0]) * np.ones_like(x)

    def exact_values(self, nq=400):
        g, w = np.polynomial.legendre.leggauss(nq)
        g, w = 0.5 * (g + 1), 0.5 * w
        X, Y = np.meshgrid(g, g, indexing="ij")
        Wq = np.outer(w, w)
        k = np.exp(-(np.hypot(X - 0.5, Y - 0.5) / ELL) ** 2)
        vm = self._vm(X, Y, self.a)
        return {"a": self.a, "strain_energy": float(np.sum(self._W(X, Y, self.a) * Wq)),
                "u_probe": float(self.u(np.array(PROBE[0]), np.array(PROBE[1]))[0]),
                "gk_avg_von_mises": float(np.sum(vm * k * Wq) / np.sum(k * Wq)),
                "max_green_strain": float(np.max(self._emax(X, Y, self.a)))}

    def solve_level(self, n):
        import pypardiso
        t0 = time.time()
        msh = fem.MeshQuad2.from_mesh(fem.MeshQuad.init_tensor(np.linspace(0, 1, n + 1),
                                                                np.linspace(0, 1, n + 1)))
        b = fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=6)
        mu, lam = self.mu, self.lam
        comp = np.zeros(b.N, int)
        comp[b.nodal_dofs[1]] = 1
        comp[b.facet_dofs[1]] = 1
        if b.interior_dofs.size:
            comp[b.interior_dofs[1]] = 1
        D = b.get_dofs().all()
        loc = b.doflocs[:, D]
        X = b.global_coordinates().value

        def kin(w):
            F = w["U"].grad + np.eye(2)[:, :, None, None]
            J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
            Fit = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J
            return F, J, Fit

        @fem.LinearForm
        def residual(v, w):
            F, J, Fit = kin(w)
            P = mu * (F - Fit) + lam * np.log(J) * Fit
            return np.einsum("ij...,ij...->...", P, v.grad) - (w["bx"] * v.value[0] + w["by"] * v.value[1])

        @fem.BilinearForm
        def tangent(du, v, w):
            F, J, Fit = kin(w)
            lnJ = np.log(J)
            dF, gv = du.grad, v.grad
            t1 = mu * np.einsum("ij...,ij...->...", dF, gv)
            t2 = (mu - lam * lnJ) * np.einsum("il...,kj...,kl...,ij...->...", Fit, Fit, dF, gv)
            t3 = lam * np.einsum("ij...,ij...->...", Fit, gv) * np.einsum("kl...,kl...->...", Fit, dF)
            return t1 + t2 + t3

        u = np.zeros(b.N)
        hist = []
        for step in range(1, NSTEPS + 1):
            frac = step / NSTEPS
            ux, uy = self.u(loc[0], loc[1], frac)
            u[D] = np.where(comp[D] == 0, ux, uy)
            bx, by = self.body(X[0], X[1], frac)
            for it in range(25):
                Uf = b.interpolate(u)
                R = fem.asm(residual, b, U=Uf, bx=bx, by=by)
                R[D] = 0.0
                nr = float(np.linalg.norm(R))
                if it > 0 and nr < 1e-9 * max(1.0, r0):
                    break
                if it == 0:
                    r0 = nr
                K = fem.asm(tangent, b, U=Uf)
                du = fem.solve(*fem.condense(K, -R, D=D),
                               solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
                u += du
            hist.append([it, nr])
        Uf = b.interpolate(u)
        F, J, Fit = kin({"U": Uf})
        C11 = F[0, 0] ** 2 + F[1, 0] ** 2
        C22 = F[0, 1] ** 2 + F[1, 1] ** 2
        W = mu / 2 * (C11 + C22 + 1 - 3 - 2 * np.log(J)) + lam / 2 * np.log(J) ** 2
        P = mu * (F - Fit) + lam * np.log(J) * Fit
        sig = np.einsum("ik...,jk...->ij...", P, F) / J
        s33 = lam * np.log(J) / J
        vm = np.sqrt(0.5 * ((sig[0, 0] - sig[1, 1]) ** 2 + (sig[1, 1] - s33) ** 2 + (s33 - sig[0, 0]) ** 2)
                     + 3 * sig[0, 1] ** 2)
        k = b.dx * np.exp(-(np.hypot(X[0] - 0.5, X[1] - 0.5) / ELL) ** 2)
        U = u[b.nodal_dofs].T
        j = int(np.argmin((msh.p[0] - PROBE[0]) ** 2 + (msh.p[1] - PROBE[1]) ** 2))
        assert np.hypot(msh.p[0, j] - PROBE[0], msh.p[1, j] - PROBE[1]) < 1e-12
        return {"n": n, "h": 1.0 / n, "n_dof": int(b.N), "newton": hist,
                "strain_energy": float((W * b.dx).sum()), "u_probe": float(U[j, 0]),
                "gk_avg_von_mises": float((vm * k).sum() / k.sum()), "min_J": float(J.min()),
                "wall_clock_s": time.time() - t0}


def exact_all():
    out = {}
    for c in case_list():
        k = Case(**c)
        out[c["id"]] = {**c, "exact": k.exact_values()}
        e = out[c["id"]]["exact"]
        print(c["id"], f"a={e['a']:.4g} max strain {e['max_green_strain']:.3f} E={e['strain_energy']:.6g}", flush=True)
    json.dump(out, open(EXACT, "w"), indent=1)


def solve_all():
    rows = json.load(open(LADDER)) if os.path.exists(LADDER) else []
    done = {(r["case"], r["n"]) for r in rows}
    for c in case_list():
        k = None
        for n in LEVELS:
            if (c["id"], n) in done:
                continue
            k = k or Case(**c)
            r = k.solve_level(n)
            r["case"] = c["id"]
            rows.append(r)
            json.dump(rows, open(LADDER, "w"), indent=1)
        print(c["id"], "done", flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    {"exact": exact_all, "solve": solve_all}.get((sys.argv[1:] or [""])[0], lambda: print(__doc__))()
