"""Hyperelastic (compressible neo-Hookean, plane strain) problems in the autonomous pipeline
(2026-11-25).

The project is about hyperelasticity, but the autonomous loop of round O solved only linear
elasticity. This module adds the neo-Hookean material to the same pipeline (pipeline2d.py),
with the same domain, boundary-condition and quantity conventions:

  W = mu/2 (tr C - 3 - 2 ln J) + lam/2 (ln J)^2,  plane strain (F_33 = 1), Q2 or P2 elements;
  P = mu (F - F^-T) + lam ln J F^-T;  tangent dP/dF = mu d_ik d_jl + (mu - lam ln J) Fit_il Fit_kj
  + lam Fit_ij Fit_kl (the form verified in finite_strain_exact.py).

Loads: tractions are DEAD loads (a constant nominal traction on the reference edge); a
pressure (a follower load) and plane stress are not implemented and are refused by validate().

Load stepping is the agent's: it starts with 4 equal steps; a step whose Newton iteration does
not converge (relative residual 1e-10 within 25 iterations) or produces J <= 0 is restarted with
twice as many steps, up to 64. Every solve records its steps, Newton iterations and min J.

Corner exponents: the singular exponents of the LINEARIZED problem are used and labelled as
such: near a singular point a neo-Hookean solution eventually leaves the small-strain range, so
the linearized order is a hypothesis for the practical range of meshes, not a theorem.

Code verification: ManufacturedNH builds a smooth manufactured field on the problem's domain
with a peak Green-Lagrange strain of about 10%, its body force -Div P and the nominal traction
P N symbolically, and mms_auto checks the L2 and H1 rates with this solver.
"""
import math
import time

import numpy as np
import skfem as fem
import sympy as sp

from omar_pfem.verification_agent import pipeline2d as P2


def validate(desc):
    err = P2.validate(desc)
    if desc.get("plane", "strain") != "strain":
        err.append("neo-Hookean: only plane strain is implemented")
    if any(b.get("type") == "pressure" for b in desc.get("bcs", [])):
        err.append("neo-Hookean: pressure is a follower load and is not implemented; only dead "
                   "(constant nominal) tractions")
    if any(b.get("type") == "clamped" and b.get("u") for b in desc.get("bcs", [])):
        err.append("neo-Hookean: prescribed non-zero displacements are not implemented")
    return err


def _kin(grad):
    F = grad + np.eye(2)[:, :, None, None]
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    Fit = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J
    return F, J, Fit


class Problem2DNH(P2.Problem2D):
    material = "neo_hookean"

    def __init__(self, desc):
        errs = validate(desc)
        if errs:
            raise ValueError("; ".join(errs))
        super().__init__(desc)
        for c in self.corners:
            c["exponent_basis"] = "linearized (small-strain) theory"

    def manufactured(self):
        return ManufacturedNH(self)

    def _solve(self, n, plan, element, mms):
        import pypardiso
        t0 = time.time()
        mq = self.mesh(n, plan)
        if element == "Q2":
            msh, el = fem.MeshQuad2.from_mesh(mq), fem.ElementQuad2()
        else:
            msh, el = fem.MeshTri2.from_mesh(mq.to_meshtri()), fem.ElementTriP2()
        mu, lam = self.mu, self.lame
        E2 = fem.ElementVector(el)
        iorder = 6
        b = fem.Basis(msh, E2, intorder=iorder)
        X = b.global_coordinates().value
        mid = msh.p[:, msh.facets].mean(axis=1) / self.a
        bf = msh.boundary_facets()
        used = np.zeros(bf.size, bool)
        D, comp_val = [], []
        neu = []                                   # (FacetBasis, traction function(frac, w))
        dl = b.doflocs
        for bc in self.desc["bcs"]:
            seg = [[c / self.a for c in p] for p in bc["segment"]]
            m = np.array([P2._on_seg(mid[0, f_], mid[1, f_], seg) for f_ in bf])
            sel = bf[m & ~used]
            used |= m
            if sel.size == 0:
                continue
            if bc["type"] == "clamped":
                d = b.get_dofs(facets=sel).all()
                D.append(d)
                if mms is not None:
                    is_x = np.isin(d, b.get_dofs(facets=sel).all("u^1"))
                    comp_val.append((d, lambda fr, d=d, is_x=is_x: np.where(
                        is_x, mms.u(dl[0, d], dl[1, d], fr)[0], mms.u(dl[0, d], dl[1, d], fr)[1])))
            elif bc["type"] == "symmetry":
                vertical = abs(seg[0][0] - seg[1][0]) < 1e-12
                d = b.get_dofs(facets=sel).all("u^1" if vertical else "u^2")
                D.append(d)
                fb = fem.FacetBasis(msh, E2, intorder=iorder, facets=sel)
                if mms is not None:
                    comp_val.append((d, lambda fr, d=d, v=vertical: mms.u(dl[0, d], dl[1, d], fr)[0 if v else 1]))
                    other = 1 if vertical else 0

                    def tf(fr, w, other=other):
                        t = mms.traction(w.x[0], w.x[1], w.n[0], w.n[1], fr)
                        z = 0 * t[0]
                        return (t[0], z) if other == 0 else (z, t[1])
                    neu.append((fb, tf))
            else:
                fb = fem.FacetBasis(msh, E2, intorder=iorder, facets=sel)
                if mms is not None:
                    neu.append((fb, lambda fr, w: mms.traction(w.x[0], w.x[1], w.n[0], w.n[1], fr)))
                else:
                    tx, ty = bc["t"]
                    neu.append((fb, lambda fr, w, tx=tx, ty=ty: (fr * tx + 0 * w.x[0], fr * ty + 0 * w.x[0])))
        if mms is not None and (~used).any():
            fb = fem.FacetBasis(msh, E2, intorder=iorder, facets=bf[~used])
            neu.append((fb, lambda fr, w: mms.traction(w.x[0], w.x[1], w.n[0], w.n[1], fr)))
        D = np.unique(np.concatenate(D)) if D else np.array([], int)

        @fem.LinearForm
        def internal(v, w):
            F, J, Fit = _kin(w["U"].grad)
            Pk = mu * (F - Fit) + lam * np.log(J) * Fit
            return np.einsum("ij...,ij...->...", Pk, v.grad)

        @fem.BilinearForm
        def tangent(du, v, w):
            F, J, Fit = _kin(w["U"].grad)
            lnJ = np.log(J)
            dF, gv = du.grad, v.grad
            return (mu * np.einsum("ij...,ij...->...", dF, gv)
                    + (mu - lam * lnJ) * np.einsum("il...,kj...,kl...,ij...->...", Fit, Fit, dF, gv)
                    + lam * np.einsum("ij...,ij...->...", Fit, gv) * np.einsum("kl...,kl...->...", Fit, dF))

        def external(fr):
            f = np.zeros(b.N)
            if mms is not None:
                bx, by = mms.body(X[0], X[1], fr)
                f += fem.asm(fem.LinearForm(lambda v, w: bx * v.value[0] + by * v.value[1]), b)
            for fb, tf in neu:
                f += fem.asm(fem.LinearForm(lambda v, w, tf=tf: tf(fr, w)[0] * v.value[0]
                                            + tf(fr, w)[1] * v.value[1]), fb)
            return f

        def attempt(nsteps):
            u = np.zeros(b.N)
            hist = []
            for s in range(1, nsteps + 1):
                fr = s / nsteps
                for d, val in comp_val:
                    u[d] = val(fr)
                fext = external(fr)
                r0, last_du = None, np.inf
                for it in range(26):
                    Uf = b.interpolate(u)
                    _, J, _ = _kin(Uf.grad)
                    if not np.all(J > 0):
                        return None, hist + [[s, it, "J<=0"]]
                    R = fem.asm(internal, b, U=Uf) - fext
                    R[D] = 0.0
                    nr = float(np.linalg.norm(R))
                    r0 = nr if r0 is None else r0
                    # converged: residual reduced 1e10 times, or the last Newton update is at
                    # round-off level relative to the solution (a tiny load has a residual floor
                    # set by round-off, not by the tolerance; found by the small-load test)
                    if it > 0 and (nr <= 1e-10 * max(r0, 1e-30) or last_du <= 1e-12 * max(
                            float(np.linalg.norm(u)), 1e-30)):
                        break
                    if it == 25:
                        return None, hist + [[s, it, "no convergence"]]
                    K = fem.asm(tangent, b, U=Uf)
                    du = fem.solve(*fem.condense(K, -R, D=D),
                                   solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
                    u += du
                    last_du = float(np.linalg.norm(du))
                hist.append([s, it, nr])
            return u, hist

        nsteps, tried = 4, []
        while True:
            u, hist = attempt(nsteps)
            tried.append(nsteps)
            if u is not None or nsteps >= 64:
                break
            nsteps *= 2
        row = {"n": n, "h": self.a / n, "element": element, "plan": plan,
               "n_elem": int(msh.t.shape[1]), "n_dof": int(b.N), "load_steps": nsteps,
               "load_steps_tried": tried, "newton": hist, "material": "neo_hookean"}
        if u is None:
            row["failed"] = True
            row["wall_clock_s"] = time.time() - t0
            return row
        Uf = b.interpolate(u)
        F, J, Fit = _kin(Uf.grad)
        row["min_detF"] = float(J.min())
        if mms is not None:
            ue = mms.u(X[0], X[1], 1.0)
            ge = mms.grad(X[0], X[1])
            e0 = sum((Uf.value[i] - ue[i]) ** 2 for i in range(2))
            e1 = sum((Uf.grad[i, j] - ge[i][j]) ** 2 for i in range(2) for j in range(2))
            row["L2"] = float(np.sqrt((e0 * b.dx).sum()))
            row["H1"] = float(np.sqrt((e1 * b.dx).sum()))
            row["wall_clock_s"] = time.time() - t0
            return row
        C11 = F[0, 0] ** 2 + F[1, 0] ** 2
        C22 = F[0, 1] ** 2 + F[1, 1] ** 2
        W = mu / 2 * (C11 + C22 + 1 - 3 - 2 * np.log(J)) + lam / 2 * np.log(J) ** 2
        row["strain_energy"] = float((W * b.dx).sum())
        Pk = mu * (F - Fit) + lam * np.log(J) * Fit
        sig = np.einsum("ik...,jk...->ij...", Pk, F) / J
        s33 = lam * np.log(J) / J
        vm = np.sqrt(0.5 * ((sig[0, 0] - sig[1, 1]) ** 2 + (sig[1, 1] - s33) ** 2
                            + (s33 - sig[0, 0]) ** 2) + 3 * sig[0, 1] ** 2)
        U = u[b.nodal_dofs].T
        for name, (px, py, comp) in self.desc.get("points", {}).items():
            j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
            if math.hypot(msh.p[0, j] - px, msh.p[1, j] - py) > 1e-9 * self.a:
                raise ValueError(f"{name} is not a mesh node")
            row[name] = float(U[j, 0 if comp == "x" else 1])
        wq = b.dx
        for k, c in enumerate(self.corners):
            d = np.hypot(X[0] - c["x"] * self.a, X[1] - c["y"] * self.a)
            kk = wq * np.exp(-(d / (P2.ELL_CELLS * self.a)) ** 2)
            row[f"corner{k}_gk_avg_von_mises"] = float((vm * kk).sum() / kk.sum())
            row[self.local_keys[k]] = float((W * wq)[d < P2.RHO_CELLS * self.a].sum())
        row["peak_von_mises"] = float(vm.max())
        row["wall_clock_s"] = time.time() - t0
        return row


class ManufacturedNH:
    """Smooth field u = A [sin(kx x + 0.3) cos(ky y + 0.2), cos(kx x + 0.1) sin(ky y + 0.4)],
    k ~ 1.3 pi / box size, amplitude for a peak Green-Lagrange strain of about 10%; its load
    is scaled by the load fraction through the amplitude (u(frac) = frac u), so every load step
    has an exact solution."""

    def __init__(self, prob, strain=0.10):
        x, y, a = sp.symbols("x y a", real=True)
        x0, x1, y0, y1 = [c * prob.a for c in prob.box]
        Lx, Ly = x1 - x0, y1 - y0
        kx, ky = 1.3 * sp.pi / Lx, 1.3 * sp.pi / Ly
        ux = a * sp.sin(kx * x + sp.Rational(3, 10)) * sp.cos(ky * y + sp.Rational(1, 5))
        uy = a * sp.cos(kx * x + sp.Rational(1, 10)) * sp.sin(ky * y + sp.Rational(2, 5))
        mu, lam = prob.mu, prob.lame
        F = sp.Matrix([[1 + sp.diff(ux, x), sp.diff(ux, y)], [sp.diff(uy, x), 1 + sp.diff(uy, y)]])
        J = F.det()
        Fit = F.inv().T
        Pm = mu * (F - Fit) + lam * sp.log(J) * Fit
        bx = -(sp.diff(Pm[0, 0], x) + sp.diff(Pm[0, 1], y))
        by = -(sp.diff(Pm[1, 0], x) + sp.diff(Pm[1, 1], y))
        Egl = (F.T * F - sp.eye(2)) / 2
        emax = sp.sqrt((Egl[0, 0] - Egl[1, 1]) ** 2 / 4 + Egl[0, 1] ** 2) + (Egl[0, 0] + Egl[1, 1]) / 2
        lf = lambda e: sp.lambdify((x, y, a), e, "numpy")
        self._u = [lf(ux), lf(uy)]
        self._g = [[lf(sp.diff(ux, x)), lf(sp.diff(ux, y))], [lf(sp.diff(uy, x)), lf(sp.diff(uy, y))]]
        self._P = [[lf(Pm[i, j]) for j in range(2)] for i in range(2)]
        self._b = [lf(bx), lf(by)]
        em = lf(emax)
        g = np.linspace(0, 1, 121)
        Xg, Yg = np.meshgrid(x0 + g * Lx, y0 + g * Ly)
        lo, hi = 0.0, 0.5 * max(Lx, Ly)
        for _ in range(60):                       # amplitude for the target strain
            midv = 0.5 * (lo + hi)
            lo, hi = (lo, midv) if np.nanmax(np.abs(em(Xg, Yg, midv))) > strain else (midv, hi)
        self.amp = 0.5 * (lo + hi)
        self.field = {"ux": str(ux), "uy": str(uy), "amplitude": self.amp, "peak_strain": strain}

    @staticmethod
    def _v(fn, x, y, aa):
        return np.asarray(fn(x, y, aa), float) * np.ones_like(np.asarray(x, float))

    def u(self, x, y, frac=1.0):
        return [self._v(f, x, y, self.amp * frac) for f in self._u]

    def grad(self, x, y):
        return [[self._v(f, x, y, self.amp) for f in r] for r in self._g]

    def body(self, x, y, frac=1.0):
        return [self._v(f, x, y, self.amp * frac) for f in self._b]

    def traction(self, x, y, nx, ny, frac=1.0):
        Pv = [[self._v(f, x, y, self.amp * frac) for f in r] for r in self._P]
        return [Pv[0][0] * nx + Pv[0][1] * ny, Pv[1][0] * nx + Pv[1][1] * ny]
