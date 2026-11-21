"""FEM against an isogeometric-type discretization on the B3 groove (3D,
finite-strain neo-Hookean): the comparison Professor Rabczuk asked for.

Discretization (written from scratch; no IGA library): trivariate B-splines of
degree p (C^(p-1)) in the parameters (theta, t, z) of the EXACT analytic map of
the half bushing,
    r = R_in(z) + t (R_out - R_in(z)),  x = r cos(theta), y = r sin(theta), z,
theta in [0, pi] (symmetry plane y = 0), t in [0, 1] (core to housing), z in
[0, 1]; R_in(z) is the raised-cosine groove profile of data_generate_B3 (C1:
its curvature jumps at z = 0.5 -/+ 0.15, where knots are placed). Smooth
splines and exact geometry as in IGA, but the geometry is the analytic map,
not a NURBS fit (the raised cosine is not a conic).

Same problem as the Q20 family: compressible neo-Hookean (E = 1000, nu = 0.45),
outer surface fixed, inner surface rigidly rotated by phi = 0.05 about the y
axis through (0, 0, 0.5) (Dirichlet data L2-projected onto the face splines),
u_y = 0 on the symmetry faces, end faces free; Newton with a consistent
predictor, energy, residual and tangent by automatic differentiation.

QoIs as mixed_b3.py (same definitions): reaction moment (work-conjugate:
reactions times the projected rotation field), strain energy, Gaussian averages
(weight exp(-(d / 2R0)^2), all quadrature points, and truncated at d < 0.25 as
in the B3 adapter), the sharp R0 average and the maximum over quadrature points
within R0; and, which only a smooth discretization defines, the stress AT the
groove-root point (theta = 0, t = 0, z = 0.5).

    python -m omar_pfem.verification_agent.iga_b3 [--levels 1,2,3] [--p 2]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.interpolate import BSpline
from torch.func import grad, jacfwd, vmap

from omar_pfem.data.data_generate_B3 import groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH
from omar_pfem.data.solve_quadratic_hex_groove_stress import E, LZ, NU, PHI, R_IN0, R_OUT

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "iga_b3_result.json")
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
ZC, W, D = LZ / 2.0, GROOVE_HALF_WIDTH, GROOVE_DEPTH
# spans (theta, t, z); n_z a multiple of 20 so that z = 0.35, 0.5, 0.65 are knots
LEVELS = {1: (12, 6, 20), 2: (24, 12, 40), 3: (36, 18, 60), 4: (48, 24, 80)}


def R_in(z):
    r = np.full_like(z, R_IN0)
    b = np.abs(z - ZC) < W
    r[b] = R_IN0 - D / 2 * (1 + np.cos(np.pi * (z[b] - ZC) / W))
    return r


def dR_in(z):
    r = np.zeros_like(z)
    b = np.abs(z - ZC) < W
    r[b] = D / 2 * np.pi / W * np.sin(np.pi * (z[b] - ZC) / W)
    return r


def geometry(th, t, z):
    """x and the parametric Jacobian d x / d(theta, t, z), (..., 3, 3)."""
    ri, dri = R_in(z), dR_in(z)
    r = ri + t * (R_OUT - ri)
    rt, rz = R_OUT - ri, dri * (1 - t)
    c, s = np.cos(th), np.sin(th)
    x = np.stack([r * c, r * s, z], -1)
    J = np.zeros(th.shape + (3, 3))
    J[..., 0, 0], J[..., 0, 1], J[..., 0, 2] = -r * s, rt * c, rz * c
    J[..., 1, 0], J[..., 1, 1], J[..., 1, 2] = r * c, rt * s, rz * s
    J[..., 2, 2] = 1.0
    return x, J


class Basis1D:
    """Open-knot B-splines of degree p on [a, b] with n uniform spans."""

    def __init__(self, a, b, n, p):
        self.edges = np.linspace(a, b, n + 1)
        self.t = np.concatenate([[a] * p, self.edges, [b] * p])
        self.n, self.p, self.nb = n, p, n + p
        self.spl = BSpline(self.t, np.eye(self.nb), p, extrapolate=False)
        self.dspl = self.spl.derivative()

    def eval(self, x):
        x = np.clip(x, self.edges[0], self.edges[-1] - 1e-14 * (self.edges[-1] - self.edges[0]))
        return np.nan_to_num(self.spl(x)), np.nan_to_num(self.dspl(x))

    def span_data(self, nq):
        g, gw = np.polynomial.legendre.leggauss(nq)
        e = self.edges
        h = e[1:] - e[:-1]
        x = 0.5 * (e[:-1, None] + e[1:, None]) + 0.5 * h[:, None] * g
        w = 0.5 * h[:, None] * gw
        v, d = self.eval(x.ravel())
        idx = np.arange(self.n)[:, None] + np.arange(self.p + 1)[None, :]   # active functions
        v = v.reshape(self.n, nq, self.nb)
        d = d.reshape(self.n, nq, self.nb)
        V = np.take_along_axis(v, idx[:, None, :].repeat(nq, 1), 2)
        Dv = np.take_along_axis(d, idx[:, None, :].repeat(nq, 1), 2)
        return x, w, V, Dv, idx


def det3(F):
    return (F[..., 0, 0] * (F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1])
            - F[..., 0, 1] * (F[..., 1, 0] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 0])
            + F[..., 0, 2] * (F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0]))


def psi(F):
    lnJ = torch.log(det3(F))
    return MU / 2 * ((F ** 2).sum((-2, -1)) - 3 - 2 * lnJ) + LAM / 2 * lnJ ** 2


def fbar(F, proj):
    """F_bar = (theta / J)^(1/3) F, theta = proj @ J (proj: element-wise L2 projection
    of J onto linear polynomials, as Q20/P1 in mixed_b3.py); identity if proj is None."""
    if proj is None:
        return F
    J = det3(F)
    theta = torch.einsum("...qr,...r->...q", proj, J)
    return (theta / J)[..., None, None] ** (1.0 / 3.0) * F


def element_energy(ue, dNdX, w):
    F = torch.eye(3) + torch.einsum("ai,qaj->qij", ue, dNdX)
    return (w * psi(F)).sum()


def element_energy_p1(ue, dNdX, w, proj):
    F = torch.eye(3) + torch.einsum("ai,qaj->qij", ue, dNdX)
    return (w * psi(fbar(F, proj))).sum()


class Problem:
    def __init__(self, level, p=2, nq=None, formulation="disp"):
        """formulation: "disp" (displacement only) or "fbar_p1" (projected
        dilatation, linear per span: the spline analogue of Q20/P1)."""
        self.level, self.p, self.formulation = level, p, formulation
        nq = nq or p + 1
        nth, nt, nz = LEVELS[level]
        self.bth, self.bt, self.bz = (Basis1D(0, np.pi, nth, p), Basis1D(0, 1, nt, p),
                                      Basis1D(0, LZ, nz, p))
        self.nb = (self.bth.nb, self.bt.nb, self.bz.nb)
        self.nctrl = int(np.prod(self.nb))
        self.ndof = 3 * self.nctrl
        xa, wa, Va, Da, ia = self.bth.span_data(nq)
        xb, wb, Vb, Db, ib = self.bt.span_data(nq)
        xc, wc, Vc, Dc, ic = self.bz.span_data(nq)
        A, B, C = [g.ravel() for g in np.meshgrid(np.arange(nth), np.arange(nt), np.arange(nz),
                                                   indexing="ij")]
        self.ne = len(A)
        # quadrature points per element (e, q_th, q_t, q_z)
        TH = xa[A][:, :, None, None] + 0 * xb[B][:, None, :, None] + 0 * xc[C][:, None, None, :]
        T = 0 * xa[A][:, :, None, None] + xb[B][:, None, :, None] + 0 * xc[C][:, None, None, :]
        Z = 0 * xa[A][:, :, None, None] + 0 * xb[B][:, None, :, None] + xc[C][:, None, None, :]
        x, Jg = geometry(TH, T, Z)
        wq = wa[A][:, :, None, None] * wb[B][:, None, :, None] * wc[C][:, None, None, :]
        detJ = np.linalg.det(Jg)
        # (theta, t, z) is left-handed: det = -r (R_out - R_in) < 0 everywhere; a sign
        # change would mean an inverted map
        assert detJ.max() < 0, "inverted map"
        detJ = -detJ
        Jinv = np.linalg.inv(Jg)
        nl = (p + 1) ** 3
        N = np.einsum("eia,ejb,ekc->eijkabc", Va[A], Vb[B], Vc[C]).reshape(self.ne, nq ** 3, nl)
        dth = np.einsum("eia,ejb,ekc->eijkabc", Da[A], Vb[B], Vc[C]).reshape(self.ne, nq ** 3, nl)
        dt_ = np.einsum("eia,ejb,ekc->eijkabc", Va[A], Db[B], Vc[C]).reshape(self.ne, nq ** 3, nl)
        dz_ = np.einsum("eia,ejb,ekc->eijkabc", Va[A], Vb[B], Dc[C]).reshape(self.ne, nq ** 3, nl)
        dxi = np.stack([dth, dt_, dz_], -1)                                  # (e, q, l, 3)
        Ji = Jinv.reshape(self.ne, nq ** 3, 3, 3)                            # d xi / d x
        self.dNdX = torch.tensor(np.einsum("eqla,eqaj->eqlj", dxi, Ji))
        self.w = torch.tensor((wq * detJ).reshape(self.ne, nq ** 3))
        self.x = x.reshape(self.ne, nq ** 3, 3)
        ctrl = ((ia[A][:, :, None, None] * self.nb[1] + ib[B][:, None, :, None]) * self.nb[2]
                + ic[C][:, None, None, :]).reshape(self.ne, nl)
        self.ctrl = ctrl
        self.dofs = (3 * ctrl[:, :, None] + np.arange(3)).reshape(self.ne, 3 * nl)
        g1 = np.polynomial.legendre.leggauss(nq)[0]
        G = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        self.phi = torch.tensor(np.concatenate([np.ones((len(G), 1)), G], 1))   # (nq^3, 4)
        if formulation == "fbar_p1":
            M = torch.einsum("eq,qa,qb->eab", self.w, self.phi, self.phi)
            self.Minv = torch.linalg.inv(M)
            self.proj = torch.einsum("qa,eab,rb,er->eqr", self.phi, self.Minv, self.phi, self.w)
            f = element_energy_p1
        else:
            self.proj = None
            f = element_energy
        self._e = vmap(f)
        self._g = vmap(grad(f))
        self._h = vmap(jacfwd(grad(f)))
        self._bc()

    # ---- boundary conditions -------------------------------------------------
    def _cid(self, i, j, k):
        return (i * self.nb[1] + j) * self.nb[2] + k

    def _face_projection(self, fun, nq=6):
        """L2 projection (parametric) of fun(theta, z) -> (.., 3) onto the t = 0
        face splines; returns (n_theta_b, n_z_b, 3) control values."""
        res = []
        for b in (self.bth, self.bz):
            g, gw = np.polynomial.legendre.leggauss(nq)
            e = b.edges
            h = e[1:] - e[:-1]
            xq = (0.5 * (e[:-1, None] + e[1:, None]) + 0.5 * h[:, None] * g).ravel()
            wq = (0.5 * h[:, None] * gw).ravel()
            V = b.eval(xq)[0]
            res.append((xq, wq, V))
        (xa, wa, Va), (xc, wc, Vc) = res
        Ma, Mc = Va.T @ (wa[:, None] * Va), Vc.T @ (wc[:, None] * Vc)
        TH, Z = np.meshgrid(xa, xc, indexing="ij")
        G = fun(TH, Z)                                                         # (na, nc, 3)
        out = np.zeros((self.bth.nb, self.bz.nb, 3))
        for k in range(3):
            rhs = Va.T @ (wa[:, None] * G[:, :, k] * wc[None, :]) @ Vc
            out[:, :, k] = np.linalg.solve(Ma, np.linalg.solve(Mc, rhs.T).T)
        return out

    def _bc(self):
        nth_b, nt_b, nz_b = self.nb
        fixed = np.zeros((self.nctrl, 3), bool)
        val = np.zeros((self.nctrl, 3))
        I, K = np.meshgrid(np.arange(nth_b), np.arange(nz_b), indexing="ij")
        inner, outer = self._cid(I, 0, K).ravel(), self._cid(I, nt_b - 1, K).ravel()
        c, s = np.cos(PHI), np.sin(PHI)

        def g(th, z):
            x0 = R_in(z) * np.cos(th)
            dz = z - LZ / 2
            return np.stack([x0 * c + dz * s - x0, 0 * z, -x0 * s + dz * c - dz], -1)
        self.g_ctrl = self._face_projection(g).reshape(-1, 3)
        self.inner = inner
        fixed[outer] = True
        val[outer] = 0.0
        J0, K0 = np.meshgrid(np.arange(nt_b), np.arange(nz_b), indexing="ij")
        for i in (0, nth_b - 1):
            fixed[self._cid(i, J0, K0).ravel(), 1] = True
        fixed[inner] = True
        val[inner] = self.g_ctrl
        self.fixed = np.where(fixed.ravel())[0]
        self.free = np.where(~fixed.ravel())[0]
        self.ubar = val.ravel()

        # virtual displacement of the inner face for a rotation increment, at the
        # deformed position (rotation velocity e_y x (x - c)): work-conjugate moment
        def v(th, z):
            x0 = np.stack([R_in(z) * np.cos(th), R_in(z) * np.sin(th), z], -1)
            xd = x0 + g(th, z)
            a = xd - np.array([0.0, 0.0, LZ / 2])
            return np.stack([a[..., 2], 0 * z, -a[..., 0]], -1)
        self.v_ctrl = self._face_projection(v).reshape(-1, 3)

    # ---- assembly ------------------------------------------------------------
    def _ue(self, u):
        return torch.tensor(u.reshape(-1, 3))[torch.tensor(self.ctrl)]

    def _args(self, s, e):
        a = (self.dNdX[s:e], self.w[s:e])
        return a + ((self.proj[s:e],) if self.proj is not None else ())

    def residual(self, u, chunk=4000):
        R = np.zeros(self.ndof)
        ue = self._ue(u)
        for s in range(0, self.ne, chunk):
            gg = self._g(ue[s:s + chunk], *self._args(s, s + chunk))
            np.add.at(R, self.dofs[s:s + chunk].ravel(), gg.reshape(-1).numpy())
        return R

    def tangent(self, u, chunk=600):
        ue = self._ue(u)
        K = None
        nd = self.dofs.shape[1]
        for s in range(0, self.ne, chunk):
            h = self._h(ue[s:s + chunk], *self._args(s, s + chunk))
            d = self.dofs[s:s + chunk]
            Kc = sp.csr_matrix((h.reshape(-1).numpy(), (np.repeat(d, nd, 1).ravel(),
                                                        np.tile(d, (1, nd)).ravel())),
                               shape=(self.ndof, self.ndof))
            K = Kc if K is None else K + Kc
        return K

    def energy(self, u):
        return float(self._e(self._ue(u), *self._args(0, self.ne)).sum())

    def _linsolve(self, A, b):
        if os.environ.get("VERIF_LINSOLVE", "pardiso") != "amg":
            import pypardiso
            return pypardiso.spsolve(A.tocsr(), b)
        import pyamg
        if not hasattr(self, "_B"):
            # rigid-body modes of the control net (control points as positions)
            xc = self._ctrl_positions()
            Bm = np.zeros((self.ndof, 6))
            for i in range(3):
                Bm[i::3, i] = 1.0
            for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
                Bm[i::3, 3 + k] = -xc[:, j]
                Bm[j::3, 3 + k] = xc[:, i]
            self._B = Bm[self.free]
        ml = pyamg.smoothed_aggregation_solver(A.tocsr(), B=self._B, symmetry="symmetric",
                                               strength=("symmetric", {"theta": 0.0}),
                                               max_coarse=2000)
        return ml.solve(b, tol=1e-12, accel="cg", maxiter=3000)

    def _ctrl_positions(self):
        # Greville abscissae mapped through the geometry
        gv = []
        for b in (self.bth, self.bt, self.bz):
            gv.append(np.array([b.t[i + 1:i + b.p + 1].mean() for i in range(b.nb)]))
        TH, T, Z = np.meshgrid(*gv, indexing="ij")
        return geometry(TH, T, Z)[0].reshape(-1, 3)

    def solve(self, nsteps=5, tol=1e-9, newton_max=30):
        u = np.zeros(self.ndof)
        hist = []
        for step in range(1, nsteps + 1):
            target = (step / nsteps * self.ubar)[self.fixed]
            K = self.tangent(u)
            R = self.residual(u)
            du_c = target - u[self.fixed]
            rhs = -R[self.free] - K[self.free][:, self.fixed] @ du_c
            u[self.free] += self._linsolve(K[self.free][:, self.free], rhs)
            u[self.fixed] = target
            for it in range(newton_max):
                R = self.residual(u)
                Rn = np.linalg.norm(R[self.free])
                if not np.isfinite(Rn):
                    raise RuntimeError(f"step {step}: non-finite residual")
                if it > 0 and Rn < tol:
                    break
                K = self.tangent(u)
                u[self.free] -= self._linsolve(K[self.free][:, self.free], R[self.free])
            else:
                raise RuntimeError(f"step {step}: |R| = {Rn:.2e}")
            hist.append((it, float(Rn)))
        self.u, self.hist = u, hist
        return u

    # ---- quantities of interest ----------------------------------------------
    def _cauchy(self, F):
        F = F.detach().requires_grad_(True)
        (P,) = torch.autograd.grad(psi(F).sum(), F)
        return (torch.einsum("...ij,...kj->...ik", P, F) / det3(F)[..., None, None]).detach().numpy()

    def root_stress(self):
        """Cauchy stress at the groove-root point (theta = 0, t = 0, z = 0.5).
        Displacement formulation: one value (C1 field). P1: the projected
        dilatation is discontinuous between the spans below and above z = 0.5;
        returns the mean of the two one-sided values and both sides."""
        th, t, z = np.array([0.0]), np.array([0.0]), np.array([ZC])
        (va, da), (vb, db), (vc, dc) = (self.bth.eval(th), self.bt.eval(t), self.bz.eval(z))
        N3 = lambda a, b, c: np.einsum("a,b,c->abc", a[0], b[0], c[0]).ravel()
        dxi = np.stack([N3(da, vb, vc), N3(va, db, vc), N3(va, vb, dc)], -1)   # (nctrl, 3)
        _, Jg = geometry(th, t, z)
        dNdX = dxi @ np.linalg.inv(Jg[0])
        U = self.u.reshape(-1, 3)
        F = torch.tensor(np.eye(3) + U.T @ dNdX)
        if self.proj is None:
            return self._cauchy(F), None
        nth, nt, nz = LEVELS[self.level]
        ue = self._ue(self.u)
        sides = []
        for C, zeta in ((nz // 2 - 1, 1.0), (nz // 2, -1.0)):      # span below / above
            e = (0 * nt + 0) * nz + C
            Fq = torch.eye(3) + torch.einsum("ai,qaj->qij", ue[e], self.dNdX[e])
            Jq = det3(Fq)
            coef = self.Minv[e] @ (self.phi * (self.w[e] * Jq)[:, None]).sum(0)
            theta = torch.tensor([1.0, -1.0, -1.0, zeta]) @ coef
            sides.append(self._cauchy((theta / det3(F)) ** (1.0 / 3.0) * F))
        return 0.5 * (sides[0] + sides[1]), sides

    def qois(self):
        ue = self._ue(self.u)
        F = torch.eye(3) + torch.einsum("eai,eqaj->eqij", ue, self.dNdX)
        if self.proj is not None:
            F = fbar(F, self.proj)
        s = self._cauchy(F).reshape(-1, 3, 3)
        w = self.w.numpy().ravel()
        pos = self.x.reshape(-1, 3)
        R = self.residual(self.u).reshape(-1, 3)
        moment = float((R[self.inner] * self.v_ctrl).sum())
        ref = np.array([R_IN0 - D, 0.0, ZC])
        R0 = 2.0 * groove_radius_of_curvature(D, W)
        d = np.linalg.norm(pos - ref, axis=1)
        p = -np.trace(s, axis1=1, axis2=2) / 3.0
        meas = {"sigma_xx": s[:, 0, 0], "sigma_yy": s[:, 1, 1], "sigma_zz": s[:, 2, 2],
                "sigma_xz": s[:, 0, 2], "pressure": p}
        out = {"reaction_moment_y": moment, "strain_energy": self.energy(self.u)}
        kg = w * np.exp(-(d / (2 * R0)) ** 2)
        for name, wt in (("gk2", kg), ("gk2trunc", kg * (d < 0.25)), ("sharp", w * (d < R0))):
            for k, v in meas.items():
                out[f"{name}_{k}_avg"] = float((v * wt).sum() / wt.sum())
        inside = d < R0
        out["n_quad_R0"] = int(inside.sum())
        for k in ("sigma_xx", "sigma_zz", "pressure"):
            out[f"max_{k}_R0"] = float(np.abs(meas[k][inside]).max())
        sr, sides = self.root_stress()
        out.update(root_sigma_xx=float(sr[0, 0]), root_sigma_zz=float(sr[2, 2]),
                   root_sigma_xz=float(sr[0, 2]), root_pressure=float(-np.trace(sr) / 3))
        if sides is not None:
            out["root_sigma_xx_sides"] = [float(x[0, 0]) for x in sides]
        return out


def run(level, p, cache, formulation="disp"):
    key = f"L{level}:p{p}" + ("" if formulation == "disp" else f":{formulation}")
    if key in cache:
        return cache[key]
    t0 = time.time()
    pb = Problem(level, p, formulation=formulation)
    pb.solve()
    row = {"level": level, "p": p, "formulation": formulation, "spans": LEVELS[level], "n_elem": pb.ne, "n_dof": pb.ndof,
           "seconds": time.time() - t0, "newton": pb.hist,
           "linsolve": os.environ.get("VERIF_LINSOLVE", "pardiso"), **pb.qois()}
    cache[key] = row
    json.dump(cache, open(OUT, "w"), indent=1)
    print(f"{key} dof {pb.ndof} {row['seconds']:.0f}s M={row['reaction_moment_y']:.6f} "
          f"W={row['strain_energy']:.6f} gk2_xx={row['gk2_sigma_xx_avg']:.5f} "
          f"max_xx_R0={row['max_sigma_xx_R0']:.4f} root_xx={row['root_sigma_xx']:.4f}", flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--formulation", default="disp", choices=("disp", "fbar_p1"))
    a = ap.parse_args(argv)
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for lv in map(int, a.levels.split(",")):
        run(lv, a.p, cache, a.formulation)


if __name__ == "__main__":
    main()
