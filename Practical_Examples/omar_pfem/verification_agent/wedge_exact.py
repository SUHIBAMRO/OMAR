"""Exact-solution generator for corner singularities (2026-11-24).

Why: the certified uncertainty has been compared with a TRUE error on 12 quantities only
(calibration.py). One exact singular problem (lshape_exact.py) is one data point. This
module generates a family of plane-strain problems with an exact solution, one per corner
type, so coverage can be measured with statistics.

Each case: a circular sector of radius 1 with its corner at the origin, edges at
theta = 0 (condition bc0) and theta = omega (condition bc1), each "free" or "clamped";
plane strain, E = 1000, Poisson ratio nu. The exact field is a sum of two Williams corner
modes (Kolosov-Muskhelishvili potentials phi = A z^lam, psi = B z^lam; the coefficients
are the null vector of the edge conditions, singularity._rows):

    u = u_1 + 0.5 u_2,   lam_1 the smallest real exponent, lam_2 the next real one below 2
                         (lam = 1 skipped: for free/free a rigid rotation); u_1 alone if
                         there is none

Both modes satisfy the corner-edge conditions exactly, so the edges carry their true
conditions (free: no load; clamped: u = 0) and the arc r = 1 carries the exact
displacement. Two terms make the sequences pre-asymptotic, as in real problems.
Only corners with a real exponent lam_1 <= 1 are generated: free/free 200-355 degrees,
clamped/free 60-130 degrees, clamped/clamped 200-315 degrees (above about 140-150 degrees
the clamped/free exponents are complex), each at nu = 0.3 and 0.45: 32 cases. The exact
field contains only the modes named here; the problem is well posed with that arc data,
so the benchmark is exact whatever other exponents the corner has.

Quantities (exact values for scoring only; the framework never sees them):
  strain_energy            1/2 of the arc integral of t.u (edges: t = 0 or u = 0), checked
                           against a polar volume integral of the energy density
  u_bisector               radial displacement at r = 0.5 on the bisector
  corner_gk_avg_von_mises  von Mises averaged with exp(-(r/0.1)^2) about the corner
  far_gk_avg_von_mises     the same about the bisector point at r = 0.6
  peak_von_mises           unbounded (r^(lam_1 - 1)): not verifiable

Discretization: 6-node triangles (P2), isoparametric, mid-edge nodes on the arc projected
onto it. Coarse mesh: rings at r = 0.5 and 1 with m (even) angular segments; levels
l = 0..5 by uniform refinement (h = 2^-l). Strategies at l = 3: the disc r < 0.5 graded
toward the corner, r' = 0.5 (r / 0.5)^g, g = 1.5 and 2.0 (r = 0.5 and the arc unchanged).

    python -m omar_pfem.verification_agent.wedge_exact exact     (exact values + checks)
    python -m omar_pfem.verification_agent.wedge_exact solve     (all ladders)
"""
import json
import os
import sys
import time

import numpy as np
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

HERE = os.path.dirname(os.path.abspath(__file__))
EXACT = os.path.join(HERE, "wedge_exact_values.json")
LADDER = os.path.join(HERE, "wedge_exact_ladder.json")
E = 1000.0
LEVELS = (0, 1, 2, 3, 4, 5)
STRATEGY_LEVEL, GRADINGS = 3, (1.5, 2.0)
WEIGHTS = (1.0, 0.5)
ELL, R_FAR, R_PROBE = 0.1, 0.6, 0.5
QOIS = ("strain_energy", "u_bisector", "corner_gk_avg_von_mises", "far_gk_avg_von_mises",
        "peak_von_mises")


def case_list():
    out = []
    for nu in (0.3, 0.45):
        for deg in (200, 225, 250, 270, 290, 315, 340, 355):
            out.append(dict(omega_deg=deg, bc0="free", bc1="free", nu=nu))
        for deg in (60, 90, 110, 130):
            out.append(dict(omega_deg=deg, bc0="clamped", bc1="free", nu=nu))
        for deg in (200, 225, 270, 315):
            out.append(dict(omega_deg=deg, bc0="clamped", bc1="clamped", nu=nu))
    for c in out:
        c["id"] = f"{c['bc0'][0]}{c['bc1'][0]}_{c['omega_deg']}_nu{int(round(100 * c['nu']))}"
    return out


class Case:
    def __init__(self, omega_deg, bc0, bc1, nu, **_):
        from omar_pfem.verification_agent.singularity import smallest_exponent
        self.omega_deg, self.bc0, self.bc1, self.nu = omega_deg, bc0, bc1, nu
        self.om = np.radians(omega_deg)
        self.mu = E / (2 * (1 + nu))
        self.lame = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = 3 - 4 * nu
        l1 = smallest_exponent(self.om, bc0, bc1, nu, lo=0.05, hi=1.0)
        # the next real exponent below 2, skipping lam = 1 (for free/free a rigid rotation,
        # no stress); none: the leading mode alone
        l2, lo = None, l1 + 0.02
        while lo < 2.0:
            c = smallest_exponent(self.om, bc0, bc1, nu, lo=lo, hi=2.0, n=8000)
            if c is None or abs(c - 1.0) > 1e-6:
                l2 = c
                break
            lo = c + 0.02
        self.lams = (l1,) if l2 is None else (l1, l2)
        self.modes = [(lam, *self._coefficients(lam)) for lam in self.lams]

    # -- the exact field ---------------------------------------------------------------
    def _theta(self, z):
        th = np.mod(np.angle(z), 2 * np.pi)
        return np.where(th > 0.5 * (self.om + 2 * np.pi), th - 2 * np.pi, th)

    def _zpow(self, z, a):
        r, th = np.abs(z), self._theta(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0, r ** a * np.exp(1j * a * th), 0.0)

    def _coefficients(self, lam):
        from omar_pfem.verification_agent.singularity import _rows
        M = np.vstack([_rows(lam, 0.0, self.bc0, self.kappa),
                       _rows(lam, self.om, self.bc1, self.kappa)])
        s = np.linalg.svd(M)
        v = s[2][-1]
        A, B = complex(v[0], v[1]), complex(v[2], v[3])
        zb = np.exp(1j * self.om / 2)                  # scale: |u| = 1 at r = 1, bisector
        u = (self.kappa * A * zb ** lam - zb * np.conj(lam * A * zb ** (lam - 1))
             - np.conj(B * zb ** lam)) / (2 * self.mu)
        return A / abs(u), B / abs(u), float(s[1][-1] / s[1][0])

    def displacement(self, x, y):
        z = np.asarray(x) + 1j * np.asarray(y)
        U = 0
        for w, (lam, A, B, _) in zip(WEIGHTS, self.modes):
            U = U + w * (self.kappa * A * self._zpow(z, lam)
                         - z * np.conj(lam * A * self._zpow(z, lam - 1))
                         - np.conj(B * self._zpow(z, lam)))
        U = U / (2 * self.mu)
        return np.real(U), np.imag(U)

    def stress(self, x, y):
        z = np.asarray(x) + 1j * np.asarray(y)
        s1, s2 = 0, 0
        for w, (lam, A, B, _) in zip(WEIGHTS, self.modes):
            d1 = lam * A * self._zpow(z, lam - 1)
            d2 = lam * (lam - 1) * A * self._zpow(z, lam - 2)
            s1 = s1 + w * 4 * np.real(d1)
            s2 = s2 + w * 2 * (np.conj(z) * d2 + lam * B * self._zpow(z, lam - 1))
        sxx, syy, sxy = (s1 - np.real(s2)) / 2, (s1 + np.real(s2)) / 2, np.imag(s2) / 2
        return sxx, syy, sxy, self.nu * (sxx + syy)

    def von_mises(self, x, y):
        sxx, syy, sxy, szz = self.stress(x, y)
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3 * sxy ** 2)

    def energy_density(self, x, y):
        sxx, syy, sxy, _ = self.stress(x, y)
        c = (1 + self.nu) / E
        exx = c * ((1 - self.nu) * sxx - self.nu * syy)
        eyy = c * ((1 - self.nu) * syy - self.nu * sxx)
        exy = c * sxy
        return 0.5 * (sxx * exx + syy * eyy + 2 * sxy * exy)

    # -- checks and exact quantities ---------------------------------------------------
    def self_checks(self):
        out = {"lam": list(self.lams), "null_space_rel_sv": [m[3] for m in self.modes]}
        r = np.linspace(0.05, 1.0, 40)
        edge = {}
        for th, bc in ((0.0, self.bc0), (self.om, self.bc1)):
            x, y = r * np.cos(th), r * np.sin(th)
            if bc == "free":
                n = (-np.sin(th), np.cos(th))
                sxx, syy, sxy, _ = self.stress(x, y)
                t = np.hypot(sxx * n[0] + sxy * n[1], sxy * n[0] + syy * n[1])
                edge[f"{bc}@{np.degrees(th):.0f}"] = float(np.max(t / self.von_mises(x, y)))
            else:
                ux, uy = self.displacement(x, y)
                edge[f"{bc}@{np.degrees(th):.0f}"] = float(np.max(np.hypot(ux, uy)))
        out["edge_residual"] = edge
        h, eq, hk = 1e-5, 0.0, 0.0
        for rr, f in ((0.3, 0.25), (0.7, 0.5), (0.5, 0.8)):
            px, py = rr * np.cos(f * self.om), rr * np.sin(f * self.om)
            S = lambda a, b: np.array(self.stress(a, b))
            dsx, dsy = (S(px + h, py) - S(px - h, py)) / (2 * h), (S(px, py + h) - S(px, py - h)) / (2 * h)
            eq = max(eq, abs(dsx[0] + dsy[2]) + abs(dsx[2] + dsy[1]))
            ux1, uy1 = self.displacement(px + h, py)
            ux0, uy0 = self.displacement(px - h, py)
            ux3, uy3 = self.displacement(px, py + h)
            ux2, uy2 = self.displacement(px, py - h)
            exx, eyy = (ux1 - ux0) / (2 * h), (uy3 - uy2) / (2 * h)
            exy = 0.5 * ((ux3 - ux2) / (2 * h) + (uy1 - uy0) / (2 * h))
            s = S(px, py)
            hk = max(hk, abs(2 * self.mu * exx + self.lame * (exx + eyy) - s[0]),
                     abs(2 * self.mu * eyy + self.lame * (exx + eyy) - s[1]),
                     abs(2 * self.mu * exy - s[2]))
        out["equilibrium_residual"], out["hooke_residual"] = float(eq), float(hk)
        return out

    def _polar(self, f, nr=240, nth=240):
        """Integral of f over the sector, r = s^4 clusters points at the corner."""
        s, ws = np.polynomial.legendre.leggauss(nr)
        s, ws = 0.5 * (s + 1), 0.5 * ws
        t, wt = np.polynomial.legendre.leggauss(nth)
        t, wt = 0.5 * self.om * (t + 1), 0.5 * self.om * wt
        S, T = np.meshgrid(s, t, indexing="ij")
        R = S ** 4
        W = np.outer(ws * 4 * s ** 3, wt) * R
        return float(np.sum(f(R * np.cos(T), R * np.sin(T)) * W))

    def exact_values(self):
        t, wt = np.polynomial.legendre.leggauss(600)
        t, wt = 0.5 * self.om * (t + 1), 0.5 * self.om * wt
        x, y = np.cos(t), np.sin(t)
        sxx, syy, sxy, _ = self.stress(x, y)
        ux, uy = self.displacement(x, y)
        e_b = 0.5 * float(np.sum(((sxx * x + sxy * y) * ux + (sxy * x + syy * y) * uy) * wt))
        e_v = self._polar(self.energy_density)
        b = self.om / 2
        ux, uy = self.displacement(R_PROBE * np.cos(b), R_PROBE * np.sin(b))
        cx, cy = R_FAR * np.cos(b), R_FAR * np.sin(b)
        kc = lambda X, Y: np.exp(-(np.hypot(X, Y) / ELL) ** 2)
        kf = lambda X, Y: np.exp(-(np.hypot(X - cx, Y - cy) / ELL) ** 2)
        return {"strain_energy": e_b, "strain_energy_volume": e_v,
                "energy_two_ways_rel": abs(e_b - e_v) / abs(e_b),
                "u_bisector": float(ux * np.cos(b) + uy * np.sin(b)),
                "corner_gk_avg_von_mises": self._polar(lambda X, Y: self.von_mises(X, Y) * kc(X, Y))
                / self._polar(kc),
                "far_gk_avg_von_mises": self._polar(lambda X, Y: self.von_mises(X, Y) * kf(X, Y))
                / self._polar(kf),
                "peak_von_mises": float("inf")}

    # -- the finite element ladder -----------------------------------------------------
    def _on_edge(self, p, th):
        return (np.abs(-np.sin(th) * p[0] + np.cos(th) * p[1]) < 1e-9) & \
               (np.cos(th) * p[0] + np.sin(th) * p[1] > -1e-12)

    def mesh(self, level, g=1.0):
        m = 2 * int(np.ceil(self.om / (np.pi / 3)))
        th = np.linspace(0, self.om, m + 1)
        pts = [[0.0, 0.0]] + [[0.5 * np.cos(a), 0.5 * np.sin(a)] for a in th] + \
              [[np.cos(a), np.sin(a)] for a in th]
        i1 = lambda k: 1 + k
        i2 = lambda k: 2 + m + k
        tri = [[0, i1(k), i1(k + 1)] for k in range(m)]
        for k in range(m):
            tri += [[i1(k), i2(k), i2(k + 1)], [i1(k), i2(k + 1), i1(k + 1)]]
        msh = fem.MeshTri(np.array(pts).T, np.array(tri).T)
        for _ in range(level):
            msh = msh.refined()
            p = msh.p.copy()
            bnd = np.zeros(p.shape[1], bool)
            bnd[msh.boundary_nodes()] = True
            arc = bnd & ~self._on_edge(p, 0.0) & ~self._on_edge(p, self.om)
            p[:, arc] /= np.linalg.norm(p[:, arc], axis=0)
            msh = fem.MeshTri(p, msh.t)
        p = msh.p.copy()
        if g != 1.0:
            r = np.linalg.norm(p, axis=0)
            inner = (r > 0) & (r < 0.5 - 1e-12)
            p[:, inner] *= (0.5 * (r[inner] / 0.5) ** g / r[inner])
            msh = fem.MeshTri(p, msh.t)
        m2 = fem.MeshTri2.from_mesh(msh)
        dof = m2.doflocs.copy()
        nv, fa = msh.p.shape[1], msh.facets
        ra = np.linalg.norm(msh.p[:, fa[0]], axis=0)
        rb = np.linalg.norm(msh.p[:, fa[1]], axis=0)
        on = np.isclose(ra, 1.0) & np.isclose(rb, 1.0)
        mid = dof[:, nv + np.where(on)[0]]
        dof[:, nv + np.where(on)[0]] = mid / np.linalg.norm(mid, axis=0)
        return fem.MeshTri2(dof, msh.t), msh

    def solve_level(self, level, g=1.0):
        import pypardiso
        t0 = time.time()
        m2, m1 = self.mesh(level, g)
        mu, lame = self.mu, self.lame

        @fem.BilinearForm
        def stiffness(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * mu * ddot(eu, ev) + lame * trace(eu) * trace(ev)

        b = fem.Basis(m2, fem.ElementVector(fem.ElementTriP2()), intorder=6)
        K = fem.asm(stiffness, b)
        x = np.zeros(K.shape[0])
        comp = np.zeros(K.shape[0], int)
        comp[b.nodal_dofs[1]] = 1
        comp[b.facet_dofs[1]] = 1
        loc = b.doflocs
        bf = m2.boundary_facets()            # arc facets: both end vertices on r = 1
        rv = np.linalg.norm(m2.p, axis=0)
        arc_f = bf[np.isclose(rv[m2.facets[0, bf]], 1.0) & np.isclose(rv[m2.facets[1, bf]], 1.0)]
        arc = b.get_dofs(facets=arc_f).all()
        ux, uy = self.displacement(loc[0, arc], loc[1, arc])
        x[arc] = np.where(comp[arc] == 0, ux, uy)
        D = [arc]
        for th, bc in ((0.0, self.bc0), (self.om, self.bc1)):
            if bc == "clamped":
                D.append(b.get_dofs(lambda X, th=th: self._on_edge(X, th)).all())
        D = np.unique(np.concatenate(D))
        u = fem.solve(*fem.condense(K, np.zeros_like(x), x=x, D=D),
                      solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
        energy = float(0.5 * u @ (K @ u))
        gu = b.interpolate(u).grad
        eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
        tr = eps[0, 0] + eps[1, 1]
        sxx, syy = 2 * mu * eps[0, 0] + lame * tr, 2 * mu * eps[1, 1] + lame * tr
        szz, sxy = lame * tr, 2 * mu * eps[0, 1]
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
        X, wq = b.global_coordinates().value, b.dx
        bis = self.om / 2
        cx, cy = R_FAR * np.cos(bis), R_FAR * np.sin(bis)
        kc = wq * np.exp(-(np.hypot(X[0], X[1]) / ELL) ** 2)
        kf = wq * np.exp(-(np.hypot(X[0] - cx, X[1] - cy) / ELL) ** 2)
        # the bisector point at r = 0.5 is a vertex on every mesh (m even; r = 0.5 is fixed)
        px, py = R_PROBE * np.cos(bis), R_PROBE * np.sin(bis)
        j = int(np.argmin((m1.p[0] - px) ** 2 + (m1.p[1] - py) ** 2))
        assert np.hypot(m1.p[0, j] - px, m1.p[1, j] - py) < 1e-9
        U = u[b.nodal_dofs].T
        return {"level": level, "g": g, "h": 2.0 ** -level, "n_elem": int(m1.t.shape[1]),
                "n_dof": int(u.size), "strain_energy": energy,
                "u_bisector": float(U[j, 0] * np.cos(bis) + U[j, 1] * np.sin(bis)),
                "corner_gk_avg_von_mises": float((vm * kc).sum() / kc.sum()),
                "far_gk_avg_von_mises": float((vm * kf).sum() / kf.sum()),
                "peak_von_mises": float(vm.max()), "wall_clock_s": time.time() - t0}


def exact_all():
    out = {}
    for c in case_list():
        k = Case(**c)
        out[c["id"]] = {**c, "lams": list(k.lams), "checks": k.self_checks(),
                        "exact": k.exact_values()}
        print(c["id"], [round(l, 4) for l in k.lams],
              f"energy two ways {out[c['id']]['exact']['energy_two_ways_rel']:.1e}", flush=True)
    json.dump(out, open(EXACT, "w"), indent=1, default=float)
    return out


def solve_all():
    rows = json.load(open(LADDER)) if os.path.exists(LADDER) else []
    done = {(r["case"], r["level"], r["g"]) for r in rows}
    for c in case_list():
        k = Case(**c)
        jobs = [(lv, 1.0) for lv in LEVELS] + [(STRATEGY_LEVEL, g) for g in GRADINGS]
        for lv, g in jobs:
            if (c["id"], lv, g) in done:
                continue
            r = k.solve_level(lv, g)
            r["case"] = c["id"]
            rows.append(r)
            json.dump(rows, open(LADDER, "w"), indent=1)
        print(c["id"], "done", flush=True)
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    if sys.argv[1:] == ["exact"]:
        exact_all()
    elif sys.argv[1:] == ["solve"]:
        solve_all()
    else:
        print(__doc__)
