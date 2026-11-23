"""FEM against an isogeometric-type discretisation: a 2D pilot on the CFM
open-hole lamina (lamina.py).

Professor Rabczuk asked for FEM against IGA on the groove peak (3D,
hyperelastic). That needs a 3D NURBS hyperelastic solver, which this code
base does not have. This pilot answers the smaller question that can be
answered now: on a problem whose FEM discretisation error turned out to be
mostly GEOMETRIC (goal_oriented.py: the Q2 compliance error equals the area
error of the mesh), does an exact-geometry spline discretisation remove it,
and do two independent implementations agree?

Discretisation (written from scratch, no IGA library): tensor-product
B-splines of degree p (C^(p-1)) in the parameters (s, theta) of the exact
analytic map x = rho(s, theta) (cos theta, sin theta), rho = (1 - s) r_h +
s R(theta), R the ellipse radius; open knots in s (hole s = 0, ellipse s = 1),
periodic knots in theta. The geometry is exact (the analytic map, not a NURBS
fit), so this is IGA-type (smooth splines, exact geometry) but not NURBS IGA.
Same material, load and rigid-motion constraints as lamina.py (three point
constraints at the hole, by Lagrange multipliers). n_theta = 16 m spans, n_s =
8 m spans, as the FEM mesh.

    python -m omar_pfem.verification_agent.iga_lamina
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import BSpline

from omar_pfem.verification_agent import lamina as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "iga_lamina_result.json")


def _open_basis(n_span, p, x):
    t = np.concatenate([np.zeros(p), np.linspace(0, 1, n_span + 1), np.ones(p)])
    n = n_span + p
    b = BSpline(t, np.eye(n), p, extrapolate=False)
    v, d = b(x), b.derivative()(x)
    return np.nan_to_num(v), np.nan_to_num(d), n


def _periodic_basis(n_span, p, x):
    h = 2 * np.pi / n_span
    t = h * np.arange(-p, n_span + p + 1)
    n = n_span + p
    b = BSpline(t, np.eye(n), p, extrapolate=False)
    v, d = np.nan_to_num(b(x)), np.nan_to_num(b.derivative()(x))
    # fold: function i + n_span is function i (periodic), i < p
    v[:, :p] += v[:, n_span:]
    d[:, :p] += d[:, n_span:]
    return v[:, :n_span], d[:, :n_span], n_span


def _geometry(s, th, beta=0.0):
    """Exact map; beta > 0 grades toward the hole smoothly: sigma(s) =
    (e^(beta s) - 1)/(e^beta - 1) (beta = 2.8 gives about the first-element size
    of the FEM grading g = 1.5 at m = 4)."""
    if beta > 0:
        sig = np.expm1(beta * s) / np.expm1(beta)
        dsig = beta * np.exp(beta * s) / np.expm1(beta)
    else:
        sig, dsig = s, np.ones_like(s)
    R = L.ellipse_radius(th)
    a, b = L.A_ELL, L.B_ELL
    q = (b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2
    dR = -a * b * (a * a - b * b) * np.sin(th) * np.cos(th) / q ** 1.5
    rho = (1 - sig) * L.RH + sig * R
    rs, rt = dsig * (R - L.RH), sig * dR
    c, sn = np.cos(th), np.sin(th)
    x = np.array([rho * c, rho * sn])
    xs = np.array([rs * c, rs * sn])
    xt = np.array([rt * c - rho * sn, rt * sn + rho * c])
    return x, xs, xt


def solve(m, p, nq=None, beta=0.0):
    t0 = time.time()
    nq = nq or p + 2
    ns_sp, nt_sp = 8 * m, 16 * m
    g, gw = np.polynomial.legendre.leggauss(nq)
    # quadrature points, span by span
    se = np.linspace(0, 1, ns_sp + 1)
    sq = (0.5 * (se[:-1, None] + se[1:, None]) + 0.5 * (se[1:, None] - se[:-1, None]) * g).ravel()
    sw = (0.5 * (se[1:, None] - se[:-1, None]) * gw).ravel()
    te = np.linspace(0, 2 * np.pi, nt_sp + 1)
    tq = (0.5 * (te[:-1, None] + te[1:, None]) + 0.5 * (te[1:, None] - te[:-1, None]) * g).ravel()
    tw = (0.5 * (te[1:, None] - te[:-1, None]) * gw).ravel()
    Bs, dBs, ns = _open_basis(ns_sp, p, np.minimum(sq, 1 - 1e-14))
    Bt, dBt, nt = _periodic_basis(nt_sp, p, tq)
    nb = ns * nt
    ndof = 2 * nb
    # active functions per span
    act_s = np.array([np.arange(e, e + p + 1) for e in range(ns_sp)])          # open
    act_t = np.array([(np.arange(j, j + p + 1)) % nt_sp for j in range(nt_sp)])  # periodic
    QB = L.stiffness_matrix()
    rows, cols, vals = [], [], []
    acc = {k: 0.0 for k in ("gk_w", "gk_g", "gk_e", "hz_w", "hz_g", "hz_e", "area")}
    store = []                                     # per s-span: data for the QoI pass
    for e in range(ns_sp):
        iq = slice(e * nq, (e + 1) * nq)
        bs, dbs = Bs[iq][:, act_s[e]], dBs[iq][:, act_s[e]]           # (nq, p+1)
        S = sq[iq][:, None, None]
        for j in range(nt_sp):
            jq = slice(j * nq, (j + 1) * nq)
            bt, dbt = Bt[jq][:, act_t[j]], dBt[jq][:, act_t[j]]
            TH = tq[jq][None, :, None]
            x, xs, xt = _geometry(S[..., 0], TH[..., 0], beta)        # (2, nq, nq)
            det = xs[0] * xt[1] - xs[1] * xt[0]
            w = sw[iq][:, None] * tw[jq][None, :] * det * L.THICK
            # parametric gradients of the local functions: (nq_s, nq_t, nloc)
            Ns = np.einsum("ia,jb->ijab", bs, bt).reshape(nq, nq, -1)
            dNs = np.einsum("ia,jb->ijab", dbs, bt).reshape(nq, nq, -1)
            dNt = np.einsum("ia,jb->ijab", bs, dbt).reshape(nq, nq, -1)
            # inverse Jacobian: [ds/dx ds/dy; dt/dx dt/dy]
            sx, sy = xt[1] / det, -xt[0] / det
            tx, ty = -xs[1] / det, xs[0] / det
            dNx = dNs * sx[..., None] + dNt * tx[..., None]
            dNy = dNs * sy[..., None] + dNt * ty[..., None]
            nl = dNx.shape[-1]
            Bm = np.zeros((nq, nq, 3, 2 * nl))
            Bm[:, :, 0, 0::2] = dNx
            Bm[:, :, 1, 1::2] = dNy
            Bm[:, :, 2, 0::2] = dNy
            Bm[:, :, 2, 1::2] = dNx
            Ke = np.einsum("ijka,kl,ijlb,ij->ab", Bm, QB, Bm, w)
            glob = (act_s[e][:, None] * nt + act_t[j][None, :]).ravel()
            gd = np.empty(2 * nl, int)
            gd[0::2], gd[1::2] = 2 * glob, 2 * glob + 1
            rows.append(np.repeat(gd, 2 * nl))
            cols.append(np.tile(gd, 2 * nl))
            vals.append(Ke.ravel())
            store.append((e, j, gd, Bm, w / L.THICK, x))
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ndof, ndof))
    # load on s = 1: t = sigma0 n_x e_x; n ds = (y_t, -x_t) dtheta (counterclockwise)
    bs1 = _open_basis(ns_sp, p, np.array([1 - 1e-14]))[0][0]          # only the last is 1
    _, _, xt1 = _geometry(np.ones_like(tq), tq, beta)
    fx = np.einsum("q,qj->j", tw * L.SIGMA0 * xt1[1] * L.THICK, Bt)    # per theta function
    f = np.zeros(ndof)
    f[2 * ((ns - 1) * nt + np.arange(nt))] = fx * bs1[ns - 1]
    # point constraints (Lagrange multipliers): u_x, u_y at (rh, 0); u_y at (-rh, 0)
    bt0 = _periodic_basis(nt_sp, p, np.array([0.0]))[0][0]
    btp = _periodic_basis(nt_sp, p, np.array([np.pi]))[0][0]
    C = np.zeros((3, ndof))
    C[0, 2 * np.arange(nt)] = bt0            # s = 0: only the first s-function is 1
    C[1, 2 * np.arange(nt) + 1] = bt0
    C[2, 2 * np.arange(nt) + 1] = btp
    A = sp.bmat([[K, sp.csr_matrix(C.T)], [sp.csr_matrix(C), None]], format="csc")
    sol = spla.spsolve(A, np.concatenate([f, np.zeros(3)]))
    u = sol[:ndof]
    compliance = float(f @ u) / L.P ** 2
    for e, j, gd, Bm, wa, x in store:
        eps = np.einsum("ijka,a->ijk", Bm, u[gd])                     # (exx, eyy, gxy)
        gu = np.zeros((2, 2) + eps.shape[:2])
        gu[0, 0], gu[1, 1], gu[0, 1] = eps[..., 0], eps[..., 1], eps[..., 2]
        _, e22, g12 = L.material_strains(gu)
        r = np.hypot(x[0], x[1])
        kg = wa * np.exp(-((r - L.RH) / L.GK_LEN) ** 2)
        hz = (r >= L.HOLE_ZONE[0]) & (r <= L.HOLE_ZONE[1])
        acc["gk_w"] += kg.sum()
        acc["gk_g"] += (g12 * kg).sum()
        acc["gk_e"] += (e22 * kg).sum()
        acc["hz_w"] += wa[hz].sum()
        acc["hz_g"] += (g12 * wa)[hz].sum()
        acc["hz_e"] += (e22 * wa)[hz].sum()
        acc["area"] += wa.sum()
    ut = float(bs1[ns - 1] * (bt0 @ u[2 * ((ns - 1) * nt + np.arange(nt))]))
    return {"m": m, "p": p, "beta": beta, "n_dof": ndof, "seconds": time.time() - t0,
            "area_mm2": acc["area"], "hole_zone_area_mm2": acc["hz_w"],   # added 2026-11-23
            "compliance": compliance, "u_tip": ut,
            "hole_gk_gamma12_avg": acc["gk_g"] / acc["gk_w"],
            "hole_gk_e22_avg": acc["gk_e"] / acc["gk_w"],
            "hole_gamma12_avg": acc["hz_g"] / acc["hz_w"],
            "hole_e22_avg": acc["hz_e"] / acc["hz_w"]}


QOIS = ("compliance", "u_tip", "hole_gk_gamma12_avg", "hole_gk_e22_avg", "hole_gamma12_avg",
        "hole_e22_avg")


def main(ms=(1, 2, 4, 8), ps=(2, 3), betas=(0.0, 2.8)):
    fem_rows = {r["m"]: r for r in json.load(open(L.CACHE)) if r["g"] == 1.5}
    ref = fem_rows[32]
    area_exact = np.pi * L.A_ELL * L.B_ELL - np.pi * L.RH ** 2
    out = {"fem_reference_m": 32, "area_exact": area_exact, "iga": [], "fem": []}
    for m in sorted(fem_rows):
        r = fem_rows[m]
        out["fem"].append({"m": m, "n_dof": r["n_dof"], **{q: r[q] for q in QOIS},
                           "rel_err": {q: (r[q] - ref[q]) / ref[q] for q in QOIS}})
    for beta in betas:
      for p in ps:
        for m in ms:
            r = solve(m, p, beta=beta)
            r["area_rel_error"] = (r["area_mm2"] - area_exact) / area_exact
            r["rel_diff_vs_fem32"] = {q: (r[q] - ref[q]) / ref[q] for q in QOIS}
            out["iga"].append(r)
            print(f"beta={beta} p={p} m={m} dof {r['n_dof']:6d} {r['seconds']:.1f}s area {r['area_rel_error']:.1e} "
                  + " ".join(f"{q} {v:+.2e}" for q, v in r["rel_diff_vs_fem32"].items()),
                  flush=True)
            json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
