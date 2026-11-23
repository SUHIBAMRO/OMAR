"""Unseen 3D benchmark (2026-11-23): thick-walled cylinder, inner surface expanded.

Why this problem: it is three-dimensional, uses the B3 element (Q20, torch-fem
Hexa2, default reduced rule), the B3 material (compressible neo-Hookean,
E = 1000, nu = 0.45) and curved iso-parametric boundaries, like B3; but unlike
B3 its continuum solution is known to ~1e-10 from a 1D radial boundary-value
problem. So it can test what no development problem could: whether the
uncertainty the agent certifies from one Q20 family contains the TRUE error.
The agent never sees the exact solution; it is used only for scoring.

Geometry: quarter of a hollow cylinder, inner radius A = 1, outer B = 2,
height L = 0.5, theta in [0, pi/2]. Boundary conditions: inner surface
prescribed u = DELTA * (x, y, 0) (radial expansion by 10%); outer surface
free; symmetry planes y = 0 (u_y = 0) and x = 0 (u_x = 0); z = 0 and z = L
rollers (u_z = 0), i.e. plane strain, so the exact field is axisymmetric:
r -> rho(r), theta -> theta, z -> z.

Exact solution: principal stretches l1 = rho', l2 = rho / r, l3 = 1;
psi = mu/2 (l1^2 + l2^2 + 1 - 3 - 2 ln J) + lam/2 (ln J)^2, J = l1 l2.
Euler-Lagrange d/dr (r dpsi/dl1) - dpsi/dl2 = 0, rho(A) = (1 + DELTA) A,
dpsi/dl1 (B) = 0 (traction-free); solved by shooting on rho'(A) with
solve_ivp (rtol 1e-12) and brentq.

Mesh levels (n_r, n_theta, n_z) = k (2, 4, 1), k = 1, 2, 3, 4, 6, 8; every
node, midside nodes included, placed by the exact map (r, theta, z) -> x.
Strategies at k = 4: radial power grading toward the inner surface
r_j = A + (B - A) (j / n_r)^g, g = 1.5, 2.0 (same element count).

Quantities (Cauchy stress, hoop = e_theta . sigma . e_theta at Gauss points):
  strain_energy           sum over Gauss points of psi(F) |det J| w
  reaction_inner_radial   |sum over inner nodes of reaction . e_r|
  u_outer_radial          u_x at the node (B, 0, 0)
  gk_avg_hoop             hoop stress averaged with exp(-(d / 0.2)^2), d the distance
                          to (A, 0, L/2), over all Gauss points
  disc_avg_hoop           hoop stress averaged over the Gauss points with d < 0.3
  peak_hoop               maximum hoop stress over all Gauss points

    python -m omar_pfem.verification_agent.cylinder3d exact
    python -m omar_pfem.verification_agent.cylinder3d solve [levels]
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cylinder3d_ladder.json")
EXACT = os.path.join(HERE, "cylinder3d_exact.json")

A, B, L = 1.0, 2.0, 0.5
E, NU = 1000.0, 0.45
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
DELTA = 0.10
REF = np.array([A, 0.0, L / 2])
ELL, R_DISC = 0.2, 0.3
LEVELS = (1, 2, 3, 4, 6, 8)
STRATEGY_LEVEL, GRADINGS = 4, (1.5, 2.0)


# ---------------------------------------------------------------------------
# exact (1D) solution
# ---------------------------------------------------------------------------
def _P(l1, l2):
    lnJ = np.log(l1 * l2)
    P1 = MU * (l1 - 1 / l1) + LAM * lnJ / l1
    P2 = MU * (l2 - 1 / l2) + LAM * lnJ / l2
    return P1, P2, LAM * lnJ


def _rhs(r, y):
    rho, l1 = y
    l2 = rho / r
    P1, P2, _ = _P(l1, l2)
    dP1_dl1 = MU * (1 + 1 / l1 ** 2) + LAM * (1 - np.log(l1 * l2)) / l1 ** 2
    dP1_dl2 = LAM / (l1 * l2)
    d2 = (P2 - P1 - dP1_dl2 * (l1 - l2)) / (r * dP1_dl1)
    return [l1, d2]


def _shoot(s, dense=False):
    sol = solve_ivp(_rhs, (A, B), [(1 + DELTA) * A, s], method="DOP853", rtol=1e-12,
                    atol=1e-14, dense_output=dense)
    if dense:
        return sol
    rho, l1 = sol.y[:, -1]
    return _P(l1, rho / B)[0]


def exact_profile():
    s = brentq(_shoot, 0.5, 1.2, xtol=1e-15)
    sol = _shoot(s, dense=True)

    def fields(r):
        r = np.asarray(r, float)
        rho, l1 = sol.sol(r)
        l2 = rho / r
        P1, P2, P3 = _P(l1, l2)
        J = l1 * l2
        psi = MU / 2 * (l1 ** 2 + l2 ** 2 + 1 - 3 - 2 * np.log(J)) + LAM / 2 * np.log(J) ** 2
        return {"rho": rho, "l1": l1, "l2": l2, "s_rr": l1 * P1 / J, "s_tt": l2 * P2 / J,
                "s_zz": P3 / J, "P1": P1, "psi": psi}
    return s, fields


def exact_qois():
    s, f = exact_profile()
    W = np.pi / 2 * L * quad(lambda r: f(r)["psi"] * r, A, B, epsabs=0, epsrel=1e-13,
                             limit=200)[0]
    reaction = abs(f(A)["P1"]) * A * np.pi / 2 * L
    u_out = float(f(B)["rho"] - B)
    # Gaussian average over the quarter cylinder: tensor Gauss-Legendre in (r, theta, z)
    xr, wr = np.polynomial.legendre.leggauss(400)
    xt, wt = np.polynomial.legendre.leggauss(400)
    xz, wz = np.polynomial.legendre.leggauss(80)
    r = A + (B - A) * (xr + 1) / 2
    th = np.pi / 4 * (xt + 1)
    z = L * (xz + 1) / 2
    wr, wt, wz = wr * (B - A) / 2, wt * np.pi / 4, wz * L / 2
    R, T, Z = np.meshgrid(r, th, z, indexing="ij")
    d2 = R ** 2 + A ** 2 - 2 * R * A * np.cos(T) + (Z - L / 2) ** 2
    Wt = np.einsum("i,j,k->ijk", wr * r, wt, wz) * np.exp(-d2 / ELL ** 2)
    stt = f(r)["s_tt"]
    gk = float((Wt * stt[:, None, None]).sum() / Wt.sum())
    # sharp disc d < R_DISC: theta range per (r, z) is analytic, [0, theta_max]

    def tmax(rr, zz):
        c = (rr ** 2 + A ** 2 + (zz - L / 2) ** 2 - R_DISC ** 2) / (2 * rr * A)
        return np.arccos(np.clip(c, -1, 1)) if c < 1 else 0.0
    r_hi = A + R_DISC
    num = den = 0.0
    for zz, wzz in zip(z, wz):
        h = R_DISC ** 2 - (zz - L / 2) ** 2
        if h <= 0:
            continue
        rmax = A + np.sqrt(h)          # on theta = 0 the disc reaches r = A + sqrt(h)
        g = lambda rr: tmax(rr, zz) * rr
        n = quad(lambda rr: f(rr)["s_tt"] * g(rr), A, min(rmax, r_hi), epsabs=0,
                 epsrel=1e-12, limit=200)[0]
        dd = quad(g, A, min(rmax, r_hi), epsabs=0, epsrel=1e-12, limit=200)[0]
        num += wzz * n
        den += wzz * dd
    disc = num / den
    return {"rho_prime_at_A": s, "strain_energy": W, "reaction_inner_radial": reaction,
            "u_outer_radial": u_out, "gk_avg_hoop": gk, "disc_avg_hoop": disc,
            "peak_hoop": float(f(A)["s_tt"]), "hoop_at_B": float(f(B)["s_tt"]),
            "radial_stress_at_B": float(f(B)["s_rr"]),
            "note": "peak_hoop = hoop Cauchy stress at r = A (the maximum, hoop stress "
                    "decreases with r); radial_stress_at_B should be ~0 (traction-free)"}


# ---------------------------------------------------------------------------
# Q20 mesh and solve
# ---------------------------------------------------------------------------
def mesh(k, g=1.0):
    from torchfem.elements import linear_to_quadratic
    nr, nt, nz = 2 * k, 4 * k, k
    s = (np.arange(nr + 1) / nr) ** g
    rr = A + (B - A) * s
    tt = np.linspace(0, np.pi / 2, nt + 1)
    zz = np.linspace(0, L, nz + 1)
    R, T, Z = np.meshgrid(rr, tt, zz, indexing="ij")             # (nr+1, nt+1, nz+1)
    par = np.stack([R.ravel(), T.ravel(), Z.ravel()], 1)
    idx = np.arange(len(par)).reshape(R.shape)
    el = []
    for i in range(nr):
        for j in range(nt):
            for m in range(nz):
                el.append([idx[i, j, m], idx[i + 1, j, m], idx[i + 1, j + 1, m],
                           idx[i, j + 1, m], idx[i, j, m + 1], idx[i + 1, j, m + 1],
                           idx[i + 1, j + 1, m + 1], idx[i, j + 1, m + 1]])
    el = np.array(el)
    X8 = np.stack([par[:, 0] * np.cos(par[:, 1]), par[:, 0] * np.sin(par[:, 1]), par[:, 2]], 1)
    n20, e20 = linear_to_quadratic(torch.tensor(X8), torch.tensor(el))
    n20, e20 = n20.numpy().copy(), e20.numpy()
    # midside nodes: parametric midpoint of the parent edge, mapped exactly
    from torchfem.elements import Hexa1
    edges = Hexa1.edges.numpy()
    mid = e20[:, 8:20].ravel()
    pa, pb = el[:, edges[:, 0]].ravel(), el[:, edges[:, 1]].ravel()
    um, first = np.unique(mid, return_index=True)
    assert np.array_equal(um, np.arange(len(par), len(par) + len(um)))
    pm = 0.5 * (par[pa[first]] + par[pb[first]])
    n20[len(par):] = np.stack([pm[:, 0] * np.cos(pm[:, 1]), pm[:, 0] * np.sin(pm[:, 1]),
                               pm[:, 2]], 1)
    return n20, e20


def solve(k, g=1.0, nsteps=10, quadrature="reduced", tol=1e-10):
    """quadrature: "reduced" (torch-fem default 2x2x2, the pre-registered setting) or
    "full" (3x3x3, q20_full.py; added 2026-11-23 for the graded-mesh study)."""
    if quadrature == "full":
        from omar_pfem.verification_agent.q20_full import SolidFull as Solid
    else:
        from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    from omar_pfem.torchfem_comparison import cauchy_from_torchfem_flux, neo_hookean_psi_3d
    X, el = mesh(k, g)
    r = np.hypot(X[:, 0], X[:, 1])
    inner = np.abs(r - A) < 1e-9
    model = Solid(torch.tensor(X), torch.tensor(el),
                  Hyperelastic3D(psi=neo_hookean_psi_3d, params=torch.tensor([MU, LAM])))
    assert model.etype.__name__ == ("Hexa2" if quadrature == "reduced" else "Hexa2Full")
    n = len(X)
    con = torch.zeros(n, 3, dtype=torch.bool)
    dis = torch.zeros(n, 3)
    con[inner, :] = True
    dis[inner, 0] = torch.tensor(DELTA * X[inner, 0])
    dis[inner, 1] = torch.tensor(DELTA * X[inner, 1])
    con[np.abs(X[:, 1]) < 1e-12, 1] = True
    con[np.abs(X[:, 0]) < 1e-12, 0] = True
    con[(np.abs(X[:, 2]) < 1e-12) | (np.abs(X[:, 2] - L) < 1e-12), 2] = True
    model.constraints, model.displacements = con, dis
    model.forces = torch.zeros(n, 3)
    t0 = time.time()
    with torch.no_grad():
        u, f, S, F, _ = model.solve(increments=torch.linspace(0, 1, nsteps + 1), max_iter=30,
                                    rtol=tol, atol=tol, stol=tol, method="direct",
                                    aggregate_integration_points=False)
    sec = time.time() - t0
    S = S.transpose(0, 1).numpy()
    F = F.transpose(0, 1).numpy()
    sig = cauchy_from_torchfem_flux(S, F)
    Ng, _, detJ = model.eval_shape_functions(model.etype.ipoints)
    wq = model.etype.iweights.numpy()[None, :] * np.abs(detJ.transpose(0, 1).numpy())
    pos = torch.einsum("gn,enD->egD", Ng, torch.tensor(X)[torch.tensor(el)]).numpy()
    th = np.arctan2(pos[..., 1], pos[..., 0])
    et = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1)
    hoop = np.einsum("egi,egij,egj->eg", et, sig, et)
    J = np.linalg.det(F)
    I1 = (F ** 2).sum((-2, -1))
    psi = MU / 2 * (I1 - 3 - 2 * np.log(J)) + LAM / 2 * np.log(J) ** 2
    d = np.linalg.norm(pos - REF, axis=-1)
    kg = wq * np.exp(-(d / ELL) ** 2)
    U = u.numpy()
    fn = f.numpy()
    ri = r[inner]
    reac = abs(float((fn[inner, 0] * X[inner, 0] / ri + fn[inner, 1] * X[inner, 1] / ri).sum()))
    j = int(np.argmin(np.linalg.norm(X - np.array([B, 0.0, 0.0]), axis=1)))
    assert np.linalg.norm(X[j] - np.array([B, 0.0, 0.0])) < 1e-12
    disc = d < R_DISC
    return {"k": k, "grading": g, "h": 1.0 / k, "quadrature": quadrature,
            "min_detF_location": pos.reshape(-1, 3)[int(np.argmin(J))].tolist(), "n_elem": int(len(el)), "n_dof": int(3 * n),
            "seconds": sec, "min_detF": float(J.min()),
            "strain_energy": float((psi * wq).sum()), "reaction_inner_radial": reac,
            "u_outer_radial": float(U[j, 0]),
            "gk_avg_hoop": float((hoop * kg).sum() / kg.sum()),
            "disc_avg_hoop": float((hoop * wq)[disc].sum() / wq[disc].sum()),
            "disc_gauss_points": int(disc.sum()),
            "peak_hoop": float(hoop.max()),
            "volume": float(wq.sum()), "volume_exact": float(np.pi / 4 * (B ** 2 - A ** 2) * L)}


def run(levels, g=1.0):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["k"], r["grading"]) for r in rows}
    for k in levels:
        if (k, g) in done:
            continue
        row = solve(k, g)
        rows.append(row)
        rows.sort(key=lambda r: (r["grading"], r["k"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"k={k} g={g} n_elem={row['n_elem']} dof={row['n_dof']} ({row['seconds']:.0f} s)",
              flush=True)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "exact":
        res = exact_qois()
        json.dump(res, open(EXACT, "w"), indent=1)
        print(json.dumps(res, indent=1))
    elif argv and argv[0] == "solve":
        levels = [int(x) for x in argv[1].split(",")] if len(argv) > 1 else LEVELS
        run(levels)
        if len(argv) <= 1:
            for g in GRADINGS:
                run([STRATEGY_LEVEL], g)


if __name__ == "__main__":
    main()
