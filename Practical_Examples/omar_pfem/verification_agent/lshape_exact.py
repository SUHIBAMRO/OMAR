"""A singular benchmark WITH an exact solution (round K, 2026-11-23).

Why: the post hoc singular-order extension certifies quantities on singular problems
(L-shape, crack, plate, block), but its certified uncertainty has been compared with a
TRUE error only on a smooth problem (the 3D cylinder). Here the continuum solution of a
singular problem is known exactly, so coverage can be tested where it matters.

Problem: the L-shaped panel of lshape.py (union of [0,1]^2, [1,2]x[0,1], [0,1]x[1,2];
plane strain, E = 1000, nu = 0.3; Q2 elements, n x n per unit square), loaded instead by
the EXACT displacement of a Williams corner solution on the four outer edges
(x = 0, x = 2, y = 0, y = 2). The two edges through the re-entrant corner (1, 1) are
traction-free, which the corner solution satisfies, so the singularity comes from the
problem and not from the data. The exact field is

    u = u_1 + 0.5 u_2,   u_k = r^lam_k f_k(theta)   (Kolosov-Muskhelishvili, phi = A z^lam,
                                                     psi = B z^lam, z centred at (1, 1))

with the two free/free exponents of the 270-degree corner, lam_1 = 0.5445 (the
leading one) and lam_2 = 0.9085: two singular terms, so the sequences are
pre-asymptotic as in real problems. Each mode is scaled to |u| = 1 at r = 1 on the
bisector. No body force (the field is an exact equilibrium solution).

Exact quantities (scoring only; the agent never sees them):
  strain_energy            1/2 of the boundary integral of t.u over the outer edges (the
                           corner edges are traction-free); checked against a polar
                           volume integral of the energy density
  u_mid_left               u_x at (0.5, 1.5);  v_mid_right  u_y at (1.5, 0.5)
  corner_gk_avg_von_mises  von Mises averaged with exp(-(d/0.1)^2) about (1, 1)
  corner_disc_avg_von_mises  its mean over the disc d < 0.2 (inside the wedge)
  far_gk_avg_syy           sigma_yy averaged with exp(-(d/0.1)^2) about (0.5, 0.5)
  peak_von_mises           unbounded (r^(lam_1 - 1)): not verifiable

    python -m omar_pfem.verification_agent.lshape_exact exact     (exact values, checks)
    python -m omar_pfem.verification_agent.lshape_exact solve     (the ladder)
"""
import json
import os
import sys
import time

import numpy as np
import skfem as fem

from omar_pfem.verification_agent import lshape as L

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "lshape_exact_ladder.json")
EXACT = os.path.join(HERE, "lshape_exact_values.json")
E, NU = L.E, L.NU
MU, LAM = L.MU, L.LAM
KAPPA = 3 - 4 * NU
C0 = complex(1.0, 1.0)
TH0, TH1 = np.pi / 2, 2 * np.pi            # the two free edges, material in between
WEIGHTS = (1.0, 0.5)
P_LEFT, P_RIGHT = (0.5, 1.5), (1.5, 0.5)
LEVELS = (2, 4, 8, 16, 32, 64, 128)
STRATEGY_N, GRADINGS = 16, (1.5, 2.0)
mesh = L.mesh                               # the L-shape mesh, unchanged
stiffness = L.stiffness


# ---------------------------------------------------------------- exact corner solution
def exponents():
    from omar_pfem.verification_agent.singularity import smallest_exponent
    l1 = smallest_exponent(1.5 * np.pi, "free", "free", NU, "strain")
    l2 = smallest_exponent(1.5 * np.pi, "free", "free", NU, "strain", lo=l1 + 0.02)
    return l1, l2


def _theta(z):
    th = np.mod(np.angle(z), 2 * np.pi)
    return np.where(th < np.pi / 4, th + 2 * np.pi, th)   # material: theta in [pi/2, 2 pi]


def _zpow(z, a):
    r, th = np.abs(z), _theta(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r ** a * np.exp(1j * a * th), 0.0)


def _coefficients(lam):
    from omar_pfem.verification_agent.singularity import _rows
    M = np.vstack([_rows(lam, TH0, "free", KAPPA), _rows(lam, TH1, "free", KAPPA)])
    v = np.linalg.svd(M)[2][-1]
    A, B = complex(v[0], v[1]), complex(v[2], v[3])
    # scale: |u| = 1 at r = 1 on the bisector theta = 5 pi / 4
    zb = np.exp(1j * 5 * np.pi / 4)
    u = (KAPPA * A * zb ** lam - zb * np.conj(lam * A * zb ** (lam - 1)) - np.conj(B * zb ** lam))
    return A / abs(u) * 2 * MU, B / abs(u) * 2 * MU, float(np.linalg.svd(M)[1][-1])


_MODES = None


def modes():
    global _MODES
    if _MODES is None:
        _MODES = [(lam, *_coefficients(lam)[:2]) for lam in exponents()]
    return _MODES


def displacement(x, y):
    z = (np.asarray(x) + 1j * np.asarray(y)) - C0
    U = 0
    for w, (lam, A, B) in zip(WEIGHTS, modes()):
        U = U + w * (KAPPA * A * _zpow(z, lam) - z * np.conj(lam * A * _zpow(z, lam - 1))
                     - np.conj(B * _zpow(z, lam)))
    U = U / (2 * MU)
    return np.real(U), np.imag(U)


def stress(x, y):
    """(sxx, syy, sxy, szz) of the exact field (plane strain)."""
    z = (np.asarray(x) + 1j * np.asarray(y)) - C0
    s1, s2 = 0, 0
    for w, (lam, A, B) in zip(WEIGHTS, modes()):
        d1 = lam * A * _zpow(z, lam - 1)
        d2 = lam * (lam - 1) * A * _zpow(z, lam - 2)
        s1 = s1 + w * 4 * np.real(d1)
        s2 = s2 + w * 2 * (np.conj(z) * d2 + lam * B * _zpow(z, lam - 1))
    sxx, syy, sxy = (s1 - np.real(s2)) / 2, (s1 + np.real(s2)) / 2, np.imag(s2) / 2
    return sxx, syy, sxy, NU * (sxx + syy)


def von_mises(x, y):
    sxx, syy, sxy, szz = stress(x, y)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)


def self_checks():
    """Traction-free corner edges, equilibrium, Hooke's law against the displacement
    gradient (finite differences), and the eigenvalue residuals."""
    out = {"lam": [m[0] for m in modes()],
           "smallest_singular_value": [_coefficients(m[0])[2] for m in modes()]}
    r = np.linspace(0.05, 1.0, 40)
    t_free = 0.0
    for th, n in ((TH0, (1.0, 0.0)), (TH1, (0.0, 1.0))):   # x = 1 edge: n = (1,0); y = 1: (0,1)
        x, y = 1 + r * np.cos(th), 1 + r * np.sin(th)
        sxx, syy, sxy, _ = stress(x, y)
        t = np.hypot(sxx * n[0] + sxy * n[1], sxy * n[0] + syy * n[1])
        t_free = max(t_free, float(np.max(t / np.abs(von_mises(x, y)))))
    out["traction_on_free_edges_rel"] = t_free
    h = 1e-5
    pts = [(0.3, 0.4), (1.6, 0.7), (0.5, 1.7), (0.8, 0.9)]
    eq, hk = 0.0, 0.0
    for px, py in pts:
        dsx = [(np.array(stress(px + h, py)) - np.array(stress(px - h, py))) / (2 * h)]
        dsy = [(np.array(stress(px, py + h)) - np.array(stress(px, py - h))) / (2 * h)]
        fx = dsx[0][0] + dsy[0][2]
        fy = dsx[0][2] + dsy[0][1]
        eq = max(eq, abs(fx) + abs(fy))
        ux1, uy1 = displacement(px + h, py)
        ux0, uy0 = displacement(px - h, py)
        ux3, uy3 = displacement(px, py + h)
        ux2, uy2 = displacement(px, py - h)
        exx, eyy = (ux1 - ux0) / (2 * h), (uy3 - uy2) / (2 * h)
        exy = 0.5 * ((ux3 - ux2) / (2 * h) + (uy1 - uy0) / (2 * h))
        tr = exx + eyy
        s = stress(px, py)
        hk = max(hk, abs(2 * MU * exx + LAM * tr - s[0]), abs(2 * MU * eyy + LAM * tr - s[1]),
                 abs(2 * MU * exy - s[2]))
    out["equilibrium_residual_max"] = float(eq)
    out["hooke_residual_max"] = float(hk)
    return out


def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def energy_boundary(n=400):
    """1/2 of the integral of t.u over the four outer edges (t = sigma n)."""
    tot = 0.0
    for (a, b), fixed, normal in ((((0, 2), ("y", 0.0)), None, None),):
        pass
    edges = [("y", 0.0, 0.0, 2.0, (0.0, -1.0)), ("x", 2.0, 0.0, 1.0, (1.0, 0.0)),
             ("y", 2.0, 0.0, 1.0, (0.0, 1.0)), ("x", 0.0, 0.0, 2.0, (-1.0, 0.0))]
    for axis, c, lo, hi, nrm in edges:
        s, w = _gl(lo, hi, n)
        x, y = (s, np.full_like(s, c)) if axis == "y" else (np.full_like(s, c), s)
        sxx, syy, sxy, _ = stress(x, y)
        ux, uy = displacement(x, y)
        tx, ty = sxx * nrm[0] + sxy * nrm[1], sxy * nrm[0] + syy * nrm[1]
        tot += 0.5 * float(((tx * ux + ty * uy) * w).sum())
    return tot


def _polar_square(f, theta_lo, theta_hi, R, nth=160, nr=120, q=3):
    """Integral of f(x, y) over {theta in [lo, hi], 0 <= r <= R(theta)} around (1, 1),
    with r = R s^q (clusters points at the corner, integrable singularity)."""
    th, wt = _gl(theta_lo, theta_hi, nth)
    s, ws = _gl(0.0, 1.0, nr)
    tot = 0.0
    for t, w in zip(th, wt):
        Rt = R(t)
        r = Rt * s ** q
        jac = Rt * q * s ** (q - 1)
        x, y = 1 + r * np.cos(t), 1 + r * np.sin(t)
        tot += w * float((f(x, y) * r * jac * ws).sum())
    return tot


def energy_density(x, y):
    sxx, syy, sxy, _ = stress(x, y)
    tr = (sxx + syy) * (1 - 2 * NU) * (1 + NU) / E          # eps_xx + eps_yy (plane strain)
    exx = ((1 - NU ** 2) * sxx - NU * (1 + NU) * syy) / E
    eyy = ((1 - NU ** 2) * syy - NU * (1 + NU) * sxx) / E
    exy = (1 + NU) * sxy / E
    del tr
    return 0.5 * (sxx * exx + syy * eyy + 2 * sxy * exy)


def energy_volume():
    """Polar integral over the three unit squares, each split at its diagonal through (1,1)."""
    tot = 0.0
    # rays from (1, 1): distance to the outer boundary of each square
    pieces = [  # (theta range, R(theta))
        ((np.pi / 2, 3 * np.pi / 4), lambda t: 1 / np.cos(t - np.pi / 2)),      # top square, to y=2
        ((3 * np.pi / 4, np.pi), lambda t: 1 / abs(np.cos(t))),                 # top square, to x=0
        ((np.pi, 5 * np.pi / 4), lambda t: 1 / abs(np.cos(t))),                 # bottom-left, x=0
        ((5 * np.pi / 4, 3 * np.pi / 2), lambda t: 1 / abs(np.sin(t))),         # bottom-left, y=0
        ((3 * np.pi / 2, 7 * np.pi / 4), lambda t: 1 / abs(np.sin(t))),         # bottom-right, y=0
        ((7 * np.pi / 4, 2 * np.pi), lambda t: 1 / abs(np.cos(t))),             # bottom-right, x=2
    ]
    for (a, b), R in pieces:
        tot += _polar_square(energy_density, a, b, R)
    return tot


def _gk(f, cx, cy, ell, wedge):
    """Weighted average with exp(-(d/ell)^2) about (cx, cy); wedge: polar around (1, 1)."""
    k = lambda x, y: np.exp(-(np.hypot(x - cx, y - cy) / ell) ** 2)
    if wedge:
        R = lambda t: 1.0                     # the material within r = 1 is the 270 wedge
        num = _polar_square(lambda x, y: f(x, y) * k(x, y), TH0, TH1, R, nth=240, nr=160)
        den = _polar_square(k, TH0, TH1, R, nth=240, nr=160)
        return num / den
    xs, wx = _gl(cx - 6 * ell, cx + 6 * ell, 200)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    W = np.outer(wx, wx)
    return float((f(X, Y) * k(X, Y) * W).sum() / (k(X, Y) * W).sum())


def _disc(f, rad):
    R = lambda t: rad
    return (_polar_square(f, TH0, TH1, R, nth=240, nr=160)
            / _polar_square(lambda x, y: np.ones_like(x), TH0, TH1, R, nth=240, nr=160))


def exact_values():
    ux, _ = displacement(*P_LEFT)
    _, vy = displacement(*P_RIGHT)
    eb, ev = energy_boundary(), energy_volume()
    return {"strain_energy": eb, "strain_energy_volume_check": ev,
            "energy_two_ways_rel": abs(eb - ev) / abs(eb),
            "u_mid_left": float(ux), "v_mid_right": float(vy),
            "corner_gk_avg_von_mises": _gk(von_mises, 1.0, 1.0, L.ELL, True),
            "corner_disc_avg_von_mises": _disc(von_mises, L.R_DISC),
            "far_gk_avg_syy": _gk(lambda x, y: stress(x, y)[1], 0.5, 0.5, L.ELL, False),
            "peak_von_mises": float("inf"),
            "lam": [m[0] for m in modes()], "weights": list(WEIGHTS),
            "checks": self_checks()}


# ---------------------------------------------------------------------- finite elements
def _outer(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[0], 2.0)
            | np.isclose(x[1], 2.0))


def solve_level(n, g=1.0):
    import pypardiso
    t0 = time.time()
    msh = mesh(n, g)
    b = fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4)
    K = fem.asm(stiffness, b)
    D = b.get_dofs(lambda x: _outer(x))
    d1, d2 = D.all("u^1"), D.all("u^2")
    xD = np.zeros(b.N)
    xD[d1] = displacement(*b.doflocs[:, d1])[0]
    xD[d2] = displacement(*b.doflocs[:, d2])[1]
    Dall = np.concatenate([d1, d2])
    u = fem.solve(*fem.condense(K, np.zeros(b.N), x=xD, D=Dall),
                  solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
    energy = float(0.5 * u @ (K @ u))
    gu = b.interpolate(u).grad
    eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
    tr = eps[0, 0] + eps[1, 1]
    sxx, syy = 2 * MU * eps[0, 0] + LAM * tr, 2 * MU * eps[1, 1] + LAM * tr
    szz, sxy = LAM * tr, 2 * MU * eps[0, 1]
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
    x, wq = b.global_coordinates().value, b.dx
    d = np.hypot(x[0] - 1.0, x[1] - 1.0)
    k = wq * np.exp(-(d / L.ELL) ** 2)
    disc = d < L.R_DISC
    df = np.hypot(x[0] - 0.5, x[1] - 0.5)
    kf = wq * np.exp(-(df / L.ELL) ** 2)
    U = u[b.nodal_dofs].T

    def node(px, py):
        j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
        if np.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9:
            return U[j]
        # AMENDMENT after the pre-registration (2026-11-23, before any scoring): graded
        # meshes have no node at the query points, so the displacement is interpolated in
        # its element (the same Q2 field). Uniform meshes always take the branch above,
        # so their values are unchanged (to round-off, checked in lshape_exact_preregister).
        lin = fem.Basis(fem.MeshQuad(msh.p, msh.t), fem.ElementVector(fem.ElementQuad2()),
                        intorder=4)
        assert np.allclose(lin.doflocs, b.doflocs)
        return lin.probes(np.array([[px], [py]])) @ u
    i = np.unravel_index(np.argmax(vm), vm.shape)
    return {"n": n, "g": g, "h": 1.0 / n, "n_elem": int(msh.t.shape[1]), "n_dof": int(u.size),
            "strain_energy": energy,
            "u_mid_left": float(node(*P_LEFT)[0]), "v_mid_right": float(node(*P_RIGHT)[1]),
            "corner_gk_avg_von_mises": float((vm * k).sum() / k.sum()),
            "corner_disc_avg_von_mises": float((vm * wq)[disc].sum() / wq[disc].sum()),
            "far_gk_avg_syy": float((syy * kf).sum() / kf.sum()),
            "peak_von_mises": float(vm.max()),
            "peak_location": [float(x[0][i]), float(x[1][i])],
            "wall_clock_s": time.time() - t0}


def run(levels=LEVELS, g=1.0):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["n"], r["g"]) for r in rows}
    for n in levels:
        if (n, g) in done:
            continue
        r = solve_level(n, g)
        rows.append(r)
        rows.sort(key=lambda r: (r["g"], r["n"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"n={n} g={g} dof={r['n_dof']} ({r['wall_clock_s']:.1f} s) "
              f"E={r['strain_energy']:.10g}", flush=True)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "exact":
        ex = exact_values()
        json.dump(ex, open(EXACT, "w"), indent=1)
        print(json.dumps(ex, indent=1))
    elif argv and argv[0] == "solve":
        run()
        for g in GRADINGS:
            run([STRATEGY_N], g)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
