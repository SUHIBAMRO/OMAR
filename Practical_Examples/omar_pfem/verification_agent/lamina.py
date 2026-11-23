"""Open-hole composite lamina of the CFM paper (Lone, Prabhakaran, Rabczuk,
Zhuang, arXiv:2611.22187, Sec. 9.3): the NUMERICAL half of their constitutive
range. HELD-OUT benchmark for the verification agent (pre-registered).

Geometry and load as in the paper: elliptical planform, semi-axes 50 mm (x)
and 10 mm (y), thickness 1 mm, central circular hole of radius 3 mm; fibres at
15 degrees to the long axis; boundary tractions of a homogeneous uniaxial
stress along x with resultant P (t = sigma0 n_x e_x, sigma0 = P / (2 b t));
three displacement components at the hole fixed to remove the rigid motions
of the balanced load. Material: the paper's reference law psi_ref, linear
orthotropic with the WWFE-I moduli of AS4/3501-6 (E1 = 126 GPa, E2 = 11 GPa,
G12 = 6.6 GPa, nu12 = 0.28), plane stress. Linear, so every QoI below is per
unit load and the load level (0.6 P_D in the paper) does not matter.

Mesh: structured mapped Q2 (9-node) mesh between the hole circle and the
ellipse, x = rho(s, theta) (cos theta, sin theta), rho = (1 - s^g) r_h + s^g
R(theta), R(theta) the ellipse radius along the ray; N_theta = 16 m around,
N_r = 8 m radially (m = 4 matches the paper's 64 x 32 divisions); mid-side
nodes on the exact curves; g > 1 grades toward the hole.

QoIs (the paper does not define its zones; these definitions are ours and
were fixed in the pre-registration before any solve):
  compliance          f . u / P^2 (work-conjugate secant compliance), mm/N
  hole_gamma12_avg    area mean of the shear strain gamma12 (material axes)
                      over the annulus 3 <= r <= 6 mm ("hole zone")
  hole_e22_avg        same for the transverse strain e22
  far_gamma12_avg     area mean of gamma12 over the disc of radius 3 mm at
                      (25, 0) ("far field")
  hole_gk_gamma12_avg, hole_gk_e22_avg
                      Gaussian-weighted versions, weight exp(-((r - 3)/1.5)^2)
                      (smooth weights converged with a rate in B1/B3/plate)
  peak_e22            max e22 over all Gauss points (hole edge, finite-radius
                      concentration, no singular point: smooth boundary)
  u_tip               u_x at (50, 0)

    python -m omar_pfem.verification_agent.lamina [--ms 1,2,4,8,16,32]
"""
import argparse
import json
import os
import time

import numpy as np
import skfem as fem

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "lamina_ladder.json")
A_ELL, B_ELL, RH, THICK = 50.0, 10.0, 3.0, 1.0
E1, E2, G12, NU12 = 126e3, 11e3, 6.6e3, 0.28          # N/mm^2 (MPa)
THETA_F = np.deg2rad(15.0)
P = 1.0                                               # N (linear problem)
SIGMA0 = P / (2 * B_ELL * THICK)
HOLE_ZONE = (3.0, 6.0)
FAR_CENTRE, FAR_RADIUS = (25.0, 0.0), 3.0
GK_LEN = 1.5


def stiffness_matrix(theta=THETA_F):
    """Plane-stress Qbar (Voigt, engineering shear) in global axes."""
    nu21 = NU12 * E2 / E1
    d = 1 - NU12 * nu21
    Q11, Q22, Q12, Q66 = E1 / d, E2 / d, NU12 * E2 / d, G12
    c, s = np.cos(theta), np.sin(theta)
    Qb = np.zeros((3, 3))
    Qb[0, 0] = Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * s**4
    Qb[1, 1] = Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * c**4
    Qb[0, 1] = Qb[1, 0] = (Q11 + Q22 - 4 * Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
    Qb[0, 2] = Qb[2, 0] = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
    Qb[1, 2] = Qb[2, 1] = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3
    Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s**2 * c**2 + Q66 * (s**4 + c**4)
    return Qb


QB = stiffness_matrix()


def ellipse_radius(th):
    return A_ELL * B_ELL / np.sqrt((B_ELL * np.cos(th)) ** 2 + (A_ELL * np.sin(th)) ** 2)


def mesh(m, g=1.5, rh=RH):
    nt, nr = 16 * m, 8 * m
    s1 = np.linspace(0.0, 1.0, nr + 1)
    t1 = 2 * np.pi * np.arange(nt) / nt
    S, TH = np.meshgrid(s1, t1, indexing="ij")
    S, TH = S.ravel(), TH.ravel()
    idx = lambda i, j: i * nt + (j % nt)
    t = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                  for i in range(nr) for j in range(nt)]).T

    def X(s, th):
        rho = (1 - s ** g) * rh + s ** g * ellipse_radius(th)
        return np.array([rho * np.cos(th), rho * np.sin(th)])

    m1 = fem.MeshQuad(X(S, TH), t)
    m2 = fem.MeshQuad2.from_mesh(m1)
    nv, nf = m1.p.shape[1], m1.facets.shape[1]

    def unwrap_mean(ths):
        ref = ths[0]
        d = (ths - ref + np.pi) % (2 * np.pi) - np.pi
        return ref + d.mean(axis=0)
    fa, fb = m1.facets
    dof = m2.doflocs.copy()
    assert np.allclose(dof[:, nv:nv + nf], 0.5 * (m1.p[:, fa] + m1.p[:, fb]))
    dof[:, nv:nv + nf] = X(0.5 * (S[fa] + S[fb]), unwrap_mean(np.stack([TH[fa], TH[fb]])))
    dof[:, nv + nf:] = X(S[m1.t].mean(axis=0), unwrap_mean(TH[m1.t]))
    return fem.MeshQuad2(dof, m1.t)


def _voigt(gu):
    return np.array([gu[0, 0], gu[1, 1], gu[0, 1] + gu[1, 0]])


@fem.BilinearForm
def stiffness(u, v, w):
    eu, ev = _voigt(u.grad), _voigt(v.grad)
    return THICK * sum(QB[i, j] * eu[j] * ev[i] for i in range(3) for j in range(3))


def material_strains(gu, theta=THETA_F):
    """(e11, e22, gamma12) in material axes from the displacement gradient."""
    exx, eyy, gxy = gu[0, 0], gu[1, 1], gu[0, 1] + gu[1, 0]
    c, s = np.cos(theta), np.sin(theta)
    e11 = c * c * exx + s * s * eyy + s * c * gxy
    e22 = s * s * exx + c * c * eyy - s * c * gxy
    g12 = -2 * s * c * exx + 2 * s * c * eyy + (c * c - s * s) * gxy
    return e11, e22, g12


# element factory and quadrature order: Q2 by default; goal_oriented.py swaps in a
# hierarchical degree-3 element (same Q2 geometry) for the enriched solution
ELEMENT = {"make": fem.ElementQuad2, "intorder": 4, "facet_intorder": 6}


def _basis(msh, elements=None):
    return fem.Basis(msh, fem.ElementVector(ELEMENT["make"]()), intorder=ELEMENT["intorder"],
                     elements=elements)


def solve_level(m, g=1.5, chunk=15000, theta_deg=None, rh=None):
    """theta_deg, rh: design parameters (active_learning.py); defaults = the
    pre-registered problem (15 deg, 3 mm). The hole zone and the Gaussian
    weight scale with rh: annulus rh <= r <= 2 rh, length rh / 2."""
    import pypardiso
    global QB
    theta = THETA_F if theta_deg is None else np.deg2rad(theta_deg)
    rh = RH if rh is None else rh
    QB = stiffness_matrix(theta)
    try:
        return _solve_level(m, g, chunk, theta, rh, pypardiso)
    finally:
        QB = stiffness_matrix()


def _solve_level(m, g, chunk, theta, rh, pypardiso):
    t0 = time.time()
    msh = mesh(m, g, rh)
    nel = msh.t.shape[1]
    chunks = np.array_split(np.arange(nel), int(np.ceil(nel / chunk)))
    b0 = _basis(msh, chunks[0])
    K = None
    for ch in chunks:
        Kc = fem.asm(stiffness, _basis(msh, ch))
        K = Kc if K is None else K + Kc
    outer = msh.facets_satisfying(lambda x: np.hypot(x[0], x[1]) > rh + 1e-6, boundaries_only=True)
    fb = fem.FacetBasis(msh, fem.ElementVector(ELEMENT["make"]()), facets=outer,
                        intorder=ELEMENT["facet_intorder"])
    f = fem.asm(fem.LinearForm(lambda v, w: THICK * SIGMA0 * w.n[0] * v.value[0]), fb)
    # rigid motions: u_x, u_y at (3, 0), u_y at (-3, 0)
    def nodal(px, py, comp):
        j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
        assert np.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9
        return b0.nodal_dofs[comp, j]
    fixed = np.array([nodal(rh, 0, 0), nodal(rh, 0, 1), nodal(-rh, 0, 1)])
    u = fem.solve(*fem.condense(K, f, D=fixed), solver=lambda A, b: pypardiso.spsolve(
        A.tocsr(), b))
    compliance = float(f @ u) / P ** 2
    res_rel = float(np.linalg.norm((K @ u - f)[np.setdiff1d(np.arange(len(f)), fixed)])
                    / np.linalg.norm(f))
    del K
    acc = {k: 0.0 for k in ("hz_w", "hz_g", "hz_e", "ff_w", "ff_g", "gk_w", "gk_g", "gk_e",
                            "area")}
    pk, pk_at = -np.inf, None
    for ch in chunks:
        bc = _basis(msh, ch)
        gu = bc.interpolate(u).grad
        _, e22, g12 = material_strains(gu, theta)
        x, wq = bc.global_coordinates().value, bc.dx
        r = np.hypot(x[0], x[1])
        hz = (r >= rh * HOLE_ZONE[0] / RH) & (r <= rh * HOLE_ZONE[1] / RH)
        ff = np.hypot(x[0] - FAR_CENTRE[0], x[1] - FAR_CENTRE[1]) <= FAR_RADIUS
        kg = wq * np.exp(-((r - rh) / (GK_LEN * rh / RH)) ** 2)
        acc["hz_w"] += wq[hz].sum()
        acc["hz_g"] += (g12 * wq)[hz].sum()
        acc["hz_e"] += (e22 * wq)[hz].sum()
        acc["ff_w"] += wq[ff].sum()
        acc["ff_g"] += (g12 * wq)[ff].sum()
        acc["gk_w"] += kg.sum()
        acc["gk_g"] += (g12 * kg).sum()
        acc["gk_e"] += (e22 * kg).sum()
        acc["area"] += wq.sum()
        i = np.unravel_index(np.argmax(e22), e22.shape)
        if e22[i] > pk:
            pk, pk_at = float(e22[i]), [float(x[0][i]), float(x[1][i])]
    j = int(np.argmin((msh.p[0] - A_ELL) ** 2 + msh.p[1] ** 2))
    return {"m": m, "g": g, "h": 1.0 / m, "n_elem": int(nel), "n_dof": int(b0.N),
            "wall_clock_s": time.time() - t0, "residual_rel": res_rel,
            "area_mm2": acc["area"],
            "hole_zone_area_mm2": acc["hz_w"],     # added 2026-11-23 (definition study)
            "compliance": compliance,
            "u_tip": float(u[b0.nodal_dofs[0, j]]),
            "hole_gamma12_avg": acc["hz_g"] / acc["hz_w"],
            "hole_e22_avg": acc["hz_e"] / acc["hz_w"],
            "far_gamma12_avg": acc["ff_g"] / acc["ff_w"],
            "hole_gk_gamma12_avg": acc["gk_g"] / acc["gk_w"],
            "hole_gk_e22_avg": acc["gk_e"] / acc["gk_w"],
            "peak_e22": pk, "peak_e22_location": pk_at}


QOIS = ("compliance", "u_tip", "hole_gamma12_avg", "hole_e22_avg", "far_gamma12_avg",
        "hole_gk_gamma12_avg", "hole_gk_e22_avg", "peak_e22")


def run_ladder(ms, g=1.5):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["m"], r["g"]) for r in rows}
    for m in ms:
        if (m, g) in done:
            continue
        row = solve_level(m, g)
        rows.append(row)
        rows.sort(key=lambda r: (r["g"], r["m"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"m={m:<3} g={g} ({row['wall_clock_s']:.0f}s, {row['n_dof']} dof) " + "  ".join(
            f"{k}={row[k]:.7g}" for k in QOIS), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8,16,32")
    ap.add_argument("--strategies", default="1.0,2.0", help="radial grading g at m = 8")
    a = ap.parse_args(argv)
    run_ladder([int(v) for v in a.ms.split(",")])
    for g in [float(v) for v in a.strategies.split(",") if v]:
        run_ladder([8], g)


if __name__ == "__main__":
    main()
