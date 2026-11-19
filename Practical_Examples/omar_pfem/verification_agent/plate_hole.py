"""Plate with a hole (the OpenPINN/PINO demonstration case): a HELD-OUT
benchmark for the verification agent.

2 m x 2 m plate, circular hole of radius 0.25 m at (1, 1), plane stress,
linear elasticity, E = 210 GPa, nu = 0.3. Left edge x = 0 fixed, right edge
x = 2 loaded by a uniform traction t_x = 1 MPa, top, bottom and hole free
(as in Wang et al., OpenPINN/PINO, Sec. 5.3, whose reference is one FEM mesh
with h = 0.025 and no convergence estimate).

Mesh: structured "polar" Q2 mesh around the hole, x = c + rho(s, theta)
(cos theta, sin theta) with rho = (1 - s^g) r_hole + s^g R(theta) and
R(theta) = 1 / max(|cos|, |sin|) the distance from the centre to the square
along the ray. N_theta = 8 m elements around (the square's corners are nodes),
N_r = 3 m radially; g = 1 uniform, g > 1 grades toward the hole. Q2
(9-node) elements; mid-side nodes on the exact circle.

Corners: the fixed edge meets the free top and bottom edges at (0,0) and
(0,2): 90-degree clamped/free corners (singular, plane stress). The hole is a
smooth, finite-radius concentration far from them.

    python -m omar_pfem.verification_agent.plate_hole [--ms 2,4,8,16,32,64]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse.linalg as spla
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "plate_hole_ladder.json")
E, NU, T = 210e9, 0.3, 1e6
C, RH, L = (1.0, 1.0), 0.25, 2.0
MU = E / (2 * (1 + NU))
LAM_PS = E * NU / (1 - NU ** 2)               # plane-stress lambda
HOLE_TOP = (1.0, 1.25)                        # sigma_xx peak of the Kirsch-type field
R_REGION = 0.1                                # sharp region: within 0.1 of the hole top
KERNEL_ELL = 0.05


def corner_exponent():
    from omar_pfem.verification_agent.singularity import smallest_exponent
    return smallest_exponent(np.pi / 2, "clamped", "free", NU, "stress")


def mesh(m, g=1.0):
    """Periodic structured Q2 mesh: vertices (s_i, theta_j), j modulo N_theta;
    every mid-side and centre node placed by the exact map from its
    parametric coordinates (theta unwrapped across the seam)."""
    nt, nr = 8 * m, 3 * m
    s1 = np.linspace(0.0, 1.0, nr + 1)
    t1 = -np.pi / 4 + 2 * np.pi * np.arange(nt) / nt
    S, TH = np.meshgrid(s1, t1, indexing="ij")              # (nr+1, nt)
    S, TH = S.ravel(), TH.ravel()
    idx = lambda i, j: i * nt + (j % nt)
    t = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                  for i in range(nr) for j in range(nt)]).T

    def X(s, th):
        R = 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
        sg = s ** g
        rho = (1 - sg) * RH + sg * R
        return np.array([C[0] + rho * np.cos(th), C[1] + rho * np.sin(th)])

    m1 = fem.MeshQuad(X(S, TH), t)
    m2 = fem.MeshQuad2.from_mesh(m1)
    nv, nf = m1.p.shape[1], m1.facets.shape[1]

    def unwrap_mean(ths):
        ref = ths[0]
        d = (ths - ref + np.pi) % (2 * np.pi) - np.pi
        return ref + d.mean(axis=0)
    fa, fb = m1.facets
    s_f = 0.5 * (S[fa] + S[fb])
    th_f = unwrap_mean(np.stack([TH[fa], TH[fb]]))
    s_c = S[m1.t].mean(axis=0)
    th_c = unwrap_mean(TH[m1.t])
    dof = m2.doflocs.copy()
    # check the assumed ordering (vertices, facets, elements) on the linear midpoints
    assert np.allclose(dof[:, nv:nv + nf], 0.5 * (m1.p[:, fa] + m1.p[:, fb]))
    dof[:, nv:nv + nf] = X(s_f, th_f)
    dof[:, nv + nf:] = X(s_c, th_c)
    return fem.MeshQuad2(dof, m1.t)


@fem.BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2 * MU * ddot(eu, ev) + LAM_PS * trace(eu) * trace(ev)


def solve_level(m, g=1.0):
    t0 = time.time()
    msh = mesh(m, g)
    ib = fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4)
    K = fem.asm(stiffness, ib)
    right = fem.FacetBasis(msh, fem.ElementVector(fem.ElementQuad2()),
                           facets=msh.facets_satisfying(lambda x: np.isclose(x[0], L)), intorder=4)
    f = fem.asm(fem.LinearForm(lambda v, w: T * v.value[0]), right)
    fixed = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    u = fem.solve(*fem.condense(K, f, D=fixed), solver=spla.spsolve)
    # stresses at the quadrature points
    gu = ib.interpolate(u).grad                               # (2, 2, nel, nq)
    eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
    tr = eps[0, 0] + eps[1, 1]
    sxx = 2 * MU * eps[0, 0] + LAM_PS * tr
    syy = 2 * MU * eps[1, 1] + LAM_PS * tr
    sxy = 2 * MU * eps[0, 1]
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
    x, wq = ib.global_coordinates().value, ib.dx
    d = np.hypot(x[0] - HOLE_TOP[0], x[1] - HOLE_TOP[1])
    near = d < R_REGION
    k = wq * np.exp(-(d / KERNEL_ELL) ** 2)
    ux, uy = u[ib.get_dofs().all("u^1")], u[ib.get_dofs().all("u^2")]
    U = u[ib.nodal_dofs].T
    def node(px, py):
        j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
        assert np.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9, (px, py)
        return U[j]
    u_rm = node(2.0, 1.0)
    u_tr = node(2.0, 2.0)
    ipk = np.unravel_index(np.argmax(vm), vm.shape)
    return {"m": m, "g": g, "h": 1.0 / m, "n_elem": int(msh.t.shape[1]), "n_dof": int(ib.N),
            "wall_clock_s": time.time() - t0,
            "u_right_mid": float(u_rm[0]), "v_top_right": float(u_tr[1]),
            "max_disp_mag": float(np.sqrt(ux ** 2 + uy ** 2).max()),
            "strain_energy": float(0.5 * u @ (K @ u)),
            "hole_avg_sigma_xx": float((sxx * wq)[near].sum() / wq[near].sum()),
            "hole_gk_avg_sigma_xx": float((sxx * k).sum() / k.sum()),
            "hole_peak_sigma_xx": float(sxx[near].max()),
            "peak_von_mises": float(vm.max()),
            "peak_location": [float(x[0][ipk]), float(x[1][ipk])],
            "hole_region_points": int(near.sum())}


def run_ladder(ms, g=1.0):
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["m"], r["g"]) for r in rows}
    for m in ms:
        if (m, g) in done:
            continue
        row = solve_level(m, g)
        rows.append(row)
        rows.sort(key=lambda r: (r["g"], r["m"]))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"m={m:<3} g={g} ({row['wall_clock_s']:.0f}s) " + "  ".join(
            f"{k}={v:.6g}" for k, v in row.items() if isinstance(v, float)
            and k not in ("h", "g", "wall_clock_s")), flush=True)
    return sorted((r for r in rows if r["m"] in ms and r["g"] == g), key=lambda r: r["m"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="2,4,8,16,32,64")
    ap.add_argument("--strategies", default="1.5,2.0", help="radial grading g at m = 16")
    a = ap.parse_args(argv)
    run_ladder([int(v) for v in a.ms.split(",")])
    for g in [float(v) for v in a.strategies.split(",") if v]:
        run_ladder([16], g)


if __name__ == "__main__":
    main()
