"""Edge-cracked panel: a second HELD-OUT test (2026-11-22) of the current rules
with the proposed singular-order extension, in a singularity class the agent has
not seen (a crack; regularity "crack"). Spec and predictions are committed
before any solve (crack_preregistration.json).

Domain [0, 2] x [-1, 1] with a traction-free edge crack along y = 0 from x = 0
to the tip at (1, 0). Plane strain, small-strain linear elasticity, E = 1000,
nu = 0.3. Bottom edge y = -1 clamped; top edge y = 1 loaded by a uniform
traction t_y = 1 (mode-I opening); all other edges and both crack faces free.

Singularities: the crack tip, lam = 0.5 (Williams; a double root of the corner
determinant, so singularity.smallest_exponent, which brackets sign changes,
cannot find it; char_det(0.5) = 0 is checked); clamped/free corners at (0, -1)
and (2, -1), lam = 0.7112 (weaker).

Mesh: structured Q2 (9-node), n x n elements per unit square (h = 1/n), the
crack faces carry separate nodes; grading g > 1 clusters the 1D coordinates
toward x = 1 and y = 0.

    python -m omar_pfem.verification_agent.crack [--ns 2,4,8,16,32,64,128] [--strategies 1.5,2.0]
"""
import argparse
import json
import os
import time

import numpy as np
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "crack_ladder.json")
E, NU, T = 1000.0, 0.3, 1.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
TIP = (1.0, 0.0)
FAR = (1.5, 0.5)
ELL = 0.1
R_DISC = 0.2


def corner_exponent():
    from omar_pfem.verification_agent.singularity import char_det
    assert abs(char_det(0.5, 2 * np.pi, "free", "free", NU)) < 1e-12
    return 0.5


def _coords(n, g):
    s = np.linspace(0.0, 1.0, n + 1)
    lo = 1.0 - (1.0 - s) ** g
    hi = 1.0 + s ** g
    return np.concatenate([lo, hi[1:]])


def mesh(n, g=1.0):
    xs = _coords(n, g)                  # [0, 2], clustered toward 1
    ys = _coords(n, g) - 1.0            # [-1, 1], clustered toward 0
    N = len(xs)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    p = [X.ravel(), Y.ravel()]
    idx = lambda i, j: i * N + j
    # duplicate the nodes on the crack (y = 0, x < 1) for the upper side
    jc = n                                              # row index of y = 0
    dup = {}
    for i in range(n):                                  # x index 0 .. n-1 -> x < 1
        dup[idx(i, jc)] = len(p[0]) + len(dup)
    px = np.concatenate([p[0], [X[i, jc] for i in range(n)]])
    py = np.concatenate([p[1], [Y[i, jc] for i in range(n)]])
    t = []
    for i in range(2 * n):
        for j in range(2 * n):
            q = [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            if j == jc:                                  # element above the crack line
                q = [dup.get(v, v) for v in q[:2]] + q[2:]
            t.append(q)
    return fem.MeshQuad2.from_mesh(fem.MeshQuad(np.array([px, py]), np.array(t).T))


@fem.BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2 * MU * ddot(eu, ev) + LAM * trace(eu) * trace(ev)


def _vb(msh, elements=None):
    return fem.Basis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4, elements=elements)


def solve_level(n, g=1.0, chunk=20000):
    import pypardiso
    t0 = time.time()
    msh = mesh(n, g)
    nel = msh.t.shape[1]
    chunks = np.array_split(np.arange(nel), int(np.ceil(nel / chunk)))
    b0 = _vb(msh, chunks[0])
    K = None
    for ch in chunks:
        Kc = fem.asm(stiffness, _vb(msh, ch))
        K = Kc if K is None else K + Kc
    top = fem.FacetBasis(msh, fem.ElementVector(fem.ElementQuad2()), intorder=4,
                         facets=msh.facets_satisfying(lambda x: np.isclose(x[1], 1.0)))
    f = fem.asm(fem.LinearForm(lambda v, w: T * v.value[1]), top)
    fixed = b0.get_dofs(lambda x: np.isclose(x[1], -1.0)).all()
    u = fem.solve(*fem.condense(K, f, D=fixed),
                  solver=lambda A, b: pypardiso.spsolve(A.tocsr(), b))
    energy = float(0.5 * u @ (K @ u))
    del K
    acc = dict(gk_w=0.0, gk_s=0.0, d_w=0.0, d_s=0.0, far_w=0.0, far_s=0.0, vm=-np.inf, vm_at=None)
    for ch in chunks:
        bc = _vb(msh, ch)
        gu = bc.interpolate(u).grad
        eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
        tr = eps[0, 0] + eps[1, 1]
        sxx, syy = 2 * MU * eps[0, 0] + LAM * tr, 2 * MU * eps[1, 1] + LAM * tr
        szz, sxy = LAM * tr, 2 * MU * eps[0, 1]
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
        x, wq = bc.global_coordinates().value, bc.dx
        d = np.hypot(x[0] - TIP[0], x[1] - TIP[1])
        k = wq * np.exp(-(d / ELL) ** 2)
        acc["gk_w"] += (vm * k).sum()
        acc["gk_s"] += k.sum()
        disc = d < R_DISC
        acc["d_w"] += (vm * wq)[disc].sum()
        acc["d_s"] += wq[disc].sum()
        df = np.hypot(x[0] - FAR[0], x[1] - FAR[1])
        kf = wq * np.exp(-(df / ELL) ** 2)
        acc["far_w"] += (syy * kf).sum()
        acc["far_s"] += kf.sum()
        i = np.unravel_index(np.argmax(vm), vm.shape)
        if vm[i] > acc["vm"]:
            acc["vm"], acc["vm_at"] = float(vm[i]), [float(x[0][i]), float(x[1][i])]
    U = u[b0.nodal_dofs].T
    nv = msh.p.shape[1]

    def nodes_at(px, py):
        return [j for j in range(nv) if np.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9]
    mouth = nodes_at(0.0, 0.0)                     # lower and upper crack-face node
    assert len(mouth) == 2, mouth
    vlo, vup = sorted(U[j][1] for j in mouth)
    corner = nodes_at(2.0, 1.0)[0]
    return {"n": n, "g": g, "h": 1.0 / n, "n_elem": int(nel), "n_dof": int(u.size),
            "strain_energy": energy,
            "u_cod_mouth": float(vup - vlo),
            "v_top_right": float(U[corner][1]),
            "tip_gk_avg_von_mises": acc["gk_w"] / acc["gk_s"],
            "tip_disc_avg_von_mises": acc["d_w"] / acc["d_s"],
            "far_gk_avg_syy": acc["far_w"] / acc["far_s"],
            "peak_von_mises": acc["vm"], "peak_location": acc["vm_at"],
            "wall_clock_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,4,8,16,32,64,128")
    ap.add_argument("--strategies", default="1.5,2.0", help="grading g at n = 16")
    a = ap.parse_args()
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {(r["n"], r["g"]) for r in rows}
    todo = [(int(n), 1.0) for n in a.ns.split(",") if n]
    todo += [(16, float(g)) for g in a.strategies.split(",") if g] + [(16, 1.0)]
    for n, g in todo:
        if (n, g) in done:
            continue
        r = solve_level(n, g)
        rows.append(r)
        done.add((n, g))
        json.dump(rows, open(CACHE, "w"), indent=1)
        print(f"n={n:4d} g={g:.1f} elem {r['n_elem']:6d} energy {r['strain_energy']:.10e} "
              f"peak {r['peak_von_mises']:.4f} at {r['peak_location']} ({r['wall_clock_s']:.1f} s)",
              flush=True)


if __name__ == "__main__":
    main()
