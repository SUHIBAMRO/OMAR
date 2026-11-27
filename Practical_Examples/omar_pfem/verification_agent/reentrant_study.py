"""Re-entrant corner: uniform vs graded h-refinement, p- and hp-refinement against the exact
solution (27 Nov, Professor Rabczuk). Rules and predictions: reentrant_preregistration.json
(frozen before any solve).

    python -m omar_pfem.verification_agent.reentrant_study run
    python -m omar_pfem.verification_agent.reentrant_study score
"""
import json
import math
import os
import sys
import time

import numpy as np
import skfem as fem

from omar_pfem.verification_agent import lshape as L
from omar_pfem.verification_agent import lshape_exact as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reentrant_result.json")
PRE = os.path.join(HERE, "reentrant_preregistration.json")
E, NU = L.E, L.NU
FAMILIES = {
    "U1": {"p": 1, "g": 1.0, "levels": [4, 8, 16, 32, 64]},
    "U2": {"p": 2, "g": 1.0, "levels": [4, 8, 16, 32, 64]},
    "G2_1.5": {"p": 2, "g": 1.5, "levels": [4, 8, 16, 32, 64]},
    "G1_2": {"p": 1, "g": 2.0, "levels": [4, 8, 16, 32, 64]},
    "G2_4": {"p": 2, "g": 4.0, "levels": [4, 8, 16, 32, 64]},
    "P": {"n": 2, "levels": [1, 2, 3, 4, 5, 6, 7, 8]},
    "HP": {"sigma": 0.15, "levels": [1, 2, 3, 4, 5, 6]},
}


def coords_graded(n, g):
    return L._coords(n, g)


def coords_geometric(m, sigma):
    lo = [1.0 - sigma ** k for k in range(0, m + 1)]           # 0, 1 - s, ..., 1 - s^m
    hi = [1.0 + sigma ** k for k in range(m, -1, -1)]          # 1 + s^m, ..., 2
    return np.array(lo + [1.0] + hi)


def mesh_from(x):
    N = len(x)
    n = (N - 1) // 2
    X_, Y_ = np.meshgrid(x, x, indexing="ij")
    idx = lambda i, j: i * N + j
    t = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(2 * n) for j in range(2 * n) if not (i >= n and j >= n)]
    t = np.array(t).T
    used = np.unique(t)
    remap = -np.ones(N * N, int)
    remap[used] = np.arange(len(used))
    return fem.MeshQuad(np.array([X_.ravel()[used], Y_.ravel()[used]]), remap[t])


def element(p):
    return {1: fem.ElementQuad1(), 2: fem.ElementQuad2()}.get(p) or fem.ElementQuadP(p)


def solve(msh, p):
    import pypardiso
    t0 = time.time()
    b = fem.Basis(msh, fem.ElementVector(element(p)), intorder=2 * p + 4)
    K = fem.asm(L.stiffness, b)
    D = b.get_dofs(lambda x: np.isclose(x[0], 0) | np.isclose(x[1], 0) | np.isclose(x[0], 2)
                   | np.isclose(x[1], 2)).all()
    # Dirichlet data: the L2 projection of the exact field (works for every element family)
    xD = b.project(lambda x: np.array(X.displacement(x[0], x[1])))
    u = fem.solve(*fem.condense(K, np.zeros(b.N), x=xD, D=D),
                  solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
    gu = b.interpolate(u).grad
    eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
    tr = eps[0, 0] + eps[1, 1]
    sh = [2 * L.MU * eps[0, 0] + L.LAM * tr, 2 * L.MU * eps[1, 1] + L.LAM * tr, 2 * L.MU * eps[0, 1]]
    x = b.global_coordinates().value
    sx = X.stress(x[0], x[1])
    ds = [sx[0] - sh[0], sx[1] - sh[1], sx[2] - sh[2]]
    # plane strain compliance for the in-plane stress difference
    dexx = ((1 - NU ** 2) * ds[0] - NU * (1 + NU) * ds[1]) / E
    deyy = ((1 - NU ** 2) * ds[1] - NU * (1 + NU) * ds[0]) / E
    dexy = (1 + NU) * ds[2] / E
    dens = ds[0] * dexx + ds[1] * deyy + 2 * ds[2] * dexy
    err2 = float((dens * b.dx).sum())
    exact = json.load(open(X.EXACT))
    energy = float(0.5 * u @ (K @ u))
    return {"n_dof": int(b.N), "n_elem": int(msh.t.shape[1]), "p": p,
            "energy_error_rel": math.sqrt(max(err2, 0.0) / (2 * exact["strain_energy"])),
            "strain_energy": energy,
            "strain_energy_error_rel": abs(energy - exact["strain_energy"]) / exact["strain_energy"],
            "seconds": time.time() - t0}


def run():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, f in FAMILIES.items():
        rows = res.setdefault(name, [])
        done = {r["level"] for r in rows}
        for lv in f["levels"]:
            if lv in done:
                continue
            if name == "P":
                msh, p = mesh_from(coords_graded(f["n"], 1.0)), lv
            elif name == "HP":
                msh, p = mesh_from(coords_geometric(lv, f["sigma"])), lv
            else:
                msh, p = mesh_from(coords_graded(lv, f["g"])), f["p"]
            r = solve(msh, p)
            r["level"] = lv
            rows.append(r)
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"{name:7s} level {lv:3d}: DOF {r['n_dof']:7d}  e_E {r['energy_error_rel']:.3e}  "
                  f"({r['seconds']:.1f} s)", flush=True)
    return res


def rate(rows):
    r = sorted(rows, key=lambda r: r["n_dof"])[-3:]
    N = np.log([x["n_dof"] for x in r])
    e = np.log([x["energy_error_rel"] for x in r])
    return float(-np.polyfit(N, e, 1)[0])


def score():
    res, pre = json.load(open(OUT)), json.load(open(PRE))["predictions"]
    res = {k: v for k, v in res.items() if "posthoc" not in k}    # labelled extensions do not score
    out = {}
    beta = {k: rate(v) for k, v in res.items() if k != "HP"}
    ok1 = all(abs(beta[k] - pre["R1_uniform"][k]) <= pre["R1_uniform"]["tol"] for k in ("U1", "U2"))
    out["R1_uniform"] = (ok1, {k: beta[k] for k in ("U1", "U2")})
    ok2 = all(abs(beta[k] - pre["R2_graded"][k]) <= pre["R2_graded"]["tol"]
              for k in ("G2_1.5", "G1_2", "G2_4"))
    out["R2_graded"] = (ok2, {k: beta[k] for k in ("G2_1.5", "G1_2", "G2_4")})
    out["R3_p"] = (abs(beta["P"] - pre["R3_p"]["P"]) <= pre["R3_p"]["tol"], {"P": beta["P"]})
    hp = sorted(res["HP"], key=lambda r: r["n_dof"])
    x = np.array([r["n_dof"] ** (1 / 3) for r in hp[-3:]])
    y = np.log([r["energy_error_rel"] for r in hp[-3:]])
    c = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fin = {k: min(v, key=lambda r: r["energy_error_rel"])["energy_error_rel"] for k, v in res.items()}
    out["R4_hp"] = (bool(r2 >= 0.98 and fin["HP"] < min(v for k, v in fin.items() if k != "HP")),
                    {"R2": r2, "finest_errors": fin})
    last = {k: max(v, key=lambda r: r["level"]) for k, v in res.items()}
    out["R5_order_of_families"] = (bool(fin["HP"] < fin["G2_4"] < fin["U2"] and
                                        last["HP"]["n_dof"] == min(r["n_dof"] for r in last.values())),
                                   {k: (last[k]["n_dof"], last[k]["energy_error_rel"]) for k in last})
    out["_rates_beta"] = beta
    json.dump({k: {"confirmed": v[0], "evidence": v[1]} if not k.startswith("_") else v
               for k, v in out.items()}, open(os.path.join(HERE, "reentrant_score.json"), "w"),
              indent=1, default=float)
    for k, v in out.items():
        print(k, v)
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    {"run": run, "score": score}[sys.argv[1]]()
