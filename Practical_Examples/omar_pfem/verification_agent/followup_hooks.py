"""Solver and code checks for problems without hand-written diagnosis facts
(2026-11-23), so that the diagnosis's S1/S2/S3 hypotheses can be tested instead of
staying NEEDS_DATA. Each check is computed once and cached in followup_hooks_result.json.

  S3 code     l_shape, edge_crack: a manufactured solution through the problem module's
              OWN bilinear form and element (scikit-fem Q2, unit square, n = 4..32);
              open_hole_lamina: the stored lamina MMS; thick_cylinder_3d, clamped_block_3d:
              the stored Q20 MMS (same torch-fem element and rule)
  S1 Newton   linear problems (L-shape, crack, lamina): one direct solve, no Newton
              tolerance; the 3D problems: one level re-solved at a 100x tighter tolerance
              and the quantities compared
  S2 linear   all five use a direct sparse solver (pypardiso or torch-fem "direct")

    python -m omar_pfem.verification_agent.followup_hooks
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "followup_hooks_result.json")


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    json.dump(d, open(OUT, "w"), indent=1)


def form_mms(module_name, ns=(4, 8, 16, 32)):
    """MMS through module.stiffness (plane-strain linear elasticity, Q2): manufactured
    u* = 0.01 (sin(pi x) sin(pi y), x y (1 - x) sin(pi y)) on the unit square, Dirichlet
    u* on the whole boundary, body force -div sigma(u*) from the module's MU, LAM."""
    import skfem as fem
    import sympy as sp
    mod = __import__(f"omar_pfem.verification_agent.{module_name}", fromlist=["stiffness"])
    x, y = sp.symbols("x y")
    u = [0.01 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
         0.01 * x * y * (1 - x) * sp.sin(sp.pi * y)]
    eps = [[(sp.diff(u[i], [x, y][j]) + sp.diff(u[j], [x, y][i])) / 2 for j in range(2)]
           for i in range(2)]
    tr = eps[0][0] + eps[1][1]
    sig = [[2 * mod.MU * eps[i][j] + (mod.LAM * tr if i == j else 0) for j in range(2)]
           for i in range(2)]
    f = [-(sp.diff(sig[i][0], x) + sp.diff(sig[i][1], y)) for i in range(2)]
    fu = [sp.lambdify((x, y), c, "numpy") for c in u]
    fg = [[sp.lambdify((x, y), sp.diff(c, v), "numpy") for v in (x, y)] for c in u]
    ff = [sp.lambdify((x, y), c, "numpy") for c in f]
    rows = []
    for n in ns:
        m = fem.MeshQuad2.from_mesh(fem.MeshQuad.init_tensor(np.linspace(0, 1, n + 1),
                                                            np.linspace(0, 1, n + 1)))
        b = fem.Basis(m, fem.ElementVector(fem.ElementQuad2()), intorder=6)
        K = fem.asm(mod.stiffness, b)
        F = fem.asm(fem.LinearForm(lambda v, w: ff[0](*w.x) * v.value[0]
                                   + ff[1](*w.x) * v.value[1]), b)
        D = b.get_dofs()
        uD = b.project(lambda X: np.array([fu[0](*X) + 0 * X[0], fu[1](*X) + 0 * X[0]]))
        uh = fem.solve(*fem.condense(K, F, x=uD, D=D))
        w = b.interpolate(uh)
        X = b.global_coordinates().value
        ue = np.array([fu[0](*X) + 0 * X[0], fu[1](*X) + 0 * X[0]])
        ge = np.array([[fg[i][j](*X) + 0 * X[0] for j in range(2)] for i in range(2)])
        dx = b.dx
        l2 = np.sqrt((((w.value - ue) ** 2).sum(0) * dx).sum() / ((ue ** 2).sum(0) * dx).sum())
        h1 = np.sqrt((((w.grad - ge) ** 2).sum((0, 1)) * dx).sum() / ((ge ** 2).sum((0, 1)) * dx).sum())
        rows.append({"n": n, "L2_rel": float(l2), "H1_rel": float(h1)})
    r = lambda k: float(np.log(rows[-2][k] / rows[-1][k]) / np.log(2))
    return {"module": module_name, "rows": rows, "rate_L2": r("L2_rel"), "rate_H1": r("H1_rel"),
            "as_expected": abs(r("L2_rel") - 3) < 0.25 and abs(r("H1_rel") - 2) < 0.25}


def tolerance_check(geometry):
    """One level re-solved with a 100x tighter Newton tolerance (3D problems)."""
    if geometry == "thick_cylinder_3d":
        from omar_pfem.verification_agent import cylinder3d as M
        k, qs = 4, ("strain_energy", "reaction_inner_radial", "u_outer_radial", "gk_avg_hoop",
                    "peak_hoop")
    else:
        from omar_pfem.verification_agent import block3d as M
        k, qs = 3, ("strain_energy", "w_tip", "mid_gk_avg_sxx", "root_gk_avg_von_mises",
                    "peak_von_mises")
    a, b = M.solve(k, tol=1e-10), M.solve(k, tol=1e-12)
    ch = max(abs(a[q] - b[q]) / abs(b[q]) for q in qs)
    return {"level_k": k, "max_rel_change": float(ch), "tolerances": [1e-10, 1e-12]}


def compute(which=("l_shape", "edge_crack", "thick_cylinder_3d", "clamped_block_3d")):
    d = _load()
    for g in which:
        if g in ("l_shape", "edge_crack") and f"mms:{g}" not in d:
            d[f"mms:{g}"] = form_mms("lshape" if g == "l_shape" else "crack")
            _save(d)
        if g in ("thick_cylinder_3d", "clamped_block_3d") and f"tol:{g}" not in d:
            d[f"tol:{g}"] = tolerance_check(g)
            _save(d)
    return d


# ---- the hooks the diagnosis reads (stored results only; never solve inside a diagnosis)
def code_check(geometry):
    d = _load()
    if geometry in ("l_shape", "edge_crack"):
        r = d.get(f"mms:{geometry}")
        if r is None:
            return None, "MMS of the module's own form not yet run (followup_hooks)"
        return r["as_expected"], (f"MMS through {r['module']}.stiffness (Q2): rates L2 "
                                  f"{r['rate_L2']:.2f} (theory 3), H1 {r['rate_H1']:.2f} (theory 2)")
    if geometry == "open_hole_lamina":
        rows = json.load(open(os.path.join(HERE, "lamina_mms.json")))["rows"]
        l2 = np.log(rows[-2]["L2_rel"] / rows[-1]["L2_rel"]) / np.log(2)
        h1 = np.log(rows[-2]["H1_semi_rel"] / rows[-1]["H1_semi_rel"]) / np.log(2)
        return abs(l2 - 3) < 0.25 and abs(h1 - 2) < 0.25, (
            f"lamina MMS (Q9 orthotropic): rates L2 {l2:.2f} (theory 3), H1 {h1:.2f} (theory 2)")
    if geometry in ("thick_cylinder_3d", "clamped_block_3d"):
        from omar_pfem.verification_agent.diagnose import _q20_mms_ok
        return _q20_mms_ok()
    return None, "no code check recorded"


def tolerance(geometry):
    if geometry in ("l_shape", "edge_crack", "open_hole_lamina"):
        return 0.0, "linear problem solved in one direct step (no Newton iteration)"
    r = _load().get(f"tol:{geometry}")
    if r is None:
        return None, "no tighter-tolerance re-solve yet (followup_hooks)"
    return r["max_rel_change"], (f"level k = {r['level_k']} re-solved at 1e-12 instead of 1e-10")


def linear_solver(geometry):
    return {"l_shape": "direct sparse LU (pypardiso)", "edge_crack": "direct sparse LU (pypardiso)",
            "open_hole_lamina": "direct sparse LU (pypardiso)",
            "thick_cylinder_3d": "torch-fem direct", "clamped_block_3d": "torch-fem direct"
            }.get(geometry)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    d = compute()
    for k, v in d.items():
        print(k, {a: b for a, b in v.items() if a != "rows"})
