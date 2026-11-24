"""Generic 2D pipeline: from a problem description to verdicts, with no problem code
(2026-11-24).

The main manual step left in the framework was writing a solver and a data adapter for
every new problem (automation assessment). This module removes it for plane-strain linear
elastic panels whose shape is a union of unit squares (L, T, U, step, ... shapes):

  1. read a JSON description: cells, material, boundary conditions on edge segments,
     displacement points of interest;
  2. extract the boundary loop and its corners, run the corner analysis
     (remedy.corner_analysis) and pick the quantities: strain energy, the requested
     displacements, a Gaussian von Mises average and the local energy at every
     rate-limiting corner, the global peak;
  3. solve the uniform Q2 ladder, two graded strategies, and an independent P2
     (6-node triangle) ladder on the same vertices;
  4. run the framework's rule plan (post hoc singular-order extension on, as in
     calibration.py) and the cross-check rule (crosscheck.assess_independent, decide);
  5. remedy loop: if a verifiable quantity is not certified, or certified with U above
     a tenth of its class tolerance, plan the intensity-aware grading
     (remedy_intensity.plan_intensity) from the uniform ladder, solve the graded Q2 and P2
     ladders, and certify again.

Description format (JSON):
  {"name": "...", "cells": [[i, j], ...], "E": 1000, "nu": 0.3,
   "bcs": [{"segment": [[x0, y0], [x1, y1]], "type": "clamped"},
           {"segment": [[x0, y0], [x1, y1]], "type": "traction", "t": [tx, ty]}],
   "points": {"v_tip": [x, y, "y"], ...}}            (points on multiples of 0.5)
Unlisted edges are traction-free.

    python -m omar_pfem.verification_agent.generic2d <description.json>
"""
import json
import math
import os
import sys
import time

import numpy as np
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "generic2d_runs")
LEVELS = (2, 4, 8, 16, 32, 64)
STRATEGY_N, GRADINGS = 8, (1.5, 2.0)
ELL, RHO = 0.1, 0.25


# ---------------------------------------------------------------- geometry ------------
def _on_segment(px, py, seg, tol=1e-9):
    (x0, y0), (x1, y1) = seg
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    s = ((px - x0) * dx + (py - y0) * dy) / L2
    return (np.abs((px - x0) * dy - (py - y0) * dx) / math.sqrt(L2) < tol) & \
        (s > -tol) & (s < 1 + tol)


def boundary_polygon(desc):
    """CCW boundary loop of the union of cells; vertices where the direction or the
    edge condition changes; each vertex carries the condition of the edge leaving it."""
    cells = {tuple(c) for c in desc["cells"]}
    edges = set()
    for i, j in cells:
        loop = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        for a in range(4):
            edges.add((loop[a], loop[(a + 1) % 4]))
    bnd = [e for e in edges if (e[1], e[0]) not in edges]
    nxt = {}
    for a, b in bnd:
        if a in nxt:
            raise ValueError("boundary touches itself at a vertex; not supported")
        nxt[a] = b
    start = min(nxt)
    loop, v = [start], nxt[start]
    while v != start:
        loop.append(v)
        v = nxt[v]
    if len(loop) != len(bnd):
        raise ValueError("domain with holes or several parts; not supported")
    clamped = [b["segment"] for b in desc["bcs"] if b["type"] == "clamped"]

    def bc(a, b):
        mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        return "clamped" if any(_on_segment(mx, my, s) for s in clamped) else "free"
    n = len(loop)
    poly = []
    for k in range(n):
        a, b, c = loop[k - 1], loop[k], loop[(k + 1) % n]
        d_in, d_out = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        if d_in != d_out or bc(a, b) != bc(b, c):
            poly.append((b[0], b[1], bc(b, c)))
    return poly


class Problem:
    def __init__(self, desc):
        from omar_pfem.verification_agent.remedy import corner_analysis
        self.desc = desc
        self.name = desc["name"]
        self.E, self.nu = float(desc["E"]), float(desc["nu"])
        self.mu = self.E / (2 * (1 + self.nu))
        self.lame = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.cells = {tuple(c) for c in desc["cells"]}
        xs = [c[0] for c in self.cells] + [c[0] + 1 for c in self.cells]
        ys = [c[1] for c in self.cells] + [c[1] + 1 for c in self.cells]
        self.box = (min(xs), max(xs), min(ys), max(ys))
        self.polygon = boundary_polygon(desc)
        self.corners = [c for c in corner_analysis(self.polygon, self.nu) if c["rate_limiting"]]
        self.lam_min = min(min(c["rate_limiting"]) for c in self.corners) if self.corners else None
        self.qois = ["strain_energy"] + list(desc.get("points", {}))
        for k, c in enumerate(self.corners):
            self.qois += [f"corner{k}_gk_avg_von_mises"]
        self.qois += ["peak_von_mises"]
        self.local_keys = [f"local_energy_{c['x']}_{c['y']}" for c in self.corners]

    # ---------------------------------------------------------- meshes and solves -----
    def mesh(self, n, plan):
        from omar_pfem.verification_agent.remedy_intensity import coords
        x0, x1, y0, y1 = self.box
        xs = coords(n, plan["x"], x0, x1)
        ys = coords(n, plan["y"], y0, y1)
        Nx, Ny = len(xs), len(ys)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        idx = lambda i, j: i * Ny + j
        t = []
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
                if (math.floor(cx), math.floor(cy)) in self.cells:
                    t.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
        t = np.array(t).T
        used = np.unique(t)
        remap = -np.ones(Nx * Ny, int)
        remap[used] = np.arange(len(used))
        return fem.MeshQuad(np.array([X.ravel()[used], Y.ravel()[used]]), remap[t])

    def solve(self, n, plan=None, element="Q2"):
        import pypardiso
        plan = plan or {"x": {}, "y": {}}
        t0 = time.time()
        mq = self.mesh(n, plan)
        if element == "Q2":
            msh, el = fem.MeshQuad2.from_mesh(mq), fem.ElementQuad2()
        else:
            msh, el = fem.MeshTri2.from_mesh(mq.to_meshtri()), fem.ElementTriP2()
        mu, lame = self.mu, self.lame

        @fem.BilinearForm
        def stiffness(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * mu * ddot(eu, ev) + lame * trace(eu) * trace(ev)

        b = fem.Basis(msh, fem.ElementVector(el), intorder=4)
        K = fem.asm(stiffness, b)
        f = np.zeros(K.shape[0])
        mid = msh.p[:, msh.facets].mean(axis=1)
        bf = msh.boundary_facets()
        D = []
        for bc in self.desc["bcs"]:
            sel = bf[_on_segment(mid[0, bf], mid[1, bf], bc["segment"])]
            if bc["type"] == "clamped":
                D.append(b.get_dofs(facets=sel).all())
            elif bc["type"] == "traction":
                fb = fem.FacetBasis(msh, fem.ElementVector(el), intorder=4, facets=sel)
                tx, ty = bc["t"]
                f += fem.asm(fem.LinearForm(lambda v, w: tx * v.value[0] + ty * v.value[1]), fb)
        load = sum(math.dist(*bc["segment"]) * np.array(bc["t"])
                   for bc in self.desc["bcs"] if bc["type"] == "traction")
        assert np.allclose([f[b.nodal_dofs[0]].sum() + f[b.facet_dofs[0]].sum()
                            + (f[b.interior_dofs[0]].sum() if b.interior_dofs.size else 0),
                            f[b.nodal_dofs[1]].sum() + f[b.facet_dofs[1]].sum()
                            + (f[b.interior_dofs[1]].sum() if b.interior_dofs.size else 0)],
                           load, atol=1e-9), "applied load differs from the description"
        D = np.unique(np.concatenate(D)) if D else np.array([], int)
        u = fem.solve(*fem.condense(K, f, D=D),
                      solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
        row = {"n": n, "h": 1.0 / n, "element": element, "plan": plan,
               "n_elem": int(msh.t.shape[1]), "n_dof": int(u.size),
               "strain_energy": float(0.5 * u @ (K @ u))}
        U = u[b.nodal_dofs].T
        for name, (px, py, comp) in self.desc.get("points", {}).items():
            j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
            assert math.hypot(msh.p[0, j] - px, msh.p[1, j] - py) < 1e-9, f"{name} is not a node"
            row[name] = float(U[j, 0 if comp == "x" else 1])
        gu = b.interpolate(u).grad
        eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
        tr = eps[0, 0] + eps[1, 1]
        sxx, syy = 2 * mu * eps[0, 0] + lame * tr, 2 * mu * eps[1, 1] + lame * tr
        szz, sxy = lame * tr, 2 * mu * eps[0, 1]
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
        dens = 0.5 * (sxx * eps[0, 0] + syy * eps[1, 1] + 2 * sxy * eps[0, 1])
        x, wq = b.global_coordinates().value, b.dx
        for k, c in enumerate(self.corners):
            d = np.hypot(x[0] - c["x"], x[1] - c["y"])
            kk = wq * np.exp(-(d / ELL) ** 2)
            row[f"corner{k}_gk_avg_von_mises"] = float((vm * kk).sum() / kk.sum())
            row[self.local_keys[k]] = float((dens * wq)[d < RHO].sum())
        row["peak_von_mises"] = float(vm.max())
        row["wall_clock_s"] = time.time() - t0
        return row


# ---------------------------------------------------------------- framework -----------
class _Adapter:
    def __init__(self, fam, strategies, lam):
        self.rows, self.hs, self.lam = fam, [r["h"] for r in fam], lam
        self.strategy_rows = strategies

    @property
    def family(self):
        return [{"n_elem": r["n_elem"], "min_detF": None} for r in self.rows]

    def series(self, q):
        return self.hs, [r[q] for r in self.rows], None

    def singular_exponent(self):
        return self.lam

    def strategies(self, q):
        if len(self.strategy_rows) < 2:
            return None
        return [r[q] for r in self.strategy_rows], None


def framework(prob, fam, strategies):
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.calibration import _limit
    from omar_pfem.verification_agent.orchestrate import combine, convergence_params, run_check
    reentrant = any(c["angle_deg"] > 180.5 for c in prob.corners)
    reg = ("reentrant_corner" if reentrant else "clamped_free_corner") if prob.corners else "smooth"
    sd = dict(name=prob.name, geometry="generic2d_" + prob.name, dim=2,
              material="linear_elastic", loading="traction", regularity=reg, element="Q9",
              reference="mesh_refinement_family", qois=list(prob.qois), well_posedness=[],
              open_questions=[])
    ad = _Adapter(fam, strategies, prob.lam_min)
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    try:
        plan = trees.rule_plan(sd)
        cp = convergence_params(plan)
        v = {}
        for c in plan["checks"]:
            for q in c["qois"]:
                v.setdefault(q, {})[c["check"]] = run_check(ad, c["check"], q, c["params"],
                                                             element="Q9", conv_params=cp.get(q))
    finally:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    out = {}
    for q in prob.qois:
        comb = combine(v[q])
        r = {"verdict": comb[0], "evidence": comb[1], "class": trees.classify_qoi(q),
             "finest": fam[-1][q]}
        if comb[0] == "PASS" and comb[1] == "rate":
            r["limit"], r["U"] = _limit(v[q])
        out[q] = r
    return out


def graded_check(fam, q):
    """Graded ladders have no strategies: the five-level check alone, class tolerances."""
    from omar_pfem.verification_agent import gates, trees
    cp = trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))
    g = gates.multilevel_convergence_gate([r["h"] for r in fam], [r[q] for r in fam],
                                          band_tol=cp["band_tol"], gci_tol=cp["gci_tol"],
                                          order_window=(0.5, 5))
    r = {"verdict": g["verdict"], "evidence": g.get("evidence"), "finest": fam[-1][q],
         "fit_order": g.get("fit_order")}
    if g["verdict"] == "PASS" and g.get("evidence") == "rate":
        r["limit"], r["U"] = g["fit_limit"], g["uncertainty_fine"]
    return r


def cross_check(verdicts, p2fam):
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.crosscheck import assess_independent, decide
    for q, r in verdicts.items():
        if "limit" not in r:
            continue
        cp = trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))
        sc = abs(r["finest"])
        ind = assess_independent([x["h"] for x in p2fam], [x[q] for x in p2fam],
                                 {"band_tol": cp["band_tol"], "gci_tol": cp["gci_tol"]}, sc)
        if ind.get("evidence") == "rate" and ind.get("verdict") == "PASS":
            st, gap = decide(r["limit"], r["U"], ind["fit_limit"], ind["uncertainty_fine"] * sc, sc)
        else:
            st, gap = "independent_unconverged", None
        r["crosscheck"], r["gap_rel"] = st, gap
        r["final"] = "CERTIFIED" if st == "cross_checked" else "INCONCLUSIVE"
    return verdicts


# ---------------------------------------------------------------- pipeline ------------
def run(desc, levels=LEVELS, log=print):
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.remedy_intensity import plan_intensity
    prob = Problem(desc)
    outdir = os.path.join(RUNS, prob.name)
    os.makedirs(outdir, exist_ok=True)
    cache_f = os.path.join(outdir, "ladders.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}

    def ladder(key, n_list, plan, element):
        rows = cache.setdefault(key, [])
        done = {r["n"] for r in rows}
        for n in n_list:
            if n not in done:
                rows.append(prob.solve(n, plan, element))
                rows.sort(key=lambda r: r["n"])
                json.dump(cache, open(cache_f, "w"), indent=1)
                log(f"  {key} n={n} dof={rows[-1]['n_dof']}")
        return [r for r in rows if r["n"] in n_list]

    log(f"[{prob.name}] polygon {prob.polygon}")
    log(f"[{prob.name}] rate-limiting corners {[(c['x'], c['y'], round(c['angle_deg']), c['edges'], [round(l, 4) for l in c['rate_limiting']]) for c in prob.corners]}")
    uni = ladder("Q2 uniform", levels, None, "Q2")
    strongest = min(prob.corners, key=lambda c: min(c["rate_limiting"])) if prob.corners else None
    strat = []
    for g in GRADINGS:
        plan = {"x": {strongest["x"]: g}, "y": {strongest["y"]: g}} if strongest else None
        strat += ladder(f"Q2 strategy g={g}", (STRATEGY_N,), plan, "Q2")
    strat = [r for r in uni if r["n"] == STRATEGY_N] + strat
    p2 = ladder("P2 uniform", levels, None, "P2")
    res = {"description": desc, "polygon": prob.polygon, "corners": prob.corners,
           "qois": prob.qois}
    res["uniform"] = cross_check(framework(prob, uni, strat), p2)
    tol = lambda q: trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))["gci_tol"]
    needs = [q for q, r in res["uniform"].items() if r["class"] != "pointwise_peak"
             and ("U" not in r or r["U"] > 0.1 * tol(q))]
    res["remedy_triggered_by"] = needs
    if needs and prob.corners:
        plan = plan_intensity(prob.polygon, prob.nu, uni)
        planned = {a: {int(k) if float(k).is_integer() else k: v for k, v in plan["plan"][a].items()}
                   for a in ("x", "y")}
        log(f"[{prob.name}] remedy plan {plan['plan']} shares "
            f"{[(c['x'], c['y'], round(c['share'], 3)) for c in plan['corners']]}")
        gq = ladder("Q2 graded", levels, planned, "Q2")
        gp = ladder("P2 graded", levels, planned, "P2")
        res["plan"] = plan
        res["graded"] = cross_check({q: graded_check(gq, q) for q in prob.qois}, gp)
    summ = {"verifiable": sum(r["class"] != "pointwise_peak" for r in res["uniform"].values())}
    for fam in ("uniform", "graded"):
        if fam in res:
            V = res[fam]
            summ[fam] = {"certified_rate": sum("limit" in r for q, r in V.items()),
                         "cross_checked": sum(r.get("crosscheck") == "cross_checked" for r in V.values()),
                         "peaks_certified": sum("limit" in r for q, r in V.items()
                                                if "peak" in q)}
    if "graded" in res:
        gain = {}
        for q in prob.qois:
            a, b = res["uniform"][q], res["graded"][q]
            if "U" in a and "U" in b and b["U"] > 0:
                gain[q] = a["U"] / b["U"]
        summ["U_gain_graded_vs_uniform"] = gain
    res["summary"] = summ
    json.dump(res, open(os.path.join(outdir, "result.json"), "w"), indent=1, default=float)
    log(json.dumps(summ, indent=1, default=float))
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run(json.load(open(sys.argv[1])))
