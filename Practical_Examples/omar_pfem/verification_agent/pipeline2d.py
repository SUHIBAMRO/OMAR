"""Autonomous 2D pipeline, round O (2026-11-24): description -> code verification -> ladders
-> verdicts (adopted rules) -> cross-check -> remedy.

Extends generic2d.py (which stays frozen for its pre-registered results) with what problems
taken from the literature need, and with one solver that serves BOTH the manufactured-solution
check and the production solves, so the code verification checks the code that is used:

  * a cell size (physical units): cells (i, j) span [i a, (i+1) a] x [j a, (j+1) a];
  * plane strain or plane stress;
  * boundary conditions on segments (physical coordinates): clamped (u = 0 or a given
    constant u), symmetry (normal displacement 0, shear traction 0; axis-parallel segments),
    traction (constant t), pressure (p on the outward normal); unlisted edges are free;
  * an optional manufactured solution (mms): body force, exact Dirichlet data on clamped and
    symmetry segments, exact tractions on every other edge, L2 and H1 errors;
  * corner analysis with symmetry edges and complex exponents (corners_ext.py).

Description (JSON):
  {"name": ..., "cell_size": 1.0, "cells": [[i, j], ...], "E": ..., "nu": ...,
   "plane": "strain" | "stress",
   "bcs": [{"segment": [[x0, y0], [x1, y1]], "type": "clamped", "u": [ux, uy]?},
           {"segment": ..., "type": "symmetry"},
           {"segment": ..., "type": "traction", "t": [tx, ty]},
           {"segment": ..., "type": "pressure", "p": p}],
   "points": {"name": [x, y, "x" | "y"], ...}}           (points must be mesh nodes)
"""
import json
import math
import os
import time

import numpy as np
import skfem as fem
from skfem.helpers import ddot, sym_grad, trace

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "pipeline2d_runs")
LEVELS = (2, 4, 8, 16, 32, 64)
STRATEGY_N, GRADINGS = 8, (1.5, 2.0)
ELL_CELLS, RHO_CELLS = 0.1, 0.25          # Gaussian width and local-energy radius, in cells
BC_TYPES = ("clamped", "symmetry", "traction", "pressure")


# ------------------------------------------------------------------ validation ----------
def validate(desc):
    """Errors that make a description unusable (empty list = usable)."""
    err = []
    for k in ("name", "cells", "E", "nu", "bcs"):
        if k not in desc:
            err.append(f"missing key {k}")
    if err:
        return err
    a = float(desc.get("cell_size", 1.0))
    if not (a > 0):
        err.append("cell_size must be positive")
    if not (0 <= float(desc["nu"]) < 0.5):
        err.append("nu must be in [0, 0.5)")
    if float(desc["E"]) <= 0:
        err.append("E must be positive")
    if desc.get("plane", "strain") not in ("strain", "stress"):
        err.append("plane must be strain or stress")
    try:
        poly = polygon(desc)
    except ValueError as e:
        return err + [str(e)]
    edges = _boundary_edges(desc)
    has_dirichlet = False
    for b in desc["bcs"]:
        if b.get("type") not in BC_TYPES:
            err.append(f"unknown bc type {b.get('type')}")
            continue
        (x0, y0), (x1, y1) = [[c / a for c in p] for p in b["segment"]]
        hit = [e for e in edges if _on_seg(0.5 * (e[0][0] + e[1][0]), 0.5 * (e[0][1] + e[1][1]),
                                           ((x0, y0), (x1, y1)))]
        if not hit:
            err.append(f"segment {b['segment']} lies on no boundary edge")
        if b["type"] == "symmetry" and not (abs(x0 - x1) < 1e-12 or abs(y0 - y1) < 1e-12):
            err.append("symmetry segments must be parallel to an axis")
        if b["type"] == "traction" and len(b.get("t", [])) != 2:
            err.append("traction needs t = [tx, ty]")
        if b["type"] == "pressure" and "p" not in b:
            err.append("pressure needs p")
        has_dirichlet |= b["type"] in ("clamped", "symmetry")
    if not any(b.get("type") == "clamped" for b in desc["bcs"]):
        sym_dirs = {("x" if abs(b["segment"][0][0] - b["segment"][1][0]) < 1e-12 else "y")
                    for b in desc["bcs"] if b.get("type") == "symmetry"}
        if sym_dirs != {"x", "y"}:
            err.append("no clamped segment and no symmetry lines in both directions: "
                       "rigid-body motion is not suppressed")
    for name, p in desc.get("points", {}).items():
        if len(p) != 3 or p[2] not in ("x", "y"):
            err.append(f"point {name} must be [x, y, 'x'|'y']")
    del poly
    return err


def _on_seg(px, py, seg, tol=1e-9):
    from omar_pfem.verification_agent.generic2d import _on_segment
    return bool(_on_segment(np.float64(px), np.float64(py), seg, tol))


def _boundary_edges(desc):
    cells = {tuple(c) for c in desc["cells"]}
    edges = set()
    for i, j in cells:
        loop = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        for k in range(4):
            edges.add((loop[k], loop[(k + 1) % 4]))
    return [e for e in edges if (e[1], e[0]) not in edges]


def polygon(desc):
    """CCW boundary loop in CELL units, each vertex carrying the condition of the edge that
    leaves it: clamped, symmetry, or free (traction, pressure and unlisted edges)."""
    from omar_pfem.verification_agent.generic2d import boundary_polygon
    a = float(desc.get("cell_size", 1.0))
    seg = [(b["type"], [[c / a for c in p] for p in b["segment"]]) for b in desc["bcs"]]
    base = boundary_polygon({"cells": desc["cells"], "bcs": []})   # loop and corners only
    # rebuild the loop on unit edges to find condition changes
    loop_edges = _boundary_edges(desc)
    nxt = {e[0]: e[1] for e in loop_edges}
    start = min(nxt)
    loop, v = [start], nxt[start]
    while v != start:
        loop.append(v)
        v = nxt[v]
    del base

    def bc(p, q):
        mx, my = 0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])
        kinds = [t for t, s in seg if _on_seg(mx, my, s)]
        if "clamped" in kinds:
            return "clamped"
        if "symmetry" in kinds:
            return "symmetry"
        return "free"
    n = len(loop)
    poly = []
    for k in range(n):
        p, q, r = loop[k - 1], loop[k], loop[(k + 1) % n]
        if (q[0] - p[0], q[1] - p[1]) != (r[0] - q[0], r[1] - q[1]) or bc(p, q) != bc(q, r):
            poly.append((q[0], q[1], bc(q, r)))
    return poly


# ------------------------------------------------------------------ problem -------------
class Problem2D:
    def __init__(self, desc):
        from omar_pfem.verification_agent.corners_ext import corner_analysis
        errs = validate(desc)
        if errs:
            raise ValueError("; ".join(errs))
        self.desc, self.name = desc, desc["name"]
        self.a = float(desc.get("cell_size", 1.0))
        self.plane = desc.get("plane", "strain")
        self.E, self.nu = float(desc["E"]), float(desc["nu"])
        self.mu = self.E / (2 * (1 + self.nu))
        lam3 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.lame = lam3 if self.plane == "strain" else 2 * self.mu * lam3 / (lam3 + 2 * self.mu)
        self.cells = {tuple(c) for c in desc["cells"]}
        xs = [c[0] for c in self.cells] + [c[0] + 1 for c in self.cells]
        ys = [c[1] for c in self.cells] + [c[1] + 1 for c in self.cells]
        self.box = (min(xs), max(xs), min(ys), max(ys))
        self.polygon = polygon(desc)
        self.all_corners = corner_analysis(self.polygon, self.nu, plane=self.plane)
        self.corners = [c for c in self.all_corners if c["rate_limiting"]]
        self.lam_min = min(min(c["rate_limiting"]) for c in self.corners) if self.corners else None
        self.qois = ["strain_energy"] + list(desc.get("points", {}))
        self.qois += [f"corner{k}_gk_avg_von_mises" for k in range(len(self.corners))]
        self.qois += ["peak_von_mises"]
        self.local_keys = [f"local_energy_{c['x']}_{c['y']}" for c in self.corners]

    def mesh(self, n, plan=None):
        from omar_pfem.verification_agent.remedy_intensity import coords
        plan = plan or {"x": {}, "y": {}}
        x0, x1, y0, y1 = self.box
        xs, ys = coords(n, plan["x"], x0, x1), coords(n, plan["y"], y0, y1)
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
        p = np.array([X.ravel()[used], Y.ravel()[used]]) * self.a
        return fem.MeshQuad(p, remap[t])

    # ---- hooks (2026-11-25), overridden by geometry2d for general domains ----
    def build_mesh(self, n, plan, element):
        mq = self.mesh(n, plan)
        if element == "Q2":
            return fem.MeshQuad2.from_mesh(mq), fem.ElementQuad2()
        return fem.MeshTri2.from_mesh(mq.to_meshtri()), fem.ElementTriP2()

    def facet_mask(self, mid, bf, bc):
        """mid: facet midpoints in cell units; bc: a boundary condition with a segment."""
        seg = [[c / self.a for c in p] for p in bc["segment"]]
        return np.array([_on_seg(mid[0, f_], mid[1, f_], seg) for f_ in bf], bool)

    def point_values(self, u, b, msh):
        U = u[b.nodal_dofs].T
        out = {}
        for name, (px, py, comp) in self.desc.get("points", {}).items():
            j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
            if math.hypot(msh.p[0, j] - px, msh.p[1, j] - py) > 1e-9 * self.a:
                raise ValueError(f"{name} is not a mesh node")
            out[name] = float(U[j, 0 if comp == "x" else 1])
        return out

    def solve(self, n, plan=None, element="Q2", mms=None):
        plan = plan or {"x": {}, "y": {}}
        if plan["x"] or plan["y"]:
            from omar_pfem.verification_agent.remedy_preregister import _relaxed_invF
            with _relaxed_invF():
                return self._solve(n, plan, element, mms)
        return self._solve(n, plan, element, mms)

    def _stress(self, eps):
        tr = eps[0, 0] + eps[1, 1]
        sxx = 2 * self.mu * eps[0, 0] + self.lame * tr
        syy = 2 * self.mu * eps[1, 1] + self.lame * tr
        sxy = 2 * self.mu * eps[0, 1]
        szz = (self.lame * tr if self.plane == "strain" else 0 * tr)
        return sxx, syy, sxy, szz

    def _solve(self, n, plan, element, mms):
        import pypardiso
        t0 = time.time()
        msh, el = self.build_mesh(n, plan, element)
        mu, lame = self.mu, self.lame
        E2 = fem.ElementVector(el)

        @fem.BilinearForm
        def stiffness(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * mu * ddot(eu, ev) + lame * trace(eu) * trace(ev)

        b = fem.Basis(msh, E2, intorder=6 if mms else 4)
        K = fem.asm(stiffness, b)
        f = np.zeros(K.shape[0])
        if mms is not None:
            f += fem.asm(fem.LinearForm(lambda v, w: mms.body(w.x[0], w.x[1])[0] * v.value[0]
                                        + mms.body(w.x[0], w.x[1])[1] * v.value[1]), b)
        mid = msh.p[:, msh.facets].mean(axis=1) / self.a
        bf = msh.boundary_facets()
        used = np.zeros(bf.size, bool)
        D, xD = [], np.zeros(K.shape[0])
        dl = b.doflocs

        def neumann(sel, tfun):
            if sel.size == 0:
                return
            fb = fem.FacetBasis(msh, E2, intorder=6 if mms else 4, facets=sel)
            f[:] += fem.asm(fem.LinearForm(lambda v, w: tfun(w)[0] * v.value[0]
                                           + tfun(w)[1] * v.value[1]), fb)

        for bc in self.desc["bcs"]:
            seg = [[c / self.a for c in p] for p in bc.get("segment", [[0, 0], [1, 0]])]
            m = self.facet_mask(mid, bf, bc)
            sel = bf[m & ~used]
            used |= m
            if bc["type"] == "clamped":
                d = b.get_dofs(facets=sel).all()
                D.append(d)
                if mms is not None:
                    ux, uy = mms.u(dl[0, d], dl[1, d])
                    xD[d] = np.where(np.isin(d, b.get_dofs(facets=sel).all("u^1")), ux, uy)
                elif "u" in bc:
                    xD[b.get_dofs(facets=sel).all("u^1")] = bc["u"][0]
                    xD[b.get_dofs(facets=sel).all("u^2")] = bc["u"][1]
            elif bc["type"] == "symmetry":
                vertical = abs(seg[0][0] - seg[1][0]) < 1e-12
                comp, other = ("u^1", 1) if vertical else ("u^2", 0)
                d = b.get_dofs(facets=sel).all(comp)
                D.append(d)
                if mms is not None:
                    xD[d] = mms.u(dl[0, d], dl[1, d])[0 if vertical else 1]

                    def shear(w, other=other):
                        t = mms.traction(w.x[0], w.x[1], w.n[0], w.n[1])
                        z = 0 * t[0]
                        return (t[0], z) if other == 0 else (z, t[1])
                    neumann(sel, shear)
            elif mms is not None:
                neumann(sel, lambda w: mms.traction(w.x[0], w.x[1], w.n[0], w.n[1]))
            elif bc["type"] == "traction":
                tx, ty = bc["t"]
                neumann(sel, lambda w, tx=tx, ty=ty: (tx + 0 * w.x[0], ty + 0 * w.x[0]))
            elif bc["type"] == "pressure":
                p = float(bc["p"])
                neumann(sel, lambda w, p=p: (-p * w.n[0], -p * w.n[1]))
        if mms is not None:                                  # free edges: exact traction
            neumann(bf[~used], lambda w: mms.traction(w.x[0], w.x[1], w.n[0], w.n[1]))
        D = np.unique(np.concatenate(D)) if D else np.array([], int)
        u = fem.solve(*fem.condense(K, f, x=xD, D=D),
                      solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
        row = {"n": n, "h": self.a / n, "element": element, "plan": plan,
               "n_elem": int(msh.t.shape[1]), "n_dof": int(u.size)}
        if mms is not None:
            uh = b.interpolate(u)
            X = b.global_coordinates().value
            ue = mms.u(X[0], X[1])
            ge = mms.grad(X[0], X[1])
            e0 = sum((uh.value[i] - ue[i]) ** 2 for i in range(2))
            e1 = sum((uh.grad[i, j] - ge[i][j]) ** 2 for i in range(2) for j in range(2))
            row["L2"] = float(np.sqrt((e0 * b.dx).sum()))
            row["H1"] = float(np.sqrt((e1 * b.dx).sum()))
            row["wall_clock_s"] = time.time() - t0
            return row
        row["strain_energy"] = float(0.5 * u @ (K @ u))
        row.update(self.point_values(u, b, msh))
        gu = b.interpolate(u).grad
        eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
        sxx, syy, sxy, szz = self._stress(eps)
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
        dens = 0.5 * (sxx * eps[0, 0] + syy * eps[1, 1] + 2 * sxy * eps[0, 1])
        x, wq = b.global_coordinates().value, b.dx
        for k, c in enumerate(self.corners):
            d = np.hypot(x[0] - c["x"] * self.a, x[1] - c["y"] * self.a)
            kk = wq * np.exp(-(d / (ELL_CELLS * self.a)) ** 2)
            row[f"corner{k}_gk_avg_von_mises"] = float((vm * kk).sum() / kk.sum())
            row[self.local_keys[k]] = float((dens * wq)[d < RHO_CELLS * self.a].sum())
        row["peak_von_mises"] = float(vm.max())
        row["wall_clock_s"] = time.time() - t0
        return row


# ------------------------------------------------------------------ verdicts ------------
class Adapter:
    def __init__(self, fam, strategies, lam):
        self.rows, self.hs, self.lam = fam, [r["h"] for r in fam], lam
        self.strategy_rows = strategies

    @property
    def family(self):
        return [{"n_elem": r["n_elem"], "min_detF": r.get("min_detF")} for r in self.rows]

    def series(self, q):
        return self.hs, [r[q] for r in self.rows], None

    def singular_exponent(self):
        return self.lam

    def strategies(self, q):
        if len(self.strategy_rows) < 2:
            return None
        return [r[q] for r in self.strategy_rows], None


def framework(prob, fam, strategies, full=False):
    """The rule plan under the ADOPTED rule set (adopted.py). full=True also returns the raw
    check results and the adapter (for the diagnosis)."""
    from omar_pfem.verification_agent import adopted, trees
    from omar_pfem.verification_agent.calibration import _limit
    from omar_pfem.verification_agent import orchestrate
    reentrant = any(c["angle_deg"] > 180.5 for c in prob.corners)
    reg = ("reentrant_corner" if reentrant else "clamped_free_corner") if prob.corners else "smooth"
    sd = dict(name=prob.name, geometry="pipeline2d_" + prob.name, dim=2,
              material=getattr(prob, "material", "linear_elastic"), loading="traction", regularity=reg, element="Q9",
              reference="mesh_refinement_family", qois=list(prob.qois), well_posedness=[],
              open_questions=[])
    ad = Adapter(fam, strategies, prob.lam_min)
    with adopted.ruleset():
        plan = trees.rule_plan(sd)
        cp = orchestrate.convergence_params(plan)
        v = {}
        for c in plan["checks"]:
            for q in c["qois"]:
                v.setdefault(q, {})[c["check"]] = orchestrate.run_check(
                    ad, c["check"], q, c["params"], element="Q9", conv_params=cp.get(q))
    out = {}
    for q in prob.qois:
        comb = orchestrate.combine(v[q])
        r = {"verdict": comb[0], "evidence": comb[1], "class": trees.classify_qoi(q),
             "finest": fam[-1][q], "checks": {k: {kk: g.get(kk) for kk in
                                                  ("verdict", "reason", "label", "evidence")}
                                              for k, g in v[q].items()}}
        if comb[0] == "PASS" and comb[1] == "rate":
            r["limit"], r["U"] = _limit(v[q])
        out[q] = r
    if full:
        return out, v, ad
    return out


def cross_check(verdicts, p2fam):
    from omar_pfem.verification_agent.generic2d import cross_check as cc
    return cc(verdicts, p2fam)


def plan_intensity(prob, uniform_rows, degree=2, share_min=None):
    """remedy_intensity.plan_intensity with this module's corner analysis."""
    from omar_pfem.verification_agent.remedy import plan_grading
    from omar_pfem.verification_agent.remedy_intensity import SHARE_MIN
    share_min = SHARE_MIN if share_min is None else share_min
    rows = sorted(uniform_rows, key=lambda r: r["n"])
    fine, coarse = rows[-1], rows[-2]
    out = []
    for c in prob.corners:
        key = f"local_energy_{c['x']}_{c['y']}"
        out.append({"x": c["x"], "y": c["y"], "lam": min(c["rate_limiting"]),
                    "delta": abs(fine[key] - coarse[key])})
    tot = sum(c["delta"] for c in out) or 1.0
    gx, gy = {}, {}
    for c in out:
        c["share"] = c["delta"] / tot
        c["graded"] = c["share"] >= share_min
        if c["graded"]:
            c["g"] = plan_grading(c["lam"], degree)["g"]
            gx[c["x"]] = max(gx.get(c["x"], 0), c["g"])
            gy[c["y"]] = max(gy.get(c["y"], 0), c["g"])
    return {"corners": out, "plan": {"x": gx, "y": gy}, "share_min": share_min,
            "levels_used": [coarse["n"], fine["n"]]}


def fingerprint(prob):
    """sha256 of the problem description and every code file the solve depends on."""
    import hashlib
    h = hashlib.sha256(json.dumps(prob.desc, sort_keys=True).encode())
    here = os.path.dirname(os.path.abspath(__file__))
    files = ["pipeline2d.py"]
    if "domain" in prob.desc:
        files.append("geometry2d.py")
    if getattr(prob, "material", "linear_elastic") == "neo_hookean":
        files.append("pipeline2d_nh.py")
    for f in files:
        h.update(open(os.path.join(here, f), "rb").read())
    return h.hexdigest()


class Ladders:
    """Cached solves of one problem (pipeline2d_runs/<name>/ladders.json)."""

    def __init__(self, prob, log=print):
        self.prob, self.log = prob, log
        self.dir = os.path.join(RUNS, prob.name)
        os.makedirs(self.dir, exist_ok=True)
        self.f = os.path.join(self.dir, "ladders.json")
        self.cache = json.load(open(self.f)) if os.path.exists(self.f) else {}
        # FIX (25 Nov): the cache is valid only for the same description AND the same solver and
        # mesh code. Before, it was keyed by the problem name alone, so a changed mesh generator
        # silently reused old solutions. A mismatched or unmarked cache is moved aside (kept).
        fp = fingerprint(prob)
        if self.cache and self.cache.get("_fingerprint") != fp:
            old = self.cache.get("_fingerprint", "unmarked")[:12]
            os.replace(self.f, os.path.join(self.dir, f"ladders_stale_{old}.json"))
            self.log(f"  ladder cache for {prob.name} was made with other code or another "
                     f"description ({old}); moved aside, recomputing")
            self.cache = {}
        self.cache["_fingerprint"] = fp

    def get(self, key, n_list, plan, element):
        rows = self.cache.setdefault(key, [])
        done = {r["n"] for r in rows}
        for n in n_list:
            if n not in done:
                r = self.prob.solve(n, plan, element)
                if r.get("failed"):
                    raise RuntimeError(f"{key} n={n}: the nonlinear solve failed with "
                                       f"{r['load_steps_tried']} load steps: {r['newton'][-1:]}")
                rows.append(r)
                rows.sort(key=lambda r: r["n"])
                json.dump(self.cache, open(self.f, "w"), indent=1)
                self.log(f"  {key} n={n} dof={rows[-1]['n_dof']}")
        return [r for r in rows if r["n"] in n_list]


def verify(prob, ladders, levels=LEVELS, log=print):
    """Uniform Q2 ladder + strategies + independent P2 ladder -> verdicts and cross-check."""
    uni = ladders.get("Q2 uniform", levels, None, "Q2")
    strongest = min(prob.corners, key=lambda c: min(c["rate_limiting"])) if prob.corners else None
    strat = []
    for g in GRADINGS:
        plan = {"x": {strongest["x"]: g}, "y": {strongest["y"]: g}} if strongest else None
        if plan:
            strat += ladders.get(f"Q2 strategy g={g}", (STRATEGY_N,), plan, "Q2")
    strat = [r for r in uni if r["n"] == STRATEGY_N] + strat
    p2 = ladders.get("P2 uniform", levels, None, "P2")
    return cross_check(framework(prob, uni, strat), p2), uni
