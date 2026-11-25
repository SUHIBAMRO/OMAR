"""General 2D domains for the autonomous pipeline (2026-11-25): any straight-edged polygon
(inclined edges, e.g. Cook's membrane) with circular or polygonal holes, meshed by gmsh.

The domains of pipeline2d were unions of squares; the automatic literature search showed that
most published examples have inclined edges or holes. This module keeps everything else of the
pipeline (solvers for linear elasticity and neo-Hookean material, boundary conditions, the
manufactured-solution check, the rules, the experiments) and replaces only the geometry:

  domain   {"outer": [[x, y], ...] (any orientation),
            "holes": [{"circle": [cx, cy, r]} | {"polygon": [[x, y], ...]}, ...]}
  bcs      as in pipeline2d on straight segments of the outer boundary or of polygonal holes
           ({"segment": [[x0, y0], [x1, y1]], "type": ...}), or on a whole hole
           ({"hole": k, "type": "clamped" | "traction" | "pressure"})
  points   {"name": [x, y, "x" | "y"]}, anywhere in the domain (evaluated by interpolation)

Meshes: gmsh, quadratic triangles (P2) as the main family and quadratic quadrilaterals (Q2, by
recombination) as the independent family, both with curved (quadratic) edges on circles. Level n
has a target element size L / n (L = the largest side of the bounding box). Graded meshes refine
toward chosen corners with h(r) = h (r / R)^(1 - 1/g) for r < R = L / 4, the unstructured
analogue of the tensor grading x = s^g of pipeline2d.

Corners: every vertex of the outer polygon and of polygonal holes, at its true angle, with the
edge conditions on both sides (corners_ext: symmetry edges, complex exponents). Circles have no
corners.
"""
import math
import os
import tempfile

import numpy as np
import skfem as fem

from omar_pfem.verification_agent import pipeline2d as P2


# ------------------------------------------------------------------ geometry --------------
def _area(poly):
    x, y = np.asarray(poly, float).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _seg_hits(p, q, seg, tol):
    """Is the midpoint of edge p-q on segment seg?"""
    m = (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))
    return P2._on_seg(m[0], m[1], seg, tol)


def _split(poly, bcs, tol):
    """Insert the end points of boundary-condition segments that lie inside an edge."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        out.append(tuple(p))
        ins = []
        for b in bcs:
            for e in b.get("segment", []):
                d = (q[0] - p[0], q[1] - p[1])
                L2 = d[0] ** 2 + d[1] ** 2
                s = ((e[0] - p[0]) * d[0] + (e[1] - p[1]) * d[1]) / L2
                cross = abs((e[0] - p[0]) * d[1] - (e[1] - p[1]) * d[0]) / math.sqrt(L2)
                if cross < tol and tol / math.sqrt(L2) < s < 1 - tol / math.sqrt(L2):
                    ins.append((s, (float(e[0]), float(e[1]))))
        for _, e in sorted(set(ins)):
            out.append(e)
    return out


def loops(desc):
    """[(points, is_hole)] with the material on the left of every loop (outer CCW, holes CW)."""
    out = []
    outer = [tuple(map(float, p)) for p in desc["domain"]["outer"]]
    if _area(outer) < 0:
        outer = outer[::-1]
    out.append((outer, False))
    for h in desc["domain"].get("holes", []):
        if "polygon" in h:
            pts = [tuple(map(float, p)) for p in h["polygon"]]
            if _area(pts) > 0:
                pts = pts[::-1]
            out.append((pts, True))
    return out


def validate(desc):
    err = []
    for k in ("name", "domain", "E", "nu", "bcs"):
        if k not in desc:
            err.append(f"missing key {k}")
    if err:
        return err
    outer = desc["domain"].get("outer", [])
    if len(outer) < 3 or abs(_area(outer)) <= 0:
        err.append("the outer boundary needs at least 3 points and a positive area")
    if not (0 <= float(desc["nu"]) < 0.5) or float(desc["E"]) <= 0:
        err.append("E must be positive and nu in [0, 0.5)")
    if desc.get("plane", "strain") not in ("strain", "stress"):
        err.append("plane must be strain or stress")
    holes = desc["domain"].get("holes", [])
    L = _scale(desc)
    for b in desc["bcs"]:
        if b.get("type") not in P2.BC_TYPES:
            err.append(f"unknown bc type {b.get('type')}")
        if "hole" in b:
            if not (0 <= int(b["hole"]) < len(holes)):
                err.append(f"hole {b['hole']} does not exist")
            if b.get("type") == "symmetry":
                err.append("symmetry is only defined on straight segments")
            continue
        seg = b.get("segment")
        if not seg:
            err.append("a boundary condition needs a segment or a hole index")
            continue
        on = any(P2._on_seg(seg[0][0] * 0.5 + seg[1][0] * 0.5, seg[0][1] * 0.5 + seg[1][1] * 0.5,
                            [p, q], 1e-9 * L)
                 for lp, _ in loops(desc) for p, q in zip(lp, lp[1:] + lp[:1]))
        if not on:
            err.append(f"segment {seg} is not on a straight boundary edge")
        if b["type"] == "symmetry" and not (abs(seg[0][0] - seg[1][0]) < 1e-12 * L
                                            or abs(seg[0][1] - seg[1][1]) < 1e-12 * L):
            err.append("symmetry segments must be parallel to an axis")
        if b["type"] == "traction" and len(b.get("t", [])) != 2:
            err.append("traction needs t = [tx, ty]")
        if b["type"] == "pressure" and "p" not in b:
            err.append("pressure needs p")
    if not any(b.get("type") == "clamped" for b in desc["bcs"]):
        dirs = {("x" if abs(b["segment"][0][0] - b["segment"][1][0]) < 1e-12 * L else "y")
                for b in desc["bcs"] if b.get("type") == "symmetry" and "segment" in b}
        if dirs != {"x", "y"}:
            err.append("rigid-body motion is not suppressed")
    for name, p in desc.get("points", {}).items():
        if len(p) != 3 or p[2] not in ("x", "y"):
            err.append(f"point {name} must be [x, y, 'x'|'y']")
        elif not _in_domain(desc, float(p[0]), float(p[1]), 1e-9 * L):
            err.append(f"point {name} {p[:2]} is outside the domain (or inside a hole)")
    return err


def _in_domain(desc, x, y, tol):
    """Inside or on the outer polygon, and not strictly inside any hole."""
    from matplotlib.path import Path
    if not Path(desc["domain"]["outer"]).contains_point((x, y), radius=tol) and \
            not Path(desc["domain"]["outer"]).contains_point((x, y), radius=-tol):
        return False
    for h in desc["domain"].get("holes", []):
        if "circle" in h:
            cx, cy, r = h["circle"]
            if math.hypot(x - cx, y - cy) < r - tol:
                return False
        elif Path(h["polygon"]).contains_point((x, y), radius=-tol) and \
                Path(h["polygon"]).contains_point((x, y), radius=tol):
            return False
    return True


def _scale(desc):
    pts = np.asarray(desc["domain"]["outer"], float)
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))


# ------------------------------------------------------------------ mixin -----------------
class GeneralGeometry:
    """Replaces the cell geometry of pipeline2d.Problem2D (or Problem2DNH) by a gmsh domain."""

    def _init_general(self, desc):
        from omar_pfem.verification_agent.corners_ext import corner_analysis
        errs = validate(desc)
        if errs:
            raise ValueError("; ".join(errs))
        self.desc, self.name = desc, desc["name"]
        self.a = _scale(desc)                              # lengths in units of a below
        self.plane = desc.get("plane", "strain")
        self.E, self.nu = float(desc["E"]), float(desc["nu"])
        self.mu = self.E / (2 * (1 + self.nu))
        lam3 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.lame = lam3 if self.plane == "strain" else 2 * self.mu * lam3 / (lam3 + 2 * self.mu)
        pts = np.asarray(desc["domain"]["outer"], float) / self.a
        self.box = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        self.all_corners = self._corners_from_gmsh()
        self.corners = [c for c in self.all_corners if c["rate_limiting"]]
        self.lam_min = min(min(c["rate_limiting"]) for c in self.corners) if self.corners else None
        self.qois = ["strain_energy"] + list(desc.get("points", {}))
        self.qois += [f"corner{k}_gk_avg_von_mises" for k in range(len(self.corners))]
        self.qois += ["peak_von_mises"]
        self.local_keys = [f"local_energy_{c['x']}_{c['y']}" for c in self.corners]

    def _add_geometry(self):
        """The domain in the current gmsh model (outer polygon minus holes; points of
        interest embedded so that they become mesh nodes)."""
        import gmsh
        occ = gmsh.model.occ
        outer = [tuple(map(float, p)) for p in self.desc["domain"]["outer"]]
        pts = [occ.addPoint(x, y, 0) for x, y in outer]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        cut = []
        for hdesc in self.desc["domain"].get("holes", []):
            if "circle" in hdesc:
                cx, cy, r = hdesc["circle"]
                cut.append((2, occ.addDisk(cx, cy, 0, r, r)))
            else:
                hp = [occ.addPoint(x, y, 0) for x, y in hdesc["polygon"]]
                hl = [occ.addLine(hp[i], hp[(i + 1) % len(hp)]) for i in range(len(hp))]
                cut.append((2, occ.addPlaneSurface([occ.addCurveLoop(hl)])))
        if cut:
            occ.cut([(2, surf)], cut)
        pp = [(0, occ.addPoint(float(x), float(y), 0))
              for x, y, _ in self.desc.get("points", {}).values()]
        if pp:
            occ.fragment(occ.getEntities(2), pp)
        occ.synchronize()

    def _hole_curves(self):
        """Tags of the boundary curves that lie on a hole (current gmsh model)."""
        import gmsh
        out = []
        holes = self.desc["domain"].get("holes", [])
        if not holes:
            return out
        for _, c in gmsh.model.getEntities(1):
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            x, y = gmsh.model.getValue(1, c, [0.5 * (lo[0] + hi[0])])[:2]
            for h in holes:
                if "circle" in h:
                    cx, cy, r = h["circle"]
                    if abs(math.hypot(x - cx, y - cy) - r) < 1e-7 * self.a:
                        out.append(c)
                elif any(P2._on_seg(x, y, [h["polygon"][i], h["polygon"][(i + 1) % len(h["polygon"])]],
                                    1e-7 * self.a) for i in range(len(h["polygon"]))):
                    out.append(c)
        return sorted(set(out))

    def _curve_kind(self, x, y):
        """Boundary condition of the boundary curve through (x, y): clamped, symmetry, free."""
        L = self.a
        kinds = []
        for b in self.desc["bcs"]:
            if "segment" in b and P2._on_seg(x, y, b["segment"], 1e-7 * L):
                kinds.append(b["type"])
            elif "hole" in b:
                h = self.desc["domain"]["holes"][int(b["hole"])]
                if "circle" in h:
                    cx, cy, r = h["circle"]
                    if abs(math.hypot(x - cx, y - cy) - r) < 1e-7 * L:
                        kinds.append(b["type"])
                else:
                    hp = h["polygon"]
                    if any(P2._on_seg(x, y, [hp[i], hp[(i + 1) % len(hp)]], 1e-7 * L)
                           for i in range(len(hp))):
                        kinds.append(b["type"])
        return "clamped" if "clamped" in kinds else "symmetry" if "symmetry" in kinds else "free"

    def _corners_from_gmsh(self):
        """Every point where the real boundary (after the holes are cut) turns or changes its
        condition, with its interior angle from the curve tangents (arcs included)."""
        import gmsh
        from omar_pfem.verification_agent.corners_ext import exponents, complex_exponents
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(self.name + "_corners")
            self._add_geometry()
            bnd = gmsh.model.getBoundary(gmsh.model.getEntities(2), oriented=True, combined=True)
            starts, ends = {}, {}
            for dim, tag in bnd:
                c = abs(tag)
                lo, hi = gmsh.model.getParametrizationBounds(1, c)
                p0, p1 = gmsh.model.getAdjacencies(1, c)[1][:2]
                a0, a1 = (lo[0], hi[0]) if tag > 0 else (hi[0], lo[0])
                q0, q1 = (p0, p1) if tag > 0 else (p1, p0)
                sg = 1.0 if tag > 0 else -1.0
                d0 = sg * np.asarray(gmsh.model.getDerivative(1, c, [a0])[:2])
                d1 = sg * np.asarray(gmsh.model.getDerivative(1, c, [a1])[:2])
                xm = gmsh.model.getValue(1, c, [0.5 * (lo[0] + hi[0])])
                kind = self._curve_kind(xm[0], xm[1])
                starts[q0] = (d0, kind)
                ends[q1] = (d1, kind)
            out = []
            for ptag, (dout, kout) in starts.items():
                if ptag not in ends:
                    continue
                din, kin = ends[ptag]
                turn = math.atan2(din[0] * dout[1] - din[1] * dout[0], float(np.dot(din, dout)))
                ang = math.pi - turn
                if abs(turn) < 1e-9 and kin == kout:
                    continue
                x, y = gmsh.model.getValue(0, ptag, [])[:2]
                lams = exponents(ang, kin, kout, self.nu, plane=self.plane)
                lim = [l for l in lams if l < 2 - 1e-9 and abs(l - round(l)) > 1e-6]
                cpx = complex_exponents(ang, kin, kout, self.nu, self.plane) \
                    if abs(ang - math.pi) > 1e-9 or kin != kout else []
                lim += [r for r, _ in cpx if r < 2 - 1e-9]
                out.append({"x": float(x) / self.a, "y": float(y) / self.a,
                            "angle_deg": math.degrees(ang),
                            "edges": [kin, kout], "exponents": [float(l) for l in lams],
                            "complex_exponents": [list(c) for c in cpx],
                            "rate_limiting": sorted(float(l) for l in lim)})
            return out
        finally:
            gmsh.finalize()

    # ---- hooks --------------------------------------------------------------------------
    def build_mesh(self, n, plan, element):
        import gmsh
        L = self.a
        h = L / n
        graded = []
        plan = plan or {"x": {}, "y": {}}
        for c in self.corners:
            gx, gy = plan["x"].get(c["x"]), plan["y"].get(c["y"])
            if gx or gy:
                graded.append((c["x"] * L, c["y"] * L, max(gx or 0, gy or 0)))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(self.name)
            self._add_geometry()
            # SELF-SIMILAR refinement (25 Nov): the same relative size field at every level,
            # scaled by h = L / n, so that the levels form a systematic family. Near holes the
            # size is 0.2 h, growing linearly to h at a distance L / 2 (no curvature-based
            # sizing, which changed regime between levels on the plate with a hole).
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            fields = []
            hole_curves = self._hole_curves()
            if hole_curves:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "CurvesList", hole_curves)
                gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
                fh = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(fh, "F", f"{h} * Min(1, 0.2 + F{fd} / {0.5 * L})")
                fields.append(fh)
            R = 0.25 * L
            for x, y, g in graded:
                f = gmsh.model.mesh.field.add("MathEval")
                e = 1.0 - 1.0 / g
                hmin = max(h * (h / L) ** (g - 1), 1e-6 * L)
                gmsh.model.mesh.field.setString(
                    f, "F", f"Max({hmin}, {h} * Min(1, (Sqrt((x-({x}))^2+(y-({y}))^2)/{R})^{e}))")
                fields.append(f)
            if fields:
                fm = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(fm)
            if element == "Q2":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
            gmsh.model.mesh.generate(2)
            fd, path = tempfile.mkstemp(suffix=".msh")
            os.close(fd)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(path)
        finally:
            gmsh.finalize()
        try:
            msh = fem.Mesh.load(path)
        finally:
            os.remove(path)
        if element == "Q2":
            if not isinstance(msh, fem.MeshQuad2):
                raise RuntimeError(f"gmsh did not return quadratic quadrilaterals ({type(msh).__name__})")
            return msh, fem.ElementQuad2()
        if not isinstance(msh, fem.MeshTri2):
            raise RuntimeError(f"gmsh did not return quadratic triangles ({type(msh).__name__})")
        return msh, fem.ElementTriP2()

    def solve(self, n, plan=None, element="Q2", mms=None):
        """As pipeline2d, but h is MEASURED on the mesh: sqrt(area / number of elements).
        gmsh meshes are not nested and the coarse levels are set by the curvature of holes,
        so the nominal size L / n does not halve exactly between levels (found by the
        manufactured-solution check on the plate with a hole, 25 Nov)."""
        row = super().solve(n, plan, element, mms)
        if row.get("n_elem"):
            row["h_nominal"] = row["h"]
            row["h"] = math.sqrt(self.area() / row["n_elem"])
        return row

    def area(self):
        if not hasattr(self, "_area"):
            A = abs(_area(self.desc["domain"]["outer"]))
            outer = np.asarray(self.desc["domain"]["outer"], float)
            for h in self.desc["domain"].get("holes", []):
                if "circle" in h:
                    cx, cy, r = h["circle"]
                    # the part of the disc inside the outer polygon, by sampling (holes may
                    # cut the boundary, e.g. quarter models)
                    from matplotlib.path import Path
                    g = np.linspace(-r, r, 801)
                    X, Y = np.meshgrid(cx + g, cy + g)
                    inside = (np.hypot(X - cx, Y - cy) <= r) & Path(outer).contains_points(
                        np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
                    A -= inside.sum() * (g[1] - g[0]) ** 2
                else:
                    A -= abs(_area(h["polygon"]))
            self._area = A
        return self._area

    def facet_mask(self, mid, bf, bc):
        if "hole" in bc:
            h = self.desc["domain"]["holes"][int(bc["hole"])]
            if "circle" in h:
                cx, cy, r = [v / self.a for v in h["circle"]]
                d = np.hypot(mid[0, bf] - cx, mid[1, bf] - cy)
                return np.abs(d - r) < 0.2 * r
            hp = [[x / self.a, y / self.a] for x, y in h["polygon"]]
            return np.array([any(P2._on_seg(mid[0, f_], mid[1, f_], [hp[i], hp[(i + 1) % len(hp)]])
                                 for i in range(len(hp))) for f_ in bf], bool)
        return super().facet_mask(mid, bf, bc)

    def point_values(self, u, b, msh):
        """Points of interest are mesh nodes (see build_mesh); the nodal value is exact."""
        U = u[b.nodal_dofs].T
        out = {}
        for name, (px, py, comp) in self.desc.get("points", {}).items():
            j = int(np.argmin((msh.p[0] - px) ** 2 + (msh.p[1] - py) ** 2))
            if math.hypot(msh.p[0, j] - px, msh.p[1, j] - py) > 1e-8 * self.a:
                raise ValueError(f"{name} is not a mesh node")
            out[name] = float(U[j, 0 if comp == "x" else 1])
        return out


class GeneralProblem(GeneralGeometry, P2.Problem2D):
    material = "linear_elastic"

    def __init__(self, desc):
        self._init_general(desc)


def make(desc):
    """The problem class for a description: 3D extruded ("dim": 3), general 2D geometry
    (key 'domain') or cells."""
    if desc.get("dim") == 3:
        from omar_pfem.verification_agent import geometry3d
        return geometry3d.make(desc)
    if "domain" not in desc:
        if desc.get("material") == "neo_hookean":
            from omar_pfem.verification_agent.pipeline2d_nh import Problem2DNH
            return Problem2DNH(desc)
        return P2.Problem2D(desc)
    if desc.get("material") == "neo_hookean":
        from omar_pfem.verification_agent import pipeline2d_nh as NH

        class GeneralNH(GeneralGeometry, NH.Problem2DNH):
            material = "neo_hookean"

            def __init__(self, d):
                errs = [e for e in NH.validate({**d, "cells": [[0, 0]]}) if "cells" not in e
                        and "segment" not in e and "rigid" not in e and "no clamped" not in e]
                if d.get("plane", "strain") != "strain":
                    errs.append("neo-Hookean: only plane strain is implemented")
                if any(b.get("type") == "pressure" for b in d["bcs"]):
                    errs.append("neo-Hookean: pressure (follower load) not implemented")
                if errs:
                    raise ValueError("; ".join(sorted(set(errs))))
                self._init_general(d)

            def manufactured(self):
                return NH.ManufacturedNH(self)
        return GeneralNH(desc)
    return GeneralProblem(desc)
