"""3D problems in the autonomous loop (25 Nov; item 4 of "automate everything").

A 3D problem is a general 2D section (geometry2d: polygon, inclined edges, circular or
polygonal holes) extruded to a thickness T along z, linear elastic:

  {"name", "dim": 3, "domain": {...as geometry2d...}, "thickness": T, "E", "nu",
   "bcs": [side faces: {"segment" | "hole", "type": clamped | symmetry | traction | pressure,
                         "t": [tx, ty(, tz)]},
           top / bottom: {"face": "z0" | "z1", "type": free | symmetry | clamped | traction,
                          "t": [tx, ty, tz]}],
   "points": {"name": [x, y, z, "x" | "y" | "z"]}}    (z = 0 or T)

Meshes (gmsh): the 2D section is meshed exactly as in geometry2d (the same self-similar size
field and corner grading) and extruded in nz = n / 2 * k layers (k = max(1, round(2 T / L))),
so every level is the same shape refined by 2 in all three directions:
  Q2 = quadratic hexahedra (27 nodes; recombined quadrilaterals extruded)    main family
  P2 = quadratic tetrahedra (10 nodes; triangles extruded and split)       independent family
h is measured: (volume / elements)^(1/3).

Singular edges (labelled as the extruded-edge approximation; vertex singularities where an edge
meets a free face are NOT analysed):
  * every 2D corner becomes an edge along z: in-plane exponents of the PLANE-STRAIN corner
    problem and the anti-plane exponent (pi / w for equal, pi / (2 w) for mixed Dirichlet /
    Neumann conditions on the out-of-plane displacement);
  * where a side face meets the top or bottom face (90 deg) with conditions that differ, the
    same analysis for that edge (e.g. a clamped side and a free top face: in-plane lambda < 1).
Only the 2D-corner edges can be graded (the grading of geometry2d, extruded).

Code verification: a smooth 3D manufactured field (manufactured()), imposed through the same
boundary-condition types as the problem. A solve that would exceed MAX_DOF is refused before
it starts (the machine has 15 GB); the refusal is an outcome the agent records.
"""
import math
import os
import tempfile
import time

import numpy as np
import skfem as fem
import sympy as sp
from skfem.helpers import ddot, sym_grad, trace

from omar_pfem.verification_agent import geometry2d as G2
from omar_pfem.verification_agent import pipeline2d as P2

# local guard (15 GB machine); a cluster job sets VERIF_MAX_DOF for its larger memory
MAX_DOF = int(os.environ.get("VERIF_MAX_DOF", 450_000))
FACES = ("z0", "z1")


def section(desc):
    """The 2D section as a geometry2d description (plane strain; side conditions only)."""
    bcs = []
    for b in desc["bcs"]:
        if "face" in b:
            continue
        d = {k: v for k, v in b.items() if k in ("segment", "hole", "type", "p", "u")}
        if "t" in b:
            d["t"] = list(b["t"][:2])
        bcs.append(d)
    pts = {k: [p[0], p[1], "x"] for k, p in desc.get("points", {}).items()}
    return {"name": desc["name"], "domain": desc["domain"], "E": desc["E"], "nu": desc["nu"],
            "plane": "strain", "bcs": bcs, "points": pts}


def validate(desc):
    err = []
    for k in ("name", "domain", "thickness", "E", "nu", "bcs"):
        if k not in desc:
            err.append(f"missing key {k}")
    if err:
        return err
    T = float(desc["thickness"])
    if T <= 0:
        err.append("thickness must be positive")
    err += [e for e in G2.validate(section(desc)) if "rigid-body" not in e
            and not e.startswith("point ")]
    for b in desc["bcs"]:
        if "face" in b:
            if b["face"] not in FACES:
                err.append(f"face must be z0 or z1, not {b['face']}")
            if b["type"] not in ("free", "symmetry", "clamped", "traction"):
                err.append(f"face condition {b['type']} not supported")
            if b["type"] == "traction" and len(b.get("t", [])) != 3:
                err.append("a face traction needs t = [tx, ty, tz]")
        elif b.get("type") == "traction" and len(b.get("t", [])) not in (2, 3):
            err.append("traction needs t = [tx, ty] or [tx, ty, tz]")
    sec = section(desc)
    for name, p in desc.get("points", {}).items():
        if len(p) != 4 or p[3] not in ("x", "y", "z"):
            err.append(f"point {name} must be [x, y, z, 'x'|'y'|'z']")
            continue
        if not (abs(p[2]) < 1e-12 * T or abs(p[2] - T) < 1e-12 * T):
            err.append(f"point {name}: z must be 0 or the thickness (mesh nodes at every level)")
        if not G2._in_domain(sec, float(p[0]), float(p[1]), 1e-9 * G2._scale(sec)):
            err.append(f"point {name} is outside the section")
    # rigid-body motion: x, y and z must each be held somewhere
    held = set()
    for b in desc["bcs"]:
        if b["type"] == "clamped":
            held |= {"x", "y", "z"}
        elif b["type"] == "symmetry":
            if "face" in b:
                held.add("z")
            elif "segment" in b:
                s = b["segment"]
                held.add("x" if abs(s[0][0] - s[1][0]) < 1e-12 else "y")
    if held != {"x", "y", "z"}:
        err.append(f"rigid-body motion is not suppressed (held: {sorted(held)})")
    return err


def _antiplane(angle, kin, kout):
    """Exponent of the out-of-plane displacement at an edge: w = 0 on a clamped side
    (Dirichlet), traction-free otherwise (free and symmetry sides: the edge direction lies in
    the symmetry plane)."""
    dir_in, dir_out = kin == "clamped", kout == "clamped"
    return math.pi / angle if dir_in == dir_out else math.pi / (2 * angle)


def _limiting(lams):
    return sorted(float(l) for l in lams if l < 2 - 1e-9 and abs(l - round(l)) > 1e-6)


class Extruded3D:
    material = "linear_elastic"
    dim = 3
    plane = "3d"
    code_files = ["pipeline2d.py", "geometry2d.py", "geometry3d.py"]

    def __init__(self, desc):
        errs = validate(desc)
        if errs:
            raise ValueError("; ".join(errs))
        self.desc, self.name = desc, desc["name"]
        self.T = float(desc["thickness"])
        self.E, self.nu = float(desc["E"]), float(desc["nu"])
        self.mu = self.E / (2 * (1 + self.nu))
        self.lame = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        sec = section(desc)
        g = G2.GeneralProblem.__new__(G2.GeneralProblem)      # the section, without the 2D
        g.desc, g.name, g.a = sec, desc["name"], G2._scale(sec)  # rigid-body check
        g.plane, g.nu = "strain", self.nu
        self.g2, self.a = g, g.a
        pts = np.asarray(desc["domain"]["outer"], float) / self.a
        self.box = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        self.kz = max(1, round(2 * self.T / self.a))
        self.face = {f: "free" for f in FACES}
        for b in desc["bcs"]:
            if "face" in b:
                self.face[b["face"]] = "clamped" if b["type"] == "clamped" else \
                    "symmetry" if b["type"] == "symmetry" else "free"
        self.corners = []
        for c in g._corners_from_gmsh():                       # 2D corners -> z edges
            lam = _limiting(c["exponents"] + [r for r, _ in c["complex_exponents"]]
                            + [_antiplane(math.radians(c["angle_deg"]), *c["edges"])])
            if lam:
                self.corners.append(dict(c, rate_limiting=lam, edge="z",
                                         antiplane=_antiplane(math.radians(c["angle_deg"]),
                                                              *c["edges"])))
        self.face_edges = self._face_edges()
        lams = [min(c["rate_limiting"]) for c in self.corners + self.face_edges]
        self.lam_min = min(lams) if lams else None
        self.qois = ["strain_energy"] + list(desc.get("points", {}))
        self.qois += [f"corner{k}_gk_avg_von_mises" for k in range(len(self.corners))]
        self.qois += ["peak_von_mises"]
        self.local_keys = [f"local_energy_{c['x']}_{c['y']}" for c in self.corners]

    def _face_edges(self):
        """Edges where a side face meets z0 / z1 at 90 deg with different conditions."""
        from omar_pfem.verification_agent.corners_ext import exponents
        kinds = {"free"}
        for b in self.g2.desc["bcs"]:
            kinds.add("clamped" if b["type"] == "clamped" else
                      "symmetry" if b["type"] == "symmetry" else "free")
        out = []
        for f in FACES:
            for k in sorted(kinds):
                if k == self.face[f]:
                    continue
                ang = math.pi / 2
                lam = _limiting(exponents(ang, k, self.face[f], self.nu, plane="strain")
                                + [_antiplane(ang, k, self.face[f])])
                if lam:
                    out.append({"face": f, "side": k, "angle_deg": 90.0,
                                "rate_limiting": lam})
        return out

    # ---- mesh --------------------------------------------------------------------------
    def n_layers(self, n):
        return max(1, n // 2) * self.kz

    def build_mesh(self, n, plan, element):
        import gmsh
        L, h = self.a, self.a / n
        g = self.g2
        g.corners = self.corners
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
            g._add_geometry()
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            fields = []
            hc = g._hole_curves()
            if hc:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "CurvesList", hc)
                gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
                fh = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(fh, "F", f"{h} * Min(1, 0.2 + F{fd} / {0.5 * L})")
                fields.append(fh)
            R = 0.25 * L
            for x, y, gg in graded:
                f = gmsh.model.mesh.field.add("MathEval")
                e = 1.0 - 1.0 / gg
                hmin = max(h * (h / L) ** (gg - 1), 1e-6 * L)
                gmsh.model.mesh.field.setString(
                    f, "F", f"Max({hmin}, {h} * Min(1, (Sqrt((x-({x}))^2+(y-({y}))^2)/{R})^{e}))")
                fields.append(f)
            if fields:
                fm = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(fm)
            rec = element == "Q2"
            if rec:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.occ.extrude(gmsh.model.getEntities(2), 0, 0, self.T,
                                   numElements=[self.n_layers(n)], recombine=rec)
            gmsh.model.occ.synchronize()
            # FIX (25 Nov, found by the volume check): mesh the section FIRST exactly as in 2D
            # (all quadrilaterals by subdivision), THEN extrude it. Meshing all in one step left
            # triangles in the coarse sections, whose prisms were dropped on loading (up to 21%
            # of the volume missing). Now the 3D mesh is the 2D mesh of the same level, layered.
            gmsh.model.mesh.generate(2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
            gmsh.model.mesh.generate(3)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
            gmsh.model.mesh.setOrder(2)
            bad = [t for t in gmsh.model.mesh.getElementTypes(3) if t != (12 if rec else 11)]
            if bad:
                raise RuntimeError(f"unexpected 3D element types {bad} (want only "
                                   f"{'27-node hexahedra' if rec else '10-node tetrahedra'})")
            nn = len(gmsh.model.mesh.getNodes()[0])
            max_dof = int(os.environ.get("VERIF_MAX_DOF", MAX_DOF))
            if 3 * nn > max_dof:
                raise MemoryError(f"level n = {n} ({element}) has {3 * nn} DOF > {max_dof}: too "
                                  "large for this machine (prepare a cluster job)")
            fd_, path = tempfile.mkstemp(suffix=".msh")
            os.close(fd_)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(path)
        finally:
            gmsh.finalize()
        try:
            msh = fem.Mesh.load(path)
        finally:
            os.remove(path)
        want = fem.MeshHex2 if element == "Q2" else fem.MeshTet2
        if not isinstance(msh, want):
            raise RuntimeError(f"gmsh did not return {want.__name__} ({type(msh).__name__})")
        el = fem.ElementHex2() if element == "Q2" else fem.ElementTetP2()
        # GUARD: the mesh must fill the domain (no element silently dropped). Curved faces
        # are quadratic, so the volume matches to ~1e-5 at the coarsest level.
        vol = fem.Basis(msh, el, intorder=4).dx.sum()
        if abs(vol / self.volume() - 1) > 1e-3:
            raise RuntimeError(f"mesh volume {vol:.6g} differs from the domain volume "
                               f"{self.volume():.6g}: elements missing")
        return msh, el

    def volume(self):
        return self.g2.area() * self.T

    # ---- boundary ----------------------------------------------------------------------
    def _facet_sets(self, msh, bf):
        """(side mask, {face: mask}) over the boundary facets."""
        z = msh.p[2, msh.facets[:, bf]]
        tol = 1e-9 * max(self.T, self.a)
        faces = {"z0": np.all(np.abs(z) < tol, axis=0), "z1": np.all(np.abs(z - self.T) < tol, axis=0)}
        return ~(faces["z0"] | faces["z1"]), faces

    # ---- manufactured solution ---------------------------------------------------------
    def manufactured(self):
        return Manufactured3D(self)

    # ---- solve -------------------------------------------------------------------------
    def solve(self, n, plan=None, element="Q2", mms=None):
        plan = plan or {"x": {}, "y": {}}
        # AMENDMENT (25 Nov; the same fix as rounds K/L, remedy_preregister._relaxed_invF):
        # scikit-fem's inverse isoparametric map uses an absolute 1e-12 tolerance, below the
        # round-off floor for 27-node hexahedra with coordinates ~ 50 (Cook, n = 16 failed);
        # the tolerance is scaled to the round-off floor. No criterion changed.
        from omar_pfem.verification_agent.remedy_preregister import _relaxed_invF
        try:
            with _relaxed_invF():
                return self._solve(n, plan, element, mms)
        except MemoryError as e:
            return {"n": n, "element": element, "plan": plan, "failed": True,
                    "reason": str(e), "load_steps_tried": None, "newton": [str(e)]}

    def _solve(self, n, plan, element, mms):
        import pypardiso
        t0 = time.time()
        msh, el = self.build_mesh(n, plan, element)
        mu, lame = self.mu, self.lame
        E3 = fem.ElementVector(el)

        @fem.BilinearForm
        def stiffness(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * mu * ddot(eu, ev) + lame * trace(eu) * trace(ev)

        io = 6 if mms else 4
        b = fem.Basis(msh, E3, intorder=io)
        K = fem.asm(stiffness, b)
        f = np.zeros(K.shape[0])
        if mms is not None:
            f += fem.asm(fem.LinearForm(lambda v, w: sum(
                mms.body(*w.x)[i] * v.value[i] for i in range(3))), b)
        bf = msh.boundary_facets()
        side, faces = self._facet_sets(msh, bf)
        mid = msh.p[:, msh.facets[:, bf]].mean(axis=1)           # (3, len(bf))
        mid2 = np.zeros((2, msh.facets.shape[1]))
        mid2[:, bf] = mid[:2] / self.a
        used = np.zeros(bf.size, bool)
        D, xD, dl = [], np.zeros(K.shape[0]), b.doflocs
        comps = ("u^1", "u^2", "u^3")

        def neumann(sel, tfun):
            if sel.size == 0:
                return
            fb = fem.FacetBasis(msh, E3, intorder=io, facets=sel)
            f[:] += fem.asm(fem.LinearForm(lambda v, w: sum(
                tfun(w)[i] * v.value[i] for i in range(3))), fb)

        def exact_traction(w):
            return mms.traction(*w.x, *w.n)

        def dirichlet(sel, cs):
            for c in cs:
                d = b.get_dofs(facets=sel).all(comps[c])
                D.append(d)
                if mms is not None:
                    xD[d] = mms.u(dl[0, d], dl[1, d], dl[2, d])[c]

        def tangential(sel, c):              # symmetry: exact shear tractions (MMS only)
            if mms is not None:
                neumann(sel, lambda w: tuple(0 * w.x[0] if i == c else exact_traction(w)[i]
                                             for i in range(3)))

        for bc in self.desc["bcs"]:
            if "face" in bc:
                m = faces[bc["face"]] & ~used
            else:
                m = side & self.g2.facet_mask(mid2, bf, bc) & ~used
            sel = bf[m]
            used |= m
            t = bc["type"]
            if t == "clamped":
                dirichlet(sel, (0, 1, 2))
                if mms is None and "u" in bc:
                    for c, val in enumerate(bc["u"]):
                        xD[b.get_dofs(facets=sel).all(comps[c])] = val
            elif t == "symmetry":
                if "face" in bc:
                    c = 2
                else:
                    s = bc["segment"]
                    c = 0 if abs(s[0][0] - s[1][0]) < 1e-12 else 1
                dirichlet(sel, (c,))
                tangential(sel, c)
            elif mms is not None:
                neumann(sel, exact_traction)
            elif t == "traction":
                tv = list(bc["t"]) + [0.0] * (3 - len(bc["t"]))
                neumann(sel, lambda w, tv=tv: tuple(tv[i] + 0 * w.x[0] for i in range(3)))
            elif t == "pressure":
                p = float(bc["p"])
                neumann(sel, lambda w, p=p: tuple(-p * w.n[i] for i in range(3)))
        if mms is not None:
            neumann(bf[~used], exact_traction)
        D = np.unique(np.concatenate(D)) if D else np.array([], int)
        u = fem.solve(*fem.condense(K, f, x=xD, D=D),
                      solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
        row = {"n": n, "h_nominal": self.a / n, "layers": self.n_layers(n), "element": element,
               "plan": plan, "n_elem": int(msh.t.shape[1]), "n_dof": int(u.size)}
        row["h"] = (self.volume() / row["n_elem"]) ** (1 / 3)
        if mms is not None:
            uh = b.interpolate(u)
            X = b.global_coordinates().value
            ue, ge = mms.u(*X), mms.grad(*X)
            e0 = sum((uh.value[i] - ue[i]) ** 2 for i in range(3))
            e1 = sum((uh.grad[i, j] - ge[i][j]) ** 2 for i in range(3) for j in range(3))
            row["L2"] = float(np.sqrt((e0 * b.dx).sum()))
            row["H1"] = float(np.sqrt((e1 * b.dx).sum()))
            row["wall_clock_s"] = time.time() - t0
            return row
        row["strain_energy"] = float(0.5 * u @ (K @ u))
        # points are nodes of the quadratic mesh (vertex or mid-side): look them up among the
        # degree-of-freedom locations of the requested component
        split = b.split_indices()
        for name, (px, py, pz, comp) in self.desc.get("points", {}).items():
            idx = split["xyz".index(comp)]
            d = np.linalg.norm(b.doflocs[:, idx].T - [px, py, pz], axis=1)
            j = int(np.argmin(d))
            if d[j] > 1e-8 * self.a:
                raise ValueError(f"{name} is not a mesh node")
            row[name] = float(u[idx[j]])
        gu = b.interpolate(u).grad
        eps = 0.5 * (gu + np.einsum("ij...->ji...", gu))
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        s = 2 * mu * eps + lame * tr * np.eye(3)[:, :, None, None]
        dev = s - (s[0, 0] + s[1, 1] + s[2, 2]) / 3 * np.eye(3)[:, :, None, None]
        vm = np.sqrt(1.5 * np.einsum("ij...,ij...->...", dev, dev))
        dens = 0.5 * np.einsum("ij...,ij...->...", s, eps)
        x, wq = b.global_coordinates().value, b.dx
        for k, c in enumerate(self.corners):         # around the edge mid-point
            d = np.sqrt((x[0] - c["x"] * self.a) ** 2 + (x[1] - c["y"] * self.a) ** 2
                        + (x[2] - 0.5 * self.T) ** 2)
            kk = wq * np.exp(-(d / (P2.ELL_CELLS * self.a)) ** 2)
            row[f"corner{k}_gk_avg_von_mises"] = float((vm * kk).sum() / kk.sum())
            de = np.hypot(x[0] - c["x"] * self.a, x[1] - c["y"] * self.a)
            row[self.local_keys[k]] = float((dens * wq)[de < P2.RHO_CELLS * self.a].sum())
        row["peak_von_mises"] = float(vm.max())
        row["wall_clock_s"] = time.time() - t0
        return row


class Manufactured3D:
    """u_i = A sin / cos products in x, y, z (smooth, not polynomial, non-zero on every face);
    in-plane wavenumbers 1.3 pi / box side, through-thickness 1.3 pi / max(T, smallest side)."""

    def __init__(self, prob):
        x, y, z = sp.symbols("x y z", real=True)
        x0, x1, y0, y1 = [c * prob.a for c in prob.box]
        # AMENDMENT (25 Nov, after the first Cook levels, before any scoring): the through-
        # thickness wavenumber was 0.7 pi / T; on thin plates (Cook: T / L = 1 / 48, elements
        # 24 times wider than thick at every level) that field converged far too slowly for
        # the levels the memory allows (tetrahedra: L2 order 2.1 at n = 8 -> 16). Now the
        # z wavenumber is 1.3 pi / max(T, smallest side of the section box).
        kx, ky = 1.3 * sp.pi / (x1 - x0), 1.3 * sp.pi / (y1 - y0)
        kz = 1.3 * sp.pi / max(prob.T, min(x1 - x0, y1 - y0))
        A = 1e-3 * prob.a
        R = sp.Rational
        u = [A * sp.sin(kx * x + R(3, 10)) * sp.cos(ky * y + R(1, 5)) * sp.cos(kz * z + R(1, 2)),
             A * sp.cos(kx * x + R(1, 10)) * sp.sin(ky * y + R(2, 5)) * sp.cos(kz * z + R(7, 10)),
             A * sp.sin(kx * x + R(1, 5)) * sp.sin(ky * y + R(3, 5)) * sp.sin(kz * z + R(1, 10))]
        X = (x, y, z)
        g = [[sp.diff(u[i], X[j]) for j in range(3)] for i in range(3)]
        eps = [[(g[i][j] + g[j][i]) / 2 for j in range(3)] for i in range(3)]
        tr = eps[0][0] + eps[1][1] + eps[2][2]
        s = [[2 * prob.mu * eps[i][j] + (prob.lame * tr if i == j else 0) for j in range(3)]
             for i in range(3)]
        body = [-sum(sp.diff(s[i][j], X[j]) for j in range(3)) for i in range(3)]
        f = lambda e: sp.lambdify(X, e, "numpy")
        self._u = [f(e) for e in u]
        self._g = [[f(e) for e in row] for row in g]
        self._s = [[f(e) for e in row] for row in s]
        self._b = [f(e) for e in body]
        self.field = {k: str(e) for k, e in zip(("ux", "uy", "uz"), u)}

    @staticmethod
    def _v(fn, *a):
        return np.asarray(fn(*a), float) * np.ones_like(np.asarray(a[0], float))

    def u(self, x, y, z):
        return [self._v(f, x, y, z) for f in self._u]

    def grad(self, x, y, z):
        return [[self._v(f, x, y, z) for f in row] for row in self._g]

    def body(self, x, y, z):
        return [self._v(f, x, y, z) for f in self._b]

    def traction(self, x, y, z, nx, ny, nz):
        s = [[self._v(f, x, y, z) for f in row] for row in self._s]
        return tuple(s[i][0] * nx + s[i][1] * ny + s[i][2] * nz for i in range(3))


def make(desc):
    if desc.get("material") == "neo_hookean":
        from omar_pfem.verification_agent.geometry3d_nh import Extruded3DNH
        return Extruded3DNH(desc)
    return Extruded3D(desc)
