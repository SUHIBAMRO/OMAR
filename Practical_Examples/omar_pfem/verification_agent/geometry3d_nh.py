"""Compressible neo-Hookean material on 3D extruded sections (25 Nov, round Q).

The same material, load stepping and Newton iteration as pipeline2d_nh (2D plane strain), in
three dimensions:

  W = mu/2 (tr C - 3 - 2 ln J) + lam/2 (ln J)^2,  P = mu (F - F^-T) + lam ln J F^-T,
  dP/dF = mu d_ik d_jl + (mu - lam ln J) Fit_il Fit_kj + lam Fit_ij Fit_kl.

Dead tractions only (pressure, a follower load, and prescribed non-zero displacements are
refused). Load steps: 4, doubled on Newton failure (relative residual 1e-10 within 25
iterations) or J <= 0, up to 64. The geometry, meshes, boundary-condition sets, guards and
quantities are those of geometry3d. Corner exponents are the linearized ones (labelled as such,
as in 2D). Code verification: ManufacturedNH3D, a smooth 3D field with a peak Green-Lagrange
strain of about 10%, its body force -Div P and nominal traction P N derived symbolically.
"""
import time

import numpy as np
import skfem as fem
import sympy as sp

from omar_pfem.verification_agent import geometry3d as G3
from omar_pfem.verification_agent import pipeline2d as P2


def _kin3(grad):
    F = grad + np.eye(3)[:, :, None, None]
    c = np.empty_like(F)                                   # cofactor matrix
    c[0, 0] = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
    c[0, 1] = -(F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
    c[0, 2] = F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]
    c[1, 0] = -(F[0, 1] * F[2, 2] - F[0, 2] * F[2, 1])
    c[1, 1] = F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0]
    c[1, 2] = -(F[0, 0] * F[2, 1] - F[0, 1] * F[2, 0])
    c[2, 0] = F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]
    c[2, 1] = -(F[0, 0] * F[1, 2] - F[0, 2] * F[1, 0])
    c[2, 2] = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    J = F[0, 0] * c[0, 0] + F[0, 1] * c[0, 1] + F[0, 2] * c[0, 2]
    return F, J, c / J                                       # F^-T = cof(F) / J


def validate_nh(desc):
    err = []
    if any(b.get("type") == "pressure" for b in desc.get("bcs", [])):
        err.append("neo-Hookean: pressure is a follower load and is not implemented")
    if any(b.get("type") == "clamped" and b.get("u") for b in desc.get("bcs", [])):
        err.append("neo-Hookean: prescribed non-zero displacements are not implemented")
    return err


class Extruded3DNH(G3.Extruded3D):
    material = "neo_hookean"
    # measured peak / basis estimate, plate, n = 8: Q2 2.21 / 0.82 GB = 2.71, P2 1.27 (25 Nov;
    # the first estimate, calibrated on the linear solver, let a run be killed)
    mem_factor = 2.8
    code_files = ["pipeline2d.py", "geometry2d.py", "geometry3d.py", "geometry3d_nh.py",
                  "pipeline2d_nh.py"]

    def __init__(self, desc):
        errs = validate_nh(desc)
        if errs:
            raise ValueError("; ".join(errs))
        super().__init__(desc)

    def manufactured(self):
        return ManufacturedNH3D(self)

    def _solve(self, n, plan, element, mms):
        import pypardiso
        t0 = time.time()
        msh, el = self.build_mesh(n, plan, element)
        mu, lam = self.mu, self.lame
        E3 = fem.ElementVector(el)
        io = 4                    # 27 points per hexahedron: exact for degree 5 per direction
        b = fem.Basis(msh, E3, intorder=io)
        X = b.global_coordinates().value
        bf = msh.boundary_facets()
        side, faces = self._facet_sets(msh, bf)
        mid = msh.p[:, msh.facets[:, bf]].mean(axis=1)
        mid2 = np.zeros((2, msh.facets.shape[1]))
        mid2[:, bf] = mid[:2] / self.a
        used = np.zeros(bf.size, bool)
        D, dl, comps = [], b.doflocs, ("u^1", "u^2", "u^3")
        dir_val, neu = [], []                    # (dofs, value(fr)), (facet basis, t(fr, w))

        def fb_of(sel):
            return fem.FacetBasis(msh, E3, intorder=io, facets=sel)

        for bc in self.desc["bcs"]:
            m = (faces[bc["face"]] if "face" in bc else side & self.g2.facet_mask(mid2, bf, bc)) & ~used
            sel = bf[m]
            used |= m
            if sel.size == 0:
                continue
            t = bc["type"]
            if t in ("clamped", "symmetry"):
                cs = (0, 1, 2) if t == "clamped" else (
                    (2,) if "face" in bc else
                    ((0,) if abs(bc["segment"][0][0] - bc["segment"][1][0]) < 1e-12 else (1,)))
                for c in cs:
                    d = b.get_dofs(facets=sel).all(comps[c])
                    D.append(d)
                    if mms is not None:
                        dir_val.append((d, lambda fr, d=d, c=c: mms.u(dl[0, d], dl[1, d], dl[2, d], fr)[c]))
                if t == "symmetry" and mms is not None:
                    c = cs[0]
                    neu.append((fb_of(sel), lambda fr, w, c=c: tuple(
                        0 * w.x[0] if i == c else mms.traction(*w.x, *w.n, fr)[i] for i in range(3))))
            elif mms is not None:
                neu.append((fb_of(sel), lambda fr, w: mms.traction(*w.x, *w.n, fr)))
            elif t == "traction":
                tv = list(bc["t"]) + [0.0] * (3 - len(bc["t"]))
                neu.append((fb_of(sel), lambda fr, w, tv=tv: tuple(fr * tv[i] + 0 * w.x[0] for i in range(3))))
        if mms is not None and (~used).any():
            neu.append((fb_of(bf[~used]), lambda fr, w: mms.traction(*w.x, *w.n, fr)))
        D = np.unique(np.concatenate(D)) if D else np.array([], int)

        # P and the tangent A = dP/dF are evaluated ONCE per Newton iteration at the quadrature
        # points and passed to the forms (recomputing the 4-index products for every pair of
        # basis functions made a 3D level take hours)
        # assembled per displacement component on the SCALAR basis (a vector basis function is
        # non-zero in one component only): K_ik = int A_ijkl d_l phi_b d_j phi_a, R_i = int P_ij
        # d_j phi_a; then mapped to the vector numbering (split_indices)
        import scipy.sparse as spr
        bs = fem.Basis(msh, el, intorder=io)
        split = b.split_indices()
        perm = np.concatenate(split)                       # block index -> vector dof

        def to_vec(Kb):
            Kb = Kb.tocoo()
            return spr.csr_matrix((Kb.data, (perm[Kb.row], perm[Kb.col])), shape=(b.N, b.N))

        @fem.BilinearForm
        def kblock(u_, v_, w):
            return np.einsum("jl...,l...,j...->...", w["Aik"], u_.grad, v_.grad)

        @fem.LinearForm
        def rblock(v_, w):
            return np.einsum("j...,j...->...", w["Pi"], v_.grad)

        def asm_internal(Pk):
            R = np.zeros(b.N)
            for i in range(3):
                R[split[i]] = fem.asm(rblock, bs, Pi=Pk[i])
            return R

        def asm_tangent(A):
            blocks = [[fem.asm(kblock, bs, Aik=A[i, :, k, :]) for k in range(3)] for i in range(3)]
            return to_vec(spr.bmat(blocks))

        I3 = np.eye(3)

        def PA(grad):
            F, J, Fit = _kin3(grad)
            lnJ = np.log(J)
            Pk = mu * (F - Fit) + lam * lnJ * Fit
            A = (mu * np.einsum("ik,jl->ijkl", I3, I3)[..., None, None]
                 + (mu - lam * lnJ) * np.einsum("il...,kj...->ijkl...", Fit, Fit)
                 + lam * np.einsum("ij...,kl...->ijkl...", Fit, Fit))
            return Pk, A

        def external(fr):
            f = np.zeros(b.N)
            if mms is not None:
                bb = mms.body(X[0], X[1], X[2], fr)
                f += fem.asm(fem.LinearForm(lambda v, w: sum(bb[i] * v.value[i] for i in range(3))), b)
            for fb, tf in neu:
                f += fem.asm(fem.LinearForm(lambda v, w, tf=tf: sum(
                    tf(fr, w)[i] * v.value[i] for i in range(3))), fb)
            return f

        def attempt(nsteps):
            u = np.zeros(b.N)
            hist = []
            for s in range(1, nsteps + 1):
                fr = s / nsteps
                for d, val in dir_val:
                    u[d] = val(fr)
                fext = external(fr)
                r0, last_du, prev = None, np.inf, np.inf
                for it in range(26):
                    Uf = b.interpolate(u)
                    _, J, _ = _kin3(Uf.grad)
                    if not np.all(J > 0):
                        return None, hist + [[s, it, "J<=0"]]
                    Pk, A = PA(Uf.grad)
                    R = asm_internal(Pk) - fext
                    R[D] = 0.0
                    nr = float(np.linalg.norm(R))
                    r0 = nr if r0 is None else r0
                    # converged: as in 2D, or (3D, found by the small-load test 25 Nov) the
                    # residual has stalled at the round-off floor: below 1e-8 r0 and not halved
                    # by the last iteration (the 3D floor lies just above the 2D thresholds)
                    if it > 0 and (nr <= 1e-10 * max(r0, 1e-30) or last_du <= 1e-12 * max(
                            float(np.linalg.norm(u)), 1e-30) or (nr <= 1e-8 * r0 and nr > 0.5 * prev)):
                        break
                    prev = nr
                    if it == 25:
                        return None, hist + [[s, it, "no convergence"]]
                    K = asm_tangent(A)
                    du = fem.solve(*fem.condense(K, -R, D=D),
                                   solver=lambda A, rhs: pypardiso.spsolve(A.tocsr(), rhs))
                    u += du
                    last_du = float(np.linalg.norm(du))
                hist.append([s, it, nr])
            return u, hist

        nsteps, tried = 4, []
        while True:
            u, hist = attempt(nsteps)
            tried.append(nsteps)
            if u is not None or nsteps >= 64:
                break
            nsteps *= 2
        row = {"n": n, "h_nominal": self.a / n, "layers": self.n_layers(n), "element": element,
               "plan": plan, "n_elem": int(msh.t.shape[1]), "n_dof": int(b.N),
               "load_steps": nsteps, "load_steps_tried": tried, "newton": hist,
               "material": "neo_hookean"}
        row["h"] = (self.volume() / row["n_elem"]) ** (1 / 3)
        if u is None:
            row["failed"] = True
            row["wall_clock_s"] = time.time() - t0
            return row
        Uf = b.interpolate(u)
        F, J, Fit = _kin3(Uf.grad)
        row["min_detF"] = float(J.min())
        if mms is not None:
            ue, ge = mms.u(X[0], X[1], X[2], 1.0), mms.grad(X[0], X[1], X[2])
            e0 = sum((Uf.value[i] - ue[i]) ** 2 for i in range(3))
            e1 = sum((Uf.grad[i, j] - ge[i][j]) ** 2 for i in range(3) for j in range(3))
            row["L2"] = float(np.sqrt((e0 * b.dx).sum()))
            row["H1"] = float(np.sqrt((e1 * b.dx).sum()))
            row["wall_clock_s"] = time.time() - t0
            return row
        lnJ = np.log(J)
        W = mu / 2 * (np.einsum("ij...,ij...->...", F, F) - 3 - 2 * lnJ) + lam / 2 * lnJ ** 2
        row["strain_energy"] = float((W * b.dx).sum())
        split = b.split_indices()
        for name, (px, py, pz, comp) in self.desc.get("points", {}).items():
            idx = split["xyz".index(comp)]
            dd = np.linalg.norm(b.doflocs[:, idx].T - [px, py, pz], axis=1)
            j = int(np.argmin(dd))
            if dd[j] > 1e-8 * self.a:
                raise ValueError(f"{name} is not a mesh node")
            row[name] = float(u[idx[j]])
        Pk = mu * (F - Fit) + lam * lnJ * Fit
        sig = np.einsum("ik...,jk...->ij...", Pk, F) / J
        dev = sig - (sig[0, 0] + sig[1, 1] + sig[2, 2]) / 3 * np.eye(3)[:, :, None, None]
        vm = np.sqrt(1.5 * np.einsum("ij...,ij...->...", dev, dev))
        wq = b.dx
        for k, c in enumerate(self.corners):
            d = np.sqrt((X[0] - c["x"] * self.a) ** 2 + (X[1] - c["y"] * self.a) ** 2
                        + (X[2] - 0.5 * self.T) ** 2)
            kk = wq * np.exp(-(d / (P2.ELL_CELLS * self.a)) ** 2)
            row[f"corner{k}_gk_avg_von_mises"] = float((vm * kk).sum() / kk.sum())
            de = np.hypot(X[0] - c["x"] * self.a, X[1] - c["y"] * self.a)
            row[self.local_keys[k]] = float((W * wq)[de < P2.RHO_CELLS * self.a].sum())
        row["peak_von_mises"] = float(vm.max())
        row["wall_clock_s"] = time.time() - t0
        return row


class ManufacturedNH3D:
    """The field of geometry3d.Manufactured3D with amplitude a (u(frac) = frac u), a chosen for
    a peak Green-Lagrange strain of about 10% (largest eigenvalue of E on a 21^3 grid).

    Only u, its gradient and its second derivatives are symbolic. P(F) is evaluated from the
    exact gradient, and the body force is the exact chain rule
        b_i = -dP_ij/dX_j = -A_ijkl(F) d2u_k / dX_l dX_j,   A = dP/dF (the closed form above);
    a fully symbolic Div P of the 3D field took more than 10 minutes to build."""

    def __init__(self, prob, strain=0.10):
        x, y, z = sp.symbols("x y z", real=True)
        x0, x1, y0, y1 = [c * prob.a for c in prob.box]
        kx, ky = 1.3 * sp.pi / (x1 - x0), 1.3 * sp.pi / (y1 - y0)
        kz = 1.3 * sp.pi / max(prob.T, min(x1 - x0, y1 - y0))
        R = sp.Rational
        u = [sp.sin(kx * x + R(3, 10)) * sp.cos(ky * y + R(1, 5)) * sp.cos(kz * z + R(1, 2)),
             sp.cos(kx * x + R(1, 10)) * sp.sin(ky * y + R(2, 5)) * sp.cos(kz * z + R(7, 10)),
             sp.sin(kx * x + R(1, 5)) * sp.sin(ky * y + R(3, 5)) * sp.sin(kz * z + R(1, 10))]
        Xs = (x, y, z)
        lf = lambda e: sp.lambdify(Xs, e, "numpy")
        self._u = [lf(e) for e in u]                                     # unit amplitude
        self._g = [[lf(sp.diff(u[i], Xs[j])) for j in range(3)] for i in range(3)]
        self._h = [[[lf(sp.diff(u[k], Xs[l], Xs[j])) for j in range(3)] for l in range(3)]
                   for k in range(3)]
        self.mu, self.lam = prob.mu, prob.lame
        s = np.linspace(0, 1, 21)
        Xg = np.meshgrid(x0 + s * (x1 - x0), y0 + s * (y1 - y0), s * prob.T, indexing="ij")
        G1 = self._grad(*Xg)

        def peak(av):
            Fn = av * G1 + np.eye(3)[:, :, None, None, None]
            E = 0.5 * (np.einsum("ki...,kj...->ij...", Fn, Fn) - np.eye(3)[:, :, None, None, None])
            return np.abs(np.linalg.eigvalsh(np.moveaxis(E, (0, 1), (-2, -1)))).max()
        lo, hi = 0.0, 0.5 * prob.a
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (lo, m) if peak(m) > strain else (m, hi)
        self.amp = 0.5 * (lo + hi)
        self.field = {k: f"a * ({e})" for k, e in zip(("ux", "uy", "uz"), u)}
        self.field.update(amplitude=self.amp, peak_strain=strain)

    @staticmethod
    def _v(fn, *a):
        return np.asarray(fn(*a), float) * np.ones_like(np.asarray(a[0], float))

    def _grad(self, x, y, z):
        return np.array([[self._v(f, x, y, z) for f in r] for r in self._g])

    def _P(self, x, y, z, aa):
        G = aa * self._grad(x, y, z)
        F, J, Fit = _kin3(G)
        return mu_P(self.mu, self.lam, F, J, Fit), F, J, Fit

    def u(self, x, y, z, frac=1.0):
        return [self.amp * frac * self._v(f, x, y, z) for f in self._u]

    def grad(self, x, y, z):
        return list(self.amp * self._grad(x, y, z))

    def body(self, x, y, z, frac=1.0):
        aa = self.amp * frac
        _, F, J, Fit = self._P(x, y, z, aa)
        H = aa * np.array([[[self._v(f, x, y, z) for f in rl] for rl in rk] for rk in self._h])
        lnJ = np.log(J)
        mu, lam = self.mu, self.lam
        # A_ijkl H_klj with A = mu d_ik d_jl + (mu - lam lnJ) Fit_il Fit_kj + lam Fit_ij Fit_kl
        t1 = mu * np.einsum("ijj...->i...", H)
        t2 = (mu - lam * lnJ) * np.einsum("il...,kj...,klj...->i...", Fit, Fit, H)
        t3 = lam * np.einsum("ij...,kl...,klj...->i...", Fit, Fit, H)
        return list(-(t1 + t2 + t3))

    def traction(self, x, y, z, nx, ny, nz, frac=1.0):
        Pv = self._P(x, y, z, self.amp * frac)[0]
        return tuple(Pv[i, 0] * nx + Pv[i, 1] * ny + Pv[i, 2] * nz for i in range(3))


def mu_P(mu, lam, F, J, Fit):
    return mu * (F - Fit) + lam * np.log(J) * Fit
