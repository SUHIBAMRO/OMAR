"""Code verification (method of manufactured solutions) of the solvers built in
the last rounds, on the REAL curved bushing geometry with the groove:

  iga_b3.py   trivariate C1 quadratic splines, displacement and projected
              dilatation (fbar_p1)
  mixed_b3.py Q20 with projected dilatation (fbar_p1); its displacement path
              already reproduces torchfem (mms_q20.py) to 4e-14

Manufactured displacement, defined through the parameters of the exact map
(theta = atan2(y, x), t = (r - R_in(z)) / (R_out - R_in(z)), z), vanishing on
every boundary so that all boundaries are homogeneous Dirichlet:
    u_x = a sin(pi t) sin(theta)   sin(pi z)
    u_y = a sin(pi t) sin(2 theta) sin(pi z)
    u_z = a sin(2 pi t) sin(theta) sin(pi z),     a = 0.005 (max |grad u| ~ 0.1)
Body force b = -Div P(F(u*)) of the SAME compressible neo-Hookean energy
(E = 1000, nu = 0.45), by automatic differentiation at every quadrature point.
The groove profile is C1 (curvature jumps at z = 0.35, 0.65), so u* is C1
there; the spline knots and the Q20 mesh planes do not all sit on those planes,
which can lower the observed rates of the Q20 mesh slightly.

Errors against u* at the quadrature points (relative L2 and H1-seminorm).
Expected for degree-2 discretizations: L2 ~ h^3, H1 ~ h^2.

    python -m omar_pfem.verification_agent.mms_new_solvers
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch.func import grad, jacrev, vmap

from omar_pfem.verification_agent import iga_b3 as I

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mms_new_solvers.json")
A_MMS = 0.005


def _R_in(z):
    band = torch.abs(z - I.ZC) < I.W
    return torch.where(band, I.R_IN0 - I.D / 2 * (1 + torch.cos(np.pi * (z - I.ZC) / I.W)),
                       torch.full_like(z, I.R_IN0))


def u_star(X):
    x, y, z = X[0], X[1], X[2]
    th = torch.atan2(y, x)
    r = torch.sqrt(x * x + y * y)
    ri = _R_in(z)
    t = (r - ri) / (I.R_OUT - ri)
    s = torch.sin(np.pi * t) * torch.sin(np.pi * z)
    return A_MMS * torch.stack([s * torch.sin(th), s * torch.sin(2 * th),
                                torch.sin(2 * np.pi * t) * torch.sin(th) * torch.sin(np.pi * z)])


def _P(X):
    F = torch.eye(3) + jacrev(u_star)(X)
    return grad(lambda G: I.psi(G))(F)


def body_force(X, chunk=20000):
    """b = -Div_X P(F(u*)) at reference points X (n, 3)."""
    out = []
    dP = vmap(jacrev(_P))
    for s in range(0, len(X), chunk):
        D = dP(torch.tensor(X[s:s + chunk]))                       # (n, 3, 3, 3)
        out.append(-torch.einsum("nijj->ni", D).numpy())
    return np.concatenate(out)


def exact(X, chunk=50000):
    u, g = [], []
    J = vmap(jacrev(u_star))
    for s in range(0, len(X), chunk):
        Xt = torch.tensor(X[s:s + chunk])
        u.append(vmap(u_star)(Xt).numpy())
        g.append(J(Xt).numpy())
    return np.concatenate(u), np.concatenate(g)


def errors(uh, gh, ue, ge, w):
    l2 = np.sqrt((w * ((uh - ue) ** 2).sum(1)).sum() / (w * (ue ** 2).sum(1)).sum())
    h1 = np.sqrt((w * ((gh - ge) ** 2).sum((1, 2))).sum() / (w * (ge ** 2).sum((1, 2))).sum())
    return float(l2), float(h1)


# ---------------------------------------------------------------------------
# spline solver
# ---------------------------------------------------------------------------
class IgaMMS(I.Problem):
    def __init__(self, level, formulation):
        super().__init__(level, formulation=formulation, keep_N=True)
        X = self.x.reshape(-1, 3)
        b = body_force(X).reshape(self.ne, -1, 3)
        fe = np.einsum("eq,eql,eqi->eli", self.w.numpy(), self.N.numpy(), b)   # (e, l, 3)
        self.f_ext = np.zeros(self.ndof)
        np.add.at(self.f_ext, self.dofs.ravel(), fe.reshape(-1))

    def _bc(self):
        nth, nt, nz = self.nb
        I3 = np.arange(self.nctrl)
        i, rem = np.divmod(I3, nt * nz)
        j, k = np.divmod(rem, nz)
        bnd = (i == 0) | (i == nth - 1) | (j == 0) | (j == nt - 1) | (k == 0) | (k == nz - 1)
        fixed = np.repeat(bnd, 3)
        self.fixed, self.free = np.where(fixed)[0], np.where(~fixed)[0]
        self.ubar = np.zeros(self.ndof)
        self.inner = np.array([], int)

    def residual(self, u, chunk=4000):
        return super().residual(u, chunk) - self.f_ext

    def errors(self):
        ue_, ge_ = exact(self.x.reshape(-1, 3))
        uel = self._ue(self.u).numpy()                                # (e, l, 3)
        uh = np.einsum("eql,eli->eqi", self.N.numpy(), uel).reshape(-1, 3)
        gh = np.einsum("eli,eqlj->eqij", uel, self.dNdX.numpy()).reshape(-1, 3, 3)
        return errors(uh, gh, ue_, ge_, self.w.numpy().ravel())


def u_star_cart(X):
    """Smooth Cartesian manufactured field for the Q20 mesh (Dirichlet data = its
    nodal values on every boundary node), independent of the C1 groove profile."""
    x, y, z = X[0], X[1], X[2]
    return A_MMS * torch.stack([torch.sin(np.pi * x) * torch.cos(np.pi * y / 2) * torch.sin(np.pi * z),
                                torch.sin(np.pi * x / 2) * torch.sin(np.pi * y) * torch.cos(np.pi * z / 2),
                                torch.cos(np.pi * x / 2) * torch.sin(np.pi * y / 2) * torch.sin(np.pi * z)])


# ---------------------------------------------------------------------------
# Q20 with projected dilatation (mixed_b3)
# ---------------------------------------------------------------------------
def q20_p1(level, field="groove", formulation="fbar_p1"):
    from omar_pfem.verification_agent import mixed_b3 as M
    from torchfem.elements import Hexa2
    global u_star
    keep = u_star
    if field == "cartesian":
        u_star = u_star_cart
    pb = M.Problem(*M.LEVELS[level], quadrature="full", formulation=formulation)
    ip, _ = M.rule("full")
    N = Hexa2().N(ip).numpy()                                         # (nq, 20)
    pos = pb.pos.reshape(-1, 3)
    w = pb.w.numpy()
    b = body_force(pos).reshape(len(pb.elements), -1, 3)
    fe = np.einsum("eq,qa,eqi->eai", w, N, b)
    f_ext = np.zeros(pb.ndof)
    np.add.at(f_ext, pb.dofs.ravel(), fe.reshape(-1))
    # every boundary node fixed to u* = 0
    X = pb.nodes
    bnd = np.zeros(len(X), bool)
    bnd[pb.inner] = bnd[pb.outer] = bnd[pb.sym] = True
    bnd |= (np.abs(X[:, 2]) < 1e-9) | (np.abs(X[:, 2] - I.LZ) < 1e-9)
    fixed = np.repeat(bnd, 3)
    pb.fixed, pb.free = np.where(fixed)[0], np.where(~fixed)[0]
    pb.ubar = np.zeros(pb.ndof)
    if field == "cartesian":
        pb.ubar = exact(X)[0].ravel()               # nodal values of u* (exact at nodes)
    base = pb.residual
    pb.residual = lambda u, chunk=4000: base(u, chunk) - f_ext
    t0 = time.time()
    pb.solve(nsteps=2)
    ue_, ge_ = exact(pos)
    U = pb.u.reshape(-1, 3)[pb.elements]                              # (e, 20, 3)
    uh = np.einsum("qa,eai->eqi", N, U).reshape(-1, 3)
    gh = np.einsum("eai,eqaj->eqij", U, pb.dNdX.numpy()).reshape(-1, 3, 3)
    l2, h1 = errors(uh, gh, ue_, ge_, w.ravel())
    u_star = keep
    name = {"fbar_p1": "Q20/P1 (mixed_b3)", "disp": "Q20 disp (mixed_b3)"}[formulation]
    return {"solver": name + ("" if field == "groove" else ", cartesian field"),
            "level": level, "n_dof": int(pb.ndof),
            "h": 1.0 / len(pb.elements) ** (1 / 3), "L2_rel": l2, "H1_rel": h1,
            "seconds": time.time() - t0, "boundary_nodes": int(bnd.sum())}


def iga(level, formulation):
    t0 = time.time()
    pb = IgaMMS(level, formulation)
    pb.solve(nsteps=2)
    l2, h1 = pb.errors()
    return {"solver": f"spline {formulation}", "level": level, "n_dof": int(pb.ndof),
            "h": 1.0 / I.LEVELS[level][0], "L2_rel": l2, "H1_rel": h1,
            "seconds": time.time() - t0}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["solver"], r["level"]) for r in res}
    plan = [("iga", "disp", lv) for lv in (1, 2, 3)] + [("iga", "fbar_p1", lv) for lv in (1, 2, 3)] \
        + [("q20", None, lv) for lv in (2376, 10200, 15960)]
    if argv:
        plan = [p for p in plan if p[0] in argv]
    for kind, form, lv in plan:
        name = f"spline {form}" if kind == "iga" else "Q20/P1 (mixed_b3)"
        if (name, lv) in done:
            continue
        r = iga(lv, form) if kind == "iga" else q20_p1(lv)
        res.append(r)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{name:<20} level {lv:>6} dof {r['n_dof']:7d} L2 {r['L2_rel']:.3e} "
              f"H1 {r['H1_rel']:.3e} ({r['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
