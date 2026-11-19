"""B3 with a mixed (mean-dilatation / F-bar) formulation: does the Gauss-point
pressure scatter at nu = 0.45 come from the volumetric part?

Own assembly on the project's curved 20-node mesh (build_quadratic_mesh_
with_curved_surfaces; torchfem's Hexa2 shape functions only). Element energy
written here, gradients/Hessians by torch.func, sparse direct solve
(pypardiso), Newton with 10 load increments, as the stored B3 solves.

  formulation "disp":  psi(F) at every Gauss point (the stored B3 model);
                       with the 8-point rule it must reproduce b3_extended.json
                       (validation of this code).
  formulation "fbar":  mean dilatation, F_bar = (theta_e / J)^(1/3) F with
                       theta_e = element average of J. Variationally
                       equivalent to a three-field Hu-Washizu element with
                       piecewise-constant pressure and dilatation (Q20/P0),
                       i.e. a MIXED u/p formulation; the pressure is constant
                       per element by construction.

QoIs: reaction moment about the y axis through (0,0,0.5), strain energy,
Gauss-point pressure scatter within 2 R0 of the groove root (as
diagnose._b3_pressure_scatter), and the sharp-sphere and Gaussian-weighted
(l = 2 R0) region averages of region_study.py.

    python -m omar_pfem.verification_agent.mixed_b3 [--levels 2376,10200]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch.func import grad, jacfwd, vmap

from omar_pfem.data.data_generate_B3 import groove_radius_of_curvature, rigid_rotation_displacement
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH
from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    E, LZ, NU, PHI, R_IN0, build_quadratic_mesh_with_curved_surfaces)

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mixed_b3_result.json")
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
G3 = (3.0 / 5.0) ** 0.5


def rule(name):
    from torchfem.elements import Hexa2
    if name == "reduced":
        return Hexa2.ipoints.double(), Hexa2.iweights.double()
    p1, w1 = (-G3, 0.0, G3), (5 / 9, 8 / 9, 5 / 9)
    ip = torch.tensor([[a, b, c] for c in p1 for b in p1 for a in p1])
    iw = torch.tensor([wa * wb * wc for wc in w1 for wb in w1 for wa in w1])
    return ip, iw


def psi(F):
    lnJ = torch.linalg.slogdet(F)[1]
    return MU / 2 * ((F ** 2).sum((-2, -1)) - 3 - 2 * lnJ) + LAM / 2 * lnJ ** 2


def det3(F):
    """Explicit 3x3 determinant: torch.linalg.det's double backward is NaN at
    F = I (see torchfem_comparison.neo_hookean_psi_3d), which made the F-bar
    tangent NaN in undeformed elements."""
    return (F[..., 0, 0] * (F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1])
            - F[..., 0, 1] * (F[..., 1, 0] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 0])
            + F[..., 0, 2] * (F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0]))


def fbar(F, w, formulation):
    if formulation == "disp":
        return F
    J = det3(F)
    theta = (w * J).sum() / w.sum()
    return (theta / J)[:, None, None] ** (1.0 / 3.0) * F


def element_energy(ue, dNdX, w, formulation):
    """ue (20,3), dNdX (nq,20,3), w (nq,)."""
    F = torch.eye(3) + torch.einsum("ai,qaj->qij", ue, dNdX)
    return (w * psi(fbar(F, w, formulation))).sum()


class Problem:
    def __init__(self, Ntheta, Nr, Nz, quadrature="reduced", formulation="disp"):
        from torchfem.elements import Hexa2
        nodes, elements, inner, outer, sym = build_quadratic_mesh_with_curved_surfaces(
            Ntheta, Nr, Nz, r_grading=1.0, groove_depth=GROOVE_DEPTH)
        self.nodes, self.elements = nodes, elements
        self.inner, self.outer, self.sym = inner, outer, sym
        self.formulation = formulation
        ip, iw = rule(quadrature)
        el = Hexa2()
        N = el.N(ip)                                     # (nq, 20)
        dNdxi = el.B(ip).transpose(1, 2)                 # (nq, 20, 3)
        X = torch.tensor(nodes)[torch.tensor(elements)]  # (ne, 20, 3)
        Jm = torch.einsum("eai,qaj->eqij", X, dNdxi)     # dX/dxi
        self.detJ = torch.linalg.det(Jm)
        self.dNdX = torch.einsum("qaj,eqji->eqai", dNdxi, torch.linalg.inv(Jm))
        self.w = iw[None, :] * self.detJ.abs()
        self.pos = torch.einsum("qa,eai->eqi", N, X).numpy()
        ne = len(elements)
        dofs = (3 * elements[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 60)
        self.dofs = dofs
        self.rows = np.repeat(dofs, 60, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 60)).ravel()
        self.ndof = 3 * len(nodes)
        f = lambda ue, d, w: element_energy(ue, d, w, formulation)
        self._g = vmap(grad(f))
        self._h = vmap(jacfwd(grad(f)))
        fixed = np.zeros((len(nodes), 3), bool)
        fixed[outer, :] = True
        fixed[inner, :] = True
        fixed[sym, 1] = True
        self.fixed = np.where(fixed.ravel())[0]
        self.free = np.where(~fixed.ravel())[0]
        ux, uy, uz = rigid_rotation_displacement(nodes[inner], LZ, PHI)
        self.ubar = np.zeros((len(nodes), 3))
        self.ubar[inner] = np.stack([ux, uy, uz], 1)

    def _ue(self, u):
        return torch.tensor(u.reshape(-1, 3))[torch.tensor(self.elements)]

    def residual(self, u, chunk=4000):
        R = np.zeros(self.ndof)
        ue = self._ue(u)
        for s in range(0, len(self.elements), chunk):
            g = self._g(ue[s:s + chunk], self.dNdX[s:s + chunk], self.w[s:s + chunk])
            np.add.at(R, self.dofs[s:s + chunk].ravel(), g.reshape(-1).numpy())
        return R

    def tangent(self, u, chunk=1000):
        ue = self._ue(u)
        vals = []
        for s in range(0, len(self.elements), chunk):
            h = self._h(ue[s:s + chunk], self.dNdX[s:s + chunk], self.w[s:s + chunk])
            vals.append(h.reshape(-1, 60, 60).numpy().ravel())
        return sp.csr_matrix((np.concatenate(vals), (self.rows, self.cols)),
                             shape=(self.ndof, self.ndof))

    def solve(self, nsteps=10, tol=1e-9, newton_max=30):
        import pypardiso
        u = np.zeros(self.ndof)
        hist = []
        for step in range(1, nsteps + 1):
            u[self.fixed] = (step / nsteps * self.ubar.ravel())[self.fixed]
            for it in range(newton_max):
                R = self.residual(u)
                Rn = np.linalg.norm(R[self.free])
                if it > 0 and Rn < tol:
                    break
                K = self.tangent(u)
                Kff = K[self.free][:, self.free]
                u[self.free] -= pypardiso.spsolve(Kff.tocsr(), R[self.free])
            else:
                raise RuntimeError(f"step {step}: |R| = {Rn:.2e}")
            hist.append((it, float(Rn)))
        self.u, self.hist = u, hist
        return u

    def gauss_stress(self):
        """Cauchy stress at the Gauss points from F_bar (constant pressure per
        element for fbar)."""
        ue = self._ue(self.u)
        F = torch.eye(3) + torch.einsum("eai,eqaj->eqij", ue, self.dNdX)
        Fb = torch.stack([fbar(F[e], self.w[e], self.formulation) for e in range(len(F))]) \
            if self.formulation != "disp" else F
        Fb = Fb.detach().requires_grad_(True)
        (P,) = torch.autograd.grad(psi(Fb).sum(), Fb)
        J = torch.linalg.det(Fb)
        sigma = torch.einsum("eqij,eqkj->eqik", P, Fb) / J[..., None, None]
        return sigma.detach().numpy(), J.detach().numpy(), psi(Fb).detach().numpy()


def qois(pb):
    sigma, Jb, ps = pb.gauss_stress()
    w = pb.w.numpy()
    R = pb.residual(pb.u).reshape(-1, 3)
    x_def = pb.nodes + pb.u.reshape(-1, 3)
    arm = x_def[pb.inner] - np.array([0.0, 0.0, LZ / 2.0])
    f = R[pb.inner]
    moment = float((arm[:, 2] * f[:, 0] - arm[:, 0] * f[:, 2]).sum())
    ref = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
    R0 = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    pos, s, ww = pb.pos.reshape(-1, 3), sigma.reshape(-1, 3, 3), w.ravel()
    d = np.linalg.norm(pos - ref[None, :], axis=1)
    p = -np.trace(s, axis1=1, axis2=2) / 3.0
    near = d < 2 * R0
    out = {"reaction_moment_y": moment, "strain_energy": float((ps * w).sum()),
           "min_detF_bar": float(Jb.min()),
           "pressure_std_2R0": float(np.std(p[near])),
           "n_gauss_2R0": int(near.sum())}
    th = np.arctan2(pos[:, 1], pos[:, 0])
    c, sn = np.cos(th), np.sin(th)
    Q = np.zeros((len(th), 3, 3))
    Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1], Q[:, 2, 2] = c, sn, -sn, c, 1.0
    sc = np.einsum("nai,nij,nbj->nab", Q, s, Q)
    dev = s + p[:, None, None] * np.eye(3)
    meas = {"sigma_xx": s[:, 0, 0], "sigma_yy": s[:, 1, 1], "sigma_zz": s[:, 2, 2],
            "sigma_xz": s[:, 0, 2], "pressure": p, "cyl_rr": sc[:, 0, 0], "cyl_tt": sc[:, 1, 1],
            "cyl_rz": sc[:, 0, 2],
            "frob": np.sqrt(np.einsum("nij,nij->n", s, s)),
            "von_mises": np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))}
    for kind, wt in (("sharp", ww * (d < R0)), ("gk2", ww * np.exp(-(d / (2 * R0)) ** 2))):
        for k, v in meas.items():
            out[f"{kind}_{k}_avg"] = float((v * wt).sum() / wt.sum())
    return out


LEVELS = {2376: (13, 12, 19), 10200: (21, 18, 31)}


def run(level, quadrature, formulation, cache):
    key = f"{level}:{quadrature}:{formulation}"
    if key in cache:
        return cache[key]
    t0 = time.time()
    pb = Problem(*LEVELS[level], quadrature=quadrature, formulation=formulation)
    pb.solve()
    row = {"n_elem": level, "quadrature": quadrature, "formulation": formulation,
           "seconds": time.time() - t0, "newton": pb.hist, **qois(pb)}
    cache[key] = row
    json.dump(cache, open(OUT, "w"), indent=1)
    print(f"{key:<24} {row['seconds']:.0f}s  M_y={row['reaction_moment_y']:.6f}  "
          f"W={row['strain_energy']:.6f}  p_std(2R0)={row['pressure_std_2R0']:.4f}  "
          f"gk2_xx={row['gk2_sigma_xx_avg']:.5f}  gk2_frob={row['gk2_frob_avg']:.5f}", flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="2376,10200")
    ap.add_argument("--cases", default="reduced:disp,full:disp,reduced:fbar,full:fbar")
    a = ap.parse_args(argv)
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for lv in map(int, a.levels.split(",")):
        for case in a.cases.split(","):
            q, f = case.split(":")
            run(lv, q, f, cache)


if __name__ == "__main__":
    main()
