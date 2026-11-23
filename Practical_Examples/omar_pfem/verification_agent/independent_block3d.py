"""A second CODE for the clamped 3D block (round J, 2026-11-23).

The block's independent value so far came from the same code (torch-fem, Q8 against Q20).
Here the block is re-solved with scikit-fem: its own mesh, its own 27-node hexahedra
(ElementHex2, triquadratic Lagrange; torch-fem's Q20 is serendipity), its own
quadrature and assembly, and the residual, tangent and Newton loop written below from the
energy (the same formulas as independent_solver.py, extended to 3D), without any code of
torch-fem or of block3d.solve. Problem data as in block3d.py: block [0,2]x[0,1]x[0,0.5],
x = 0 clamped, dead traction (0, 0, -1) on x = 2, compressible neo-Hookean
psi = mu/2 (I1 - 3 - 2 ln J) + lam/2 (ln J)^2, E = 1000, nu = 0.3, 10 load increments.
Levels (n_x, n_y, n_z) = k (4, 2, 1). Stored apart: block3d_hex27_ladder.json.

    python -m omar_pfem.verification_agent.independent_block3d [--ks 1,2,3,4,6]
"""
import argparse
import json
import os
import time

import numpy as np
import skfem as fem

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "block3d_hex27_ladder.json")
LX, LY, LZ = 2.0, 1.0, 0.5
E, NU, TAU = 1000.0, 0.3, 1.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MID, ROOT, TIP = (1.0, 0.5, 0.5), (0.0, 0.5, 0.5), (2.0, 0.5, 0.5)


def _F(gu):
    F = gu.copy()
    for i in range(3):
        F[i, i] += 1.0
    return F


def _mat(F):                     # (3, 3, ...) -> (..., 3, 3)
    return np.moveaxis(np.moveaxis(F, 0, -1), 0, -1)


def _unmat(A):                   # (..., 3, 3) -> (3, 3, ...)
    return np.moveaxis(np.moveaxis(A, -1, 0), -1, 0)


def _det(F):
    return np.linalg.det(_mat(F))


def _invT(F):
    return _unmat(np.swapaxes(np.linalg.inv(_mat(F)), -1, -2))


def psi(F):
    J = _det(F)
    return MU / 2 * ((F ** 2).sum(axis=(0, 1)) - 3 - 2 * np.log(J)) + LAM / 2 * np.log(J) ** 2


def pk1(F):
    Fit = _invT(F)
    return MU * (F - Fit) + LAM * np.log(_det(F)) * Fit


def dpk1(F, dF):
    Fit = _invT(F)
    lnJ = np.log(_det(F))
    dFT = np.einsum("ji...->ij...", dF)
    A = np.einsum("ik...,kl...,lj...->ij...", Fit, dFT, Fit)
    tr = np.einsum("ij...,ij...->...", Fit, dF)
    return MU * dF + (MU - lnJ * LAM) * A + LAM * tr * Fit


def tangent(F):
    """dP_ij/dF_kl = mu d_ik d_jl + (mu - lam ln J) Fit_il Fit_kj + lam Fit_ij Fit_kl
    (the same derivative as dpk1, as a tensor; computed once per Newton iteration)."""
    Fit = _invT(F)
    lnJ = np.log(_det(F))
    I = np.eye(3)
    C = MU * np.einsum("ik,jl->ijkl", I, I)[..., None, None] * np.ones_like(lnJ)
    C = C + (MU - LAM * lnJ) * np.einsum("il...,kj...->ijkl...", Fit, Fit)
    return C + LAM * np.einsum("ij...,kl...->ijkl...", Fit, Fit)


@fem.BilinearForm
def a_form(du, v, w):
    return np.einsum("ijkl...,kl...,ij...->...", w["C"], du.grad, v.grad)


@fem.LinearForm
def r_form(v, w):
    return np.einsum("ij...,ij...->...", w["P"], v.grad)


@fem.LinearForm
def traction(v, w):
    return -TAU * v.value[2]


@fem.Functional
def energy_form(w):
    return psi(_F(w["u"].grad))


def solve(k, nsteps=10, newton_max=30, tol=1e-9):
    import pypardiso
    t0 = time.time()
    m = fem.MeshHex.init_tensor(np.linspace(0, LX, 4 * k + 1), np.linspace(0, LY, 2 * k + 1),
                                np.linspace(0, LZ, k + 1))
    e = fem.ElementVector(fem.ElementHex2())
    ib = fem.Basis(m, e, intorder=5)
    fb = fem.FacetBasis(m, e, facets=m.facets_satisfying(lambda x: np.isclose(x[0], LX)),
                        intorder=5)
    fext = fem.asm(traction, fb)
    assert abs(fext.sum() + TAU * LY * LZ) < 1e-10
    fixed = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    u = ib.zeros()
    free = np.setdiff1d(np.arange(ib.N), fixed)
    iters = []
    for step in range(1, nsteps + 1):
        f = step / nsteps * fext
        for it in range(newton_max):
            Fq = _F(ib.interpolate(u).grad)
            R = fem.asm(r_form, ib, P=pk1(Fq)) - f
            Rn = np.linalg.norm(R[free]) / np.linalg.norm(fext)
            if it > 0 and Rn < tol:
                break
            K = fem.asm(a_form, ib, C=tangent(Fq))
            du = fem.solve(*fem.condense(K, -R, D=fixed),
                           solver=lambda A, b: pypardiso.spsolve(A.tocsr(), b))
            u = u + du
        else:
            raise RuntimeError(f"Newton did not converge at step {step} (|R| = {Rn:.2e})")
        iters.append(it)
    w = dict(u=ib.interpolate(u))
    F = _F(w["u"].grad)
    J = _det(F)
    sig = np.einsum("ik...,jk...->ij...", pk1(F), F) / J            # Cauchy = P F^T / J
    trs = sig[0, 0] + sig[1, 1] + sig[2, 2]
    dev = sig - trs / 3 * np.eye(3)[:, :, None, None]
    vm = np.sqrt(1.5 * (dev ** 2).sum(axis=(0, 1)))
    x, wq = ib.global_coordinates().value, ib.dx
    dist = lambda c: np.sqrt(sum((x[i] - c[i]) ** 2 for i in range(3)))
    km = wq * np.exp(-(dist(MID) / 0.15) ** 2)
    kr = wq * np.exp(-(dist(ROOT) / 0.1) ** 2)
    U = u[ib.nodal_dofs].T
    j = int(np.argmin(sum((m.p[i] - TIP[i]) ** 2 for i in range(3))))
    assert np.sqrt(sum((m.p[i, j] - TIP[i]) ** 2 for i in range(3))) < 1e-12
    return {"k": k, "h": 0.5 / k, "n_elem": int(m.t.shape[1]), "n_dof": int(ib.N),
            "seconds": time.time() - t0, "newton_iterations": iters,
            "min_detF": float(J.min()),
            "strain_energy": float(fem.asm(energy_form, ib, **w)), "w_tip": float(U[j, 2]),
            "mid_gk_avg_sxx": float((sig[0, 0] * km).sum() / km.sum()),
            "root_gk_avg_von_mises": float((vm * kr).sum() / kr.sum()),
            "peak_von_mises": float(vm.max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,3,4,6")
    a = ap.parse_args(argv)
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    for k in [int(s) for s in a.ks.split(",")]:
        if k in {r["k"] for r in rows}:
            continue
        r = solve(k)
        rows.append(r)
        rows.sort(key=lambda r: r["k"])
        json.dump(rows, open(OUT, "w"), indent=1)
        print(f"Hex27 k={k} dof={r['n_dof']} ({r['seconds']:.0f} s) E={r['strain_energy']:.8g} "
              f"w_tip={r['w_tip']:.6g} mid={r['mid_gk_avg_sxx']:.6g} "
              f"root={r['root_gk_avg_von_mises']:.6g} min det F={r['min_detF']:.4f}", flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
