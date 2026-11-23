"""Independent values for rate-certified quantities that had none (2026-11-23).

  b1   interior_gk_PK1_frob of B1 with the second FE code on Q2 elements
       (independent_solver.py; kernel length 0.1 as in b1_ladder.KERNEL_ELL). The
       other B1 Q2 quantities already stored are recomputed and must reproduce the
       stored values (check of this script, not of the physics).
  cook u_tip, v_tip, max_disp_mag, interior_gk_PK1_frob of Cook's membrane on
       Q2 elements (3x3 Gauss). Cook's primary family is Q4 in the same scikit-fem
       code, so this is an independent DISCRETIZATION (element family), not an
       independent code; the code itself reproduces torch-fem on B1/B2 to 1e-13.

Results: independent_extra_result.json. No acceptance criterion is involved here;
the comparison is made by crosscheck.py.

    python -m omar_pfem.verification_agent.crosscheck_extra b1 cook
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla
import skfem as fem

from omar_pfem.verification_agent import cook_ladder as C
from omar_pfem.verification_agent import independent_solver as I

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "independent_extra_result.json")
B1_NS = (11, 21, 41, 81, 161)
COOK_NS = (4, 8, 16, 32, 64, 128)


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(res):
    json.dump(res, open(OUT, "w"), indent=1)


def b1(res):
    stored = json.load(open(os.path.join(HERE, "independent_solver_result.json")))
    I.PROBLEMS["b1"]["kernel"] = 0.1
    for N in B1_NS:
        key = f"b1:q2:{N}"
        if key in res:
            continue
        m, ib, u, mu, lam, hist, sec = I.solve("b1", N, "q2")
        q = I.qois("b1", N, "q2", m, ib, u, mu, lam)
        old = stored[key]
        repro = max(abs(q[k] - old[k]) / max(abs(old[k]), 1e-30)
                    for k in ("strain_energy", "max_disp_mag", "interior_avg_PK1_frob"))
        q.update(seconds=sec, reproduces_stored_rel=repro, h=1.0 / (N - 1))
        res[key] = q
        _save(res)
        print(f"b1 q2 N={N}: gk={q['interior_gk_PK1_frob']:.8g} repro={repro:.1e} ({sec:.0f} s)",
              flush=True)


def cook_q2(N, nsteps=10, tol=1e-9, newton_max=30):
    m = C.mesh(N)
    e = fem.ElementQuad2()
    ib = fem.Basis(m, fem.ElementVector(e), intorder=4)
    mu, lam = I.lame(np.full((m.t.shape[1], ib.X.shape[1]), C.E_MOD), C.NU)
    right = fem.FacetBasis(m, fem.ElementVector(e),
                           facets=m.facets_satisfying(lambda x: np.isclose(x[0], 48.0)),
                           intorder=4)
    fext = fem.asm(fem.LinearForm(lambda v, w: C.TAU * v.value[1]), right)
    fixed = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    u = ib.zeros()
    t0 = time.time()
    for step in range(1, nsteps + 1):
        f = step / nsteps * fext
        prev = np.inf
        for it in range(newton_max):
            w = dict(u=ib.interpolate(u), mu=mu, lam=lam)
            R = fem.asm(I.r_form, ib, **w) - f
            Rn = np.linalg.norm(np.delete(R, fixed))
            if it > 0 and (Rn < tol or (Rn < 1e-7 and Rn > 0.5 * prev)):
                break
            prev = Rn
            K = fem.asm(I.a_form, ib, **w)
            u = u + fem.solve(*fem.condense(K, -R, D=fixed), solver=spla.spsolve)
        else:
            raise RuntimeError(f"Newton did not converge at step {step}")
    w = dict(u=ib.interpolate(u), mu=mu, lam=lam)
    F = I._F(w["u"].grad)
    Pn = np.sqrt((I.pk1(F, mu, lam) ** 2).sum(axis=(0, 1)))
    x, wq = ib.global_coordinates().value, ib.dx
    (cx, cy), r = C.REGION
    d = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2)
    k = wq * np.exp(-(d / C.KERNEL_ELL) ** 2)
    U = u[ib.nodal_dofs].T
    j = int(np.argmin((m.p[0] - C.TIP[0]) ** 2 + (m.p[1] - C.TIP[1]) ** 2))
    ux, uy = u[ib.get_dofs().all("u^1")], u[ib.get_dofs().all("u^2")]
    return {"N": N, "h": 1.0 / N, "n_dof": int(ib.N), "seconds": time.time() - t0,
            "u_tip": float(U[j, 0]), "v_tip": float(U[j, 1]),
            "max_disp_mag": float(np.sqrt(ux ** 2 + uy ** 2).max()),
            "strain_energy": float(fem.asm(I.energy_form, ib, **w)),
            "interior_gk_PK1_frob": float((Pn * k).sum() / k.sum()),
            "interior_avg_PK1_frob": float((Pn * wq)[d < r].sum() / wq[d < r].sum()),
            "peak_PK1_frob": float(Pn.max()), "min_detF": float(I._det(F).min())}


def cook(res):
    for N in COOK_NS:
        key = f"cook:q2:{N}"
        if key in res:
            continue
        res[key] = cook_q2(N)
        _save(res)
        r = res[key]
        print(f"cook q2 N={N}: u_tip={r['u_tip']:.8g} v_tip={r['v_tip']:.8g} "
              f"gk={r['interior_gk_PK1_frob']:.8g} ({r['seconds']:.0f} s)", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    res = _load()
    if "b1" in argv:
        b1(res)
    if "cook" in argv:
        cook(res)


if __name__ == "__main__":
    main()
