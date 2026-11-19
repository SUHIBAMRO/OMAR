"""Independent solver check of B1 and B2 with a second finite-element code.

B1 (unit square, clamped bottom, dead traction t_y = -5 - 2 sin(pi x) on the
top edge in 10 increments; plane-strain compressible neo-Hookean
psi = mu/2 (I1 - 2 - 2 ln J) + lam/2 (ln J)^2, E = 1000 + 200 sin(pi x)
cos(pi y), nu = 0.3) re-solved with scikit-fem: its own mesh, element,
quadrature and assembly; the residual, tangent and Newton loop below are
written here from the energy, without any code of the project's solver.

Two modes:

  same    the SAME discrete problem as the project's B1 solver (Q4, 2x2
          Gauss, material at the element centroid, 2-point edge rule for
          the traction). The QoIs must agree to round-off/Newton tolerance:
          this checks the project's implementation.
  q2      a DIFFERENT discretisation (Q2 elements, 3x3 Gauss, material at
          every quadrature point, 3-point edge rule). Its values must
          approach the limits the verification agent extrapolated from the
          project's Q4 ladder, within the agent's stated uncertainty: this
          checks the agent's reference values independently.

B2 (quarter ring, see setup_b2) is checked the same way.

    python -m omar_pfem.verification_agent.independent_solver [--problems b1,b2]
                                                              [--mode same|q2|all]
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse.linalg as spla
import skfem as fem

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "independent_solver_result.json")


def E_field(x, y):
    return 1000.0 + 200.0 * np.sin(np.pi * x) * np.cos(np.pi * y)


def ty_field(x):
    return -5.0 - 2.0 * np.sin(np.pi * x)


def lame(E, nu=0.3):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


# ---- continuum: neo-Hookean, written out from psi -------------------------------------------
def _F(gu):
    F = gu.copy()
    F[0, 0] += 1.0
    F[1, 1] += 1.0
    return F


def _det(F):
    return F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]


def _invT(F):
    J = _det(F)
    return np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J


def psi(F, mu, lam):
    J = _det(F)
    I1 = (F ** 2).sum(axis=(0, 1))
    return mu / 2 * (I1 - 2 - 2 * np.log(J)) + lam / 2 * np.log(J) ** 2


def pk1(F, mu, lam):
    """P = mu (F - F^-T) + lam ln J F^-T."""
    Fit = _invT(F)
    return mu * (F - Fit) + lam * np.log(_det(F)) * Fit


def dpk1(F, dF, mu, lam):
    """Directional derivative of P: mu dF + (mu - lam ln J) F^-T dF^T F^-T
    + lam (F^-T : dF) F^-T."""
    Fit = _invT(F)
    lnJ = np.log(_det(F))
    dFT = np.einsum("ji...->ij...", dF)
    A = np.einsum("ik...,kl...,lj...->ij...", Fit, dFT, Fit)
    tr = np.einsum("ij...,ij...->...", Fit, dF)
    return mu * dF + (mu - lnJ * lam) * A + lam * tr * Fit


@fem.BilinearForm
def a_form(du, v, w):
    F = _F(w["u"].grad)
    return np.einsum("ij...,ij...->...", dpk1(F, du.grad, w["mu"], w["lam"]), v.grad)


@fem.LinearForm
def r_form(v, w):
    F = _F(w["u"].grad)
    return np.einsum("ij...,ij...->...", pk1(F, w["mu"], w["lam"]), v.grad)


@fem.Functional
def energy_form(w):
    return psi(_F(w["u"].grad), w["mu"], w["lam"])


# ---- problems ------------------------------------------------------------------------------
def _elem(mode):
    # same: Q4, 2x2 Gauss, 2-point edge rule; q2: Q2, 3x3 Gauss, 3-point edge rule
    return (fem.ElementQuad1(), 2, 2) if mode == "same" else (fem.ElementQuad2(), 4, 4)


def setup_b1(N, mode):
    """N = nodes per side of the Q4 mesh (as in b1_ladder); for q2 the mesh
    has N-1 Q2 elements per side (same element size, ~4x the dof)."""
    m = fem.MeshQuad.init_tensor(np.linspace(0, 1, N), np.linspace(0, 1, N))
    e, iq, iqf = _elem(mode)
    ib = fem.Basis(m, fem.ElementVector(e), intorder=iq)
    if mode == "same":
        c = m.p[:, m.t].mean(axis=1)                    # element centroids, as the project
        E = np.repeat(E_field(c[0], c[1])[:, None], ib.X.shape[1], axis=1)
    else:
        x = ib.global_coordinates().value
        E = E_field(x[0], x[1])
    mu, lam = lame(E)
    top = fem.FacetBasis(m, fem.ElementVector(e), facets=m.facets_satisfying(
        lambda x: np.isclose(x[1], 1.0)), intorder=iqf)
    fext = fem.asm(fem.LinearForm(lambda v, w: ty_field(w.x[0]) * v.value[1]), top)
    fixed = ib.get_dofs(lambda x: np.isclose(x[1], 0.0)).all()
    return m, ib, mu, lam, fext, fixed


def setup_b2(N, mode):
    """B2 quarter ring R_in = 1, R_out = 2: E = 1000 + 200 sin(2 theta), nu = 0.3,
    dead pressure p = 5 + 2 sin(2 theta) on the inner arc (pushing outward),
    u_y = 0 on theta = 0, u_x = 0 on theta = pi/2, outer arc free.
    same: Q4 on the N x N polar grid, material at every Gauss point, traction
    along each chord's outward-from-centre normal (as the project); q2: Q2
    with the arcs represented exactly at the mid-side nodes, traction along
    the curved boundary's normal."""
    mp = fem.MeshQuad.init_tensor(np.linspace(1.0, 2.0, N), np.linspace(0.0, np.pi / 2, N))
    inner = mp.facets_satisfying(lambda x: np.isclose(x[0], 1.0))   # in (r, theta)
    if mode != "same":
        mp = fem.MeshQuad2.from_mesh(mp)
    m = mp.morphed(lambda p: p[0] * np.cos(p[1]), lambda p: p[0] * np.sin(p[1]))
    e, iq, iqf = _elem(mode)
    ib = fem.Basis(m, fem.ElementVector(e), intorder=iq)
    x = ib.global_coordinates().value
    E = 1000.0 + 200.0 * np.sin(2.0 * np.arctan2(x[1], x[0]))
    mu, lam = lame(E)
    fb = fem.FacetBasis(m, fem.ElementVector(e), facets=inner, intorder=iqf)

    def trac(v, w):
        th = np.arctan2(w.x[1], w.x[0])
        pr = 5.0 + 2.0 * np.sin(2.0 * th)
        return -pr * (w.n[0] * v.value[0] + w.n[1] * v.value[1])   # -n: away from the centre
    fext = fem.asm(fem.LinearForm(trac), fb)
    fixed = np.concatenate([ib.get_dofs(lambda x: np.isclose(x[1], 0.0)).all("u^2"),
                            ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all("u^1")])
    return m, ib, mu, lam, fext, fixed


PROBLEMS = {
    "b1": dict(setup=setup_b1, geometry="unit_square", tol=1e-10,
               ladder="b1_ladder_neo_hookean.json",
               query={"top_off_center": (0.7, 1.0), "mid_domain": (0.3, 0.6)},
               region=((0.5, 0.5), 0.15), kernel=None,
               qois=("max_disp_mag", "strain_energy", "v_top_off_center", "u_top_off_center",
                     "u_mid_domain", "v_mid_domain", "interior_avg_PK1_frob",
                     "peak_PK1_frob")),
    "b2": dict(setup=setup_b2, geometry="quarter_ring", tol=1e-9,
               ladder="b2_ladder_neo_hookean.json",
               query={"inner": (0.5, 0.5 * np.sqrt(3.0)), "mid": (0.75, 0.75 * np.sqrt(3.0))},
               region=((1.5 * np.cos(np.pi / 4), 1.5 * np.sin(np.pi / 4)), 0.15), kernel=0.1,
               qois=("max_disp_mag", "u_inner", "v_inner", "u_mid", "v_mid", "strain_energy",
                     "interior_avg_PK1_frob", "interior_gk_PK1_frob", "peak_PK1_frob")),
}


def solve(prob, N, mode, nsteps=10, newton_max=30):
    P = PROBLEMS[prob]
    m, ib, mu, lam, fext, fixed = P["setup"](N, mode)
    u = ib.zeros()
    t0 = time.time()
    hist = []
    for step in range(1, nsteps + 1):
        f = step / nsteps * fext
        prev = np.inf
        for it in range(newton_max):
            w = dict(u=ib.interpolate(u), mu=mu, lam=lam)
            R = fem.asm(r_form, ib, **w) - f
            Rn = np.linalg.norm(np.delete(R, fixed))
            # converged, or stagnating at the round-off floor (recorded in hist)
            if it > 0 and (Rn < P["tol"] or (Rn < 1e-7 and Rn > 0.5 * prev)):
                break
            prev = Rn
            K = fem.asm(a_form, ib, **w)
            du = fem.solve(*fem.condense(K, -R, D=fixed), solver=spla.spsolve)
            u = u + du
        else:
            raise RuntimeError(f"Newton did not converge at step {step} (|R| = {Rn:.2e})")
        hist.append((it, float(Rn)))
    return m, ib, u, mu, lam, hist, time.time() - t0


def qois(prob, N, mode, m, ib, u, mu, lam):
    P = PROBLEMS[prob]
    w = dict(u=ib.interpolate(u), mu=mu, lam=lam)
    F = _F(w["u"].grad)
    Pn = np.sqrt((pk1(F, mu, lam) ** 2).sum(axis=(0, 1)))         # (nel, nqp)
    x = ib.global_coordinates().value
    wq = ib.dx                                                     # quadrature weight x detJ
    (cx, cy), r = P["region"]
    d = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2)
    inside = d < r
    ux, uy = u[ib.get_dofs().all("u^1")], u[ib.get_dofs().all("u^2")]   # all nodes
    out = {"problem": prob, "N": N, "mode": mode, "n_dof": int(ib.N),
           "max_disp_mag": float(np.sqrt(ux ** 2 + uy ** 2).max()),
           "strain_energy": float(fem.asm(energy_form, ib, **w)),
           "peak_PK1_frob": float(Pn.max()),
           "interior_avg_PK1_frob": float((Pn * wq)[inside].sum() / wq[inside].sum()),
           "interior_region_gauss_points": int(inside.sum()),
           "min_detF": float(_det(F).min())}
    if P["kernel"]:
        k = wq * np.exp(-(d / P["kernel"]) ** 2)
        out["interior_gk_PK1_frob"] = float((Pn * k).sum() / k.sum())
    U = u[ib.nodal_dofs].T                                         # (n_vertices, 2)
    for name, (qx, qy) in P["query"].items():
        j = int(np.argmin((m.p[0] - qx) ** 2 + (m.p[1] - qy) ** 2))
        assert np.hypot(m.p[0, j] - qx, m.p[1, j] - qy) < 1e-12, "query point not a vertex"
        out[f"u_{name}"], out[f"v_{name}"] = float(U[j, 0]), float(U[j, 1])
    return out


def run(prob, N, mode, cache):
    key = f"{prob}:{mode}:{N}"
    if key in cache:
        return cache[key]
    m, ib, u, mu, lam, hist, sec = solve(prob, N, mode)
    row = qois(prob, N, mode, m, ib, u, mu, lam)
    row.update(wall_clock_s=sec, newton=hist)
    cache[key] = row
    json.dump(cache, open(OUT, "w"), indent=1)
    print(f"{prob} {mode:<5} N={N:<4} {sec:6.1f}s  " + "  ".join(
        f"{k}={v:.8g}" for k, v in row.items() if isinstance(v, float) and k != "wall_clock_s"),
        flush=True)
    return row


def _ladder(prob):
    rows = json.load(open(os.path.join(HERE, PROBLEMS[prob]["ladder"])))
    if prob == "b1":
        return {r["N"]: r for r in rows if r.get("grading", 0.0) == 0.0}
    return {r["N"]: r for r in rows if r["r_grading"] == 1.0 and r["newton_tol"] == 1e-8}


def compare_same(prob, cache):
    lad = _ladder(prob)
    rows = []
    for key, r in sorted(cache.items()):
        if not key.startswith(f"{prob}:same:") or r["N"] not in lad:
            continue
        ref = lad[r["N"]]
        for q in PROBLEMS[prob]["qois"]:
            if q in ref:
                rel = abs(r[q] - ref[q]) / max(abs(ref[q]), 1e-300)
                rows.append({"N": r["N"], "qoi": q, "project": ref[q], "scikit_fem": r[q],
                             "rel_diff": rel})
    return rows


def compare_q2(prob, cache):
    """Q2 values vs the agent's extrapolated limits (5-level fit) and their
    stated uncertainty (relative), for the QoIs the agent passed with a rate."""
    from omar_pfem.verification_agent.verify_report import rules
    R = rules(PROBLEMS[prob]["geometry"])
    q2 = sorted((r for k, r in cache.items() if k.startswith(f"{prob}:q2:")),
                key=lambda r: r["N"])
    out = []
    for q in PROBLEMS[prob]["qois"]:
        if q not in R["v"] or not q2:
            continue
        g = R["v"][q].get("qoi_convergence_multilevel", {})
        ov, ev = R["combine"][q][:2]
        lim, U = g.get("fit_limit"), g.get("uncertainty_fine")
        vals = [r[q] for r in q2]
        row = {"qoi": q, "agent_verdict": f"{ov}/{ev}", "agent_limit": lim,
               "agent_uncertainty_rel": U, "q2_N": [r["N"] for r in q2], "q2_values": vals}
        if len(vals) >= 3:
            # Aitken / Richardson from the finest three Q2 values (ratio-2 family)
            a, b, c = vals[-3:]
            den = (c - b) - (b - a)
            if den != 0 and (c - b) * (b - a) > 0 and abs(c - b) < abs(b - a):
                row["q2_extrapolated"] = c - (c - b) ** 2 / den
        if lim is not None and ev == "rate" and U is not None:
            dd = abs(vals[-1] - lim) / abs(lim)
            row["q2_finest_vs_limit_rel"] = dd
            row["within_uncertainty"] = bool(dd <= U)
            if "q2_extrapolated" in row:
                de = abs(row["q2_extrapolated"] - lim) / abs(lim)
                row["q2_extrapolated_vs_limit_rel"] = de
                row["extrapolated_within_uncertainty"] = bool(de <= U)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="b1,b2")
    ap.add_argument("--mode", default="all", choices=("same", "q2", "all"))
    ap.add_argument("--b1-Ns", default="11,21,41,81,161")
    ap.add_argument("--b2-Ns", default="7,13,25,49,97,193")
    ap.add_argument("--b2-q2-Ns", default="7,13,25,49,97")
    a = ap.parse_args(argv)
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    # rows of the first (B1-only) version: "same:N" -> "b1:same:N"
    cache = {(k if k.count(":") == 2 else "b1:" + k): v for k, v in cache.items() if ":" in k}
    Ns = {("b1", "same"): a.b1_Ns, ("b1", "q2"): a.b1_Ns,
          ("b2", "same"): a.b2_Ns, ("b2", "q2"): a.b2_q2_Ns}
    res = {}
    for prob in a.problems.split(","):
        for mode in ("same", "q2"):
            if a.mode in (mode, "all"):
                for N in map(int, Ns[(prob, mode)].split(",")):
                    run(prob, N, mode, cache)
        res[prob] = {"same": compare_same(prob, cache), "q2": compare_q2(prob, cache)}
        print(f"\n=== {prob}")
        if res[prob]["same"]:
            worst = max(res[prob]["same"], key=lambda r: r["rel_diff"])
            print(f"same discretisation: max relative difference {worst['rel_diff']:.2e} "
                  f"({worst['qoi']}, N={worst['N']}) over {len(res[prob]['same'])} values")
        for r in res[prob]["q2"]:
            if "q2_finest_vs_limit_rel" in r:
                print(f"q2 {r['qoi']:<24} finest {r['q2_values'][-1]:.8g}  agent limit "
                      f"{r['agent_limit']:.8g} +- {r['agent_uncertainty_rel']:.2e}  diff "
                      f"{r['q2_finest_vs_limit_rel']:.2e}  "
                      f"{'OK' if r['within_uncertainty'] else 'OUTSIDE'}"
                      + (f"; Q2 extrapolated {r['q2_extrapolated']:.8g} diff "
                         f"{r['q2_extrapolated_vs_limit_rel']:.2e} "
                         f"{'OK' if r['extrapolated_within_uncertainty'] else 'OUTSIDE'}"
                         if "q2_extrapolated_vs_limit_rel" in r else ""))
            else:
                print(f"q2 {r['qoi']:<24} agent {r['agent_verdict']} (no rate limit); Q2 values "
                      + ", ".join(f"{v:.6g}" for v in r["q2_values"]))
    cache["comparison"] = res
    json.dump(cache, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
