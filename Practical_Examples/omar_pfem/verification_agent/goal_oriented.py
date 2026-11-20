"""Goal-oriented (adjoint-weighted) error estimate from ONE mesh, checked
against the verified error of the refinement family (old item H2).

For a linear problem a(u, v) = f(v) and a linear quantity J, the error of the
Q2 solution u_h is J(u) - J(u_h) = f(z) - a(u_h, z), z the dual (adjoint)
solution a(v, z) = J(v). With z approximated in an enriched space V+ (here:
hierarchical degree-3 elements on the same mesh and the same Q2 geometry,
Q2 in V+), the dual-weighted residual equals J(u+) - J(u_h) exactly, so the
estimate is computed as that difference. Quantities that are linear in u:
compliance, u_tip and the region / Gaussian averages; the peak is not (max).

Test: the open-hole lamina (lamina.py, CFM paper), m = 1, 2, 4, 8 (m = 4 is
the paper's mesh). "True" error: against the m = 32 Q2 solution of the stored
ladder (lamina_ladder.json, verified: compliance error at m = 32 ~ 1e-8 rel).
Effectivity index = estimate / true error; 1 is exact.

What it cannot see (by construction): the error of the geometry. The enriched
solution lives on the same Q2 geometry, so a QoI whose error is mostly the
boundary approximation (here the compliance: its relative error equals the
relative area error of the mesh at every level) gets effectivity << 1. Sharp
regions are defined by which Gauss points fall inside them, so the quantity
itself changes with the quadrature of the enriched space: no estimate there.

    python -m omar_pfem.verification_agent.goal_oriented
"""
import json
import os

import numpy as np
import skfem as fem

from omar_pfem.verification_agent import lamina

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "goal_oriented_result.json")
LINEAR_QOIS = ("compliance", "u_tip", "hole_gamma12_avg", "hole_e22_avg", "far_gamma12_avg",
               "hole_gk_gamma12_avg", "hole_gk_e22_avg")


def solve_enriched(m):
    old = dict(lamina.ELEMENT)
    lamina.ELEMENT.update(make=lambda: fem.ElementQuadP(3), intorder=6, facet_intorder=8)
    try:
        return lamina.solve_level(m)
    finally:
        lamina.ELEMENT.clear()
        lamina.ELEMENT.update(old)


def main(ms=(1, 2, 4, 8), ref_m=32):
    ladder = {r["m"]: r for r in json.load(open(lamina.CACHE)) if r["g"] == 1.5}
    ref = ladder[ref_m]
    rows = []
    for m in ms:
        base = ladder[m] if m in ladder else lamina.solve_level(m)
        plus = solve_enriched(m)
        area_exact = np.pi * lamina.A_ELL * lamina.B_ELL - np.pi * lamina.RH ** 2
        row = {"m": m, "n_elem": base["n_elem"], "dof_q2": base["n_dof"],
               "area_rel_error": (base["area_mm2"] - area_exact) / area_exact,
               "dof_enriched": plus["n_dof"], "seconds_q2": base["wall_clock_s"],
               "seconds_enriched": plus["wall_clock_s"], "qois": {}}
        for q in LINEAR_QOIS:
            est = plus[q] - base[q]
            true = ref[q] - base[q]
            row["qois"][q] = {"q2": base[q], "enriched": plus[q], "reference": ref[q],
                              "estimate": est, "true_error": true,
                              "effectivity": est / true if true != 0 else None,
                              "true_rel": abs(true / ref[q]), "estimate_rel": abs(est / ref[q])}
        rows.append(row)
        print(f"m={m}: " + "; ".join(
            f"{q} eff {v['effectivity']:.3f} (true {v['true_rel']:.1e})"
            for q, v in row["qois"].items()), flush=True)
    json.dump({"reference_m": ref_m, "rows": rows}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
