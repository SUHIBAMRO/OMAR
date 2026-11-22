"""Rule (2026-11-22): a rate-certified uncertainty is FINAL only if an independent
discretization agrees with the certified limit within the stated uncertainties.

Why: the B3 Gaussian groove averages were certified with 0.11-0.18% from the
Q20 family, yet an independent spline discretization lies about 2% away
(report Section 25). Convergence of one family bounds that family's sequence,
not the distance to the continuum value.

For every quantity the agent passes with a rate (rules plan, as `agent rules`):
  limit, U    the multilevel fit limit and its uncertainty (relative to the
              quantity's scale), else the three-level Richardson value and GCI
  independent value and its own uncertainty, where one exists:
    unit_square, quarter_ring   second FE code with Q2 elements (independent_solver.py):
                                finest level, U = change over its last step
    open_hole_lamina            splines p = 3 on exact geometry (iga_lamina.py), m = 16,
                                U = change from m = 8
    bushing_groove              moment, energy: spline extrapolated limit (iga_b3.py, p = 2),
                                U = its multilevel uncertainty; Gaussian averages: spline
                                P1 at the finest level, U = its last change (not converged)
  status      cross_checked  |limit - independent| <= U + U_ind
              contradicted   otherwise: the certified U is not a bound on the error;
                             the gap is the honest uncertainty
              single_family  no independent value: the uncertainty is not final

    python -m omar_pfem.verification_agent.crosscheck
"""
import json
import os

import numpy as np

from omar_pfem.verification_agent import gates

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "crosscheck_result.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape", "edge_crack")


def _j(name):
    return json.load(open(os.path.join(HERE, name)))


def independent(geometry, qoi):
    """(value, uncertainty as an absolute number, source) or None."""
    if geometry in ("unit_square", "quarter_ring"):
        tag = "b1" if geometry == "unit_square" else "b2"
        d = _j("independent_solver_result.json")
        rows = sorted((v for k, v in d.items() if k.startswith(f"{tag}:q2:") and qoi in v),
                      key=lambda r: r["N"])
        if len(rows) < 2:
            return None
        return rows[-1][qoi], abs(rows[-1][qoi] - rows[-2][qoi]), "second FE code, Q2"
    if geometry == "open_hole_lamina":
        d = _j("iga_lamina_result.json")["iga"]
        pick = lambda m: next((r for r in d if r["p"] == 3 and r["m"] == m and r["beta"] == 2.8
                               and qoi in r), None)
        a, b = pick(16), pick(8)
        if a is None or b is None:
            return None
        return a[qoi], abs(a[qoi] - b[qoi]), "splines p = 3, exact geometry"
    if geometry == "bushing_groove":
        d = _j("iga_b3_result.json")
        if qoi in ("reaction_moment_y", "strain_energy"):
            rows = [d[f"L{l}:p2"] for l in (1, 2, 3, 4)]
            g = gates.multilevel_convergence_gate([1.0 / r["spans"][0] for r in rows],
                                                  [r[qoi] for r in rows], band_tol=0.01,
                                                  gci_tol=0.02, order_window=(0.5, 5))
            lim = g.get("fit_limit")
            return lim, g.get("uncertainty_fine", 0.0) * abs(rows[-1][qoi]), "splines p = 2"
        if qoi.startswith("gk2_"):
            key = "gk2trunc_" + qoi[len("gk2_"):]
            if f"L4:p2:fbar_p1" not in d or key not in d["L4:p2:fbar_p1"]:
                return None
            a, b = d["L4:p2:fbar_p1"][key], d["L3:p2:fbar_p1"][key]
            return a, abs(a - b), "spline P1, finest level (not converged)"
    return None


def decide(limit, U_rel, independent_value, U_ind_abs, scale):
    """The rule itself: cross_checked iff the gap is within both uncertainties."""
    if independent_value is None:
        return "single_family", None
    gap = abs(limit - independent_value) / scale
    return ("cross_checked" if gap <= U_rel + U_ind_abs / scale else "contradicted"), gap


def agent_limit(R, qoi):
    v = R["v"][qoi]
    ml = v.get("qoi_convergence_multilevel", {})
    if ml.get("evidence") == "rate":
        return ml["fit_limit"], ml["uncertainty_fine"]
    c3 = v.get("qoi_convergence", {})
    if c3.get("evidence") == "rate":
        return c3["richardson_extrapolate"], c3["gci_fine"]
    return None


def run():
    from omar_pfem.verification_agent.verify_report import rules
    rows = []
    for geo in PROBLEMS:
        R = rules.__wrapped__(geo)
        ad = R["adapter"]
        for q, comb in R["combine"].items():
            if comb[0] != "PASS" or comb[1] != "rate":
                continue
            al = agent_limit(R, q)
            if al is None:
                continue
            lim, U = al
            _, vals, scale = ad.series(q)
            sc = scale if scale else abs(vals[-1])
            ind = independent(geo, q)
            row = {"problem": geo, "qoi": q, "limit": lim, "U_rel": U, "scale": sc}
            if ind is None or ind[0] is None:
                row["status"] = "single_family"
            else:
                val, uind, src = ind
                status, gap = decide(lim, U, val, uind, sc)
                row.update(independent=val, U_ind_rel=uind / sc, gap_rel=gap, source=src,
                           status=status)
            rows.append(row)
    summary = {s: sum(r["status"] == s for r in rows)
               for s in ("cross_checked", "contradicted", "single_family")}
    res = {"summary": summary, "rows": rows}
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    res = run()
    print(res["summary"])
    for r in res["rows"]:
        if r["status"] != "single_family":
            print(f"{r['problem']:<17}{r['qoi']:<24}{r['status']:<14} gap {100 * r['gap_rel']:.3f}% "
                  f"vs U {100 * r['U_rel']:.3f}% + {100 * r['U_ind_rel']:.3f}%  ({r['source']})")
