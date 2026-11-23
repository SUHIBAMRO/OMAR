"""Normalisation study (2026-11-23; an ANALYSIS, no criterion is changed).

Region and peak quantities are judged relative to a fixed stress scale (the
finest-level region average of the stress norm, report Section 4), so that
near-zero components are not amplified. On B3 that scale (3.10 for the Gaussian
region) is inflated by the reduced-integration pressure oscillation, and the
normal-component averages (0.43-0.54) are 6-7 times smaller. This script asks,
for every quantity the agent rate-certifies (post hoc singular-order option on, as
in the final comparison): what are its uncertainty and its tolerance margin on the
quantity's OWN value, and would it still meet the same tolerance there?

Quantities whose own value is below 10% of the scale are listed apart: for them the
own-value basis is the one the scale was introduced to avoid.

    python -m omar_pfem.verification_agent.normalisation_study
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "normalisation_study.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape", "edge_crack", "thick_cylinder_3d")


def run():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent.verify_report import rules
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    rows = []
    try:
        for geo in PROBLEMS:
            R = rules.__wrapped__(geo)
            ad = R["adapter"]
            for q, comb in R["combine"].items():
                if comb[0] != "PASS" or comb[1] != "rate":
                    continue
                ml = R["v"][q].get("qoi_convergence_multilevel", {})
                if ml.get("evidence") == "rate":
                    lim, U, tol = ml["fit_limit"], ml["uncertainty_fine"], ml["gci_tol"]
                else:
                    c3 = R["v"][q]["qoi_convergence"]
                    lim, U, tol = c3["richardson_extrapolate"], c3["gci_fine"], c3["gci_tol"]
                _, vals, scale = ad.series(q)
                sc = scale if scale else abs(vals[-1])
                own = abs(lim)
                ratio = sc / own if own else float("inf")
                rows.append({"problem": geo, "qoi": q, "scale": sc, "value": lim,
                             "scale_over_value": ratio, "U_of_scale": U,
                             "U_of_value": U * ratio, "tol": tol,
                             "meets_tol_on_own_value": U * ratio <= tol,
                             "near_zero": own < 0.1 * sc})
    finally:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    scaled = [r for r in rows if r["scale_over_value"] > 1.01]
    res = {"rate_certified": len(rows),
           "judged_on_a_scale_larger_than_the_value": len(scaled),
           "fail_tol_on_own_value": [f"{r['problem']}:{r['qoi']}" for r in rows
                                     if not r["meets_tol_on_own_value"] and not r["near_zero"]],
           "near_zero_components": [f"{r['problem']}:{r['qoi']}" for r in rows if r["near_zero"]],
           "max_scale_over_value": max((r["scale_over_value"] for r in rows if not r["near_zero"]),
                                       default=None),
           "rows": rows}
    res["crosscheck_on_own_value"] = crosscheck_own_value(rows)
    json.dump(res, open(OUT, "w"), indent=1)
    return res


def crosscheck_own_value(rows):
    """Cross-check v2 repeated with every uncertainty and tolerance taken on the quantity's
    own value instead of the stress scale (B3 Gaussian averages: the only quantities where
    the two bases differ). An analysis of sensitivity to the basis, not a new rule."""
    from omar_pfem.verification_agent.crosscheck import (assess_independent, decide,
                                                          independent_series)
    out = {}
    for r in rows:
        if r["scale_over_value"] <= 1.01:
            continue
        own = abs(r["value"])
        U_own = r["U_of_value"]
        ml = {"band_tol": 0.01, "gci_tol": r["tol"]}
        st = []
        for src, hs, vs in independent_series(r["problem"], r["qoi"]):
            if hs == "exact":
                continue
            a = assess_independent(hs, vs, ml, own)
            if a.get("evidence") == "rate" and a.get("verdict") == "PASS":
                s, gap = decide(r["value"], U_own, a["fit_limit"], a["uncertainty_fine"] * own, own)
                st.append({"source": src, "status": s, "gap_of_value": gap,
                           "U_ind_of_value": a["uncertainty_fine"]})
            else:
                st.append({"source": src, "status": "independent_unconverged",
                           "reason": a.get("reason")})
        statuses = [x["status"] for x in st]
        out[f"{r['problem']}:{r['qoi']}"] = {
            "status": ("cross_checked" if "cross_checked" in statuses and "contradicted"
                       not in statuses else "contradicted" if "contradicted" in statuses
                       else "independent_unconverged" if statuses else "single_family"),
            "sources": st}
    return out


if __name__ == "__main__":
    r = run()
    print({k: v for k, v in r.items() if k != "rows"})
    for x in r["rows"]:
        if x["scale_over_value"] > 1.01:
            print(f"{x['problem']:<17}{x['qoi']:<24} scale/value {x['scale_over_value']:6.2f} "
                  f"U {100 * x['U_of_scale']:.3f}% of scale = {100 * x['U_of_value']:.3f}% of "
                  f"value (tol {100 * x['tol']:.0f}%)")
