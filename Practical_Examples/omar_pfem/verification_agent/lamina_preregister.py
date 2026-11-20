"""Write lamina_preregistration.json BEFORE any lamina solve (same protocol as
plate_hole_preregister.py). Refuses to run once lamina_ladder.json exists.

    python -m omar_pfem.verification_agent.lamina_preregister            (before solving)
    python -m omar_pfem.verification_agent.lamina_preregister --compare  (after solving)
"""
import datetime
import json
import os

from omar_pfem.verification_agent import trees
from omar_pfem.verification_agent.agent import DEFAULT_SPECS
from omar_pfem.verification_agent.lamina import CACHE
from omar_pfem.verification_agent.orchestrate import LaminaAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lamina_preregistration.json")
# CFM paper, Table 6: width of the first-order constitutive range, % of the value
CFM_WIDTH_PCT = {"compliance": (6.19, 3.18), "hole_gamma12_avg": (7.12, 5.61),
                 "hole_e22_avg": (146.0, 107.0), "far_gamma12_avg": (5.27, 4.71)}


def main():
    if os.path.exists(CACHE):
        raise SystemExit("lamina_ladder.json exists: not a pre-registration any more")
    spec = dict(DEFAULT_SPECS["open_hole_lamina"])
    spec["qois"] = list(LaminaAdapter.qois)
    rules = trees.apply_rules(spec)
    prereg = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "Lone, Prabhakaran, Rabczuk, Zhuang, Constitutive-fibre mechanics, "
                  "arXiv:2611.22187, Sec. 9.3 (open-hole lamina); their result is for one mesh "
                  "(4,096 quadratic triangles, 64 x 32 divisions)",
        "spec": spec, "adapter_description": LaminaAdapter.description,
        "qoi_classes": {q: trees.classify_qoi(q) for q in LaminaAdapter.qois},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"]} for q, r in rules.items()},
        "known_limitations_stated_in_advance": [
            "the QoI vocabulary does not know 'compliance' (class unknown): it is a global, "
            "energy-type quantity (f.u/P^2), which the rules cannot see",
            "our zone definitions are not the paper's (the paper does not define them); the "
            "comparison with the CFM widths is for quantities of the same kind, not identical "
            "ones",
            "linear reference law only (the paper's widths are at 0.6 P_D with the nonlinear "
            "fitted law); relative discretization errors of a linear problem are used as the "
            "estimate"],
        "cfm_constitutive_width_percent_without_with_convexity": CFM_WIDTH_PCT,
        "developer_expectations": {
            "compliance, u_tip": "PASS with rate; observed order 3-5 (smooth, Q2: energy-type "
                                 "order 2k = 4)",
            "hole/far sharp averages": "PASS, band or rate",
            "hole Gaussian averages": "PASS with rate",
            "peak_e22": "PASS (converging, bounded; order about 1 from the Gauss-point offset), "
                        "located at the hole edge",
            "CFM resolution (m = 4) against the CFM widths": "discretization uncertainty of "
                "compliance and of the hole/far-field shear strain at m = 4 below 1/3 of the "
                "CFM width (margin >= 3), i.e. the paper's mesh is adequate for its ranges",
        },
        "rules_changed_for_this_problem": False,
    }
    json.dump(prereg, open(OUT, "w"), indent=1)
    print(json.dumps({q: v["predicted"] for q, v in prereg["rule_predictions"].items()}))
    print(prereg["qoi_classes"])




def compare():
    """Score the unchanged rules and the stated expectations; compare the
    discretization error at the paper's resolution (m = 4) with the CFM widths."""
    import numpy as np
    from omar_pfem.verification_agent.verify_report import rules
    pre = json.load(open(OUT))
    R = rules("open_hole_lamina")
    ad = R["adapter"]
    rows, hits, n_pred = {}, 0, 0
    for q, pr in pre["rule_predictions"].items():
        ov, ev = R["combine"][q][:2]
        g3 = R["v"][q].get("qoi_convergence", {})
        g5 = R["v"][q].get("qoi_convergence_multilevel", {})
        row = {"verdict": ov, "evidence": ev, "rule_prediction": pr["predicted"],
               "order_3level": g3.get("observed_order"), "order_5level": g5.get("fit_order"),
               "U_fine": g5.get("uncertainty_fine")}
        if pr["predicted"] is not None:
            n_pred += 1
            row["rule_hit"] = pr["predicted"] == ov
            hits += row["rule_hit"]
        rows[q] = row
    byq = {r["m"]: r for r in ad.rows}
    fine = byq[max(byq)]
    strat = {r["g"]: r for r in ad.strategy_rows}
    cfm = {}
    for q, (w_nc, w_c) in pre["cfm_constitutive_width_percent_without_with_convexity"].items():
        err4 = 100 * abs(byq[4][q] / fine[q] - 1)
        spread = 100 * (max(r[q] for r in strat.values()) - min(r[q] for r in strat.values())) \
            / abs(fine[q])
        num = max(err4, spread)
        cfm[q] = {"error_at_m4_vs_m%d_pct" % max(byq): err4, "strategy_spread_m8_pct": spread,
                  "cfm_width_pct": [w_nc, w_c],
                  "margin_half_width_over_numerical": (w_c / 2) / num if num > 0 else None}
    peak_at_hole = all(abs(np.hypot(*r["peak_e22_location"]) - 3.0) < 0.5 for r in ad.rows[-3:])
    expect = {
        "compliance and u_tip PASS with rate": all(rows[q]["verdict"] == "PASS" and
                                                   rows[q]["evidence"] == "rate"
                                                   for q in ("compliance", "u_tip")),
        "compliance/u_tip 5-level order in [3, 5]": all(
            rows[q]["order_5level"] is not None and 3 <= rows[q]["order_5level"] <= 5
            for q in ("compliance", "u_tip")),
        "sharp averages PASS (band or rate)": all(rows[q]["verdict"] == "PASS" for q in (
            "hole_gamma12_avg", "hole_e22_avg", "far_gamma12_avg")),
        "Gaussian averages PASS with rate": all(rows[q]["verdict"] == "PASS" and
                                                rows[q]["evidence"] == "rate"
                                                for q in ("hole_gk_gamma12_avg",
                                                          "hole_gk_e22_avg")),
        "peak PASS, at the hole edge": rows["peak_e22"]["verdict"] == "PASS" and peak_at_hole,
        "margin >= 3 at m = 4 for compliance and the shear strains": all(
            (cfm[q]["margin_half_width_over_numerical"] or 0) >= 3
            for q in ("compliance", "hole_gamma12_avg", "far_gamma12_avg")),
    }
    out = {"n_levels": len(ad.hs), "levels_m": [r["m"] for r in ad.rows], "rule_hits": hits,
           "rule_predictions": n_pred, "qois": rows, "cfm_comparison": cfm,
           "developer_expectations_met": expect}
    path = os.path.join(HERE, f"lamina_result_{len(ad.hs)}levels.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"rules {hits}/{n_pred}; expectations {sum(expect.values())}/{len(expect)}")
    for k, v in expect.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    for q, r in rows.items():
        print(f"  {q:<22} {r['verdict']}/{r['evidence']:<5} p3={r['order_3level']} "
              f"p5={r['order_5level']} U={r['U_fine']}")
    for q, c in cfm.items():
        print(f"  CFM {q:<18} {c}")


if __name__ == "__main__":
    import sys
    compare() if "--compare" in sys.argv else main()
