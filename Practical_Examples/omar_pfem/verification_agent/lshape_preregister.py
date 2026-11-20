"""Write lshape_preregistration.json BEFORE any L-shape solve (same protocol as
cook_preregister.py and plate_hole_preregister.py): spec, rule predictions
(rules unchanged), theory and the developer's expectations for the two POST
HOC singular-order options (VERIF_POSTHOC_SINGULAR=1: degree-k elements with
lam < k; pre-asymptotic label), which this problem tests for the first time
on unseen data. Refuses to run once lshape_ladder.json exists.

    python -m omar_pfem.verification_agent.lshape_preregister            (before solving)
    python -m omar_pfem.verification_agent.lshape_preregister --compare  (after solving)
"""
import datetime
import json
import os

from omar_pfem.verification_agent import trees
from omar_pfem.verification_agent.agent import DEFAULT_SPECS
from omar_pfem.verification_agent.lshape import CACHE, corner_exponent
from omar_pfem.verification_agent.orchestrate import LShapeAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lshape_preregistration.json")


def main():
    if os.path.exists(CACHE):
        raise SystemExit("lshape_ladder.json exists: not a pre-registration any more")
    spec = dict(DEFAULT_SPECS["l_shape"])
    spec["qois"] = list(LShapeAdapter.qois)
    lam = corner_exponent()
    rules = trees.apply_rules(spec)
    prereg = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "first unseen test of the post hoc singular-order options (degree-k "
                   "elements, pre-asymptotic label), proposed after the Cook and plate-with-"
                   "hole results",
        "spec": spec, "adapter_description": LShapeAdapter.description,
        "corner_exponent_lam": lam,
        "theory": {"energy_order": 2 * lam, "displacement_and_region_order_at_least": 2 * lam,
                   "corner_peak_slope": lam - 1, "singular_order_tol": 0.15,
                   "note": "Q2, lam = 0.5445 < 2: the re-entrant corner limits every rate to "
                           "2 lam = 1.089 (pollution); the clamped/free corners (lam 0.711) "
                           "are weaker"},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"], "role": r["role"]}
                             for q, r in rules.items()},
        "stated_in_advance": {
            "without the post hoc option": "every singular_order check is INCONCLUSIVE "
                                           "(theory implemented for degree-1 elements only)",
            "pre-asymptotic label": "NOT expected to trigger: the re-entrant corner is the "
                                    "strongest singularity of the classic cases and is loaded "
                                    "directly (the arm bends about it), so its coefficient is "
                                    "large and the asymptotic order should show at moderate h. "
                                    "If it does trigger, that counts against this expectation.",
        },
        "developer_expectations": {
            "strain_energy": "singular_order PASS (order 2 lam = %.3f within 0.15); "
                             "convergence PASS with rate" % (2 * lam),
            "u_top_left, v_right_top": "singular_order PASS (order >= %.3f); convergence PASS "
                                       "with rate" % (2 * lam - 0.15),
            "far_gk_avg_syy": "singular_order PASS (order >= %.3f, polluted by the corner); "
                              "convergence PASS with rate" % (2 * lam - 0.15),
            "corner_gk_avg_von_mises": "convergence PASS; singular_order PASS (order >= %.3f). "
                                       "LOW confidence: von Mises is not linear in the stress "
                                       "and the weight centred on the corner is not smooth in "
                                       "the stress direction, so the order may lie between "
                                       "lam and 2 lam" % (2 * lam - 0.15),
            "corner_disc_avg_von_mises": "convergence PASS (band or rate); singular_order not "
                                         "predicted (sharp disc: Gauss points enter and leave)",
            "peak_von_mises": "convergence FAIL (grows); log-log slope lam - 1 = %.3f within "
                              "0.15 (singular_order PASS); located within 0.1 of (1, 1) on the "
                              "finest three levels" % (lam - 1),
        },
        "rules_changed_for_this_problem": False,
    }
    json.dump(prereg, open(OUT, "w"), indent=1)
    print(json.dumps(prereg["rule_predictions"], indent=1))
    print(prereg["theory"])


def compare():
    from omar_pfem.verification_agent.verify_report import rules
    pre = json.load(open(OUT))
    out = {}
    for mode in ("default", "posthoc"):
        if mode == "posthoc":
            os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        else:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        R = rules.__wrapped__("l_shape")      # not the lru_cache: the mode differs
        rows, hits, n_pred = {}, 0, 0
        for q, pr in pre["rule_predictions"].items():
            ov, ev = R["combine"][q][:2]
            sg = R["v"][q].get("singular_order", {})
            row = {"verdict": ov, "evidence": ev, "rule_prediction": pr["predicted"],
                   "singular_order": {k: sg.get(k) for k in ("verdict", "observed", "expected",
                                                             "label")},
                   "convergence": R["v"][q].get("qoi_convergence_multilevel", {}).get(
                       "evidence")}
            if pr["predicted"] is not None:
                n_pred += 1
                row["rule_hit"] = pr["predicted"] == ov
                hits += row["rule_hit"]
            rows[q] = row
        out[mode] = {"rule_hits": hits, "rule_predictions": n_pred, "qois": rows}
    os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    ad = R["adapter"]
    P = out["posthoc"]["qois"]
    sing = lambda q: P[q]["singular_order"]["verdict"] == "PASS"
    near = lambda p: abs(p[0] - 1) < 0.1 and abs(p[1] - 1) < 0.1
    expect = {
        "default: every singular_order INCONCLUSIVE": all(
            r["singular_order"]["verdict"] == "INCONCLUSIVE"
            for r in out["default"]["qois"].values()),
        "pre-asymptotic label not triggered": not any(
            r["singular_order"].get("label") and "pre-asymptotic" in r["singular_order"]["label"]
            for r in P.values()),
        "strain energy singular_order PASS, convergence rate":
            sing("strain_energy") and P["strain_energy"]["convergence"] == "rate",
        "displacements singular_order PASS, convergence rate": all(
            sing(q) and P[q]["convergence"] == "rate" for q in ("u_top_left", "v_right_top")),
        "far Gaussian average singular_order PASS, convergence rate":
            sing("far_gk_avg_syy") and P["far_gk_avg_syy"]["convergence"] == "rate",
        "corner Gaussian average singular_order PASS (low confidence)":
            sing("corner_gk_avg_von_mises"),
        "corner Gaussian average convergence PASS":
            P["corner_gk_avg_von_mises"]["convergence"] in ("rate", "band"),
        "corner disc average convergence PASS (band or rate)":
            P["corner_disc_avg_von_mises"]["convergence"] in ("rate", "band"),
        "peak convergence FAIL": P["peak_von_mises"]["convergence"] not in ("rate", "band"),
        "peak slope = lam - 1 within 0.15": sing("peak_von_mises"),
        "peak located at the re-entrant corner": all(
            near(r["peak_location"]) for r in ad.rows[-3:]),
    }
    res = {"n_levels": len(ad.hs), "levels_n": [r["n"] for r in ad.rows],
           "lam": pre["corner_exponent_lam"], "modes": out,
           "developer_expectations_met": expect}
    path = os.path.join(HERE, f"lshape_result_{len(ad.hs)}levels.json")
    json.dump(res, open(path, "w"), indent=1)
    for mode, o in out.items():
        print(f"{mode}: rule predictions {o['rule_hits']}/{o['rule_predictions']}")
    print(f"developer expectations {sum(expect.values())}/{len(expect)}")
    for k, v in expect.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    for q, r in P.items():
        print(f"  {q:<26} {r['verdict']}/{r['evidence']:<5} conv {r['convergence']} "
              f"singular {r['singular_order']}")


if __name__ == "__main__":
    import sys
    compare() if "--compare" in sys.argv else main()
