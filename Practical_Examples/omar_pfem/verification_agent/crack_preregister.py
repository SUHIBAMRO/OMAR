"""Write crack_preregistration.json BEFORE any crack solve (same protocol as
cook_preregister.py and plate_hole_preregister.py): spec, rule predictions
(rules unchanged), theory and the developer's expectations for the two POST
HOC singular-order options (VERIF_POSTHOC_SINGULAR=1: degree-k elements with
lam < k; pre-asymptotic label), which this problem tests for the first time
on unseen data. Refuses to run once crack_ladder.json exists.

    python -m omar_pfem.verification_agent.crack_preregister            (before solving)
    python -m omar_pfem.verification_agent.crack_preregister --compare  (after solving)
"""
import datetime
import json
import os

from omar_pfem.verification_agent import trees
from omar_pfem.verification_agent.agent import DEFAULT_SPECS
from omar_pfem.verification_agent.crack import CACHE, corner_exponent
from omar_pfem.verification_agent.orchestrate import CrackAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "crack_preregistration.json")


def main():
    if os.path.exists(CACHE):
        raise SystemExit("crack_ladder.json exists: not a pre-registration any more")
    spec = dict(DEFAULT_SPECS["edge_crack"])
    spec["qois"] = list(CrackAdapter.qois)
    lam = corner_exponent()
    rules = trees.apply_rules(spec)
    prereg = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "second unseen test of the current rules with the proposed singular-order "
                   "extension (VERIF_POSTHOC_SINGULAR=1), in a new singularity class (crack); "
                   "written after the L-shape result, whose lesson (a displacement may approach "
                   "its rate from below) is used below",
        "spec": spec, "adapter_description": CrackAdapter.description,
        "corner_exponent_lam": lam,
        "theory": {"energy_order": 2 * lam, "displacement_and_region_order_at_least": 2 * lam,
                   "corner_peak_slope": lam - 1, "singular_order_tol": 0.15,
                   "note": "Q2, lam = 0.5 < 2: the crack tip limits every rate to 2 lam = 1.0 "
                           "(pollution); the clamped/free corners (lam 0.711) are weaker"},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"], "role": r["role"]}
                             for q, r in rules.items()},
        "stated_in_advance": {
            "without the post hoc option": "every singular_order check is INCONCLUSIVE "
                                           "(theory implemented for degree-1 elements only)",
            "pre-asymptotic label": "NOT expected to trigger: the tip is loaded directly in "
                                    "mode I (strong singular coefficient)",
        },
        "developer_expectations": {
            "strain_energy": "singular_order PASS (order 2 lam = 1.0 within 0.15); convergence "
                             "PASS with rate",
            "u_cod_mouth": "singular_order PASS (order >= 0.85); convergence PASS with rate",
            "v_top_right": "singular_order PASS (order >= 0.85); LOW confidence: a displacement at "
                           "a domain corner approached its rate from below on the L-shape",
            "far_gk_avg_syy": "singular_order PASS (order >= 0.85); convergence PASS with rate",
            "tip_gk_avg_von_mises": "convergence PASS; singular_order PASS (order >= 0.85), LOW "
                                    "confidence (nonlinear in the stress, weight centred on the "
                                    "singular point)",
            "tip_disc_avg_von_mises": "convergence PASS (band or rate); singular_order not "
                                      "predicted",
            "peak_von_mises": "convergence FAIL (grows); log-log slope lam - 1 = -0.5 within 0.15 "
                              "(singular_order PASS); located within 0.1 of the tip (1, 0) on the "
                              "finest three levels",
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
        R = rules.__wrapped__("edge_crack")      # not the lru_cache: the mode differs
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
    near = lambda p: abs(p[0] - 1) < 0.1 and abs(p[1]) < 0.1
    expect = {
        "default: every singular_order INCONCLUSIVE": all(
            r["singular_order"]["verdict"] == "INCONCLUSIVE"
            for r in out["default"]["qois"].values()),
        "pre-asymptotic label not triggered": not any(
            r["singular_order"].get("label") and "pre-asymptotic" in r["singular_order"]["label"]
            for r in P.values()),
        "strain energy singular_order PASS, convergence rate":
            sing("strain_energy") and P["strain_energy"]["convergence"] == "rate",
        "crack opening singular_order PASS, convergence rate":
            sing("u_cod_mouth") and P["u_cod_mouth"]["convergence"] == "rate",
        "corner displacement singular_order PASS (low confidence)": sing("v_top_right"),
        "far Gaussian average singular_order PASS, convergence rate":
            sing("far_gk_avg_syy") and P["far_gk_avg_syy"]["convergence"] == "rate",
        "tip Gaussian average singular_order PASS (low confidence)":
            sing("tip_gk_avg_von_mises"),
        "tip Gaussian average convergence PASS":
            P["tip_gk_avg_von_mises"]["convergence"] in ("rate", "band"),
        "tip disc average convergence PASS (band or rate)":
            P["tip_disc_avg_von_mises"]["convergence"] in ("rate", "band"),
        "peak convergence FAIL": P["peak_von_mises"]["convergence"] not in ("rate", "band"),
        "peak slope = lam - 1 within 0.15": sing("peak_von_mises"),
        "peak located at the crack tip": all(near(r["peak_location"]) for r in ad.rows[-3:]),
    }
    res = {"n_levels": len(ad.hs), "levels_n": [r["n"] for r in ad.rows],
           "lam": pre["corner_exponent_lam"], "modes": out,
           "developer_expectations_met": expect}
    path = os.path.join(HERE, f"crack_result_{len(ad.hs)}levels.json")
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
