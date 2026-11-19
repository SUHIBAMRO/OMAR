"""Write plate_hole_preregistration.json BEFORE any plate-with-hole solve: the
fixed spec, the rule predictions (trees.apply_rules, rules unchanged), the
theoretical orders and the developer's own expectations. Refuses to run once
plate_hole_ladder.json exists. Same protocol as cook_preregister.py.

    python -m omar_pfem.verification_agent.plate_hole_preregister            (before solving)
    python -m omar_pfem.verification_agent.plate_hole_preregister --compare  (after solving)
"""
import datetime
import json
import os

from omar_pfem.verification_agent import trees
from omar_pfem.verification_agent.agent import DEFAULT_SPECS
from omar_pfem.verification_agent.orchestrate import PlateHoleAdapter
from omar_pfem.verification_agent.plate_hole import CACHE, corner_exponent

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plate_hole_preregistration.json")


def main():
    if os.path.exists(CACHE):
        raise SystemExit("plate_hole_ladder.json exists: results already computed, not a "
                         "pre-registration any more")
    spec = dict(DEFAULT_SPECS["plate_hole"])
    spec["qois"] = list(PlateHoleAdapter.qois)
    lam = corner_exponent()
    rules = trees.apply_rules(spec)
    prereg = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "Wang et al., OpenPINN/PINO, Sec. 5.3 (attachment of Professor Rabczuk, "
                  "19 Nov 2026); their reference is one FEM mesh, h = 0.025",
        "spec": spec, "adapter_description": PlateHoleAdapter.description,
        "corner_exponent_lam": lam,
        "theory": {"energy_order": 2 * lam, "displacement_order_at_least": 2 * lam,
                   "corner_peak_slope": lam - 1, "singular_order_tol": 0.15,
                   "note": "Q2 elements, lam < 2: the corner limits the global rate to 2 lam "
                           "(pollution); quantities at the hole are smooth locally but "
                           "polluted by the corners, so their order is at most about 2 lam"},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"], "role": r["role"]}
                             for q, r in rules.items()},
        "known_rule_limitation_stated_in_advance": (
            "The rules classify by QoI class and problem regularity only. In a singular "
            "problem every pointwise_peak gets the singular-order check (slope lam - 1). "
            "hole_peak_sigma_xx is a peak at a smooth, finite-radius concentration far from "
            "the corners: it is expected to CONVERGE, so the singular-order gate is expected "
            "to FAIL for it although nothing is wrong. The rules do not know where a peak "
            "is; the diagnosis engine (location, h-trend) should separate the two peaks."),
        "developer_expectations": {
            "u_right_mid, v_top_right, max_disp_mag":
                "PASS with rate; fitted order >= 2 lam - 0.15 = %.2f" % (2 * lam - 0.15),
            "strain_energy": "PASS with rate; order 2 lam = %.2f within 0.15" % (2 * lam),
            "hole_avg_sigma_xx": "PASS (band or rate): sharp disc, Gauss points enter and "
                                 "leave the region between levels",
            "hole_gk_avg_sigma_xx": "PASS with rate (smooth weight)",
            "hole_peak_sigma_xx": "convergence PASS (bounded, converging; order about 1 from "
                                  "the Gauss-point offset from the hole surface, at most "
                                  "about 2 lam); singular_order FAIL (not a corner peak); "
                                  "located at the hole top (1, 1.25) within 0.05",
            "peak_von_mises": "convergence FAIL (grows); log-log slope lam - 1 = %.3f within "
                              "0.15; located at a clamped/free corner (0,0) or (0,2)"
                              % (lam - 1),
        },
        "rules_changed_for_this_problem": False,
    }
    json.dump(prereg, open(OUT, "w"), indent=1)
    print(json.dumps(prereg["rule_predictions"], indent=1))
    print(prereg["theory"])


def compare():
    from omar_pfem.verification_agent.verify_report import rules
    pre = json.load(open(OUT))
    R = rules("plate_hole")
    ad = R["adapter"]
    rows, hits, n_pred = {}, 0, 0
    for q, pr in pre["rule_predictions"].items():
        ov, ev = R["combine"][q][:2]
        sg = R["v"][q].get("singular_order", {})
        row = {"verdict": ov, "evidence": ev, "rule_prediction": pr["predicted"],
               "singular_order": {k: sg.get(k) for k in ("verdict", "observed", "expected")},
               "convergence": R["v"][q].get("qoi_convergence_multilevel", {}).get("evidence")}
        if pr["predicted"] is not None:
            n_pred += 1
            row["rule_hit"] = pr["predicted"] == ov
            hits += row["rule_hit"]
        rows[q] = row
    sing = lambda q: rows[q]["singular_order"]["verdict"] == "PASS"
    lam = pre["corner_exponent_lam"]
    fine = ad.rows[-3:]
    corner = lambda p: abs(p[0]) < 0.1 and min(abs(p[1]), abs(p[1] - 2)) < 0.1
    expect = {
        "displacements PASS with rate, order >= 2 lam - 0.15":
            all(rows[q]["verdict"] == "PASS" and rows[q]["evidence"] == "rate" and sing(q)
                for q in ("u_right_mid", "v_top_right", "max_disp_mag")),
        "strain energy PASS with rate, order = 2 lam within 0.15":
            rows["strain_energy"]["verdict"] == "PASS" and sing("strain_energy"),
        "sharp hole average PASS (band or rate)": rows["hole_avg_sigma_xx"]["verdict"] == "PASS",
        "Gaussian hole average PASS with rate": rows["hole_gk_avg_sigma_xx"]["verdict"] == "PASS"
        and rows["hole_gk_avg_sigma_xx"]["evidence"] == "rate",
        "hole peak converges (convergence gate rate)":
            rows["hole_peak_sigma_xx"]["convergence"] == "rate",
        "hole peak singular_order FAIL": rows["hole_peak_sigma_xx"]["singular_order"][
            "verdict"] == "FAIL",
        "von Mises peak convergence FAIL": rows["peak_von_mises"]["convergence"] != "rate"
        and rows["peak_von_mises"]["verdict"] == "FAIL",
        "von Mises peak slope = lam - 1 within 0.15": sing("peak_von_mises"),
        "von Mises peak at a clamped/free corner": all(corner(r["peak_location"]) for r in fine),
    }
    out = {"n_levels": len(ad.hs), "levels_m": [r["m"] for r in ad.rows], "lam": lam,
           "rule_hits": hits, "rule_predictions": n_pred, "qois": rows,
           "developer_expectations_met": expect}
    path = os.path.join(HERE, f"plate_hole_result_{len(ad.hs)}levels.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(ad.hs)} levels: rule predictions {hits}/{n_pred}; developer expectations "
          f"{sum(expect.values())}/{len(expect)}")
    for k, v in expect.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    for q, r in rows.items():
        print(f"  {q:<22} {r['verdict']}/{r['evidence']:<5} conv {r['convergence']} "
              f"singular {r['singular_order']}")


if __name__ == "__main__":
    import sys
    compare() if "--compare" in sys.argv else main()
