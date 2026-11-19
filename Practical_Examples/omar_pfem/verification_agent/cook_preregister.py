"""Write cook_preregistration.json BEFORE any Cook solve: the fixed spec, the
rule predictions (trees.apply_rules, unchanged rules) and the theoretical
orders, plus the developer's own expectations where the rules make none.
Refuses to run once cook_ladder.json exists.

    python -m omar_pfem.verification_agent.cook_preregister            (before solving)
    python -m omar_pfem.verification_agent.cook_preregister --compare  (after solving)

--compare scores the unchanged rules and the developer's stated expectations
against the verdicts on the levels in cook_ladder.json and writes
cook_result_<n>levels.json.
"""
import datetime
import json
import os

from omar_pfem.verification_agent import trees
from omar_pfem.verification_agent.agent import DEFAULT_SPECS
from omar_pfem.verification_agent.cook_ladder import CACHE, cook_corner_exponent
from omar_pfem.verification_agent.orchestrate import CookAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cook_preregistration.json")


def main():
    if os.path.exists(CACHE):
        raise SystemExit("cook_ladder.json exists: results already computed, not a "
                         "pre-registration any more")
    spec = dict(DEFAULT_SPECS["cook_membrane"])
    spec["qois"] = list(CookAdapter.qois)
    lam = cook_corner_exponent()
    rules = trees.apply_rules(spec)
    prereg = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spec": spec, "adapter_description": CookAdapter.description,
        "corner_exponent_lam": lam,
        "theory": {"energy_order": 2 * lam, "displacement_order_at_least": 2 * lam,
                   "peak_slope": lam - 1, "singular_order_tol": 0.15},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"], "role": r["role"]}
                             for q, r in rules.items()},
        "developer_expectations": {
            "u_tip, v_tip, max_disp_mag, strain_energy":
                "PASS with rate; strain-energy order 2 lam = %.2f within 0.15; displacement "
                "orders >= 2 lam - 0.15" % (2 * lam),
            "interior_avg_PK1_frob":
                "PASS, possibly only on the band (sharp disc: Gauss points enter/leave the "
                "region between levels, as in B1/B2/B3)",
            "interior_gk_PK1_frob": "PASS with rate (smooth weight)",
            "peak_PK1_frob": "convergence FAIL (grows); singular_order PASS with log-log "
                             "slope lam - 1 = %.3f within 0.15; strategy spread large; located "
                             "at the 108.4-degree corner (0,44)" % (lam - 1),
        },
        "rules_changed_for_this_problem": False,
    }
    json.dump(prereg, open(OUT, "w"), indent=1)
    print(json.dumps(prereg["rule_predictions"], indent=1))
    print(prereg["theory"])


def compare():
    from omar_pfem.verification_agent.verify_report import rules
    pre = json.load(open(OUT))
    R = rules("cook_membrane")
    n_levels = len(R["adapter"].hs)
    rows, hits, n_pred = {}, 0, 0
    for q, pr in pre["rule_predictions"].items():
        ov, ev = R["combine"][q][:2]
        sg = R["v"][q].get("singular_order", {})
        row = {"verdict": ov, "evidence": ev, "rule_prediction": pr["predicted"],
               "singular_order": {k: sg.get(k) for k in ("verdict", "observed", "expected")}}
        if pr["predicted"] is not None:
            n_pred += 1
            row["rule_hit"] = pr["predicted"] == ov
            hits += row["rule_hit"]
        rows[q] = row
    sing = lambda q: rows[q]["singular_order"]["verdict"] == "PASS"
    conv = lambda q: R["v"][q]["qoi_convergence_multilevel"].get("evidence")
    expect = {   # the developer expectations of the pre-registration, as checks
        "displacements PASS with rate, order >= 2 lam - 0.15":
            all(rows[q]["verdict"] == "PASS" and rows[q]["evidence"] == "rate" and sing(q)
                for q in ("u_tip", "v_tip", "max_disp_mag")),
        "strain energy PASS with rate, order = 2 lam within 0.15":
            rows["strain_energy"]["verdict"] == "PASS" and sing("strain_energy"),
        "sharp disc average PASS (band or rate)": rows["interior_avg_PK1_frob"]["verdict"] == "PASS",
        "Gaussian average PASS with rate": rows["interior_gk_PK1_frob"]["verdict"] == "PASS"
        and rows["interior_gk_PK1_frob"]["evidence"] == "rate",
        "peak convergence FAIL": conv("peak_PK1_frob") != "rate"
        and rows["peak_PK1_frob"]["verdict"] == "FAIL",
        "peak slope = lam - 1 within 0.15": sing("peak_PK1_frob"),
        "peak at the 108.4-degree corner (0,44)": all(
            abs(r["peak_location"][0]) < 3 and abs(r["peak_location"][1] - 44) < 3
            for r in R["adapter"].rows[-5:]),
    }
    out = {"n_levels": n_levels, "levels_N": [r["N"] for r in R["adapter"].rows],
           "rule_hits": hits, "rule_predictions": n_pred, "qois": rows,
           "developer_expectations_met": expect}
    path = os.path.join(HERE, f"cook_result_{n_levels}levels.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{n_levels} levels: rule predictions {hits}/{n_pred}; developer expectations "
          f"{sum(expect.values())}/{len(expect)}")
    for k, v in expect.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    for q, r in rows.items():
        print(f"  {q:<24} {r['verdict']}/{r['evidence']:<5} singular {r['singular_order']}")


if __name__ == "__main__":
    import sys
    compare() if "--compare" in sys.argv else main()
