"""Write cook_preregistration.json BEFORE any Cook solve: the fixed spec, the
rule predictions (trees.apply_rules, unchanged rules) and the theoretical
orders, plus the developer's own expectations where the rules make none.
Refuses to run once cook_ladder.json exists.

    python -m omar_pfem.verification_agent.cook_preregister
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


if __name__ == "__main__":
    main()
