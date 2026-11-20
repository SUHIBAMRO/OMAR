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


if __name__ == "__main__":
    main()
