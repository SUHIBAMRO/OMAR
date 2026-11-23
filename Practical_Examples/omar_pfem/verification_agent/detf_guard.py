"""Admissibility guard for stored solves (2026-11-23, POST HOC: added after the
inverted g = 2 strategy of cylinder3d was found; no verdict rule is changed).

Finding: a Newton solve with too few load increments can converge to an inverted state
(min det F <= 0) without any error being raised (cylinder3d, g = 2.0, 10 increments:
min det F = -1.03, errors 13-27%). The full 3x3x3 rule does not prevent it; 40
increments do (graded_q20_study.json). The agent's rules had no check for this.

This guard
  1. scans every stored solve of every adapter (family levels and strategy meshes) and
     lists those with min det F <= 0 as INADMISSIBLE,
  2. states for each whether any verdict check consumed it (the strategy checks are the
     only consumers of strategy rows),
  3. records the remedy and, where the re-solve exists (graded_q20_study.json), its result.
The rule it proposes, for future runs (not applied to the frozen pre-registered scoring):
a solve with min det F <= 0 is re-solved with four times the increments; if it stays
inverted, it is excluded and every check that needs it is INCONCLUSIVE.

    python -m omar_pfem.verification_agent.detf_guard
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "detf_guard.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape", "edge_crack", "thick_cylinder_3d", "clamped_block_3d")
STRATEGY_CHECKS = ("strategy_robustness", "anchored_robustness")


def _det(r):
    for k in ("min_detF", "posthoc_min_detF"):
        if r.get(k) is not None:
            return r[k]
    return None


def scan():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent.verify_report import rules
    out, recorded, missing = [], 0, []
    for geo in PROBLEMS:
        R = rules.__wrapped__(geo)
        ad = R["adapter"]
        fam = getattr(ad, "rows", None) or []
        strat = getattr(ad, "strategy_rows", None) or []
        used = sorted(q for q, v in R["v"].items() if any(c in v for c in STRATEGY_CHECKS))
        n_with = 0
        for role, rows in (("family", fam), ("strategy", strat)):
            for r in rows:
                d = _det(r)
                if d is None:
                    continue
                n_with += 1
                if d <= 0:
                    out.append({"problem": geo, "role": role, "min_detF": d,
                                "grading": r.get("grading"), "level": r.get("k", r.get("n")),
                                "consumed_by_a_verdict": role == "family" or bool(used),
                                "strategy_checks_on": used})
        recorded += n_with
        if not n_with:
            missing.append(geo)
    return out, recorded, missing


def main():
    bad, recorded, missing = scan()
    gq = json.load(open(os.path.join(HERE, "graded_q20_study.json")))
    resolve = [r for r in gq if r["grading"] == 2.0]
    res = {"posthoc": True, "solves_with_min_detF_recorded": recorded,
           "problems_without_min_detF_records": missing,
           "inadmissible": bad,
           "remedy_tested": {
               "case": "thick_cylinder_3d, k = 4, g = 2.0",
               "rows": [{k: r.get(k) for k in ("quadrature", "nsteps", "min_detF", "errors")}
                        for r in resolve],
               "conclusion": "10 increments: inverted with both rules (min det F -1.03 reduced, "
                             "-0.74 full); 40 increments: min det F 1.004, every global "
                             "quantity within 1.2e-5 of the exact value (reduced 2.6e-7)"},
           "proposed_rule_for_future_runs": "min det F <= 0 -> re-solve with 4x increments; "
                                            "still <= 0 -> exclude, dependent checks "
                                            "INCONCLUSIVE (not applied to the frozen scoring)"}
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "remedy_tested"}, indent=1))
