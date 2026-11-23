"""Autonomous follow-up (2026-11-23): act on a diagnosis instead of only naming it.

When the diagnosis supports D4 (pre-asymptotic: pairwise orders still drifting) for
a quantity of a problem whose level runner is known to the agent, the agent

  1. estimates the cost of the next level (doubled n) from its own stored solves:
     time t ~ dof^alpha fitted on the finest three levels, dof ~ n^2 (2D);
     memory ~ 3 kB per unknown (a measured upper bound for these scikit-fem solves);
  2. runs it only if it fits the budget (default 300 s, 8 GB), else reports why not;
  3. stores it in <problem>_followup.json (never in the pre-registered ladder);
  4. re-runs the problem's own plan (the same, frozen checks; post hoc singular-order
     option on, as in the comparison it continues) on the extended family and reports
     the old and new verdicts.

Safeguards: no tolerance, window or criterion changes; the original verdicts stay
on record; a new level can only add evidence. Predictions for the first use
(L-shape, v_right_top) were committed before the run (followup_preregistration.json).

    python -m omar_pfem.verification_agent.followup l_shape v_right_top
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNERS = {"l_shape": "lshape", "edge_crack": "crack"}
BYTES_PER_DOF = 3000.0


def _rows(geometry):
    mod = __import__(f"omar_pfem.verification_agent.{RUNNERS[geometry]}", fromlist=["CACHE"])
    rows = sorted((r for r in json.load(open(mod.CACHE)) if r["g"] == 1.0), key=lambda r: r["n"])
    fpath = os.path.join(HERE, f"{geometry}_followup.json")
    extra = json.load(open(fpath)) if os.path.exists(fpath) else []
    return mod, rows, extra, fpath


def plan_next_level(geometry, budget_s=300.0, budget_gb=8.0):
    mod, rows, extra, _ = _rows(geometry)
    fam = sorted(rows + extra, key=lambda r: r["n"])
    n_next = 2 * fam[-1]["n"]
    dof = np.array([r["n_dof"] for r in fam[-3:]], float)
    t = np.array([r["wall_clock_s"] for r in fam[-3:]], float)
    alpha = float(np.polyfit(np.log(dof), np.log(t), 1)[0])
    dof_next = fam[-1]["n_dof"] * 4.0
    t_next = float(t[-1] * (dof_next / dof[-1]) ** alpha)
    gb_next = dof_next * BYTES_PER_DOF / 1e9
    ok = t_next <= budget_s and gb_next <= budget_gb
    return {"n_next": n_next, "dof_estimate": dof_next, "alpha": alpha,
            "seconds_estimate": t_next, "gb_estimate": gb_next, "budget_s": budget_s,
            "budget_gb": budget_gb, "within_budget": ok}


def evaluate(geometry, rows_extended):
    """The problem's plan (frozen checks), on an extended family."""
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.orchestrate import (ADAPTERS, combine, convergence_params,
                                                          run_check)
    spec = _default_spec(geometry)
    sd = spec.to_dict()
    plan = trees.rule_plan(sd)
    ad = ADAPTERS[geometry]()
    ad.rows = rows_extended
    ad.hs = [r["h"] for r in rows_extended]
    cp = convergence_params(plan)
    v = {}
    for c in plan["checks"]:
        for q in c["qois"]:
            v.setdefault(q, {})[c["check"]] = run_check(ad, c["check"], q, c["params"],
                                                         element=spec.element,
                                                         conv_params=cp.get(q))
    return {q: (combine(v[q])[:2], v[q]) for q in spec.qois}


def act(geometry, qoi, budget_s=300.0, budget_gb=8.0):
    from omar_pfem.verification_agent.diagnose import diagnose_problem, pairwise_orders
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    try:
        d = diagnose_problem(geometry, [qoi]).get(qoi, {})
        decision = {"diagnosis_supported": d.get("supported", []),
                    "next_investigation": d.get("next_investigation")}
        if "D4" not in decision["diagnosis_supported"]:
            decision["action"] = "none: D4 (pre-asymptotic) not supported"
            return decision
        plan = plan_next_level(geometry, budget_s, budget_gb)
        decision["plan"] = plan
        if not plan["within_budget"]:
            decision["action"] = "not run: estimate exceeds the budget"
            return decision
        mod, rows, extra, fpath = _rows(geometry)
        t0 = time.time()
        row = mod.solve_level(plan["n_next"], 1.0)
        row["agent_followup"] = {"reason": f"D4 supported for {qoi}", "seconds": time.time() - t0}
        extra.append(row)
        json.dump(extra, open(fpath, "w"), indent=1)
        before = evaluate(geometry, rows)
        after = evaluate(geometry, sorted(rows + extra, key=lambda r: r["n"]))
        ext = sorted(rows + extra, key=lambda r: r["n"])
        decision.update(action=f"ran n = {plan['n_next']}",
                        measured_seconds=row["agent_followup"]["seconds"],
                        n_dof=row["n_dof"],
                        pairwise_orders_after=pairwise_orders([r["h"] for r in ext][-6:],
                                                              [r[qoi] for r in ext][-6:]),
                        verdicts={q: {"before": before[q][0], "after": after[q][0]}
                                  for q in before},
                        fit_after={q: after[q][1].get("qoi_convergence_multilevel", {})
                                   .get("fit_order") for q in after},
                        singular_order_after={q: {k: after[q][1].get("singular_order", {})
                                                  .get(k) for k in ("verdict", "observed",
                                                                    "expected")}
                                              for q in after})
        return decision
    finally:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)


def score_lshape(decision):
    pre = json.load(open(os.path.join(HERE, "followup_preregistration.json")))
    po = decision.get("pairwise_orders_after") or []
    so = decision.get("singular_order_after", {})
    lam2 = pre["theory_2lam"]
    return {
        "P1 next pairwise order in [0.97, 1.15]": bool(po) and po[-1] is not None
        and 0.97 <= po[-1] <= 1.15,
        "P2 v_right_top fit >= 0.94 and singular_order PASS":
            (decision["fit_after"]["v_right_top"] or 0) >= 0.94
            and so.get("v_right_top", {}).get("verdict") == "PASS",
        "P3 energy fit within 0.15 of 2 lambda":
            abs((decision["fit_after"]["strain_energy"] or 0) - lam2) <= 0.15,
        "P4 under 120 s": decision.get("measured_seconds", 1e9) < 120,
    }


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    argv = sys.argv[1:] if argv is None else argv
    geometry, qoi = argv[0], argv[1]
    dec = act(geometry, qoi)
    if geometry == "l_shape" and qoi == "v_right_top" and "verdicts" in dec:
        dec["predictions_met"] = score_lshape(dec)
    json.dump(dec, open(os.path.join(HERE, f"followup_{geometry}_{qoi}.json"), "w"), indent=1,
              default=str)
    print(json.dumps({k: v for k, v in dec.items() if k != "singular_order_after"}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
