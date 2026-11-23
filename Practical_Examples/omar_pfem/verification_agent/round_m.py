"""Round M (2026-11-23): billed LLM-vs-rules ablation on seven more problems, and the
check-level half of the det F guard (POST HOC).

1. Aggregates ablation_round_M/<problem>.json (7 problems x 3 runs, approved by Omar). A run in
   which the LLM chose no quantity at all (degenerate; the ablation script counts it as zero
   misses and zero false alarms) is reported apart and excluded from the means.
2. Runs the two strategy checks (strategy_robustness, anchored_robustness) on every
   quantity of the thick cylinder with VERIF_POSTHOC_DETF off and on. With it on, the stored
   inverted strategy solve (g = 2.0, min det F = -1.03) is replaced by its guarded re-solve
   (cylinder3d_strategy_guarded.json, 40 increments); this is the case the ablation exposed.

    python -m omar_pfem.verification_agent.round_m
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "ablation_round_M")
OUT = os.path.join(HERE, "round_m_result.json")
PROBLEMS = ("cook", "plate", "lamina", "lshape", "crack", "cylinder", "block")


def ablation():
    out, cost, runs_all = {}, 0.0, 0
    for p in PROBLEMS:
        d = json.load(open(os.path.join(DIR, f"{p}.json")))["problems"][p]
        runs = d["runs"]
        runs_all += len(runs)
        cost += sum(r["llm_cost_usd"] for r in runs)
        valid = [r for r in runs if r["error"] is None and r["qois_evaluated"]]
        mean = lambda k: sum(len(r[k] or []) for r in valid) / len(valid)
        out[p] = {
            "runs": len(runs), "valid_runs": len(valid),
            "degenerate_runs": len(runs) - len(valid),
            "spec_fully_correct": sum(all(r["spec_match"].values()) for r in runs),
            "spec_field_wrong": sorted({k for r in runs for k, v in r["spec_match"].items() if not v}),
            "agreement_mean": sum(r["verdict_agreement"] for r in valid) / len(valid),
            "llm_missed_mean": mean("llm_missed"), "llm_false_alarm_mean": mean("llm_false_alarm"),
            "llm_missed_runs": [r["llm_missed"] for r in valid],
            "rules_missed": d["rule_only"]["missed"], "rules_false_alarm": d["rule_only"]["false_alarm"],
            "cost_usd": sum(r["llm_cost_usd"] for r in runs),
        }
    return {"problems": out, "runs": runs_all, "cost_usd": cost,
            "spec_fully_correct": sum(v["spec_fully_correct"] for v in out.values()),
            "degenerate_runs": sum(v["degenerate_runs"] for v in out.values()),
            "block_rule_miss_caught_by_llm": sum(
                "root_gk_avg_von_mises" not in (m or []) for m in out["block"]["llm_missed_runs"]),
            "cook_energy_caught_by_llm": sum(
                "strain_energy" not in (m or []) for m in out["cook"]["llm_missed_runs"])}


def cylinder_guard():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.orchestrate import ADAPTERS, run_check, convergence_params
    spec = _default_spec("thick_cylinder_3d")
    cp = convergence_params(trees.rule_plan(spec.to_dict()))
    ad = ADAPTERS["thick_cylinder_3d"]()
    res = {}
    for sw in ("0", "1"):
        os.environ["VERIF_POSTHOC_DETF"] = sw
        try:
            for q in spec.qois:
                for chk in ("strategy_robustness", "anchored_robustness"):
                    r = run_check(ad, chk, q, trees.check_params(chk, q, "thick_cylinder_3d"),
                                  element=spec.element, conv_params=cp.get(q))
                    res.setdefault(q, {})[f"{chk}:{'on' if sw == '1' else 'off'}"] = {
                        "verdict": r["verdict"], "reason": r.get("reason")}
        finally:
            os.environ.pop("VERIF_POSTHOC_DETF", None)
    g = json.load(open(os.path.join(HERE, "cylinder3d_strategy_guarded.json")))[0]
    stored = [r for r in ad.strategy_rows if r["grading"] == 2.0][0]
    uniform = [r for r in ad.strategy_rows if r["grading"] == 1.0][0]
    return {"verdicts": res,
            "stored_min_detF": stored["min_detF"], "guarded_min_detF": g["min_detF"],
            "guarded_nsteps": g["detf_guard"]["resolved_nsteps"],
            "guarded_energy_rel_to_uniform": g["strain_energy"] / uniform["strain_energy"] - 1,
            "fail_off": sorted({q for q, v in res.items() for k, x in v.items()
                                if k.endswith(":off") and x["verdict"] == "FAIL"}),
            "fail_on": sorted({q for q, v in res.items() for k, x in v.items()
                               if k.endswith(":on") and x["verdict"] == "FAIL"})}


def main():
    res = {"posthoc": "the check-level guard was added after seeing the ablation results",
           "ablation": ablation(), "cylinder_guard": cylinder_guard()}
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    a, c = res["ablation"], res["cylinder_guard"]
    print(f"ablation: {a['runs']} runs, USD {a['cost_usd']:.2f}, spec fully correct "
          f"{a['spec_fully_correct']}, degenerate {a['degenerate_runs']}")
    for p, v in a["problems"].items():
        print(f"  {p:9s} agree {v['agreement_mean']:.2f} missed {v['llm_missed_mean']:.2f} "
              f"false alarms {v['llm_false_alarm_mean']:.2f} (valid {v['valid_runs']})")
    print("cylinder FAIL off:", c["fail_off"], " on:", c["fail_on"])
    return res


if __name__ == "__main__":
    main()
