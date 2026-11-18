"""What does the LLM add? Repeated LLM-pipeline runs against the rule-only
baseline, per problem.

For each problem text and each of `--repeats` runs, the full pipeline
(formalizer -> proposer -> critic -> checks -> advisor) is executed and
compared with the deterministic rule-only run on the same data
(`agent rules`). Per run it records:

  spec       did the formalizer recover the reference specification
             (geometry, regularity, element, reference type)? spec errors
  plan       checks the LLM omitted that the rules had to add; tolerances
             the code had to replace
  verdicts   agreement with the rule-only verdicts (overall + evidence) on
             the QoIs both evaluated; QoIs the LLM chose that the rules'
             default list lacks, and vice versa
  prediction which failures the LLM predicted vs the rules: missed and
             false alarms
  advisor    claim-audit flags
  cost       LLM calls, USD, wall time

and summarises mean and spread over the repeats. The question it answers:
does the LLM contribute anything the rules alone do not (better predictions,
useful QoI choices, problem formalisation from free text), and is that
contribution stable across runs?

BILLED: with --backend anthropic or claude-cli every run costs money
(roughly USD 0.5-1.1 per run on these problems). --backend replay runs the
code path at no cost with the canned answers in fixtures/ (only B1 has one;
a replay run checks the script, it measures nothing).

Each run uses a COPY of the current solved-case repository, so neighbour
retrieval behaves as in a real run but repository.jsonl is not modified.

    python -m omar_pfem.verification_agent.ablation --backend claude-cli \\
        --problems b1,b2,b3 --repeats 5 --out ablation_result.json
"""
import argparse
import json
import os
import shutil
import statistics
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = {"b1": ("problems/b1_square.txt", "unit_square"),
            "b2": ("problems/b2_ring.txt", "quarter_ring"),
            "b3": ("problems/b3_groove.txt", "bushing_groove")}


def rule_only(geometry):
    """{qoi: (overall, evidence)}, rule predictions and QoI list, no LLM."""
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.orchestrate import (ADAPTERS, run_check, combine,
                                                          convergence_params)
    spec = _default_spec(geometry)
    sd = spec.to_dict()
    plan, analysis = trees.rule_plan(sd), trees.apply_rules(sd)
    adapter = ADAPTERS[geometry]()
    cp = convergence_params(plan)
    v = {}
    for c in plan["checks"]:
        for q in c["qois"]:
            v.setdefault(q, {})[c["check"]] = run_check(adapter, c["check"], q, c["params"],
                                                         element=spec.element,
                                                         conv_params=cp.get(q))
    verdicts = {q: combine(v[q])[:2] for q in spec.qois}
    failed = {q for q, (o, _) in verdicts.items() if o == "FAIL"}
    predicted = {q for q, a in analysis.items() if a["predicted"] == "FAIL"}
    return {"spec": sd, "verdicts": verdicts, "qois": list(spec.qois),
            "missed": sorted(failed - predicted), "false_alarm": sorted(predicted - failed)}


def one_run(backend_name, replay, problem_path, geometry, baseline):
    from omar_pfem.verification_agent import llm_roles
    from omar_pfem.verification_agent.orchestrate import orchestrate
    backend = llm_roles.make_backend(backend_name, replay)
    trace, t0, error = {}, time.time(), None
    try:
        orchestrate(backend, open(os.path.join(HERE, problem_path)).read(),
                    log=lambda *a, **k: None, trace=trace)
    except Exception as e:                           # a failed run is a result too
        error = f"{type(e).__name__}: {e}"
    ref = baseline["spec"]
    spec = trace.get("spec", {})
    gs = trace.get("gate_summary", {})
    common = [q for q in gs if q in baseline["verdicts"]]
    agree = [q for q in common
             if (gs[q]["verdict"], gs[q]["convergence_evidence"]) == baseline["verdicts"][q]]
    pred = trace.get("prediction", {}).get("llm", {})
    return {
        "error": error,
        "wall_s": time.time() - t0,
        "llm_calls": len(getattr(backend, "calls", [])),
        "llm_cost_usd": getattr(backend, "cost_usd", 0.0),
        "spec_match": {k: spec.get(k) == ref.get(k)
                       for k in ("geometry", "regularity", "element", "reference")},
        "spec_errors": len(trace.get("spec_errors", [])),
        "checks_added_by_rules": len(trace.get("rule_augmented", []) or []),
        "tolerances_replaced": len(trace.get("tolerances_overridden", []) or []),
        "qois_evaluated": sorted(gs),
        "qois_not_in_rule_list": sorted(set(gs) - set(baseline["qois"])),
        "rule_qois_not_chosen": sorted(set(baseline["qois"]) - set(gs)),
        "verdict_agreement": (len(agree) / len(common)) if common else None,
        "verdict_disagreements": sorted(set(common) - set(agree)),
        "llm_missed": pred.get("missed"), "llm_false_alarm": pred.get("false_alarm"),
        "advisor_flags": len(trace.get("advisor_flags", []) or []),
    }


def summarise(runs):
    ok = [r for r in runs if r["error"] is None]

    def stat(key):
        vals = [r[key] for r in ok if r[key] is not None]
        if not vals:
            return None
        return {"mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals)}

    return {
        "runs": len(runs), "failed_runs": len(runs) - len(ok),
        "spec_fully_correct": sum(all(r["spec_match"].values()) for r in ok),
        "verdict_agreement": stat("verdict_agreement"),
        "checks_added_by_rules": stat("checks_added_by_rules"),
        "tolerances_replaced": stat("tolerances_replaced"),
        "llm_missed_count": stat("llm_missed_n"),
        "llm_false_alarm_count": stat("llm_false_alarm_n"),
        "advisor_flags": stat("advisor_flags"),
        "cost_usd_total": sum(r["llm_cost_usd"] for r in runs),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("anthropic", "claude-cli", "replay"), default="replay")
    ap.add_argument("--replay", default=os.path.join(HERE, "fixtures", "replay_b1_example.json"))
    ap.add_argument("--problems", default="b1")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "ablation_result.json"))
    args = ap.parse_args(argv)
    if args.backend == "replay" and args.problems != "b1":
        raise SystemExit("replay answers exist for b1 only")

    scratch = tempfile.mkdtemp(prefix="ablation_repo_")
    result = {"backend": args.backend, "repeats": args.repeats, "problems": {}}
    for name in args.problems.split(","):
        path, geometry = PROBLEMS[name]
        baseline = rule_only(geometry)
        runs = []
        for i in range(args.repeats):
            repo = os.path.join(scratch, f"{name}_{i}.jsonl")
            shutil.copy(os.path.join(HERE, "repository.jsonl"), repo)
            from omar_pfem.verification_agent import repository
            old = repository.DEFAULT_PATH
            # the repository functions bind their path default at import, so
            # point those defaults at this run's copy and restore them after
            patched = {f: f.__defaults__ for f in (repository.record, repository.load,
                                                   repository.nearest)}
            for f in patched:
                f.__defaults__ = tuple(repo if d == old else d for d in f.__defaults__)
            try:
                r = one_run(args.backend, args.replay, path, geometry, baseline)
            finally:
                for f, d in patched.items():
                    f.__defaults__ = d
            r["llm_missed_n"] = len(r["llm_missed"] or [])
            r["llm_false_alarm_n"] = len(r["llm_false_alarm"] or [])
            runs.append(r)
            print(f"{name} run {i + 1}/{args.repeats}: "
                  + (f"ERROR {r['error']}" if r["error"] else
                     f"agreement {r['verdict_agreement']}, missed {r['llm_missed']}, "
                     f"false alarms {r['llm_false_alarm']}, ${r['llm_cost_usd']:.2f}"), flush=True)
        result["problems"][name] = {
            "rule_only": {"missed": baseline["missed"], "false_alarm": baseline["false_alarm"],
                          "qois": baseline["qois"]},
            "runs": runs, "summary": summarise(runs)}
        json.dump(result, open(args.out, "w"), indent=1, default=str)
    shutil.rmtree(scratch, ignore_errors=True)
    for name, p in result["problems"].items():
        s = p["summary"]
        print(f"\n{name}: {s['runs']} runs ({s['failed_runs']} failed), spec fully correct "
              f"{s['spec_fully_correct']}, verdict agreement {s['verdict_agreement']}, "
              f"rules missed {p['rule_only']['missed']} / false alarms "
              f"{p['rule_only']['false_alarm']}, LLM missed {s['llm_missed_count']} / false "
              f"alarms {s['llm_false_alarm_count']}, cost ${s['cost_usd_total']:.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
