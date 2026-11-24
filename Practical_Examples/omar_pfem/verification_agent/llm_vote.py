"""Does a majority vote over three LLM runs cut the false alarms? (2026-11-24, billed,
approved by Omar).

Each LLM run predicts which quantities will FAIL. One run's prediction varies (round M:
1-3 false alarms per run). Here the three runs of each problem vote: a quantity is predicted
FAIL when at least two of three runs predict it (majority), or when all three do
(unanimous). Every prediction is scored against the same reference: the FAIL verdicts of the
deterministic rule plan (ablation.rule_only), so single runs, votes and the rules' own
predictions are comparable. A run in which the LLM chose no quantity is left out of its vote.

    python -m omar_pfem.verification_agent.llm_vote --preregister
    python -m omar_pfem.verification_agent.llm_vote --score
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "ablation_round_vote")
PRE = os.path.join(HERE, "llm_vote_preregistration.json")
OUT = os.path.join(HERE, "llm_vote_result.json")
PROBLEMS = ("b1", "b2", "b3", "cook", "plate", "lamina", "lshape", "crack", "cylinder", "block")
FROZEN = ("ablation.py", "llm_vote.py", "orchestrate.py", "llm_roles.py", "trees.py", "gates.py")


def _sha(n):
    return hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest()


def preregister():
    if os.path.isdir(DIR) and os.listdir(DIR):
        raise SystemExit("vote runs exist: not a pre-registration")
    pre = {"written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
           "problems": list(PROBLEMS), "repeats": 3, "backend": "claude-cli",
           "predictions": {
               "V1 fewer false alarms": "summed over the ten problems, the majority vote has "
                                        "fewer false alarms than the mean single run (MEDIUM)",
               "V2 no more misses": "the majority vote misses no more failures than the mean "
                                    "single run (MEDIUM)",
               "V3 still above rules": "the majority vote still has more false alarms than "
                                       "the rules' own predictions (MEDIUM)",
               "V4 unanimity": "the unanimous vote has the fewest false alarms of the three "
                               "LLM variants and misses more than the majority (LOW-MEDIUM)"}}
    json.dump(pre, open(PRE, "w"), indent=1)
    print(json.dumps(pre["predictions"], indent=1))


def score():
    pre = json.load(open(PRE))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    from omar_pfem.verification_agent.ablation import PROBLEMS as P, rule_only
    res, tot = {}, {k: [0, 0] for k in ("single_mean", "majority", "unanimous", "rules")}
    cost = 0.0
    for p in pre["problems"]:
        d = json.load(open(os.path.join(DIR, f"{p}.json")))["problems"][p]
        base = rule_only(P[p][1])
        failed = {q for q, (o, _) in base["verdicts"].items() if o == "FAIL"}
        qs = set(base["qois"])
        runs = [r for r in d["runs"] if r["error"] is None and r["qois_evaluated"]
                and r.get("llm_predicted_fail") is not None]
        cost += sum(r["llm_cost_usd"] for r in d["runs"])
        sets = [set(r["llm_predicted_fail"]) & qs for r in runs]
        sc = lambda S: (len(failed - S), len(S - failed))
        single = [sc(S) for S in sets]
        smean = (sum(m for m, _ in single) / len(single), sum(f for _, f in single) / len(single)) if single else (None, None)
        maj = {q for q in qs if sum(q in S for S in sets) * 2 > len(sets)} if sets else set()
        una = set.intersection(*sets) if sets else set()
        rules_pred = set(failed) - set(base["missed"]) | set(base["false_alarm"])
        row = {"runs_used": len(sets), "failed_by_rules": sorted(failed),
               "single": single, "single_mean": smean, "majority": sc(maj),
               "unanimous": sc(una), "rules": sc(rules_pred),
               "majority_set": sorted(maj), "unanimous_set": sorted(una)}
        res[p] = row
        if sets:
            for k, v in (("single_mean", smean), ("majority", row["majority"]),
                         ("unanimous", row["unanimous"]), ("rules", row["rules"])):
                tot[k][0] += v[0]
                tot[k][1] += v[1]
    met = {
        "V1 fewer false alarms": tot["majority"][1] < tot["single_mean"][1],
        "V2 no more misses": tot["majority"][0] <= tot["single_mean"][0],
        "V3 still above rules": tot["majority"][1] > tot["rules"][1],
        "V4 unanimity": tot["unanimous"][1] <= min(tot["majority"][1], tot["single_mean"][1])
        and tot["unanimous"][0] > tot["majority"][0],
    }
    out = {"per_problem": res, "totals_missed_false_alarm": tot, "cost_usd": cost,
           "predictions_met": met}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(tot, indent=1), f"cost USD {cost:.2f}")
    for k, v in met.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    return out


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    preregister() if "--preregister" in sys.argv else score()
