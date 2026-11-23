"""Three-way evaluation (2026-11-23): what does each layer of the agent add?

Same stored solves, same quantities, same tolerances, same reference labels:

  A  conventional      three finest levels of one family, Richardson order and GCI,
                       accepted iff monotone, shrinking and GCI <= tolerance
                       (conventional.py; Celik et al. 2008 / ASME V&V 20 practice)
  B  deterministic     the framework WITHOUT an LLM: rules plan, gates (three- and
                       five-level), strategy and anchored robustness, singular-order
                       check, combination ("agent rules"); criteria as pre-registered
  B+ deterministic     B with the post hoc singular-order extension
     + post hoc        (VERIF_POSTHOC_SINGULAR=1); labelled post hoc throughout
  B* final             B+ and, for a rate certification, the cross-check v2: certified
                       only if an independent discretization (or the exact solution)
                       agrees; otherwise INCONCLUSIVE
  C  LLM agent         the full pipeline with LLM roles (formalizer, proposer, critic,
                       advisor), measured in the 9 stored billed runs on B1, B2, B3
                       (ablation_result_live.json; no new billed run was made). Code
                       decides every verdict, so C's verdicts equal B's on the quantities
                       it evaluates; what C adds or removes is measured separately:
                       coverage, the LLM's verdict predictions, cost.

Reference labels ("truth"), from conventional.truth(): a pointwise peak at a
singular point is not verifiable; the B3 groove maxima are not verifiable; the B3
sharp-sphere averages are band-only; all else verifiable. For the 3D benchmark the
exact solution decides: the hoop peak is verifiable but its finest-level error
(2.03%) exceeds the 2% tolerance, so NOT certifying it is the correct decision
(label "needs_finer"). These labels rest on the same corner theory B encodes; only
the 3D labels are independent of it.

Metrics
  certification   correct certifications (verifiable and PASS), false alarms
                  (verifiable, not PASS), missed failures (not verifiable but PASS)
  rate-certified  certifications with a quantified uncertainty (PASS with rate)
  true coverage   3D benchmark only: does the certified interval contain the exact value?
  diagnosis       for every non-certified quantity of the problems the diagnosis
                  engine covers: is the supported cause the reference cause?
  cost            unknowns solved (sum of n_dof over the solves each method needs;
                  solve times are not stored for every problem); LLM USD and seconds
  human decisions an operational proxy, stated so it can be disputed: set-up inputs a
                  person must write (A: choose the family; B: specification, data
                  adapter, problem facts for the diagnosis; C: adapter only, the LLM
                  writes the specification) plus every quantity a person must still
                  examine (A: every non-certified quantity, no cause is given; B: every
                  non-certified quantity without a sufficient automatic diagnosis; C: as
                  B plus every LLM false alarm, which a person must dismiss)

    python -m omar_pfem.verification_agent.three_way
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "three_way_result.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape", "edge_crack", "thick_cylinder_3d")
LLM_KEYS = {"unit_square": "b1", "quarter_ring": "b2", "bushing_groove": "b3"}


def _j(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def label(geo, q):
    from omar_pfem.verification_agent.conventional import truth
    if geo == "thick_cylinder_3d" and q == "peak_hoop":
        return "needs_finer"
    return truth(geo, q)


# ---------------------------------------------------------------------------
# cost: unknowns solved
# ---------------------------------------------------------------------------
def family_and_all_dofs(geo, ad):
    """(family n_dof sorted by h descending, n_dof of every solve the framework uses)."""
    if geo == "bushing_groove":
        fam = [3 * r["n_nodes"] for r in ad.family]
        extra = [3 * r["n_nodes"] for _, r in ad.zbeta]
        radial = [3 * r["n_nodes"] for r in json.load(open(os.path.join(HERE, "b3_radial.json")))
                  if "n_nodes" in r] if os.path.exists(os.path.join(HERE, "b3_radial.json")) else []
        return fam, fam + extra + radial
    rows = ad.rows
    fam = [r["n_dof"] for r in rows]
    strat = [r["n_dof"] for r in getattr(ad, "strategy_rows", []) if r not in rows]
    return fam, fam + strat


def independent_dofs(geo):
    if geo in ("unit_square", "quarter_ring"):
        t = "b1" if geo == "unit_square" else "b2"
        d = _j("independent_solver_result.json")
        return sum(v["n_dof"] for k, v in d.items() if k.startswith(f"{t}:q2:"))
    if geo == "cook_membrane":
        return sum(v["n_dof"] for k, v in _j("independent_extra_result.json").items()
                   if k.startswith("cook:q2:"))
    if geo == "open_hole_lamina":
        return sum(r["n_dof"] for r in _j("iga_lamina_result.json").get("iga", [])
                   if r["p"] == 3 and r["beta"] == 2.8)
    if geo == "bushing_groove":
        return sum(v["n_dof"] for v in _j("iga_b3_result.json").values())
    return 0


# ---------------------------------------------------------------------------
def run():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent import diagnose
    from omar_pfem.verification_agent.conventional import conventional
    from omar_pfem.verification_agent.verify_report import rules
    xc = {(r["problem"], r["qoi"]): r for r in _j("crosscheck_result.json")["rows"]}
    rows, per_problem = [], {}
    for geo in PROBLEMS:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        R = rules.__wrapped__(geo)
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        Rp = rules.__wrapped__(geo)
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        ad = R["adapter"]
        diag = {}
        if geo in diagnose.FACTS:
            try:
                diag = diagnose.diagnose_problem(geo)
            except Exception as e:                       # recorded, not hidden
                diag = {"_error": repr(e)}
        for q, comb in R["combine"].items():
            A, _ = conventional(ad, q)
            B, Bev = comb[0], comb[1]
            Bp, Bpev = Rp["combine"][q][0], Rp["combine"][q][1]
            x = xc.get((geo, q))
            if Bp == "PASS" and Bpev == "rate":
                Bstar = "PASS" if (x and x["status"] == "cross_checked") else "INCONCLUSIVE"
            else:
                Bstar = Bp
            d = diag.get(q) if isinstance(diag, dict) else None
            rows.append({"problem": geo, "qoi": q, "truth": label(geo, q), "A": A,
                         "B": B, "B_evidence": Bev, "Bp": Bp, "Bp_evidence": Bpev,
                         "Bstar": Bstar, "crosscheck": x["status"] if x else None,
                         "diagnosis_covered": geo in diagnose.FACTS,
                         "diagnosis_sufficient": bool(d and d.get("evidence_sufficient")),
                         "diagnosis_supported": (d or {}).get("supported", [])})
        fam, alls = family_and_all_dofs(geo, ad)
        per_problem[geo] = {"unknowns_A": sum(sorted(fam)[-3:]),
                            "unknowns_B": sum(alls),
                            "unknowns_Bstar": sum(alls) + independent_dofs(geo),
                            "diagnosis_error": diag.get("_error") if isinstance(diag, dict)
                            else None}

    def score(key, subset=None):
        rs = [r for r in rows if subset is None or r["problem"] in subset]
        ver = [r for r in rs if r["truth"] == "verifiable"]
        bad = [r for r in rs if r["truth"] in ("not_verifiable", "needs_finer")]
        out = {"quantities": len(rs), "verifiable": len(ver), "must_not_certify": len(bad),
               "certified_correctly": sum(r[key] == "PASS" for r in ver),
               "false_alarms": sum(r[key] != "PASS" for r in ver),
               "false_alarms_FAIL": sum(r[key] == "FAIL" for r in ver),
               "false_alarms_INCONCLUSIVE": sum(r[key] == "INCONCLUSIVE" for r in ver),
               "missed_failures": sum(r[key] == "PASS" for r in bad)}
        return out

    scores = {k: score(k) for k in ("A", "B", "Bp", "Bstar")}
    # rate-certified (quantified uncertainty) and true coverage on the 3D benchmark
    for k, ev in (("B", "B_evidence"), ("Bp", "Bp_evidence")):
        scores[k]["rate_certified"] = sum(r[k] == "PASS" and r[ev] == "rate" for r in rows)
    scores["Bstar"]["rate_certified_and_cross_checked"] = sum(
        r["Bstar"] == "PASS" and r["Bp_evidence"] == "rate" for r in rows)
    cyl = _j("cylinder3d_result.json")
    coverage = {"T2_covered": cyl.get("T2_covered"), "T2_of": cyl.get("T2_of"),
                "note": "3D benchmark: for every rate-certified quantity the exact value lies "
                        "inside the certified interval"}
    # human decisions (operational proxy)
    human = {}
    for k, setup in (("A", 1), ("Bp", 3), ("Bstar", 3)):
        per = {}
        for geo in PROBLEMS:
            rs = [r for r in rows if r["problem"] == geo]
            if k == "A":
                exam = sum(r["A"] != "PASS" for r in rs)
            else:
                key = k
                exam = sum(r[key] != "PASS" and not r["diagnosis_sufficient"] for r in rs)
            per[geo] = setup + exam
        human[k] = {"per_problem": per, "total": sum(per.values())}
    # C: the LLM agent, from the stored billed runs
    abl = _j("ablation_result_live.json").get("problems", {})
    llm = {}
    for geo, key in LLM_KEYS.items():
        P = abl.get(key)
        if not P:
            continue
        rs = {r["qoi"]: r for r in rows if r["problem"] == geo}
        gate_fail = {q for q, r in rs.items() if r["B"] == "FAIL"}
        runs = []
        for run in P["runs"]:
            pred_fail = (gate_fail - set(run["llm_missed"])) | set(run["llm_false_alarm"])
            ev = set(run["qois_evaluated"])
            not_ver = {q for q, r in rs.items() if r["truth"] != "verifiable"}
            runs.append({
                "cost_usd": run["llm_cost_usd"], "wall_s": run["wall_s"],
                "llm_calls": run["llm_calls"], "spec_errors": run["spec_errors"],
                "coverage": len(ev & set(rs)) / len(rs),
                "verdict_agreement_with_B": run["verdict_agreement"],
                "pred_fail_on_verifiable": len(pred_fail - not_ver),
                "pred_fail_on_not_verifiable": len(pred_fail & not_ver),
                "not_verifiable_total": len(not_ver),
                "missed_not_verifiable": len(not_ver - pred_fail)})
        rule = P["rule_only"]
        llm[geo] = {"runs": runs,
                    "mean": {k: statistics.mean(r[k] for r in runs)
                             for k in ("cost_usd", "wall_s", "coverage",
                                       "pred_fail_on_verifiable", "missed_not_verifiable")},
                    "rules_prediction_missed": len(rule["missed"]),
                    "rules_prediction_false_alarm": len(rule["false_alarm"])}
        human.setdefault("C", {"per_problem": {}})
        human["C"]["per_problem"][geo] = (2 + human["Bp"]["per_problem"][geo] - 3
                                          + round(statistics.mean(r["pred_fail_on_verifiable"]
                                                                  for r in runs)))
    human["C"]["total_on_B1_B3"] = sum(human["C"]["per_problem"].values())
    human["Bp"]["total_on_B1_B3"] = sum(human["Bp"]["per_problem"][g] for g in LLM_KEYS)
    human["A"]["total_on_B1_B3"] = sum(human["A"]["per_problem"][g] for g in LLM_KEYS)
    sub = score("Bp", LLM_KEYS)
    res = {"scores": scores, "scores_on_B1_B3": {"A": score("A", LLM_KEYS), "B+": sub},
           "true_coverage_3d": coverage, "per_problem": per_problem, "human": human,
           "llm": llm, "rows": rows,
           "diagnosis": diagnosis_accuracy(rows)}
    json.dump(res, open(OUT, "w"), indent=1)
    return res


# reference causes of the non-certified quantities on the problems the diagnosis covers;
# the B1/B2/B3 entries are diagnose.MANUAL (manual investigations), the others follow
# from theory and from what later runs showed (stated so they can be disputed)
REFERENCE_CAUSE = {
    "M1": "singular point (unbounded peak)",
    "Q1": "sharp region samples a changing set of Gauss points",
    "D3+Q2": "pressure oscillation accumulated by a magnitude measure",
    "Q3": "maximum taken at interior Gauss points",
    "C1": "verification capability missing (singular-order theory only for degree 1)",
}


def reference_cause(geo, q, r):
    from omar_pfem.verification_agent.diagnose import MANUAL
    if (geo, q) in MANUAL:
        return sorted(MANUAL[(geo, q)][0])
    if r["truth"] == "not_verifiable" and "peak" in q:
        return ["M1"]
    if geo == "plate_hole" and r["Bp"] == "PASS" and r["B"] != "PASS":
        return ["C1"]                    # INCONCLUSIVE by default only for lack of theory
    return None


def diagnosis_accuracy(rows):
    out = []
    for r in rows:
        if not r["diagnosis_covered"] or r["B"] == "PASS":
            continue
        ref = reference_cause(r["problem"], r["qoi"], r)
        if ref is None:
            continue
        out.append({"problem": r["problem"], "qoi": r["qoi"], "reference": ref,
                    "supported": r["diagnosis_supported"],
                    "correct": set(ref) <= set(r["diagnosis_supported"])})
    return {"cases": out, "correct": sum(c["correct"] for c in out), "of": len(out)}


if __name__ == "__main__":
    res = run()
    for k, v in res["scores"].items():
        print(f"{k:<6}", v)
    print("B1-B3:", res["scores_on_B1_B3"])
    print("3D coverage:", res["true_coverage_3d"])
    print("diagnosis:", res["diagnosis"]["correct"], "/", res["diagnosis"]["of"])
    for c in res["diagnosis"]["cases"]:
        if not c["correct"]:
            print("   wrong:", c)
    print("human:", {k: v.get("total", v.get("total_on_B1_B3")) for k, v in res["human"].items()},
          {k: v.get("total_on_B1_B3") for k, v in res["human"].items()})
    for g, v in res["llm"].items():
        print("LLM", g, v["mean"], "rules missed/fa", v["rules_prediction_missed"],
              v["rules_prediction_false_alarm"])
    for g, v in res["per_problem"].items():
        print(g, v)
