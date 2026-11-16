"""Orchestrated verification run: LLM roles around deterministic gates.

    formalize -> validate spec -> retrieve nearest solved cases
      -> propose plan -> validate plan (code) -> critic -> (one revision)
      -> execute catalogued checks (code) -> record verdicts (code)
      -> advisor explanation -> audit advisor against verdicts (code)

The data a check runs on comes from an adapter keyed by geometry; adapters
expose a fixed QoI vocabulary, so the formalizer cannot invent observables.
"""
import json
import os

from omar_pfem.verification_agent import gates, repository, trees
from omar_pfem.verification_agent import llm_roles as R
from omar_pfem.verification_agent.spec import ProblemSpec

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Data adapters: geometry -> mesh family + QoI series (all precomputed or
# computed by deterministic solvers; no LLM involvement)
# ----------------------------------------------------------------------------


class B1Adapter:
    geometry = "unit_square"
    description = ("B1: 2D unit square, clamped bottom edge, top-edge traction, smooth "
                   "spatially varying E, nu (mesh_convergence.AnalyticFieldB1), Q4, "
                   "global refinement N=11,21,41,81")
    qois = ("max_disp_mag", "u_top_off_center", "v_top_off_center", "u_mid_domain",
            "v_mid_domain", "strain_energy", "interior_avg_PK1_frob", "peak_PK1_frob")

    def __init__(self, material="neo_hookean", Ns=(11, 21, 41, 81), strategy_N=81):
        from omar_pfem.verification_agent.b1_ladder import run_ladder
        cache = os.path.join(HERE, f"b1_ladder_{material}.json")
        self.rows = run_ladder(list(Ns), material, cache)
        self.hs = [r["h"] for r in self.rows]
        # same node count, three refinement strategies: uniform and corner
        # grading a=0.5, 0.8 (b1_ladder.grade_nodes), read from the cache only
        all_rows = json.load(open(cache))
        self.strategy_rows = sorted(
            (r for r in all_rows if r["N"] == strategy_N), key=lambda r: r.get("grading", 0.0))

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


class B3Adapter:
    geometry = "bushing_groove"
    description = ("B3: 3D bushing with a circumferential groove, prescribed rotation, "
                   "neo-Hookean, Q20 global refinement family of 7 levels, 2,376-45,000 elements, plus "
                   "three local z-clustering strategies at 10,200 elements")

    def __init__(self):
        from omar_pfem.verification_agent.agent import _load_b3_family, _load_b3_zbeta_10200
        self.family = _load_b3_family()
        self.zbeta = _load_b3_zbeta_10200(self.family)
        self.hs = [g["n_elem"] ** (-1.0 / 3.0) for g in self.family]
        self.scale = self.family[-1]["region_avg_sigma_frob"]
        self.get = {"sigma_frob_avg": lambda g: g["region_avg_sigma_frob"]}
        for c in ("xx", "yy", "zz", "xy", "xz", "yz"):
            for s in ("avg", "max"):
                self.get[f"sigma_{c}_{s}"] = lambda g, c=c, s=s: g["per_component"][c][s]
        self.qois = tuple(self.get)

    def series(self, qoi):
        return self.hs, [self.get[qoi](g) for g in self.family], self.scale

    def strategies(self, qoi):
        return [self.get[qoi](z) for _, z in self.zbeta], self.scale


ADAPTERS = {"unit_square": B1Adapter, "bushing_groove": B3Adapter}
ADAPTER_VOCAB = {
    "unit_square": (B1Adapter.description, B1Adapter.qois),
    "bushing_groove": (B3Adapter.description,
                       ("sigma_frob_avg",) + tuple(f"sigma_{c}_{s}" for c in
                                                   ("xx", "yy", "zz", "xy", "xz", "yz")
                                                   for s in ("avg", "max"))),
}


# ----------------------------------------------------------------------------
# Execution of a validated plan (deterministic)
# ----------------------------------------------------------------------------


def run_check(adapter, check, qoi, params, element=None, conv_params=None):
    """Execute one catalogued check. The order window for qoi_convergence
    comes from the element (trees.order_window), never from the plan."""
    if check == "qoi_convergence":
        hs, vals, scale = adapter.series(qoi)
        return gates.qoi_convergence_gate(hs, vals, scale=scale,
                                          order_window=trees.order_window(element), **params)
    if check == "anchored_robustness":
        s = adapter.strategies(qoi)
        if s is None:
            return {"gate": "anchored_robustness", "verdict": "INCONCLUSIVE",
                    "reason": "no alternative refinement strategy computed for this problem"}
        hs, vals, scale = adapter.series(qoi)
        conv = gates.qoi_convergence_gate(hs, vals, scale=scale,
                                          order_window=trees.order_window(element),
                                          **(conv_params or {}))
        anchor, kind = gates.family_anchor(conv)
        return gates.anchored_robustness_gate(s[0], anchor, scale=s[1], anchor_kind=kind,
                                              **params)
    if check == "strategy_robustness":
        s = adapter.strategies(qoi)
        if s is None:
            return {"gate": "strategy_robustness", "verdict": "INCONCLUSIVE",
                    "reason": "no alternative refinement strategy computed for this problem"}
        vals, scale = s
        return gates.strategy_robustness_gate(vals, scale=scale, **params)
    if check == "mms_rates":
        return {"gate": "mms_rates", "verdict": "INCONCLUSIVE",
                "reason": "run `agent mms` for code verification; not a per-QoI check"}
    raise ValueError(check)


def orchestrate(backend, problem_text, log=print, trace=None):
    """`trace` (a dict) is filled in place, so a caller can save it even if
    a later step raises."""
    trace = {} if trace is None else trace
    trace["problem_text"] = problem_text

    # 1. formalize
    vocab = "\n".join(f"- geometry '{g}': {d}\n  available QoIs: {list(q)}"
                      for g, (d, q) in ADAPTER_VOCAB.items())
    spec_d, errs = R.formalize(backend, problem_text + "\n\nAVAILABLE DATA ADAPTERS "
                               "(geometry must be one of these; QoIs must come from its "
                               "list):\n" + vocab)
    if spec_d["geometry"] not in ADAPTERS:
        errs.append(f"no data adapter for geometry '{spec_d['geometry']}'")
    else:
        allowed = ADAPTER_VOCAB[spec_d["geometry"]][1]
        errs += [f"QoI '{q}' not available for {spec_d['geometry']}"
                 for q in spec_d["qois"] if q not in allowed]
    trace["spec"], trace["spec_errors"] = spec_d, errs
    log(f"[formalizer] {spec_d['name']}: {len(spec_d['qois'])} QoIs, "
        f"regularity={spec_d['regularity']}")
    for w in spec_d["well_posedness"]:
        log(f"   well-posedness: {w}")
    for q in spec_d["open_questions"]:
        log(f"   open question:  {q}")
    if errs:
        log(f"[guard] spec rejected: {errs}")
        return trace
    spec = ProblemSpec(**{k: (tuple(v) if k == "qois" else v) for k, v in spec_d.items()
                          if k in ProblemSpec.__dataclass_fields__})

    # 2. retrieve
    neigh = [{"similarity": round(s, 3), "problem": r["problem"]["name"],
              "verdicts": {k: v["verdict"] for k, v in r["verdicts"].items()}}
             for s, r in repository.nearest(spec, k=5)]
    trace["neighbours"] = neigh
    log(f"[repository] nearest: {[(n['problem'], n['similarity']) for n in neigh]}")

    # 3. propose / validate / critique, at most one revision
    baseline = trees.rule_plan(spec_d)
    trace["rule_plan"], trace["rule_analysis"] = baseline, trees.apply_rules(spec_d)
    log(f"[rules] baseline requires: " + "; ".join(
        f"{c['check']} on {len(c['qois'])} QoIs" for c in baseline["checks"])
        + f"; predicts FAIL: {[e['qoi'] for e in baseline['expected_failures']]}")
    plan, perrs = R.propose(backend, spec_d, neigh)
    perrs = perrs + trees.plan_rule_errors(plan, spec_d)
    crit = R.critique(backend, spec_d, plan) if not perrs else None
    if perrs or not crit["approve"]:
        feedback = {"validation_errors": perrs, "critic": crit}
        log(f"[guard/critic] revision requested: {json.dumps(feedback)}")
        plan, perrs = R.propose(backend, spec_d, neigh + [{"revise_previous_plan": feedback}])
        perrs = perrs + trees.plan_rule_errors(plan, spec_d)
        crit = R.critique(backend, spec_d, plan) if not perrs else None
    # Rules are authoritative: if the only remaining problems are checks the
    # rules require but the LLM left out, add them (catalogue defaults) and
    # record the augmentation, instead of discarding the whole plan.
    rule_only = perrs and all(e.startswith(("rules ", "ceiling: ")) for e in perrs)
    if rule_only:
        base = {c["check"]: c for c in baseline["checks"]}
        need = trees.apply_rules(spec_d)
        added = []
        for q, r in need.items():
            have = {c["check"] for c in plan["checks"] if q in c["qois"]}
            for chk in r["required"]:
                if chk not in have:
                    plan["checks"].append({"check": chk, "qois": [q],
                                           "params": dict(base[chk]["params"]),
                                           "rationale": f"added by rules {r['rules']}"})
                    added.append(f"{chk}:{q}")
        # clamp tolerances looser than the class ceiling
        clamped = []
        for c in plan["checks"]:
            for q in list(c["qois"]):
                if q not in need:
                    continue
                ceil = trees.class_params(c["check"], need[q]["qoi_class"])
                if any(c["params"].get(k, 0) > v + 1e-12 for k, v in ceil.items()):
                    c["qois"].remove(q)
                    plan["checks"].append({"check": c["check"], "qois": [q],
                                           "params": {k: min(c["params"].get(k, v), v)
                                                      for k, v in ceil.items()},
                                           "rationale": "tolerance clamped to class ceiling"})
                    clamped.append(f"{c['check']}:{q}")
        plan["checks"] = [c for c in plan["checks"] if c["qois"]]
        trace["rule_augmented"], trace["ceiling_clamped"] = added, clamped
        log(f"[rules] plan augmented with required checks the LLM omitted: {added}; "
            f"tolerances clamped to class ceilings: {clamped}")
        perrs = R.validate_plan(plan, spec_d["qois"]) + trees.plan_rule_errors(plan, spec_d)
    if not perrs and crit is None:
        # the final plan has not been reviewed yet (e.g. after augmentation)
        crit = R.critique(backend, spec_d, plan)
    trace.update(plan=plan, plan_errors=perrs, critique=crit)
    if perrs:
        log(f"[guard] plan rejected after revision: {perrs}")
        return trace
    if not crit["approve"]:
        log(f"[critic] plan still not approved: {crit['objections']}; executing anyway, "
            "objections stored with the record")
    log(f"[proposer] {len(plan['checks'])} checks; expected failures: "
        f"{[e['qoi'] for e in plan['expected_failures']]}")

    # 4. execute (code only)
    adapter = ADAPTERS[spec.geometry]()
    per_qoi = {}
    conv_params = {q: c["params"] for c in plan["checks"] if c["check"] == "qoi_convergence"
                   for q in c["qois"]}
    for c in plan["checks"]:
        for q in c["qois"]:
            g = run_check(adapter, c["check"], q, c["params"], element=spec.element,
                          conv_params=conv_params.get(q))
            per_qoi.setdefault(q, {})[c["check"]] = g
    gate_summary = {}
    for q, verdicts in per_qoi.items():
        sub = ProblemSpec(**{**spec.to_dict(), "qois": (q,), "name": f"{spec.name} {q}"})
        rec = repository.record(sub, method={"orchestrated": True, "plan": plan,
                                             "critic": crit},
                                observables={}, verdicts=verdicts)
        # a QoI's overall verdict: FAIL if any gate failed, PASS if all passed
        vs = [v["verdict"] for v in verdicts.values()]
        overall = "FAIL" if "FAIL" in vs else ("PASS" if all(v == "PASS" for v in vs)
                                               else "INCONCLUSIVE")
        gate_summary[q] = {"verdict": overall,
                           "gates": {k: {kk: vv for kk, vv in v.items()
                                         if kk in ("verdict", "reason", "relative_spread",
                                                   "observed_order", "gci_fine", "evidence",
                                                   "finest_three", "anchor_kind")}
                                     for k, v in verdicts.items()}}
        log(f"[gates] {q:<24} {overall:<13} accepted={rec['accepted']}")
    trace["gate_summary"] = gate_summary

    # prediction score: did the proposer anticipate the failures?
    failed = {q for q, g in gate_summary.items() if g["verdict"] == "FAIL"}
    trace["prediction"] = {}
    for who, exp in (("llm", plan["expected_failures"]),
                     ("rules", baseline["expected_failures"])):
        predicted = {e["qoi"] for e in exp}
        trace["prediction"][who] = {"predicted_fail": sorted(predicted),
                                    "actual_fail": sorted(failed),
                                    "missed": sorted(failed - predicted),
                                    "false_alarm": sorted(predicted - failed)}
        log(f"[score:{who}] predicted FAIL {sorted(predicted)} | actual FAIL {sorted(failed)}"
            f" | missed {sorted(failed - predicted)} | false alarm {sorted(predicted - failed)}")

    # 5. advise and audit
    advice, flags = R.advise(backend, spec_d, gate_summary)
    trace.update(advice=advice, advisor_flags=flags)
    log(f"[advisor] {advice['summary']}")
    log(f"[audit] {'advisor consistent with gates' if not flags else flags}")
    return trace
