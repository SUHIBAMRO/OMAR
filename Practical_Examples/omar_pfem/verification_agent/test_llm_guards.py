"""Tests for the deterministic guards around the LLM roles (no LLM needed).

    cd Practical_Examples && python -m pytest omar_pfem/verification_agent/test_llm_guards.py -q
"""
import pytest

from omar_pfem.verification_agent import llm_roles as R

QOIS = ["strain_energy", "peak_PK1_frob"]


def _plan(**over):
    p = {"checks": [{"check": "qoi_convergence", "qois": list(QOIS),
                     "params": {"band_tol": 0.01, "gci_tol": 0.05}, "rationale": "x"}],
         "expected_failures": []}
    p.update(over)
    return p


def test_valid_plan_passes():
    assert R.validate_plan(_plan(), QOIS) == []


def test_unknown_check_rejected():
    p = _plan(checks=[{"check": "eyeball_it", "qois": QOIS, "params": {}, "rationale": ""}])
    assert any("not in the catalogue" in e for e in R.validate_plan(p, QOIS))


def test_loose_tolerance_rejected():
    p = _plan()
    p["checks"][0]["params"]["gci_tol"] = 0.5
    assert any("outside" in e for e in R.validate_plan(p, QOIS))


def test_foreign_parameter_rejected():
    p = _plan()
    p["checks"][0]["params"]["tol"] = 0.01
    assert any("not allowed" in e for e in R.validate_plan(p, QOIS))


def test_undeclared_and_uncovered_qois_rejected():
    p = _plan()
    p["checks"][0]["qois"] = ["strain_energy", "made_up_qoi"]
    errs = R.validate_plan(p, QOIS)
    assert any("made_up_qoi" in e for e in errs)
    assert any("'peak_PK1_frob' has no check" in e for e in errs)


def test_extra_keys_rejected():
    with pytest.raises(ValueError):
        R.validate_plan({**_plan(), "accepted": True}, QOIS)


def test_spec_manufactured_needs_smooth():
    d = {"name": "x", "geometry": "unit_square", "dim": 2, "material": "neo_hookean",
         "loading": "t", "regularity": "clamped_free_corner", "element": "Q4",
         "reference": "exact_manufactured", "qois": ["a"], "well_posedness": [],
         "open_questions": []}
    assert any("smooth" in e for e in R.validate_spec(d))


def test_advisor_cannot_overturn_gate():
    advice = {"per_qoi": [{"qoi": "strain_energy", "stated_verdict": "PASS", "interpretation": ""},
                          {"qoi": "peak_PK1_frob", "stated_verdict": "PASS", "interpretation": ""}],
              "summary": "", "manuscript_wording": ""}
    flags = R.audit_advice(advice, {"strain_energy": "PASS", "peak_PK1_frob": "FAIL"})
    assert flags == ["advisor says PASS for 'peak_PK1_frob' but the gate says FAIL"]


def test_advisor_omission_flagged():
    advice = {"per_qoi": [], "summary": "", "manuscript_wording": ""}
    assert R.audit_advice(advice, {"strain_energy": "PASS"}) == [
        "advisor omitted QoI 'strain_energy' (gate: PASS)"]


def test_gate_flags_divergent_peak():
    # B1 peak PK1 on N=11,21,41,81: increments grow with refinement
    from omar_pfem.verification_agent import gates
    g = gates.qoi_convergence_gate([0.1, 0.05, 0.025, 0.0125],
                                   [9.33502, 11.0406, 13.3427, 16.28])
    assert g["verdict"] == "FAIL" and g["reason"].startswith("divergent")
    assert "richardson_extrapolate" not in g


# ---- problem/method trees -------------------------------------------------

def _spec(reg="clamped_free_corner", ref="mesh_refinement_family", qois=None):
    return {"reference": ref, "regularity": reg,
            "qois": qois or ["strain_energy", "max_disp_mag", "interior_avg_PK1_frob",
                             "peak_PK1_frob", "sigma_xx_max", "L2_rel"]}


def test_qoi_classes():
    from omar_pfem.verification_agent.trees import classify_qoi
    assert [classify_qoi(q) for q in _spec()["qois"]] == [
        "global", "displacement", "region_average", "pointwise_peak", "pointwise_peak",
        "error_norm"]


def test_rules_singular_peak_needs_robustness_and_predicts_fail():
    from omar_pfem.verification_agent.trees import apply_rules
    a = apply_rules(_spec())["peak_PK1_frob"]
    assert "strategy_robustness" in a["required"] and a["predicted"] == "FAIL"


def test_rules_smooth_peak_not_predicted_to_fail():
    from omar_pfem.verification_agent.trees import apply_rules
    assert apply_rules(_spec(reg="smooth"))["peak_PK1_frob"]["predicted"] is None


def test_rule_plan_satisfies_its_own_rules():
    from omar_pfem.verification_agent import trees
    for reg in ("smooth", "clamped_free_corner", "stress_concentration", "unknown"):
        s = _spec(reg=reg)
        assert trees.plan_rule_errors(trees.rule_plan(s), s) == []


def test_plan_missing_required_check_flagged():
    from omar_pfem.verification_agent import trees
    s = _spec()
    plan = {"checks": [{"check": "qoi_convergence", "qois": s["qois"], "params": {},
                        "rationale": ""}], "expected_failures": []}
    errs = trees.plan_rule_errors(plan, s)
    assert any("strategy_robustness" in e and "peak_PK1_frob" in e for e in errs)


# ---- adaptive refinement controller --------------------------------------

def _fake_solver(calls):
    # smooth QoI converging at O(h^2) to 1.0, and a peak growing like log(1/h)
    import math

    def solve(N):
        calls.append(N)
        h = 1.0 / (N - 1)
        return {"N": N, "h": h, "n_dof": 2 * N * N, "wall_clock_s": 1e-4 * N * N,
                "smooth": 1.0 + 0.5 * h ** 2, "slow": 1.0 + 2.0 * h ** 0.5,
                "peak": 3.0 + math.log(1.0 / h) + 0.1 * math.log(1.0 / h) ** 2,
                # stable within the band but oscillating: no rate can ever be shown
                "wobble": 1.0 + 1e-3 * (-1) ** round(math.log2(N - 1))}
    return solve


def test_adaptive_stops_early_and_classifies():
    from omar_pfem.verification_agent import adaptive
    calls = []
    res = adaptive.adaptive_refine(_fake_solver(calls), ["smooth", "peak"], budget_s=1e6,
                                   log=lambda *a: None)
    assert res["decisions"]["smooth"]["verdict"] == "PASS"
    assert res["decisions"]["peak"]["verdict"] == "FAIL"
    assert res["decisions"]["peak"]["reason"].startswith("divergent")
    # the peak is decided FAIL (divergent) at 3 levels; the smooth QoI needs the
    # multilevel gate, which needs >= 4 levels, before it can be decided PASS
    assert calls == [11, 21, 41, 81]
    assert res["decisions"]["smooth"]["evidence"] == "rate"
    assert "qoi_convergence_multilevel" in res["decisions"]["smooth"]["evidence_from"]
    assert res["decisions"]["peak"]["levels"] == [11, 21, 41]


def test_adaptive_respects_budget():
    from omar_pfem.verification_agent import adaptive
    calls = []
    res = adaptive.adaptive_refine(_fake_solver(calls), ["slow"], budget_s=0.3,
                                   log=lambda *a: None)
    assert res["stop"].startswith("budget")
    assert res["decisions"]["slow"]["verdict"] == "INCONCLUSIVE"


# ---- end-to-end, offline: plan omits rule-required checks -> augmented ----

def test_orchestrate_replay_augments_and_reviews(tmp_path):
    import json, os, subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = tmp_path / "repo.jsonl"
    out = tmp_path / "trace.json"
    env = {**os.environ, "VERIFICATION_REPO": str(repo)}
    subprocess.run([sys.executable, "-m", "omar_pfem.verification_agent.agent", "orchestrate",
                    "--backend", "replay", "--replay",
                    os.path.join(here, "fixtures", "replay_b1_example.json"),
                    "--problem", "B1 test", "--trace_out", str(out)],
                   cwd=os.path.dirname(os.path.dirname(here)), env=env, check=True,
                   capture_output=True)
    t = json.load(open(out))
    assert "error" not in t
    # the fixture predates qoi_convergence_multilevel, so rule R2 adds it for every QoI
    assert sorted(t["rule_augmented"]) == sorted(
        ["anchored_robustness:interior_avg_PK1_frob", "anchored_robustness:peak_PK1_frob",
         "strategy_robustness:interior_avg_PK1_frob", "strategy_robustness:peak_PK1_frob"]
        + [f"qoi_convergence_multilevel:{q}" for q in (
            "strain_energy", "max_disp_mag", "v_top_off_center", "interior_avg_PK1_frob",
            "peak_PK1_frob")])
    # the fixture's gci_tol 0.05 / band 0.01 exceed the displacement/global ceilings
    assert sorted(t["ceiling_clamped"]) == sorted(
        f"qoi_convergence:{q}" for q in ("strain_energy", "max_disp_mag", "v_top_off_center",
                                          "interior_avg_PK1_frob", "peak_PK1_frob"))
    assert t["critique"] is not None
    assert t["gate_summary"]["peak_PK1_frob"]["verdict"] == "FAIL"
    assert t["gate_summary"]["strain_energy"]["verdict"] == "PASS"
    assert t["gate_summary"]["strain_energy"]["convergence_evidence"] == "rate"
    assert "qoi_convergence_multilevel" in t["gate_summary"]["strain_energy"]["gates"]
    assert t["gate_summary"]["peak_PK1_frob"]["convergence_evidence"] == "none"
    assert t["advisor_flags"] == ["advisor says PASS for 'peak_PK1_frob' but the gate says FAIL"]


# ---- critic follow-ups: order window, evidence, anchored robustness, ceilings ----

def test_order_window_rejects_non_asymptotic_rate():
    from omar_pfem.verification_agent import gates
    hs = [0.4, 0.2, 0.1, 0.05]
    vals = [1.0 + 0.3 * h ** 0.2 for h in hs]   # monotone, p = 0.2
    g = gates.qoi_convergence_gate(hs, vals, band_tol=1e-4, gci_tol=10.0, order_window=(0.5, 3))
    assert g["verdict"] == "FAIL" and "not asymptotic" in g["reason"]
    g2 = gates.qoi_convergence_gate(hs, [1.0 + h ** 2 for h in hs], band_tol=1e-4,
                                    gci_tol=0.05, order_window=(0.5, 3))
    assert g2["verdict"] == "PASS" and g2["evidence"] == "rate"
    assert abs(g2["observed_order"] - 2.0) < 1e-6


def test_band_pass_is_labeled_weaker_evidence():
    from omar_pfem.verification_agent import gates
    # oscillating inside the band: stable, but no rate can be shown
    g = gates.qoi_convergence_gate([0.1, 0.05, 0.025], [1.0, 1.0006, 1.0003], band_tol=0.01)
    assert g["verdict"] == "PASS" and g["evidence"] == "band"
    # monotone and shrinking inside the band: rate evidence takes precedence
    g2 = gates.qoi_convergence_gate([0.1, 0.05, 0.025], [1.0, 1.0005, 1.0006], band_tol=0.01,
                                    order_window=(0.5, 3))
    assert g2["evidence"] == "rate" and abs(g2["observed_order"] - 2.3219) < 1e-3


def test_anchored_robustness_catches_shared_error():
    from omar_pfem.verification_agent import gates
    # three strategies agree with each other (0.2%) but all sit 5% off the limit
    g_s = gates.strategy_robustness_gate([1.05, 1.051, 1.052], tol=0.02)
    g_a = gates.anchored_robustness_gate([1.05, 1.051, 1.052], anchor=1.0, tol=0.02)
    assert g_s["verdict"] == "PASS" and g_a["verdict"] == "FAIL"


def test_ceiling_flags_loosened_tolerance():
    from omar_pfem.verification_agent import trees
    s = _spec(qois=["strain_energy"])
    plan = trees.rule_plan(s)
    assert trees.plan_rule_errors(plan, s) == []
    plan["checks"][0]["params"]["gci_tol"] = 0.05
    assert any(e.startswith("ceiling:") for e in trees.plan_rule_errors(plan, s))


def test_order_window_from_element_degree():
    from omar_pfem.verification_agent.trees import order_window
    assert order_window("Q4") == (0.5, 3) and order_window("Q20") == (0.5, 5)


# ---- multilevel (5-level least-squares) convergence gate -------------------

_HS5 = [0.07, 0.058, 0.05, 0.044, 0.04]   # non-uniform ratios, like the B3 family


def test_multilevel_recovers_power_law():
    from omar_pfem.verification_agent import gates
    vals = [1.0 + 3.0 * h ** 2 for h in _HS5]
    g = gates.multilevel_convergence_gate(_HS5, vals, gci_tol=0.05, order_window=(0.5, 5))
    assert g["evidence"] == "rate" and g["verdict"] == "PASS" and g["failed_criteria"] == []
    assert abs(g["fit_order"] - 2.0) < 1e-4 and abs(g["fit_limit"] - 1.0) < 1e-7
    a, kind = gates.family_anchor(g)
    assert kind == "multilevel fit limit" and abs(a - 1.0) < 1e-7


def test_multilevel_rejects_oscillation():
    from omar_pfem.verification_agent import gates
    vals = [1.0 + 3.0 * h ** 2 + 1e-3 * (-1) ** i for i, h in enumerate(_HS5)]
    g = gates.multilevel_convergence_gate(_HS5, vals, band_tol=0.01, order_window=(0.5, 5))
    assert g["evidence"] == "band" and "monotone" in g["failed_criteria"]


def test_multilevel_rejects_non_asymptotic_order():
    from omar_pfem.verification_agent import gates
    vals = [1.0 + 0.3 * h ** 0.2 for h in _HS5]
    g = gates.multilevel_convergence_gate(_HS5, vals, band_tol=1e-6, order_window=(0.5, 5))
    assert g["verdict"] == "FAIL" and "order_window" in g["failed_criteria"]


def test_multilevel_needs_four_levels():
    from omar_pfem.verification_agent import gates
    g = gates.multilevel_convergence_gate(_HS5[:3], [1.0, 1.1, 1.2][::-1])
    assert g["verdict"] == "INCONCLUSIVE"


def test_multilevel_two_term_error_still_rate():
    from omar_pfem.verification_agent import gates
    vals = [2.0 + 1.0 * h ** 2 + 5.0 * h ** 3 for h in _HS5]
    g = gates.multilevel_convergence_gate(_HS5, vals, gci_tol=0.05, order_window=(0.5, 5))
    assert g["evidence"] == "rate" and 2.0 < g["fit_order"] < 3.0


def test_combine_takes_strongest_passing_evidence_and_any_fail():
    from omar_pfem.verification_agent.orchestrate import combine
    v = {"qoi_convergence": {"verdict": "PASS", "evidence": "band"},
         "qoi_convergence_multilevel": {"verdict": "PASS", "evidence": "rate"}}
    assert combine(v) == ("PASS", "rate", ["qoi_convergence_multilevel"])
    v["strategy_robustness"] = {"verdict": "FAIL"}
    assert combine(v)[0] == "FAIL"
    v = {"qoi_convergence": {"verdict": "FAIL", "evidence": "none"},
         "qoi_convergence_multilevel": {"verdict": "PASS", "evidence": "rate"}}
    assert combine(v)[0] == "FAIL"   # one convergence gate never relaxes the other


def test_rule_plan_includes_multilevel_for_solution_verification():
    from omar_pfem.verification_agent import trees
    plan = trees.rule_plan(_spec())
    ml = [c for c in plan["checks"] if c["check"] == "qoi_convergence_multilevel"]
    assert sorted(q for c in ml for q in c["qois"]) == sorted(_spec()["qois"])


def test_adaptive_band_only_refines_until_rate_or_limit():
    from omar_pfem.verification_agent import adaptive
    calls = []
    res = adaptive.adaptive_refine(_fake_solver(calls), ["wobble"], budget_s=1e6,
                                   max_levels=6, log=lambda *a: None)
    d = res["decisions"]["wobble"]
    assert calls == [11, 21, 41, 81, 161, 321] and res["stop"] == "max levels"
    assert d["verdict"] == "PASS" and d["evidence"] == "band"
    assert "rate not certified" in d["reason"]
    calls = []
    res = adaptive.adaptive_refine(_fake_solver(calls), ["wobble"], budget_s=1e6,
                                   require_rate=False, log=lambda *a: None)
    assert calls == [11, 21, 41, 81]
    assert res["decisions"]["wobble"]["evidence"] == "band"


def test_multilevel_four_levels_says_no_leave_one_out():
    from omar_pfem.verification_agent import gates
    hs = _HS5[1:]
    g = gates.multilevel_convergence_gate(hs, [1.0 + 3.0 * h ** 2 for h in hs],
                                          order_window=(0.5, 5))
    assert g["evidence"] == "rate" and "leave_one_out_orders" not in g
    assert "no leave-one-out check with 4 levels" in g["reason"]


def test_repository_latest_record_supersedes(tmp_path):
    from omar_pfem.verification_agent import repository
    from omar_pfem.verification_agent.spec import ProblemSpec
    p = str(tmp_path / "r.jsonl")
    s = ProblemSpec(name="X q", geometry="g", dim=2, material="m", loading="l",
                    regularity="smooth", element="Q4", reference="mesh_refinement_family",
                    qois=("q",))
    repository.record(s, {}, {}, {"a": {"verdict": "FAIL"}}, path=p)
    repository.record(s, {}, {}, {"a": {"verdict": "PASS"}}, path=p)
    assert len(repository.load(p, latest_only=False)) == 2
    recs = repository.load(p)
    assert len(recs) == 1 and recs[0]["accepted"] is True


def test_adaptive_min_levels_five_uses_full_multilevel_check():
    from omar_pfem.verification_agent import adaptive
    calls = []
    res = adaptive.adaptive_refine(_fake_solver(calls), ["smooth"], budget_s=1e6,
                                   min_levels=5, log=lambda *a: None)
    assert calls == [11, 21, 41, 81, 161]
    g = res["decisions"]["smooth"]["gates"]["qoi_convergence_multilevel"]
    assert "leave_one_out_orders" in g and res["decisions"]["smooth"]["evidence"] == "rate"
