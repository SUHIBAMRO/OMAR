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
    # a FAIL needs min_levels (4) levels too, like a PASS
    assert res["decisions"]["peak"]["levels"] == [11, 21, 41, 81]


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
    # the fixture predates qoi_convergence_multilevel (R2) and singular_order (R8)
    assert sorted(t["rule_augmented"]) == sorted(
        ["anchored_robustness:interior_avg_PK1_frob", "anchored_robustness:peak_PK1_frob",
         "strategy_robustness:interior_avg_PK1_frob", "strategy_robustness:peak_PK1_frob"]
        + [f"{chk}:{q}" for chk in ("qoi_convergence_multilevel", "singular_order")
           for q in ("strain_energy", "max_disp_mag", "v_top_off_center",
                     "interior_avg_PK1_frob", "peak_PK1_frob")])
    # tolerances are always set by code: the fixture's band 0.01 / gci 0.05 are replaced
    over = {(o["check"], o["qoi"]) for o in t["tolerances_overridden"]}
    assert {("qoi_convergence", q) for q in ("strain_energy", "max_disp_mag",
                                              "v_top_off_center", "interior_avg_PK1_frob",
                                              "peak_PK1_frob")} <= over
    for c in t["plan"]["checks"]:
        for q in c["qois"]:
            from omar_pfem.verification_agent import trees
            assert c["params"] == trees.check_params(c["check"], q, "unit_square")
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


def test_three_level_rate_contradicted_by_non_monotone_five_levels_is_band():
    from omar_pfem.verification_agent.orchestrate import combine
    v = {"qoi_convergence": {"verdict": "PASS", "evidence": "rate"},
         "qoi_convergence_multilevel": {"verdict": "PASS", "evidence": "band",
                                        "failed_criteria": ["monotone", "fit_quality"]}}
    assert combine(v) == ("PASS", "band", ["qoi_convergence", "qoi_convergence_multilevel"])
    # a multilevel failure that is not a monotonicity failure does not contradict it
    v["qoi_convergence_multilevel"]["failed_criteria"] = ["leave_one_out"]
    assert combine(v) == ("PASS", "rate", ["qoi_convergence"])


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


def test_rule_r7_strategy_needs_anchor():
    from omar_pfem.verification_agent import trees
    s = _spec(reg="smooth", qois=["max_disp_mag"])
    plan = trees.rule_plan(s)
    plan["checks"].append({"check": "strategy_robustness", "qois": ["max_disp_mag"],
                           "params": {"tol": 0.01}, "rationale": ""})
    errs = trees.plan_rule_errors(plan, s)
    assert any("R7" in e and "anchored_robustness" in e for e in errs)


def test_anchored_inconclusive_without_converged_family():
    from omar_pfem.verification_agent import orchestrate as O

    class Fake:
        def series(self, q):        # divergent: increments grow
            return [0.1, 0.05, 0.025, 0.0125, 0.00625], [9.3, 11.0, 13.3, 16.3, 19.9], None

        def strategies(self, q):
            return [16.3, 19.9, 26.1], None
    g = O.run_check(Fake(), "anchored_robustness", "peak", {"tol": 0.02}, element="Q4")
    assert g["verdict"] == "INCONCLUSIVE" and "no limit" in g["reason"]


def test_requirement_tightens_ceiling_only_with_recorded_error():
    from omar_pfem.verification_agent import trees, requirements
    # B3 reaction moment: 1.38% / 3 = 0.46% < 1% GCI ceiling -> tightened
    p = trees.check_params("qoi_convergence", "reaction_moment_y", "bushing_groove")
    assert abs(p["gci_tol"] - 0.0138 / 3) < 1e-12 and p["band_tol"] == 0.002
    # groove stresses: >= 600% surrogate error -> requirement far above ceiling
    assert trees.check_params("qoi_convergence", "sigma_xx_avg", "bushing_groove") == \
        trees.class_params("qoi_convergence", "region_average")
    # nothing recorded -> class ceiling, and it says so
    assert requirements.required_uncertainty("unit_square", "strain_energy")[0] is None
    assert trees.check_params("qoi_convergence", "strain_energy", "unit_square") == \
        trees.class_params("qoi_convergence", "global")


def test_llm_tolerances_are_replaced_by_code():
    from omar_pfem.verification_agent.orchestrate import set_code_tolerances
    spec = {"geometry": "bushing_groove", "qois": ["reaction_moment_y", "sigma_xx_avg"]}
    plan = {"checks": [{"check": "qoi_convergence", "qois": spec["qois"],
                        "params": {"band_tol": 0.002, "gci_tol": 0.003}, "rationale": ""}],
            "expected_failures": []}
    new, changed = set_code_tolerances(plan, spec)
    params = {q: c["params"] for c in new["checks"] for q in c["qois"]}
    assert abs(params["reaction_moment_y"]["gci_tol"] - 0.0046) < 1e-12
    assert params["sigma_xx_avg"] == {"band_tol": 0.01, "gci_tol": 0.02}
    assert {c["qoi"] for c in changed} == {"reaction_moment_y", "sigma_xx_avg"}


def test_b3_kernel_qois_present_and_scaled():
    from omar_pfem.verification_agent import orchestrate as O
    a = O.B3Adapter()
    if not hasattr(a, "kernel_values"):
        pytest.skip("Gauss-point dumps not available")
    assert "gk2_sigma_zz_avg" in a.qois and "sigma_zz_avg" in a.qois
    _, vals, scale = a.series("gk2_sigma_zz_avg")
    # one kernel value per family level (every level has a Gauss-point dump)
    assert len(vals) == len(a.series("sigma_zz_avg")[1]) >= 7
    assert abs(scale - a.kernel_scale) < 1e-15
    assert a.series("sigma_zz_avg")[2] == a.scale     # sphere QoIs keep their scale


def test_audit_flags_convergence_claim_without_rate():
    adv = {"per_qoi": [
        {"qoi": "a", "stated_verdict": "PASS",
         "interpretation": "The average converges monotonically to 0.43."},
        {"qoi": "b", "stated_verdict": "PASS",
         "interpretation": "Stable in the band; it does not converge at a demonstrable rate "
                           "and is not monotone."},
        {"qoi": "c", "stated_verdict": "FAIL",
         "interpretation": "Monotone on the finest three levels but not asymptotic."},
        {"qoi": "d", "stated_verdict": "FAIL",
         "interpretation": "The convergence verdict is FAIL; the differences grow."}],
        "summary": "", "manuscript_wording": ""}
    flags = R.audit_advice(adv, {"a": "PASS", "b": "PASS", "c": "FAIL", "d": "FAIL"},
                           {"a": "band", "b": "band", "c": "none", "d": "none"},
                           {"a": False, "b": False, "c": True, "d": False})
    assert flags and all("'a'" in f for f in flags)


def test_singular_exponent_reproduces_known_values():
    import numpy as np
    from omar_pfem.verification_agent import singularity as S
    l1 = S.smallest_exponent(1.5 * np.pi, "free", "free", 0.3)        # L-shaped corner
    assert abs(l1 - 0.5444837) < 1e-6
    assert abs(S.smallest_exponent(1.5 * np.pi, "free", "free", 0.3, lo=l1 + 1e-3)
               - 0.9085292) < 1e-6
    assert abs(S.b1_corner_exponent(0.3) - 0.7111729) < 1e-6


def test_singular_order_gate_modes():
    from omar_pfem.verification_agent import gates
    hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    lam = 0.7
    g = gates.singular_order_gate(hs, [1 + h ** 1.4 for h in hs], lam, "global")
    assert g["verdict"] == "PASS" and abs(g["observed"] - 1.4) < 1e-3
    assert gates.singular_order_gate(hs, [1 + h ** 1.0 for h in hs], lam,
                                     "displacement")["verdict"] == "FAIL"
    assert gates.singular_order_gate(hs, [1 + h ** 2.0 for h in hs], lam,
                                     "displacement")["verdict"] == "PASS"   # lower bound
    g = gates.singular_order_gate(hs, [3 * h ** (lam - 1) for h in hs], lam, "pointwise_peak")
    assert g["verdict"] == "PASS" and abs(g["observed"] - (lam - 1)) < 1e-9


def test_mms_q20_reduced_rule_converges_at_theoretical_rates():
    # two coarse cubes are enough to see the asymptotic slopes (3 and 2)
    from omar_pfem.verification_agent import mms_q20 as M
    rows = [M.solve(n, 0.3, "reduced") for n in (2, 4)]
    assert M.rates(rows, "L2_rel")[0] > 2.8
    assert M.rates(rows, "H1_rel")[0] > 1.8


def test_b2_is_registered_as_smooth_held_out_problem():
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.orchestrate import ADAPTERS, ADAPTER_VOCAB
    assert "quarter_ring" in ADAPTERS and "quarter_ring" in ADAPTER_VOCAB
    sd = _default_spec("quarter_ring").to_dict()
    assert sd["regularity"] == "smooth" and sd["element"] == "Q4"
    req = trees.required_checks(trees.rule_plan(sd), sd)
    # smooth problem: both convergence checks, no singular-order or robustness requirement
    for q, r in req.items():
        assert {"qoi_convergence", "qoi_convergence_multilevel"} <= r["required"]
        assert "singular_order" not in r["required"]


def test_ablation_does_not_touch_the_repository(tmp_path):
    import filecmp
    import os
    import shutil
    from omar_pfem.verification_agent import ablation
    repo = os.path.join(os.path.dirname(ablation.__file__), "repository.jsonl")
    before = tmp_path / "before.jsonl"
    shutil.copy(repo, before)
    ablation.main(["--backend", "replay", "--problems", "b1", "--repeats", "1",
                   "--out", str(tmp_path / "abl.json")])
    assert filecmp.cmp(repo, before, shallow=False)


def test_diagnosis_matches_manual_findings():
    """The diagnosis engine reproduces every manually established cause
    (supported AND refuted) from the stored data alone, and leaves no
    insufficient-evidence item open on the three benchmarks."""
    from omar_pfem.verification_agent import diagnose as D
    res = {g: D.diagnose_problem(g) for g in ("unit_square", "quarter_ring", "bushing_groove")}
    rows = D.consistency(res)
    assert rows and all(ok for _, _, ok, _ in rows), [r for r in rows if not r[2]]
    assert all(d["evidence_sufficient"] for g in res.values() for d in g.values())
    # B1 corner peak is a mathematical singularity, not a solver problem
    assert "M1" in res["unit_square"]["peak_PK1_frob"]["supported"]
    assert {"S1", "S2", "S3"} <= set(res["unit_square"]["peak_PK1_frob"]["refuted"])


def test_diagnosis_actions_are_reachable():
    """Investigations the agent runs itself must be able to succeed: the B2
    Newton-tolerance re-solve uses a tolerance above the residual round-off
    floor (~8e-11 at N=49), so Newton can meet it."""
    import inspect
    from omar_pfem.verification_agent import diagnose as D
    src = inspect.getsource(D._act_b2_tolerance)
    assert "tol=1e-10" in src and "1e-11" not in src
    assert ("quarter_ring", "S1") in D.ACTIONS and ("bushing_groove", "S2") in D.ACTIONS


def test_conv_only_ignores_strategy_checks():
    from omar_pfem.verification_agent import diagnose as D
    rate = {"verdict": "PASS", "evidence": "rate"}
    g = {"qoi_convergence": dict(rate), "qoi_convergence_multilevel": dict(rate),
         "strategy_robustness": {"verdict": "FAIL", "evidence": "none"}}
    assert D._conv_only(g) == ("PASS", "rate")


def test_independent_solver_reproduces_project_solvers():
    """scikit-fem residual/tangent, same discrete problem: B1 (N=11) and B2
    (N=7) QoIs equal the project's stored ladder values to round-off."""
    import json
    from omar_pfem.verification_agent import independent_solver as S
    for prob, N in (("b1", 11), ("b2", 7)):
        ref = S._ladder(prob)[N]
        row = S.qois(prob, N, "same", *S.solve(prob, N, "same")[:5])
        for q in S.PROBLEMS[prob]["qois"]:
            assert abs(row[q] - ref[q]) <= 1e-9 * abs(ref[q]), (prob, q, row[q], ref[q])


def test_corner_lambda_ignores_regular_corners():
    """Cook: the 47.5-degree clamped/free corner has no singular exponent
    below 1 (None); the 108.4-degree one has lam = 0.6442. The minimum must
    skip the regular corner (this crashed before the held-out Cook test)."""
    from omar_pfem.verification_agent import diagnose as D
    assert abs(D.corner_lambda(D.FACTS["cook_membrane"]) - 0.6442) < 1e-3
    assert D.corner_lambda(D.FACTS["quarter_ring"]) is None


def test_cook_is_registered_as_singular_held_out_problem():
    import json, os
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import DEFAULT_SPECS
    from omar_pfem.verification_agent.orchestrate import ADAPTERS
    assert "cook_membrane" in ADAPTERS
    spec = DEFAULT_SPECS["cook_membrane"]
    assert trees.regularity_class(spec["regularity"]) == "singular" and spec["element"] == "Q4"
    pre = json.load(open(os.path.join(os.path.dirname(__file__), "cook_preregistration.json")))
    assert pre["rules_changed_for_this_problem"] is False
    assert pre["rule_predictions"]["peak_PK1_frob"]["predicted"] == "FAIL"


def test_cauchy_from_torchfem_flux_handles_both_conventions():
    """Cauchy (symmetric) passes through; a first Piola-Kirchhoff stress
    (non-symmetric) is converted. Converting the Cauchy stress again was
    the 2026-11-19 bug (stored B3 stresses were sigma F^T / J)."""
    import numpy as np
    from omar_pfem.torchfem_comparison import cauchy_from_torchfem_flux
    rng = np.random.default_rng(0)
    F = np.eye(3) + 0.05 * rng.standard_normal((5, 3, 3))
    S = rng.standard_normal((5, 3, 3)); sig = S + np.swapaxes(S, 1, 2)
    J = np.linalg.det(F)
    P = J[:, None, None] * sig @ np.linalg.inv(np.swapaxes(F, 1, 2))   # P = J sigma F^-T
    assert np.allclose(cauchy_from_torchfem_flux(sig, F), sig)
    assert np.allclose(cauchy_from_torchfem_flux(P, F), sig)


def test_exact_midside_mesh_is_valid_where_project_builder_inverts():
    from omar_pfem.verification_agent.b3_local_refine import quality
    q = quality(21, 26, 31, 5.0, 1.5, 0.9, exact=True)
    assert q["min_detJ"] > 0 and q["h_over_rho"] < 0.25
    assert quality(21, 26, 31, 5.0, 1.5, 0.9, exact=False)["min_detJ"] < 0


def test_confirmation_respects_memory_limit():
    from omar_pfem.verification_agent import diagnose as D
    t, gb, what = D._cook_finer_estimate()
    assert gb > D.MEM_LIMIT_GB      # N = 1024 would need ~13 GB: the agent must not run it


def test_literature_agent_rejects_invented_quotes():
    """A card field whose quote is not in the paper is unsupported and is not
    used in the problem text (CFM paper cites [49] for the moduli)."""
    import os
    import pytest
    from omar_pfem.verification_agent import literature_agent as L
    pdf = os.path.join(L.FEEDBACK, L.TRUTH["cfm_lamina"]["pdf"])
    if not os.path.exists(pdf):
        pytest.skip("CFM paper not present")
    txt = L.extract_text(pdf)
    sec, where = L.select_section(txt, "9.3", "open-hole lamina")
    assert where.startswith("section 9.3")
    f = lambda v, q: {"value": v, "quote": q}
    card = {"title": "t", "geometry": f("g", "a central hole of radius 3 mm"),
            "material": f("E1 = 126 GPa", "AS4/3501-6 has E1 = 126 GPa"),
            "boundary_conditions": f("b", "three displacement components at the hole are fixed"),
            "loading": f("l", "homogeneous uniaxial stress along the long axis"),
            "discretisation": f("d", "4096 isoparametric quadratic triangles"),
            "reference": {"type": "single_mesh", "convergence_evidence": "none_stated",
                          "quote": "4096 isoparametric quadratic triangles"},
            "qois": [{"name": "secant compliance", "kind": "compliance", "definition": "",
                      "reported_value": "", "quote": "secant compliance"}],
            "singular_features": [], "not_stated": []}
    ok = L.check_quotes(card, txt)
    assert ok["material"] is False and ok["geometry"] and ok["qois[0]"]
    assert "126" not in L.problem_text(card, ok)
    assert any(n.startswith("reference not verified") for n in L.needs(card, ok))
    rows, gaps = L.vocabulary(card)
    assert not gaps and rows[0]["class"] == "global"     # compliance known since 2026-11-20
    card["qois"].append({"name": "training loss", "kind": "other", "definition": "",
                         "reported_value": "", "quote": "secant compliance"})
    rows, gaps = L.vocabulary(card)
    assert not gaps and rows[1]["class"] == "out_of_scope"
    card["qois"].pop()
    # a finite concentration (singular=false) must not trigger the singular-order check
    card["singular_features"] = [{"description": "hole edge", "singular": False,
                                  "inferred": False, "quote": "secant compliance"}]
    assert not any(n.startswith("singular features") for n in L.needs(card, ok))
    card["singular_features"][0]["singular"] = True
    assert any(n.startswith("singular features") for n in L.needs(card, ok))


def test_literature_quote_check_is_unicode_normalised():
    """A quote with a composed accent must match paper text that carries the
    base letter + combining accent (pdftotext output)."""
    from omar_pfem.verification_agent import literature_agent as L
    paper = "the energy ⟨ĉref , T⟩ is the mechanism-controlled energy"
    assert L.normalise("⟨ĉref , T⟩ is the mechanism") in L.normalise(paper)


# ---------------------------------------------------------------------------
# Modules added on 2026-11-20/21 (held-out L-shape, spline discretizations,
# goal-oriented estimate, active learning)
# ---------------------------------------------------------------------------
def test_lshape_mesh_and_exponent():
    import numpy as np
    from omar_pfem.verification_agent import lshape
    for n, g in ((2, 1.0), (4, 2.0)):
        m = lshape.mesh(n, g)
        assert m.t.shape[1] == 3 * n * n
        assert not np.any((m.p[0] > 1 + 1e-12) & (m.p[1] > 1 + 1e-12))
    assert abs(lshape.corner_exponent() - 0.5444837) < 1e-6


def test_lamina_element_factory_default_unchanged():
    """The element switch used by goal_oriented.py leaves the Q2 results unchanged."""
    import json
    import os
    import pytest
    from omar_pfem.verification_agent import lamina
    if not os.path.exists(lamina.CACHE):
        pytest.skip("lamina ladder not present")
    ref = next(r for r in json.load(open(lamina.CACHE)) if r["m"] == 1 and r["g"] == 1.5)
    r = lamina.solve_level(1)
    for q in ("compliance", "u_tip", "hole_gk_gamma12_avg"):
        assert abs(r[q] / ref[q] - 1) < 1e-10


def test_iga_lamina_exact_geometry_and_convergence():
    import numpy as np
    from omar_pfem.verification_agent import iga_lamina, lamina
    r = iga_lamina.solve(2, 3)
    exact = np.pi * lamina.A_ELL * lamina.B_ELL - np.pi * lamina.RH ** 2
    assert abs(r["area_mm2"] / exact - 1) < 1e-7            # exact geometry (quadrature only)
    assert abs(r["compliance"] / 6.980613985e-05 - 1) < 0.01  # FEM m = 32 value


def test_iga_b3_geometry_residual_and_projector():
    import numpy as np
    import torch
    from scipy.integrate import quad
    from omar_pfem.verification_agent import iga_b3 as I
    pb = I.Problem(1, formulation="fbar_p1")
    ex = quad(lambda z: np.pi / 2 * (I.R_OUT ** 2 - I.R_in(np.array([z]))[0] ** 2), 0, 1,
              points=[0.35, 0.5, 0.65], limit=200)[0]
    assert abs(pb.w.numpy().sum() / ex - 1) < 1e-12           # exact volume
    assert np.abs(pb.residual(np.zeros(pb.ndof))).max() < 1e-10
    f = pb.phi @ torch.tensor([0.3, 0.1, -0.2, 0.05])
    assert float((pb.proj[3] @ f - f).abs().max()) < 1e-12    # P1 projector is exact on P1


def test_active_learning_gp_interpolates():
    import numpy as np
    from omar_pfem.verification_agent.active_learning import GP
    rng = np.random.default_rng(0)
    X = rng.random((12, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    gp = GP(X, y, np.full(12, 1e-10))
    assert np.abs(gp.predict(X) - y).max() < 1e-3


def test_crosscheck_rule():
    """A rate-certified uncertainty is final only if an independent value agrees."""
    from omar_pfem.verification_agent.crosscheck import decide
    assert decide(1.000, 0.002, 1.001, 0.0005, 1.0)[0] == "cross_checked"
    assert decide(1.000, 0.002, 1.020, 0.0010, 1.0)[0] == "contradicted"   # B3-like 2% gap
    assert decide(1.000, 0.002, None, None, 1.0)[0] == "single_family"


def test_crosscheck_v2_independent_must_converge():
    """v2: an independent sequence counts only if it passes the gate itself (rate AND
    within the tolerance); a non-converging or loose one leaves the quantity unresolved."""
    from omar_pfem.verification_agent.crosscheck import assess_independent
    ml = {"band_tol": 0.01, "gci_tol": 0.02}
    hs = [1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
    good = [1 + 0.5 * h ** 2 for h in hs]                         # clean order 2
    a = assess_independent(hs, good, ml, 1.0)
    assert a["evidence"] == "rate" and a["verdict"] == "PASS" and abs(a["fit_limit"] - 1) < 1e-6
    wobble = [1.000, 1.004, 0.998, 1.003, 1.001]                 # no rate (band at best)
    assert assess_independent(hs, wobble, ml, 1.0)["evidence"] != "rate"
    slow = [1 + 0.5 * h ** 0.6 for h in hs]                       # rate, but U above 2%
    s = assess_independent(hs, slow, ml, 1.0)
    assert not (s["evidence"] == "rate" and s["verdict"] == "PASS")
    assert assess_independent(hs[:3], good[:3], ml, 1.0)["evidence"] == "too_few_levels"


def test_crosscheck_v2_stored_statuses():
    """The stored v2 result keeps unresolved cases INCONCLUSIVE, never CERTIFIED."""
    import json
    import os
    d = json.load(open(os.path.join(os.path.dirname(__file__), "crosscheck_result.json")))
    for r in d["rows"]:
        assert (r["final"] == "CERTIFIED") == (r["status"] == "cross_checked")
    gk = {r["qoi"]: r["status"] for r in d["rows"] if r["problem"] == "bushing_groove"}
    assert gk["gk2_sigma_xx_avg"] != "cross_checked"          # spline sequences unconverged


def test_diagnosis_pairwise_orders_and_drift():
    """D4 uses pairwise orders: a clean sequence has steady orders, an approach from
    below has rising ones (the L-shape displacement)."""
    import numpy as np
    from omar_pfem.verification_agent.diagnose import pairwise_orders
    hs = [1 / 2 ** k for k in range(2, 8)]
    clean = pairwise_orders(hs, [1 + h ** 2 for h in hs])
    assert all(abs(p - 2) < 1e-6 for p in clean)
    # two power terms of opposite sign: the observed order drifts toward the leading one
    rising = pairwise_orders(hs, [1 + h ** 1.1 - 2 * h ** 1.6 for h in hs])
    assert all(np.diff([p for p in rising if p is not None]) != 0)
    # round J bug fix: the 3D ladders k = 1, 2, 3, 4, 6, 8 have no constant ratio; an exact
    # h^1.4 sequence must still give 1.4 at every triple (the old formula gave 3.28, 2.90, 0.33, 2.90)
    h3 = [1 / k for k in (1, 2, 3, 4, 6, 8)]
    assert all(abs(p - 1.4) < 1e-8 for p in pairwise_orders(h3, [1 + h ** 1.4 for h in h3]))


def test_capability_gap_hypothesis():
    """C1: a check that reports it lacks the theory is a framework gap, not evidence
    about the quantity; it must not be read as non-convergence."""
    from types import SimpleNamespace
    from omar_pfem.verification_agent.diagnose import SUPPORTED, NOT_APPLICABLE, t_capability_gap
    c = SimpleNamespace(gates={
        "qoi_convergence": {"verdict": "PASS", "evidence": "rate"},
        "singular_order": {"verdict": "INCONCLUSIVE",
                           "reason": "theory implemented for degree-1 elements only"}})
    r = t_capability_gap(c)
    assert r["status"] == SUPPORTED and "lack of a check" in r["evidence"]
    c2 = SimpleNamespace(gates={"qoi_convergence": {"verdict": "PASS", "evidence": "rate"}})
    assert t_capability_gap(c2)["status"] == NOT_APPLICABLE


def test_derived_facts_never_guess():
    """Facts derived for a problem without hand-written ones: known items from the
    specification/adapter, unknown items make tests ask for data."""
    from omar_pfem.verification_agent.diagnose import derived_facts
    f = derived_facts("thick_cylinder_3d")
    assert f["degree"] == 2 and f["quadrature"] == "reduced" and f["nu"] == 0.45
    # round J: the code check now reads the stored Q20 MMS (followup_hooks); a problem
    # with no stored check still gets None, never a guessed value
    assert f["corners"] == [] and f["code_check"]()[0] is True
    from omar_pfem.verification_agent.followup_hooks import code_check, tolerance
    assert code_check("no_such_problem")[0] is None
    assert tolerance("thick_cylinder_3d")[0] < 1e-8
    g = derived_facts("l_shape")
    assert g["corners"] and abs(g["corners"][0]["lam"] - 0.5445) < 1e-3


def test_followup_respects_budget():
    """The autonomous follow-up runs a level only within its time/memory budget."""
    from omar_pfem.verification_agent.followup import plan_next_level
    p = plan_next_level("l_shape", budget_s=1e-3, budget_gb=1e-3)
    assert p["within_budget"] is False and p["n_next"] >= 256


def test_detf_guard_resolves_and_marks():
    """Round J (post hoc switch VERIF_POSTHOC_DETF): off by default; when on, an
    inverted converged solve is re-solved with 4x increments, and still-inverted rows
    are marked inadmissible."""
    import os
    from omar_pfem.verification_agent.detf_guard import guarded
    calls = []

    def fake(k, nsteps=10):
        calls.append(nsteps)
        return {"min_detF": -1.0 if nsteps < 40 else 1.0}

    os.environ.pop("VERIF_POSTHOC_DETF", None)
    assert guarded(fake, 4)["min_detF"] == -1.0 and calls == [10]
    os.environ["VERIF_POSTHOC_DETF"] = "1"
    try:
        r = guarded(fake, 4)
        assert calls == [10, 10, 40] and r["detf_inadmissible"] is False
        assert r["detf_guard"]["first_min_detF"] == -1.0
        r2 = guarded(lambda k, nsteps=10: {"min_detF": -0.5}, 4)
        assert r2["detf_inadmissible"] is True
    finally:
        os.environ.pop("VERIF_POSTHOC_DETF", None)


def test_report_text_forms():
    """Round J: the report text check looks for the printed value in the forms the report
    uses (thousands separator, Unicode minus or 'below', x 10^n with superscripts)."""
    from omar_pfem.verification_agent.report_text_check import forms, found
    assert "109,395" in forms(109395, 0)
    assert "0.5445" in forms(0.5445, 0.0005)            # typed digits, not only tol digits
    assert "1.1 × 10⁻⁵" in forms(1.1e-5, 0.05e-5)
    assert found("P1 is 0.55% below the limit", -0.55, 0.005) == "0.55"
    assert found("value 10.55 here", 0.55, 0.005) is None   # no match inside a longer number


def test_remedy_corner_analysis_and_plan():
    """Round K: the corner analysis finds every rate-limiting corner of the L-shape and the
    crack from the polygon alone, and the plan grades with g > p / lam."""
    from omar_pfem.verification_agent.remedy import (POLYGONS, corner_analysis, graded_coords,
                                                     plan_grading, plan_problem)
    import numpy as np
    assert plan_grading(0.5444837367824591, 2)["g"] == 3.7
    assert plan_grading(0.5, 2)["g"] == 4.1                 # strictly above p / lam = 4
    sing = [c for c in corner_analysis(POLYGONS["l_shape"], 0.3) if c["rate_limiting"]]
    assert sorted((c["x"], c["y"]) for c in sing) == [(0, 0), (1, 1), (2, 0)]
    re = next(c for c in sing if (c["x"], c["y"]) == (1, 1))
    assert abs(re["angle_deg"] - 270) < 1e-9 and abs(re["rate_limiting"][0] - 0.5445) < 1e-4
    tip = next(c for c in corner_analysis(POLYGONS["edge_crack"], 0.3) if c["x"] == 1)
    assert tip["angle_deg"] == 360 and tip["rate_limiting"] == [0.5, 1.5]
    p = plan_problem(POLYGONS["l_shape"], 0.3)
    assert p["x_targets"] == [0, 1, 2] and p["y_targets"] == [0, 1]
    x = graded_coords(8, 3.0, {0, 1, 2})
    assert len(x) == 17 and np.all(np.diff(x) > 0) and x[8] == 1.0


def test_exact_corner_solution():
    """Round K: the exact solution of the singular benchmark is traction-free on the corner
    edges and its energy agrees computed two independent ways."""
    import json
    import os
    from omar_pfem.verification_agent import lshape_exact as X
    c = X.self_checks()
    assert c["traction_on_free_edges_rel"] < 1e-12
    assert max(c["smallest_singular_value"]) < 1e-12
    ex = json.load(open(os.path.join(X.HERE, "lshape_exact_values.json")))
    assert ex["energy_two_ways_rel"] < 1e-12


def test_calibration_framework_covers_true_error():
    """Round K: on every quantity with an exact value, the framework's certified interval
    contains it (stored result of calibration.py)."""
    import json
    import os
    from omar_pfem.verification_agent.calibration import OUT
    s = json.load(open(OUT))["summary"]
    assert s["framework_covered"] == s["framework_certified"] >= 12
    assert s["gci_covered"] < s["gci_certified"]            # GCI made false certifications


def test_second_code_tangent():
    """Round K: the second code's tangent tensor equals the derivative formula and the
    finite-difference derivative of its first Piola-Kirchhoff stress."""
    import numpy as np
    from omar_pfem.verification_agent import independent_block3d as B
    rng = np.random.default_rng(0)
    F = np.eye(3)[:, :, None, None] + 0.1 * rng.standard_normal((3, 3, 2, 4))
    dF = rng.standard_normal((3, 3, 2, 4))
    assert np.abs(np.einsum("ijkl...,kl...->ij...", B.tangent(F), dF) - B.dpk1(F, dF)).max() < 1e-10
    e = 1e-7
    fd = (B.pk1(F + e * dF) - B.pk1(F - e * dF)) / (2 * e)
    assert np.abs(fd - B.dpk1(F, dF)).max() / np.abs(B.dpk1(F, dF)).max() < 1e-7


def test_intensity_plan_reproduces():
    """Round L: the intensity-aware plan is a deterministic function of the uniform family
    (stored ladder): it grades the corners with share >= 0.10, each with g > p / lam."""
    import json
    import os
    from omar_pfem.verification_agent import ushape as U
    from omar_pfem.verification_agent.remedy_intensity import coords, plan_intensity
    here = os.path.dirname(U.__file__)
    rows = [r for r in json.load(open(os.path.join(here, "ushape_ladder.json")))
            if r["variant"] == "uniform"]
    p = plan_intensity(U.POLYGON, U.NU, rows)
    stored = json.load(open(os.path.join(here, "ushape_plan.json")))
    assert {k: {str(a): b for a, b in v.items()} for k, v in p["plan"].items()} == stored["plan"]
    assert all(c["graded"] == (c["share"] >= 0.10) for c in p["corners"])
    x = coords(8, {1: 3.7}, 0, 3)
    assert len(x) == 25 and x[8] == 1.0 and (x[1:] > x[:-1]).all()


def test_detf_check_level_guard():
    """Round M (post hoc, VERIF_POSTHOC_DETF): an inverted strategy row is replaced by its
    admissible re-solve; without one the check is INCONCLUSIVE; with the switch off the
    strategy values are unchanged."""
    import os
    from omar_pfem.verification_agent.detf_guard import admissible_strategies

    class A:
        geometry = "no_such_problem"
        strategy_rows = [{"k": 4, "grading": 1.0, "min_detF": 1.0, "e": 1.0},
                         {"k": 4, "grading": 2.0, "min_detF": -1.0, "e": 9.0}]
    s = ([1.0, 9.0], None)
    r = admissible_strategies(A(), "e", s)
    assert isinstance(r, str) and "inadmissible" in r
    A.strategy_rows[1]["min_detF"] = 0.5
    assert admissible_strategies(A(), "e", s) == s
    from omar_pfem.verification_agent.orchestrate import ADAPTERS, run_check
    from omar_pfem.verification_agent import trees
    ad = ADAPTERS["thick_cylinder_3d"]()
    p = trees.check_params("strategy_robustness", "u_outer_radial", "thick_cylinder_3d")
    os.environ.pop("VERIF_POSTHOC_DETF", None)
    assert run_check(ad, "strategy_robustness", "u_outer_radial", p)["verdict"] == "FAIL"
    os.environ["VERIF_POSTHOC_DETF"] = "1"
    try:
        assert run_check(ad, "strategy_robustness", "u_outer_radial", p)["verdict"] == "PASS"
    finally:
        os.environ.pop("VERIF_POSTHOC_DETF", None)


def test_switch_exemption_is_checked():
    """Round M: the frozen-file exemption accepts orchestrate.py only with the switch blocks
    stripped; another switch name (negative control) is refused."""
    from omar_pfem.verification_agent.prereg_util import unchanged_except_switch
    assert unchanged_except_switch("orchestrate.py", "b850330", "VERIF_POSTHOC_DETF")
    assert not unchanged_except_switch("orchestrate.py", "b850330", "VERIF_NOT_A_SWITCH")


def test_round_m_excludes_degenerate_runs():
    """Round M: a run in which the LLM chose no quantity is excluded from the means."""
    from omar_pfem.verification_agent.round_m import ablation
    a = ablation()
    assert a["degenerate_runs"] == 1 and a["problems"]["crack"]["valid_runs"] == 2
    assert a["runs"] == 21


# ---- adopted rule set, 24 Nov 2026 (advisor decisions Q1, Q2) ---------------------------
def _sing(observed, expected=1.0, tol=0.15, mode="lower_bound"):
    return {"gate": "singular_order", "verdict": "FAIL", "observed": observed,
            "expected": expected, "mode": mode, "tol": tol, "reason": "r"}


def test_adopted_below_rising_is_inconclusive():
    from omar_pfem.verification_agent.adopted import below_rule
    hs = [1 / 2 ** k for k in range(5)]
    # error ~ h^p with p rising 0.4 -> 0.7 -> 0.85 -> 0.95 toward 1
    errs, e = [], 1.0
    for p in (0.0, 0.4, 0.7, 0.85, 0.95):
        e *= 2 ** -p
        errs.append(e)
    vals = [1 + x for x in errs]
    g = below_rule(_sing(0.6), hs, vals)
    assert g["verdict"] == "INCONCLUSIVE" and "from below" in g["label"]


def test_adopted_below_not_rising_stays_fail():
    from omar_pfem.verification_agent.adopted import below_rule
    hs = [1 / 2 ** k for k in range(5)]
    vals = [1 + 2 ** (-0.5 * k) for k in range(5)]          # steady order 0.5 < 1
    assert below_rule(_sing(0.5), hs, vals)["verdict"] == "FAIL"


def test_adopted_above_and_peak_untouched():
    from omar_pfem.verification_agent.adopted import below_rule
    hs = [1 / 2 ** k for k in range(5)]
    vals = [1 + 2 ** (-0.5 * k) for k in range(5)]
    assert below_rule(_sing(1.8, mode="equal"), hs, vals)["verdict"] == "FAIL"
    assert below_rule(_sing(-0.1, expected=-0.3, mode="equal"), hs, vals)["verdict"] == "FAIL"


def test_adopted_absolute_targets_qoi_dependent():
    from omar_pfem.verification_agent import requirements
    from omar_pfem.verification_agent.adopted import ruleset
    with ruleset():
        assert requirements.absolute_target("strain_energy")[0] == 0.01
        assert requirements.absolute_target("u_tip")[0] == 0.01
        assert requirements.absolute_target("gk_avg_von_mises")[0] == 0.05
        assert requirements.absolute_target("peak_von_mises")[0] is None
    assert requirements.absolute_target("strain_energy")[0] is None      # default unchanged
    assert abs(requirements.constitutive_requirement(0.06)[0] - 0.02) < 1e-15


# ---- round O: autonomous workflow ---------------------------------------------------------
def test_symmetry_edge_exponents_match_mirror_argument():
    import numpy as np
    from omar_pfem.verification_agent.corners_ext import exponents
    assert not [l for l in exponents(np.pi / 2, "free", "symmetry", 0.3) if abs(l - round(l)) > 1e-6]
    assert abs(min(exponents(np.pi, "free", "symmetry", 0.3)) - 0.5) < 1e-8


def test_complex_exponent_of_clamped_free_corner():
    import numpy as np
    from omar_pfem.verification_agent.corners_ext import complex_exponents
    re_, im = complex_exponents(np.pi, "clamped", "free", 0.3)[0]
    assert abs(re_ - 0.5) < 1e-6 and abs(im - np.log(3 - 4 * 0.3) / (2 * np.pi)) < 1e-6
    assert abs(complex_exponents(1.5 * np.pi, "clamped", "free", 0.3)[0][0] - 0.3417) < 1e-3


def test_pipeline2d_reproduces_generic2d():
    import json, os
    from omar_pfem.verification_agent.generic2d import Problem
    from omar_pfem.verification_agent.pipeline2d import Problem2D
    d = json.load(open(os.path.join(os.path.dirname(__file__), "problems2d", "t_bracket.json")))
    a, b = Problem2D(d).solve(4), Problem(d).solve(4)
    assert abs(a["strain_energy"] / b["strain_energy"] - 1) < 1e-12
    assert abs(a["v_left_tip"] / b["v_left_tip"] - 1) < 1e-12


def test_experiment_freeze_detects_tampering(tmp_path, monkeypatch):
    import json
    import pytest
    from omar_pfem.verification_agent import experiment as X
    monkeypatch.setattr(X, "EXP_DIR", str(tmp_path))
    monkeypatch.setattr(X, "MEMORY", str(tmp_path / "mem.json"))
    path, rec = X.freeze("p", 1, {"action": "a", "situation": "s", "quantities": ["q"],
                                  "prediction": {"text": "t", "criterion": "c"},
                                  "cost": {"minutes": 1.0}})
    d = json.load(open(path))
    d["preregistration"]["prediction"]["text"] = "changed"
    json.dump(d, open(path, "w"))
    with pytest.raises(RuntimeError):
        X.complete(path, {"confirmed": True})


def test_untraced_numbers_are_rejected():
    from omar_pfem.verification_agent.autonomous import trace_numbers
    d = {"name": "x", "cell_size": 10.0, "cells": [[0, 0], [1, 0]], "E": 210000.0, "nu": 0.3,
         "plane": "stress", "bcs": [{"segment": [[0, 0], [0, 10]], "type": "clamped"},
                                    {"segment": [[20, 0], [20, 10]], "type": "traction",
                                     "t": [77.0, 0.0]}], "points": {}}
    text = "A 20 by 10 mm plate, E = 210000 MPa, nu = 0.3, clamped left, traction 50 MPa."
    unsup, flagged = trace_numbers(d, text, [])
    assert any("77" in u for u in unsup) and not flagged


# ---- neo-Hookean in the loop, literature search (2026-11-25) -------------------------------
def test_neo_hookean_refuses_follower_pressure_and_plane_stress():
    from omar_pfem.verification_agent.pipeline2d_nh import validate
    d = {"name": "x", "cell_size": 1.0, "cells": [[0, 0], [1, 0]], "E": 1.0, "nu": 0.3,
         "plane": "stress", "bcs": [{"segment": [[0, 0], [0, 1]], "type": "clamped"},
                                    {"segment": [[2, 0], [2, 1]], "type": "pressure", "p": 1.0}]}
    e = validate(d)
    assert any("plane strain" in x for x in e) and any("follower" in x for x in e)


def test_neo_hookean_small_load_matches_linear():
    from omar_pfem.verification_agent.pipeline2d import Problem2D
    from omar_pfem.verification_agent.pipeline2d_nh import Problem2DNH
    d = {"name": "x", "cell_size": 1.0, "cells": [[0, 0], [1, 0]], "E": 1000.0, "nu": 0.3,
         "plane": "strain", "bcs": [{"segment": [[0, 0], [0, 1]], "type": "clamped"},
                                    {"segment": [[2, 0], [2, 1]], "type": "traction",
                                     "t": [0.0, 0.01]}], "points": {"v_tip": [2, 1, "y"]}}
    a, b = Problem2DNH(d).solve(4), Problem2D(d).solve(4)
    assert abs(a["v_tip"] / b["v_tip"] - 1) < 1e-3      # finite strain -> linear as load -> 0


def test_literature_code_screen_excludes_shells():
    from omar_pfem.verification_agent.literature_search import code_screen
    ok, why = code_screen({"title": "A shell element", "abstract": "finite element benchmark "
                           "for Reissner-Mindlin shell elasticity, numerical examples"})
    assert not ok and why["excluded_by"]


# ---- general 2D geometry (geometry2d, 25 Nov) ----
def _load(name):
    import json, os
    return json.load(open(os.path.join(os.path.dirname(__file__), "problems2d", name)))


def test_geometry_validate_rejects_point_outside_domain():
    from omar_pfem.verification_agent.geometry2d import validate
    d = _load("plate_hole_quarter.json")
    d["points"] = {"bad": [0.2, 0.2, "x"]}           # inside the hole
    assert validate(d)


def test_cook_membrane_matches_literature_at_mid_edge():
    """Cook's membrane, plane stress, E = 1, nu = 1/3: the reference tip value 23.96 is the
    vertical displacement at the mid-point of the loaded edge (48, 52)."""
    from omar_pfem.verification_agent.geometry2d import make
    r = make(_load("cook_linear.json")).solve(32, element="Q2")
    assert abs(r["v_C"] - 23.96) < 0.01


def test_plate_with_hole_has_no_singular_corners_and_converges_regularly():
    """Corners come from the real gmsh boundary: where the hole meets a symmetry edge the angle
    is 90 deg with symmetry on one side and a free curve on the other -> no singularity.
    Self-similar refinement: energy differences drop at least 4x per level."""
    from omar_pfem.verification_agent.geometry2d import make
    p = make(_load("plate_hole_quarter.json"))
    assert p.corners == []
    e = [p.solve(n, element="Q2")["strain_energy"] for n in (4, 8, 16)]
    assert abs(e[2] - e[1]) < abs(e[1] - e[0]) / 4


def test_ladder_cache_is_invalidated_when_description_changes(tmp_path, monkeypatch):
    """A cached ladder made for another description (or other code) must not be reused."""
    from omar_pfem.verification_agent import pipeline2d as P
    monkeypatch.setattr(P, "RUNS", str(tmp_path))
    d = {"name": "c", "cell_size": 1.0, "cells": [[0, 0], [1, 0]], "E": 1.0, "nu": 0.3,
         "plane": "strain", "bcs": [{"segment": [[0, 0], [0, 1]], "type": "clamped"},
                                    {"segment": [[2, 0], [2, 1]], "type": "traction",
                                     "t": [0.0, 0.01]}]}
    a = P.Ladders(P.Problem2D(d), log=lambda *_: None).get("Q2 uniform", (2,), None, "Q2")[0]
    d["E"] = 2.0
    b = P.Ladders(P.Problem2D(d), log=lambda *_: None).get("Q2 uniform", (2,), None, "Q2")[0]
    assert abs(b["strain_energy"] / a["strain_energy"] - 0.5) < 1e-9
    assert any(f.name.startswith("ladders_stale_") for f in (tmp_path / "c").iterdir())


# ---- sensitivity of assumed inputs (25 Nov) ----
def _beam():
    return {"name": "sens_beam", "cell_size": 1.0, "cells": [[0, 0], [1, 0]], "E": 1000.0,
            "nu": 0.3, "plane": "strain",
            "bcs": [{"segment": [[0, 0], [0, 1]], "type": "clamped"},
                    {"segment": [[2, 0], [2, 1]], "type": "traction", "t": [0.0, 1.0]}],
            "points": {"v_tip": [2, 1, "y"]}}


def test_sensitivity_recovers_exact_scalings(tmp_path, monkeypatch):
    """Linear, zero-displacement + traction: displacement ~ t / E, energy ~ t^2 / E."""
    from omar_pfem.verification_agent import sensitivity as S
    monkeypatch.setattr(S, "OUT", str(tmp_path))
    monkeypatch.setattr(S, "LEVELS", (2, 4))
    r = S.analyse(_beam(), ["E", "bcs[1].t"], None, log=lambda *_: None)["params"]
    assert abs(r["E"]["quantities"]["v_tip"]["S_log_posthoc"] + 1) < 1e-9
    assert abs(r["E"]["quantities"]["v_tip"]["S"] + 1 / (1 - S.REL ** 2)) < 1e-9
    assert abs(r["bcs[1].t"]["quantities"]["strain_energy"]["S_log_posthoc"] - 2) < 1e-9


def test_sensitivity_decision_against_certified_uncertainty(tmp_path, monkeypatch):
    from omar_pfem.verification_agent import sensitivity as S
    monkeypatch.setattr(S, "OUT", str(tmp_path))
    monkeypatch.setattr(S, "LEVELS", (2, 4))
    cert = {"v_tip": {"U": 1e-9, "limit": 1.0}}
    r = S.analyse(_beam(), ["E", "vertex x"], cert, log=lambda *_: None)["params"]
    assert r["E"]["needs_person"] and r["E"]["matters_for"] == ["v_tip"]
    assert r["vertex x"]["needs_person"]                   # cannot be perturbed automatically
    cert = {"v_tip": {"U": 0.5, "limit": 1.0}}              # 50% numerical uncertainty
    r = S.analyse(_beam(), ["E"], cert, log=lambda *_: None)["params"]
    assert not r["E"]["needs_person"]


def test_hole_radius_perturbation_moves_points_on_the_hole():
    from omar_pfem.verification_agent import sensitivity as S
    from omar_pfem.verification_agent.geometry2d import validate
    d, step = S.perturbed(_load("plate_hole_quarter.json"), "hole[0].radius", +1)
    assert abs(step - 0.1) < 1e-12 and d["points"]["v_hole_top"][:2] == [0.0, 1.1]
    assert d["bcs"][0]["segment"][0] == [1.1, 0.0] and validate(d) == []


def test_experiment_memory_counts_a_repeat_once_and_records_are_not_overwritten(tmp_path, monkeypatch):
    from omar_pfem.verification_agent import experiment as X
    monkeypatch.setattr(X, "MEMORY", str(tmp_path / "m.json"))
    monkeypatch.setattr(X, "EXP_DIR", str(tmp_path))
    e = {"problem": "p", "action": "a", "situation": "s", "quantities": ["q"], "confirmed": True}
    X.remember(dict(e)); X.remember(dict(e)); X.remember(dict(e, confirmed=False))
    assert len(X.memory()) == 2
    rec = {"action": "a", "situation": "s", "quantities": ["q"], "cost": {"minutes": 1}}
    p1, _ = X.freeze("p", 1, rec)
    p2, _ = X.freeze("p", 1, rec)
    assert p1 != p2 and p2.endswith("_run2.json")


def test_sensitivity_treats_certified_U_as_relative(tmp_path, monkeypatch):
    """The framework's U is relative (gates.uncertainty_fine). With limit 100 and U 0.5 (50%),
    a 10% change of E is negligible; dividing U by |limit| again (the first version) made it
    'matters'."""
    from omar_pfem.verification_agent import sensitivity as S
    monkeypatch.setattr(S, "OUT", str(tmp_path))
    monkeypatch.setattr(S, "LEVELS", (2, 4))
    r = S.analyse(_beam(), ["E"], {"v_tip": {"U": 0.5, "limit": 100.0}}, log=lambda *_: None)
    assert r["params"]["E"]["quantities"]["v_tip"]["U_rel"] == 0.5
    assert not r["params"]["E"]["needs_person"]


def test_run_report_is_built_from_stored_files_only():
    """The per-run report states the stored certified values (relative U turned into +- abs)."""
    import json, os, re
    from omar_pfem.verification_agent import run_report
    here = os.path.dirname(__file__)
    h, _ = run_report.build("cook_linear", pdf=False)
    page = open(h).read()
    c = json.load(open(os.path.join(here, "autonomous_runs", "cook_linear", "run.json")))["certified_info"]
    for q, x in c.items():
        u = abs(x.get("U_rel", x["U"]))
        assert f"{x['limit']:.6g} &plusmn; {u * abs(x['limit']):.3g}" in page, q
    assert "&amp;lambda;" not in page


# ---- 3D extruded problems (geometry3d, 25 Nov) ----
def _load3(name):
    import json, os
    return json.load(open(os.path.join(os.path.dirname(__file__), "problems3d", name)))


def test_3d_validate_rejects_bad_points_and_free_z_motion():
    from omar_pfem.verification_agent.geometry3d import validate
    d = _load3("plate_hole_quarter_3d_pe.json")
    d["points"] = {"mid": [10, 0, 0.5, "x"]}                 # not on a z face
    assert any("z must be 0" in e for e in validate(d))
    d = _load3("plate_hole_quarter_3d_pe.json")
    d["bcs"] = [b for b in d["bcs"] if "face" not in b]       # nothing holds z
    assert any("rigid-body" in e for e in validate(d))


def test_3d_meshes_fill_the_domain_and_match_the_2d_section():
    """The volume guard (elements silently dropped on loading, found 25 Nov) and the layered
    structure: the Q2 hexahedral mesh is the 2D Q2 mesh of the same level times the layers."""
    import skfem as fem
    from omar_pfem.verification_agent.geometry2d import make
    p3 = make(_load3("cook_3d.json"))
    p2 = make(_load("cook_linear.json"))
    for n in (2, 4):
        for el in ("Q2", "P2"):
            m, e = p3.build_mesh(n, None, el)
            assert abs(fem.Basis(m, e, intorder=4).dx.sum() / p3.volume() - 1) < 1e-6
        assert p3.build_mesh(n, None, "Q2")[0].t.shape[1] == \
            p2.build_mesh(n, None, "Q2")[0].t.shape[1] * p3.n_layers(n)


def test_2d_meshes_fill_the_domain():
    import skfem as fem
    from omar_pfem.verification_agent.geometry2d import make
    for f in ("cook_linear.json", "plate_hole_quarter.json"):
        p = make(_load(f))
        for n in (2, 4, 8):
            for el in ("Q2", "P2"):
                m, e = p.build_mesh(n, None, el)
                assert abs(fem.Basis(m, e, intorder=4).dx.sum() / p.area() - 1) < 1e-4, (f, n, el)


def test_3d_cook_singular_edges():
    from omar_pfem.verification_agent.geometry2d import make
    p = make(_load3("cook_3d.json"))
    assert len(p.corners) == 3
    assert {(e["face"], e["side"]) for e in p.face_edges} == {("z0", "clamped"), ("z1", "clamped")}
    assert p.lam_min < 1



# ---- round Q (25 Nov) ----
def test_code_verification_classes():
    from omar_pfem.verification_agent.mms_auto import classify
    assert classify("PASS", {"L2": [1, 2]}) == "PASS"
    assert classify("FAIL", {"L2": [1, .2, .03], "H1": [1, .5, .2]}) == "INCONCLUSIVE"
    assert classify("FAIL", {"L2": [1, .2, .3], "H1": [1, .5, .2]}) == "FAIL"


def test_reference_domain_keeps_the_condition_types():
    from omar_pfem.verification_agent.reference_domain import reference_description
    ref, _ = reference_description(_load3("cook_3d.json"))
    assert sorted(b["type"] for b in ref["bcs"]) == ["clamped", "traction"]
    ref, _ = reference_description(_load3("plate_hole_quarter_3d_pe.json"))
    kinds = sorted((b.get("face", "side"), b["type"]) for b in ref["bcs"])
    assert kinds == [("side", "symmetry"), ("side", "symmetry"), ("side", "traction"),
                     ("z0", "symmetry"), ("z1", "symmetry")] and ref["domain"]["holes"]


def test_nh3d_body_force_matches_finite_differences_of_P():
    import numpy as np
    from omar_pfem.verification_agent.geometry2d import make
    m = make(_load3("nh_plate_hole_3d_pe.json")).manufactured()
    x, y, z, h = np.array([[3.3]]), np.array([[4.1]]), np.array([[0.4]]), 1e-5
    b = [float(np.ravel(v)[0]) for v in m.body(x, y, z)]
    for i in range(3):
        s = 0
        for j, d in enumerate(np.eye(3) * h):
            s += (m._P(x + d[0], y + d[1], z + d[2], m.amp)[0][i, j]
                  - m._P(x - d[0], y - d[1], z - d[2], m.amp)[0][i, j]) / (2 * h)
        assert abs(b[i] + float(np.ravel(s)[0])) < 1e-6 * max(1, abs(b[i]))


def test_nh3d_small_load_equals_linear_3d():
    from omar_pfem.verification_agent.geometry2d import make
    d = _load3("nh_plate_hole_3d_pe.json")
    for bc in d["bcs"]:
        if bc["type"] == "traction":
            bc["t"] = [v * 1e-4 for v in bc["t"]]
    lin = dict(d, material="linear_elastic")
    a, b = make(d).solve(2, element="Q2"), make(lin).solve(2, element="Q2")
    assert abs(a["u_right"] / b["u_right"] - 1) < 1e-3


def test_cluster_import_refuses_wrong_md5_and_other_code(tmp_path, monkeypatch):
    import json, hashlib, pytest
    from omar_pfem.verification_agent import cluster_ladder as C
    monkeypatch.setattr(C, "JOBS", str(tmp_path))
    d = _load3("cook_3d.json")
    json.dump({"description": d}, open(tmp_path / "x_spec.json", "w"))
    f = tmp_path / "ret.json"
    json.dump({"name": "x", "fingerprint": "not-this-code", "rows": {}}, open(f, "w"))
    with pytest.raises(SystemExit, match="md5"):
        C.import_result(str(f), "0" * 32)
    md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
    with pytest.raises(SystemExit, match="changed since the job"):
        C.import_result(str(f), md5)


def test_campaign_index_builds():
    from omar_pfem.verification_agent import campaign
    f = campaign.index(pdf=False)
    assert "Open questions" in open(f).read()


def test_cluster_mms_import_only_under_local_keys(tmp_path, monkeypatch):
    import json, pytest
    from omar_pfem.verification_agent import cluster_ladder as C, mms_auto
    from omar_pfem.verification_agent.geometry2d import make
    monkeypatch.setattr(C, "JOBS", str(tmp_path))
    monkeypatch.setattr(mms_auto, "RECORD", str(tmp_path / "rec.json"))
    d = _load3("cook_3d.json")
    k = mms_auto.key(make(d), "Q2")
    json.dump({"checks": [{"description": d, "suffix": ""}], "elements": ["Q2"]},
              open(tmp_path / "m_spec.json", "w"))
    good = {"verdict": "PASS", "levels": [2, 4, 8, 16, 32]}
    r = C._import_mms({"name": "m", "results": {k: good, "other-code|Q2": good}})
    assert r["added"] == [(k, "PASS")] and r["refused"] == ["other-code|Q2"]
    with pytest.raises(SystemExit, match="PASS is on record"):          # never downgrade a PASS
        C._import_mms({"name": "m", "results": {k: {"verdict": "FAIL"}}})
    assert json.load(open(tmp_path / "rec.json"))[k]["verdict"] == "PASS"
    with pytest.raises(SystemExit):
        C._import_mms({"name": "m", "results": {"x|Q2": good}})


def test_cluster_mms_record_not_repeated_or_overwritten(tmp_path, monkeypatch):
    """26 Nov (after job 3322812): a re-run must not replace the cluster's n = 32 check by a
    local n <= 16 one, nor ask for the same cluster job again."""
    import json
    from omar_pfem.verification_agent import mms_auto
    from omar_pfem.verification_agent.geometry2d import make
    monkeypatch.setattr(mms_auto, "RECORD", str(tmp_path / "rec.json"))
    prob = make(_load3("cook_3d.json"))
    k = mms_auto.key(prob, "Q2")
    cl = {"verdict": "INCONCLUSIVE", "levels": [2, 4, 8, 16, 32], "rates": {"L2": {"observed": 3.79}},
          "levels_added": ["computed on the cluster"], "problem": "cook_3d"}
    json.dump({k: cl}, open(tmp_path / "rec.json", "w"))
    need = mms_auto.needed(prob)
    assert need["Q2"][0] is False and "cluster" in need["Q2"][1] and need["P2"][0] is True
    out = mms_auto.run(prob, elements=("Q2",), log=lambda s: None)
    assert out["Q2"]["skipped"] and out["Q2"]["levels"] == cl["levels"]
    assert not any("refused" in s for s in out["Q2"]["levels_added"])   # -> no new cluster job
    rec = json.load(open(tmp_path / "rec.json"))
    mms_auto._store(rec, k, {"verdict": "INCONCLUSIVE", "levels": [2, 4, 8, 16]})
    assert json.load(open(tmp_path / "rec.json"))[k]["levels"] == cl["levels"]


def test_cached_cluster_level_costs_nothing():
    """26 Nov: a level already in the cache (returned from the cluster) costs 0 minutes; an
    absent level is grown from the finest level below it, not from the finest stored one."""
    from omar_pfem.verification_agent.experiment import _level_cost
    rows = [{"n": n, "n_dof": 10 * 8 ** i, "wall_clock_s": 1.0} for i, n in enumerate((2, 4, 8, 16, 32))]
    cm = lambda dof: dof / 1e3
    assert _level_cost(rows, 32, 8, cm) == (rows[-1]["n_dof"], 0.0)
    assert _level_cost(rows[:4], 32, 8, cm)[0] == 8 * rows[3]["n_dof"]
    assert _level_cost(rows, 64, 8, cm)[0] == 8 * rows[-1]["n_dof"]


def test_machine_refused_experiment_is_not_evidence(tmp_path, monkeypatch):
    """26 Nov: an experiment refused by the local memory guard never ran; it must not lower the
    prior of its action (it used to be remembered as refuted)."""
    import json
    from omar_pfem.verification_agent import experiment as X
    monkeypatch.setattr(X, "MEMORY", str(tmp_path / "mem.json"))
    f = tmp_path / "e.json"
    rec = {"problem": "p", "action": "a", "situation": "s", "quantities": ["q"],
           "cost": {"minutes": 1.0}}
    json.dump(rec, open(f, "w"))
    monkeypatch.setattr(X, "_verify_frozen", lambda path: rec)
    X.complete(str(f), {"confirmed": False, "failed": "level n = 16 needs about 16.6 GB > 13 GB: "
                                                      "too large for this machine"})
    assert X.prior("a", "s") == (0.5, 0)
    X.complete(str(f), {"confirmed": False, "failed": "J<=0"})
    assert X.prior("a", "s") == (1 / 3, 1)


def test_cluster_paste_is_rebuilt_byte_for_byte():
    """26 Nov: the pasted cluster print (prompt lines around it, CRLF line ends, the newline that
    echo adds) is rebuilt to the exact bytes; a changed file is detected by its md5."""
    import hashlib
    from omar_pfem.verification_agent.cluster_ladder import parse_paste, print_command
    a, b = '{\n "x": 1\n}', '{\n "y": [\n  2\n ]\n}\n'
    md = lambda t: hashlib.md5(t.encode()).hexdigest()
    paste = ("(base) $ " + print_command(["cluster_jobs/a_result.json"]) + "\r\n"
             f"=== a_result.json {md(a)}\r\n{a}\r\n\r\n=== end\r\n"
             f"=== b_result.json {md(b)}\r\n{b}\r\n=== end\r\n"
             f"=== c_result.json {md(a)}\r\n{a} \r\nX\r\n\r\n=== end\r\n(base) $ ")
    got = {n: d for n, _, d in parse_paste(paste)}
    assert got["a_result.json"] == a.encode() and got["b_result.json"] == b.encode()
    assert got["c_result.json"] is None


def test_nearly_incompressible_description_is_refused_by_code():
    """26 Nov: nu > 0.49 is refused by code (displacement elements lock); it is not sent back to
    the LLM for correction, which could change nu silently."""
    from omar_pfem.verification_agent import autonomous as A

    class Log:
        steps = []

        def step(self, *a, **k):
            self.steps.append(a)
    desc = {"name": "cook_nearly_incompressible", "material": "linear_elastic",
            "geometry_kind": "polygon", "cell_size": 1, "cells": [],
            "outer": [[0, 0], [48, 44], [48, 60], [0, 44]], "holes": [], "E": 1.0, "nu": 0.4999,
            "plane": "strain", "bcs": [], "points": []}

    class B:
        def ask(self, role, prompt, schema):
            return {"supported": True, "unsupported_reason": "", "description": desc,
                    "sources": [], "assumptions": []}
    import omar_pfem.verification_agent.llm_roles as R
    orig = R.formalize
    R.formalize = lambda b, t: ({}, [])
    try:
        d, m = A.define(B(), "text", Log())
    finally:
        R.formalize = orig
    assert d is None and m["supported"] is False and "incompressible" in m["unsupported_reason"]


def test_scout_prompt_contains_the_paper():
    """26 Nov: an edit dropped the f-prefix and the scout received '{c['title']}' instead of the
    paper (round-2 search attempt 1, all scouts empty). The prompt must carry title and passage."""
    from omar_pfem.verification_agent import literature_search as L
    seen = {}

    class B:
        def ask(self, role, prompt, schema):
            seen["p"] = prompt
            return {"likely_representable": False, "example": "", "reason": ""}
    L.scout(B(), {"title": "A plate with a hole", "text_screen": {"passage": "E = 210 GPa"}},
            "linear_elastic")
    assert "A plate with a hole" in seen["p"] and "E = 210 GPa" in seen["p"]
    assert "{c[" not in seen["p"] and "{CAPABILITIES" not in seen["p"] and "{DOMAIN" not in seen["p"]
