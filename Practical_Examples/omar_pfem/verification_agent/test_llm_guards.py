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
    assert L.vocabulary(card)[1]          # compliance: vocabulary gap reported
    # a finite concentration (singular=false) must not trigger the singular-order check
    card["singular_features"] = [{"description": "hole edge", "singular": False,
                                  "inferred": False, "quote": "secant compliance"}]
    assert not any(n.startswith("singular features") for n in L.needs(card, ok))
    card["singular_features"][0]["singular"] = True
    assert any(n.startswith("singular features") for n in L.needs(card, ok))
