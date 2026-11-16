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
                "peak": 3.0 + math.log(1.0 / h) + 0.1 * math.log(1.0 / h) ** 2}
    return solve


def test_adaptive_stops_early_and_classifies():
    from omar_pfem.verification_agent import adaptive
    calls = []
    res = adaptive.adaptive_refine(_fake_solver(calls), ["smooth", "peak"], budget_s=1e6,
                                   log=lambda *a: None)
    assert res["decisions"]["smooth"]["verdict"] == "PASS"
    assert res["decisions"]["peak"]["reason"].startswith("divergent")
    assert calls == [11, 21, 41]  # both decided without a fourth level


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
    assert sorted(t["rule_augmented"]) == ["strategy_robustness:interior_avg_PK1_frob",
                                           "strategy_robustness:peak_PK1_frob"]
    assert t["critique"] is not None
    assert t["gate_summary"]["peak_PK1_frob"]["verdict"] == "FAIL"
    assert t["gate_summary"]["strain_energy"]["verdict"] == "PASS"
    assert t["advisor_flags"] == ["advisor says PASS for 'peak_PK1_frob' but the gate says FAIL"]
