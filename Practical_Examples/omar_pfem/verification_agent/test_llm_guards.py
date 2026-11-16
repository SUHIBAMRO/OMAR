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
