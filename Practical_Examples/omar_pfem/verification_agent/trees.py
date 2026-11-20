"""Problem and method trees with explicit rules (GRAFT-style, solid mechanics).

GRAFT represents scientific semantics as a problem tree, a method/action
tree, and rules linking them; this is the verification counterpart for
hyperelastic FE references:

  problem tree   regularity class x reference type x QoI class
  method tree    code verification (mms_rates) / solution verification
                 (qoi_convergence, qoi_convergence_multilevel,
                 strategy_robustness, anchored_robustness)
  rules          which checks a (problem, QoI) REQUIRES, and which verdict
                 theory PREDICTS, each with a stated reason (R1-R6 from the
                 spec; R7 from the plan: strategy robustness needs an anchor)

The rules give (i) a deterministic baseline plan with no LLM, (ii) a
completeness check every LLM plan must pass, and (iii) predictions that are
scored against the gate verdicts, so a rule that predicts wrongly is visible.
"""
import re

PROBLEM_TREE = {
    "regularity": {
        "smooth": ["smooth"],
        "singular": ["clamped_free_corner", "reentrant_corner", "crack"],
        "near_singular": ["stress_concentration", "contact"],
        "unknown": ["unknown"],
    },
    "reference": {
        "code_verification": ["exact_manufactured"],
        "solution_verification": ["mesh_refinement_family", "higher_order_reference"],
        "validation": ["experiment"],
    },
    "qoi_class": {
        "global": "integral over the whole body (energy, reaction force)",
        "displacement": "displacement at a point or its maximum",
        "region_average": "stress/strain averaged over a finite region",
        "pointwise_peak": "maximum of a stress/strain field, SCF",
        "error_norm": "error norm against an exact solution",
    },
}

METHOD_TREE = {
    "code_verification": {"mms_rates": "observed vs theoretical convergence rates"},
    "solution_verification": {
        "qoi_convergence": "stability / Richardson-GCI on one refinement family, with the "
                           "observed order required to lie in the element's asymptotic window",
        "qoi_convergence_multilevel": "least-squares power-law fit over the finest 5 levels; "
                                      "rate evidence needs monotonicity, fit quality, p in "
                                      "the window and leave-one-out stability",
        "strategy_robustness": "agreement across independent refinement strategies",
        "singular_order": "observed orders consistent with the corner singular exponent",
        "anchored_robustness": "every strategy agrees with the family's Richardson limit "
                               "(or finest value), not only with the other strategies; "
                               "inconclusive when the family itself did not converge",
    },
}

# Polynomial degree k of the displacement interpolation.
ELEMENT_DEGREE = {"Q4": 1, "T3": 1, "H8": 1, "TET4": 1,
                  "Q8": 2, "Q9": 2, "T6": 2, "Q20": 2, "H27": 2, "TET10": 2}


def order_window(element):
    """Admissible observed order for a converging FE QoI: at least 0.5
    (slower means not yet asymptotic or stagnating) and at most 2k+1, one
    above the fastest rate any standard QoI attains with degree-k elements
    (energy, O(h^2k)); a larger apparent order signals error cancellation or
    pre-asymptotic behaviour, not convergence."""
    k = ELEMENT_DEGREE.get(element)
    return None if k is None else (0.5, 2 * k + 1)


# Tolerance CEILINGS per QoI class (upper bounds). The actual tolerance of a
# QoI is check_params(): the ceiling, tightened by the requirement derived
# from the surrogate errors the reference has to resolve
# (requirements.py). Tolerances are set by code only; an LLM plan's
# parameters are replaced by these (orchestrate.py).
CLASS_CEILINGS = {
    "global": {"band_tol": 0.002, "gci_tol": 0.01, "robust_tol": 0.01},
    "displacement": {"band_tol": 0.002, "gci_tol": 0.01, "robust_tol": 0.01},
    "region_average": {"band_tol": 0.01, "gci_tol": 0.02, "robust_tol": 0.02},
    "pointwise_peak": {"band_tol": 0.01, "gci_tol": 0.02, "robust_tol": 0.02},
    "error_norm": {"band_tol": 0.01, "gci_tol": 0.02, "robust_tol": 0.02},
    "unknown": {"band_tol": 0.01, "gci_tol": 0.02, "robust_tol": 0.02},
}


def regularity_class(regularity):
    for cls, members in PROBLEM_TREE["regularity"].items():
        if regularity in members:
            return cls
    return "unknown"


def reference_class(reference):
    for cls, members in PROBLEM_TREE["reference"].items():
        if reference in members:
            return cls
    return "unknown"


_PATTERNS = [
    ("error_norm", r"(L2|H1|energy_norm|_rel$|error_norm)"),
    ("pointwise_peak", r"(peak|_max$|^max_(?!disp)|scf)"),
    ("region_average", r"(_avg|avg_|average|interior)"),
    # compliance, eigenvalues (frequencies, buckling loads) are energy-type functionals
    # of the whole solution (added 2026-11-20 for the literature agent's vocabulary gaps)
    ("global", r"(energy|reaction|force_total|compliance|eigenvalue|frequenc|buckling)"),
    ("displacement", r"(disp|^u_|^v_|^w_)"),
]


def classify_qoi(name):
    for cls, pat in _PATTERNS:
        if re.search(pat, name, re.I):
            return cls
    return "unknown"


# Each rule: (id, condition(spec, qoi_class) -> bool, required checks,
#             predicted verdict or None, reason)
RULES = [
    ("R1", lambda s, q: reference_class(s["reference"]) == "code_verification",
     {"mms_rates"}, None,
     "an exact manufactured solution allows rate-based code verification"),
    ("R2", lambda s, q: reference_class(s["reference"]) == "solution_verification",
     {"qoi_convergence", "qoi_convergence_multilevel"}, None,
     "without an exact solution, every QoI must be shown converged on a refinement family, "
     "both on the finest three levels (Celik) and by a fit over the finest five"),
    ("R3", lambda s, q: q == "pointwise_peak"
     and regularity_class(s["regularity"]) in ("singular", "near_singular"),
     {"strategy_robustness", "anchored_robustness"}, "FAIL",
     "a peak at a singular or near-singular feature grows or wanders with refinement and "
     "depends on where the mesh is refined"),
    ("R4", lambda s, q: q == "region_average"
     and regularity_class(s["regularity"]) in ("singular", "near_singular"),
     {"strategy_robustness", "anchored_robustness"}, None,
     "a region average is integrable even near a singularity, but its robustness to the "
     "refinement strategy must be shown, not assumed"),
    ("R5", lambda s, q: q in ("global", "displacement")
     and regularity_class(s["regularity"]) != "unknown",
     set(), "PASS",
     "energy and displacements converge in H1-type norms even with corner singularities"),
    ("R8", lambda s, q: regularity_class(s["regularity"]) == "singular"
     and q in ("global", "displacement", "region_average", "pointwise_peak"),
     {"singular_order"}, None,
     "at a singular corner theory fixes the convergence order (2 lam for energy, at least "
     "2 lam elsewhere) and the peak growth (h^(lam-1)); the observed orders must match it"),
    ("R6", lambda s, q: regularity_class(s["regularity"]) == "unknown",
     {"qoi_convergence", "qoi_convergence_multilevel", "strategy_robustness",
      "anchored_robustness"}, None,
     "unknown regularity: apply every solution-verification check"),
]


# Rules whose QoIs are DIAGNOSTICS, not reference candidates: their checks
# still run (they document the expected behaviour), but a FAIL there is the
# expected outcome, not a failed reference target.
DIAGNOSTIC_RULES = {"R3"}


def apply_rules(spec):
    """spec: dict with reference, regularity, qois. Returns per-QoI required
    checks, predictions, the rules that fired and the role: "diagnostic" if a
    DIAGNOSTIC_RULES rule fired, else "candidate"."""
    out = {}
    for qoi in spec["qois"]:
        cls = classify_qoi(qoi)
        req, pred, fired = set(), None, []
        for rid, cond, checks, verdict, reason in RULES:
            if cond(spec, cls):
                req |= checks
                fired.append(rid)
                if verdict is not None:
                    pred = verdict
        out[qoi] = {"qoi_class": cls, "required": sorted(req), "predicted": pred,
                    "rules": fired,
                    "role": "diagnostic" if DIAGNOSTIC_RULES & set(fired) else "candidate"}
    return out


def class_params(check, qoi_class):
    c = CLASS_CEILINGS.get(qoi_class, CLASS_CEILINGS["unknown"])
    if check in ("qoi_convergence", "qoi_convergence_multilevel"):
        return {"band_tol": c["band_tol"], "gci_tol": c["gci_tol"]}
    if check in ("strategy_robustness", "anchored_robustness"):
        return {"tol": c["robust_tol"]}
    if check == "mms_rates":
        return {"tol": 0.2}
    return {}


def check_params(check, qoi, geometry=None):
    """The tolerances a check uses for one QoI: the class ceiling, tightened
    (never loosened) to the required uncertainty of requirements.py when a
    surrogate error is recorded for that QoI."""
    from omar_pfem.verification_agent import requirements
    p = class_params(check, classify_qoi(qoi))
    req, _ = requirements.required_uncertainty(geometry, qoi)
    if req is not None:
        p = {k: (min(v, req) if k in ("band_tol", "gci_tol", "tol") and check != "mms_rates"
                 else v) for k, v in p.items()}
    return p


def rule_plan(spec):
    """Deterministic baseline plan (no LLM), in the proposer's schema, with
    the per-class ceiling tolerances."""
    rules = apply_rules(spec)
    geometry = spec.get("geometry")
    checks = []
    for chk in ("mms_rates", "qoi_convergence", "qoi_convergence_multilevel",
                "singular_order", "strategy_robustness", "anchored_robustness"):
        groups = {}
        for q, r in rules.items():
            if chk in r["required"]:
                p = check_params(chk, q, geometry)
                groups.setdefault((r["qoi_class"], tuple(sorted(p.items()))), []).append(q)
        for (cls, p), qs in groups.items():
            checks.append({"check": chk, "qois": qs, "params": dict(p),
                           "rationale": f"{cls}: required by rules " + ", ".join(sorted(
                               {rid for q in qs for rid in rules[q]["rules"]}))})
    return {"checks": checks,
            "expected_failures": [{"qoi": q, "why": "rules " + ", ".join(r["rules"])}
                                  for q, r in rules.items() if r["predicted"] == "FAIL"]}


# Plan-level rule (depends on the plan, not only on the spec):
# R7  a QoI checked for strategy robustness must also be checked against the
#     family's limit: strategies that agree with each other can share one
#     discretization error.
R7_REASON = ("strategies that agree with each other can share one discretization error, so "
             "strategy agreement must be checked against the family's limit as well")


def required_checks(plan, spec):
    """{qoi: {"required", "rules", "qoi_class"}}: the spec rules (R1-R6) plus
    the plan-level rule R7."""
    out = {q: {"required": set(r["required"]), "rules": list(r["rules"]),
               "qoi_class": r["qoi_class"]} for q, r in apply_rules(spec).items()}
    for c in plan["checks"]:
        if c["check"] == "strategy_robustness":
            for q in c["qois"]:
                if q in out and "anchored_robustness" not in out[q]["required"]:
                    out[q]["required"].add("anchored_robustness")
                    out[q]["rules"].append("R7")
    return out


def plan_rule_errors(plan, spec):
    """Every check the rules require must be in the plan for that QoI."""
    have = {}
    for c in plan["checks"]:
        for q in c["qois"]:
            have.setdefault(q, set()).add(c["check"])
    errs = []
    analysis = apply_rules(spec)
    for q, r in required_checks(plan, spec).items():
        for chk in sorted(r["required"]):
            if chk not in have.get(q, set()):
                errs.append(f"rules {r['rules']} require '{chk}' for '{q}' "
                            f"({r['qoi_class']}, regularity {spec['regularity']})")
    # tolerances may be tightened, never loosened beyond the class ceiling
    for c in plan["checks"]:
        for q in c["qois"]:
            if q not in analysis:
                continue
            ceil = check_params(c["check"], q, spec.get("geometry"))
            for k, v in c["params"].items():
                if k in ceil and v > ceil[k] + 1e-12:
                    errs.append(f"ceiling: {c['check']} {k}={v} for '{q}' exceeds the "
                                f"{analysis[q]['qoi_class']} limit {ceil[k]:g}")
    return errs
