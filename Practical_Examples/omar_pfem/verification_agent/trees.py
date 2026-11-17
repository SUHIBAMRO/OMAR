"""Problem and method trees with explicit rules (GRAFT-style, solid mechanics).

GRAFT represents scientific semantics as a problem tree, a method/action
tree, and rules linking them; this is the verification counterpart for
hyperelastic FE references:

  problem tree   regularity class x reference type x QoI class
  method tree    code verification (mms_rates) / solution verification
                 (qoi_convergence, qoi_convergence_multilevel,
                 strategy_robustness, anchored_robustness)
  rules          which checks a (problem, QoI) REQUIRES, and which verdict
                 theory PREDICTS, each with a stated reason

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
        "anchored_robustness": "every strategy agrees with the family's Richardson limit "
                               "(or finest value), not only with the other strategies",
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


# Tolerance CEILINGS per QoI class. The reference uncertainty must be well
# below the surrogate errors it is used to judge (best surrogate errors in
# this project: ~1.4-4% on B3/B1 QoIs), so global and displacement QoIs,
# which converge fastest, get the tightest ceilings. LLM plans may tighten
# these, never loosen them.
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
    ("error_norm", r"(L2|H1|energy_norm|_rel$)"),
    ("pointwise_peak", r"(peak|_max$|^max_(?!disp)|scf)"),
    ("region_average", r"(_avg|avg_|average|interior)"),
    ("global", r"(energy|reaction|force_total)"),
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
    ("R6", lambda s, q: regularity_class(s["regularity"]) == "unknown",
     {"qoi_convergence", "qoi_convergence_multilevel", "strategy_robustness",
      "anchored_robustness"}, None,
     "unknown regularity: apply every solution-verification check"),
]


def apply_rules(spec):
    """spec: dict with reference, regularity, qois. Returns per-QoI required
    checks, predictions and the rules that fired."""
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
                    "rules": fired}
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


def rule_plan(spec):
    """Deterministic baseline plan (no LLM), in the proposer's schema, with
    the per-class ceiling tolerances."""
    rules = apply_rules(spec)
    checks = []
    for chk in ("mms_rates", "qoi_convergence", "qoi_convergence_multilevel",
                "strategy_robustness", "anchored_robustness"):
        for cls in CLASS_CEILINGS:
            qs = [q for q, r in rules.items() if chk in r["required"] and r["qoi_class"] == cls]
            if qs:
                checks.append({"check": chk, "qois": qs, "params": class_params(chk, cls),
                               "rationale": f"{cls}: required by rules " + ", ".join(sorted(
                                   {rid for q in qs for rid in rules[q]["rules"]}))})
    return {"checks": checks,
            "expected_failures": [{"qoi": q, "why": "rules " + ", ".join(r["rules"])}
                                  for q, r in rules.items() if r["predicted"] == "FAIL"]}


def plan_rule_errors(plan, spec):
    """Every check the rules require must be in the plan for that QoI."""
    have = {}
    for c in plan["checks"]:
        for q in c["qois"]:
            have.setdefault(q, set()).add(c["check"])
    errs = []
    analysis = apply_rules(spec)
    for q, r in analysis.items():
        for chk in r["required"]:
            if chk not in have.get(q, set()):
                errs.append(f"rules {r['rules']} require '{chk}' for '{q}' "
                            f"({r['qoi_class']}, regularity {spec['regularity']})")
    # tolerances may be tightened, never loosened beyond the class ceiling
    for c in plan["checks"]:
        for q in c["qois"]:
            if q not in analysis:
                continue
            ceil = class_params(c["check"], analysis[q]["qoi_class"])
            for k, v in c["params"].items():
                if k in ceil and v > ceil[k] + 1e-12:
                    errs.append(f"ceiling: {c['check']} {k}={v} for '{q}' exceeds the "
                                f"{analysis[q]['qoi_class']} ceiling {ceil[k]}")
    return errs
