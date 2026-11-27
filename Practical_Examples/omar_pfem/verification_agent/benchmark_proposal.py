"""The agent proposes the verification benchmarks for a new code itself (27 Nov, Professor
Rabczuk: "ideally the agent should identify these as relevant benchmark problems for a new code
itself"). Pre-registered in benchmark_proposal_preregistration.json; the prompt does NOT name any
benchmark, singularity or refinement strategy.

    python -m omar_pfem.verification_agent.benchmark_proposal
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "benchmark_proposal_result.json")

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["benchmarks"],
          "properties": {"benchmarks": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["name", "purpose", "features_tested", "refinement_studies",
                           "expected_rates", "reference_solution", "literature"],
              "properties": {k: {"type": "string"} for k in
                             ("name", "purpose", "features_tested", "refinement_studies",
                              "expected_rates", "reference_solution", "literature")}}}}}


def prompt():
    from omar_pfem.verification_agent.capabilities import REGISTRY
    sup = {cat: [k for k, (ok, _) in v.items() if ok] for cat, v in REGISTRY.items()}
    return ("You plan the verification of a NEW finite element code before it is used for "
            "research. Its features: " + json.dumps(sup) + ". Elements: quadratic quadrilaterals "
            "and triangles in 2D, 27-node hexahedra and 10-node tetrahedra in 3D. Propose the "
            "set of verification benchmark problems (6 to 10) that would establish that the code "
            "is correct and how it converges, and for each: its purpose, the features it tests, "
            "the refinement studies to run, the convergence rates theory predicts (with the "
            "norm or quantity), the reference solution, and the literature that establishes it.")


def checks(res):
    txt = lambda b: " ".join(b.values()).lower()
    bs = res["benchmarks"]
    corner = [b for b in bs if re.search(r"re-?entrant|l-shap|corner singular|singular corner|"
                                         r"crack|notch", txt(b))]
    return {
        "P1_singular_corner": bool(corner),
        "P2_uniform_vs_graded": any(re.search(r"uniform", txt(b)) and
                                    re.search(r"grad|adaptiv|geometric", txt(b)) for b in corner),
        "P3_p_or_hp": any(re.search(r"\bhp\b|p-refine|p-version|polynomial degree|higher.order|"
                                    r"p-convergence", txt(b)) for b in bs),
        "P4_mms_separate": any(re.search(r"manufactured", txt(b)) for b in bs),
        "P5_reference_from_literature": any(re.search(r"cook|kirsch|23\.9|plate with a (circular )?hole",
                                                      txt(b)) for b in bs)}


def run(backend_name="claude-cli"):
    from omar_pfem.verification_agent import llm_roles
    b = llm_roles.make_backend(backend_name)
    res = b.ask("benchmark planner", prompt(), SCHEMA)
    out = {"prompt": prompt(), "result": res, "checks": checks(res), "llm_cost_usd": b.cost_usd}
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in out["checks"].items():
        print(k, v)
    print("benchmarks:", [x["name"] for x in res["benchmarks"]], "cost", b.cost_usd)
    return out


if __name__ == "__main__":
    run()
