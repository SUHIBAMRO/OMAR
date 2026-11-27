"""A-priori convergence results extracted from the literature by the agent (27 Nov, Professor
Rabczuk: "the agent can identify a-priori convergence results from the literature, including
their assumptions and the solution's regularity"). Pre-registered in
apriori_literature_preregistration.json.

Code: arXiv search, download, passage selection, quote check. LLM (autonomous decision):
extracts each convergence result with its assumptions and evaluates it for the re-entrant
corner of reentrant_study.py. Code compares the extracted rates with the observed ones.

    python -m omar_pfem.verification_agent.apriori_literature
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "apriori_literature_result.json")
QUERIES_V2 = [  # POST HOC attempt 2 (after attempt 1 read unrelated papers)
    'abs:"graded meshes" AND abs:"optimal convergence" AND abs:corner',
    'abs:"L-shaped" AND abs:"graded" AND abs:"finite element"',
    'abs:"corner singularities" AND abs:elasticity AND abs:"finite element"',
    'abs:"mesh grading" AND abs:"singular"']
QUERIES = ['abs:"corner singularity" AND abs:"finite element" AND abs:"convergence"',
           'abs:"graded mesh" AND abs:"singularity" AND abs:"finite element"',
           'abs:"re-entrant corner" AND abs:"error estimate"',
           'abs:"hp-FEM" AND abs:"corner" AND abs:"exponential convergence"']
HINT = "convergence rate corner singularity graded mesh a priori error estimate regularity"
CASE = {"lambda": 0.5444837367824591, "degrees": [1, 2],
        "refinements": ["uniform h", "graded h with nodes s^g", "p-version on a fixed mesh",
                        "hp on geometric meshes"]}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["results"],
          "properties": {"results": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["statement", "refinement", "norm", "rate_formula", "assumptions",
                           "regularity", "quote", "beta_for_case"],
              "properties": {"statement": {"type": "string"},
                             "refinement": {"type": "string", "enum": CASE["refinements"] + ["other"]},
                             "norm": {"type": "string"}, "rate_formula": {"type": "string"},
                             "assumptions": {"type": "string"}, "regularity": {"type": "string"},
                             "quote": {"type": "string"},
                             "beta_for_case": {"type": "array", "items": {
                                 "type": "object", "additionalProperties": False,
                                 "required": ["degree", "g", "beta"],
                                 "properties": {"degree": {"type": "integer"},
                                                "g": {"type": "number"},
                                                "beta": {"type": "number"}}}}}}}}}


def prompt(passage):
    return ("Extract every a-priori convergence result for the finite element method stated in "
            "the passage below (energy norm, L2 norm or a functional), with: the refinement type, "
            "the norm, the rate formula as stated, ALL assumptions (domain, regularity of the "
            "solution, mesh conditions, polynomial degree), the regularity class, and a short "
            "verbatim quote. Then evaluate each result for this case: a 2D elasticity problem whose "
            f"solution has a corner singularity r^lambda with lambda = {CASE['lambda']:.4f}, "
            f"element degrees {CASE['degrees']}. Give beta_for_case as the predicted exponent beta "
            "in error ~ N^(-beta) (N = number of unknowns, 2D) for each degree; for graded meshes "
            "with nodes s^g give it for g = 1.5, 2 and 4 (set g = 0 otherwise); leave the list empty "
            "if the result does not apply or gives no algebraic rate (hp: state the exponential "
            "law in rate_formula).\n\nPASSAGE:\n" + passage)


def run(backend_name="claude-cli", budget_usd=1.2, per_query=8, read=3, queries=None, strict=False):
    from omar_pfem.verification_agent import literature_search as LS, llm_roles
    from omar_pfem.verification_agent.literature_agent import extract_text
    from omar_pfem.verification_agent.literature_select import select_passage
    b = llm_roles.make_backend(backend_name)
    seen, cands = set(), []
    for q in (queries or QUERIES):
        try:
            for c in LS.arxiv(q, per_query):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    cands.append(c)
        except Exception as e:
            print("query failed", q, e)
    scored = []
    for c in cands:
        t = (c["title"] + " " + c["abstract"]).lower()
        s = sum(k in t for k in ("a priori", "error estimate", "convergence rate", "graded",
                                 "corner", "singular", "hp", "elasticity", "regularity"))
        if strict:   # POST HOC: the paper must be about corner singularities AND mesh/convergence
            must = (("corner" in t or "singular" in t) and
                    any(k in t for k in ("graded", "grading", "convergence rate", "error estimate", "hp")))
            s = s if must else -1
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    out = {"candidates": len(cands), "read": []}
    for s, c in scored[:read]:
        if b.cost_usd >= budget_usd:
            break
        try:
            pdf = LS.download(c["id"], os.path.join(HERE, "literature_search_runs", "papers"))
            text = extract_text(pdf)
        except Exception as e:
            print("not readable", c["id"], e)
            continue
        passage, where = select_passage(text, HINT)
        res = b.ask("a-priori extractor", prompt(passage), SCHEMA)
        flat = re.sub(r"\s+", " ", text).lower()
        for r in res["results"]:
            q = re.sub(r"\s+", " ", r["quote"]).lower().strip()
            r["quote_found"] = bool(q) and q[:60] in flat
        out["read"].append({"id": c["id"], "title": c["title"], "passage_at": where,
                            "results": res["results"]})
        print(c["id"], c["title"][:70], len(res["results"]), "results")
    out["llm_cost_usd"] = b.cost_usd
    out["comparison"] = compare(out)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["comparison"], indent=1)[:3000])
    return out


def compare(out):
    """Extracted predictions vs the observed rates of reentrant_result.json."""
    obs = json.load(open(os.path.join(HERE, "reentrant_score.json")))["_rates_beta"]
    key = {("uniform h", 1, 0): "U1", ("uniform h", 2, 0): "U2",
           ("graded h with nodes s^g", 2, 1.5): "G2_1.5", ("graded h with nodes s^g", 1, 2.0): "G1_2",
           ("graded h with nodes s^g", 2, 4.0): "G2_4", ("p-version on a fixed mesh", 0, 0): "P"}
    rows = []
    for paper in out["read"]:
        for r in paper["results"]:
            for x in r["beta_for_case"]:
                g = x["g"] if r["refinement"].startswith("graded") else 0
                deg = 0 if r["refinement"].startswith("p-version") else x["degree"]
                k = key.get((r["refinement"], deg, float(g)))
                if k:
                    rows.append({"paper": paper["id"], "family": k, "predicted_beta": x["beta"],
                                 "observed_beta": obs[k], "agrees_0.1": abs(x["beta"] - obs[k]) <= 0.1,
                                 "quote_found": r["quote_found"], "assumptions": r["assumptions"][:300]})
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run()
