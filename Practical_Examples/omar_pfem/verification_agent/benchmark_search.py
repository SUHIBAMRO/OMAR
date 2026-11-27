"""The agent finds established benchmarks itself and decides whether it can carry them through
(27 Nov; pre-registered in benchmark_search_preregistration.json).

For each candidate: requirements by the LLM (autonomous decision), capability assessment by
code (predefined procedure, capabilities.assess); feasible 2D candidates are carried through by
autonomous.run; infeasible ones are reported with the missing capabilities, which are NOT
implemented.

    python -m omar_pfem.verification_agent.benchmark_search --budget-usd 2.5
"""
import argparse
import collections
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "benchmark_search_runs")
QUERIES = [  # 3D hyperelastic with curved geometry / stress concentration first (the challenge)
    ("3D hyperelastic", 'abs:hyperelastic AND abs:"three-dimensional" AND abs:benchmark',
     "three-dimensional hyperelastic benchmark hole cylinder torsion"),
    ("3D hyperelastic", 'abs:"neo-Hookean" AND abs:"stress concentration"',
     "neo-Hookean stress concentration hole notch three-dimensional"),
    ("3D hyperelastic", 'abs:hyperelastic AND abs:"finite element" AND abs:cylinder AND abs:"numerical examples"',
     "hyperelastic cylinder hole numerical example"),
    ("2D established", 'abs:"benchmark" AND abs:"plane strain" AND abs:elasticity AND abs:"finite element"',
     "benchmark plane strain plate hole Cook membrane L-shaped"),
    ("2D established", 'abs:"Cook" AND abs:"membrane" AND abs:"finite element"',
     "Cook's membrane"),
]


def run(backend_name="claude-cli", budget_usd=2.5, per_query=12, assess_top=3, max_carry=2,
        log=print):
    from omar_pfem.verification_agent import autonomous, capabilities as CAP, llm_roles
    from omar_pfem.verification_agent import literature_search as LS
    from omar_pfem.verification_agent.literature_agent import extract_text
    from omar_pfem.verification_agent.literature_select import select_passage
    made = []
    orig = llm_roles.make_backend

    def capture(*a, **k):
        b = orig(*a, **k)
        made.append(b)
        return b
    llm_roles.make_backend = capture
    cost = lambda: sum(getattr(b, "cost_usd", 0.0) for b in made)
    backend = llm_roles.make_backend(backend_name)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rec = {"started": stamp, "candidates": [], "carried": []}
    seen = set()
    try:
        for group, q, hint in QUERIES:
            try:
                found = LS.arxiv(q, per_query)
            except Exception as e:
                log(f"[bench] query failed {q!r}: {e!s:.100}")
                continue
            ok = [c for c in found if LS.code_screen(c)[0] and c["id"] not in seen]
            log(f"[bench] {group}: {q!r}: {len(found)} found, {len(ok)} pass the abstract screen")
            ranked = []
            for c in ok:
                seen.add(c["id"])
                try:
                    pdf = LS.download(c["id"], os.path.join(HERE, "literature_search_runs", "papers"))
                    text = extract_text(pdf)
                    passage, where = select_passage(text, hint)
                    ranked.append((len([w for w in hint.split() if w.lower() in passage.lower()]),
                                   c, pdf, passage, where))
                except Exception as e:
                    log(f"[bench] {c['id']}: not readable ({e!s:.80})")
            ranked.sort(key=lambda t: -t[0])
            for _, c, pdf, passage, where in ranked[:assess_top]:
                if cost() >= budget_usd:
                    break
                req = CAP.requirements(backend, passage, hint)
                a = CAP.assess(req)
                row = {"group": group, "id": c["id"], "title": c["title"], "passage_at": where,
                       "requirements": req, "assessment": a}
                rec["candidates"].append(row)
                log(f"[bench] {c['id']} ({req['example'][:60]}): {a['decision']}"
                    + ("" if a["feasible"] else ": " + "; ".join(m["need"] for m in a["missing"])))
                if a["feasible"] and req["size"].get("dimension", 2) == 2 and \
                        len(rec["carried"]) < max_carry and cost() < budget_usd:
                    res = autonomous.run(pdf=pdf, hint=req["example"], backend_name=backend_name,
                                         budget_minutes=20.0, max_experiments=4,
                                         chosen_by="the benchmark search (code + LLM requirements)")
                    rec["carried"].append({"id": c["id"], "example": req["example"],
                                           "final": res.get("final"),
                                           "run_dir": os.path.join("autonomous_runs", c["id"])})
                    log(f"[bench] carried {c['id']}: final {res.get('final')}")
    finally:
        llm_roles.make_backend = orig
        gaps = collections.Counter(m["need"] for r in rec["candidates"]
                                   for m in r["assessment"]["missing"])
        rec["missing_capabilities_ranked"] = gaps.most_common()
        rec["llm_cost_usd"] = cost()
        os.makedirs(OUT, exist_ok=True)
        json.dump(rec, open(os.path.join(OUT, f"{stamp}.json"), "w"), indent=1, default=str)
        log(f"[bench] {len(rec['candidates'])} assessed, {len(rec['carried'])} carried through; "
            f"gaps {gaps.most_common(5)}; LLM cost USD {cost():.2f}")
    return rec


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude-cli")
    ap.add_argument("--budget-usd", type=float, default=2.5)
    a = ap.parse_args()
    run(a.backend, a.budget_usd)
