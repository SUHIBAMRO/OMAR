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
import re

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


def reassess(run_file, backend_name="claude-cli", budget_usd=1.9, max_carry=2, log=print):
    """POST HOC v2 assessment of the same candidates (same passages), then carry the feasible
    2D cases through. Labelled post hoc; the first run's score stands."""
    from omar_pfem.verification_agent import autonomous, capabilities as CAP, llm_roles
    from omar_pfem.verification_agent import literature_search as LS
    from omar_pfem.verification_agent.literature_agent import extract_text
    from omar_pfem.verification_agent.literature_select import select_passage
    first = json.load(open(run_file))
    made = []
    orig = llm_roles.make_backend

    def capture(*a, **k):
        b = orig(*a, **k)
        made.append(b)
        return b
    llm_roles.make_backend = capture
    cost = lambda: sum(getattr(b, "cost_usd", 0.0) for b in made)
    backend = llm_roles.make_backend(backend_name)
    hints = {g: h for g, _, h in QUERIES}
    rec = {"post_hoc": True, "first_run": os.path.basename(run_file), "candidates": [], "carried": []}
    try:
        for c in first["candidates"]:
            if cost() >= budget_usd:
                break
            pdf = LS.download(c["id"], os.path.join(HERE, "literature_search_runs", "papers"))
            hint = next(h for g, _, h in QUERIES if g == c["group"] and
                        select_passage(extract_text(pdf), h)[1] == c["passage_at"]) \
                if any(g == c["group"] for g, _, _ in QUERIES) else hints[c["group"]]
            passage, _ = select_passage(extract_text(pdf), hint)
            req = CAP.requirements_v2(backend, passage, hint)
            a = CAP.assess_v2(req)
            rec["candidates"].append({"id": c["id"], "group": c["group"], "requirements": req,
                                      "assessment": a})
            log(f"[bench v2] {c['id']}: {a['decision']}")
            dim = req["size"].get("dimension") or 2
            if a["feasible"] and dim == 2 and len(rec["carried"]) < max_carry and cost() < budget_usd:
                res = autonomous.run(pdf=pdf, hint=a["feasible_cases"][0], backend_name=backend_name,
                                     budget_minutes=20.0, max_experiments=4,
                                     chosen_by="the benchmark search v2 (code + LLM requirements)")
                rec["carried"].append({"id": c["id"], "case": a["feasible_cases"][0],
                                       "final": res.get("final")})
                log(f"[bench v2] carried {c['id']}: final {res.get('final')}")
    finally:
        llm_roles.make_backend = orig
        gaps = collections.Counter(m["need"] for r in rec["candidates"] for cs in r["assessment"]["cases"]
                                   for m in cs["missing"])
        rec["missing_capabilities_ranked"] = gaps.most_common()
        rec["llm_cost_usd"] = cost()
        json.dump(rec, open(run_file.replace(".json", "_v2_posthoc.json"), "w"), indent=1, default=str)
        log(f"[bench v2] {len(rec['candidates'])} reassessed, {len(rec['carried'])} carried; gaps "
            f"{gaps.most_common(6)}; LLM cost USD {cost():.2f}")
    return rec


QUERIES_R5 = [
    ("2D established", 'abs:"plate with a hole" AND abs:elasticity', "plate circular hole tension Kirsch"),
    ("2D established", 'abs:"Cook\'s membrane" AND abs:"linear elastic"', "Cook's membrane"),
    ("2D established", 'abs:"L-shaped domain" AND abs:"linear elasticity"', "L-shaped domain"),
    ("2D established", 'abs:"cantilever" AND abs:"plane stress" AND abs:"finite element"', "cantilever beam plane stress"),
    ("3D hyperelastic", 'abs:"neo-Hookean" AND abs:"finite element" AND abs:"three-dimensional"',
     "neo-Hookean three-dimensional block hole"),
]


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower().replace("$", "")).strip()[:80]


def previously_seen_titles():
    """27 Nov (after round 6 read the journal version of arXiv 1911.08728 under an OpenAlex id):
    duplicates are also recognised by their normalised title."""
    out = set()
    for d in (os.path.join(HERE, "literature_search_runs"), OUT):
        if os.path.isdir(d):
            for g in os.listdir(d):
                if g.endswith(".json") and g != "seen.json":
                    r = json.load(open(os.path.join(d, g)))
                    for key in ("candidates", "read"):
                        for c in r.get(key, []):
                            if isinstance(c, dict) and c.get("title"):
                                out.add(norm_title(c["title"]))
    return out


def previously_seen():
    ids = set()
    f = os.path.join(HERE, "literature_search_runs", "seen.json")
    if os.path.exists(f):
        ids |= set(json.load(open(f)))
    for d in (os.path.join(HERE, "literature_search_runs"), OUT):
        if os.path.isdir(d):
            for g in os.listdir(d):
                if g.endswith(".json") and g != "seen.json":
                    r = json.load(open(os.path.join(d, g)))
                    ids |= {c["id"] for c in r.get("candidates", []) if isinstance(c, dict) and "id" in c}
                    ids |= {c["id"] for c in r.get("read", []) if isinstance(c, dict) and "id" in c}
    ids |= {d for d in os.listdir(os.path.join(HERE, "autonomous_runs")) if re.match(r"\d{4}\.\d{4,5}$", d)}
    return ids


def run_r5(backend_name="claude-cli", budget_usd=2.0, per_query=10, assess_top=3, max_carry=2,
           log=print):
    """Round 5 (pre-registered): only papers never seen before count; v2 assessment."""
    from omar_pfem.verification_agent import autonomous, capabilities as CAP, llm_roles
    from omar_pfem.verification_agent import literature_search as LS
    from omar_pfem.verification_agent.literature_agent import extract_text
    from omar_pfem.verification_agent.literature_select import select_passage
    made, orig = [], llm_roles.make_backend

    def capture(*a, **k):
        b = orig(*a, **k)
        made.append(b)
        return b
    llm_roles.make_backend = capture
    cost = lambda: sum(getattr(b, "cost_usd", 0.0) for b in made)
    backend = llm_roles.make_backend(backend_name)
    seen = previously_seen()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rec = {"round": 5, "started": stamp, "excluded_previously_seen": len(seen), "candidates": [],
           "carried": []}
    try:
        for group, q, hint in QUERIES_R5:
            try:
                found = LS.arxiv(q, per_query)
            except Exception as e:
                log(f"[r5] query failed {q!r}: {e!s:.100}")
                continue
            new = [c for c in found if c["id"] not in seen and LS.code_screen(c)[0]]
            log(f"[r5] {q!r}: {len(found)} found, {len(new)} new and pass the abstract screen")
            ranked = []
            for c in new:
                seen.add(c["id"])
                try:
                    pdf = LS.download(c["id"], os.path.join(HERE, "literature_search_runs", "papers"))
                    passage, where = select_passage(extract_text(pdf), hint)
                    ranked.append((sum(w.lower() in passage.lower() for w in hint.split()), c, pdf,
                                   passage, where))
                except Exception as e:
                    log(f"[r5] {c['id']}: not readable ({e!s:.80})")
            ranked.sort(key=lambda t: -t[0])
            for _, c, pdf, passage, where in ranked[:assess_top]:
                if cost() >= budget_usd:
                    break
                req = CAP.requirements_v2(backend, passage, hint)
                a = CAP.assess_v2(req)
                rec["candidates"].append({"group": group, "id": c["id"], "title": c["title"],
                                          "requirements": req, "assessment": a})
                log(f"[r5] {c['id']}: {a['decision']}")
                dim = req["size"].get("dimension") or 2
                if a["feasible"] and dim == 2 and len(rec["carried"]) < max_carry and cost() < budget_usd:
                    res = autonomous.run(pdf=pdf, hint=a["feasible_cases"][0], backend_name=backend_name,
                                         budget_minutes=20.0, max_experiments=4,
                                         chosen_by="search round 5 (code + LLM requirements)")
                    refused = next((s["detail"] for s in json.load(open(os.path.join(
                        HERE, "autonomous_runs", c["id"], "run.json")))["steps"]
                        if s["status"] == "unsupported"), None) if os.path.exists(os.path.join(
                            HERE, "autonomous_runs", c["id"], "run.json")) else None
                    rec["carried"].append({"id": c["id"], "case": a["feasible_cases"][0],
                                           "final": res.get("final"), "refused": refused})
                    log(f"[r5] carried {c['id']}: final {res.get('final')} refused {refused}")
    finally:
        llm_roles.make_backend = orig
        rec["llm_cost_usd"] = cost()
        os.makedirs(OUT, exist_ok=True)
        json.dump(rec, open(os.path.join(OUT, f"round5_{stamp}.json"), "w"), indent=1, default=str)
        log(f"[r5] {len(rec['candidates'])} assessed, {len(rec['carried'])} carried; "
            f"LLM cost USD {cost():.2f}")
    return rec


QUERIES_R6 = [
    ("2D established", "benchmark problems solid mechanics finite element", "benchmark Cook's membrane plate hole"),
    ("2D established", "Cook's membrane linear elasticity finite element", "Cook's membrane"),
    ("2D established", "plate with a circular hole linear elasticity finite element convergence", "plate circular hole tension"),
    ("2D established", "L-shaped domain linear elasticity corner singularity finite element", "L-shaped domain re-entrant corner"),
    ("3D hyperelastic", "neo-Hookean three-dimensional benchmark finite element", "neo-Hookean three-dimensional benchmark"),
]


def run_r6(backend_name="claude-cli", budget_usd=1.5, per_query=10, assess_top=2, max_carry=2,
           log=print):
    """Round 6 (pre-registered): OpenAlex open-access works, unseen only, v2 assessment."""
    from omar_pfem.verification_agent import autonomous, capabilities as CAP, llm_roles
    from omar_pfem.verification_agent import literature_search as LS
    from omar_pfem.verification_agent.literature_agent import extract_text
    from omar_pfem.verification_agent.literature_select import select_passage
    made, orig = [], llm_roles.make_backend

    def capture(*a, **k):
        b = orig(*a, **k)
        made.append(b)
        return b
    llm_roles.make_backend = capture
    cost = lambda: sum(getattr(b, "cost_usd", 0.0) for b in made)
    backend = llm_roles.make_backend(backend_name)
    seen = previously_seen()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rec = {"round": 6, "source": "OpenAlex", "started": stamp, "candidates": [], "carried": [],
           "unreadable": []}
    try:
        for group, q, hint in QUERIES_R6:
            try:
                found = LS.openalex(q, per_query)
            except Exception as e:
                log(f"[r6] query failed {q!r}: {e!s:.100}")
                continue
            titles = previously_seen_titles()
            new = [c for c in found if c["id"] not in seen and norm_title(c["title"]) not in titles
                   and LS.code_screen(c)[0]]
            log(f"[r6] {q!r}: {len(found)} with a PDF link, {len(new)} new and pass the abstract screen")
            ranked = []
            for c in new:
                seen.add(c["id"])
                try:
                    pdf = LS.download_url(c["id"], c["pdf_url"],
                                          os.path.join(HERE, "literature_search_runs", "papers"))
                    passage, where = select_passage(extract_text(pdf), hint)
                    ranked.append((sum(w.lower() in passage.lower() for w in hint.split()), c, pdf,
                                   passage, where))
                except Exception as e:
                    rec["unreadable"].append({"id": c["id"], "reason": f"{e!s:.80}"})
            ranked.sort(key=lambda t: -t[0])
            for _, c, pdf, passage, where in ranked[:assess_top]:
                if cost() >= budget_usd:
                    break
                req = CAP.requirements_v2(backend, passage, hint)
                a = CAP.assess_v2(req)
                rec["candidates"].append({"group": group, "id": c["id"], "title": c["title"],
                                          "doi": c.get("doi"), "requirements": req, "assessment": a})
                log(f"[r6] {c['id']} {c['title'][:60]!r}: {a['decision']}")
                dim = req["size"].get("dimension") or 2
                # 27 Nov (after round 7): a case whose missing information concerns the material,
                # the loads or the supports is refused by the definition step anyway; not carried
                case = next((x for x in a["cases"] if x["case"] == (a["feasible_cases"] or [None])[0]), {})
                blocking = [m for m in case.get("missing_information", [])
                            if re.search(r"material|modulus|poisson|load|traction|pressure|support|"
                                         r"boundary condition|clamp", m.lower())]
                if blocking:
                    log(f"[r6] {c['id']}: feasible but not carried: missing {blocking[:2]}")
                    rec.setdefault("not_carried_missing_information", []).append(
                        {"id": c["id"], "missing": blocking})
                elif a["feasible"] and dim == 2 and len(rec["carried"]) < max_carry and cost() < budget_usd:
                    res = autonomous.run(pdf=pdf, hint=a["feasible_cases"][0], backend_name=backend_name,
                                         budget_minutes=20.0, max_experiments=4,
                                         chosen_by="search round 6 (OpenAlex, code + LLM requirements)")
                    rf = os.path.join(HERE, "autonomous_runs", c["id"], "run.json")
                    refused = next((s["detail"] for s in json.load(open(rf))["steps"]
                                    if s["status"] == "unsupported"), None) if os.path.exists(rf) else None
                    rec["carried"].append({"id": c["id"], "case": a["feasible_cases"][0],
                                           "final": res.get("final"), "refused": refused})
                    log(f"[r6] carried {c['id']}: final {res.get('final')} refused {refused}")
    finally:
        llm_roles.make_backend = orig
        rec["llm_cost_usd"] = cost()
        os.makedirs(OUT, exist_ok=True)
        json.dump(rec, open(os.path.join(OUT, f"round6_{stamp}.json"), "w"), indent=1, default=str)
        log(f"[r6] {len(rec['candidates'])} assessed, {len(rec['carried'])} carried, "
            f"{len(rec['unreadable'])} unreadable; LLM cost USD {cost():.2f}")
    return rec
