"""Automatic literature search (2026-11-25): the step "choosing the paper" without a person.

Until now a person chose the paper and wrote a hint. Here the agent:

  1. QUERIES   builds arXiv queries from what its pipeline can represent (CAPABILITIES) and
               from its memory: a problem class with fewer recorded runs is searched first
               (the experience it lacks), so the search follows what it has not learned yet;
  2. SEARCHES  the arXiv API (titles, abstracts, PDF links);
  3. SCREENS   by code on the abstract (keywords; excluded: shells, plates in bending, dynamics,
               fracture, contact, fluids), then by code on the FULL TEXT (the passage that best
               matches the class's hint, penalised for curved edges, 3D, other materials),
               then by the LLM on the best few passages only ("scout": is a representable example
               stated in this passage, and what is it called?);
  4. READS     the most promising papers: the passage is chosen by literature_select from the
               scout's hint, then the card and the problem definition run as in autonomous.py,
               which refuses what it cannot represent;
  5. RECORDS   every candidate, decision and cost (literature_search_runs/<stamp>.json); papers
               already read are not read again.

    python -m omar_pfem.verification_agent.literature_search --max-read 3 --backend claude-cli
"""
import datetime
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "literature_search_runs")
SEEN = os.path.join(OUTDIR, "seen.json")
API = "https://export.arxiv.org/api/query"

CAPABILITIES = {
    "linear_elastic": {"queries": [
        'abs:"linear elasticity" AND abs:"numerical examples"',
        'abs:elasticity AND abs:"L-shaped"',
        'abs:"plane stress" AND abs:"finite element" AND abs:numerical',
        'abs:"plate with a hole" AND abs:"finite element"',
        'abs:"Cook" AND abs:membrane AND abs:"linear elasticity"'],
        "hint": "plate hole L-shaped Cook membrane clamped traction Young modulus Poisson "
                "ratio plane strain plane stress",
        "text": "2D small-strain linear elasticity, plane strain or plane stress"},
    "neo_hookean": {"queries": [
        'abs:"neo-Hookean" AND abs:"numerical examples"',
        'abs:hyperelastic AND abs:"finite element" AND abs:"numerical examples"',
        'abs:"neo-Hookean" AND abs:"plane strain"',
        'abs:"neo-Hookean" AND abs:"Cook"'],
        "hint": "neo-Hookean plane strain block clamped traction compression Young modulus "
                "Poisson ratio",
        "text": "2D compressible neo-Hookean hyperelasticity at finite strain, plane strain, "
                "dead tractions"},
}
# 26 Nov: brought in line with what the problem definition (autonomous.MODELER) and geometry2d
# represent since 25 Nov: straight-edged polygons (inclined edges allowed) with circular or
# polygonal holes. Before, holes and inclined edges were penalised and declared not
# representable, so the search could not find the problems the pipeline now solves.
PENALTY = ("three-dimensional", "3d", "cylinder", "sphere", "ellipse", "elliptical", "spline",
           "nurbs", "shell", "contact", "incompressible", "mooney", "ogden", "crack")
DOMAIN_TEXT = ("2D domains bounded by straight edges (inclined edges allowed, e.g. Cook's "
               "membrane or a trapezoid) with circular or polygonal holes (e.g. a plate with a "
               "hole, also as a quarter model with symmetry edges); clamped, symmetry, roller and "
               "constant traction boundary conditions (pressure for linear elasticity only)")

INCLUDE = ("numerical example", "benchmark", "test case", "finite element", "plane strain",
           "plane stress", "elasticity", "hyperelastic", "neo-hookean")
EXCLUDE = ("shell", "kirchhoff", "reissner", "mindlin", "plate bending", "dynamic", "wave",
           "crack propagation", "phase-field fracture", "contact", "fluid", "thermal")

SCOUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["likely_representable", "example", "reason"],
    "properties": {"likely_representable": {"type": "boolean"},
                   "example": {"type": "string"}, "reason": {"type": "string"}}}


def queries_by_need():
    """Material classes ordered by how few runs the agent has recorded for them."""
    from omar_pfem.verification_agent.autonomous import RUNS
    count = {k: 0 for k in CAPABILITIES}
    if os.path.isdir(RUNS):
        for d in os.listdir(RUNS):
            f = os.path.join(RUNS, d, "run.json")
            if os.path.exists(f):
                desc = json.load(open(f)).get("description") or {}
                m = desc.get("material", "linear_elastic")
                if m in count:
                    count[m] += 1
    order = sorted(count, key=lambda k: count[k])
    return order, count


def arxiv(query, n=15):
    url = f"{API}?search_query={urllib.parse.quote(query, safe=':')}&start=0&max_results={n}"
    req = urllib.request.Request(url, headers={"User-Agent": "verification-agent/0.1"})
    xml = urllib.request.urlopen(req, timeout=60).read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
        out.append({"id": re.sub(r"v\d+$", "", aid),
                    "title": " ".join(e.find("a:title", ns).text.split()),
                    "abstract": " ".join(e.find("a:summary", ns).text.split()),
                    "query": query})
    return out


def code_screen(c):
    t = (c["title"] + " " + c["abstract"]).lower()
    inc = sum(k in t for k in INCLUDE)
    exc = [k for k in EXCLUDE if k in t]
    return inc >= 2 and not exc, {"include_hits": inc, "excluded_by": exc}


def text_screen(pdf, material):
    """Code, on the full text: the best passage for the class's hint and its penalties."""
    from omar_pfem.verification_agent.literature_agent import extract_text
    from omar_pfem.verification_agent.literature_select import select_passage
    text = extract_text(pdf)
    passage, where = select_passage(text, CAPABILITIES[material]["hint"])
    low = passage.lower()
    score = float(re.search(r"score ([0-9.]+)", where).group(1))
    pen = {k: low.count(k) for k in PENALTY if k in low}
    return {"passage_at": where, "score": score, "penalties": pen,
            "net": score - 5 * sum(pen.values()), "passage": passage}


def scout(backend, c, material):
    prompt = (f"You screen papers for a verification agent. Its pipeline can solve "
              f"{CAPABILITIES[material]['text']}, on {DOMAIN_TEXT}. Below is the passage of the "
              "paper that code found most likely to describe a numerical example. Does it "
              "contain an example that this pipeline can represent exactly from the text "
              "(geometry, material constants, supports and loads stated)? If yes, name the "
              "example in a few words that would find it again (e.g. 'L-shaped panel', 'block "
              "under compression'). Straight inclined edges and circular or polygonal holes ARE "
              "representable. Be strict otherwise: curved outer edges, 3D, shells, contact, "
              "near-incompressible materials (Poisson ratio above 0.49), other material models, "
              "or data given only in figures are not representable, except the geometry of a "
              "named standard benchmark (Cook's membrane). If one case of several is "
              "representable (e.g. Poisson ratio 1/3 next to 0.4999), answer yes and name it."
              f"\n\nTITLE: {c['title']}\n\nPASSAGE:\n{c['text_screen']['passage'][:6000]}")
    return backend.ask("scout", prompt, SCOUT_SCHEMA)


def download(aid, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{aid}.pdf")
    if not os.path.exists(path):
        req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}",
                                     headers={"User-Agent": "verification-agent/0.1"})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=120).read())
    return path


def search(backend_name="claude-cli", max_read=3, per_query=15, budget_usd=5.0,
           budget_minutes=20.0, scout_top=6, log=print):
    from omar_pfem.verification_agent import autonomous, llm_roles
    os.makedirs(OUTDIR, exist_ok=True)
    seen = set(json.load(open(SEEN))) if os.path.exists(SEEN) else set()
    made = []
    orig = llm_roles.make_backend

    def capture(*a, **k):
        b = orig(*a, **k)
        made.append(b)
        return b
    llm_roles.make_backend = capture
    backend = llm_roles.make_backend(backend_name)
    cost = lambda: sum(getattr(b, "cost_usd", 0.0) for b in made)
    order, count = queries_by_need()
    rec = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "material_order": order, "runs_per_material": count, "candidates": [], "read": []}
    log(f"[search] material classes by need (fewest runs first): {count} -> {order}")
    try:
        for material in order:
            for q in CAPABILITIES[material]["queries"]:
                try:
                    cands = arxiv(q, per_query)
                except Exception as e:
                    log(f"[search] query failed {q!r}: {e!r}")
                    continue
                time.sleep(3)                                   # arXiv API etiquette
                for c in cands:
                    if c["id"] in seen or any(x["id"] == c["id"] for x in rec["candidates"]):
                        continue
                    ok, why = code_screen(c)
                    c.update(material=material, code_screen=ok, code_screen_detail=why)
                    if ok:
                        try:
                            pdf = download(c["id"], os.path.join(OUTDIR, "papers"))
                            c["text_screen"] = text_screen(pdf, material)
                            time.sleep(1)
                        except Exception as e:
                            c["text_screen"] = {"error": repr(e)[:200], "net": -1e9}
                    rec["candidates"].append(c)
                log(f"[search] {q!r}: {len(cands)} found, "
                    f"{sum(1 for c in rec['candidates'] if c.get('query') == q and c['code_screen'])} "
                    f"pass the keyword screen")
            ranked = sorted([c for c in rec["candidates"] if c["material"] == material
                             and c.get("text_screen", {}).get("net", -1e9) > 0],
                            key=lambda c: -c["text_screen"]["net"])
            for c in ranked[:scout_top]:
                if cost() < budget_usd:
                    c["scout"] = scout(backend, c, material)
                    log(f"[search] scout {c['id']}: {c['scout']['likely_representable']} "
                        f"({c['scout']['example']!r}; {c['scout']['reason'][:120]})")
            picks = [c for c in ranked if c.get("scout", {}).get("likely_representable")]
            for c in picks:
                if len(rec["read"]) >= max_read or cost() >= budget_usd:
                    break
                pdf = download(c["id"], os.path.join(OUTDIR, "papers"))
                log(f"[search] reading {c['id']}: {c['title'][:80]} (hint {c['scout']['example']!r})")
                res = autonomous.run(pdf=pdf, hint=c["scout"]["example"], backend_name=backend_name,
                                     budget_minutes=budget_minutes, max_experiments=4,
                                     chosen_by="the literature search (code + scout)")
                seen.add(c["id"])
                rec["read"].append({"id": c["id"], "title": c["title"], "material": material,
                                    "hint": c["scout"]["example"],
                                    "run_dir": os.path.join("autonomous_runs", c["id"]),
                                    "reached_verdicts": res.get("final") is not None,
                                    "final": res.get("final")})
                json.dump(sorted(seen), open(SEEN, "w"))
            if len(rec["read"]) >= max_read or cost() >= budget_usd:
                break
    finally:
        llm_roles.make_backend = orig
        rec["llm_cost_usd"] = cost()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json.dump(rec, open(os.path.join(OUTDIR, f"{stamp}.json"), "w"), indent=1, default=float)
        log(f"[search] read {len(rec['read'])} papers, "
            f"{sum(r['reached_verdicts'] for r in rec['read'])} reached verdicts; "
            f"LLM cost USD {rec['llm_cost_usd']:.2f}")
    return rec


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude-cli")
    ap.add_argument("--max-read", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    a = ap.parse_args()
    search(a.backend, a.max_read, budget_usd=a.budget_usd)


# ------------------------------------------------------------------ OpenAlex (27 Nov) ------
OPENALEX = "https://api.openalex.org/works"


def openalex(query, n=10):
    """Open-access works with a PDF link (journals and repositories, not only arXiv). The id is
    the OpenAlex work id; the abstract is rebuilt from its inverted index."""
    url = (f"{OPENALEX}?search={urllib.parse.quote(query)}&filter=open_access.is_oa:true,"
           f"has_abstract:true&per-page={n}")
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "verification-agent/0.1"}), timeout=60))
    out = []
    for w in d["results"]:
        pdf = (w.get("best_oa_location") or {}).get("pdf_url")
        if not pdf:
            continue
        inv = w.get("abstract_inverted_index") or {}
        words = sorted((i, k) for k, v in inv.items() for i in v)
        out.append({"id": "OA_" + w["id"].rsplit("/", 1)[-1], "title": w.get("title") or "",
                    "abstract": " ".join(k for _, k in words), "pdf_url": pdf, "query": query,
                    "doi": w.get("doi")})
    return out


def download_url(cid, url, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{cid}.pdf")
    if not os.path.exists(path):
        data = urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (verification-agent)"}), timeout=120).read()
        if not data.startswith(b"%PDF"):
            raise ValueError("not a PDF (publisher page or access wall)")
        open(path, "wb").write(data)
    return path
