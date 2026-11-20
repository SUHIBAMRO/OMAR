"""Literature agent: a paper (PDF) -> a quote-checked benchmark card -> the
problem text the formalizer takes, plus what the reference still needs.

The first of the three agents in Prof. Rabczuk's plan (12 Nov: "a literature
agent to formalize benchmark problems"); here it does by itself what was done
by hand for the OpenPINN plate (Section 18 of the report) and the CFM lamina
(Section 23).

Authority split, as for the other roles (llm_roles.py):
  code   extracts the text, selects the benchmark section, CHECKS EVERY QUOTE
         against the paper, maps QoIs to the agent's vocabulary, decides what
         verification the reference needs, writes the problem text;
  LLM    ("librarian") fills the card and must give, for every field, a
         verbatim quote from the section. A field whose quote is not found in
         the paper is marked unsupported and is NOT used in the problem text
         (no invented moduli, meshes or claims of convergence).

    python -m omar_pfem.verification_agent.literature_agent card <pdf> --section 9.3 \\
        --hint "open-hole lamina" --backend claude-cli --out card.json
    python -m omar_pfem.verification_agent.literature_agent eval --backend claude-cli
"""
import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FEEDBACK = os.path.normpath(os.path.join(HERE, "..", "..", "..", "advisor_feedback"))

QOI_KINDS = ["displacement", "global_energy", "compliance", "reaction", "region_average",
             "pointwise_peak", "field_error_norm", "eigenvalue", "other"]
REF_TYPES = ["single_mesh", "refinement_family", "analytical", "experimental", "unspecified"]
EVIDENCE = ["none_stated", "mesh_study", "error_estimate", "independent_code"]

_field = {"type": "object", "additionalProperties": False, "required": ["value", "quote"],
          "properties": {"value": {"type": "string"}, "quote": {"type": "string"}}}
CARD_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "geometry", "material", "boundary_conditions", "loading",
                 "discretisation", "reference", "qois", "singular_features",
                 "not_stated"],
    "properties": {
        "title": {"type": "string"},
        "geometry": _field, "material": _field, "boundary_conditions": _field,
        "loading": _field, "discretisation": _field,
        "reference": {"type": "object", "additionalProperties": False,
                      "required": ["type", "convergence_evidence", "quote"],
                      "properties": {"type": {"type": "string", "enum": REF_TYPES},
                                     "convergence_evidence": {"type": "string",
                                                              "enum": EVIDENCE},
                                     "quote": {"type": "string"}}},
        "qois": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "kind", "definition", "reported_value", "quote"],
            "properties": {"name": {"type": "string"},
                           "kind": {"type": "string", "enum": QOI_KINDS},
                           "definition": {"type": "string"},
                           "reported_value": {"type": "string"},
                           "quote": {"type": "string"}}}},
        "singular_features": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["description", "inferred", "quote"],
            "properties": {"description": {"type": "string"},
                           "inferred": {"type": "boolean"},
                           "quote": {"type": "string"}}}},
        "not_stated": {"type": "array", "items": {"type": "string"}},
    },
}

PROMPT = (
    "You are the literature agent of a verification framework. From the paper section "
    "below, fill the benchmark card for the numerical example it describes. Rules: (1) every "
    "field's `quote` must be copied VERBATIM from the section (one contiguous phrase, at most "
    "about 40 words, no ellipsis); code will reject any quote not found in the paper; (2) "
    "never supply a number, material constant, mesh size or claim the section does not "
    "state: if the paper cites another source for it (e.g. 'the moduli of [49]'), say so in "
    "`value` and list the missing item in `not_stated`; (3) reference.type says how the "
    "reference solution was obtained; reference.convergence_evidence is 'none_stated' unless "
    "the section reports a mesh-convergence study, an error estimate or an independent code "
    "for that reference; (4) list every quantity of interest the section reports, with its "
    "kind; (5) singular_features: corners where a fixed edge meets a free edge, re-entrant "
    "corners, cracks, point loads or sharp notches; set inferred=true if you deduce it from "
    "the geometry and boundary conditions rather than from an explicit statement, and quote "
    "the sentence it is deduced from.\n\nSECTION:\n")


# ---------------------------------------------------------------------------
# Deterministic parts
# ---------------------------------------------------------------------------
def extract_text(pdf):
    return subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, "-"], capture_output=True,
                          text=True, check=True).stdout


def normalise(s):
    s = s.replace("­", "")
    s = re.sub(r"-\s*\n\s*", "", s)                 # hyphenation at line breaks
    for a, b in (("–", "-"), ("—", "-"), ("−", "-"), ("‘", "'"),
                 ("’", "'"), ("“", '"'), ("”", '"'), ("ﬁ", "fi"),
                 ("ﬂ", "fl"), (" ", " ")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


_HEAD = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{2,80})$", re.M)


def sections(text):
    """[(number, title, start, end)] from numbered headings."""
    heads = [(m.group(1), m.group(2).strip(), m.start()) for m in _HEAD.finditer(text)]
    out = []
    for i, (num, title, start) in enumerate(heads):
        end = heads[i + 1][2] if i + 1 < len(heads) else len(text)
        out.append((num, title, start, end))
    return out


def select_section(text, section=None, hint=None, max_chars=30000):
    """The section `section` (e.g. '9.3', with its subsections), or the one
    whose title/body best matches `hint`; trimmed to max_chars."""
    secs = sections(text)
    if section:
        chosen = [s for s in secs if s[0] == section or s[0].startswith(section + ".")]
        if chosen:
            start, end = chosen[0][2], max(s[3] for s in chosen)
            return text[start:end][:max_chars], f"section {section}: {chosen[0][1]}"
    if hint:
        words = [w for w in re.findall(r"\w+", hint.lower()) if len(w) > 2]
        best = max(secs, key=lambda s: sum(text[s[2]:s[3]].lower().count(w) for w in words)
                   + 5 * sum(w in s[1].lower() for w in words))
        return text[best[2]:best[3]][:max_chars], f"section {best[0]}: {best[1]}"
    return text[:max_chars], "start of the paper"


def check_quotes(card, text):
    """{field path: True/False}; a quote counts if its normalised form occurs
    in the normalised paper text."""
    nt = normalise(text)
    res = {}

    def chk(path, q):
        res[path] = bool(q.strip()) and normalise(q) in nt
    for f in ("geometry", "material", "boundary_conditions", "loading", "discretisation"):
        chk(f, card[f]["quote"])
    chk("reference", card["reference"]["quote"])
    for i, q in enumerate(card["qois"]):
        chk(f"qois[{i}]", q["quote"])
    for i, s in enumerate(card["singular_features"]):
        chk(f"singular_features[{i}]", s["quote"])
    return res


KIND_TO_CLASS = {"displacement": "displacement", "global_energy": "global",
                 "compliance": "global", "reaction": "global",
                 "region_average": "region_average", "pointwise_peak": "pointwise_peak",
                 "field_error_norm": "global", "eigenvalue": "global"}


def vocabulary(card):
    """QoI kinds mapped to the agent's classes; kinds the rules' name-based
    classifier does not recognise are reported as vocabulary gaps."""
    from omar_pfem.verification_agent import trees
    rows, gaps = [], []
    for q in card["qois"]:
        cls = KIND_TO_CLASS.get(q["kind"])
        rows.append({"name": q["name"], "kind": q["kind"], "class": cls})
        probe = {"compliance": "compliance", "eigenvalue": "eigenvalue",
                 "reaction": "reaction_force", "field_error_norm": "field_error_norm"}.get(
            q["kind"])
        if cls is None or (probe and trees.classify_qoi(probe) == "unknown"):
            gaps.append(f"'{q['name']}' ({q['kind']}): no rule class recognises this kind")
    return rows, gaps


def needs(card, ok):
    """What verification the reference still needs, decided in code."""
    out = []
    ref = card["reference"]
    if ref["type"] in ("single_mesh", "unspecified") or ref["convergence_evidence"] == \
            "none_stated":
        out.append("reference not verified: a refinement family (and strategies) is needed "
                   "before its values can be used to measure surrogate errors")
    if ref["convergence_evidence"] != "independent_code":
        out.append("no independent code: cross-check with a second solver")
    peaks = [q["name"] for q in card["qois"] if q["kind"] == "pointwise_peak"]
    if card["singular_features"]:
        out.append("singular features present: run the singular-order check; pointwise "
                   "peaks there have no finite limit" + (f" ({', '.join(peaks)})"
                                                         if peaks else ""))
    elif peaks:
        out.append("pointwise peaks: check boundedness (finite-radius concentration or "
                   "singularity)")
    if card["not_stated"]:
        out.append("missing inputs must be fixed (and declared) before solving: "
                   + "; ".join(card["not_stated"]))
    unsupported = [k for k, v in ok.items() if not v]
    if unsupported:
        out.append("unsupported fields (quote not found in the paper), not used: "
                   + ", ".join(unsupported))
    return out


def problem_text(card, ok):
    """Problem text for the formalizer, from quote-checked fields only."""
    parts = [card["title"] + "."]
    for f, label in (("geometry", "Geometry"), ("material", "Material"),
                     ("boundary_conditions", "Boundary conditions"), ("loading", "Loading"),
                     ("discretisation", "Discretisation in the paper")):
        if ok.get(f):
            parts.append(f"{label}: {card[f]['value']}")
    feats = [s["description"] for i, s in enumerate(card["singular_features"])
             if ok.get(f"singular_features[{i}]")]
    if feats:
        parts.append("Features: " + "; ".join(feats) + ".")
    qs = [f"{q['name']} ({q['definition']})" for i, q in enumerate(card["qois"])
          if ok.get(f"qois[{i}]")]
    parts.append("Question: which of these quantities can serve as trustworthy reference "
                 "values: " + "; ".join(qs) + "?")
    return "\n".join(parts)


def run_card(backend, pdf, section=None, hint=None):
    text = extract_text(pdf)
    sec, where = select_section(text, section, hint)
    card = backend.ask("librarian", PROMPT + sec, CARD_SCHEMA)
    ok = check_quotes(card, text)
    vocab, gaps = vocabulary(card)
    return {"pdf": os.path.basename(pdf), "section_used": where, "card": card,
            "quotes_found": ok, "quote_rate": sum(ok.values()) / max(len(ok), 1),
            "qoi_classes": vocab, "vocabulary_gaps": gaps, "needs": needs(card, ok),
            "problem_text": problem_text(card, ok)}


# ---------------------------------------------------------------------------
# Evaluation against the two problems formalised by hand (report 18, 23)
# ---------------------------------------------------------------------------
# facts the card must contain (all lower-case substrings of the field value),
# and facts it must NOT invent (numbers the section does not state)
TRUTH = {
    "openpinn_plate": {
        "pdf": "2026-11-19_Rabczuk_attachment_OpenPINN.pdf", "hint": "plate with a hole",
        "section": None,
        "must": {"geometry": ["0.25"], "material": ["210", "0.3"],
                 "reference.type": "single_mesh",
                 "reference.convergence_evidence": "none_stated",
                 "discretisation": ["0.025"]},
        "must_not": {"material": []},
        "singular_expected": True,
    },
    "cfm_lamina": {
        "pdf": "2026-11-20_Rabczuk_CFM_Constitutive-fibre_mechanics_arXiv2611.22187.pdf",
        "hint": "open-hole lamina", "section": "9.3",
        "must": {"geometry": ["50", "10", "3"], "material": ["as4"],
                 "discretisation": ["4096"],
                 "reference.type": "single_mesh"},
        # the section cites [49] for the moduli: values must not be invented
        "must_not": {"material": ["126", "11 gpa", "6.6"]},
        "qoi_names": ["compliance", "shear", "transverse"],
        "singular_expected": False,
    },
}


def score(result, truth):
    card, hits = result["card"], {}
    for key, want in truth["must"].items():
        if "." in key:
            a, b = key.split(".")
            hits[key] = card[a][b] == want
        else:
            v = card[key]["value"].lower()
            hits[key] = all(w in v for w in want)
    for key, bad in truth.get("must_not", {}).items():
        v = card[key]["value"].lower()
        hits[f"{key} invents nothing"] = not any(b in v for b in bad)
    if "qoi_names" in truth:
        names = " ".join(q["name"].lower() + " " + q["definition"].lower()
                         for q in card["qois"])
        hits["QoIs found"] = all(w in names for w in truth["qoi_names"])
    hits["singular features as expected"] = bool(card["singular_features"]) == \
        truth["singular_expected"]
    hits["reference flagged as unverified"] = any(n.startswith("reference not verified")
                                                  for n in result["needs"])
    return hits


def evaluate(backend, repeats=1):
    out = {}
    for name, t in TRUTH.items():
        runs = []
        for _ in range(repeats):
            r = run_card(backend, os.path.join(FEEDBACK, t["pdf"]), t["section"], t["hint"])
            r["score"] = score(r, t)
            runs.append(r)
            print(f"{name}: section '{r['section_used']}', quotes {r['quote_rate']:.0%}, "
                  f"score {sum(r['score'].values())}/{len(r['score'])} "
                  f"{[k for k, v in r['score'].items() if not v]}", flush=True)
        out[name] = runs
    if hasattr(backend, "cost_usd"):
        out["_cost_usd"] = backend.cost_usd
    return out


def main(argv=None):
    from omar_pfem.verification_agent.llm_roles import make_backend
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["card", "eval"])
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--section")
    ap.add_argument("--hint")
    ap.add_argument("--backend", default="replay")
    ap.add_argument("--replay")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    be = make_backend(a.backend, a.replay)
    res = run_card(be, a.pdf, a.section, a.hint) if a.mode == "card" else evaluate(be, a.repeats)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    if a.mode == "card":
        print(json.dumps({k: res[k] for k in ("section_used", "quote_rate", "vocabulary_gaps",
                                              "needs", "problem_text")}, indent=1))


if __name__ == "__main__":
    main()
