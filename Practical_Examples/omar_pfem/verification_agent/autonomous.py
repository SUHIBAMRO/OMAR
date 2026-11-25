"""The autonomous verification workflow, end to end (round O, 2026-11-24).

Professor Rabczuk (24 Nov): "aim for an end-to-end workflow first, starting from literature
over problem definition, meshing, MMS (if suitable), simulation, convergence analysis and
diagnosis ... the agent should decide what to investigate next and learn from the results."

One command runs every step and records, for each, who did it (code, LLM, or a person), so
the autonomy is measured, not claimed:

  L  literature    literature_agent: a quote-checked benchmark card from a paper (LLM fills,
                   code checks every quote). Skipped when a problem text is given (logged as
                   a person's input).
  D  definition    formalizer (LLM) -> specification (code checks it); modeler (LLM) -> a
                   pipeline2d description; code checks it (pipeline2d.validate) and traces
                   every number in it to the source text; up to two corrections are asked
                   of the LLM with the errors; numbers it had to ASSUME are flagged for a
                   person, never silently accepted.
  M  meshing       pipeline2d: the domain, the corner analysis and the meshes, by code.
  V  MMS           mms_auto: decides whether code verification is needed, generates the
                   manufactured solution on the problem's own domain, checks the rates.
  S  simulation    uniform Q2 ladder, strategies, independent P2 ladder (pipeline2d).
  C  convergence   rule plan with the ADOPTED rules (adopted.py), cross-check.
  G  diagnosis     diagnose.py's hypotheses, on facts derived from the description.
  N  next steps    experiment.py: candidates, choice by prior x value / cost, frozen
                   prediction, run within the budget, scored, remembered (experience.json).

    python -m omar_pfem.verification_agent.autonomous --description problems2d/x.json
    python -m omar_pfem.verification_agent.autonomous --text problem.txt --backend claude-cli
    python -m omar_pfem.verification_agent.autonomous --pdf paper.pdf --hint "..." --backend claude-cli
"""
import argparse
import json
import math
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "autonomous_runs")

MODELER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["supported", "unsupported_reason", "description", "sources", "assumptions"],
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_reason": {"type": "string"},
        "description": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "material", "geometry_kind", "cell_size", "cells", "outer",
                         "holes", "E", "nu", "plane", "bcs", "points"],
            "properties": {
                "name": {"type": "string"},
                "geometry_kind": {"type": "string", "enum": ["cells", "polygon"]},
                "outer": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "holes": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["kind", "circle", "polygon"],
                    "properties": {"kind": {"type": "string", "enum": ["circle", "polygon"]},
                                   "circle": {"type": "array", "items": {"type": "number"}},
                                   "polygon": {"type": "array", "items": {
                                       "type": "array", "items": {"type": "number"}}}}}},
                "material": {"type": "string", "enum": ["linear_elastic", "neo_hookean"]},
                "cell_size": {"type": "number"},
                "cells": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "E": {"type": "number"}, "nu": {"type": "number"},
                "plane": {"type": "string", "enum": ["strain", "stress"]},
                "bcs": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["segment", "hole", "type", "t", "p", "u"],
                    "properties": {
                        "hole": {"type": "integer"},
                        "segment": {"type": "array", "items": {"type": "array",
                                                               "items": {"type": "number"}}},
                        "type": {"type": "string",
                                 "enum": ["clamped", "symmetry", "traction", "pressure"]},
                        "t": {"type": "array", "items": {"type": "number"}},
                        "p": {"type": "number"},
                        "u": {"type": "array", "items": {"type": "number"}}}}},
                "points": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "x", "y", "component"],
                    "properties": {"name": {"type": "string"}, "x": {"type": "number"},
                                   "y": {"type": "number"},
                                   "component": {"type": "string", "enum": ["x", "y"]}}}}}},
        "sources": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["item", "value", "quote"],
            "properties": {"item": {"type": "string"}, "value": {"type": "string"},
                           "quote": {"type": "string"}}}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
}

MODELER_PROMPT = (
    "You are the modeler of a verification framework. Turn the problem below into an input "
    "description for a 2D finite element pipeline, or say it is not supported. Materials: "
    "linear_elastic (small strain; plane strain or plane stress) or neo_hookean (compressible "
    "neo-Hookean hyperelasticity at finite strain, W = mu/2 (tr C - 3 - 2 ln J) + lam/2 "
    "(ln J)^2 with mu, lam from E and nu; plane strain only; only dead tractions, no pressure "
    "and no prescribed non-zero displacement). Choose neo_hookean only if the problem states a "
    "neo-Hookean (or compatible hyperelastic) material; another hyperelastic model "
    "(Mooney-Rivlin, Ogden, incompressible) is NOT supported. Geometry: either "
    "geometry_kind='polygon' (preferred for anything but a union of squares): `outer` lists "
    "the vertices of a straight-edged outer boundary (inclined edges allowed), `holes` lists "
    "circular holes (kind='circle', circle=[cx, cy, r], polygon=[]) or polygonal holes "
    "(kind='polygon', circle=[], polygon=[[x, y], ...]); holes may cut the outer boundary (e.g. "
    "a quarter model of a plate with a hole); set cell_size=1 and cells=[] then; a boundary "
    "condition on a whole hole uses hole=<index> and segment=[]; on a straight edge it uses "
    "hole=-1 and a segment; points of interest may lie anywhere in the domain. Or "
    "geometry_kind='cells' as described next (outer=[], holes=[]). For geometry_kind='cells' the pipeline can only represent domains that are unions of equal square "
    "cells: cell [i, j] covers [i a, (i+1) a] x [j a, (j+1) a] with a = cell_size; no holes, "
    "no curved or inclined edges (use geometry_kind='polygon' for those). Boundary conditions act on straight boundary segments "
    "given by their end points in physical coordinates: clamped (all displacement zero; "
    "optionally a constant prescribed displacement u = [ux, uy]), symmetry (normal "
    "displacement zero, shear traction zero; only on segments parallel to an axis), traction "
    "(a constant traction vector t = [tx, ty] per unit length), pressure (p acting against "
    "the outward normal). Edges not listed are traction-free. Use t = [], p = 0, u = [] where "
    "a field does not apply. Points of interest must lie on cell corners or edge midpoints "
    "(multiples of a/2); give each a short snake_case name starting with u_ or v_ for the "
    "x or y displacement. Rules: (1) use ONLY numbers stated in the problem; for each number "
    "you use, add a `sources` entry with a verbatim quote containing it; (2) if the problem "
    "does not state a value the pipeline needs (e.g. Young's modulus for a displacement "
    "quantity), choose a value, and list it in `assumptions` with the reason; (3) if the "
    "geometry or a condition cannot be represented exactly (a curved OUTER edge, an ellipse, "
    "a distributed load that is not constant, a point load, contact, another nonlinearity), set "
    "supported=false and explain in unsupported_reason; never approximate silently.\n\n"
    "PROBLEM:\n")


# ------------------------------------------------------------------ autonomy log ----------
class Log:
    def __init__(self, name):
        self.name, self.steps, self.t0 = name, [], time.time()
        self.dir = os.path.join(RUNS, name)
        os.makedirs(self.dir, exist_ok=True)

    def step(self, stage, actor, status, detail, **extra):
        self.steps.append({"stage": stage, "actor": actor, "status": status,
                           "detail": detail, **extra})
        print(f"[{stage}] ({actor}) {status}: {detail}", flush=True)

    def human_points(self):
        return [s for s in self.steps if s["actor"] == "person" or s["status"] == "needs_person"]

    def save(self, result):
        json.dump({"steps": self.steps, "human_points": self.human_points(),
                   "wall_clock_s": time.time() - self.t0, **result},
                  open(os.path.join(self.dir, "run.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ definition ------------
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text):
    out = []
    for m in _NUM.finditer(text.replace("×", "x").replace("−", "-")):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    # "10^5" / "10^{-3}" forms
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[x*]?\s*10\^\{?([-+]?\d+)\}?", text):
        out.append(float(m.group(1)) * 10 ** int(m.group(2)) if "x" in m.group(0) or "*" in
                   m.group(0) else 10 ** int(m.group(2)))
    return out


def trace_numbers(desc, text, assumptions):
    """Every physical number of the description must be found in the source text (relative
    1e-9), be 0, or be declared as an assumption.
    POST HOC (24 Nov, after the pre-registered autonomy test): the cell size is a meshing
    choice, not a physical input, so it is no longer traced; instead every vertex coordinate of
    the domain (physical units) must be a stated number, or half or a quarter of one
    (symmetry models), or 0."""
    src = numbers_in(text)
    declared = numbers_in(" ".join(assumptions))
    found = lambda v: any(abs(v - s) <= 1e-9 * max(1.0, abs(s)) for s in src)
    assumed = lambda v: any(abs(v - s) <= 1e-9 * max(1.0, abs(s)) for s in declared)
    a = float(desc.get("cell_size", 1.0))
    items = [("E", desc["E"]), ("nu", desc["nu"])]
    for k, b in enumerate(desc["bcs"]):
        items += [(f"bcs[{k}].t", v) for v in b.get("t", [])]
        if b["type"] == "pressure":
            items.append((f"bcs[{k}].p", b["p"]))
        items += [(f"bcs[{k}].u", v) for v in b.get("u", [])]
    if "domain" in desc:
        a = 1.0
        for x, y in desc["domain"]["outer"]:
            items += [("vertex x", abs(x)), ("vertex y", abs(y))]
        for h in desc["domain"].get("holes", []):
            if "circle" in h:
                items += [("hole centre", abs(h["circle"][0])), ("hole centre", abs(h["circle"][1])),
                          ("hole radius", abs(h["circle"][2]))]
            else:
                for x, y in h["polygon"]:
                    items += [("vertex x", abs(x)), ("vertex y", abs(y))]
    else:
        from omar_pfem.verification_agent.pipeline2d import polygon
        for x, y, _ in polygon(desc):
            items += [("vertex x", abs(x * a)), ("vertex y", abs(y * a))]
    unsupported, flagged = [], []
    for name, v in items:
        if v == 0 or found(v):
            continue
        if (name.startswith("vertex") or name.startswith("hole")) and any(
                found(v * k) for k in (2, 4)):
            continue        # half or quarter of a stated dimension (symmetry models)
        if assumed(v):
            flagged.append(f"{name} = {v:g} (declared assumption)")
        else:
            unsupported.append(f"{name} = {v:g} is not in the source text and not declared")
    return unsupported, flagged


def to_pipeline(desc):
    """Modeler output -> pipeline2d / geometry2d description (drop empty optional fields)."""
    if desc.get("geometry_kind") == "polygon":
        out = {k: desc[k] for k in ("name", "E", "nu", "plane")}
        holes = [{"circle": h["circle"]} if h["kind"] == "circle" else {"polygon": h["polygon"]}
                 for h in desc.get("holes", [])]
        out["domain"] = {"outer": desc["outer"], "holes": holes}
    else:
        out = {k: desc[k] for k in ("name", "cell_size", "cells", "E", "nu", "plane")}
    out["material"] = desc.get("material", "linear_elastic")
    out["bcs"] = []
    for b in desc["bcs"]:
        if b.get("hole", -1) is not None and b.get("hole", -1) >= 0:
            d = {"hole": b["hole"], "type": b["type"]}
        else:
            d = {"segment": b["segment"], "type": b["type"]}
        if b["type"] == "traction":
            d["t"] = b["t"]
        if b["type"] == "pressure":
            d["p"] = b["p"]
        if b["type"] == "clamped" and b.get("u"):
            d["u"] = b["u"]
        out["bcs"].append(d)
    out["points"] = {p["name"]: [p["x"], p["y"], p["component"]] for p in desc["points"]}
    return out


def define(backend, text, log, retries=2):
    from omar_pfem.verification_agent import llm_roles
    from omar_pfem.verification_agent.pipeline2d import validate as validate_le
    from omar_pfem.verification_agent.pipeline2d_nh import validate as validate_nh
    from omar_pfem.verification_agent import geometry2d as G

    def validate(d):
        if "domain" in d:
            e = G.validate(d)
            if not e:
                try:
                    G.make(d)
                except ValueError as ex:
                    e = [str(ex)]
            return e
        return validate_nh(d) if d.get("material") == "neo_hookean" else validate_le(d)
    spec, problems = llm_roles.formalize(backend, text)
    log.step("D", "LLM", "done" if not problems else "checked",
             f"formalizer specification; code check: {problems or 'no problems'}",
             spec=spec)
    prompt, feedback = MODELER_PROMPT + text, ""
    for attempt in range(retries + 1):
        m = backend.ask("modeler", prompt + feedback, MODELER_SCHEMA)
        if not m["supported"]:
            log.step("D", "LLM", "unsupported", m["unsupported_reason"], modeler=m)
            return None, m
        desc = to_pipeline(m["description"])
        errs = validate(desc)
        unsup, flagged = trace_numbers(desc, text, m["assumptions"])
        log.step("D", "code", "checked", f"attempt {attempt + 1}: validation "
                 f"{errs or 'ok'}; untraced numbers {unsup or 'none'}; assumptions {flagged or 'none'}",
                 description=desc)
        if not errs and not unsup:
            # POST HOC (24 Nov): only an assumed PHYSICAL number needs a person; prose
            # clarifications in `assumptions` are recorded but not flagged
            if flagged:
                # NEW RULE (25 Nov, sensitivity_preregistration.json, defined before any run):
                # assumed inputs are checked by the sensitivity step after certification; a
                # person is asked only where an input matters for a certified quantity, or
                # where it cannot be perturbed automatically
                from omar_pfem.verification_agent.sensitivity import param_name
                names = [param_name(f) for f in flagged]
                desc["assumed"] = sorted({n for n in names if n})
                manual = [f for f, n in zip(flagged, names) if n is None]
                log.step("D", "code", "deferred", "assumed inputs, checked by sensitivity after "
                         "certification: " + "; ".join(flagged))
                if manual:
                    log.step("D", "person", "needs_person", "assumed inputs that cannot be "
                             "perturbed automatically: " + "; ".join(manual))
            elif m["assumptions"]:
                log.step("D", "LLM", "noted", "modeler notes (no assumed physical input): "
                         + "; ".join(m["assumptions"])[:400])
            return desc, m
        feedback = ("\n\nYOUR PREVIOUS DESCRIPTION WAS REJECTED BY CODE:\n- "
                    + "\n- ".join(errs + unsup) + "\nCorrect it.")
    log.step("D", "code", "failed", "no valid description after corrections")
    return None, m


# ------------------------------------------------------------------ diagnosis -------------
def diagnose_generic(prob, verdicts, v, ad, mms):
    """diagnose.py's hypotheses with facts derived from the description (no hand-written
    facts). Tests that need facts this problem does not have return 'needs data'."""
    from omar_pfem.verification_agent import diagnose as D
    from omar_pfem.verification_agent import trees
    code_ok = all(r.get("verdict") == "PASS" for r in mms.values())
    facts = {"name": prob.name, "element": "Q9", "degree": 2, "quadrature": "full",
             "nu": prob.nu, "material": getattr(prob, "material", "linear_elastic"),
             "plane": prob.plane,
             "corners": [{"where": f"corner ({c['x']}, {c['y']}) {c['edges']}",
                          "lam": min(c["rate_limiting"])} for c in prob.corners],
             "corners_in_qoi_region": bool(prob.corners), "feature_radius": None,
             "code_check": lambda: (code_ok, "automatic MMS (mms_auto): " +
                                    ", ".join(f"{k} {r.get('verdict')}" for k, r in mms.items())),
             "tolerance_check": lambda: (None, "direct sparse solver; no tolerance to vary"),
             "linear_solver": ("PARDISO direct", lambda: (None, "no second linear solver run")),
             "fullint": None, "facts_source": "derived from the description"}
    allv = {q: (r["verdict"], r["evidence"]) for q, r in verdicts.items()}
    out = {}
    for q, r in verdicts.items():
        needs = r["verdict"] != "PASS" or r["evidence"] != "rate"
        c = D.Ctx(prob.name, facts, q, trees.classify_qoi(q), r["verdict"], r["evidence"],
                  v[q], allv, ad, v)
        res = []
        for h in D.HYPOTHESES:
            if not needs and h.id != "M2":
                continue
            try:
                if not h.applies(c):
                    continue
                t = h.test(c)
            except Exception as e:                    # recorded, not hidden
                t = {"status": D.NEEDS_DATA, "evidence": f"test not available here ({e!r:.80})"}
            if t["status"] == D.NOT_APPLICABLE:
                continue
            res.append({"id": h.id, "class": h.cls, **t})
        if res:
            sup = [x for x in res if x["status"] == D.SUPPORTED]
            out[q] = {"supported": [x["id"] for x in sup],
                      "diagnosis": sorted({x["class"] for x in sup}),
                      "open": [x["id"] for x in res if x["status"] == D.NEEDS_DATA],
                      "tests": res}
    return out


# ------------------------------------------------------------------ the loop --------------
def verify_state(prob, ladders, levels):
    from omar_pfem.verification_agent import pipeline2d as P
    uni = ladders.get("Q2 uniform", levels, None, "Q2")
    strongest = min(prob.corners, key=lambda c: min(c["rate_limiting"])) if prob.corners else None
    strat = []
    for g in P.GRADINGS:
        if strongest:
            strat += ladders.get(f"Q2 strategy g={g}", (P.STRATEGY_N,),
                                 {"x": {strongest["x"]: g}, "y": {strongest["y"]: g}}, "Q2")
    strat = [r for r in uni if r["n"] == P.STRATEGY_N] + strat
    p2 = ladders.get("P2 uniform", levels, None, "P2")
    out, v, ad = P.framework(prob, uni[-5:] if len(uni) > 5 else uni, strat, full=True)
    return P.cross_check(out, p2[-5:] if len(p2) > 5 else p2), v, ad


def graded_state(prob, ladders, levels, uni_rows):
    from omar_pfem.verification_agent import pipeline2d as P
    from omar_pfem.verification_agent.generic2d import graded_check
    plan = P.plan_intensity(prob, uni_rows)
    planned = {a: {int(k) if float(k).is_integer() else k: g for k, g in plan["plan"][a].items()}
               for a in ("x", "y")}
    gq = ladders.get("Q2 graded", levels, planned, "Q2")
    gp = ladders.get("P2 graded", levels, planned, "P2")
    res = P.cross_check({q: {**graded_check(gq, q), "class": __import__(
        "omar_pfem.verification_agent.trees", fromlist=["x"]).classify_qoi(q), "checks": {}}
        for q in prob.qois}, gp)
    return res, plan


def summary(V):
    return {"certified": sorted(q for q, r in V.items() if r.get("final") == "CERTIFIED"),
            "open": sorted(q for q, r in V.items() if r.get("final") != "CERTIFIED")}


def _small_load(prob, factor, levels):
    """The same problem with every traction scaled by `factor` (a diagnostic, not the
    problem asked): its five-level strain-energy order."""
    import copy
    from omar_pfem.verification_agent import gates
    from omar_pfem.verification_agent import pipeline2d as P
    d = copy.deepcopy(prob.desc)
    d["name"] = prob.name + f"_load{factor:g}"
    for b in d["bcs"]:
        if b["type"] == "traction":
            b["t"] = [factor * t for t in b["t"]]
    p2 = type(prob)(d)
    rows = P.Ladders(p2, log=lambda s: None).get("Q2 uniform", levels, None, "Q2")
    g = gates.multilevel_convergence_gate([r["h"] for r in rows], [r["strain_energy"] for r in rows],
                                          order_window=(0.05, 20))
    return g["fit_order"], [r["min_detF"] for r in rows]


def _execute(c, prob, ladders, V, v, ad, graded, lv, glv, result, small_load_done):
    kind = c["run"]["kind"]
    if kind in ("uniform_level", "p2_level"):
        lv = lv + [c["run"]["n"]]
        V, v, ad = verify_state(prob, ladders, lv)
        new = V
    elif kind == "graded_level":
        glv = glv + [c["run"]["n"]]
        uni = ladders.get("Q2 uniform", lv, None, "Q2")
        graded, plan = graded_state(prob, ladders, glv[-5:], uni)
        new = graded
    elif kind == "graded":
        uni = ladders.get("Q2 uniform", lv, None, "Q2")
        graded, plan = graded_state(prob, ladders, glv, uni)
        new = graded
        result["graded_plan"] = plan["plan"]
    elif kind == "small_load":
        order, js = _small_load(prob, c["run"]["factor"], lv)
        new = dict(V)
        new["_small_load_order"] = order
        result["small_load_check"] = {"factor": c["run"]["factor"], "energy_order": order,
                                      "min_detF": js, "linearized_2lam": 2 * prob.lam_min}
        small_load_done = True
    else:
        raise ValueError(kind)
    return new, V, v, ad, graded, lv, glv, small_load_done


def run(desc=None, text=None, pdf=None, hint=None, section=None, backend_name="replay",
        replay=None, budget_minutes=20.0, max_experiments=4, levels=(2, 4, 8, 16, 32),
        chosen_by="person"):
    from omar_pfem.verification_agent import experiment as X
    from omar_pfem.verification_agent import llm_roles, mms_auto
    from omar_pfem.verification_agent import pipeline2d as P
    backend = llm_roles.make_backend(backend_name, replay) if (text or pdf) else None
    name = (desc or {}).get("name") or os.path.splitext(os.path.basename(pdf or text or "x"))[0]
    log = Log(name)
    result = {"source": {"pdf": pdf, "text": text, "description_given": desc is not None}}
    # L
    if pdf:
        if section is None:       # POST HOC (24 Nov): passage selector, not headings
            from omar_pfem.verification_agent.literature_select import run_card as rc
            card = rc(backend, pdf, hint)
        else:
            from omar_pfem.verification_agent.literature_agent import run_card
            card = run_card(backend, pdf, section, hint)
        log.step("L", "LLM", "done", f"card from {card['section_used']}; quotes found "
                 f"{card['quote_rate']:.0%}", card=card)
        if section is None:
            log.step("L", "code", "done", "section chosen by code from the hint")
        problem_text = card["problem_text"]
        if chosen_by == "person":
            log.step("L", "person", "input", f"paper and hint chosen by a person ({hint!r})")
        else:       # 2026-11-25: the literature search chose them (fixed after its first run)
            log.step("L", "code", "done", f"paper and hint chosen by {chosen_by} ({hint!r})")
    elif text:
        problem_text = open(text).read() if os.path.exists(text) else text
        log.step("L", "person", "input", "problem text written by a person (literature step "
                 "not used)")
    else:
        problem_text = None
        log.step("L", "person", "input", "description written by a person (steps L, D skipped)")
    # D
    if desc is None:
        desc, m = define(backend, problem_text, log)
        result["modeler"] = m
        if desc is None:
            log.save(result)
            return result
    from omar_pfem.verification_agent import geometry2d as G
    prob = G.make(desc)
    if "domain" in desc:
        log.step("M", "code", "note", "general domain meshed by gmsh (quadratic triangles and "
                 "independent quadratic quadrilaterals, curved edges on circles); corners from "
                 "the real boundary after the holes are cut")
    if desc.get("material") == "neo_hookean":
        log.step("M", "code", "note", "neo-Hookean: corner exponents from the linearized theory "
                 "(a hypothesis for the practical mesh range); dead loads; load steps chosen by "
                 "the solver (doubled on Newton failure or J <= 0)")
    log.step("M", "code", "done", f"rate-limiting corners (x, y, smallest exponent) "
             f"{[(round(c['x'] * prob.a, 6), round(c['y'] * prob.a, 6), round(min(c['rate_limiting']), 4)) for c in prob.corners]}")
    # V
    need = mms_auto.needed(prob)
    log.step("V", "code", "decided", "; ".join(f"{k}: {'needed' if n else 'not needed'} ({r})"
                                               for k, (n, r) in need.items()))
    mms = mms_auto.run(prob, log=lambda s: None)
    log.step("V", "code", "PASS" if all(r["verdict"] == "PASS" for r in mms.values()) else "FAIL",
             "; ".join(f"{k} {r['verdict']}" + ("" if r.get("skipped") else
                                                 f" (L2 {r['rates']['L2']['observed']:.2f}, "
                                                 f"H1 {r['rates']['H1']['observed']:.2f}, levels "
                                                 f"{r['levels']})") for k, r in mms.items()))
    result["mms"] = mms
    if not all(r["verdict"] == "PASS" for r in mms.values()):
        log.step("V", "code", "stop", "code verification failed: no solution verification on an "
                 "unverified solver")
        log.save(result)
        return result
    # S, C
    ladders = P.Ladders(prob, log=lambda s: None)
    lv = list(levels)
    V, v, ad = verify_state(prob, ladders, lv)
    log.step("S", "code", "done", f"Q2 and P2 ladders n = {lv}")
    log.step("C", "code", "done", f"adopted rules: {summary(V)}")
    # G
    diag = diagnose_generic(prob, V, v, ad, mms)
    log.step("G", "code", "done", "; ".join(f"{q}: supported {d['supported']}" for q, d in diag.items()))
    # N
    graded, spent, k = None, 0.0, 0
    history, glv = [], list(lv)
    certified_any = set(summary(V)["certified"])
    cert_info = {}

    def note_certified(state):
        for q, r in state.items():
            if not q.startswith("_") and r.get("final") == "CERTIFIED" and q not in cert_info:
                cert_info[q] = {"U": r.get("U"), "limit": r.get("limit"), "finest": r.get("finest")}
    note_certified(V)
    banned, small_load_done = set(), False
    while k < max_experiments:
        cur = graded if graded is not None else V
        cands, concl = X.candidates({"prob": prob, "verdicts": cur, "graded": graded,
                                     "levels": glv if graded is not None else lv,
                                     "ladders": ladders, "diagnosis": diag,
                                     "certified_any": certified_any, "banned": banned,
                                     "small_load_done": small_load_done,
                                     # 2026-11-25: n = 128 neo-Hookean solves took far longer
                                     # than the cost model predicted; cap the level
                                     "max_n": 64 if getattr(prob, "material", "") == "neo_hookean"
                                     else 128})
        for c in concl:
            if not any(h.get("quantity") == c["quantity"] for h in history):
                log.step("N", "code", "concluded", f"{c['quantity']}: {c['conclusion']} "
                         f"({c['justification']})")
                history.append(c)
        cands = [c for c in cands if spent + c["cost"]["minutes"] <= budget_minutes]
        if not cands:
            log.step("N", "code", "stop", "no further experiment applies within the budget "
                     f"({spent:.1f} of {budget_minutes} min used)")
            break
        c = cands[0]
        k += 1
        path, rec = X.freeze(prob.name, k, c)
        log.step("N", "code", "pre-registered", f"#{k} {c['action']} on {c['quantities']}: "
                 f"{c['hypothesis']}; prediction: {c['prediction']['text']}; prior "
                 f"{c['prior']:.2f} (n={c['prior_n']}); est. {c['cost']['minutes']:.1f} min",
                 file=os.path.relpath(path, HERE), sha256=rec["sha256"])
        t0 = time.time()
        old = V
        try:
            new, V, v, ad, graded, lv, glv, small_load_done = _execute(
                c, prob, ladders, V, v, ad, graded, lv, glv, result, small_load_done)
        except Exception as e:                          # a failed experiment is an outcome
            minutes = (time.time() - t0) / 60
            spent += minutes
            banned.add(c["action"])
            X.complete(path, {"confirmed": False, "failed": repr(e)[:300],
                              "observed_minutes": minutes})
            log.step("N", "code", "failed", f"#{k} {c['action']} could not be run: {e!s:.200}; "
                     "recorded as refuted, not tried again on this problem")
            history.append({"k": k, "action": c["action"], "confirmed": False, "failed": True})
            continue
        minutes = (time.time() - t0) / 60
        spent += minutes
        old = {q: ({**old[q], "final": "CERTIFIED"} if q in certified_any else old[q]) for q in old}
        ok = X.evaluate(c, old, new)
        certified_any |= set(summary({q: r for q, r in new.items() if not q.startswith("_")})["certified"])
        note_certified({q: r for q, r in new.items() if not q.startswith("_")})
        X.complete(path, {"confirmed": ok, "observed_minutes": minutes,
                          "after": summary({q: r for q, r in new.items() if not q.startswith("_")}),
                          **({"small_load_order": new["_small_load_order"]}
                             if "_small_load_order" in new else {})})
        log.step("N", "code", "confirmed" if ok else "refuted",
                 f"#{k} {c['action']}: prediction {'confirmed' if ok else 'refuted'} "
                 f"({minutes:.1f} min); certified so far {sorted(certified_any)}")
        history.append({"k": k, "action": c["action"], "confirmed": ok})
    final = {"certified": sorted(certified_any),
             "open": sorted(q for q in prob.qois if q not in certified_any)}
    result["certified_info"] = cert_info
    # A: sensitivity of assumed inputs (only certified quantities have a known accuracy)
    assumed = desc.get("assumed", [])
    if assumed and cert_info:
        from omar_pfem.verification_agent import sensitivity as SE
        rec = SE.analyse({k: v for k, v in desc.items() if k != "assumed"}, assumed, cert_info,
                         log=lambda s: None)
        result["sensitivity"] = rec
        for name, r in rec["params"].items():
            if r.get("needs_person"):
                why = (r["status"] if "status" in r else "changes " + ", ".join(
                    f"{q} by {r['quantities'][q]['change_rel']:.3g} (numerical U "
                    f"{r['quantities'][q]['U_rel']:.2g})" for q in r["matters_for"])
                    + f" [post hoc proposal: person needed only for "
                    f"{r['posthoc_person_needed_for'] or 'none'}]")
                log.step("A", "person", "needs_person", f"assumed {name}: {why}; a person "
                         "must confirm this input")
            else:
                log.step("A", "code", "negligible", f"assumed {name}: changes every certified "
                         "quantity by less than its numerical uncertainty over the range; "
                         "no person needed")
    elif assumed:
        log.step("A", "person", "needs_person", f"assumed inputs {assumed}: nothing certified, "
                 "their effect cannot be judged; a person must confirm them")
    result.update(final=final, verdicts=V, graded=graded, diagnosis=diag, experiments=history,
                  description=desc)
    log.save(result)
    auto = [s for s in log.steps if s["actor"] in ("code", "LLM")]
    print(f"steps by code or LLM: {len(auto)}; points needing a person: {len(log.human_points())}")
    return result


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--description")
    ap.add_argument("--text")
    ap.add_argument("--pdf")
    ap.add_argument("--hint")
    ap.add_argument("--section")
    ap.add_argument("--backend", default="replay")
    ap.add_argument("--replay")
    ap.add_argument("--budget-minutes", type=float, default=20.0)
    ap.add_argument("--max-experiments", type=int, default=4)
    ap.add_argument("--assume", nargs="+", help="inputs of a given description to treat as "
                    "assumed (checked by the sensitivity step), e.g. E nu 'bcs[2].t'")
    a = ap.parse_args(argv)
    desc = json.load(open(a.description)) if a.description else None
    if desc is not None and a.assume:
        desc["assumed"] = a.assume
    run(desc, a.text, a.pdf, a.hint, a.section, a.backend, a.replay, a.budget_minutes,
        a.max_experiments)


if __name__ == "__main__":
    main()
