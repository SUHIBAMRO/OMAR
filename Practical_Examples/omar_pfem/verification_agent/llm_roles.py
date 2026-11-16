"""LLM orchestration layer on top of the deterministic verification core.

Four roles, GRAFT-style (Toscano et al.), adapted to computational solid
mechanics:

  formalizer  natural-language problem -> ProblemSpec + well-posedness notes
  proposer    ProblemSpec + nearest solved cases -> verification plan, chosen
              ONLY from the registered check catalogue below
  critic      reviews the plan against the spec and repository; may veto
  advisor     interprets the gate outputs in words for the report

The authority split is the point of the design:
  * the LLM never computes a verdict and never writes `accepted`;
  * every plan is validated in code against CHECK_CATALOGUE before anything
    runs (unknown checks, out-of-range tolerances, missing QoIs -> rejected);
  * the advisor's text is cross-checked against the gate verdicts, and any
    QoI it calls "converged"/"admitted" that a gate failed is flagged.

Backends (pick with --backend):
  anthropic  Anthropic Python SDK; needs ANTHROPIC_API_KEY or `ant auth login`
  claude-cli headless `claude -p --json-schema` (what GRAFT's harness does)
  replay     offline: answers come from a JSON file, for tests and audits
"""
import json
import os
import re
import shutil
import subprocess

MODEL = "claude-opus-5-5"

# ----------------------------------------------------------------------------
# Check catalogue: the only actions a plan may contain. Each entry names the
# deterministic command that implements it and the bounds the code enforces
# on LLM-chosen parameters.
# ----------------------------------------------------------------------------
CHECK_CATALOGUE = {
    "mms_rates": {
        "implements": "agent.py mms (gates.mms_rate_gate)",
        "applies_when": "code verification; smooth manufactured solution available",
        "params": {"tol": (0.05, 0.3)},
    },
    "qoi_convergence": {
        "implements": "gates.qoi_convergence_gate on a mesh-refinement family",
        "applies_when": "solution verification of a physical-problem QoI",
        "params": {"band_tol": (0.001, 0.02), "gci_tol": (0.01, 0.10)},
    },
    "strategy_robustness": {
        "implements": "gates.strategy_robustness_gate across refinement strategies",
        "applies_when": "QoI sensitive to where the mesh is refined (local features)",
        "params": {"tol": (0.005, 0.05)},
    },
}
ELEMENTS = ("Q4", "Q8", "Q9", "Q20", "H8", "H27", "T3", "T6", "TET4", "TET10")
REGULARITY = ("smooth", "clamped_free_corner", "stress_concentration",
              "reentrant_corner", "crack", "contact", "unknown")

# ----------------------------------------------------------------------------
# JSON schemas for structured output
# ----------------------------------------------------------------------------
_STR = {"type": "string"}
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": _STR, "geometry": _STR, "dim": {"type": "integer", "enum": [1, 2, 3]},
        "material": _STR, "loading": _STR,
        "regularity": {"type": "string", "enum": list(REGULARITY)},
        "element": {"type": "string", "enum": list(ELEMENTS)},
        "reference": {"type": "string", "enum": ["exact_manufactured", "mesh_refinement_family",
                                                 "higher_order_reference", "experiment"]},
        "qois": {"type": "array", "items": _STR},
        "well_posedness": {"type": "array", "items": _STR},
        "open_questions": {"type": "array", "items": _STR},
    },
    "required": ["name", "geometry", "dim", "material", "loading", "regularity", "element",
                 "reference", "qois", "well_posedness", "open_questions"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "check": {"type": "string", "enum": list(CHECK_CATALOGUE)},
                "qois": {"type": "array", "items": _STR},
                "params": {"type": "object", "properties": {
                    k: {"type": "number"} for c in CHECK_CATALOGUE.values() for k in c["params"]},
                    "additionalProperties": False},
                "rationale": _STR,
            },
            "required": ["check", "qois", "params", "rationale"],
            "additionalProperties": False}},
        "expected_failures": {"type": "array", "items": {
            "type": "object", "properties": {"qoi": _STR, "why": _STR},
            "required": ["qoi", "why"], "additionalProperties": False}},
    },
    "required": ["checks", "expected_failures"],
    "additionalProperties": False,
}
CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "objections": {"type": "array", "items": _STR},
        "missing_checks": {"type": "array", "items": _STR},
    },
    "required": ["approve", "objections", "missing_checks"],
    "additionalProperties": False,
}
ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "per_qoi": {"type": "array", "items": {
            "type": "object",
            "properties": {"qoi": _STR, "stated_verdict": {"type": "string",
                                                          "enum": ["PASS", "FAIL", "INCONCLUSIVE"]},
                           "interpretation": _STR},
            "required": ["qoi", "stated_verdict", "interpretation"],
            "additionalProperties": False}},
        "summary": _STR,
        "manuscript_wording": _STR,
    },
    "required": ["per_qoi", "summary", "manuscript_wording"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are part of a verification pipeline for finite-element and neural-operator "
    "results in nonlinear solid mechanics (hyperelasticity). Deterministic code computes "
    "every verdict; your job is to formalize problems, choose checks from a fixed catalogue, "
    "critique plans, and explain results. Never invent numbers. If information is missing, "
    "say so in open_questions or objections instead of guessing. Known theory you may rely "
    "on: corners where a clamped edge meets a traction-free edge, re-entrant corners, cracks "
    "and sharp grooves produce singular or near-singular stress, so pointwise peak stress "
    "need not converge under refinement while energy, displacements and region averages do; "
    "Q4 converges at O(h^2) in L2 and O(h) in H1/stress for smooth solutions, Q9 one order "
    "higher.")

# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------


class AnthropicBackend:
    """Anthropic SDK. Structured JSON via output_config.format; adaptive
    thinking (always on for this model); server-side refusal fallback."""

    def __init__(self, model=MODEL, effort="high"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort

    def ask(self, role, prompt, schema):
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": f"[role: {role}]\n\n{prompt}"}],
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"{role}: model declined ({msg.stop_details})")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError(f"{role}: output truncated at max_tokens")
        text = next(b.text for b in msg.content if b.type == "text")
        return json.loads(text)


class ClaudeCLIBackend:
    """Headless Claude Code (`claude -p`), no tools, JSON-schema output."""

    def __init__(self, model=MODEL, exe=None):
        import tempfile
        self.exe = exe or shutil.which("claude") or "/opt/node22/bin/claude"
        self.model = model
        # empty working directory: no project CLAUDE.md or files leak into the role
        self.cwd = tempfile.mkdtemp(prefix="va_llm_")
        self.cost_usd, self.calls = 0.0, []

    def ask(self, role, prompt, schema):
        cmd = [self.exe, "-p", "--model", self.model, "--output-format", "json",
               "--json-schema", json.dumps(schema), "--tools", "",
               "--append-system-prompt", SYSTEM]
        out = subprocess.run(cmd, input=f"[role: {role}]\n\n{prompt}", text=True,
                             capture_output=True, timeout=1800, cwd=self.cwd)
        if out.returncode != 0:
            raise RuntimeError(f"{role}: claude CLI failed: {out.stderr[-2000:]}")
        envelope = json.loads(out.stdout)
        self.cost_usd += envelope.get("total_cost_usd") or 0.0
        self.calls.append({"role": role, "cost_usd": envelope.get("total_cost_usd"),
                           "model": self.model})
        if envelope.get("is_error"):
            raise RuntimeError(f"{role}: {envelope.get('result')}")
        data = envelope.get("structured_output")
        if data is None:  # fall back to the text result
            m = re.search(r"\{.*\}", envelope.get("result", ""), re.S)
            data = json.loads(m.group(0))
        return data


class ReplayBackend:
    """Offline backend: role -> canned JSON answer, for tests and audits."""

    def __init__(self, path):
        self.answers = json.load(open(path))

    def ask(self, role, prompt, schema):
        return self.answers[role]


def make_backend(name, replay_path=None):
    if name == "anthropic":
        return AnthropicBackend()
    if name == "claude-cli":
        return ClaudeCLIBackend()
    if name == "replay":
        return ReplayBackend(replay_path)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Deterministic guards around every LLM output
# ----------------------------------------------------------------------------


def _check_schema_keys(obj, schema, where):
    missing = [k for k in schema["required"] if k not in obj]
    extra = [k for k in obj if k not in schema["properties"]]
    if missing or extra:
        raise ValueError(f"{where}: missing {missing}, unexpected {extra}")


def validate_spec(d):
    _check_schema_keys(d, SPEC_SCHEMA, "formalizer")
    errs = []
    if d["element"] not in ELEMENTS:
        errs.append(f"unknown element {d['element']}")
    if d["regularity"] not in REGULARITY:
        errs.append(f"unknown regularity {d['regularity']}")
    if not d["qois"]:
        errs.append("no QoIs named")
    if d["reference"] == "exact_manufactured" and d["regularity"] != "smooth":
        errs.append("manufactured reference requires a smooth solution")
    return errs


def validate_plan(plan, spec_qois):
    """Hard guard: a plan may only contain catalogued checks, with parameters
    inside the catalogue bounds, on QoIs the spec actually declared."""
    _check_schema_keys(plan, PLAN_SCHEMA, "proposer")
    errs = []
    for i, c in enumerate(plan["checks"]):
        cat = CHECK_CATALOGUE.get(c["check"])
        if cat is None:
            errs.append(f"check {i}: '{c['check']}' is not in the catalogue")
            continue
        for k, v in c["params"].items():
            if k not in cat["params"]:
                errs.append(f"check {i} ({c['check']}): parameter '{k}' not allowed")
                continue
            lo, hi = cat["params"][k]
            if not lo <= v <= hi:
                errs.append(f"check {i} ({c['check']}): {k}={v} outside [{lo}, {hi}]")
        for q in c["qois"]:
            if q not in spec_qois:
                errs.append(f"check {i} ({c['check']}): QoI '{q}' not declared in the spec")
    covered = {q for c in plan["checks"] for q in c["qois"]}
    for q in spec_qois:
        if q not in covered:
            errs.append(f"QoI '{q}' has no check")
    return errs


def audit_advice(advice, gate_verdicts):
    """Flag every place where the advisor's stated verdict disagrees with the
    deterministic gate. gate_verdicts: {qoi: verdict}."""
    _check_schema_keys(advice, ADVICE_SCHEMA, "advisor")
    flags = []
    stated = {p["qoi"]: p["stated_verdict"] for p in advice["per_qoi"]}
    for q, v in gate_verdicts.items():
        if q not in stated:
            flags.append(f"advisor omitted QoI '{q}' (gate: {v})")
        elif stated[q] != v:
            flags.append(f"advisor says {stated[q]} for '{q}' but the gate says {v}")
    for q in stated:
        if q not in gate_verdicts:
            flags.append(f"advisor discusses '{q}', which no gate judged")
    return flags


# ----------------------------------------------------------------------------
# Role prompts
# ----------------------------------------------------------------------------


def formalize(backend, problem_text):
    prompt = ("Formalize this verification problem into the schema. Use only the listed "
              "enum values; put anything you cannot determine in open_questions. In "
              "well_posedness list the conditions (boundary conditions, material stability, "
              "load level vs. limit points, singular features) that decide which QoIs can "
              "converge.\n\nPROBLEM:\n" + problem_text)
    d = backend.ask("formalizer", prompt, SPEC_SCHEMA)
    return d, validate_spec(d)


def propose(backend, spec_dict, neighbours):
    prompt = ("Choose verification checks for this problem, ONLY from the catalogue. Cover "
              "every declared QoI. Predict which QoIs you expect to fail and why.\n\n"
              f"CATALOGUE:\n{json.dumps(CHECK_CATALOGUE, indent=1)}\n\n"
              f"PROBLEM SPEC:\n{json.dumps(spec_dict, indent=1)}\n\n"
              f"NEAREST SOLVED CASES (Jaccard similarity, verdicts):\n"
              f"{json.dumps(neighbours, indent=1)}")
    plan = backend.ask("proposer", prompt, PLAN_SCHEMA)
    return plan, validate_plan(plan, spec_dict["qois"])


def critique(backend, spec_dict, plan):
    prompt = ("Review this verification plan. Object to any check that cannot establish "
              "what it claims, to tolerances that are too loose for a validation target, "
              "and to QoIs that are physically ill-posed for pointwise convergence. Approve "
              "only if you would accept the plan in a journal review.\n\n"
              f"SPEC:\n{json.dumps(spec_dict, indent=1)}\n\nPLAN:\n{json.dumps(plan, indent=1)}")
    return backend.ask("critic", prompt, CRITIQUE_SCHEMA)


def advise(backend, spec_dict, gate_results):
    prompt = ("Explain these deterministic gate results for a manuscript. For each QoI, "
              "restate the gate verdict exactly (do not change it) and interpret it "
              "physically. Then propose one paragraph of manuscript wording that claims no "
              "more than the gates support.\n\n"
              f"SPEC:\n{json.dumps(spec_dict, indent=1)}\n\n"
              f"GATE RESULTS:\n{json.dumps(gate_results, indent=1)}")
    advice = backend.ask("advisor", prompt, ADVICE_SCHEMA)
    verdicts = {q: g["verdict"] for q, g in gate_results.items()}
    return advice, audit_advice(advice, verdicts)
