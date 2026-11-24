"""Runs the pre-registered autonomy test (autonomy_test/preregistration.json) and records the
LLM cost of every case (the backend instance is captured; autonomous.py is not changed after its
hash was frozen).

    python -m omar_pfem.verification_agent.autonomy_test_run [case ...]
    python -m omar_pfem.verification_agent.autonomy_test_run --score
"""
import hashlib
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(HERE, "autonomy_test")
PRE = os.path.join(TDIR, "preregistration.json")
OUT = os.path.join(TDIR, "results.json")


def check_frozen():
    pre = json.load(open(PRE))
    bad = []
    for f, h in pre["frozen_sha256"].items():
        p = os.path.normpath(os.path.join(TDIR, f))
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            bad.append(f)
    if bad:
        raise SystemExit(f"frozen files changed since the pre-registration: {bad}")
    return pre


def run_case(name, case):
    from omar_pfem.verification_agent import autonomous, llm_roles
    made = []
    orig = llm_roles.make_backend

    def capture(*a, **k):
        b = orig(*a, **k)
        made.append(b)
        return b
    llm_roles.make_backend = capture
    src = os.path.join(TDIR, case["source"])
    try:
        if src.endswith(".pdf"):
            res = autonomous.run(pdf=src, hint=case.get("hint"), backend_name="claude-cli",
                                 budget_minutes=15, max_experiments=4)
        else:
            res = autonomous.run(text=src, backend_name="claude-cli", budget_minutes=15,
                                 max_experiments=4)
        err = None
    except Exception as e:
        res, err = {}, traceback.format_exc()[-3000:]
    finally:
        llm_roles.make_backend = orig
    cost = sum(getattr(b, "cost_usd", 0.0) for b in made)
    run_name = os.path.splitext(os.path.basename(src))[0]
    return {"run_dir": os.path.join("autonomous_runs", run_name), "llm_cost_usd": cost,
            "error": err, "final": res.get("final"),
            "stopped_at_definition": "final" not in res and err is None}


def main(names):
    import warnings
    warnings.filterwarnings("ignore")
    pre = check_frozen()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in names or list(pre["cases"]):
        if name in out and out[name].get("error") is None:
            continue
        print(f"===== {name}", flush=True)
        out[name] = run_case(name, pre["cases"][name])
        json.dump(out, open(OUT, "w"), indent=1, default=float)
        print(f"===== {name} done: cost {out[name]['llm_cost_usd']:.2f} error "
              f"{bool(out[name]['error'])}", flush=True)




def score():
    """Checks every pre-registered expectation by code (results.json + the run logs)."""
    pre = json.load(open(PRE))
    res = json.load(open(OUT))
    run = lambda c: json.load(open(os.path.join(HERE, res[c]["run_dir"], "run.json")))
    steps = lambda c: run(c)["steps"]
    section = lambda c: next((s["detail"] for s in steps(c) if s["stage"] == "L"
                              and s["actor"] == "LLM"), "")
    stopped = lambda c: run(c).get("final") is None and any(
        s["status"] in ("unsupported", "failed") for s in steps(c))
    persons = lambda c: len(run(c)["human_points"])
    mms_ok = lambda c: all(r["verdict"] == "PASS" for r in run(c).get("mms", {}).values())
    cert = lambda c: (run(c).get("final") or {}).get("certified", [])
    corner_lams = lambda c: [s["detail"] for s in steps(c) if s["stage"] == "M"]
    out = {"cases": {}}
    for c in ("T1_deep_ritz_lshape", "T2_qtt_lshaped_panel", "T3_polivem_hole"):
        out["cases"][c] = {"stopped": stopped(c), "section_used": section(c),
                           "quote_rate_100": "quotes found 100%" in section(c)}
    out["cases"]["T1_deep_ritz_lshape"]["right_section"] = "5.2" in section("T1_deep_ritz_lshape")
    out["cases"]["T2_qtt_lshaped_panel"]["right_section"] = "section 5" in section("T2_qtt_lshaped_panel")
    out["cases"]["T3_polivem_hole"]["right_section"] = "hole" in section("T3_polivem_hole").lower()
    out["cases"]["T6_cantilever_parabolic"] = {"stopped": stopped("T6_cantilever_parabolic")}
    for c in ("T4_step_plate", "T5_quarter_cutout"):
        d = run(c)["description"]
        out["cases"][c] = {"mms_pass": mms_ok(c), "energy_certified": "strain_energy" in cert(c),
                           "certified": cert(c), "persons": persons(c),
                           "cells": len(d["cells"]), "cell_size": d["cell_size"],
                           "plane": d["plane"], "corners": corner_lams(c)}
    out["cases"]["T5_quarter_cutout"]["symmetry_used"] = sum(
        b["type"] == "symmetry" for b in run("T5_quarter_cutout")["description"]["bcs"])
    out["cases"]["T7_missing_modulus"] = {"persons": persons("T7_missing_modulus"),
                                          "assumed": [s["detail"] for s in steps("T7_missing_modulus")
                                                      if s["status"] == "needs_person"]}
    cost = sum(r["llm_cost_usd"] for r in res.values())
    C = out["cases"]
    out["predictions"] = {
        "S1 stops on T1, T2, T3, T6": all(C[c]["stopped"] for c in (
            "T1_deep_ritz_lshape", "T2_qtt_lshaped_panel", "T3_polivem_hole", "T6_cantilever_parabolic")),
        "S2 T4, T5 end to end, no person beyond the text, MMS PASS, energy certified": all(
            C[c]["mms_pass"] and C[c]["energy_certified"] and C[c]["persons"] == 1
            for c in ("T4_step_plate", "T5_quarter_cutout")),
        "S3 T7 flagged": C["T7_missing_modulus"]["persons"] >= 2,
        "S4 cost below USD 10": cost < 10}
    out["details_against_expectations"] = {
        "T4 6 cells of 10 mm, plane stress": C["T4_step_plate"]["cells"] == 6 and
        C["T4_step_plate"]["cell_size"] == 10 and C["T4_step_plate"]["plane"] == "stress",
        "T5 3 cells of 20 mm": C["T5_quarter_cutout"]["cells"] == 3,
        "T5 symmetry on two lines": C["T5_quarter_cutout"]["symmetry_used"] == 2,
        "literature sections right": {c: C[c]["right_section"] for c in (
            "T1_deep_ritz_lshape", "T2_qtt_lshaped_panel", "T3_polivem_hole")}}
    out["llm_cost_usd"] = cost
    json.dump(out, open(os.path.join(TDIR, "score.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("predictions", "details_against_expectations",
                                          "llm_cost_usd")}, indent=1))
    return out


if __name__ == "__main__":
    score() if "--score" in sys.argv else main(sys.argv[1:])
