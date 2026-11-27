"""A-priori convergence predictions compared with the observed rates (27 Nov, Professor
Rabczuk: "predict convergence rates for the selected elements, norms and QoIs, account for
singularities and compare these predictions with observed rates").

Rules, theory and expectations were frozen in apriori_preregistration.json before any
comparison was computed. This is a PREDEFINED PROCEDURE (a theory table in code, with its
citations and assumptions); finding such results in the literature by the LLM is a later step.

    python -m omar_pfem.verification_agent.apriori            (all stored runs)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "autonomous_runs")
LADDERS = os.path.join(HERE, "pipeline2d_runs")
OUT = os.path.join(HERE, "apriori_comparison.json")
TOL = 0.3
P_DEG = {"Q2": 2, "P2": 2}
CITE = {"mms": "Ciarlet 1978, Thm 3.2.2",
        "energy": "Strang & Fix 1973 Ch. 8; Babuska & Kellogg 1979",
        "functional": "duality argument (Aubin-Nitsche type), e.g. Brenner & Scott 2008 Ch. 0/5",
        "peak": "Williams 1952 corner fields; stress ~ r^(lam-1)",
        "graded": "Babuska, Kellogg & Pitkaranta 1979; Apel 1999"}


def predict(qclass, p, lam, graded=False):
    """{'order', 'mode', 'basis', 'citation'} for one quantity class."""
    smooth = graded or lam is None or lam >= p
    m = p if smooth else lam
    if qclass == "global":
        return {"order": 2 * m, "mode": "equal",
                "basis": f"strain-energy error = energy-norm error^2, order 2 min(p, lam) = {2 * m:.3f}"
                         + (" (graded: restored)" if graded and lam is not None and lam < p else ""),
                "citation": CITE["graded" if graded else "energy"]}
    if qclass in ("displacement", "region_average"):
        return {"order": 2 * m, "mode": "lower_bound",
                "basis": f"linear functional: order >= 2 min(p, lam) = {2 * m:.3f} (duality)",
                "citation": CITE["functional"]}
    if qclass == "pointwise_peak":
        # FIX (27 Nov, before scoring): grading does not bound a singular stress; the
        # pre-registered theory restores only energy and functionals on graded meshes
        if lam is not None and lam < 1:
            return {"order": lam - 1.0, "mode": "equal",
                    "basis": f"stress unbounded at the corner: peak ~ h^(lam - 1), slope {lam - 1:.3f}",
                    "citation": CITE["peak"]}
        return {"order": float(p), "mode": "lower_bound",
                "basis": f"pointwise stress, no singular corner (or graded): order >= p = {p}",
                "citation": CITE["energy"]}
    return None


def observed(rows, q, qclass, diverging=False):
    """Order over the finest three levels (h measured). FIX (27 Nov, before scoring): the
    log-log slope of the value itself is the measure only for a diverging peak; a bounded peak
    has an error order like any other quantity."""
    rows = sorted([r for r in rows if q in r and r.get(q) is not None], key=lambda r: r["n"])[-3:]
    if len(rows) < 3:
        return None, "fewer than 3 levels"
    h = [r["h"] for r in rows]
    v = [r[q] for r in rows]
    if qclass == "pointwise_peak" and diverging:
        return float(np.polyfit(np.log(h), np.log(np.abs(v)), 1)[0]), "log-log slope"
    d1, d2 = v[1] - v[0], v[2] - v[1]
    scale = max(abs(v[2]), 1e-30)
    if abs(d2) <= 1e-10 * scale or d1 * d2 <= 0:
        return None, "no measurable order (converged to round-off or not monotone)"
    # three-level estimate (levels halve h): p = ln|d1 / d2| / ln(h1 / h2)
    return math.log(abs(d1 / d2)) / math.log(h[1] / h[2]), "three-level"


def classify(pred, obs):
    if obs is None:
        return "no measurable order"
    d = obs - pred["order"]
    if pred["mode"] == "equal":
        if abs(d) <= TOL:
            return "agrees"
        return ("faster than predicted (pre-asymptotic or small singular coefficient)" if d > 0
                else "slower than predicted: CONTRADICTION")
    return "agrees" if d >= -TOL else "slower than predicted: CONTRADICTION"


def assumptions(prob, desc):
    a = ["element degree p = 2 (Q2 and P2)",
         "lam from the meshed boundary (singularity.py): smallest rate-limiting exponent",
         "h = measured element size; observed order from the finest three levels"]
    if desc.get("material") == "neo_hookean":
        a.append("neo-Hookean: linearized corner exponents (hypothesis)")
    if getattr(prob, "dim", 2) == 3:
        a.append("3D extruded: edge exponents only, vertex singularities not analysed (hypothesis)")
    return a


def analyse_run(name):
    run = json.load(open(os.path.join(RUNS, name, "run.json")))
    desc = run.get("description") or {}
    lf = os.path.join(LADDERS, desc.get("name", name), "ladders.json")
    if not desc or not os.path.exists(lf) or not run.get("verdicts"):
        return None
    from omar_pfem.verification_agent.geometry2d import make
    prob = make({k: v for k, v in desc.items() if k not in ("assumed", "geometry_source")})
    lam = getattr(prob, "lam_min", None)
    lad = json.load(open(lf))
    out = {"run": name, "lam": lam, "assumptions": assumptions(prob, desc), "rows": []}
    for q, v in run["verdicts"].items():
        if q.startswith("_") or "class" not in v:
            continue
        for key, graded in (("Q2 uniform", False), ("Q2 graded", True)):
            if key not in lad:
                continue
            pr = predict(v["class"], P_DEG["Q2"], lam, graded)
            ob, how = observed(lad[key], q, v["class"], diverging=pr["order"] < 0)
            out["rows"].append({"quantity": q, "class": v["class"], "ladder": key,
                                "predicted": pr["order"], "mode": pr["mode"], "basis": pr["basis"],
                                "citation": pr["citation"], "observed": ob, "how": how,
                                "outcome": classify(pr, ob)})
    return out


def mms_rows():
    rec = json.load(open(os.path.join(HERE, "code_verification_record.json")))
    rows = []
    for k, r in rec.items():
        if "rates" not in r:
            continue
        el = k.split("|")[1]
        p = P_DEG.get(el)
        for norm, pred in (("L2", p + 1), ("H1", p)):
            ob = r["rates"][norm]["observed"]
            rows.append({"problem": r.get("problem"), "element": el, "norm": norm,
                         "predicted": pred, "observed": ob, "verdict": r["verdict"],
                         "within_gate_tol_0.2": abs(ob - pred) <= 0.2})
    return rows


def main():
    import warnings
    warnings.filterwarnings("ignore")
    res = {"preregistration": "apriori_preregistration.json", "runs": [], "mms": mms_rows()}
    for name in sorted(os.listdir(RUNS)):
        if os.path.exists(os.path.join(RUNS, name, "run.json")) and not name.endswith("_posthoc"):
            try:
                a = analyse_run(name)
            except Exception as e:                      # one run must not stop the table
                a = {"run": name, "error": f"{e!s:.200}"}
            if a:
                res["runs"].append(a)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    from collections import Counter
    c = Counter((r["ladder"], r["class"], r["outcome"]) for a in res["runs"] for r in a.get("rows", []))
    for k, v in sorted(c.items()):
        print(f"{v:3d}  {k[0]:11s} {k[1]:15s} {k[2]}")
    bad = [(a["run"], r["quantity"], r["ladder"], r["predicted"], r["observed"])
           for a in res["runs"] for r in a.get("rows", []) if "CONTRADICTION" in r["outcome"]]
    print("contradictions:", bad)
    m = res["mms"]
    print(f"MMS: {sum(x['within_gate_tol_0.2'] for x in m if x['verdict'] == 'PASS')} of "
          f"{sum(x['verdict'] == 'PASS' for x in m)} PASS rows within 0.2 of p+1 / p")
    return res


if __name__ == "__main__":
    main()
