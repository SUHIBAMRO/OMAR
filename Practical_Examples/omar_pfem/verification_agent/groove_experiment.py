"""Decision 3 of 24 Nov 2026: one more targeted spline refinement of the B3 groove peak, only
if the agent justifies it, predicts the outcome and evaluates the hypothesis (experiment.py).

    python -m omar_pfem.verification_agent.groove_experiment propose   (justify, freeze, job)
    python -m omar_pfem.verification_agent.groove_experiment run-l6    (inside the cluster job)
    python -m omar_pfem.verification_agent.groove_experiment score     (after the import)

Everything the proposal says is computed here from the stored spline levels; nothing is typed
in by hand except the level's span counts, which follow the existing pattern (12 per level
around, 6 through, 20 along).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "iga_b3_result.json")
OUT6 = os.path.join(HERE, "iga_b3_l6_result.json")
L6 = (72, 36, 120)
PROBLEM, ACTION = "bushing_groove", "spline_level_6"


def _peaks(d, form):
    suf = "" if form == "disp" else ":fbar_p1"
    return {lv: d[f"L{lv}:p2{suf}"]["max_sigma_xx_R0"] for lv in (2, 3, 4, 5)
            if f"L{lv}:p2{suf}" in d}


def analyse():
    d = json.load(open(DATA))
    P, U = _peaks(d, "fbar_p1"), _peaks(d, "disp")
    rel = lambda s: {f"{a}->{b}": s[b] / s[a] - 1 for a, b in zip(list(s)[:-1], list(s)[1:])}
    gap = {lv: U[lv] - P[lv] for lv in P if lv in U}
    ch = list(rel(P).values())
    contracting = abs(ch[-1]) < abs(ch[-2])
    dof5 = d["L5:p2:fbar_p1"]["n_dof"]
    dof6 = 3 * (L6[0] + 2) * (L6[1] + 2) * (L6[2] + 2)
    t5 = max(d["L5:p2:fbar_p1"]["seconds"], d["L5:p2"]["seconds"])
    alpha = 1.3                     # AMG-CG solves scaled with dof^1.0-1.3 on levels 3-5
    t6 = t5 * (dof6 / dof5) ** alpha
    return {"P1_peaks": P, "disp_peaks": U, "P1_changes": rel(P), "disp_changes": rel(U),
            "gap_disp_minus_P1": gap, "P1_contracting": contracting,
            "dof5": dof5, "dof6": dof6, "hours_per_formulation": t6 / 3600}


def proposal(a):
    from omar_pfem.verification_agent import experiment as X
    ch = list(a["P1_changes"].items())
    gaps = list(a["gap_disp_minus_P1"].items())
    p, n = X.prior(ACTION, "pointwise_peak:finite_radius_not_contracting")
    return {
        "action": ACTION, "situation": "pointwise_peak:finite_radius_not_contracting",
        "quantities": ["max_sigma_xx_R0"],
        "justification": (
            "The groove root has a finite radius (a problem fact), so the continuum stress is "
            "bounded and the peak has a finite limit; no singular exponent applies. The "
            "projected-dilatation "
            f"(P1) peak changed {', '.join(f'{k}: {v:+.1%}' for k, v in ch)}: the last change "
            f"is {'smaller' if a['P1_contracting'] else 'larger'} than the one before, so the "
            "sequence does not yet contract. The displacement-only peak approaches it: the gap "
            f"{', '.join(f'L{k}: {v:.3f}' for k, v in gaps)} shrinks every level. Without a "
            "contracting sequence no extrapolation or rate is possible; one more level is the "
            "cheapest test that can decide between a converging and a non-converging peak."),
        "hypothesis": ("H1: the P1 peak is converging and the L4->L5 drop is pre-asymptotic: at "
                       "level 6 the change contracts and the formulations keep approaching."),
        "prediction": {
            "text": (f"|P1(L6)/P1(L5) - 1| < {abs(ch[-1][1]):.3f} (the L4->L5 change) AND "
                     f"|disp(L6) - P1(L6)| < {gaps[-1][1]:.3f} (the L5 gap)"),
            "criterion": "groove_contracts",
            "last_change": abs(ch[-1][1]), "last_gap": gaps[-1][1]},
        "if_confirmed": "the peak may become certifiable: at least one more level is needed for "
                        "three contracting changes; still reported as not verified until then",
        "if_refuted": "report the groove peak and the stress-concentration factor as not "
                      "verifiable with the present discretizations; stop refining",
        "confidence": "low-medium: theory says bounded, but the last change grew",
        "prior": p, "prior_n": n,
        "run": {"kind": "cluster_job", "levels": {"6": list(L6)},
                "formulations": ["fbar_p1", "disp"]},
        "cost": {"dof": a["dof6"], "minutes": 2 * a["hours_per_formulation"] * 60,
                 "where": "cluster (above the local budget)"},
    }


def propose():
    from omar_pfem.verification_agent import experiment as X
    from omar_pfem.verification_agent.job_factory import make_job
    a = analyse()
    c = proposal(a)
    path, rec = X.freeze(PROBLEM, 1, c)
    hours = math.ceil(2 * a["hours_per_formulation"] * 1.5)
    job = make_job("groove_l6", "-m omar_pfem.verification_agent.groove_experiment run-l6",
                   mem_mb=64000, hours=min(max(hours, 12), 48),
                   outputs=["iga_b3_l6_result.json"])
    print(json.dumps({"analysis": a, "frozen": os.path.relpath(path, HERE),
                      "sha256": rec["sha256"], "job": job}, indent=1, default=float))


def run_l6():
    from omar_pfem.verification_agent import iga_b3 as I
    os.environ.setdefault("VERIF_LINSOLVE", "amg")
    I.LEVELS[6] = L6
    I.OUT = OUT6
    cache = json.load(open(OUT6)) if os.path.exists(OUT6) else {}
    I.run(6, 2, cache, "fbar_p1")
    I.run(6, 2, cache)


def score():
    from omar_pfem.verification_agent import experiment as X
    path = os.path.join(X.EXP_DIR, PROBLEM, f"01_{ACTION}.json")
    rec = X._verify_frozen(path)
    d, r6 = json.load(open(DATA)), json.load(open(OUT6))
    p5, u5 = d["L5:p2:fbar_p1"]["max_sigma_xx_R0"], d["L5:p2"]["max_sigma_xx_R0"]
    p6, u6 = r6["L6:p2:fbar_p1"]["max_sigma_xx_R0"], r6["L6:p2"]["max_sigma_xx_R0"]
    pr = rec["prediction"]
    change, gap = abs(p6 / p5 - 1), abs(u6 - p6)
    ok = change < pr["last_change"] and gap < pr["last_gap"]
    X.complete(path, {"confirmed": ok, "P1_L6": p6, "disp_L6": u6, "change_L5_L6": change,
                      "gap_L6": gap, "observed_minutes": (r6["L6:p2:fbar_p1"]["seconds"]
                                                          + r6["L6:p2"]["seconds"]) / 60,
                      "conclusion": rec["if_confirmed"] if ok else rec["if_refuted"]})
    print("confirmed" if ok else "refuted", change, gap)



def value_of_information(level=7, grid=126):
    """(2026-11-25) Before proposing another level: can ANY outcome of it change the verdict
    under the frozen rules? Every candidate value of the next peak is run through the same two
    convergence checks (three- and five-level, peak class tolerances); the verdict combination
    fails if either check fails. Writes the decision to experiments/bushing_groove/."""
    import numpy as np
    from omar_pfem.verification_agent import experiment as X
    from omar_pfem.verification_agent import gates, trees
    d, n = json.load(open(DATA)), json.load(open(OUT6))
    P = [d[f"L{l}:p2:fbar_p1"]["max_sigma_xx_R0"] for l in (2, 3, 4, 5)] + \
        [n["L6:p2:fbar_p1"]["max_sigma_xx_R0"]]
    spans = {l: 12 * l for l in range(2, 12)}
    cp = trees.class_params("qoi_convergence_multilevel", "pointwise_peak")
    hs = [1 / spans[l] for l in range(level - 4, level + 1)]
    certifying = []
    for x in np.linspace(0.8 * P[-1], 1.2 * P[-1], grid):
        seq = (P + [x])[-5:]
        g5 = gates.multilevel_convergence_gate(hs, seq, order_window=trees.order_window("Q20"), **cp)
        g3 = gates.qoi_convergence_gate(hs[-3:], seq[-3:], order_window=trees.order_window("Q20"), **cp)
        if g5["verdict"] == "PASS" and g3["verdict"] == "PASS":
            certifying.append(float(x))
    dof = lambda l: 3 * (12 * l + 2) * (6 * l + 2) * (20 * l + 2)
    t6 = max(n["L6:p2:fbar_p1"]["seconds"], n["L6:p2"]["seconds"]) / 3600
    hours = {l: t6 * (dof(l) / dof(6)) ** 1.3 for l in (7, 8, 9)}
    rec = {"action": "spline_level_7_declined", "situation": "pointwise_peak:non_monotone",
           "quantities": ["max_sigma_xx_R0"],
           "decision": "not proposed" if not certifying else "proposed",
           "justification": (f"P1 peak sequence {[round(p, 3) for p in P]}: the five-level window "
                             f"L{level - 4}-L{level} contains the reversal; of {grid} candidate values "
                             f"of L{level} ({0.8 * P[-1]:.3f}-{1.2 * P[-1]:.3f}), "
                             f"{len(certifying)} pass both convergence checks. A monotone five-level "
                             "window needs levels up to 9."),
           "hours_per_formulation_estimate": hours,
           "prediction": {"text": "none (no experiment)", "criterion": "none"},
           "cost": {"minutes": 0.0}}
    path, frozen = X.freeze(PROBLEM, 2, rec)
    print(json.dumps({k: rec[k] for k in ("decision", "justification", "hours_per_formulation_estimate")},
                     indent=1, default=float))
    return rec


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    {"propose": propose, "run-l6": run_l6, "score": score,
     "voi": value_of_information}[sys.argv[1]]()
