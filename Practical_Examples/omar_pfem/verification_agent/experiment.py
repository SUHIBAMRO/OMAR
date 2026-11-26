"""Experiment protocol and experience memory (round O, 2026-11-24).

Professor Rabczuk (24 Nov): "the agent should decide what to investigate next and learn from
the results"; a further refinement is acceptable "provided the agent independently justifies
the experiment, predicts the expected outcome and evaluates whether the hypothesis is
confirmed". This module is that loop, for every next action of the agent:

  1. CANDIDATES  each action states when it applies, from the verdicts and check results
                 (not from a person): finer uniform level, intensity-graded meshes, a finer
                 independent level, or a conclusion without an experiment (a peak at a corner
                 whose exponent is below 1 is unbounded: not verifiable).
  2. CHOICE      score = prior x quantities it can decide / (1 + estimated minutes); the prior
                 is the confirmed fraction of earlier experiments of the same action in the
                 same situation (Laplace: (confirmed + 1) / (n + 2)), read from the memory.
  3. FREEZE      before running: justification, hypothesis, a machine-checkable prediction,
                 the cost estimate, the git HEAD and a SHA-256 of the record are written to
                 experiments/<problem>/<k>_<action>.json (and committed locally if
                 AGENT_GIT_COMMIT=1).
  4. RUN         within the budget; an experiment above it becomes a cluster job (job_factory)
                 and stays pending until its result is imported.
  5. SCORE       the prediction is checked by code; confirmed / refuted, observed cost.
  6. LEARN       the outcome is appended to experience.json; it changes the priors of step 2.

Nothing here changes a tolerance, a criterion or a verdict rule.
"""
import datetime
import hashlib
import json
import math
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "experiments")
MEMORY = os.path.join(HERE, "experience.json")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


# ------------------------------------------------------------------ memory ----------------
def memory():
    return json.load(open(MEMORY)) if os.path.exists(MEMORY) else []


def prior(action, situation):
    rec = [m for m in memory() if m["action"] == action and m["situation"] == situation
           and m.get("confirmed") is not None]
    k = sum(m["confirmed"] for m in rec)
    return (k + 1) / (len(rec) + 2), len(rec)


def remember(entry):
    mem = memory()
    # BOOKKEEPING RULE (25 Nov): the same experiment on the same problem with the same outcome
    # (e.g. a repeat run from cached solutions) is not new evidence and is counted once
    same = ("problem", "action", "situation", "quantities", "confirmed")
    if any(all(m.get(k) == entry.get(k) for k in same) for m in mem):
        return
    mem.append(entry)
    json.dump(mem, open(MEMORY, "w"), indent=1)


# ------------------------------------------------------------------ freezing --------------
def _head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def freeze(problem, k, record):
    d = os.path.join(EXP_DIR, problem)
    os.makedirs(d, exist_ok=True)
    rec = dict(record, problem=problem, git_head=_head(),
               frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    body = json.dumps(rec, sort_keys=True, default=float)
    rec["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    path = os.path.join(d, f"{k:02d}_{record['action']}.json")
    run = 2
    while os.path.exists(path):        # FIX (25 Nov): a repeat run never overwrites a record
        path = os.path.join(d, f"{k:02d}_{record['action']}_run{run}.json")
        run += 1
    json.dump({"preregistration": rec}, open(path, "w"), indent=1, default=float)
    if os.environ.get("AGENT_GIT_COMMIT") == "1":
        subprocess.run(["git", "add", path], cwd=ROOT, check=False)
        subprocess.run(["git", "commit", "-q", "-m", f"agent pre-registration: {problem} "
                        f"{record['action']} ({rec['sha256'][:12]})"], cwd=ROOT, check=False)
    return path, rec


def _verify_frozen(path):
    rec = dict(json.load(open(path))["preregistration"])
    sha = rec.pop("sha256")
    body = json.dumps(rec, sort_keys=True, default=float)
    if hashlib.sha256(body.encode()).hexdigest() != sha:
        raise RuntimeError(f"pre-registration {path} was changed after freezing")
    return rec


def complete(path, outcome):
    rec = _verify_frozen(path)
    data = json.load(open(path))
    data["outcome"] = outcome
    json.dump(data, open(path, "w"), indent=1, default=float)
    # FIX (26 Nov, bookkeeping): an experiment the local memory guard refused never ran, so it
    # is no evidence about its prediction; it is remembered with confirmed = None (the prior
    # counts only confirmed True / False). It used to count as refuted.
    machine = "too large for this machine" in str(outcome.get("failed", ""))
    remember({"problem": rec["problem"], "action": rec["action"], "situation": rec["situation"],
              "quantities": rec["quantities"],
              "confirmed": None if machine else outcome["confirmed"],
              **({"not_run": "refused by the local memory guard (cluster job)"} if machine else {}),
              "predicted_minutes": rec["cost"]["minutes"],
              "observed_minutes": outcome.get("observed_minutes"),
              "file": os.path.relpath(path, HERE)})


# ------------------------------------------------------------------ situations ------------
def situation(r):
    """A short description of a quantity's state, the key of the memory."""
    labels = [g.get("label") or "" for g in r.get("checks", {}).values()]
    if any("from below" in l for l in labels):
        tag = "preasymptotic_below"
    elif any("pre-asymptotic" in l for l in labels):
        tag = "preasymptotic_above"
    elif r.get("crosscheck") == "independent_unconverged":
        tag = "independent_unconverged"
    elif r["verdict"] == "PASS" and r["evidence"] == "rate":
        tag = "certified"
    elif r["verdict"] == "PASS":
        tag = "band_only"
    elif r["verdict"] == "FAIL":
        tag = "fail"
    else:
        tag = "inconclusive"
    return f"{r['class']}:{tag}"


def _cost_model(rows):
    """minutes(dof) ~ c dof^alpha from the stored solves (finest three), and the next level's
    dof (x4 in 2D)."""
    rows = sorted(rows, key=lambda r: r["n_dof"])[-3:]
    d = np.array([r["n_dof"] for r in rows], float)
    t = np.array([max(r.get("wall_clock_s", 0.01), 1e-3) for r in rows], float)
    if len(rows) >= 2 and np.ptp(np.log(d)) > 0:
        alpha, logc = np.polyfit(np.log(d), np.log(t), 1)
        alpha = float(np.clip(alpha, 1.0, 2.0))
    else:
        alpha, logc = 1.2, math.log(t[-1]) - 1.2 * math.log(d[-1])
    return lambda dof: math.exp(logc) * dof ** alpha / 60.0


def _level_cost(rows, n, grow, cm):
    """(dof, minutes) of level n. FIX (26 Nov, after the plate's P2 n = 32 row came back from
    the cluster): a level already in the cache costs nothing, and the next level's size is
    grown from the finest level BELOW n (it was grown from the finest stored level, so a
    cached n = 32 row made the n = 32 experiment look like an n = 64 one and exceed the
    budget)."""
    hit = [r for r in rows if r["n"] == n]
    if hit:
        return hit[0]["n_dof"], 0.0
    below = [r for r in rows if r["n"] < n] or rows
    base = max(below, key=lambda r: r["n"])
    dof = base["n_dof"] * grow ** max(1, round(math.log2(n / base["n"])))
    return dof, cm(dof)


# ------------------------------------------------------------------ actions (2D) ----------
def candidates(state):
    """state: {"prob", "verdicts" (uniform), "graded" (or None), "levels", "ladders"}.
    Returns candidate experiments (dicts) and conclusions (no experiment)."""
    from omar_pfem.verification_agent import trees
    prob, V = state["prob"], state["verdicts"]
    tol = lambda q: trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))["gci_tol"]
    uni = state["ladders"].cache.get("Q2 uniform", [])
    p2 = state["ladders"].cache.get("P2 uniform", [])
    cm_q2, cm_p2 = _cost_model(uni), _cost_model(p2)
    n_next = 2 * max(state["levels"])
    grow = 2 ** getattr(state["prob"], "dim", 2)       # DOF per level: 4 in 2D, 8 in 3D
    dof_next, min_q2 = _level_cost(uni, n_next, grow, cm_q2)
    dof_next_p2, min_p2 = _level_cost(p2, n_next, grow, cm_p2)
    out, conclusions = [], []
    done = set(state.get("certified_any", ()))
    diag = state.get("diagnosis", {})
    open_q = {q: r for q, r in V.items() if not (r.get("final") == "CERTIFIED") and q not in done}
    # conclusions: unbounded peaks
    for q, r in list(open_q.items()):
        if r["class"] == "pointwise_peak" and prob.lam_min is not None and prob.lam_min < 1:
            conclusions.append({"quantity": q, "conclusion": "not verifiable",
                                "justification": f"a corner exponent {prob.lam_min:.4f} < 1: the "
                                                 "stress is unbounded (r^(lam-1)), the peak grows "
                                                 "with refinement; no experiment can certify it"})
            open_q.pop(q)
    # A1 finer uniform level
    pre = [q for q, r in open_q.items() if situation(r).split(":")[1] in
           ("preasymptotic_below", "preasymptotic_above", "band_only", "inconclusive")
           or "D4" in diag.get(q, {}).get("supported", [])]
    graded = state.get("graded") is not None
    if graded:
        gq = state["ladders"].cache.get("Q2 graded", [])
        gp = state["ladders"].cache.get("P2 graded", gq)
        dof_next, min_q2 = _level_cost(gq, n_next, grow, _cost_model(gq))
        dof_next_p2, min_p2 = _level_cost(gp, n_next, grow, _cost_model(gp))
    if pre and n_next <= state.get("max_n", 128):
        minutes = min_q2 + min_p2
        fam = "graded" if graded else "uniform"
        why = lambda q: (situation(open_q[q]) + (" + diagnosis D4 (pre-asymptotic)" if "D4" in
                         diag.get(q, {}).get("supported", []) else ""))
        out.append({"action": f"finer_{fam}_level", "quantities": pre,
                    "situation": situation(open_q[pre[0]]),
                    "justification": "; ".join(f"{q}: {why(q)}" for q in pre),
                    "hypothesis": f"the sequences are pre-asymptotic; one finer {fam} level "
                                  f"(n = {n_next}) lets the checks decide",
                    "prediction": {"text": "at least one listed quantity becomes certified with "
                                           "a rate and cross-checked",
                                   "criterion": "any_certified"},
                    "run": {"kind": f"{fam}_level", "n": n_next},
                    "cost": {"dof": dof_next + dof_next_p2, "minutes": minutes}})
    # A2 intensity-graded meshes
    need = [q for q, r in open_q.items() if r["class"] != "pointwise_peak"]
    need += [q for q, r in V.items() if r.get("final") == "CERTIFIED" and r["class"] !=
             "pointwise_peak" and r.get("U", 0) > 0.1 * tol(q)]
    if need and prob.corners and state.get("graded") is None:
        dof = 2 * sum(r["n_dof"] for r in uni)
        out.append({"action": "graded_meshes", "quantities": sorted(set(need)),
                    "situation": "corner_limited",
                    "justification": f"rate-limiting corners {[(c['x'], c['y'], round(min(c['rate_limiting']), 4)) for c in prob.corners]}; "
                                     f"{len(need)} quantities uncertified or with U above a tenth "
                                     "of the class tolerance",
                    "hypothesis": "the corner singularities limit the rate; grading toward the "
                                  "corners with the largest local-energy change restores it",
                    "prediction": {"text": "more quantities are certified (and cross-checked) on "
                                           "the graded ladders than on the uniform ones, or the "
                                           "strain-energy U falls at least 3 times",
                                   "criterion": "graded_better"},
                    "run": {"kind": "graded"},
                    "cost": {"dof": dof, "minutes": cm_q2(dof / 2) * 2}})
    # A3 finer independent level
    ind = [q for q, r in V.items() if r.get("crosscheck") == "independent_unconverged"]
    n_next_p2 = 2 * max(state.get("p2_levels") or state["levels"])
    if ind and n_next_p2 <= state.get("max_n", 128):
        dof_ind, min_ind = _level_cost(p2, n_next_p2, grow, cm_p2)
        out.append({"action": "finer_independent_level", "quantities": ind,
                    "situation": "any:independent_unconverged",
                    "justification": f"{ind} are rate-certified but their independent P2 "
                                     "sequence has no rate yet",
                    "hypothesis": "the independent sequence is pre-asymptotic; one more level "
                                  "gives it a rate that agrees",
                    "prediction": {"text": "at least one listed quantity becomes cross-checked",
                                   "criterion": "any_crosschecked"},
                    "run": {"kind": "p2_level", "n": n_next_p2},
                    "cost": {"dof": dof_ind, "minutes": min_ind}})
    # A4 (2026-11-25) finite-strain corner check: a neo-Hookean problem whose global quantities
    # converge more slowly than the linearized corner theory while min det F falls with refinement
    if getattr(prob, "material", "") == "neo_hookean" and prob.corners \
            and "small_load_check" not in state.get("banned", ()):
        js = [r.get("min_detF") for r in sorted(uni, key=lambda r: r["n"])]
        falling = len(js) >= 3 and all(js[i + 1] < js[i] for i in range(len(js) - 1))
        slow = [q for q, r in V.items() if r["class"] in ("global", "displacement")
                and (r.get("checks", {}).get("singular_order", {}).get("verdict") == "FAIL")]
        if falling and slow and not state.get("small_load_done"):
            dof = sum(r["n_dof"] for r in uni)
            out.append({"action": "small_load_check", "quantities": slow,
                        "situation": "neo_hookean:order_below_linearized_and_J_falling",
                        "justification": f"orders below the linearized corner theory for {slow}; "
                                         f"min det F falls with refinement {[round(j, 3) for j in js]}",
                        "hypothesis": "the loss of rate is a finite-strain effect at the corner: at "
                                      "one tenth of the load the strain energy converges with the "
                                      "linearized order 2 lam",
                        "prediction": {"text": "at 1/10 of the load the five-level energy order is "
                                               f"within 0.15 of 2 lam = {2 * prob.lam_min:.3f}",
                                       "criterion": "order_near_linear",
                                       "target": 2 * prob.lam_min, "tol": 0.15},
                        "run": {"kind": "small_load", "factor": 0.1},
                        "cost": {"dof": dof, "minutes": cm_q2(dof)}})
    out = [c for c in out if c["action"] not in state.get("banned", ())]
    for c in out:
        p, n = prior(c["action"], c["situation"])
        c["prior"], c["prior_n"] = p, n
        c["score"] = p * len(c["quantities"]) / (1.0 + c["cost"]["minutes"])
    return sorted(out, key=lambda c: -c["score"]), conclusions


def evaluate(c, old, new):
    """Code checks the frozen prediction against the new verdicts."""
    crit, qs = c["prediction"]["criterion"], c["quantities"]
    if crit == "any_certified":
        ok = any(new.get(q, {}).get("final") == "CERTIFIED" and old[q].get("final") != "CERTIFIED"
                 for q in qs)
    elif crit == "any_crosschecked":
        ok = any(new.get(q, {}).get("crosscheck") == "cross_checked" for q in qs)
    elif crit == "order_near_linear":
        p = new.get("_small_load_order")
        ok = p is not None and abs(p - c["prediction"]["target"]) <= c["prediction"]["tol"]
    elif crit == "graded_better":
        n_old = sum(r.get("final") == "CERTIFIED" for r in old.values())
        n_new = sum(r.get("final") == "CERTIFIED" for r in new.values())
        uo, un = old["strain_energy"].get("U"), new["strain_energy"].get("U")
        ok = n_new > n_old or (uo is not None and un is not None and un <= uo / 3)
    else:
        raise ValueError(crit)
    return bool(ok)
