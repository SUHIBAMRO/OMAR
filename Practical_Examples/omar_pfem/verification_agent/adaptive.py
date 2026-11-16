"""Adaptive layer of the verification agent (prototype).

Two pieces, both deterministic:

1. outcome_prior: before computing anything, predict each QoI's verdict
   from the solved-case repository, weighting past records by fingerprint
   similarity (GRAFT's similarity-weighted prior, with the reward replaced
   by the gate verdict). This is what the repository "learns".

2. adaptive_refine: a refinement controller. Starting from the three
   coarsest levels, it re-evaluates the QoI gate after each level and
   requests the next (finer) level only for QoIs that are still undecided
   and only if the predicted cost fits the remaining budget. It stops a QoI
   on PASS, on a divergent or oscillatory sequence (more uniform refinement
   cannot fix those), or on budget.

What this is NOT: active learning for the neural-operator surrogate
(choosing which training samples or fidelities to generate next). That is
the third agent in the research plan and needs training runs; it is not
implemented here.
"""
import math

import numpy as np

from omar_pfem.verification_agent import gates, repository, trees
from omar_pfem.verification_agent.spec import jaccard


def outcome_prior(spec, path=repository.DEFAULT_PATH, min_weight=1e-9):
    """P(PASS) for each QoI of `spec`, from records of the same QoI class.
    weight = Jaccard(fingerprint of spec restricted to that QoI, record
    fingerprint); a record votes 1 if accepted, 0 otherwise. Returns
    {qoi: {"p_pass", "n_records", "weight"}}; p_pass is None with no data."""
    recs = repository.load(path)
    out = {}
    for q in spec.qois:
        cls = trees.classify_qoi(q)
        fp = spec.__class__(**{**spec.to_dict(), "qois": (q,)}).fingerprint()
        num = den = 0.0
        n = 0
        for r in recs:
            rq = r["problem"]["qois"]
            if len(rq) != 1 or trees.classify_qoi(rq[0]) != cls:
                continue
            if r["problem"]["name"].endswith(q) and r["problem"]["geometry"] == spec.geometry:
                continue  # leave-one-out: never predict a QoI from its own record
            w = jaccard(fp, r["fingerprint"])
            if w <= min_weight:
                continue
            num += w * (1.0 if r["accepted"] else 0.0)
            den += w
            n += 1
        out[q] = {"qoi_class": cls, "p_pass": (num / den) if den else None,
                  "n_records": n, "weight": den}
    return out


def _cost_model(rows, fallback_exponent=1.5):
    """Fit wall_clock = c * n_dof^alpha on rows that recorded timing."""
    t = [(r["n_dof"], r["wall_clock_s"]) for r in rows if r.get("wall_clock_s")]
    if len(t) >= 2:
        x, y = np.log([a for a, _ in t]), np.log([b for _, b in t])
        alpha, logc = np.polyfit(x, y, 1)
        return lambda n: float(math.exp(logc) * n ** alpha), f"fitted alpha={alpha:.2f}"
    if len(t) == 1:
        n0, t0 = t[0]
        return (lambda n: t0 * (n / n0) ** fallback_exponent,
                f"one timing point, assumed alpha={fallback_exponent}")
    return None, "no timing data"


def adaptive_refine(solve, qois, N0=(11, 21, 41), budget_s=1800.0, max_levels=6,
                    band_tol=0.01, gci_tol=0.05, timing_rows=(), log=print):
    """solve(N) -> row dict with 'h', 'n_dof', optional 'wall_clock_s', and
    the QoIs. Returns per-QoI decisions and the levels used."""
    rows = [solve(N) for N in N0]
    spent = sum(r.get("wall_clock_s") or 0.0 for r in rows)
    open_qois, decided = list(qois), {}
    while True:
        hs = [r["h"] for r in rows]
        for q in list(open_qois):
            g = gates.qoi_convergence_gate(hs, [r[q] for r in rows], band_tol, gci_tol)
            final = (g["verdict"] == "PASS" or g["reason"].startswith(("divergent",
                                                                        "oscillatory")))
            if final:
                decided[q] = {**g, "levels": [r["N"] for r in rows]}
                open_qois.remove(q)
        Ns = [r["N"] for r in rows]
        log(f"[adaptive] levels {Ns}: decided {len(decided)}/{len(qois)}, open {open_qois}")
        if not open_qois:
            stop = "all QoIs decided"
            break
        if len(rows) >= max_levels:
            stop = "max levels"
            break
        N_next = 2 * Ns[-1] - 1
        n_next = 2 * N_next ** 2
        model, how = _cost_model(list(rows) + list(timing_rows))
        est = model(n_next) if model else None
        if est is None or spent + est > budget_s:
            stop = (f"budget: next level N={N_next} predicted "
                    f"{'unknown' if est is None else f'{est:.0f}s'} ({how}), "
                    f"spent {spent:.0f}s of {budget_s:.0f}s")
            break
        log(f"[adaptive] refining to N={N_next} for {open_qois} (predicted {est:.0f}s, {how})")
        r = solve(N_next)
        rows.append(r)
        spent += r.get("wall_clock_s") or est
    for q in open_qois:
        g = gates.qoi_convergence_gate([r["h"] for r in rows], [r[q] for r in rows],
                                       band_tol, gci_tol)
        decided[q] = {**g, "verdict": "INCONCLUSIVE" if g["verdict"] != "PASS" else "PASS",
                      "reason": f"{g['reason']}; stopped: {stop}",
                      "levels": [r["N"] for r in rows]}
    return {"decisions": decided, "levels": [r["N"] for r in rows], "stop": stop,
            "spent_s": spent}
