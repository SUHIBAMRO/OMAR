"""Adaptive layer of the verification agent (prototype).

Two pieces, both deterministic:

1. outcome_prior: before computing anything, predict each QoI's verdict
   from the solved-case repository, weighting past records by fingerprint
   similarity (GRAFT's similarity-weighted prior, with the reward replaced
   by the gate verdict). This is what the repository "learns".

2. adaptive_refine: a refinement controller. Starting from the three
   coarsest levels, it re-evaluates BOTH convergence gates after each level
   (three-level Celik, and from 4 levels on the multilevel least-squares
   gate over the finest 5), combines them with the same policy as the
   orchestrator (gates.combine_verdicts), and requests the next (finer)
   level only for QoIs that are still undecided and only if the predicted
   cost fits the remaining budget. A QoI is decided
     PASS  when the multilevel gate is available (>= 4 levels), every gate
           passes, and -- with require_rate (default) -- the evidence is a
           rate; a band-only PASS keeps the QoI open, since more levels may
           certify a rate;
     FAIL  on a divergent or oscillatory three-level sequence outside the
           band (more uniform refinement cannot fix those).
   Otherwise it stays open until max_levels or the budget; then a QoI whose
   gates all pass is reported PASS with its (weaker) evidence and the stop
   reason, anything else INCONCLUSIVE. Tolerances default to the per-class
   ceilings (trees.CLASS_CEILINGS) and the order window to the element's.

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


def _gates(hs, vals, band_tol, gci_tol, window):
    out = {"qoi_convergence": gates.qoi_convergence_gate(hs, vals, band_tol, gci_tol,
                                                         order_window=window)}
    if len(vals) >= 4:
        out["qoi_convergence_multilevel"] = gates.multilevel_convergence_gate(
            hs, vals, band_tol, gci_tol, order_window=window)
    return out


def adaptive_refine(solve, qois, N0=(11, 21, 41), budget_s=1800.0, max_levels=6,
                    band_tol=None, gci_tol=None, element=None, require_rate=True,
                    timing_rows=(), log=print):
    """solve(N) -> row dict with 'h', 'n_dof', optional 'wall_clock_s', and
    the QoIs. band_tol/gci_tol: None -> the QoI class ceiling. Returns
    per-QoI decisions (combined verdict, evidence, both gate results, levels)
    and the levels used."""
    window = trees.order_window(element)

    def tols(q):
        c = trees.class_params("qoi_convergence", trees.classify_qoi(q))
        return (c["band_tol"] if band_tol is None else band_tol,
                c["gci_tol"] if gci_tol is None else gci_tol)

    def evaluate(q):
        hs, vals = [r["h"] for r in rows], [r[q] for r in rows]
        g = _gates(hs, vals, *tols(q), window)
        overall, evidence, src = gates.combine_verdicts(g)
        return g, overall, evidence, src

    rows = [solve(N) for N in N0]
    spent = sum(r.get("wall_clock_s") or 0.0 for r in rows)
    open_qois, decided = list(qois), {}
    while True:
        for q in list(open_qois):
            g, overall, evidence, src = evaluate(q)
            g3 = g["qoi_convergence"]
            final_fail = (g3["verdict"] == "FAIL"
                          and g3["reason"].startswith(("divergent", "oscillatory")))
            final_pass = (overall == "PASS" and "qoi_convergence_multilevel" in g
                          and (evidence == "rate" or not require_rate))
            if final_fail or final_pass:
                decided[q] = {"verdict": "FAIL" if final_fail else "PASS",
                              "evidence": "none" if final_fail else evidence,
                              "evidence_from": [] if final_fail else src,
                              "reason": g3["reason"] if final_fail else
                              " | ".join(f"{k}: {v['reason']}" for k, v in g.items()),
                              "gates": g, "levels": [r["N"] for r in rows]}
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
        g, overall, evidence, src = evaluate(q)
        verdict = "PASS" if overall == "PASS" else "INCONCLUSIVE"
        note = ("; rate not certified" if verdict == "PASS" and evidence != "rate" else
                "; multilevel gate not reached" if "qoi_convergence_multilevel" not in g else "")
        decided[q] = {"verdict": verdict, "evidence": evidence if verdict == "PASS" else "none",
                      "evidence_from": src if verdict == "PASS" else [],
                      "reason": " | ".join(f"{k}: {v['reason']}" for k, v in g.items())
                      + f"; stopped: {stop}{note}",
                      "gates": g, "levels": [r["N"] for r in rows]}
    return {"decisions": decided, "levels": [r["N"] for r in rows], "stop": stop,
            "spent_s": spent}
