"""The agent against conventional solution verification (Professor Rabczuk:
"agent against conventional verification"), on every problem with stored data.

Conventional procedure (Celik et al. 2008 / ASME V&V 20 practice): for each
quantity, the three finest levels of one refinement family; Richardson order
and GCI; accepted iff the three values are monotone with shrinking differences
and GCI <= the tolerance. No order window, no band acceptance, no second
refinement strategy, no singularity theory, no five-level fit. Same tolerances
as the agent (class ceilings), same stored solves.

Agent: its final verdict (rules plan, trees + gates + combination), as `agent
rules` computes it.

Reference ("truth"), stated so it can be disputed: a pointwise peak at a
singular corner (B1, Cook, plate corner, L-shape) has no finite limit; the B3
groove maxima are not convergent on any mesh family tried (Sections 9, 15, 19,
25); every other quantity is taken as verifiable. This truth rests on the same
corner theory the agent encodes, so the comparison shows what the extra checks
add, not an independent ground truth. The B3 sharp-sphere averages are a third
class: stable, but without a demonstrated rate ("band").

    python -m omar_pfem.verification_agent.conventional
"""
import json
import os

from omar_pfem.verification_agent import gates, trees
from omar_pfem.verification_agent.agent import DEFAULT_SPECS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "conventional_result.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape")


def truth(geometry, qoi):
    cls = trees.classify_qoi(qoi)
    reg = trees.regularity_class(DEFAULT_SPECS[geometry]["regularity"])
    if geometry == "bushing_groove":
        if qoi in ("sigma_xy_max", "cyl_rt_max", "cyl_tz_max"):
            # zero by symmetry on the theta = 0 plane through the root: stored maxima
            # 0.01-0.03 against a stress scale of 0.91 are noise around zero; label
            # corrected after the first run (it had been "not_verifiable")
            return "verifiable"
        if qoi.endswith("_max") or "max_" in qoi or "scf" in qoi:
            return "not_verifiable"
        if qoi.startswith("sigma_") and qoi.endswith("_avg"):
            return "band_only"                       # sharp sphere: stable, no rate
        return "verifiable"
    if geometry == "plate_hole" and qoi == "hole_peak_sigma_xx":
        return "verifiable"                          # smooth finite-radius peak
    if cls == "pointwise_peak" and reg == "singular":
        return "not_verifiable"
    return "verifiable"


def conventional(adapter, qoi):
    hs, vals, scale = adapter.series(qoi)
    tol = trees.class_params("qoi_convergence", trees.classify_qoi(qoi))["gci_tol"]
    g = gates.qoi_convergence_gate(hs, vals, band_tol=0.0, gci_tol=tol, scale=scale)
    ok = g["verdict"] == "PASS" and g.get("evidence") == "rate"
    return ("PASS" if ok else "FAIL"), g


def main():
    from omar_pfem.verification_agent.verify_report import rules
    rows = []
    for geo in PROBLEMS:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        R = rules.__wrapped__(geo)
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        Rp = rules.__wrapped__(geo)
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        ad = R["adapter"]
        for q, comb in R["combine"].items():
            conv, g = conventional(ad, q)
            t = truth(geo, q)
            rows.append({"problem": geo, "qoi": q, "truth": t, "conventional": conv,
                         "conventional_reason": g.get("reason", ""), "agent": comb[0],
                         "agent_evidence": comb[1], "agent_posthoc": Rp["combine"][q][0],
                         "n_levels": len(ad.hs)})

    def score(key):
        verif = [r for r in rows if r["truth"] == "verifiable"]
        bad = [r for r in rows if r["truth"] == "not_verifiable"]
        band = [r[key] for r in rows if r["truth"] == "band_only"]
        return {"verifiable": len(verif), "not_verifiable": len(bad),
                "certified": sum(r[key] == "PASS" for r in verif),
                "wrongly_failed": sum(r[key] == "FAIL" for r in verif),
                "inconclusive": sum(r[key] == "INCONCLUSIVE" for r in verif),
                "missed_failures": sum(r[key] == "PASS" for r in bad),
                "band_only_passed": sum(1 for b in band if b == "PASS"), "band_only": len(band)}
    res = {"conventional": score("conventional"), "agent": score("agent"),
           "agent_posthoc": score("agent_posthoc"),
           "per_problem": {}, "rows": rows,
           "disagreements": [r for r in rows if r["conventional"] != r["agent"]]}
    for geo in PROBLEMS:
        sub = [r for r in rows if r["problem"] == geo]
        res["per_problem"][geo] = {
            "n": len(sub),
            "conventional_missed": [r["qoi"] for r in sub if r["truth"] == "not_verifiable"
                                    and r["conventional"] == "PASS"],
            "agent_missed": [r["qoi"] for r in sub if r["truth"] == "not_verifiable"
                             and r["agent"] == "PASS"],
            "conventional_false_alarms": [r["qoi"] for r in sub if r["truth"] == "verifiable"
                                          and r["conventional"] != "PASS"],
            "agent_false_alarms": [r["qoi"] for r in sub if r["truth"] == "verifiable"
                                   and r["agent"] != "PASS"],
            "agent_posthoc_not_passed": [r["qoi"] for r in sub if r["truth"] == "verifiable"
                                         and r["agent_posthoc"] != "PASS"]}
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("conventional", "agent", "agent_posthoc"):
        print(f"{k:<14}", res[k])
    for geo, p in res["per_problem"].items():
        print(geo, {k: v for k, v in p.items() if v})


if __name__ == "__main__":
    main()
