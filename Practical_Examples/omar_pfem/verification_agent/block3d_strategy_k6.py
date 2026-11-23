"""Block strategies one level finer (round J, 2026-11-23; an ANALYSIS, stored apart).

The pre-registered plan checks strategy robustness at k = 4 (block3d.STRATEGY_LEVEL).
There the root Gaussian von Mises average FAILs: spread 3.5% across g = 1, 1.5, 2
(tolerance 2%), anchored deviation 3.0%. Question: is that a property of the quantity
or of the coarse k = 4 meshes? Same gradings at k = 6; the same gates with the same
tolerances are applied, the family limit (anchored check) is the stored one. The
pre-registered verdict is not replaced; this is reported next to it.

    python -m omar_pfem.verification_agent.block3d_strategy_k6
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "block3d_strategy_k6.json")
QOIS = ("strain_energy", "w_tip", "mid_gk_avg_sxx", "root_gk_avg_von_mises")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent import block3d as B, gates
    from omar_pfem.verification_agent.verify_report import rules
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": []}
    done = {r["grading"] for r in res["rows"]}
    ladder = json.load(open(B.CACHE))
    for g in (1.0, 1.5, 2.0):
        if g in done:
            continue
        if g == 1.0:
            row = next(r for r in ladder if r["k"] == 6 and r["grading"] == 1.0)
        else:
            row = B.solve(6, g)
        res["rows"].append({k: row.get(k) for k in ("grading", "min_detF", "seconds") + QOIS})
        json.dump(res, open(OUT, "w"), indent=1)
    rows = sorted(res["rows"], key=lambda r: r["grading"])
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    try:
        R = rules.__wrapped__("clamped_block_3d")
    finally:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    out = {}
    for q in QOIS:
        v = [r[q] for r in rows]
        ml = R["v"][q].get("qoi_convergence_multilevel", {})
        c3 = R["v"][q].get("qoi_convergence", {})
        lim = ml.get("fit_limit") if ml.get("evidence") == "rate" else c3.get("richardson_extrapolate")
        sr = gates.strategy_robustness_gate(v, tol=0.02)
        k4 = R["v"][q].get("strategy_robustness", {})
        out[q] = {"k6_values": v, "k6_spread": sr["relative_spread"], "k6_verdict": sr["verdict"],
                  "k6_max_dev_from_limit": (max(abs(x - lim) for x in v) / abs(lim)
                                            if lim else None),
                  "k4_spread": k4.get("relative_spread"), "k4_verdict": k4.get("verdict")}
    res["comparison"] = out
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    r = main()
    for q, v in r["comparison"].items():
        print(q, {k: v[k] for k in v if k != "k6_values"})
    print([(x["grading"], x["min_detF"]) for x in r["rows"]])
