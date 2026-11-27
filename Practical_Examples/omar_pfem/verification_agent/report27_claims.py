"""Claims check of advisor_feedback/2026-11-27_Autonomous_Workflow_Progress_Report.pdf: the
numbers are recomputed from the stored files and searched for in the PDF text.

    python -m omar_pfem.verification_agent.report27_claims
"""
import collections
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "advisor_feedback",
                                   "2026-11-27_Autonomous_Workflow_Progress_Report.pdf"))
j = lambda *p: json.load(open(os.path.join(HERE, *p)))


def _e(x):
    """2.0e-3 style."""
    m, ex = f"{x:.1e}".split("e")
    return f"{m}e{int(ex)}"


def claims():
    c = []
    mms = j("apriori_comparison.json")["mms"]
    npass = sum(m["verdict"] == "PASS" for m in mms)
    c.append(("MMS agree", f"{sum(m['within_gate_tol_0.2'] for m in mms if m['verdict'] == 'PASS')} of {npass}", "76 of 76"))
    v1 = collections.Counter(x["outcome"].split(":")[0].split(" (")[0]
                             for a in j("apriori_comparison.json")["runs"] for x in a.get("rows", []))
    c.append(("v1 counts", f"Of {sum(v1.values())} comparisons on the stored runs, {v1['agrees']} agree, "
              f"{v1['slower than predicted']} are slower than predicted and {v1['no measurable order']} have",
              "Of 126 comparisons on the stored runs, 60 agree, 56 are slower than predicted and 10 have"))
    v2 = collections.Counter(x["outcome"] for x in j("apriori_comparison_v2_posthoc.json")["rows"])
    c.append(("v2 counts", f"{v2['agrees']} comparisons agree, {v2['no a-priori prediction']} fall",
              "88 comparisons agree, 22 fall"))
    c.append(("v2 open", str(v2["slower than predicted: CONTRADICTION"]) + " remain unexplained", "2 remain unexplained"))
    b = j("reentrant_score.json")["_rates_beta"]
    for k, s in (("U1", "0.277"), ("U2", "0.276"), ("G2_1.5", "0.41"), ("G1_2", "0.47"), ("G2_4", "0.96"), ("P", "0.59")):
        c.append((f"beta {k}", f"{b[k]:.{len(s) - 2}f}", s))
    res = j("reentrant_result.json")
    hp = max(res["HP"], key=lambda r: r["n_dof"])
    g4 = max(res["G2_4"], key=lambda r: r["n_dof"])
    ext = max(res["HP_posthoc_extension"], key=lambda r: r["n_dof"])
    c += [("hp finest", f"With {hp['n_dof']:,} unknowns hp reached {_e(hp['energy_error_rel'])}", "With 10,922 unknowns hp reached 2.0e-3"),
          ("g4 finest", f"reached {_e(g4['energy_error_rel'])} with {g4['n_dof']:,}", "reached 3.5e-4 with 99,330"),
          ("hp ext", f"reach {_e(ext['energy_error_rel'])} with {ext['n_dof']:,} unknowns", "reach 3.4e-4 with 31,682 unknowns")]
    r2 = j("reentrant_score.json")["R4_hp"]["evidence"]["R2"]
    c.append(("hp R2", f"R² = {r2:.4f}", "R² = 0.9993"))
    ck = j("autonomous_runs", "1911.08728", "run.json")["certified_info"]["v_q"]
    c.append(("v_Q", f"{ck['limit']:.4f} ± {ck['U_abs']:.4f}", "21.5141 ± 0.0133"))
    s3 = j("literature_round3_score.json")
    c.append(("round 3", "4/5" if sum(v["confirmed"] is True for v in s3["predictions"].values()) == 4 else "?", "4/5"))
    bs = j("benchmark_search_score.json")
    gaps = dict(bs["v2_posthoc"]["missing_capabilities_by_paper"])
    c.append(("gap 3D definition", str(gaps["3D problem definition from a paper"]), "4"))
    c.append(("gap curved", str(gaps["curved outer boundary"]), "3"))
    first = j(bs["run"])
    c.append(("bench cost", f"USD {first['llm_cost_usd']:.2f}", "USD 0.61"))
    c.append(("bench v2 cost", f"USD {bs['v2_posthoc']['cost_usd']:.2f}", "USD 0.77"))
    return c


def main():
    flat = re.sub(r"\s+", " ", subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                                             text=True).stdout).replace("−", "-")
    bad = 0
    for name, computed, printed in claims():
        ok = computed == printed and printed in flat
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: data {computed!r}"
              + ("" if printed in flat else f" (PDF text {printed!r} not found)"))
    n = len(claims())
    print(f"{n - bad} of {n} claims found and consistent")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
