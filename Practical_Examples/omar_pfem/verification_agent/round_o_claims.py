"""Claims check of the round-O new-results report: every number printed there is recomputed from
the stored data and searched for in the exported PDF.

    python -m omar_pfem.verification_agent.round_o_claims
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "..", "..", "advisor_feedback",
                   "2026-11-24_Autonomous_Workflow_New_Results.pdf")
j = lambda p: json.load(open(os.path.join(HERE, p)))


def claims():
    from omar_pfem.verification_agent.experiment import prior
    imp = j("adopted_impact.json")
    ch = imp["changed"]
    run = lambda p: j(f"autonomous_runs/{p}/run.json")
    mem = j("experience.json")
    sc = j("autonomy_test/score.json")
    ph = j("autonomy_test/posthoc/results_posthoc.json")
    g = j("experiments/bushing_groove/01_spline_level_6.json")["preregistration"]
    from omar_pfem.verification_agent.groove_experiment import analyse
    a = analyse()
    lsh = None
    with __import__("omar_pfem.verification_agent.adopted", fromlist=["x"]).ruleset():
        from omar_pfem.verification_agent.verify_report import rules
        lsh = rules.__wrapped__("l_shape")["v"]["v_right_top"]["singular_order"]
    t_mms = j("code_verification_record.json")
    tb = next(v for k, v in t_mms.items() if v["problem"] == "t_bracket" and "|Q2|" in k)
    e = tb["errors"]["L2"]
    import math
    pw = [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
    from omar_pfem.verification_agent.corners_ext import complex_exponents, exponents
    from omar_pfem.verification_agent.remedy import corner_exponents
    C = [
        ("changed verdicts", len(ch), "{:d}", lambda v: v == 1),
        ("changed one is the L-shape tip", ch[0]["problem"] + ch[0]["qoi"], "{}", lambda v: v == "l_shapev_right_top"),
        ("certified of 83", imp["adopted_star"]["certified_of_verifiable"], "{:d}", lambda v: v == 58),
        ("missed failures", imp["adopted_star"]["missed_failures"], "{:d}", lambda v: v == 0),
        ("pairwise 1", lsh["pairwise_orders"][0], "{:.2f}", None),
        ("pairwise 2", lsh["pairwise_orders"][1], "{:.2f}", None),
        ("pairwise 3", lsh["pairwise_orders"][2], "{:.2f}", None),
        ("bound", lsh["expected"], "{:.2f}", None),
        ("T-bracket certified", len(run("t_bracket")["final"]["certified"]), "{:d}", lambda v: v == 7),
        ("experiments", len(mem), "{:d}", None),
        ("confirmed", sum(m["confirmed"] for m in mem), "{:d}", None),
        ("prior graded", prior("graded_meshes", "corner_limited")[0], "{:.2f}", None),
        ("trials graded", prior("graded_meshes", "corner_limited")[1], "{:d}", None),
        ("prior finer graded", prior("finer_graded_level", "region_average:band_only")[0], "{:.2f}", None),
        ("trials finer graded", prior("finer_graded_level", "region_average:band_only")[1], "{:d}", None),
        ("MMS pairwise 1", pw[0], "{:.2f}", None), ("MMS pairwise 2", pw[1], "{:.2f}", None),
        ("MMS pairwise 3", pw[2], "{:.2f}", None),
        ("MMS fitted after level", tb["rates"]["L2"]["observed"], "{:.2f}", None),
        ("complex clamped-free 270", complex_exponents(4.71238898038469, "clamped", "free", 0.3)[0][0], "{:.2f}", None),
        ("former analysis 270", min(corner_exponents(4.71238898038469, "clamped", "free", 0.3)), "{:.2f}", None),
        ("T4 certified", len(run("t4_step_plate")["final"]["certified"]), "{:d}", lambda v: v == 6),
        ("test cost", sc["llm_cost_usd"], "{:.2f}", None),
        ("predictions met", sum(sc["predictions"].values()), "{:d}", lambda v: v == 3),
        ("post hoc cost", sum(r["llm_cost_usd"] for r in ph.values()), "{:.2f}", None),
        ("P1 change 2-3 %", 100 * a["P1_changes"]["2->3"], "{:.1f}", None),
        ("P1 change 3-4 %", 100 * a["P1_changes"]["3->4"], "{:.1f}", None),
        ("P1 change 4-5 %", 100 * a["P1_changes"]["4->5"], "{:.1f}", None),
        ("gap L2", a["gap_disp_minus_P1"][2], "{:.3f}", None),
        ("gap L3", a["gap_disp_minus_P1"][3], "{:.3f}", None),
        ("gap L4", a["gap_disp_minus_P1"][4], "{:.3f}", None),
        ("gap L5", a["gap_disp_minus_P1"][5], "{:.3f}", None),
        ("predicted change", g["prediction"]["last_change"], "{:.3f}", None),
        ("predicted gap", g["prediction"]["last_gap"], "{:.3f}", None),
        ("L6 unknowns", a["dof6"], "{:,}", None),
        ("hours per formulation", a["hours_per_formulation"], "{:.0f}", None),
    ]
    return C


def main():
    import warnings
    warnings.filterwarnings("ignore")
    txt = subprocess.run(["pdftotext", PDF, "-"], capture_output=True, text=True).stdout
    flat = re.sub(r"\s+", " ", txt).replace("−", "-")
    bad = 0
    for name, val, fmt, cond in claims():
        s = fmt.format(val)
        found = (s.lstrip("-") in flat) if isinstance(val, (int, float)) else True
        ok = found and (cond is None or cond(val))
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {s}")
    print(f"{len(claims()) - bad} of {len(claims())} claims found and consistent")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
