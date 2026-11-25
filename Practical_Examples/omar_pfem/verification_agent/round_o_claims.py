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
        ("experiments", len(mem), "made {:d} experiments", None),
        ("confirmed", sum(m["confirmed"] for m in mem), "and {:d} confirmed", None),
        ("prior graded", prior("graded_meshes", "corner_limited")[0], "now {:.2f} for graded", None),
        ("trials graded", prior("graded_meshes", "corner_limited")[1], "problems ({:d} trials)", None),
        ("prior finer graded", prior("finer_graded_level", "region_average:band_only")[0], "and {:.2f} for a further", None),
        ("trials finer graded", prior("finer_graded_level", "region_average:band_only")[1], "level ({:d} trials)", None),
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
    ] + claims_25nov()
    return C


def claims_25nov():
    """Section 8 (25 November): neo-Hookean in the loop and the literature search."""
    from omar_pfem.verification_agent import gates
    rec = j("code_verification_record.json")
    nh = next(v for k, v in rec.items() if k.endswith("|neo_hookean") and "|Q2|" in k
              and v["problem"] == "nh_l_bracket_demo")
    lad = sorted(j("pipeline2d_runs/nh_l_bracket_demo/ladders.json")["Q2 uniform"], key=lambda r: r["n"])
    r5 = [r for r in lad if r["n"] <= 32]
    g = gates.multilevel_convergence_gate([r["h"] for r in r5], [r["strain_energy"] for r in r5],
                                          order_window=(0.5, 5))
    run = j("autonomous_runs/nh_l_bracket_demo/run.json")
    import glob
    fail = json.load(open(sorted(glob.glob(os.path.join(HERE, "experiments/nh_l_bracket_demo/*graded_meshes.json")))[0]))
    sc = j("literature_search_score.json")
    return [
        ("NH MMS L2", nh["rates"]["L2"]["observed"], "{:.2f}", None),
        ("NH MMS H1", nh["rates"]["H1"]["observed"], "{:.2f}", None),
        ("NH energy order", g["fit_order"], "{:.2f}", None),
        ("NH 2 lam", run["small_load_check"]["linearized_2lam"], "{:.2f}", None),
        ("NH min J n=2", lad[0]["min_detF"], "{:.2f}", lambda v: lad[0]["n"] == 2),
        ("NH min J n=64", lad[-1]["min_detF"], "{:.2f}", lambda v: lad[-1]["n"] == 64),
        ("graded failed with 64 steps", int("64]" in fail["outcome"]["failed"]) * 64, "{:d}", lambda v: v == 64),
        ("small-load order", run["small_load_check"]["energy_order"], "{:.2f}", None),
        ("papers found", sc["candidates"], "{:d}", None),
        ("abstract screen", sc["keyword_pass"], "{:d}", None),
        ("judged by LLM", sc["scouted"], "{:d}", None),
        ("rated representable", sc["scout_positive"], "{:d}", lambda v: v == 1),
        ("search cost", sc["llm_cost_usd"], "{:.2f}", None),
        ("Cook mentions", len(sc["cook_membrane_mentions"]), "{:d}", lambda v: v == 4),
        ("search predictions met", sum(sc["predictions"].values()), "{:d}", lambda v: v == 3),
    ] + claims_l6()


def claims_l6():
    """Section 8.3: the level-6 groove experiment."""
    from omar_pfem.verification_agent import gates, trees
    d, n = j("iga_b3_result.json"), j("iga_b3_l6_result.json")
    o = j("experiments/bushing_groove/01_spline_level_6.json")["outcome"]
    row = lambda lv, f: (d.get(f"L{lv}:p2{f}") or n.get(f"L{lv}:p2{f}"))
    h = {lv: 1.0 / row(lv, "")["spans"][0] for lv in (2, 3, 4, 5, 6)}
    fit = lambda q, f: gates.multilevel_convergence_gate(
        [h[l] for l in (2, 3, 4, 5, 6)], [row(l, f)[q] for l in (2, 3, 4, 5, 6)],
        order_window=trees.order_window("Q20"), **trees.class_params(
            "qoi_convergence_multilevel", trees.classify_qoi(q)))
    xr = next(r for r in j("crosscheck_result.json")["rows"]
              if r["problem"] == "bushing_groove" and r["qoi"] == "gk2_sigma_xx_avg")
    return [
        ("prediction confirmed", int(o["confirmed"]), "{:d}", lambda v: v == 1),
        ("P1 hours", n["L6:p2:fbar_p1"]["seconds"] / 3600, "{:.1f}", None),
        ("disp hours", n["L6:p2"]["seconds"] / 3600, "{:.1f}", None),
        ("P1 peak L6", n["L6:p2:fbar_p1"]["max_sigma_xx_R0"], "{:.2f}", None),
        ("disp peak L6", n["L6:p2"]["max_sigma_xx_R0"], "{:.2f}", None),
        ("change L5-L6 %", 100 * o["change_L5_L6"], "{:.1f}", None),
        ("gap L6", o["gap_L6"], "{:.3f}", None),
        ("moment order P1", fit("reaction_moment_y", ":fbar_p1")["fit_order"], "{:.2f}", None),
        ("energy order P1", fit("strain_energy", ":fbar_p1")["fit_order"], "{:.2f}", None),
        ("moment rate", int(fit("reaction_moment_y", ":fbar_p1")["evidence"] == "rate"), "{:d}", lambda v: v == 1),
        ("Gaussian P1 no rate", int(fit("gk2trunc_sigma_xx_avg", ":fbar_p1").get("evidence") != "rate"), "{:d}", lambda v: v == 1),
        ("P1 Gaussian L5", d["L5:p2:fbar_p1"]["gk2trunc_sigma_xx_avg"], "{:.4f}", None),
        ("P1 Gaussian L6", n["L6:p2:fbar_p1"]["gk2trunc_sigma_xx_avg"], "{:.4f}", None),
        ("P1 above Q20 %", 100 * (n["L6:p2:fbar_p1"]["gk2trunc_sigma_xx_avg"] / xr["limit"] - 1), "{:.1f}", None),
        ("disp above Q20 %", 100 * (n["L6:p2"]["gk2trunc_sigma_xx_avg"] / xr["limit"] - 1), "{:.2f}", None),
    ]


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
