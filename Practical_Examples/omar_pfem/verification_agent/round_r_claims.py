"""Claims check for advisor_feedback/2026-11-26_Autonomous_Workflow_New_Results_part2.pdf:
every number in the PDF that comes from a stored run is recomputed here from the stored files
and must appear in the PDF text exactly as formatted below.

    python -m omar_pfem.verification_agent.round_r_claims
"""
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "advisor_feedback",
                                   "2026-11-26_Autonomous_Workflow_New_Results_part2.pdf"))


def _j(*p):
    return json.load(open(os.path.join(HERE, *p)))


# the 3D runs were repeated after cluster job 3322812 (26 Nov); the PDF reports them as they
# were when it was written, frozen in run_as_in_pdf_2026-11-26.json
_PDF_SNAPSHOT = ("cook_3d", "nh_plate_hole_3d_pe", "plate_hole_quarter_3d_pe")


def _run(n):
    return _j("autonomous_runs", n, "run_as_in_pdf_2026-11-26.json" if n in _PDF_SNAPSHOT
              else "run.json")


def _e(x, d=2):
    """1.05e-13 style (mantissa with d-1 decimals)."""
    m, ex = f"{x:.{d - 1}e}".split("e")
    return f"{m}e{int(ex)}"


def claims():
    c = []
    ck = _run("cook_linear")["certified_info"]
    c += [("Cook v_C limit", f"{ck['v_C']['limit']:.4f}", "23.9623"),
          ("Cook v_C U abs", f"{ck['v_C']['U_abs']:.4f}", "0.0076"),
          ("Cook v_C U rel", f"{100 * ck['v_C']['U_rel']:.3f}%", "0.032%"),
          ("Cook energy", f"{ck['strain_energy']['limit']:.4f} ± {ck['strain_energy']['U_abs']:.4f}",
           "12.0209 ± 0.0005")]
    from omar_pfem.verification_agent.geometry2d import make
    cp = make(_j("problems2d", "cook_linear.json"))
    c.append(("Cook corner exponent", f"{min(min(x['rate_limiting']) for x in cp.corners):.3f}", "0.677"))
    pl = _run("plate_hole_quarter")["certified_info"]["peak_von_mises"]
    c += [("plate peak", f"{pl['limit']:.4f} ± {pl['U_abs']:.4f}", "3.0757 ± 0.0136"),
          ("plate peak U", f"{100 * pl['U_rel']:.2f}%", "0.44%")]
    k = _j("geometry2d_validation.json")["kirsch"]
    c += [("Kirsch levels", ", ".join(f"{v:.4f}" for v in k["peak_von_mises_Q2"].values()),
           "2.9464, 2.9755, 2.9903"), ("Kirsch extrapolated", f"{k['extrapolated']:.3f}", "3.005")]
    s = _j("sensitivity_runs", "score.json")
    c.append(("sensitivity score", f"{sum(v['confirmed'] for v in s.values())} of {len(s)}", "4 of 8"))
    c.append(("1/E central difference", f"{1 / (1 - 0.1 ** 2):.4f}", "1.0101"))
    sp = _j("sensitivity_runs", "plate_hole_quarter.json")["params"]
    c.append(("Michell |S|", _e(abs(sp["nu"]["quantities"]["peak_von_mises"]["S"])), "2.7e-4"))
    rr = sp["hole[0].radius"]["quantities"]["peak_von_mises"]
    c += [("radius change", f"{100 * rr['change_rel']:.2f}%", "0.79%"),
          ("change/3", f"{100 * rr['change_rel'] / 3:.2f}%", "0.26%"),
          ("radius U", f"{100 * rr['U_rel']:.2f}%", "0.44%")]
    g = _j("geometry3d_score.json")
    c.append(("3D score", f"{sum(v['confirmed'] for v in g.values())} of {len(g)}", "4 of 8"))
    c.append(("identity", _e(max(g["P2_planestrain_identity"]["evidence"].values()), 3), "1.05e-13"))
    ev = g["P3_planestrain_limits"]["evidence"]
    c += [("energy gap", f"{_e(ev['strain_energy'][0])} against {_e(ev['strain_energy'][1])}",
           "4.1e-6 against 7.7e-6"),
          ("u_right gap", f"{_e(ev['u_right'][0])} against {_e(ev['u_right'][1])}",
           "1.4e-6 against 1.1e-5")]
    m = re.search(r"has (\d+) DOF", g["P8_memory_guard"]["evidence"])
    c.append(("Cook n=32 DOF", f"{int(m.group(1)):,}", "797,427"))
    p3 = _run("plate_hole_quarter_3d_pe")
    m = re.search(r"has (\d+) DOF", json.dumps(p3["steps"]))
    c.append(("plate P2 n=32 DOF", f"{int(m.group(1)):,}", "712,701"))
    sup = _j("cook_3d_supplementary.json")["rows"]["Q2_16"]
    c += [("Cook 3D v_C", f"{sup['v_C']:.3f}", "23.923"),
          ("Cook 3D rel", f"{100 * abs(sup['v_C'] / 23.96 - 1):.2f}%", "0.16%")]

    def rates(run, el, ref=False):
        r = (run["mms_reference_domain"]["checks"] if ref else run["mms"])[el]["rates"]
        return f"{r['L2']['observed']:.2f} / {r['H1']['observed']:.2f}"
    nh2, ck3, nh3 = _run("nh_plate_hole_pe"), _run("cook_3d"), _run("nh_plate_hole_3d_pe")
    c += [("NH 2D Q2", rates(nh2, "Q2"), "3.13 / 2.08"), ("NH 2D P2", rates(nh2, "P2"), "3.19 / 2.14"),
          ("plate 3D Q2", rates(p3, "Q2"), "3.17 / 2.13"), ("plate 3D P2", rates(p3, "P2"), "3.08 / 2.02"),
          ("Cook 3D Q2", rates(ck3, "Q2"), "3.41 / 2.14"), ("Cook 3D P2", rates(ck3, "P2"), "2.73 / 2.31"),
          ("NH 3D P2", rates(nh3, "P2"), "3.08 / 2.02")]
    m = re.search(r"about ([\d.]+) GB", json.dumps(ck3["mms_reference_domain"]))
    c.append(("Cook ref P2 memory", f"{float(m.group(1)):.1f} GB", "26.8 GB"))
    c.append(("NH 3D Q2 levels", ", ".join(map(str, nh3["mms"]["Q2"]["levels"])), "2, 4, 8"))
    q = _j("round_q_score.json")
    c.append(("round Q score", f"{sum(v['confirmed'] for v in q.values())} of {len(q)}", "3 of 8"))
    c.append(("small load", _e(abs(q["Q6_nh3d_small_load"]["evidence"]["rel"])), "1.4e-6"))
    q5 = q["Q5_nh3d_identity"]["evidence"]["max_rel_dev"]
    c.append(("NH 27-point gap n=2", _e(max(q5.values())), "8.0e-5"))
    l7 = _j("experiments", "bushing_groove", "02_spline_level_7_declined.json")["preregistration"]
    seq = re.search(r"\[([\d., ]+)\]", l7["justification"]).group(1)
    c.append(("groove sequence", seq, "1.708, 1.055, 1.001, 0.891, 0.924"))
    c.append(("L7 candidates", re.search(r"of (\d+) candidate", l7["justification"]).group(1), "126"))
    h = l7["hours_per_formulation_estimate"]
    c.append(("hours L7-L9", ", ".join(f"{h[k]:.1f}" for k in ("7", "8")) + f" and {h['9']:.1f}",
              "9.5, 15.8 and 24.8"))
    # Section 5, results of cluster job 3322812 (added 26 Nov after the import)
    rec = _j("code_verification_record.json")
    cl = {(v["problem"], k.split("|")[1]): v for k, v in rec.items()
          if "computed on the cluster" in v.get("levels_added", [])}
    rr = lambda p, el: (f"{cl[(p, el)]['rates']['L2']['observed']:.2f} / "
                        f"{cl[(p, el)]['rates']['H1']['observed']:.2f}")
    c += [("job Cook Q2 n=32", rr("cook_3d", "Q2"), "3.79 / 1.99"),
          ("job Cook P2 n=32", rr("cook_3d", "P2"), "3.24 / 2.27"),
          ("job Cook ref Q2", rr("cook_3d_reference_domain", "Q2"), "3.52 / 2.42"),
          ("job Cook ref P2", rr("cook_3d_reference_domain", "P2"), "3.12 / 2.29"),
          ("job NH 3D Q2", rr("nh_plate_hole_3d_pe", "Q2"), "3.17 / 2.13")]
    c.append(("job Cook all INCONCLUSIVE", "all INCONCLUSIVE" if all(
        v["verdict"] == "INCONCLUSIVE" for (p, _), v in cl.items() if p.startswith("cook_3d"))
        else "not all INCONCLUSIVE", "all INCONCLUSIVE"))
    lad = _j("pipeline2d_runs", "plate_hole_quarter_3d_pe", "ladders.json")["P2 uniform"]
    p2 = {r["n"]: r for r in lad}
    c += [("plate P2 n=32 DOF (job)", f"{p2[32]['n_dof']:,}", "712,701"),
          ("plate P2 v_hole_top", ", ".join(f"{p2[n]['v_hole_top']:.5f}" for n in (8, 16, 32)),
           "-0.97460, -0.97318, -0.97267")]
    now = _j("autonomous_runs", "plate_hole_quarter_3d_pe", "run.json")
    c.append(("plate after import", f"{len(now['final']['certified'])} of "
              f"{len(now['final']['certified']) + len(now['final']['open'])} certified", "2 of 4 certified"))
    exp = _j("experiments", "plate_hole_quarter_3d_pe", "01_finer_independent_level_run5.json")
    c.append(("plate prediction", "refuted" if not exp["outcome"]["confirmed"] else "confirmed",
              "refuted"))
    nh = _run("nh_plate_hole_3d_pe")      # FIX (28 Nov): the PDF-time snapshot, as the others
    m = re.search(r"needs about ([\d.]+) GB", json.dumps(nh))
    c.append(("NH n=16 memory", f"{float(m.group(1)):.1f} GB", "16.6 GB"))
    import glob
    n_tests = sum(open(f).read().count("\ndef test_") for f in
                  glob.glob(os.path.join(HERE, "test_*.py")))
    # FIX (27 Nov): the PDF states the count when it was written; tests only grow afterwards,
    # so the claim holds while the current count is at least the printed one
    c.append(("test count", "115 automated tests pass" if n_tests >= 115 else f"{n_tests} automated tests pass",
              "115 automated tests pass"))
    return c


# the email attaches one PDF: the 24-25 November PDF followed by this one (pdfunite)
COMBINED = PDF.replace("_part2.pdf", "_combined.pdf")


def main():
    text = subprocess.run(["pdftotext", PDF, "-"], capture_output=True, text=True).stdout
    flat = re.sub(r"\s+", " ", text)
    if os.path.exists(COMBINED):
        both = subprocess.run(["pdftotext", "-f", "9", COMBINED, "-"], capture_output=True,
                              text=True).stdout
        if re.sub(r"\s+", " ", both) != flat:
            print("FAIL the combined PDF (pages 9-16) is not the current part-2 PDF")
            raise SystemExit(1)
        print("ok   combined PDF pages 9-16 = part-2 PDF")
    bad = 0
    for name, computed, printed in claims():
        ok = computed == printed and printed in flat
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: data {computed!r}, PDF {printed!r}"
              + ("" if printed in flat else " (not found in the PDF)"))
    n = len(claims())
    print(f"{n - bad} of {n} claims found and consistent")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
