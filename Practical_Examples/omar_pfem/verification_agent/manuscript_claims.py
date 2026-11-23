"""Claims check for the manuscript's B3 groove-stress passage (paper/main.tex,
Section "Regional Cauchy stress at the groove"), written 2026-11-23 for the audit.

Every number printed in that passage is either recomputed here from the stored,
stress-corrected data (Cauchy conversion fixed 2026-11-19) or listed as not
recomputable with the reason. Each claim also checks that its printed text is
still present in main.tex, so an edit of the manuscript that changes a number
without updating the data (or the reverse) is caught.

Sources: manuscript_recompute/f1_values.json (family re-solved with the fix),
verify_report_result.json (the report's recomputed claims, same data),
crosscheck_result.json (independent spline values).

Bases: sharp-sphere spreads are relative to the sharp-region stress scale,
Gaussian uncertainties relative to the Gaussian-region stress scale (both: the
finest-level average of the stress norm over that region, as in the report's
Section 4); the "about 2%" spline gap is relative to the averages themselves.

    python -m omar_pfem.verification_agent.manuscript_claims
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TEX = os.path.join(ROOT, "paper", "main.tex")
OUT = os.path.join(HERE, "manuscript_claims_result.json")


def _j(*p):
    return json.load(open(os.path.join(HERE, *p)))


def _vr(section, claim):
    for r in _j("verify_report_result.json"):
        if r["section"] == section and r["claim"] == claim:
            return r["recomputed"]
    raise KeyError((section, claim))


def _gk_rows():
    return [r for r in _j("crosscheck_result.json")["rows"]
            if r["problem"] == "bushing_groove" and r["qoi"] in
            ("gk2_sigma_xx_avg", "gk2_sigma_yy_avg", "gk2_sigma_zz_avg", "gk2_pressure_avg")]


def claims():
    f1 = _j("manuscript_recompute", "f1_values.json")["manuscript_vs_new"]
    new = lambda k: f1[k]["new"]
    rng = new("avg range over eight resolutions 2,376-47,040 ('0.38-0.44')")
    mx = new("max first six levels ('4.13 to 0.706')")
    rise = new("rise at the seventh level ('26%', '0.891 at 45,000')")
    zb = new("z_beta 0-3 at 10,200 ('0.615-0.826')")
    wall = new("inner wall, finest two levels inside the region ('0.3-1.0%')")
    comps = ("sigma_xx", "sigma_yy", "sigma_zz", "pressure")
    gk_p = [_vr("8.3", f"gk2_{c}_avg 5-level p") for c in comps]
    gk_u = [_vr("8.3", f"gk2_{c}_avg U %") for c in comps]
    gk_s = [_vr("8.3", f"gk2_{c}_avg strategy spread %") for c in comps]
    rows = _gk_rows()
    u_val = [100 * r["U_rel"] * r["scale"] / abs(r["limit"]) for r in rows]
    gap_val = [100 * abs(r["v1"]["independent"] - r["limit"]) / abs(r["limit"]) for r in rows]
    mixed = _j("mixed_b3_result.json")
    p1_10200 = [100 * abs(mixed["10200:full:fbar_p1"][f"gk2_{c}_avg"]
                          / mixed["10200:reduced:disp"][f"gk2_{c}_avg"] - 1) for c in comps]
    return [
        # (printed text in main.tex, printed value, recomputed value, tolerance, note)
        ("$0.42$--$0.47$, without trend", 0.42, rng[0], 0.005, "min of the 8 sharp averages"),
        ("$0.42$--$0.47$, without trend", 0.47, rng[1], 0.005, "max of the 8 sharp averages"),
        ("($4.16$ to $0.729$)", 4.16, mx[0], 0.005, "peak, first level"),
        ("($4.16$ to $0.729$)", 0.729, mx[1], 0.0005, "peak, sixth level"),
        ("then rises by $25\\%$", 25, 100 * rise[0], 0.5, "rise at the seventh level"),
        ("($0.910$ at $45{,}000$ elements)", 0.910, rise[1], 0.0005, "peak at 45,000"),
        ("between $0.621$ and $0.836$", 0.621, zb[0], 0.0005, "z_beta range low"),
        ("between $0.621$ and $0.836$", 0.836, zb[1], 0.0005, "z_beta range high"),
        ("at most $0.3$--$1.0\\%$ inside the region", 0.3, wall[0], 0.05, "inner wall low"),
        ("at most $0.3$--$1.0\\%$ inside the region", 1.0, wall[1], 0.05, "inner wall high"),
        ("observed orders $2.3$--$2.7$", 2.3, min(gk_p), 0.05, "Gaussian fit order low"),
        ("observed orders $2.3$--$2.7$", 2.7, max(gk_p), 0.05, "Gaussian fit order high"),
        ("uncertainty of $0.11$--$0.18\\%$", 0.11, min(gk_u), 0.005, "U, % of stress scale"),
        ("uncertainty of $0.11$--$0.18\\%$", 0.18, max(gk_u), 0.005, "U, % of stress scale"),
        ("($0.8$--$1.0\\%$ of the averages", 0.8, min(u_val), 0.05, "U, % of the averages"),
        ("($0.8$--$1.0\\%$ of the averages", 1.0, max(u_val), 0.05, "U, % of the averages"),
        ("by at most $0.5\\%$ across the graded", 0.5, max(gk_s), 0.05, "strategy spread"),
        ("averages about $2\\%$ higher", 2, sum(gap_val) / len(gap_val), 0.15,
         "spline P1 gap, % of the averages (range 1.9-2.1)"),
        ("by $1.4\\%$ at $23{,}520$", 1.4, _vr("25", "gap sigma_xx 23520 %"), 0.05, "P1 vs disp"),
        ("$1.6\\%$ at $15{,}960$", 1.6, _vr("19", "15960 P1 vs disp xx %"), 0.05, "P1 vs disp"),
        ("$1.9$--$2.1\\%$ at $10{,}200$", 1.9, min(p1_10200), 0.05, "P1 vs disp, 4 comps"),
        ("$1.9$--$2.1\\%$ at $10{,}200$", 2.1, max(p1_10200), 0.05, "P1 vs disp, 4 comps"),
        ("still err by $146$--$173\\%$", 146, _vr("20", "best normal gk2 min %"), 0.5, "best"),
        ("still err by $146$--$173\\%$", 173, _vr("20", "best variant gk2 xx %"), 0.5, "best"),
    ]


NOT_RECOMPUTED = {
    "range $1{,}745$--$5{,}448\\%$": "surrogate errors against the 47,040-element reference; "
                                    "evaluator path (P from the displacement, one conversion), "
                                    "unaffected by the stress fix; outputs not in this repository",
    "stays at $2.5$--$8.7\\%$": "same evaluator path, displacement metric, unaffected",
    "by $290\\%$ or more": "other surrogate variants; cluster job output (Section 20 of the "
                          "report quotes it from the same file); checked by hand",
}


def sharp_checks():
    """Sharp-sphere spreads: three finest levels and across the graded meshes."""
    from omar_pfem.verification_agent.verify_report import rules
    R = rules.__wrapped__("bushing_groove")
    ad = R["adapter"]
    out = []
    fin, strat = [], []
    for q in ("sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg"):
        _, v, sc = ad.series(q)
        fin.append(100 * (max(v[-3:]) - min(v[-3:])) / sc)
        strat.append(100 * R["v"][q]["strategy_robustness"]["relative_spread"])
    out += [("within $0.4$--$0.7\\%$ over the three finest levels", 0.4, min(fin), 0.05,
             "sharp, % of sharp stress scale"),
            ("within $0.4$--$0.7\\%$ over the three finest levels", 0.7, max(fin), 0.05,
             "sharp, % of sharp stress scale"),
            ("within\n$1.1$--$2.0\\%$ across the graded meshes", 1.1, min(strat), 0.05,
             "sharp strategy spread"),
            ("within\n$1.1$--$2.0\\%$ across the graded meshes", 2.0, max(strat), 0.05,
             "sharp strategy spread")]
    return out


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    tex = open(TEX).read()
    norm = lambda s: " ".join(s.split())
    tex_n = norm(tex)
    res = []
    for text, printed, value, tol, note in claims() + sharp_checks():
        present = norm(text) in tex_n
        ok = present and abs(value - printed) <= tol
        res.append(dict(text=text, printed=printed, recomputed=value, tol=tol, note=note,
                        present_in_tex=present, ok=ok))
    for text, why in NOT_RECOMPUTED.items():
        res.append(dict(text=text, present_in_tex=norm(text) in tex_n, ok=None, note=why))
    json.dump(res, open(OUT, "w"), indent=1)
    checked = [r for r in res if r["ok"] is not None]
    bad = [r for r in checked if not r["ok"]]
    for r in bad:
        print("MISMATCH", r)
    print(f"{len(checked) - len(bad)} of {len(checked)} manuscript claims match the stored data; "
          f"{len(res) - len(checked)} listed as not recomputed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
