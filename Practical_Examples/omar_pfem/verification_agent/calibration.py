"""Calibration of the certified uncertainty against TRUE errors (round K, 2026-11-23).

Every quantity whose continuum value is known exactly is collected:
  thick_cylinder_3d   smooth 3D problem, exact 1D solution (cylinder3d_exact.json)
  l_shape_exact       singular 2D problem, exact Williams solution (lshape_exact_values.json)
  l_shape_exact, g = 3.7   the same problem on the agent's graded meshes (remedy.py)
For each quantity certified with a RATE, by the framework (deterministic rules with the
post hoc singular-order extension) and by conventional GCI (conventional.py), the
effectivity is |exact - certified limit| / (U x scale), scale = |finest value|:
below 1 the interval contains the exact value; far below 1 it is conservative.

    python -m omar_pfem.verification_agent.calibration          (numbers + figures)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calibration_result.json")
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "advisor_feedback", "report_build",
                                      "figures", "verification"))


def _j(name):
    return json.load(open(os.path.join(HERE, name)))


def _limit(gv):
    ml, c3 = gv.get("qoi_convergence_multilevel", {}), gv.get("qoi_convergence", {})
    if ml.get("evidence") == "rate":
        return ml["fit_limit"], ml["uncertainty_fine"]
    if c3.get("evidence") == "rate":
        return c3["richardson_extrapolate"], c3["gci_fine"]
    return None


def collect():
    from omar_pfem.verification_agent import gates, trees
    from omar_pfem.verification_agent.conventional import conventional
    from omar_pfem.verification_agent.verify_report import rules
    exact = {"thick_cylinder_3d": _j("cylinder3d_exact.json"),
             "l_shape_exact": _j("lshape_exact_values.json")}
    xc = _j("crosscheck_result.json")
    final = {(r["problem"], r["qoi"]): r["status"]
             for r in xc["rows"] + xc.get("rows_posthoc_only", [])}
    rows = []
    for geo, ex in exact.items():
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        try:
            R = rules.__wrapped__(geo)
        finally:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        ad = R["adapter"]
        for q, comb in R["combine"].items():
            e = ex.get(q)
            if not isinstance(e, float) or not np.isfinite(e):
                continue
            _, vals, _ = ad.series(q)
            sc = abs(vals[-1])
            row = {"problem": geo, "family": "uniform", "qoi": q, "exact": e, "scale": sc,
                   "finest_true_error": abs(vals[-1] - e) / sc}
            if comb[0] == "PASS" and comb[1] == "rate":
                lim, U = _limit(R["v"][q])
                row["framework"] = {"limit": lim, "U": U, "true_error": abs(e - lim) / sc,
                                    "effectivity": abs(e - lim) / (U * sc),
                                    "covered": abs(e - lim) <= U * sc,
                                    "final_status": final.get((geo, q), "single_family")}
            A, g = conventional(ad, q)
            if A == "PASS":
                row["gci"] = {"limit": g["richardson_extrapolate"], "U": g["gci_fine"],
                              "true_error": abs(e - g["richardson_extrapolate"]) / sc,
                              "effectivity": abs(e - g["richardson_extrapolate"])
                              / (g["gci_fine"] * sc),
                              "covered": abs(e - g["richardson_extrapolate"]) <= g["gci_fine"] * sc}
            rows.append(row)
    # the graded family of the exact benchmark (same five-level check, class tolerances)
    ex = exact["l_shape_exact"]
    lad = _j("lshape_exact_ladder.json")
    gr = sorted((r for r in lad if r["g"] == 3.7), key=lambda r: r["n"])
    for q in ("strain_energy", "u_mid_left", "v_mid_right", "corner_gk_avg_von_mises",
              "corner_disc_avg_von_mises", "far_gk_avg_syy"):
        cp = trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))
        hs, vals = [r["h"] for r in gr], [r[q] for r in gr]
        gm = gates.multilevel_convergence_gate(hs, vals, band_tol=cp["band_tol"],
                                               gci_tol=cp["gci_tol"], order_window=(0.5, 5))
        sc = abs(vals[-1])
        row = {"problem": "l_shape_exact", "family": "graded g = 3.7", "qoi": q,
               "exact": ex[q], "scale": sc, "finest_true_error": abs(vals[-1] - ex[q]) / sc,
               "evidence": gm.get("evidence")}
        if gm.get("evidence") == "rate" and gm.get("verdict") == "PASS":
            lim, U = gm["fit_limit"], gm["uncertainty_fine"]
            row["framework"] = {"limit": lim, "U": U, "true_error": abs(ex[q] - lim) / sc,
                                "effectivity": abs(ex[q] - lim) / (U * sc),
                                "covered": abs(ex[q] - lim) <= U * sc}
        rows.append(row)
    fw = [r["framework"] for r in rows if "framework" in r]
    gc = [r["gci"] for r in rows if "gci" in r]
    summ = {
        "framework_certified": len(fw),
        "framework_covered": sum(f["covered"] for f in fw),
        "framework_effectivity_median": float(np.median([f["effectivity"] for f in fw])),
        "framework_effectivity_max": float(max(f["effectivity"] for f in fw)),
        "gci_certified": len(gc), "gci_covered": sum(f["covered"] for f in gc),
        "gci_effectivity_max": float(max(f["effectivity"] for f in gc)) if gc else None,
        "gci_not_covered": [f"{r['problem']}:{r['qoi']}" for r in rows
                            if "gci" in r and not r["gci"]["covered"]],
    }
    res = {"summary": summ, "rows": rows}
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    return res


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omar_pfem.verification_agent.report_figures import C, _save
    style = {("thick_cylinder_3d", "uniform"): (C["blue"], "o", "Cylinder 3D (smooth), uniform"),
             ("l_shape_exact", "uniform"): (C["orange"], "s", "L-shape (singular), uniform"),
             ("l_shape_exact", "graded g = 3.7"): (C["green"], "^",
                                                    "L-shape (singular), agent-graded")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True, sharey=True)
    lo, hi = 1e-10, 1e-1
    for ax, key, title in ((axes[0], "framework", "(a) Framework (post hoc extension)"),
                           (axes[1], "gci", "(b) Conventional GCI, three levels")):
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="0.93", lw=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color="0.35", lw=1.0, zorder=1)
        ax.text(3e-4, 2e-9, "exact value inside\nthe interval", fontsize=7.5, color="0.35")
        for (geo, fam), (col, mk, lab) in style.items():
            pts = [(r[key]["U"], max(r[key]["true_error"], 2e-10)) for r in res["rows"]
                   if r["problem"] == geo and r["family"] == fam and key in r]
            if pts:
                u, e = zip(*pts)
                ax.scatter(u, e, s=34, marker=mk, color=col, edgecolor="white", lw=0.8,
                           label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("certified uncertainty U (of |value|)")
        ax.grid(True, which="major", color="0.9", lw=0.5)
    axes[0].set_ylabel("true error of the certified limit")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "fig32_1_calibration")
    # remedy: true energy error against unknowns on the exact benchmark
    lad = _j("lshape_exact_ladder.json")
    ex = _j("lshape_exact_values.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, q, title in ((axes[0], "strain_energy", "(a) Strain energy"),
                         (axes[1], "corner_gk_avg_von_mises", "(b) Gaussian von Mises at the corner")):
        for g, col, mk, lab in ((1.0, C["orange"], "s", "uniform"),
                                (3.7, C["green"], "^", "agent-graded, g = 3.7")):
            rs = sorted((r for r in lad if r["g"] == g), key=lambda r: r["n"])
            ax.plot([r["n_dof"] for r in rs], [abs(r[q] - ex[q]) / abs(ex[q]) for r in rs],
                    color=col, marker=mk, ms=5, lw=2, mec="white", mew=0.8, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("unknowns")
        ax.set_title(title)
        ax.grid(True, which="major", color="0.9", lw=0.5)
    axes[0].set_ylabel("true relative error")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig32_2_remedy")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    r = collect()
    print(json.dumps(r["summary"], indent=1))
    for x in r["rows"]:
        f, g = x.get("framework"), x.get("gci")
        print(f"{x['problem']:<18}{x['family']:<15}{x['qoi']:<27}"
              + (f"FW eff {f['effectivity']:.3f} cov {f['covered']} " if f else "FW -            ")
              + (f"GCI eff {g['effectivity']:.3f} cov {g['covered']}" if g else "GCI -"))
    figures(r)
