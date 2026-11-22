"""Figures for the technical report, drawn only from the stored results (no new solves).

    python -m omar_pfem.verification_agent.report_figures

Writes PNG files to advisor_feedback/report_build/figures/verification/:
  fig_b1_convergence.png        B1: errors against the fitted limit; corner peak growth
  fig_singular_unseen.png       L-shape and edge crack: energy error and peak growth vs theory
  fig_b3_gaussian_crosscheck.png  B3 Gaussian sigma_xx: Q20 family, its certified limit +- U, splines
  fig_b3_groove_peak.png        B3 groove peak: displacement vs projected-dilatation splines
  fig_mms.png                   manufactured-solution errors of the new solvers
  fig_crosscheck.png            gap to the independent value against the allowed uncertainty
  fig_active_learning.png       surrogate error against cost for the four strategies
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "advisor_feedback", "report_build", "figures", "verification")

# Okabe-Ito (colour-blind safe)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000", "grey": "#7f7f7f"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})


def _j(name):
    return json.load(open(os.path.join(HERE, name)))


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def slope_triangle(ax, x0, y0, slope, dx=1.6, label=None, below=True, color="0.3"):
    """A log-log slope triangle anchored at (x0, y0), going right by factor dx."""
    x1 = x0 * dx
    y1 = y0 * dx ** slope
    ax.plot([x0, x1, x1 if below else x0, x0], [y0, y1, y0 if below else y1, y0],
            color=color, lw=0.8)
    ax.text(x1 * 1.04 if below else x0 / 1.04, np.sqrt(y0 * y1),
            label or f"{slope:g}", fontsize=7.5, color=color,
            ha="left" if below else "right", va="center")


def plain(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter, LogLocator
    f = FuncFormatter(lambda v, _: f"{v:g}")
    for a in (("x", ax.xaxis), ("y", ax.yaxis)):
        if a[0] in axis:
            a[1].set_major_formatter(f)
            a[1].set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _ml(R, q):
    return R["v"][q].get("qoi_convergence_multilevel", {})


def fig_b1(rules):
    R = rules("unit_square")
    ad = R["adapter"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    names = {"strain_energy": "strain energy", "max_disp_mag": "max |u|",
             "v_top_off_center": "v, top", "u_mid_domain": "u, mid-domain",
             "interior_gk_PK1_frob": "Gaussian interior |P|"}
    cols = [C["blue"], C["orange"], C["green"], C["purple"], C["sky"]]
    for (q, lab), c in zip(names.items(), cols):
        hs, vals, _ = ad.series(q)
        m = _ml(R, q)
        lim, p = m["fit_limit"], m["fit_order"]
        e = np.abs(np.array(vals) - lim) / abs(lim)
        a.loglog(hs, e, "o-", color=c, ms=3.5, lw=1.1, label=f"{lab} (p = {p:.2f})")
    slope_triangle(a, 0.012, 2.2e-5, 2 * 0.7112, label="2λ = 1.42")
    a.set_xlabel("element size h")
    a.set_ylabel("|v(h) − v∞| / |v∞|")
    a.set_title("(a) smooth quantities: error to the fitted limit")
    a.legend(frameon=False, loc="upper left")
    hs, vals, _ = ad.series("peak_PK1_frob")
    b.loglog(hs, vals, "s-", color=C["red"], ms=3.5, lw=1.1, label="corner peak |P|")
    slope_triangle(b, 0.03, vals[-2] * 0.95, 0.7112 - 1, label="λ − 1 = −0.289", below=False)
    b.set_xlabel("element size h")
    b.set_ylabel("peak |P|")
    b.set_title("(b) corner peak grows: no limit exists")
    plain(b, "y")
    b.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_b1_convergence.png")


def fig_singular(rules):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    specs = [("l_shape", "L-shaped panel", 0.5445, C["blue"]),
             ("edge_crack", "edge crack", 0.5, C["red"])]
    for geo, lab, lam, c in specs:
        R = rules(geo)
        ad = R["adapter"]
        hs, vals, _ = ad.series("strain_energy")
        m = _ml(R, "strain_energy")
        e = np.abs(np.array(vals) - m["fit_limit"]) / abs(m["fit_limit"])
        a.loglog(hs, e, "o-", color=c, ms=3.5, lw=1.1,
                 label=f"{lab}: {m['fit_order']:.2f} (2λ = {2 * lam:.2f})")
        hs, vals, _ = ad.series("peak_von_mises")
        sl = np.polyfit(np.log(hs[-4:]), np.log(vals[-4:]), 1)[0]
        b.loglog(hs, vals, "s-", color=c, ms=3.5, lw=1.1,
                 label=f"{lab}: slope {sl:.2f} (theory {lam - 1:.2f})".replace(" -", " −").replace("(-", "(−"))
    a.set_xlabel("element size h")
    a.set_ylabel("|W(h) − W∞| / W∞")
    a.set_title("(a) strain energy, Q9 elements")
    a.legend(frameon=False, loc="upper left", title="observed order", title_fontsize=7.5)
    b.set_xlabel("element size h")
    b.set_ylabel("peak von Mises stress")
    b.set_yticks([5, 10, 20, 40])
    b.set_title("(b) peak at the singular point")
    b.legend(frameon=False, loc="lower left")
    b.set_ylim(3.2, None)
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi * 8)
    plain(b, "y")
    fig.tight_layout()
    _save(fig, "fig_singular_unseen.png")


def fig_b3_gaussian(rules):
    R = rules("bushing_groove")
    ad = R["adapter"]
    q = "gk2_sigma_xx_avg"
    hs, vals, scale = ad.series(q)
    m = _ml(R, q)
    lim, U = m["fit_limit"], m["uncertainty_fine"] * scale
    d = _j("iga_b3_result.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.axhspan(lim - U, lim + U, color=C["blue"], alpha=0.12, lw=0)
    ax.axhline(lim, color=C["blue"], lw=0.8, ls="--")
    ax.text(0.066, lim - 0.0009, f"certified Q20 limit {lim:.3f} ± {U:.4f}",
            color=C["blue"], fontsize=7.5, ha="left", va="top")
    ax.plot(hs, vals, "o-", color=C["blue"], ms=3.5, lw=1.1, label="Q20, displacement (8 levels)")
    for key, lab, c, mk in (("", "splines p = 2, displacement", C["orange"], "^"),
                            (":fbar_p1", "splines p = 2, projected dilatation", C["red"], "s")):
        rows = [d[f"L{lv}:p2{key}"] for lv in (1, 2, 3, 4)]
        h = [r["n_elem"] ** (-1 / 3) for r in rows]
        ax.plot(h, [r["gk2trunc_sigma_xx_avg"] for r in rows], mk + "-", color=c, ms=4,
                lw=1.1, label=lab)
    ax.invert_xaxis()
    ax.set_xlabel(r"$h = N_{\mathrm{el}}^{-1/3}$  ($N_{\mathrm{el}}$: number of elements; finer to the right)")
    ax.set_ylabel("Gaussian average of σxx")
    ax.set_title("B3 groove: the certified Q20 limit and an independent discretization")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_b3_gaussian_crosscheck.png")


def fig_b3_peak():
    d = _j("iga_b3_result.json")
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for key, lab, c, mk in (("", "displacement", C["orange"], "^"),
                            (":fbar_p1", "projected dilatation (P1)", C["red"], "s")):
        rows = [d[f"L{lv}:p2{key}"] for lv in (1, 2, 3, 4)]
        x = [r["n_dof"] for r in rows]
        y = [r["max_sigma_xx_R0"] for r in rows]
        ax.loglog(x, y, mk + "-", color=c, ms=4, lw=1.1, label=f"splines p = 2, {lab}")
        for i, (xi, yi) in enumerate(zip(x, y)):
            off = {(0, ""): (8, -4), (0, ":fbar_p1"): (-34, 2)}.get((i, key), (4, 4))
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=off,
                        fontsize=7, color=c)
    ax.set_xlim(5e3, 6e5)
    plain(ax, "y")
    ax.set_xlabel("unknowns")
    ax.set_ylabel("max σxx at the groove root (R₀)")
    ax.set_title("B3 groove peak: settles only with a projected dilatation (not yet certified)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_b3_groove_peak.png")


def fig_mms():
    rows = _j("mms_new_solvers.json")
    series = [("spline disp", "splines, displacement", C["orange"], "^"),
              ("spline fbar_p1", "splines, projected dilatation", C["red"], "s"),
              ("Q20 disp (mixed_b3), cartesian field", "Q20, displacement", C["blue"], "o"),
              ("Q20/P1 (mixed_b3), cartesian field", "Q20/P1", C["green"], "D")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for name, lab, c, mk in series:
        rr = sorted((r for r in rows if r["solver"] == name), key=lambda r: -r["h"])
        h = [r["h"] for r in rr]
        a.loglog(h, [r["L2_rel"] for r in rr], mk + "-", color=c, ms=3.5, lw=1.1, label=lab)
        b.loglog(h, [r["H1_rel"] for r in rr], mk + "-", color=c, ms=3.5, lw=1.1, label=lab)
    slope_triangle(a, 0.03, 3e-5, 3, dx=1.5)
    slope_triangle(b, 0.03, 1.2e-3, 2, dx=1.5)
    for ax, t in ((a, "(a) relative L2 error"), (b, "(b) relative H1 error")):
        ax.set_xlabel("h")
        ax.set_title(t)
    a.set_ylabel("error against the manufactured solution")
    a.legend(frameon=False, loc="upper left")
    b.text(0.05, 0.97, "spline curves coincide", transform=b.transAxes, fontsize=7, color="0.4",
           va="top")
    plain(a, "x")
    plain(b, "xy")
    for ax in (a, b):
        ax.set_xticks([0.03, 0.04, 0.06, 0.08])
    b.set_yticks([0.003, 0.01, 0.03])
    fig.tight_layout()
    _save(fig, "fig_mms.png")


def fig_crosscheck():
    d = _j("crosscheck_result.json")
    rows = [r for r in d["rows"] if "gap_rel" in r]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    allow = np.array([r["U_rel"] + r["U_ind_rel"] for r in rows])
    gap = np.maximum(np.array([r["gap_rel"] for r in rows]), 1e-8)
    bad = np.array([r["status"] == "contradicted" for r in rows])
    lo, hi = 1e-7, 1e-2
    ax.loglog([lo, hi], [lo, hi], color="0.5", lw=0.8)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=C["red"], alpha=0.06, lw=0)
    ax.text(2e-7, 3e-3, "contradicted:\ngap larger than allowed", fontsize=7.5, color=C["red"])
    ax.loglog(allow[~bad], gap[~bad], "o", color=C["green"], ms=4,
              label=f"cross-checked ({(~bad).sum()})")
    ax.loglog(allow[bad], gap[bad], "o", color=C["red"], ms=5,
              label=f"contradicted ({bad.sum()})")
    b3 = [i for i, r in enumerate(rows) if r["status"] == "contradicted"
          and r["problem"] == "bushing_groove"]
    lam = [i for i, r in enumerate(rows) if r["status"] == "contradicted"
           and r["problem"] != "bushing_groove"]
    ax.annotate(f"B3 Gaussian groove averages ({len(b3)})", (allow[b3].min(), gap[b3].mean()),
                textcoords="offset points", xytext=(-8, 0), fontsize=7, ha="right",
                va="center", color=C["red"])
    for i in lam:
        ax.annotate("lamina sharp hole average", (allow[i], gap[i]), textcoords="offset points",
                    xytext=(-8, 0), fontsize=7, ha="right", va="center", color=C["red"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("allowed: certified U + independent U (relative to scale)")
    ax.set_ylabel("gap: |certified limit − independent value|")
    ax.set_title(f"Cross-check of {len(rows)} quantities with an independent value")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()
    _save(fig, "fig_crosscheck.png")


def fig_al():
    d = _j("al_lamina_result.json")["strategies"]
    labs = {"random": ("random sampling, fine mesh", C["grey"]),
            "al_fine": ("active learning, fine mesh", C["blue"]),
            "mf_naive": ("multi-fidelity, unverified levels", C["green"]),
            "mf_verified": ("multi-fidelity, verified levels", C["red"])}
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for k, (lab, c) in labs.items():
        for i, hist in enumerate(d[k]["history"]):
            ax.loglog([h["cost_s"] for h in hist], [h["rmse_rel"] for h in hist], "-", color=c,
                      lw=1.0, alpha=0.75, label=lab if i == 0 else None)
    for t in (1e-2, 1e-3):
        ax.axhline(t, color="0.6", lw=0.7, ls=":")
        ax.text(4.0, t * 1.12, f"{100 * t:g}% target", fontsize=7, color="0.4")
    ax.set_xlabel("cumulative solver cost (s)")
    ax.set_ylabel("surrogate error (relative RMSE)")
    ax.set_title("CFM lamina surrogate: error against cost (five seeds per strategy)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig_active_learning.png")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent.verify_report import rules
    r = rules.__wrapped__
    fig_b1(r)
    fig_singular(r)
    fig_b3_gaussian(r)
    fig_b3_peak()
    fig_mms()
    fig_crosscheck()
    fig_al()


if __name__ == "__main__":
    main()
