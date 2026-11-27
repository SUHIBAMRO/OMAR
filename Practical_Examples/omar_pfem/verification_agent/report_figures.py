"""Figures of the combined progress report (24-26 November): every figure is drawn from the
stored files only.

    cd Practical_Examples && python -m omar_pfem.verification_agent.report_figures

Writes advisor_feedback/figures_2026-11-26/fig*.png.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "advisor_feedback", "figures_2026-11-26"))
# categorical slots 1-4 (validated: CVD separation and normal-vision floor pass; contrast below
# 3:1 for slots 2-4, so every series also has its own marker and a legend or direct label)
C1, C2, C3, C4 = "#257de0", "#f47230", "#00b894", "#f2a600"
INK, INK2, GRID = "#1f1f1f", "#5f5f5f", "#e3e3e3"
EL_STYLE = {"Q2": dict(color=C1, marker="o"), "P2": dict(color=C2, marker="s")}


def _j(*p):
    return json.load(open(os.path.join(HERE, *p)))


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "lines.markersize": 6, "legend.frameon": False, "savefig.dpi": 200,
        "figure.dpi": 100})
    return plt


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


# ---------------------------------------------------------------- 1 workflow -------------
def fig_workflow():
    plt = _style()
    from matplotlib.patches import FancyBboxPatch
    stages = [("L", "Literature\nsearch", "LLM + code"), ("D", "Problem\ndefinition", "LLM + code"),
              ("M", "Meshing", "code"), ("V", "Code check\n(MMS)", "code"),
              ("S", "Simulation\nladders", "code"), ("C", "Convergence\nanalysis", "code"),
              ("G", "Diagnosis", "code"), ("N", "Next\nexperiment", "code"),
              ("A", "Input\nsensitivity", "code"), ("R", "Run\nreport", "code")]
    col = {"code": C1, "LLM + code": C3}
    fig, ax = plt.subplots(figsize=(10, 2.9))
    w, h, gap = 0.96, 0.95, 0.08
    for i, (k, name, actor) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=col[actor], lw=2))
        ax.text(x + w / 2, h * 0.70, k, ha="center", va="center", fontsize=11, weight="bold",
                color=INK)
        ax.text(x + w / 2, h * 0.33, name, ha="center", va="center", fontsize=7, color=INK)
        if i:
            ax.annotate("", (x - 0.01, h / 2), (x - gap + 0.01, h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    # loop back from N to S, and the cluster branch
    xs, xn = 4 * (w + gap) + w / 2, 7 * (w + gap) + w / 2
    ax.annotate("", (xs, -0.04), (xn, -0.04), arrowprops=dict(
        arrowstyle="-|>", color=INK2, lw=1, connectionstyle="arc3,rad=-0.25"))
    ax.text((xs + xn) / 2, -0.62, "frozen prediction -> run -> scored by code -> next choice",
            ha="center", fontsize=7.5, color=INK2)
    ax.annotate("", (xs, h + 0.42), (xs, h + 0.05), arrowprops=dict(arrowstyle="-|>", color=C2,
                                                                     lw=1.5))
    ax.text(xs, h + 0.5, "level too large for the local machine:\ncluster job prepared by code, "
            "submitted by a person", ha="center", va="bottom", fontsize=7.5, color=INK)
    for i, (lab, c) in enumerate((("code only", C1), ("LLM proposes, code validates", C3),
                                  ("person", C2))):
        ax.plot([0.05 + 2.6 * i], [-1.05], "s", ms=9, mfc="white", mec=c, mew=2)
        ax.text(0.22 + 2.6 * i, -1.05, lab, va="center", fontsize=7.5, color=INK)
    ax.set_xlim(-0.1, 10 * (w + gap))
    ax.set_ylim(-1.3, h + 1.15)
    ax.axis("off")
    return _save(fig, "fig1_workflow.png")


# ---------------------------------------------------------------- 2 meshes ---------------
def _mesh_ax(ax, desc, n, title):
    from matplotlib.collections import PolyCollection
    from omar_pfem.verification_agent.geometry2d import make
    msh, _ = make(desc).build_mesh(n, None, "Q2")
    k = 4 if "Quad" in type(msh).__name__ else 3
    polys = [msh.p[:, msh.t[:k, i]].T for i in range(msh.t.shape[1])]
    ax.add_collection(PolyCollection(polys, facecolors="#f4f8fd", edgecolors=INK2, lw=0.35))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(f"{title}\nQ2 mesh, level n = {n} ({msh.t.shape[1]} elements)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    return msh


def fig_meshes():
    plt = _style()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    _mesh_ax(ax[0], _j("problems2d", "cook_linear.json"), 8, "Cook's membrane")
    ax[0].plot([48], [52], "o", color=C2, ms=6)
    ax[0].annotate("C (48, 52): v_C", (48, 52), xytext=(-70, 8), textcoords="offset points",
                   fontsize=7.5, color=INK)
    _mesh_ax(ax[1], _j("problems2d", "plate_hole_quarter.json"), 8,
             "Quarter plate with a hole (10 x 10, r = 1)")
    return _save(fig, "fig2_meshes.png")


# ---------------------------------------------------------------- 3 Cook v_C -------------
def fig_cook():
    plt = _style()
    lad = _j("pipeline2d_runs", "cook_linear", "ladders.json")
    ci = _j("autonomous_runs", "cook_linear", "run.json")["certified_info"]["v_C"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lim, ua = ci["limit"], ci["U_abs"]
    ax.axhspan(lim - ua, lim + ua, color=C3, alpha=0.18, lw=0)
    ax.axhline(23.96, color=INK2, lw=1, ls=":")
    ax.text(70, 23.96 - 0.012, "reference 23.96", fontsize=7.5, color=INK2, va="top")
    for key, c, m, ls in (("Q2 uniform", C1, "o", "-"), ("P2 uniform", C2, "s", "-"),
                          ("Q2 graded", C1, "^", "--"), ("P2 graded", C2, "v", "--")):
        rows = sorted(lad[key], key=lambda r: r["n"])
        ax.semilogx([r["n_dof"] for r in rows], [r["v_C"] for r in rows], ls, color=c,
                    marker=m, label=key)
    ax.text(3.3e5, lim + ua + 0.003, f"certified {lim:.4f} ± {ua:.4f}", fontsize=7.5,
            color=INK, ha="right", va="bottom")
    ax.set_ylim(23.80, 24.0)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("v_C")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    return _save(fig, "fig3_cook_vC.png")


# ---------------------------------------------------------------- 4 sensitivity ----------
def fig_sensitivity():
    """Rows: certified quantities; colour: the perturbed input (same colour in both panels);
    black tick: the quantity's certified relative uncertainty U; dashed: the 1% target."""
    plt = _style()
    inp = {"E": ("E ±10%", C1, "o"), "nu": ("ν ±0.05", C2, "s"), "load": ("load ±10%", C3, "^"),
           "hole[0].radius": ("hole radius ±10%", C4, "D")}
    floor = 1e-10
    panels = (("plate_hole_quarter", "Quarter plate with a hole"), ("cook_linear", "Cook's membrane"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharex=True)
    for ax, (prob, title) in zip(axes, panels):
        rec = _j("sensitivity_runs", prob + ".json")["params"]
        qs = list(next(v for v in rec.values() if "quantities" in v)["quantities"])
        for i, q in enumerate(qs):
            for k, r in rec.items():
                if "quantities" not in r:
                    continue
                lab, c, m = inp["load" if ".t" in k else k]
                ch = r["quantities"][q]["change_rel"]
                off = {"E": -0.18, "nu": -0.06, "load": 0.06}.get("load" if ".t" in k else k, 0.18)
                ax.semilogx([max(ch, floor)], [i + off], m, color=c, ms=7, mec="white", mew=0.6,
                            mfc=c if ch >= floor else "white", label=lab)
            u = r["quantities"][q].get("U_rel")
            ax.plot([u, u], [i - 0.32, i + 0.32], color=INK, lw=2, solid_capstyle="butt",
                    label="certified uncertainty U")
        ax.axvline(0.01, color=INK2, lw=1, ls="--")
        ax.set_yticks(range(len(qs)), [q.replace("_gk_avg_von_mises", " average") for q in qs],
                      fontsize=8)
        ax.set_ylim(-0.6, len(qs) - 0.4)
        ax.set_xlim(floor / 2, 1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("relative change of the quantity")
        ax.text(0.0115, len(qs) - 0.5, "1%", fontsize=7.5, color=INK2, va="top")
    h, lab = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.15, f"a change below {floor:.0e} (none in theory) is drawn hollow at the left edge",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout()
    return _save(fig, "fig4_sensitivity.png")


# ---------------------------------------------------------------- 5 3D mesh --------------
def fig_mesh3d():
    plt = _style()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from omar_pfem.verification_agent.geometry2d import make
    fig = plt.figure(figsize=(9.4, 4.0))
    for i, (f, n, title) in enumerate((("cook_3d.json", 4, "Cook's membrane, extruded"),
                                       ("plate_hole_quarter_3d_pe.json", 4,
                                        "Quarter plate with a hole, extruded"))):
        prob = make(_j("problems3d", f))
        msh, _ = prob.build_mesh(n, None, "Q2")
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # boundary faces of the hexahedra (corner nodes of each quadrilateral face)
        fac = msh.facets[:4, msh.boundary_facets()] if msh.facets.shape[0] >= 4 else None
        polys = [msh.p[:, fac[:, k]].T for k in range(fac.shape[1])]
        polys = [p[[0, 1, 3, 2]] if _crossed(p) else p for p in polys]
        ax.add_collection3d(Poly3DCollection(polys, facecolor="#dbe9fa", edgecolor=INK2,
                                             linewidths=0.25, alpha=1.0))
        lo, hi = msh.p.min(axis=1), msh.p.max(axis=1)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=28, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"{title}\n27-node hexahedra, n = {n} ({msh.t.shape[1]} elements)",
                     fontsize=8.5)
    return _save(fig, "fig5_mesh3d.png")


def _crossed(p):
    """A quadrilateral given in tensor order (0,1,2,3 = bottom-left, bottom-right, top-left,
    top-right) is drawn as a bow-tie; detect it by the diagonals' direction."""
    a, b = p[1] - p[0], p[3] - p[2]
    return np.dot(a, b) > 0 and np.dot(p[2] - p[0], p[3] - p[1]) > 0 and \
        np.linalg.norm(p[0] + p[3] - p[1] - p[2]) < np.linalg.norm(p[0] + p[2] - p[1] - p[3])


# ---------------------------------------------------------------- 6 code checks ----------
def fig_mms():
    plt = _style()
    rec = _j("code_verification_record.json")

    def best(problem, el, mat=None):
        cand = [v for k, v in rec.items() if v.get("problem") == problem and
                k.split("|")[1] == el and (mat is None or k.endswith(mat))]
        return max(cand, key=lambda v: (len(v["levels"]), "computed on the cluster" in
                                         v.get("levels_added", [])))
    panels = [("3D plate, plane strain", "plate_hole_quarter_3d_pe", None),
              ("3D neo-Hookean plate", "nh_plate_hole_3d_pe", "neo_hookean"),
              ("Cook 3D (n = 32 on the cluster)", "cook_3d", None)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=False)
    for ax, (title, prob, mat) in zip(axes, panels):
        verdicts = []
        for el in ("Q2", "P2"):
            try:
                r = best(prob, el, mat)
            except ValueError:
                continue
            n = np.array(r["levels"], float)
            st = EL_STYLE[el]
            for norm, ls in (("L2", "-"), ("H1", "--")):
                e = np.array(r["errors"][norm], float)
                ax.loglog(n, e, ls, color=st["color"], marker=st["marker"], ms=5,
                          label=f"{el} {norm}")
            verdicts.append(f"{el}: {r['verdict']}")
        _slope(ax, 3)
        _slope(ax, 2, low=True)
        ax.set_title(title + "\n" + "; ".join(verdicts), fontsize=8.5)
        ax.set_xlabel("level n (h ~ 1/n)")
        ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        ax.minorticks_off()
    axes[0].set_ylabel("relative error of the manufactured solution")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    return _save(fig, "fig6_mms.png")


def _slope(ax, p, low=False):
    ys = np.concatenate([l.get_ydata() for l in ax.get_lines()])
    m = ys.min()
    y0 = m * (2.5 if low else 40)
    x0, x1 = 2.0, 4.0
    y1 = y0 * (x1 / x0) ** (-p)
    ax.plot([x0, x1, x0, x0], [y0, y1, y1, y0], color=INK2, lw=0.8)
    ax.text(x1 * 1.06, y1, f"slope {p}", fontsize=7, color=INK2, va="center", ha="left")


# ---------------------------------------------------------------- 7 groove peak ----------
def fig_groove():
    plt = _style()
    d, n = _j("iga_b3_result.json"), _j("iga_b3_l6_result.json")
    row = lambda lv, f: (d.get(f"L{lv}:p2{f}") or n.get(f"L{lv}:p2{f}"))
    lv = [2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for f, lab, c, m in ((":fbar_p1", "splines with projected dilatation (P1)", C1, "o"),
                         ("", "splines, displacement only", C2, "s")):
        y = [row(l, f)["max_sigma_xx_R0"] for l in lv]
        ax.plot(lv, y, "-", color=c, marker=m, label=lab)
        ax.annotate(f"{y[-1]:.2f}", (lv[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=INK, va="center")
    ax.axvspan(5.5, 6.5, color=C4, alpha=0.15, lw=0)
    ax.text(6, ax.get_ylim()[1] * 0.97, "level 6:\ncluster", ha="center", va="top", fontsize=7.5,
            color=INK2)
    ax.set_xticks(lv)
    ax.set_xlabel("spline level")
    ax.set_ylabel("maximum of σxx within R₀")
    ax.legend(fontsize=7.5)
    return _save(fig, "fig7_groove.png")


# ---------------------------------------------------------------- 8 plate cross-check ----
def fig_plate_crosscheck():
    plt = _style()
    lad = _j("pipeline2d_runs", "plate_hole_quarter_3d_pe", "ladders.json")
    v = _j("autonomous_runs", "plate_hole_quarter_3d_pe", "run.json")["verdicts"]["v_hole_top"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for key, el in (("Q2 uniform", "Q2"), ("P2 uniform", "P2")):
        rows = sorted(lad[key], key=lambda r: r["n"])[1:]
        ax.semilogx([r["n_dof"] for r in rows], [r["v_hole_top"] for r in rows], "-",
                    label=f"{el} ({'27-node hexahedra' if el == 'Q2' else '10-node tetrahedra'})",
                    **EL_STYLE[el])
    p32 = max(lad["P2 uniform"], key=lambda r: r["n"])
    ax.annotate("P2, n = 32\n(cluster)", (p32["n_dof"], p32["v_hole_top"]), xytext=(-20, -34),
                textcoords="offset points", fontsize=7.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ua = abs(v["U"] * v["limit"])
    ax.axhspan(v["limit"] - ua, v["limit"] + ua, color=C3, alpha=0.25, lw=0)
    ax.text(3e3, v["limit"] + ua, f"Q2 limit {v['limit']:.5f}", fontsize=7.5, color=INK,
            va="bottom")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("v_hole_top")
    ax.legend(fontsize=7.5, loc="lower right")
    return _save(fig, "fig8_plate_crosscheck.png")


# ---------------------------------------------------------------- 9 re-entrant corner ---
def fig_reentrant():
    plt = _style()
    res = _j("reentrant_result.json")
    pre = _j("reentrant_preregistration.json")["predictions"]
    sty = {"U1": (C1, "o", "--", "uniform h, Q1"), "U2": (C1, "s", "-", "uniform h, Q2"),
           "G1_2": (C3, "o", "--", "graded g = 2, Q1"), "G2_1.5": (C4, "s", "-", "graded g = 1.5, Q2"),
           "G2_4": (C3, "s", "-", "graded g = 4, Q2"), "P": (C2, "^", "--", "p-version (n = 2), p = 1..8"),
           "HP": (C2, "D", "-", "hp, geometric mesh, p = m = 1..6")}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, (c, m, ls, lab) in sty.items():
        rows = sorted(res[k], key=lambda r: r["n_dof"])
        ax.loglog([r["n_dof"] for r in rows], [r["energy_error_rel"] for r in rows], ls, color=c,
                  marker=m, ms=5, label=lab)
    ext = sorted(res.get("HP_posthoc_extension", []), key=lambda r: r["n_dof"])
    if ext:
        last = max(res["HP"], key=lambda r: r["n_dof"])
        ax.loglog([last["n_dof"]] + [r["n_dof"] for r in ext],
                  [last["energy_error_rel"]] + [r["energy_error_rel"] for r in ext], ":",
                  color=C2, marker="D", mfc="white", ms=5, label="hp, p = 7, 8 (added after scoring)")
    ax.set_xlabel("degrees of freedom N")
    ax.set_ylabel("relative energy-norm error")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.text(2.2e4, 9e-2, f"uniform: N^-{pre['R1_uniform']['U1']:.3f} (lam/2)", fontsize=7, color=INK2)
    return _save(fig, "fig9_reentrant.png")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    for f in (fig_workflow, fig_meshes, fig_cook, fig_sensitivity, fig_mesh3d, fig_mms,
              fig_groove, fig_plate_crosscheck, fig_reentrant):
        try:
            print("ok  ", f())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("FAIL", f.__name__, e)


if __name__ == "__main__":
    main()
