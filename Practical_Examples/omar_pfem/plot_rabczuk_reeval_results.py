"""Generates every figure that can reasonably be drawn from the raw JSON
output of run_b3_rabczuk_reeval_v2_local.py and
evaluate_B1_pi_vs_dd_fine_reference.py -- Omar's own explicit request
("كل الصور الممكن تطلع حطها خليه يطلعها كلشي ممكن", 2026-11-02): every
possible figure, not a curated subset.

Colors follow this account's own validated data-viz palette (fixed
categorical hue order, single-hue sequential ramp for heatmaps -- see
the dataviz skill's references/palette.md) rather than matplotlib's
defaults, so FEM/PI/DD (or 6,840/43,400-element) keep the SAME color
across every figure in this report.

Either --b3_json or --b1_json may be omitted if that job isn't done yet
-- this script draws whichever one(s) it's given, exactly like
summarize_rabczuk_reeval_results.py already does for the Markdown
tables.

Usage:
  python -m omar_pfem.plot_rabczuk_reeval_results \\
      --b3_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \\
      --b1_json /scratch/gole3157/pfem_run/b1_pi_vs_dd_fine_reference.json \\
      --out_dir /scratch/gole3157/pfem_run/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Fixed categorical order (never cycled/reassigned per-chart) -- this
# account's own validated palette, references/palette.md.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
# Sequential (magnitude) ramp, light -> dark, for heatmaps.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

STRESS_COMPONENTS = ["xx", "yy", "zz", "xy", "xz", "yz"]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _pct(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v * 100:.1f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def _grouped_bar(ax, categories, series, colors, ylabel, title, pct=True):
    """series: dict[label] -> (means array, stds array or None), one entry
    per bar-group member (e.g. FEM(N)/PI/DD, or 6,840-el/43,400-el)."""
    n_groups = len(categories)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    for i, (label, (means, stds)) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.92, label=label, color=colors[i],
               yerr=stds, capsize=3, error_kw={"ecolor": INK_SECONDARY, "linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    if pct:
        _pct(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_axes(ax)


# ============================================================================
# B1: PI vs DD vs FEM(N), each vs the fine_N reference
# ============================================================================

B1_METRICS = [
    ("l2_rel", "Displacement relative L2 error"),
    ("h1_semi_rel", "H1 semi-norm relative error"),
    ("energy_rel", "Tangent energy relative error"),
    ("reaction_resultant_rel_err", "Reaction resultant relative error"),
    ("cauchy_avg_rel_err", "Region Cauchy stress, mean relative error"),
    ("cauchy_p99_rel_err", "Region Cauchy stress, p99 relative error"),
    ("cauchy_max_rel_err", "Region Cauchy stress, max relative error"),
]


def plot_b1(b1_json_path, out_dir):
    with open(b1_json_path) as f:
        data = json.load(f)
    resolutions = data["resolutions"]
    rows = data["rows"]
    fine_N = data["fine_N"]

    for metric_key, metric_label in B1_METRICS:
        fem_mean, fem_std, pi_mean, pi_std, dd_mean, dd_std = [], [], [], [], [], []
        for N in resolutions:
            rows_n = [r for r in rows if r["N"] == N]
            if not rows_n:
                fem_mean.append(np.nan); fem_std.append(0)
                pi_mean.append(np.nan); pi_std.append(0)
                dd_mean.append(np.nan); dd_std.append(0)
                continue
            fem_vals = [r["fem_vs_fine"][metric_key] for r in rows_n]
            pi_vals = [r["pi_vs_fine"][metric_key] for r in rows_n]
            dd_vals = [r["dd_vs_fine"][metric_key] for r in rows_n]
            fem_mean.append(np.mean(fem_vals)); fem_std.append(np.std(fem_vals))
            pi_mean.append(np.mean(pi_vals)); pi_std.append(np.std(pi_vals))
            dd_mean.append(np.mean(dd_vals)); dd_std.append(np.std(dd_vals))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        series = {
            f"FEM(N)": (np.array(fem_mean), np.array(fem_std)),
            "PI": (np.array(pi_mean), np.array(pi_std)),
            "DD": (np.array(dd_mean), np.array(dd_std)),
        }
        _grouped_bar(ax, [f"N={N}" for N in resolutions], series, [BLUE, ORANGE, AQUA],
                     "Relative error", f"B1 -- {metric_label}\nvs. fine FEM reference (N={fine_N})")
        _save(fig, out_dir, f"b1_{metric_key}.png")

    # One combined overview: disp-L2 and region Cauchy mean, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (metric_key, metric_label) in zip(axes, [B1_METRICS[0], B1_METRICS[4]]):
        fem_mean, pi_mean, dd_mean = [], [], []
        for N in resolutions:
            rows_n = [r for r in rows if r["N"] == N]
            fem_mean.append(np.mean([r["fem_vs_fine"][metric_key] for r in rows_n]))
            pi_mean.append(np.mean([r["pi_vs_fine"][metric_key] for r in rows_n]))
            dd_mean.append(np.mean([r["dd_vs_fine"][metric_key] for r in rows_n]))
        series = {"FEM(N)": (np.array(fem_mean), None), "PI": (np.array(pi_mean), None),
                  "DD": (np.array(dd_mean), None)}
        _grouped_bar(ax, [f"N={N}" for N in resolutions], series, [BLUE, ORANGE, AQUA],
                     "Relative error", metric_label)
    fig.suptitle(f"B1 overview -- vs. fine FEM reference (N={fine_N})", color=INK_PRIMARY, fontsize=12)
    _save(fig, out_dir, "b1_overview.png")


# ============================================================================
# B3: every checkpoint vs. the single 950,400-element reference
# ============================================================================

B3_U_ONLY_METRICS = [
    ("mean_disp_rel_l2_fine_combined", "Displacement relative L2"),
    ("mean_H1_seminorm_rel_fine", "H1 semi-norm relative error"),
    ("mean_rel_err_energy_fine", "Energy relative error"),
    ("mean_rel_err_reaction_moment_y_fine", "Reaction moment M_y relative error"),
    ("mean_region_cauchy_field_rel_fine", "Region Cauchy stress field relative error"),
]


def _is_u_only(r):
    return "mean_region_cauchy_field_rel_fine" in r["summary"]


def plot_b3(b3_json_path, out_dir):
    with open(b3_json_path) as f:
        data = json.load(f)
    checkpoints = data["checkpoints"]
    u_only = {name: r for name, r in checkpoints.items() if _is_u_only(r)}
    mixed = {name: r for name, r in checkpoints.items() if not _is_u_only(r)}

    # --- one horizontal bar chart per QoI, across every u-only checkpoint ---
    for metric_key, metric_label in B3_U_ONLY_METRICS:
        names = list(u_only.keys())
        vals = [u_only[n]["summary"].get(metric_key, np.nan) for n in names]
        order = np.argsort(vals)
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
        y = np.arange(len(names))
        ax.barh(y, vals, color=BLUE, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Relative error")
        ax.set_title(f"B3 -- {metric_label} (every checkpoint vs. 950,400-el reference)",
                     fontsize=11, loc="left")
        _pct(ax, axis="x")
        _style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        _save(fig, out_dir, f"b3_u_only_{metric_key}.png")

    # --- heatmap: u-only checkpoints x stress components ---
    if u_only:
        names = list(u_only.keys())
        mat = np.array([[u_only[n]["summary"].get(f"mean_region_sigma_{c}_rel_fine", np.nan)
                          for c in STRESS_COMPONENTS] for n in names])
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))
        im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto")
        ax.set_xticks(range(len(STRESS_COMPONENTS)))
        ax.set_xticklabels([f"$\\sigma_{{{c}}}$" for c in STRESS_COMPONENTS])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                text_color = "white" if v > np.nanmax(mat) * 0.6 else INK_PRIMARY
                ax.text(j, i, f"{v * 100:.1f}%", ha="center", va="center",
                         fontsize=8, color=text_color)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        cbar.ax.tick_params(colors=INK_SECONDARY)
        ax.set_title("B3 -- per-component stress relative error, by checkpoint",
                      fontsize=11, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
        fig.set_facecolor(SURFACE)
        _save(fig, out_dir, "b3_u_only_stress_components_heatmap.png")

    # --- mixed-stress checkpoints: implicit vs direct ---
    if mixed:
        names = list(mixed.keys())
        h1 = [mixed[n]["summary"]["mean_H1_seminorm_rel_fine"] for n in names]
        implicit = [mixed[n]["summary"]["mean_region_cauchy_field_rel_implicit_fine"] for n in names]
        direct = [mixed[n]["summary"]["mean_region_cauchy_field_rel_direct_fine"] for n in names]

        fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
        series = {"implicit": (np.array(implicit), None), "direct": (np.array(direct), None)}
        y = np.arange(len(names))
        width = 0.35
        for i, (label, (means, _)) in enumerate(series.items()):
            offset = (i - 0.5) * width
            ax.barh(y + offset, means, width * 0.9, label=label,
                    color=[ORANGE, VIOLET][i])
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Region Cauchy stress field relative error")
        ax.set_title("B3 -- mixed-stress checkpoints, implicit vs. direct stress path",
                      fontsize=11, loc="left")
        _pct(ax, axis="x")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
        _style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        _save(fig, out_dir, "b3_mixed_stress_implicit_vs_direct.png")

        # per-component heatmap for mixed-stress, direct path (the physically
        # meaningful one -- see this project's own convention elsewhere)
        mat = np.array([[mixed[n]["summary"].get(f"mean_region_sigma_{c}_rel_direct_fine", np.nan)
                          for c in STRESS_COMPONENTS] for n in names])
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
        im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto")
        ax.set_xticks(range(len(STRESS_COMPONENTS)))
        ax.set_xticklabels([f"$\\sigma_{{{c}}}$" for c in STRESS_COMPONENTS])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                text_color = "white" if v > np.nanmax(mat) * 0.6 else INK_PRIMARY
                ax.text(j, i, f"{v * 100:.1f}%", ha="center", va="center",
                         fontsize=8, color=text_color)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        cbar.ax.tick_params(colors=INK_SECONDARY)
        ax.set_title("B3 -- mixed-stress, per-component error (direct path)", fontsize=11, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
        fig.set_facecolor(SURFACE)
        _save(fig, out_dir, "b3_mixed_stress_components_heatmap.png")

    # --- FEM-vs-FEM discretization error (no neural network) ---
    disc = data.get("fem_discretization_error")
    if disc:
        metrics = [
            ("mean_disp_rel_l2_vs_950400", "Displacement relative L2"),
            ("mean_H1_seminorm_rel_vs_950400", "H1 semi-norm relative error"),
            ("mean_energy_rel_vs_950400", "Energy relative error"),
            ("mean_reaction_moment_y_rel_vs_950400", "Reaction moment M_y relative error"),
            ("mean_region_cauchy_field_rel_vs_950400", "Region Cauchy stress field relative error"),
        ]
        labels_short = ["disp L2", "H1", "energy", "reaction M_y", "region sigma"]
        v6840 = [disc["vs_6840"]["summary"][k] for k, _ in metrics]
        v43400 = [disc["vs_43400"]["summary"][k] for k, _ in metrics]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        series = {"6,840-element (training mesh)": (np.array(v6840), None),
                  "43,400-element (accuracy-matched)": (np.array(v43400), None)}
        _grouped_bar(ax, labels_short, series, [GREEN, VIOLET], "Relative error",
                     "B3 -- pure FEM discretization error vs. 950,400-el reference\n"
                     "(no neural network involved)")
        _save(fig, out_dir, "b3_fem_discretization_error.png")

        # --- overlay: best u-only checkpoint vs. the two discretization errors ---
        if u_only:
            best_name = min(u_only, key=lambda n: u_only[n]["summary"].get(
                "mean_region_cauchy_field_rel_fine", np.inf))
            best = u_only[best_name]["summary"]
            v_best = [
                best.get("mean_disp_rel_l2_fine_combined", np.nan),
                best.get("mean_H1_seminorm_rel_fine", np.nan),
                best.get("mean_rel_err_energy_fine", np.nan),
                best.get("mean_rel_err_reaction_moment_y_fine", np.nan),
                best.get("mean_region_cauchy_field_rel_fine", np.nan),
            ]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            series = {
                f"Best NN checkpoint ({best_name})": (np.array(v_best), None),
                "6,840-element FEM (training mesh)": (np.array(v6840), None),
                "43,400-element FEM (accuracy-matched)": (np.array(v43400), None),
            }
            _grouped_bar(ax, labels_short, series, [BLUE, GREEN, VIOLET], "Relative error",
                         "B3 -- is the neural operator as accurate as a finer mesh?\n"
                         "best checkpoint vs. pure FEM discretization error")
            _save(fig, out_dir, "b3_nn_vs_discretization_error.png")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--b3_json", type=str, default=None)
    parser.add_argument("--b1_json", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    if not args.b3_json and not args.b1_json:
        raise SystemExit("give at least one of --b3_json / --b1_json")

    if args.b1_json:
        print("B1 figures:")
        plot_b1(args.b1_json, args.out_dir)
    if args.b3_json:
        print("B3 figures:")
        plot_b3(args.b3_json, args.out_dir)

    print(f"\nAll figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
