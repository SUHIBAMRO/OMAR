"""Turns the raw JSON output of run_b3_rabczuk_reeval_v2_local.py and
evaluate_B1_pi_vs_dd_fine_reference.py into clean Markdown tables ready
to paste into the advisor report or an email to Professor Rabczuk --
built while the real HPC batch jobs (3112377/3112378) were still
queued, so the summary is ready the moment they finish (no need to
hand-read the raw JSON under time pressure).

Usage:
  python -m omar_pfem.summarize_rabczuk_reeval_results \\
      --b3_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \\
      --b1_json /scratch/gole3157/pfem_run/b1_pi_vs_dd_fine_reference.json \\
      --out_md /scratch/gole3157/pfem_run/rabczuk_reeval_summary.md

Either --b3_json or --b1_json may be omitted if that job isn't done yet
-- the script summarizes whichever one(s) it's given.
"""
import argparse
import json


def summarize_b3(b3_json_path):
    with open(b3_json_path) as f:
        data = json.load(f)

    lines = ["## B3 — every checkpoint vs. the single 950,400-element reference\n"]
    lines.append("| Checkpoint | disp L2 | H1 | energy | reaction M_y | region σ (field) | "
                  "σxx | σyy | σzz | σxy | σxz | σyz |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in data["checkpoints"].items():
        s = r["summary"]
        if "mean_region_cauchy_field_rel_fine" in s:  # u-only checkpoint
            lines.append(
                f"| {name} | {s['mean_disp_rel_l2_fine_combined']:.4f} | "
                f"{s['mean_H1_seminorm_rel_fine']:.4f} | {s['mean_rel_err_energy_fine']:.4f} | "
                f"{s['mean_rel_err_reaction_moment_y_fine']:.4f} | "
                f"{s['mean_region_cauchy_field_rel_fine']:.4f} | "
                f"{s['mean_region_sigma_xx_rel_fine']:.4f} | {s['mean_region_sigma_yy_rel_fine']:.4f} | "
                f"{s['mean_region_sigma_zz_rel_fine']:.4f} | {s['mean_region_sigma_xy_rel_fine']:.4f} | "
                f"{s['mean_region_sigma_xz_rel_fine']:.4f} | {s['mean_region_sigma_yz_rel_fine']:.4f} |")
        else:  # mixed-stress checkpoint: one row each for implicit and direct
            for path in ("implicit", "direct"):
                lines.append(
                    f"| {name} ({path}) | — | {s['mean_H1_seminorm_rel_fine']:.4f} | — | — | "
                    f"{s[f'mean_region_cauchy_field_rel_{path}_fine']:.4f} | "
                    f"{s[f'mean_region_sigma_xx_rel_{path}_fine']:.4f} | "
                    f"{s[f'mean_region_sigma_yy_rel_{path}_fine']:.4f} | "
                    f"{s[f'mean_region_sigma_zz_rel_{path}_fine']:.4f} | "
                    f"{s[f'mean_region_sigma_xy_rel_{path}_fine']:.4f} | "
                    f"{s[f'mean_region_sigma_xz_rel_{path}_fine']:.4f} | "
                    f"{s[f'mean_region_sigma_yz_rel_{path}_fine']:.4f} |")

    disc = data.get("fem_discretization_error")
    if disc:
        lines.append("\n## B3 — FEM-vs-FEM discretization error (no neural network), vs. the "
                      "950,400-element reference\n")
        lines.append("| Solution | disp L2 | H1 | energy | reaction M_y | region σ (field) |")
        lines.append("|---|---|---|---|---|---|")
        for tag, label in (("vs_6840", "6,840-element (training mesh)"),
                            ("vs_43400", "43,400-element (accuracy-matched)")):
            s = disc[tag]["summary"]
            lines.append(
                f"| {label} | {s['mean_disp_rel_l2_vs_950400']:.4f} | "
                f"{s['mean_H1_seminorm_rel_vs_950400']:.4f} | {s['mean_energy_rel_vs_950400']:.4f} | "
                f"{s['mean_reaction_moment_y_rel_vs_950400']:.4f} | "
                f"{s['mean_region_cauchy_field_rel_vs_950400']:.4f} |")
    return "\n".join(lines)


def summarize_b1(b1_json_path):
    with open(b1_json_path) as f:
        data = json.load(f)

    import numpy as np
    lines = [f"## B1 — PI vs. DD, inference meshes {data['resolutions']} vs. the "
             f"N={data['fine_N']} fine reference ({len(data['seeds'])} seed(s))\n"]
    lines.append(f"PI checkpoint fingerprint: `{data['pi_fingerprint']}`  \n"
                 f"DD checkpoint fingerprint: `{data['dd_fingerprint']}`\n")
    lines.append("| N | FEM(N) vs fine (L2) | PI vs fine (L2) | DD vs fine (L2) | "
                 "PI vs fine (H1) | DD vs fine (H1) |")
    lines.append("|---|---|---|---|---|---|")
    for N in data["resolutions"]:
        rows_n = [r for r in data["rows"] if r["N"] == N]
        if not rows_n:
            continue
        fem_l2 = float(np.mean([r["fem_vs_fine"]["l2_rel"] for r in rows_n]))
        pi_l2 = float(np.mean([r["pi_vs_fine"]["l2_rel"] for r in rows_n]))
        dd_l2 = float(np.mean([r["dd_vs_fine"]["l2_rel"] for r in rows_n]))
        pi_h1 = float(np.mean([r["pi_vs_fine"]["h1_semi_rel"] for r in rows_n]))
        dd_h1 = float(np.mean([r["dd_vs_fine"]["h1_semi_rel"] for r in rows_n]))
        lines.append(f"| {N} | {fem_l2 * 100:.2f}% | {pi_l2 * 100:.2f}% | {dd_l2 * 100:.2f}% | "
                     f"{pi_h1 * 100:.2f}% | {dd_h1 * 100:.2f}% |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--b3_json", type=str, default=None)
    parser.add_argument("--b1_json", type=str, default=None)
    parser.add_argument("--out_md", type=str, required=True)
    args = parser.parse_args()

    sections = []
    if args.b3_json:
        sections.append(summarize_b3(args.b3_json))
    if args.b1_json:
        sections.append(summarize_b1(args.b1_json))
    if not sections:
        raise SystemExit("give at least one of --b3_json / --b1_json")

    out = "\n\n".join(sections) + "\n"
    with open(args.out_md, "w") as f:
        f.write(out)
    print(out)
    print(f"\nWritten to {args.out_md}")


if __name__ == "__main__":
    main()
