"""Three summary plots for Timon's 2026-11-10 email response (local
z-refinement convergence, linear-vs-quadratic peak stress, and the
singularity check), built from already-committed real numbers in
PROJECT_STATUS.md (jobs 3141745/3141748, plus the extended CPU sweeps)
-- no new FEM solving, this just visualizes results already obtained
and recorded elsewhere. Matches this project's own convention of
committing the plotting SCRIPT, not the generated PNGs (see
.gitignore's *.png exception list).

Usage: python -m omar_pfem.plot_item1_summary_figures --out_dir <dir>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plots(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Local refinement at 47,040 elements (job 3141745)
    n_gauss = [2356, 3870, 8346, 9700, 10602]
    avg_xx = [3.027, 2.975, 2.850, 2.815, 2.800]
    max_xx = [47.65, 44.49, 47.78, 54.19, 63.35]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(n_gauss, avg_xx, "o-", color="#2a6099")
    axes[0].set_xlabel("Gauss points inside groove region")
    axes[0].set_ylabel("region-average |sigma_xx|")
    axes[0].set_title("Region-average stress: converges")
    axes[0].grid(alpha=0.3)
    axes[1].plot(n_gauss, max_xx, "o-", color="#c0504d")
    axes[1].set_xlabel("Gauss points inside groove region")
    axes[1].set_ylabel("pointwise max |sigma_xx|")
    axes[1].set_title("Pointwise max: noisy at fixed element order")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Local z-refinement at a fixed 47,040-element mesh")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "item1a_local_refinement_convergence.png"), dpi=150)
    plt.close(fig)

    # Linear vs quadratic peak stress (CPU checks + job 3141748)
    lin_max = {2376: 61.90, 5760: 61.19}
    quad_max = {2376: 28.97, 5760: 29.90, 10200: 29.61}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_lin = list(lin_max.keys())
    x_quad = list(quad_max.keys())
    ax.plot(x_lin, [lin_max[e] for e in x_lin], "o-", color="#c0504d",
            label="linear (8-node) elements")
    ax.plot(x_quad, [quad_max[e] for e in x_quad], "s-", color="#2a6099",
            label="quadratic (20-node) elements")
    ax.set_xlabel("number of elements")
    ax.set_ylabel("pointwise max |sigma_xx| at the groove")
    ax.set_title("Peak groove stress: linear vs. quadratic elements")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "item1c_linear_vs_quadratic.png"), dpi=150)
    plt.close(fig)

    # Singularity check: max stress vs smallest local z-element size (extended CPU sweep)
    min_z = [0.0417, 0.0231, 0.00129, 0.000042]
    max_xx2 = [61.19, 57.26, 58.13, 49.95]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(min_z, max_xx2, "o-", color="#2a6099")
    ax.set_xlabel("smallest local z-element size (log scale)")
    ax.set_ylabel("pointwise max |sigma_xx|")
    ax.set_title("Max stress stays bounded as local element size shrinks ~1000x\n"
                  "(linear elements, 5,760-element mesh) -- not a singularity signature")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "item1d_singularity_check.png"), dpi=150)
    plt.close(fig)

    print(f"wrote 3 figures to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    make_plots(args.out_dir)
