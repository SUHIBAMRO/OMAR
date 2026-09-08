"""Item #12: build a side-by-side grid of displacement-field snapshots
across mesh resolutions, matching the figure style used in Timon's own
group's papers (VINO Figs. 7a/8d/9d; the PFEM/NOWS screenshots Omar
shared: one panel per resolution, arranged in a row, labeled by point
count / DOF).

Reuses item #4's own already-converged checkpoints on Drive
(coarse_{geometry}_{material}_Q4_N{N}.pt) directly -- these already
hold the converged solution's free-DOF displacement (u_free) from the
real GPU runs already committed to this project (see
matrix_free_solver.py's own checkpoint format: {"u_free": ..., "stats":
..., "next_step": ...}). No new solve is needed: build_mesh_and_bcs is
cheap (pure numpy mesh generation), and loading + reshaping the already-
converged field is essentially free, so this whole figure costs no new
GPU time.

B1's own Q4 mesh (generate_grid_Q4) is a plain (Ny, Nx) rectangular
grid with node index j*Nx+i -- confirmed directly from its own source,
not assumed -- so the free-DOF displacement can be scattered back into
the full (N, N, 2) array and reshaped into an image with no
interpolation step.
"""
import os

import numpy as np
import torch


def load_converged_field_B1(N, checkpoint_dir, material='neo_hookean', dtype=torch.float64):
    """Returns (X, Y, Umag) as (N, N) arrays: the grid coordinates and the
    converged displacement magnitude at every node, for B1's own Q4 mesh
    at resolution N, read from item #4's own checkpoint on Drive."""
    from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs

    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        'B1', 'Q4', N, material, torch.device('cpu'), dtype)

    ckpt_path = os.path.join(checkpoint_dir, f'coarse_B1_{material}_Q4_N{N}.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    u_free = ckpt['u_free'].numpy() if torch.is_tensor(ckpt['u_free']) else ckpt['u_free']

    n_nodes = nodes.shape[0]
    u_full = np.zeros(2 * n_nodes, dtype=u_free.dtype)
    u_full[free_dofs] = u_free

    u_full = u_full.reshape(n_nodes, 2)
    umag = np.sqrt((u_full ** 2).sum(axis=1))

    X = nodes[:, 0].reshape(N, N)
    Y = nodes[:, 1].reshape(N, N)
    Umag = umag.reshape(N, N)
    return X, Y, Umag, ckpt['stats']


def make_resolution_grid_figure(Ns, checkpoint_dir, out_path, material='neo_hookean',
                                 dtype=torch.float64):
    """One combined figure: |u| contour for each N in Ns, side by side,
    each panel titled by its own DOF count (matching the papers' own
    "N points" column-header convention)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fields = []
    for N in Ns:
        X, Y, Umag, stats = load_converged_field_B1(N, checkpoint_dir, material, dtype)
        fields.append((N, X, Y, Umag))

    vmax = max(f[3].max() for f in fields)
    vmin = 0.0

    fig, axes = plt.subplots(1, len(fields), figsize=(2.6 * len(fields), 2.9), dpi=200)
    if len(fields) == 1:
        axes = [axes]

    im = None
    for ax, (N, X, Y, Umag) in zip(axes, fields):
        im = ax.contourf(X, Y, Umag, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        n_dof = 2 * N * N
        ax.set_title(f'N={N}\n({n_dof:,} DOF)', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    fig.subplots_adjust(right=0.88, wspace=0.15)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label='|u|')

    fig.suptitle(f'Mesh convergence (B1, {material.replace("_", "-").title()})',
                  fontsize=10, y=1.05)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print('Saved', out_path)
    return out_path
