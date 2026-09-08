"""Item #12: field-panel-grid figures for the neural-operator studies
(Table 12 zero-shot resolution, Table 19/25 OOD shift, Table 26 DD-NO
coarse-vs-fine), in the same side-by-side style as
field_snapshot_grid.py's FEM convergence figure and the papers Omar
pointed to.

Reuses each study's OWN already-validated pipeline directly --
build_model/mesh_tensors_of/loss_and_pred/install_input_norm_for_checkpoint
from resolution_invariance_zeroshot.py, build_shifted_b1/b2 from
ood_progressive.py -- rather than re-deriving sample construction,
input normalization, or soft-Dirichlet enforcement independently. Only
the LAST step (extracting the predicted field and handing it to
panel_grid_plot) is new; the physics/model-loading code is exactly what
already produced the real numbers in Tables 12/19/25/26.

Model hyperparameters are reconstructed from add_common_args's own
defaults, matching every training run in this project that did not
override them; if a given checkpoint used different values,
model.load_state_dict fails loudly on a shape mismatch rather than
silently loading garbage.
"""
import argparse

import numpy as np
import torch

from omar_pfem.panel_grid_plot import plot_panel_grid, displacement_magnitude


def _default_args(geometry, material):
    from omar_pfem.resolution_invariance_zeroshot import add_common_args
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args([f'--geometry={geometry}', f'--material={material}'])
    return args


def _load_model(checkpoint, geometry, material, device):
    from omar_pfem.resolution_invariance_zeroshot import build_model
    from omar_pfem.train_B1 import install_input_norm_for_checkpoint
    args = _default_args(geometry, material)
    model = build_model(args, device)
    install_input_norm_for_checkpoint(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, args


@torch.no_grad()
def _predict(geometry, sample, model, args, device, dtype=torch.float32):
    from omar_pfem.resolution_invariance_zeroshot import mesh_tensors_of, loss_and_pred
    mesh_t = mesh_tensors_of(geometry, sample, device, dtype)
    E_b = torch.tensor(sample['E_node'][None], device=device, dtype=dtype)
    nu_b = torch.tensor(sample['nu_node'][None], device=device, dtype=dtype)
    f_b = torch.tensor(sample['node_forces'][None], device=device, dtype=dtype)
    _, _, _, uv_pred, _ = loss_and_pred(geometry, mesh_t, model, E_b, nu_b, f_b, args, dtype)
    return uv_pred[0].cpu().numpy()


def make_zeroshot_resolution_grid(checkpoint, geometry, material, resolutions,
                                   seed, out_path, device=None):
    """Table 12 style: ONE trained checkpoint, evaluated at several
    resolutions never seen together, no retraining -- the network's own
    predicted |u| field, side by side across N."""
    from omar_pfem.resolution_invariance_zeroshot import build_sample_b1, build_sample_b2

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, args = _load_model(checkpoint, geometry, material, device)
    build_fn = build_sample_b1 if geometry == 'B1' else build_sample_b2

    panels = []
    for N in resolutions:
        sample, _ = build_fn(N, seed=seed, material=material, solve_fem=False)
        uv_pred = _predict(geometry, sample, model, args, device)
        panels.append({
            'title': f'N={N}\n({2 * sample["xy"].shape[0]:,} DOF)',
            'x': sample['xy'][:, 0], 'y': sample['xy'][:, 1],
            'values': displacement_magnitude(uv_pred),
        })

    return plot_panel_grid(
        panels, out_path, cmap='viridis',
        suptitle=f'One trained operator -> {len(resolutions)} resolutions (no retraining)\n'
                  f'({geometry}, {material.replace("_", "-").title()}, predicted |u|; Table 12)')


def make_ood_shift_grid(checkpoint, geometry, material, factor, shift_sigmas,
                         out_path, seed=90_000_000, device=None):
    """Table 19/25 style: ONE trained checkpoint, evaluated at baseline plus
    progressively shifted material/loading distributions -- predicted |u|
    field side by side, one panel per shift level (matching the
    "Nominal / Unseen a / Unseen b" panels Omar pointed to)."""
    from omar_pfem.ood_progressive import build_shifted_b1, build_shifted_b2, field_kwargs

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, args = _load_model(checkpoint, geometry, material, device)
    build_fn = build_shifted_b1 if geometry == 'B1' else build_shifted_b2
    N = 21  # matches ood_progressive.py's own default study resolution

    panels = []
    for k in shift_sigmas:
        kw = field_kwargs(geometry, factor, k)
        sample = build_fn(N, seed, material, kw)
        uv_pred = _predict(geometry, sample, model, args, device)
        label = 'Nominal (baseline)' if k == 0 else f'{factor} shift, k={k}σ'
        panels.append({
            'title': label, 'x': sample['xy'][:, 0], 'y': sample['xy'][:, 1],
            'values': displacement_magnitude(uv_pred),
        })

    return plot_panel_grid(
        panels, out_path, cmap='viridis',
        suptitle=f'Out-of-distribution shift ({factor}), predicted |u|\n'
                  f'({geometry}, {material.replace("_", "-").title()}; Tables 19/25)')


def make_dd_no_coarse_vs_fine_grid(coarse_checkpoint, fine_checkpoint, resolutions,
                                    out_path, seed=20_000_000, geometry='B1',
                                    material='neo_hookean', device=None):
    """Table 26 style: the SAME test resolutions, predicted by the
    coarse-trained (N=13) and fine-trained (N=33) DD-NO checkpoints, one
    row per model -- visualizes the finding Table 26 already reports in
    numbers (fine-trained generalizes more evenly across resolutions)."""
    from omar_pfem.resolution_invariance_zeroshot import build_sample_b1, build_sample_b2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_fn = build_sample_b1 if geometry == 'B1' else build_sample_b2

    rows = []
    for ckpt, row_label in [(coarse_checkpoint, 'Coarse-trained (N=13)'),
                             (fine_checkpoint, 'Fine-trained (N=33)')]:
        model, args = _load_model(ckpt, geometry, material, device)
        panels = []
        for N in resolutions:
            sample, _ = build_fn(N, seed=seed, material=material, solve_fem=False)
            uv_pred = _predict(geometry, sample, model, args, device)
            panels.append({
                'title': f'N={N}', 'x': sample['xy'][:, 0], 'y': sample['xy'][:, 1],
                'values': displacement_magnitude(uv_pred),
            })
        rows.append((row_label, panels))

    vmin = min(p['values'].min() for _, panels in rows for p in panels)
    vmax = max(p['values'].max() for _, panels in rows for p in panels)

    fig, axes = plt.subplots(2, len(resolutions), figsize=(2.6 * len(resolutions), 6.0), dpi=200)
    im = None
    for r, (row_label, panels) in enumerate(rows):
        for c, p in enumerate(panels):
            ax = axes[r, c]
            im = ax.tricontourf(p['x'], p['y'], p['values'], levels=20, cmap='viridis',
                                 vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if r == 0:
                ax.set_title(p['title'], fontsize=9)
        axes[r, 0].set_ylabel(row_label, fontsize=9)

    fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.1)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax)
    fig.suptitle('DD-NO trained on coarse vs. fine FEM labels, predicted |u|\n'
                  'zero-shot across resolutions (Table 26)', fontsize=10, y=1.02)

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print('Saved', out_path)
    return out_path
