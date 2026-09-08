"""Item #12: one shared plotting utility for every "side-by-side field
panel" figure across this project's studies (FEM convergence, zero-shot
resolution, OOD shift, DD-NO coarse-vs-fine), in the style Omar pointed
to from Timon's own group's papers (VINO Figs. 7a/8d/9d; the PFEM/NOWS
screenshots he shared).

Uses `tricontourf` (unstructured, triangulation-based contouring)
rather than `contourf` on a reshaped regular grid: B1's mesh IS a plain
rectangular grid (reshaping would work there), but B2's is a polar-
mapped ring, and re-deriving its exact (Ntheta, Nr) reshape order for
every caller would be one more place to get subtly wrong. `tricontourf`
takes the raw (x, y, values) point cloud directly, with no assumption
about node ordering, so ONE function works correctly for every geometry
this project has, at the cost of a slightly less pixel-crisp contour
than a reshaped image would give (visually indistinguishable at the
resolutions these figures use).
"""
import os

import numpy as np


def plot_panel_grid(panels, out_path, suptitle=None, cmap='viridis', shared_scale=True):
    """panels: list of dicts, each {"title": str, "x": array, "y": array,
    "values": array} (all 1D, one value per mesh node). Renders one row
    of side-by-side tricontourf panels, saved as one PNG.

    shared_scale: if True (the papers' own convention when comparing the
    same quantity across cases), every panel uses the same colour scale
    (global min/max across all panels) so panel-to-panel differences are
    visually comparable, not artifacts of independent auto-scaling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if shared_scale:
        vmin = min(p['values'].min() for p in panels)
        vmax = max(p['values'].max() for p in panels)
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(1, len(panels), figsize=(2.8 * len(panels), 3.1), dpi=200)
    if len(panels) == 1:
        axes = [axes]

    im = None
    for ax, p in zip(axes, panels):
        kwargs = dict(levels=20, cmap=cmap)
        if shared_scale:
            kwargs['vmin'], kwargs['vmax'] = vmin, vmax
        im = ax.tricontourf(p['x'], p['y'], p['values'], **kwargs)
        ax.set_title(p['title'], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    fig.subplots_adjust(right=0.88, wspace=0.15)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.05)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print('Saved', out_path)
    return out_path


def displacement_magnitude(uv):
    return np.sqrt((uv ** 2).sum(axis=1))
