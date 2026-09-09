"""Item #12: one shared plotting utility for every "side-by-side field
panel" figure across this project's studies (FEM convergence, zero-shot
resolution, OOD shift, DD-NO coarse-vs-fine), in the style Omar pointed
to from Timon's own group's papers (VINO Figs. 7a/8d/9d; the PFEM/NOWS
screenshots he shared).

Uses `tricontourf` (unstructured, triangulation-based contouring)
rather than `contourf` on a reshaped regular grid: B1's mesh IS a plain
rectangular grid (reshaping would work there), but B2's is a polar-
mapped ring, and re-deriving its exact (Ntheta, Nr) reshape order for
every caller would be one more place to get subtly wrong.

IMPORTANT (found and fixed 2026-09-09, after Omar spotted it directly
in a rendered figure): passing only the raw (x, y, values) point cloud,
with no `quad` connectivity, makes `tricontourf` fall back to a plain
Delaunay triangulation of the point cloud's CONVEX HULL. For B1 (a
solid rectangle) that is harmless. For B2 -- a quarter RING, R_in=1 to
R_out=2, an annulus with a genuinely empty hole from r=0 to r=1 -- the
convex hull spans straight across that empty hole, so Delaunay quietly
fills it with interpolated colour as if material were there, rendering
a solid quarter-disk instead of the true quarter-ring-with-a-hole. Any
caller that has the real mesh's Q4 element connectivity available
should now pass it as `quad` on each panel dict: each quad [n1,n2,n3,n4]
(CCW, per data_generate_B1.py/data_generate_B2.py's own convention) is
split into triangles (n1,n2,n3) and (n1,n3,n4), giving a triangulation
that exactly matches the real mesh -- it cannot span a hole the mesh
itself does not have, unlike a blind Delaunay of the scattered nodes.
`quad` is optional and backward compatible: omitting it reproduces the
old Delaunay-of-point-cloud behaviour (still fine for B1-only figures).
"""
import os

import numpy as np


def _triangulation_for(p):
    """Builds a matplotlib Triangulation from real mesh connectivity when
    a panel provides 'quad' (each row [n1,n2,n3,n4], CCW); otherwise
    returns None so the caller falls back to plain tricontourf(x, y, ...)
    (a scattered-point Delaunay -- fine only when the domain has no
    interior hole, e.g. B1)."""
    quad = p.get('quad')
    if quad is None:
        return None
    from matplotlib.tri import Triangulation
    quad = np.asarray(quad)
    tris = np.vstack([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
    return Triangulation(p['x'], p['y'], tris)


def plot_panel_grid(panels, out_path, suptitle=None, cmap='viridis', shared_scale=True):
    """panels: list of dicts, each {"title": str, "x": array, "y": array,
    "values": array} (all 1D, one value per mesh node), plus an optional
    "quad" array (real Q4 element connectivity, [n1,n2,n3,n4] per row --
    see the module docstring for why this matters for ring-shaped
    domains like B2). Renders one row of side-by-side tricontourf
    panels, saved as one PNG.

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
        triang = _triangulation_for(p)
        if triang is not None:
            im = ax.tricontourf(triang, p['values'], **kwargs)
        else:
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
