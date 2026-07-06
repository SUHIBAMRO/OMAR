"""
Visualizations matching Fig. 7/8 in the VINO paper (Eshaghi et al., CMAME
437 (2025) 117785) and the plotting section of VINO_Hyperelasticity.py:
  - box plot of the relative L2 error across the whole test set
  - exact / predicted / error field comparison (X- and Y-displacement)
    for one representative test sample

Reuses utils/postprocessing.py's plot_field_2d unchanged for B1 (flat
domain, same convention as the original beam/plate examples). B2's
domain is curved (quarter ring), so plot_field_2d's flat L x W axes
would show the (theta, r) computational rectangle instead of the true
annulus shape -- plot_field_2d_polar below is the same jet/bwr contourf
style, just evaluated at each grid point's real physical (x, y) location.
"""
import os

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.fno_utils import collate_fn
from utils.postprocessing import plot_field_2d


def test_errors(model, model_params, loss_fn, test_dataset, normalizer_y):
    """
    Same per-sample relative L2 error as utils.fno_utils.model_evaluation,
    but returns the full array (needed for the box plot) instead of only
    printing summary statistics.
    """
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    errors = []
    for x, y in test_loader:
        y = normalizer_y.decode(y)
        y_pred = normalizer_y.decode(model.apply(model_params, x))
        errors.append(loss_fn(y, y_pred).item())
    return np.array(errors)


def plot_box_error(errors, title, folder=None, file=None):
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.figure(figsize=(4, 4))
    plt.boxplot(errors, vert=True)
    plt.yscale('log')
    plt.ylabel('Relative L2 error')
    plt.title(title)
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{file}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_field_2d_polar(F, r_in, r_out, title, folder=None, file=None, isError=False):
    """Same style as utils.postprocessing.plot_field_2d, evaluated on the
    true physical (x, y) quarter-ring instead of a flat rectangle. F is on
    the (theta, r) grid: rows = theta in [0, pi/2], columns = r in [r_in, r_out]."""
    plt.rcParams["font.family"] = "DejaVu Serif"
    ny, nx = F.shape
    theta = np.linspace(0, np.pi / 2, ny)
    r = np.linspace(r_in, r_out, nx)
    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    plt.figure()
    color_type = 'bwr' if isError else 'jet'
    contour = plt.contourf(x, y, F, levels=512, cmap=color_type)
    plt.colorbar(contour)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{file}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_case_results(name, model, model_data, model_params, loss_fn, test_dataset, normalizers, folder,
                      is_b2=False):
    """
    Generates the box plot + exact/predicted/error field comparison for
    one trained case (the loss-curve plot is handled separately). Returns
    the per-sample test error array.
    """
    normalizer_x, normalizer_y = normalizers
    errors = test_errors(model, model_params, loss_fn, test_dataset, normalizer_y)
    plot_box_error(errors, f'{name}: test set relative L2 error', folder=folder, file=f'{name}_error_box')

    # a "typical" (median-error) sample, not a cherry-picked best case
    index = int(np.argsort(errors)[len(errors) // 2])
    x_test, y_test = test_dataset[index]
    x_test = jnp.expand_dims(jnp.array(x_test), 0)
    y_test = jnp.expand_dims(jnp.array(y_test), 0)
    y_pred = model.apply(model_params, x_test)
    if model_data["normalized"]:
        y_test = normalizer_y.decode(y_test)
        y_pred = normalizer_y.decode(y_pred)

    if is_b2:
        r_in, r_out = model_data["beam"]["R_inner"], model_data["beam"]["R_outer"]

        def plot_fn(field, title, file, is_error=False):
            plot_field_2d_polar(field, r_in, r_out, title, folder=folder, file=file, isError=is_error)
    else:
        length, width = model_data["beam"]["length"], model_data["beam"]["width"]

        def plot_fn(field, title, file, is_error=False):
            plot_field_2d(field, length, width, title, folder=folder, file=file, isError=is_error)

    for comp, comp_name in [(0, 'X'), (1, 'Y')]:
        exact = np.array(y_test[0, :, :, comp])
        pred = np.array(y_pred[0, :, :, comp])
        plot_fn(exact, f'{name}: {comp_name}-displacement Exact', f'{name}_{comp_name}_exact')
        plot_fn(pred, f'{name}: {comp_name}-displacement Predicted', f'{name}_{comp_name}_pred')
        plot_fn(exact - pred, f'{name}: {comp_name}-displacement Error', f'{name}_{comp_name}_error', is_error=True)

    return errors
