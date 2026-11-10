"""Timon's 2026-11-10 email, item 1: "Can you plot the stress
distribution (in different components) along the groove contour" --
plus the related question "Is the groove sufficiently smooth that we
can exclude singularities due to the geometry?"

Smoothness answer (worked out analytically + confirmed numerically
BEFORE writing this plotting script, see PROJECT_STATUS.md): the groove
profile r(z) = R_in0 - (d/2)(1+cos(pi(z-zc)/w)) is a raised-cosine dip.
At its own deepest point (z=zc) it is perfectly smooth -- a finite,
nonzero radius of curvature (rho ~ 0.0228, `groove_radius_of_curvature`),
not a singularity (a true geometric singularity, e.g. a V-notch or crack
tip, has an UNBOUNDED curvature/zero radius; this groove's curvature is
bounded). HOWEVER, the profile is only C1 (not C2) at the two band
edges z = zc +/- w: confirmed numerically that the second derivative
(curvature) jumps from a finite nonzero value just inside the band to
exactly 0 just outside it (a real, if mild, curvature DISCONTINUITY --
no kink in slope, but a kink in curvature). This is a SEPARATE
geometric feature from the main groove root, and this project's
existing region-stress mask (radius = 2*rho ~ 0.0456) is centered on
the deepest point and does NOT reach anywhere near the band edges
(w=0.15, over 3x farther out) -- so the existing 274%/region-stress
numbers say nothing about whether this second feature is also a problem.
This script's whole-contour plot is what actually answers that, rather
than reasoning about it.

Reuses the exact, already-validated `compute_region_sigma_full` Cauchy-
stress machinery (same P=dpsi/dF via autograd, same sigma=(1/detF)*P*F^T)
with an ALL-Gauss-points mask instead of the small region mask, then
selects the single innermost-radial, first-angular-ring of elements
(i==0, j==0 in the structured (theta,r,z) index -- `_element_ijk`) to
get a clean z-sweep right at the inner wall near theta=0, matching the
existing region mask's own theta=0 center. Plots every stress component
at both 839,040 and 950,400 elements (the SAME cached fine-reference
solves already used for the convergence check) on the same axes, so the
plot itself shows whether the groove ROOT converges but something near
the band edges does not, or neither does -- real data, not conjecture.

Usage:
  python -m omar_pfem.plot_groove_contour_stress \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \\
      --sample_index 0 --out_dir /scratch/gole3157/pfem_run/groove_contour_plots
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, FINE_RESOLUTION_STRESS
from omar_pfem.evaluate_B3_qois import compute_region_sigma_full
from omar_pfem.data.mesh_convergence_B3 import HEXA1_IPOINTS, _gauss_point_parametric, _element_ijk
from omar_pfem.data.data_generate_B3 import groove_R_in, groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH, R_GRADING
from omar_pfem.data.mesh_convergence_B3 import R_IN0, LZ

RESOLUTION_839040 = (97, 96, 93)

STRESS_COMPONENTS = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]


def inner_wall_contour_sigma(u, E_node, nu_node, geom, Ntheta, Nr, Nz, sample_index):
    """sigma restricted to the FIRST angular ring (j==0, closest to
    theta=0 -- the existing region mask's own center) and FIRST radial
    ring (i==0, right at the inner/groove wall), spanning every z -- a
    clean inner-wall contour sweep. Only computes stress at these
    (n_z-1)*n_gauss Gauss points, not the whole domain (compute_region_
    sigma_full takes a mask and only builds F/P/sigma where it's True,
    so passing the contour mask directly -- not an all-True one -- keeps
    this as cheap as the existing small-region evaluation, not a full-
    domain stress field). Returns (z_sorted, {component_name: sigma_sorted})."""
    n_gauss = HEXA1_IPOINTS.shape[0]
    j, i, k = _element_ijk(Ntheta, Nr, Nz)
    theta_g, t_g, z_g = _gauss_point_parametric(Ntheta, Nr, Nz, R_GRADING, HEXA1_IPOINTS)

    contour_elem_mask = (j == 0) & (i == 0)  # (n_elem,) boolean
    contour_mask = np.broadcast_to(contour_elem_mask[:, None], (contour_elem_mask.shape[0], n_gauss))
    contour_mask_torch = torch.from_numpy(contour_mask.copy())

    sigma_contour_all, _ = compute_region_sigma_full(u, E_node, nu_node, geom, contour_mask_torch)
    sigma_contour = sigma_contour_all[sample_index]  # (n_contour_gauss, 3, 3), same row-major order as z_g[mask]
    z_contour = z_g[contour_elem_mask].reshape(-1)

    order = np.argsort(z_contour)
    z_sorted = z_contour[order]
    out = {}
    for name, a, b in STRESS_COMPONENTS:
        out[name] = sigma_contour[:, a, b].cpu().numpy()[order]
    return z_sorted, out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    os.makedirs(args.out_dir, exist_ok=True)
    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"

    print("Loading cached 950,400-element reference (no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, args.n_samples, device, dtype,
                                       cache_path=fine_cache, require_region_mask=True)
    print("Loading cached 839,040-element solution (no new solve, from the convergence check) ...")
    mid_cache = args.out_json + ".res_839040.cache.pt"
    mid_ref = generate_fine_reference(HOLDOUT, RESOLUTION_839040, args.n_samples, device, dtype,
                                       cache_path=mid_cache, require_region_mask=False, n_increments=25)

    s_fine = fine_ref["samples"][args.sample_index]
    s_mid = mid_ref["samples"][args.sample_index]

    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    Ntheta_m, Nr_m, Nz_m = mid_ref["fine_resolution"]

    z_fine, sigma_fine = inner_wall_contour_sigma(
        s_fine["u_true_f"], s_fine["E_f_t"], s_fine["nu_f_t"], fine_ref["geom_fine"],
        Ntheta_f, Nr_f, Nz_f, 0)
    z_mid, sigma_mid = inner_wall_contour_sigma(
        s_mid["u_true_f"], s_mid["E_f_t"], s_mid["nu_f_t"], mid_ref["geom_fine"],
        Ntheta_m, Nr_m, Nz_m, 0)

    zc = LZ / 2.0
    band_lo, band_hi = zc - GROOVE_HALF_WIDTH, zc + GROOVE_HALF_WIDTH
    rho = groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    region_radius = 2.0 * rho

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, (name, _, _) in zip(axes.ravel(), STRESS_COMPONENTS):
        ax.plot(z_mid, sigma_mid[name], "o-", ms=3, label="839,040-element", alpha=0.8)
        ax.plot(z_fine, sigma_fine[name], "o-", ms=3, label="950,400-element", alpha=0.8)
        ax.axvline(zc, color="k", linestyle=":", linewidth=1, label="groove deepest point (zc)" if name == "xx" else None)
        ax.axvline(band_lo, color="r", linestyle="--", linewidth=1, label="band edge (curvature jump)" if name == "xx" else None)
        ax.axvline(band_hi, color="r", linestyle="--", linewidth=1)
        ax.axvspan(zc - region_radius, zc + region_radius, color="gray", alpha=0.15,
                   label="existing region mask extent" if name == "xx" else None)
        ax.set_title(f"sigma_{name}")
        ax.set_xlabel("z")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"B3 inner-wall (theta~0) stress along z, sample {args.sample_index}: "
                 f"839,040 vs. 950,400 elements\n"
                 f"groove band: [{band_lo:.3f}, {band_hi:.3f}], deepest point zc={zc:.3f}, "
                 f"existing region mask radius={region_radius:.4f}")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"groove_contour_stress_sample{args.sample_index}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    print("\n=== Quick read: does each component look converged along the WHOLE contour (not just the region mask)? ===")
    for name, _, _ in STRESS_COMPONENTS:
        # crude common-z comparison: interpolate the coarser (839,040) onto the finer sample's z grid
        interp = np.interp(z_fine, z_mid, sigma_mid[name])
        rel = np.abs(interp - sigma_fine[name]) / (np.abs(sigma_fine[name]).max() + 1e-12)
        worst_z = z_fine[np.argmax(rel)]
        print(f"  sigma_{name}: max pointwise rel. diff along contour = {rel.max()*100:.1f}% at z={worst_z:.4f} "
              f"(groove root at z={zc:.4f}, band edges at {band_lo:.4f}/{band_hi:.4f})")


if __name__ == "__main__":
    main()
