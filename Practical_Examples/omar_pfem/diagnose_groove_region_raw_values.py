"""Follow-up to the contour plot (plot_groove_contour_stress.py): the
plot showed the 839,040- and 950,400-element normal-stress curves
(sigma_xx/yy/zz) visually overlapping almost everywhere, INCLUDING at
the groove -- apparently at odds with the 274% region_cauchy_field_rel
number. Two explanations: (1) the true stress there is itself small
(near the rocking/bending pattern's own zero-crossing around z=Lz/2),
so even a modest absolute mismatch becomes a large RELATIVE number; or
(2) a genuinely large absolute mismatch that just isn't visible at the
contour plot's y-axis scale (set by the much larger stress away from
the groove).

This uses the EXACT already-validated comparison method
(compute_one_resolution_vs_fine's own interpolate-then-compare steps --
not the contour plot's cruder 1D z-only interpolation) and reports RAW
per-Gauss-point values within the real region mask, not just the
aggregated percentage, to settle which explanation is correct, not
guess.

Usage:
  python -m omar_pfem.diagnose_groove_region_raw_values \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \\
      --sample_index 0
"""
import argparse

import numpy as np
import torch

from omar_pfem.evaluate_B3_qois_finer import (
    generate_fine_reference, interpolate_node_field_to_finer_mesh, FINE_RESOLUTION_STRESS,
)
from omar_pfem.evaluate_B3_qois import compute_region_sigma_full
from omar_pfem.data.data_generate_B3_dataset import R_GRADING

RESOLUTION_839040 = (97, 96, 93)
STRESS_COMPONENTS = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"

    print("Loading cached references (no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, args.n_samples, device, dtype,
                                       cache_path=fine_cache, require_region_mask=True)
    mid_cache = args.out_json + ".res_839040.cache.pt"
    mid_ref = generate_fine_reference(HOLDOUT, RESOLUTION_839040, args.n_samples, device, dtype,
                                       cache_path=mid_cache, require_region_mask=False, n_increments=25)

    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    Ntheta_c, Nr_c, Nz_c = mid_ref["fine_resolution"]
    geom_fine = fine_ref["geom_fine"]
    region_mask_fine_torch = fine_ref["region_mask_fine_torch"]

    s_f = fine_ref["samples"][args.sample_index]
    s_c = mid_ref["samples"][args.sample_index]

    # EXACT same step compute_one_resolution_vs_fine uses: interpolate the
    # 839,040-element solution's displacement field onto the 950,400-element
    # mesh's own nodes.
    u_c_np = s_c["u_true_f"][0].cpu().numpy()
    u_c_on_fine_np = interpolate_node_field_to_finer_mesh(
        u_c_np, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING, n_components=3)
    u_c_on_fine = torch.tensor(u_c_on_fine_np, dtype=dtype, device=device).unsqueeze(0)
    u_true_f = s_f["u_true_f"]
    E_f_t, nu_f_t = s_f["E_f_t"], s_f["nu_f_t"]

    sigma_true, w_region = compute_region_sigma_full(u_true_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
    sigma_interp, _ = compute_region_sigma_full(u_c_on_fine, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
    sigma_true = sigma_true[0]      # (n_region, 3, 3)
    sigma_interp = sigma_interp[0]
    w_region = w_region / w_region.sum()  # normalize to a volume-fraction weight for a weighted mean below

    print(f"\nn_region (Gauss points in the real groove region mask) = {sigma_true.shape[0]}")
    print(f"\n{'component':>10} {'true: mean':>12} {'true: |mean|':>14} {'true: RMS':>12} "
          f"{'true: max|.|':>14} {'interp: RMS':>12} {'|diff|: RMS':>14} {'rel (RMS ratio)':>16}")
    for name, a, b in STRESS_COMPONENTS:
        t = sigma_true[:, a, b]
        p = sigma_interp[:, a, b]
        w = w_region
        true_mean = (t * w).sum().item()
        true_rms = torch.sqrt((t ** 2 * w).sum()).item()
        true_max_abs = t.abs().max().item()
        interp_rms = torch.sqrt((p ** 2 * w).sum()).item()
        diff_rms = torch.sqrt(((p - t) ** 2 * w).sum()).item()
        rel = diff_rms / (true_rms + 1e-12)
        print(f"{name:>10} {true_mean:>12.4f} {abs(true_mean):>14.4f} {true_rms:>12.4f} "
              f"{true_max_abs:>14.4f} {interp_rms:>12.4f} {diff_rms:>14.4f} {rel*100:>15.2f}%")

    print("\n=== Reading this table ===")
    print("If 'true: RMS' is SMALL relative to the stress magnitude seen elsewhere in the domain")
    print("(the contour plot showed |sigma| up to several hundred away from the groove), the 274%-type")
    print("numbers are at least partly a small-denominator effect -- a modest absolute mismatch read")
    print("as a huge percentage. If 'true: RMS' is itself large AND 'diff: RMS' is comparably large,")
    print("it is a genuine large absolute discrepancy, not a denominator artifact.")


if __name__ == "__main__":
    main()
