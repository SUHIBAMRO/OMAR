"""Diagnostic requested explicitly by Prof. Rabczuk's 2026-11-04 email,
point 4: "Can you please also test the stress error in another region
far away from the groove. If the error is very high there as well, it
indicates something is awkward."

This reuses the EXACT SAME methodology, samples, and already-solved/
cached FEM fields as evaluate_B3_fem_discretization_error.py (no new
FEM solves -- it loads the same `.fine_950400.cache.pt` /
`.fine_43400.cache.pt` sidecar files the real re-evaluation run already
produced), but swaps the region mask from the groove root to a point at
the SAME radius and angle, one quarter of the way along the bushing's
axis (z = LZ/4 = 0.25), where the inner radius is flat (R_in(z) = R_IN0
= 0.5, confirmed directly -- no groove influence at all there) and far
from both the groove (z = LZ/2) and the two free end faces (z = 0, LZ)
that could otherwise introduce their own boundary-related stress
artifacts. Same region RADIUS (2x the groove's own radius of curvature)
as the production region mask, for an apples-to-apples region SIZE.

If this far-field region's discretization error is also huge, that
would point to a genuine interpolation/mapping bug. If it is small
(consistent with every other QoI's few-percent discretization error),
that confirms the groove-region error is a real, localized geometric
effect, not an artifact of how coarse and fine meshes are compared.

Usage (same conda env, same run_dir as the real re-evaluation run):
  python -m omar_pfem.verify_far_region_discretization_error \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json
"""
import argparse

import numpy as np
import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION, R_GRADING
from omar_pfem.data.mesh_convergence_B3 import (
    R_IN0, R_OUT, LZ, HEXA1_IPOINTS, _gauss_point_parametric,
)
from omar_pfem.data.data_generate_B3 import groove_R_in, groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH
from omar_pfem.evaluate_B3_qois import (
    compute_region_sigma_full, cauchy_field_rel_from_full, sigma_components_rel_from_full,
    STRESS_COMPONENTS,
)
from omar_pfem.evaluate_B3_qois_finer import (
    generate_fine_reference, interpolate_node_field_to_finer_mesh,
    FINE_RESOLUTION_STRESS, FINE_RESOLUTION_DISP_ENERGY_REACTION,
)
from omar_pfem.evaluate_B3_fem_discretization_error import load_coarse_reference


def build_far_region_mask(Ntheta, Nr, Nz, r_grading, ref_point, region_radius):
    """Same construction as evaluate_B3_qois.build_region_mask, but at an
    arbitrary center -- here the groove-free point (R_IN0, theta=0,
    z=LZ/4) instead of the groove root."""
    theta_g, t_g, z_g = _gauss_point_parametric(Ntheta, Nr, Nz, r_grading, HEXA1_IPOINTS)
    R_in_eff_g = groove_R_in(z_g, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    Rr_g = R_in_eff_g + (R_OUT - R_in_eff_g) * t_g
    x_g = Rr_g * np.cos(theta_g)
    y_g = Rr_g * np.sin(theta_g)
    pos_g = np.stack([x_g, y_g, z_g], axis=-1)
    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    return dist < region_radius


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True,
                         help="same --out_json the real re-evaluation run used, so this "
                              "script reuses its cached fine/mid FEM solutions instead of "
                              "re-solving them")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"device: {device}")

    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    N_SAMPLES = args.n_samples

    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    far_ref_point = np.array([R_IN0, 0.0, LZ / 4.0])
    print(f"Far region: center={far_ref_point.tolist()}, radius={region_radius:.6f} "
          f"(same size as the groove-root region; R_in at z=LZ/4 is R_IN0={R_IN0} -- "
          f"confirmed flat, no groove influence there)")

    print(f"\n{'=' * 90}\nLoading cached 950,400-element ({FINE_RESOLUTION_STRESS}) reference "
          f"(no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, N_SAMPLES, device, dtype,
                                        cache_path=fine_cache)

    print(f"\n{'=' * 90}\nLoading cached 43,400-element ({FINE_RESOLUTION_DISP_ENERGY_REACTION}) "
          f"solution (no new solves) ...")
    mid_cache = args.out_json + ".fine_43400.cache.pt"
    mid_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_DISP_ENERGY_REACTION, N_SAMPLES,
                                       device, dtype, cache_path=mid_cache)

    print(f"\n{'=' * 90}\nLoading already-solved 6,840-element solution from dataset.h5 ...")
    coarse_ref = load_coarse_reference(HOLDOUT, N_SAMPLES, device, dtype)

    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    geom_fine = fine_ref["geom_fine"]
    far_mask_fine_np = build_far_region_mask(Ntheta_f, Nr_f, Nz_f, R_GRADING,
                                              far_ref_point, region_radius)
    far_mask_fine_torch = torch.tensor(far_mask_fine_np, device=device)
    n_far_fine = int(far_mask_fine_np.sum())
    print(f"\nFar-region Gauss points at 950,400 elements: {n_far_fine} "
          f"(groove-region count at this resolution was 794, for comparison)")

    n_samples_eff = min(len(coarse_ref["samples"]), len(mid_ref["samples"]), len(fine_ref["samples"]))

    for ref, tag in ((coarse_ref, "6,840-element"), (mid_ref, "43,400-element")):
        Ntheta_c, Nr_c, Nz_c = ref["fine_resolution"]
        n_elem = (Ntheta_c - 1) * (Nr_c - 1) * (Nz_c - 1)
        far_mask_c_np = build_far_region_mask(Ntheta_c, Nr_c, Nz_c, R_GRADING,
                                               far_ref_point, region_radius)
        n_far_c = int(far_mask_c_np.sum())
        print(f"\n{'=' * 90}\n=== [FAR region, vs. 950,400 el] {tag} solution "
              f"({n_far_c} far-region Gauss points at this resolution) ===")

        field_rels, comp_rels_per_name = [], {name: [] for name, _, _ in STRESS_COMPONENTS}
        for idx in range(n_samples_eff):
            s_c = ref["samples"][idx]
            s_f = fine_ref["samples"][idx]
            assert s_c["sample_index"] == s_f["sample_index"], "sample ordering mismatch"

            u_c_np = s_c["u_true_f"][0].cpu().numpy()
            u_c_on_fine_np = interpolate_node_field_to_finer_mesh(
                u_c_np, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING, n_components=3)
            u_c_on_fine = torch.tensor(u_c_on_fine_np, dtype=dtype, device=device).unsqueeze(0)
            u_true_f = s_f["u_true_f"]
            E_f_t, nu_f_t = s_f["E_f_t"], s_f["nu_f_t"]

            sigma_true_f, w_true_f = compute_region_sigma_full(
                u_true_f, E_f_t, nu_f_t, geom_fine, far_mask_fine_torch)
            sigma_c_on_fine, _ = compute_region_sigma_full(
                u_c_on_fine, E_f_t, nu_f_t, geom_fine, far_mask_fine_torch)
            field_rel = float(cauchy_field_rel_from_full(sigma_true_f, w_true_f, sigma_c_on_fine)[0].item())
            comp_rel = sigma_components_rel_from_full(sigma_true_f, w_true_f, sigma_c_on_fine)
            field_rels.append(field_rel)
            for name, _, _ in STRESS_COMPONENTS:
                comp_rels_per_name[name].append(float(comp_rel[name][0].item()))

        field_rels = np.array(field_rels)
        print(f"  region_cauchy_field_rel (FAR region): mean={np.nanmean(field_rels)*100:.4f}%  "
              f"median={np.nanmedian(field_rels)*100:.4f}%")
        comp_str = "  ".join(f"{name}={np.nanmean(comp_rels_per_name[name])*100:.4f}%"
                              for name, _, _ in STRESS_COMPONENTS)
        print(f"  region_sigma_components_rel (FAR region): {comp_str}")

    print(f"\n{'=' * 90}\nDone. Compare these FAR-region numbers against the already-reported "
          f"GROOVE-region numbers (6,840-el: 21,703% field / 24,021-26,665% normal components; "
          f"43,400-el: 6,100% field / 6,714-7,473% normal components). If the far numbers are "
          f"small (a few percent, like every other QoI), the groove-region error is a real, "
          f"localized geometric effect, not an interpolation bug.")


if __name__ == "__main__":
    main()
