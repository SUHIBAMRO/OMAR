"""Timon's 2026-11-10 email, item 2: "For the far-field control, here
we can evaluate PI and the different DD variants. This gives us a
meaningful PI-vs-DD stress comparison at least away from the unresolved
groove concentration."

Reuses EVERY piece of the already-validated groove-region checkpoint
evaluation (`run_b3_rabczuk_reeval_v2_local.py`'s own `U_ONLY_NAMES`
dict and `evaluate_u_only_checkpoint_against_fine_reference`) completely
unmodified -- the only change is which region mask gets used.
`generate_fine_reference` always builds the GROOVE region mask
internally (`build_region_mask`, hardcoded, not parameterized) and
stores it at `fine_ref["region_mask_fine_torch"]`; since
`evaluate_u_only_checkpoint_against_fine_reference` only ever READS
that dict entry (confirmed by reading its source, not assumed), simply
loading the already-cached fine reference (zero new FEM solves) and then
OVERWRITING that one entry with the far-field mask
(`verify_far_region_discretization_error.py`'s own
`build_far_region_mask`, same center/radius already used and reported
for the far-region FEM-vs-FEM control check) makes every checkpoint
evaluation run at the far-field region instead, with no other code
path touched.

Usage:
  python -m omar_pfem.evaluate_b3_checkpoints_far_region \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \\
      --results_out /scratch/gole3157/pfem_run/b3_far_region_checkpoints.json
"""
import argparse
import json
import os

import numpy as np
import torch

from omar_pfem.data.data_generate_B3_dataset import R_GRADING, GROOVE_DEPTH, GROOVE_HALF_WIDTH
from omar_pfem.data.mesh_convergence_B3 import R_IN0, LZ
from omar_pfem.data.data_generate_B3 import groove_radius_of_curvature
from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, evaluate_u_only_checkpoint_against_fine_reference, FINE_RESOLUTION_STRESS
from omar_pfem.verify_far_region_discretization_error import build_far_region_mask
from omar_pfem.run_b3_rabczuk_reeval_v2_local import U_ONLY_NAMES


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True,
                         help="same --out_json the real re-evaluation run used, so this "
                              "script reuses its cached 950,400-element reference")
    parser.add_argument("--results_out", type=str, default=None,
                         help="default: <out_json>.far_region_checkpoints.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"device: {device}")

    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    results_out = args.results_out or (args.out_json + ".far_region_checkpoints.json")

    print(f"\n{'=' * 90}\nLoading cached 950,400-element reference (no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, args.n_samples, device, dtype,
                                        cache_path=fine_cache, require_region_mask=True)

    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    far_ref_point = np.array([R_IN0, 0.0, LZ / 4.0])
    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    far_mask_np = build_far_region_mask(Ntheta_f, Nr_f, Nz_f, R_GRADING, far_ref_point, region_radius)
    n_far = int(far_mask_np.sum())
    print(f"Swapping the groove region mask ({fine_ref['region_mask_fine_torch'].sum().item()} Gauss "
          f"points) for the FAR-field one ({n_far} Gauss points, center={far_ref_point.tolist()}, "
          f"radius={region_radius:.5f} -- same size, groove-free location already validated in "
          f"verify_far_region_discretization_error.py) -- no other part of fine_ref changes.")
    fine_ref["region_mask_fine_torch"] = torch.tensor(far_mask_np, device=device)

    results = {}
    if os.path.exists(results_out):
        with open(results_out) as f:
            results = json.load(f)
        print(f"Resuming from {results_out}: {list(results.keys())} already done.")

    for name, rel_path in U_ONLY_NAMES.items():
        if name in results:
            print(f"\n[skip, already done] {name}")
            continue
        ckpt = f"{R}/{rel_path}"
        print(f"\n{'=' * 90}\n=== [FAR region, vs. 950,400 el] {name} ===")
        if not os.path.exists(ckpt):
            print(f"  SKIPPED -- checkpoint not found at {ckpt}")
            continue
        results[name] = evaluate_u_only_checkpoint_against_fine_reference(ckpt, fine_ref, device, dtype)
        with open(results_out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 90}\n=== SUMMARY: every checkpoint, FAR-FIELD region vs. the 950,400-element reference ===")
    for name in U_ONLY_NAMES:
        if name not in results:
            continue
        s = results[name]["summary"]
        print(f"  {name:<65} disp={s['mean_disp_rel_l2_fine_combined']*100:6.2f}%  "
              f"region_field={s['mean_region_cauchy_field_rel_fine']*100:8.2f}%  "
              f"sigma_xx={s['mean_region_sigma_xx_rel_fine']*100:8.2f}%")
    print(f"\nResults in {results_out}")


if __name__ == "__main__":
    main()
