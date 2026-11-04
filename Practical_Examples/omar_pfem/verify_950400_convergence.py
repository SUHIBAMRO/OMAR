"""Per Prof. Rabczuk's 2026-11-04 email (Point 4's remaining sub-item):
the project's own mesh-convergence history already checked that the
950,400-element reference ((101,100,97)) is converged against the
839,040-element row ((97,96,93)), the ladder's second-finest rung --
but that OLD check used a single deterministic case and an OLDER
formula version of the region-stress metric, not the current 10-random-
held-out-sample methodology and current region-stress formula
(cauchy_field_rel_from_full / sigma_components_rel_from_full) used
everywhere else in this re-evaluation. That old result should not be
assumed to carry over automatically. This script re-checks it properly:
solves 839,040 elements for real (genuine new FEM solve, 10 held-out
samples, cacheable/resumable exactly like every other reference in this
project), and compares it against the already-cached 950,400-element
reference using the EXACT SAME, already-validated function
(evaluate_B3_fem_discretization_error.compute_one_resolution_vs_fine)
used for the 6,840-/43,400-vs-950,400 discretization-error numbers
already in the report -- not a re-implementation, so there is no risk of
a silently different formula giving a falsely reassuring (or falsely
alarming) answer.

If the 839,040-vs-950,400 errors (on every QoI, INCLUDING the
groove-region stress field) are small, 950,400 is confirmed converged
with the current methodology and can keep being used as the common
reference everywhere else in this re-evaluation. If they are large, the
950,400 reference itself would need to be reconsidered before trusting
anything computed against it.

Usage:
  python -m omar_pfem.verify_950400_convergence \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json
"""
import argparse
import json

import torch

from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, FINE_RESOLUTION_STRESS
from omar_pfem.evaluate_B3_fem_discretization_error import compute_one_resolution_vs_fine

# (97, 96, 93) -> 839,040 elements: the 950,400-element row's ((101,100,97))
# immediate neighbour in this project's own B3 groove mesh-convergence
# ladder (cell_b3_groove_gpu_mesh_convergence.py) -- its second-finest rung.
RESOLUTION_839040 = (97, 96, 93)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True,
                         help="same --out_json the real re-evaluation run used, so this "
                              "script reuses its cached 950,400-element reference")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results_out", type=str, default=None,
                         help="where to write this script's own JSON results "
                              "(default: <out_json>.convergence_839040_vs_950400.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"device: {device}")

    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    N_SAMPLES = args.n_samples
    results_out = args.results_out or (args.out_json + ".convergence_839040_vs_950400.json")

    print(f"\n{'=' * 90}\nLoading cached 950,400-element ((101, 100, 97)) reference (no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, N_SAMPLES, device, dtype,
                                        cache_path=fine_cache, require_region_mask=True)

    print(f"\n{'=' * 90}\nSolving 839,040 elements ((97, 96, 93)) for {N_SAMPLES} samples "
          f"(real new FEM solve, cacheable/resumable) ...")
    mid_cache = args.out_json + ".res_839040.cache.pt"
    mid_ref = generate_fine_reference(HOLDOUT, RESOLUTION_839040, N_SAMPLES, device, dtype,
                                       cache_path=mid_cache, require_region_mask=True)

    print(f"\n{'=' * 90}\n=== 839,040-element solution vs. the 950,400-element reference "
          f"(SAME methodology/samples/formula as every other QoI in this re-evaluation) ===")
    result = compute_one_resolution_vs_fine(mid_ref, fine_ref, device, dtype, tag="839,040-element")
    s = result["summary"]

    print(f"\n{'=' * 90}\n=== SUMMARY ===")
    print(f"  disp_rel_l2:              {s['mean_disp_rel_l2_vs_950400'] * 100:.4f}%")
    print(f"  H1_seminorm_rel:          {s['mean_H1_seminorm_rel_vs_950400'] * 100:.4f}%")
    print(f"  energy_rel:               {s['mean_energy_rel_vs_950400'] * 100:.4f}%")
    print(f"  reaction_moment_y_rel:    {s['mean_reaction_moment_y_rel_vs_950400'] * 100:.4f}%")
    print(f"  region_cauchy_field_rel:  {s['mean_region_cauchy_field_rel_vs_950400'] * 100:.4f}%  "
          f"(the groove-region stress metric itself -- this is the number that matters most "
          f"for deciding whether 950,400 is converged enough to trust for the groove-region "
          f"numbers already reported elsewhere)")
    print(f"  region_avg_sigma_xx_rel:  {s['mean_region_avg_sigma_xx_rel_vs_950400'] * 100:.4f}%")
    for name in ("xx", "yy", "zz", "xy", "xz", "yz"):
        print(f"  region_sigma_{name}_rel:      {s[f'mean_region_sigma_{name}_rel_vs_950400'] * 100:.4f}%")

    with open(results_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults in {results_out}")


if __name__ == "__main__":
    main()
