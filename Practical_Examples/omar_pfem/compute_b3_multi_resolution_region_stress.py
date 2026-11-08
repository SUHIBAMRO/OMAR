"""Per Prof. Rabczuk's 2026-11-04 email, point 4's explicit request:
"Can you please report the errors of all the other mesh sizes as well."
`compute_b3_multi_resolution_vs_950400.py` already solved and cached
336/1,320/3,360/18,200/79,464/201,780 elements vs. the 950,400-element
reference, but only for the 4 scalar/global QoIs (displacement, H1,
energy, reaction) -- it deliberately did NOT build the groove region
mask for those coarse solves (`require_region_mask=False`, needed at the
time to avoid a hard assertion on coarse meshes), so the per-component
groove-region stress breakdown (the one Section 6/7 of the report is
actually about) was never computed for these 6 resolutions, only for
6,840 and 43,400.

This script closes that gap WITHOUT any new FEM solving: the already-
cached per-sample FEM fields for all 6 resolutions (and the 950,400
reference) are loaded straight from `generate_fine_reference`'s own
cache files (same `--out_json` as the original run), and run through
`compute_one_resolution_vs_fine` -- the exact, already-validated
function used for the 6,840/43,400/839,040-vs-950,400 numbers already in
the report, not a re-implementation. That function only ever needs the
FINE reference's own region mask (already built, since
FINE_RESOLUTION_STRESS was always solved with require_region_mask=True);
the coarser side only needs its own raw (u, E, nu) fields, which the
existing caches already have regardless of require_region_mask. Pure
post-processing: no GPU time beyond loading tensors and a handful of
interpolation/stress evaluations, safe to run alongside any other job
using the GPU.

Usage:
  python -m omar_pfem.compute_b3_multi_resolution_region_stress \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json
  (same --run_dir/--out_json as compute_b3_multi_resolution_vs_950400.py's
  own run, so it finds the exact same cache files on disk)
"""
import argparse
import json

import torch

from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, FINE_RESOLUTION_STRESS
from omar_pfem.evaluate_B3_fem_discretization_error import compute_one_resolution_vs_fine
from omar_pfem.compute_b3_multi_resolution_vs_950400 import CANDIDATE_RESOLUTIONS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True,
                         help="SAME --out_json compute_b3_multi_resolution_vs_950400.py used, "
                              "so this script finds its exact cache files on disk")
    parser.add_argument("--resolutions", type=str, nargs="+", default=list(CANDIDATE_RESOLUTIONS))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results_out", type=str, default=None,
                         help="default: <out_json>.multi_resolution_region_stress.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"device: {device}")

    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    N_SAMPLES = args.n_samples
    results_out = args.results_out or (args.out_json + ".multi_resolution_region_stress.json")

    print(f"\n{'=' * 90}\nLoading cached 950,400-element reference (no new solves) ...")
    fine_cache = args.out_json + ".fine_950400.cache.pt"
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, N_SAMPLES, device, dtype,
                                        cache_path=fine_cache, require_region_mask=True)

    results = {}
    try:
        with open(results_out) as f:
            results = json.load(f)
        print(f"Resuming from {results_out}: {list(results.keys())} already done.")
    except FileNotFoundError:
        pass

    for label in args.resolutions:
        if label not in CANDIDATE_RESOLUTIONS:
            print(f"  SKIPPED -- unknown resolution label {label!r}, known: {list(CANDIDATE_RESOLUTIONS)}")
            continue
        if label in results:
            print(f"\n[skip, already done] {label} elements")
            continue
        Ntheta, Nr, Nz = CANDIDATE_RESOLUTIONS[label]
        n_elem = (Ntheta - 1) * (Nr - 1) * (Nz - 1)
        cache_path = args.out_json + f".res_{label}.cache.pt"
        print(f"\n{'=' * 90}\n=== Loading cached {label} elements ({(Ntheta, Nr, Nz)}, "
              f"n_elem={n_elem}) -- must already exist from the earlier "
              f"compute_b3_multi_resolution_vs_950400.py run, no new solve ===")
        ref = generate_fine_reference(HOLDOUT, (Ntheta, Nr, Nz), N_SAMPLES, device, dtype,
                                       cache_path=cache_path, require_region_mask=False)
        result = compute_one_resolution_vs_fine(ref, fine_ref, device, dtype, tag=f"{label}-element")
        s = result["summary"]
        results[label] = result
        print(f"  {label} el: region_cauchy_field={s['mean_region_cauchy_field_rel_vs_950400']*100:.2f}%  "
              f"region_avg_sigma_xx={s['mean_region_avg_sigma_xx_rel_vs_950400']*100:.2f}%")
        for name in ("xx", "yy", "zz", "xy", "xz", "yz"):
            print(f"    sigma_{name}={s[f'mean_region_sigma_{name}_rel_vs_950400']*100:.2f}%")
        with open(results_out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 90}\n=== SUMMARY: groove-region stress, every tested resolution vs. 950,400 ===")
    for label in args.resolutions:
        if label not in results:
            continue
        s = results[label]["summary"]
        print(f"  {label:>7} el: field={s['mean_region_cauchy_field_rel_vs_950400']*100:8.2f}%  "
              f"xx={s['mean_region_sigma_xx_rel_vs_950400']*100:8.2f}%  "
              f"yy={s['mean_region_sigma_yy_rel_vs_950400']*100:8.2f}%  "
              f"zz={s['mean_region_sigma_zz_rel_vs_950400']*100:8.2f}%  "
              f"xy={s['mean_region_sigma_xy_rel_vs_950400']*100:8.2f}%  "
              f"xz={s['mean_region_sigma_xz_rel_vs_950400']*100:8.2f}%  "
              f"yz={s['mean_region_sigma_yz_rel_vs_950400']*100:8.2f}%")
    print(f"\nResults in {results_out}")


if __name__ == "__main__":
    main()
