"""Professor Rabczuk's 2026-11-12 email: evaluate PI and the DD
variants against the now-converged QUADRATIC (Q20) groove reference,
report the quadratic meshes' DOF, and include the SCF.

Mirrors evaluate_b3_checkpoints_far_region.py's own established
pattern exactly, swapping generate_fine_reference (linear, 950,400-
element) for generate_fine_reference_quadratic (Q20, resolution given
on the command line) -- same U_ONLY_NAMES checkpoint list, same
resumable per-checkpoint JSON caching, same
evaluate_u_only_checkpoint_against_fine_reference call, completely
unmodified (it is already generic over element type, confirmed
directly in evaluate_B3_qois_quadratic.py's own docstring).

SCF is computed separately, using QUADRATIC (Q20) elements on BOTH
sides (groove and no-groove counterfactual) via
solve_quadratic_hex_groove_stress.solve_and_measure_groove_region_quadratic,
at a representative, homogeneous-material verification case -- the
SAME methodology already used and reported earlier for the matched-
element-type SCF, not per real dataset sample (keeping this run's
scope bounded; a per-sample SCF would need a groove_depth-override on
the heterogeneous-material solve path, not built here). NOTE: an
earlier version of this script mistakenly reused compute_b3_scf.
compute_scf, which computes the groove/nominal pair with LINEAR
elements -- inconsistent with the Q20 reference used for the
checkpoint evaluation above. Fixed to keep element type consistent
throughout this script's own SCF number.

Usage:
  python -m omar_pfem.evaluate_b3_checkpoints_vs_quadratic \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --resolution 21 18 31 \\
      --out_json /scratch/gole3157/pfem_run/b3_quadratic_reference.json \\
      --results_out /scratch/gole3157/pfem_run/b3_quadratic_checkpoints.json
"""
import argparse
import json
import os

import torch

import numpy as np

from omar_pfem.evaluate_B3_qois_quadratic import generate_fine_reference_quadratic
from omar_pfem.evaluate_B3_qois_finer import evaluate_u_only_checkpoint_against_fine_reference
from omar_pfem.run_b3_rabczuk_reeval_v2_local import U_ONLY_NAMES
from omar_pfem.data.solve_quadratic_hex_groove_stress import (
    solve_and_measure_groove_region_quadratic, R_IN0, LZ)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--resolution", type=int, nargs=3, required=True,
                         help="Ntheta Nr Nz for the Q20 reference mesh")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--results_out", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    results_out = args.results_out or (args.out_json + ".quadratic_checkpoints.json")
    Ntheta, Nr, Nz = args.resolution

    print(f"\n{'=' * 90}\nBuilding/loading the Q20 (quadratic) reference "
          f"({Ntheta}x{Nr}x{Nz}), {args.n_samples} samples ...")
    fine_cache = args.out_json + f".fine_q20_{Ntheta}_{Nr}_{Nz}.cache.pt"
    fine_ref = generate_fine_reference_quadratic(
        HOLDOUT, (Ntheta, Nr, Nz), args.n_samples, device, dtype, cache_path=fine_cache)

    geom = fine_ref["geom_fine"]
    print(f"\nQ20 reference mesh: {geom['n_elements']:,} elements, {geom['n_nodes']:,} nodes, "
          f"DOF={geom['dof']:,}  |  {fine_ref['n_region_fine']} Gauss points in the groove region")

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
        print(f"\n{'=' * 90}\n=== [Q20 reference, {Ntheta}x{Nr}x{Nz}] {name} ===")
        if not os.path.exists(ckpt):
            print(f"  SKIPPED -- checkpoint not found at {ckpt}")
            continue
        results[name] = evaluate_u_only_checkpoint_against_fine_reference(ckpt, fine_ref, device, dtype)
        with open(results_out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 90}\n=== SUMMARY: every checkpoint vs. the Q20 ({Ntheta}x{Nr}x{Nz}) reference ===")
    for name in U_ONLY_NAMES:
        if name not in results:
            continue
        s = results[name]["summary"]
        print(f"  {name:<65} disp={s['mean_disp_rel_l2_fine_combined']*100:6.2f}%  "
              f"region_field={s['mean_region_cauchy_field_rel_fine']*100:8.2f}%  "
              f"sigma_xx={s['mean_region_sigma_xx_rel_fine']*100:8.2f}%")

    print(f"\n{'=' * 90}\nComputing SCF with QUADRATIC (Q20) elements on BOTH sides "
          f"(representative homogeneous-material case, same resolution as the reference "
          f"above, no local z-clustering) ...")
    groove_q20 = solve_and_measure_groove_region_quadratic(
        Ntheta, Nr, Nz, device=device)
    same_point = np.array([R_IN0, 0.0, LZ / 2.0])
    nominal_q20 = solve_and_measure_groove_region_quadratic(
        Ntheta, Nr, Nz, device=device, groove_depth=0.0, ref_point=same_point)
    scf_max = groove_q20["region_max_sigma_xx"] / nominal_q20["region_avg_sigma_xx"]
    scf_avg = groove_q20["region_avg_sigma_xx"] / nominal_q20["region_avg_sigma_xx"]
    scf_result = {
        "groove": groove_q20, "nominal_no_groove": nominal_q20,
        "SCF_max_sigma_xx_over_nominal_avg": scf_max, "SCF_avg_sigma_xx": scf_avg,
    }
    print(f"  groove (Q20) avg|sigma_xx|={groove_q20['region_avg_sigma_xx']:.4f}  "
          f"max|sigma_xx|={groove_q20['region_max_sigma_xx']:.4f}")
    print(f"  nominal(no groove, Q20) avg|sigma_xx|={nominal_q20['region_avg_sigma_xx']:.4f}")
    print(f"  SCF (max groove / nominal avg, both Q20) = {scf_max:.3f}")

    combined = {
        "resolution": [Ntheta, Nr, Nz], "n_elements": geom["n_elements"], "n_nodes": geom["n_nodes"],
        "dof": geom["dof"], "n_region_gauss": fine_ref["n_region_fine"],
        "checkpoints": results, "scf": scf_result,
    }
    with open(args.out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nFull combined results written to {args.out_json}")
    print(f"Per-checkpoint results in {results_out}")


if __name__ == "__main__":
    main()
