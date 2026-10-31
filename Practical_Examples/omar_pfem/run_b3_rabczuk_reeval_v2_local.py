"""Plain-script (non-Colab) version of B3_Rabczuk_Reeval_v2.ipynb, for
running directly on the TU Ilmenau HPC cluster (or any machine with a
local checkout of this repo and the checkpoints already on local disk)
instead of Colab + Google Drive mount.

Same logic as the notebook's own code cell: generate ONE common
950,400-element reference and evaluate every existing B3 checkpoint
against it for every QoI (Prof. Rabczuk's second email), plus the
FEM-vs-FEM discretization-error report (6,840- and 43,400-element
solutions vs. the same 950,400-element reference). No retraining --
evaluation of existing checkpoints only.

FULLY RESUMABLE (added 2026-10-31, after a real ~10h run on an
interactive HPC node was lost with NOTHING saved anywhere when the
node's session time limit killed it): every expensive step -- each
fine-mesh FEM solve, each checkpoint's evaluation, the discretization-
error report -- writes its own progress to disk immediately, and is
skipped on a re-run if already done. Re-running this exact command
after any interruption (killed session, crashed job, network drop)
picks up from wherever it left off, at the granularity of one FEM
solve or one checkpoint, never the whole run.

Usage (from the HPC, inside the `omar` conda env, from the
Practical_Examples/ directory so `omar_pfem` is importable):
  python -m omar_pfem.run_b3_rabczuk_reeval_v2_local \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json
"""
import argparse
import json
import os
import time

import torch

from omar_pfem.evaluate_B3_qois_finer import (
    generate_fine_reference, evaluate_u_only_checkpoint_against_fine_reference,
    FINE_RESOLUTION_STRESS, FINE_RESOLUTION_DISP_ENERGY_REACTION,
)
from omar_pfem.evaluate_B3_mixed_stress_finer import evaluate_mixed_stress_checkpoint_against_fine_reference
from omar_pfem.evaluate_B3_fem_discretization_error import load_coarse_reference, compute_fem_discretization_errors

U_ONLY_NAMES = {
    "PI baseline": "b3_training_normalized/checkpoint_50000.pt",
    "Items 2/3, attempt 1 (equilibrium, w=1.0)": "b3_equilibrium_augmented/checkpoint_50000.pt",
    "Items 2/3, attempt 2 (equilibrium, w=0.12)": "b3_equilibrium_augmented_v2/checkpoint_50000.pt",
    "Item 4, DD rel_l2 @ N=50": "b3_data_driven_rel_l2_N50/model_best.pt",
    "Item 4, DD rel_l2 @ N=200 (break-even baseline)": "b3_data_driven_rel_l2/model_best.pt",
    "Item 4, DD rel_l2 @ N=400": "b3_data_driven_rel_l2_N400/model_best.pt",
    "Item 4, DD sobolev": "b3_data_driven_sobolev/model_best.pt",
    "Item 4, DD hybrid": "b3_data_driven_hybrid/model_best.pt",
}
MIXED_STRESS_NAMES = {
    "Item 1, attempt 1 (mixed-stress, scale=500)": "b3_mixed_stress_training/checkpoint_50000.pt",
    "Item 1, attempt 5 (mixed-stress, local-refine)": "b3_mixed_stress_local_refine/checkpoint_50000.pt",
    "Item 1, attempt 6 (mixed-stress, relative + local-refine)":
        "b3_mixed_stress_local_refine_relative/checkpoint_50000.pt",
}


def _load_partial(out_json):
    if os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)
    return {"checkpoints": {}, "fem_discretization_error": None}


def _save_partial(out_json, partial):
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(partial, f, indent=2)
    os.replace(tmp, out_json)  # atomic -- a crash mid-write never corrupts the real output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_dir", type=str, required=True,
                         help="local directory holding the checkpoint subfolders and "
                              "b3_dataset_clean_holdout/dataset.h5 (e.g. /scratch/<login>/pfem_run)")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    R = args.run_dir
    HOLDOUT = f"{R}/b3_dataset_clean_holdout/dataset.h5"
    N_SAMPLES = args.n_samples
    _started = time.time()

    partial = _load_partial(args.out_json)
    if partial["checkpoints"]:
        print(f"Resuming from {args.out_json}: {len(partial['checkpoints'])} checkpoint(s) already done.")

    fine_cache = args.out_json + ".fine_950400.cache.pt"
    print(f"\n{'=' * 90}")
    print(f"=== Generating the ONE common 950,400-element ({FINE_RESOLUTION_STRESS}) reference, {N_SAMPLES} samples ===")
    t0 = time.time()
    fine_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_STRESS, N_SAMPLES, device, dtype,
                                        cache_path=fine_cache)
    print(f"Fine reference ready in {time.time() - t0:.1f}s.")

    for name, rel_path in U_ONLY_NAMES.items():
        if name in partial["checkpoints"]:
            print(f"\n[skip, already done] {name}")
            continue
        ckpt = f"{R}/{rel_path}"
        print(f"\n{'=' * 90}\n=== [vs. 950,400 el] {name} ===")
        if not os.path.exists(ckpt):
            print(f"  SKIPPED -- checkpoint not found at {ckpt}")
            continue
        partial["checkpoints"][name] = evaluate_u_only_checkpoint_against_fine_reference(ckpt, fine_ref, device, dtype)
        _save_partial(args.out_json, partial)

    for name, rel_path in MIXED_STRESS_NAMES.items():
        if name in partial["checkpoints"]:
            print(f"\n[skip, already done] {name}")
            continue
        ckpt = f"{R}/{rel_path}"
        print(f"\n{'=' * 90}\n=== [vs. 950,400 el, implicit + direct] {name} ===")
        if not os.path.exists(ckpt):
            print(f"  SKIPPED -- checkpoint not found at {ckpt}")
            continue
        partial["checkpoints"][name] = evaluate_mixed_stress_checkpoint_against_fine_reference(ckpt, fine_ref, device, dtype)
        _save_partial(args.out_json, partial)

    if partial["fem_discretization_error"] is None:
        mid_cache = args.out_json + ".fine_43400.cache.pt"
        print(f"\n{'=' * 90}")
        print(f"=== Generating the 43,400-element ({FINE_RESOLUTION_DISP_ENERGY_REACTION}) solution, {N_SAMPLES} samples ===")
        print("=== (needed only for the FEM-vs-FEM discretization-error report below) ===")
        t0 = time.time()
        mid_ref = generate_fine_reference(HOLDOUT, FINE_RESOLUTION_DISP_ENERGY_REACTION, N_SAMPLES, device, dtype,
                                           cache_path=mid_cache)
        print(f"43,400-element solution ready in {time.time() - t0:.1f}s.")

        print(f"\n{'=' * 90}")
        print("=== Loading the already-solved 6,840-element solution from dataset.h5 (no new FEM solve) ===")
        coarse_ref = load_coarse_reference(HOLDOUT, N_SAMPLES, device, dtype)

        print(f"\n{'=' * 90}")
        print("=== FEM-vs-FEM discretization error: 6,840-el and 43,400-el solutions vs. the 950,400-el reference ===")
        print("=== (NO neural network involved -- isolates pure mesh-discretization error) ===")
        partial["fem_discretization_error"] = compute_fem_discretization_errors(coarse_ref, mid_ref, fine_ref, device, dtype)
        _save_partial(args.out_json, partial)
    else:
        print(f"\n[skip, already done] FEM-vs-FEM discretization error")

    results = partial["checkpoints"]
    disc = partial["fem_discretization_error"]

    print(f"\n{'=' * 90}")
    print("=== SUMMARY: every checkpoint vs. the single 950,400-element reference ===")
    for name, r in results.items():
        s = r["summary"]
        if "mean_region_cauchy_field_rel_fine" in s:
            print(f"  {name}:")
            print(f"    disp_rel_l2 combined:        {s['mean_disp_rel_l2_fine_combined']:.4f}")
            print(f"    H1_seminorm_rel:             {s['mean_H1_seminorm_rel_fine']:.4f}")
            print(f"    energy rel err:              mean={s['mean_rel_err_energy_fine']:.4f}")
            print(f"    reaction_moment_y rel err:   mean={s['mean_rel_err_reaction_moment_y_fine']:.4f}")
            print(f"    region_cauchy_field_rel:     mean={s['mean_region_cauchy_field_rel_fine']:.4f}")
            print(f"    region sigma components:    xx={s['mean_region_sigma_xx_rel_fine']:.4f}  "
                  f"yy={s['mean_region_sigma_yy_rel_fine']:.4f}  zz={s['mean_region_sigma_zz_rel_fine']:.4f}  "
                  f"xy={s['mean_region_sigma_xy_rel_fine']:.4f}  xz={s['mean_region_sigma_xz_rel_fine']:.4f}  "
                  f"yz={s['mean_region_sigma_yz_rel_fine']:.4f}")
        else:
            print(f"  {name}:")
            print(f"    H1_seminorm_rel:                     {s['mean_H1_seminorm_rel_fine']:.4f}")
            print(f"    region_cauchy_field_rel IMPLICIT:    {s['mean_region_cauchy_field_rel_implicit_fine']:.4f}")
            print(f"    region_cauchy_field_rel DIRECT:      {s['mean_region_cauchy_field_rel_direct_fine']:.4f}")

    print(f"\n{'=' * 90}")
    print("=== SUMMARY: FEM-vs-FEM discretization error (no NN) vs. the 950,400-element reference ===")
    for tag, label in (("vs_6840", "6,840-element"), ("vs_43400", "43,400-element")):
        s = disc[tag]["summary"]
        print(f"  {label} solution:")
        print(f"    disp_rel_l2:               {s['mean_disp_rel_l2_vs_950400']:.4f}")
        print(f"    H1_seminorm_rel:           {s['mean_H1_seminorm_rel_vs_950400']:.4f}")
        print(f"    energy_rel:                {s['mean_energy_rel_vs_950400']:.4f}")
        print(f"    reaction_moment_y_rel:     {s['mean_reaction_moment_y_rel_vs_950400']:.4f}")
        print(f"    region_cauchy_field_rel:   {s['mean_region_cauchy_field_rel_vs_950400']:.4f}")

    print(f"\n{'=' * 90}")
    print(f"Total wall-clock THIS invocation: {time.time() - _started:.1f}s")
    print(f"Results in {args.out_json} (already saved incrementally throughout the run)")


if __name__ == "__main__":
    main()
