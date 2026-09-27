"""Finer-resolution region-stress evaluation for B3, to separate two
hypotheses `region_cauchy_field_rel`'s real result (71.77% mean, 68.45%
median on the 100-sample held-out set, at the 6,840-element TRAINING
resolution) could not distinguish on its own (see evaluate_B3_qois.py's
own docstring and this project's PROJECT_STATUS.md, 2026-09-27 entries):

  (a) an EVALUATION-resolution artifact: only 6 Gauss points fall in the
      fixed groove-neighborhood region at 6,840 elements, far below this
      project's own MIN_RELIABLE_N_P99=20 threshold -- maybe the true
      region-stress accuracy is fine, and 6 points is just too few to
      measure it well (for either the true field or the prediction).
  (b) a genuine, resolution-independent accuracy gap in the network's
      local stress prediction, unrelated to how finely it is sampled.

This script tests (a) vs (b) directly, for a SUBSET of the existing
100-sample clean held-out set, WITHOUT retraining: for each chosen
sample,
  1. take its already-generated (21,20,19) E_node/nu_node fields (from
     dataset.h5) and interpolate them onto a much finer mesh's node
     positions -- reusing mesh_convergence_B3.py's own
     `_field_interpolator` (RegularGridInterpolator, genuine multilinear
     interpolation, not nearest-node snapping -- the same machinery this
     project already uses to compare different resolutions for the SAME
     physical sample).
  2. solve a REAL new FEM problem at the finer mesh with this
     interpolated material field and the SAME phi (rocking angle) --
     reusing data_generate_B3_dataset.solve_one_sample directly, no new
     solver code.
  3. query the SAME trained network (checkpoint_50000.pt, unchanged) at
     the finer mesh's own node positions, with the interpolated
     material field as its function input -- a genuine zero-shot
     resolution-generalization query, the same kind of cross-resolution
     evaluation this project's B1/B2 work already relies on.
  4. compute region_cauchy_field_rel between the finer FEM truth and the
     finer network prediction, using the SAME resolution-agnostic
     function evaluate_B3_qois.py already has, but now at a resolution
     with far more Gauss points in the region (e.g. 36 at (41,36,32),
     vs. 6 at production resolution) -- reused unmodified, not
     reimplemented.

IMPORTANT CAVEAT (present by construction, not a bug): interpolating the
coarse-grid E/nu SAMPLE onto a finer mesh does not recover a genuinely
finer-scale material realization -- it is a piecewise-multilinear
reconstruction of the SAME coarse-grid values, smoother between the
original grid points than a real finer GRF draw would be. This is the
right test for exactly the question asked here (does refining the MESH
and the STRESS-EVALUATION sampling change the measured accuracy for a
FIXED, unchanged physical field), not a claim that this reproduces what
training on a finer physical dataset would show.

Usage:
  python -m omar_pfem.evaluate_B3_qois_finer \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_training_normalized/checkpoint_50000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_training_normalized/qois_finer_resolution.json \
      --fine_resolution 41 36 32 \
      --n_samples 10
"""
import argparse
import json

import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import (
    build_fixed_geometry, apply_dirichlet_b3, build_model,
)
from omar_pfem.train_B1 import install_input_norm_for_checkpoint, _apply_input_norm
from omar_pfem.data.data_generate_B3_dataset import (
    DEFAULT_RESOLUTION, R_GRADING, solve_one_sample,
)
from omar_pfem.data.mesh_convergence_B3 import MIN_RELIABLE_N_P99, _theta_t_axes, _field_interpolator
from omar_pfem.evaluate_B3_qois import (
    build_region_mask, compute_region_cauchy_field_rel, compute_region_sigma_xx,
)


def interpolate_node_field_to_finer_mesh(field_coarse, Ntheta_c, Nr_c, Nz_c,
                                          Ntheta_f, Nr_f, Nz_f, r_grading):
    """field_coarse: (n_nodes_coarse,) node-ordered as index =
    k*(Ntheta*Nr) + j*Nr + i (matches generate_grid_hex8_bushing's own
    convention, and build_fixed_geometry's node ordering). Returns
    (n_nodes_fine,) interpolated onto the finer mesh's own node
    positions, using the exact same query-point construction
    mesh_convergence_B3.py's `compare_case_to_reference` already
    verified for interpolating displacement fields across resolutions."""
    thetas_c, ts_c, zs_c = _theta_t_axes(Ntheta_c, Nr_c, Nz_c, r_grading)
    interp_fn = _field_interpolator(field_coarse.reshape(-1, 1), Ntheta_c, Nr_c, Nz_c,
                                     thetas_c, ts_c, zs_c, 1)

    from omar_pfem.data.mesh_convergence_B3 import LZ
    thetas_f = np.linspace(0.0, np.pi, Ntheta_f)
    ts_f = np.linspace(0.0, 1.0, Nr_f) ** r_grading
    zs_f = np.linspace(0.0, LZ, Nz_f)
    TH, T, Z = np.meshgrid(thetas_f, ts_f, zs_f, indexing="ij")
    theta_q = np.transpose(TH, (2, 0, 1)).ravel()
    t_q = np.transpose(T, (2, 0, 1)).ravel()
    z_q = np.transpose(Z, (2, 0, 1)).ravel()

    field_fine = interp_fn(theta_q, t_q, z_q).reshape(-1)
    return field_fine


def main():
    parser = argparse.ArgumentParser(
        "Finer-resolution region-stress evaluation for B3 (evaluation-resolution "
        "artifact vs. genuine network accuracy gap).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--fine_resolution", type=int, nargs=3, default=[41, 36, 32],
                         metavar=("NTHETA", "NR", "NZ"))
    parser.add_argument("--n_samples", type=int, default=10,
                         help="how many of the held-out samples to re-solve at the finer resolution")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta_c, Nr_c, Nz_c = DEFAULT_RESOLUTION
    Ntheta_f, Nr_f, Nz_f = args.fine_resolution
    n_elem_f = (Ntheta_f - 1) * (Nr_f - 1) * (Nz_f - 1)
    print(f"Coarse (training) resolution: {(Ntheta_c, Nr_c, Nz_c)}")
    print(f"Fine (evaluation-only) resolution: {(Ntheta_f, Nr_f, Nz_f)} -- {n_elem_f:,} elements")

    region_mask_fine_np = build_region_mask(Ntheta_f, Nr_f, Nz_f, R_GRADING)
    n_region_fine = int(region_mask_fine_np.sum())
    print(f"  fine-mesh fixed region: {n_region_fine} Gauss points "
          f"(reliable p99 needs >= {MIN_RELIABLE_N_P99}) -- vs. 6 at production resolution")
    assert n_region_fine >= MIN_RELIABLE_N_P99, (
        f"chosen --fine_resolution only gives {n_region_fine} region Gauss points, "
        f"below MIN_RELIABLE_N_P99={MIN_RELIABLE_N_P99} -- pick a finer --fine_resolution")

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node_all = h5["E_node"][:]
        nu_node_all = h5["nu_node"][:]
        phi_all = h5["phi"][:]
    n_available = E_node_all.shape[0]
    n_samples = min(args.n_samples, n_available)
    print(f"  {n_available} FEM samples available, evaluating {n_samples} of them at the finer resolution")

    geom_fine = build_fixed_geometry(Ntheta_f, Nr_f, Nz_f, device, dtype=dtype)
    print(f"  fine geometry: {geom_fine['n_elements']} elements, {geom_fine['n_nodes']} nodes")
    region_mask_fine_torch = torch.tensor(region_mask_fine_np, device=device)

    rows = []
    for idx in range(n_samples):
        E_c = E_node_all[idx].astype(np.float64)
        nu_c = nu_node_all[idx].astype(np.float64)
        phi = float(phi_all[idx])

        E_f = interpolate_node_field_to_finer_mesh(E_c, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING)
        nu_f = interpolate_node_field_to_finer_mesh(nu_c, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING)

        r = solve_one_sample(Ntheta_f, Nr_f, Nz_f, E_f, nu_f, phi, device=device, dtype=dtype)
        u_true_f = torch.tensor(r["u"], dtype=dtype, device=device).unsqueeze(0)  # (1, n_nodes_f, 3)

        E_f_t = torch.tensor(E_f, dtype=dtype, device=device).unsqueeze(0)
        nu_f_t = torch.tensor(nu_f, dtype=dtype, device=device).unsqueeze(0)
        phi_t = torch.tensor([phi], dtype=dtype, device=device)

        with torch.no_grad():
            fun_material = torch.stack(
                [E_f_t, nu_f_t, phi_t[:, None].expand(-1, geom_fine["n_nodes"])], dim=2)
            fun_material = _apply_input_norm(fun_material)
            xyz_batch = geom_fine["nodes"].unsqueeze(0)
            u_net = model(xyz_batch, fun_material)
            u_pred_f = apply_dirichlet_b3(u_net, geom_fine, phi_t)

        field_rel_fine = compute_region_cauchy_field_rel(
            u_true_f, u_pred_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)

        # Raw region_avg_sigma_xx (true and pred) at the FINE resolution --
        # reusing the already-verified compute_region_sigma_xx unmodified,
        # not new math. Needed to directly check, against this SAME
        # sample's own coarse-resolution (6-point) true value already in
        # qois_50000.json, whether the TRUE region-stress reference itself
        # shifts substantially between 6,840 and this finer resolution --
        # the key question for interpreting why region_cauchy_field_rel_fine
        # came out higher, not lower, than at production resolution.
        avg_true_f, _, _ = compute_region_sigma_xx(u_true_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
        avg_pred_f, _, _ = compute_region_sigma_xx(u_pred_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)

        rows.append({
            "sample_index": idx, "phi": phi,
            "force_rel_residual_fine_solve": r["force_rel_residual"],
            "region_cauchy_field_rel_fine": float(field_rel_fine[0].item()),
            "region_avg_sigma_xx_true_fine": float(avg_true_f[0].item()),
            "region_avg_sigma_xx_pred_fine": float(avg_pred_f[0].item()),
        })
        print(f"  [{idx + 1}/{n_samples}] fine solve force_rel_residual={r['force_rel_residual']:.2e}  "
              f"region_cauchy_field_rel_fine={field_rel_fine[0].item():.4f}  "
              f"region_avg_sigma_xx_true_fine={avg_true_f[0].item():.4f}  "
              f"region_avg_sigma_xx_pred_fine={avg_pred_f[0].item():.4f}")

    vals = np.array([row["region_cauchy_field_rel_fine"] for row in rows])
    vals_valid = vals[~np.isnan(vals)]
    summary = {
        "n_samples": n_samples, "n_region_gauss_points_fine": n_region_fine,
        "n_region_gauss_points_coarse": 6,
        "mean_region_cauchy_field_rel_fine": float(np.mean(vals_valid)) if len(vals_valid) else float("nan"),
        "median_region_cauchy_field_rel_fine": float(np.median(vals_valid)) if len(vals_valid) else float("nan"),
        "std_region_cauchy_field_rel_fine": float(np.std(vals_valid)) if len(vals_valid) else float("nan"),
    }
    print(f"\nSummary over {n_samples} samples at ({Ntheta_f},{Nr_f},{Nz_f}) "
          f"({n_region_fine} region Gauss points):")
    print(f"  mean={summary['mean_region_cauchy_field_rel_fine']:.4f}  "
          f"median={summary['median_region_cauchy_field_rel_fine']:.4f}  "
          f"std={summary['std_region_cauchy_field_rel_fine']:.4f}")
    print("  Compare against the SAME samples' coarse-resolution (6-point) "
          "region_cauchy_field_rel in qois_50000.json's per_sample rows.")

    result = {
        "checkpoint": args.checkpoint, "checkpoint_iter": ckpt["iter"],
        "dataset": args.dataset,
        "coarse_resolution": [Ntheta_c, Nr_c, Nz_c],
        "fine_resolution": [Ntheta_f, Nr_f, Nz_f],
        "summary": summary,
        "per_sample": rows,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
