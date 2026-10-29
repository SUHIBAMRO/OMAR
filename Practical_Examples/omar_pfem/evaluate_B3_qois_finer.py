"""Finer-resolution evaluation for B3: region-stress, H1 semi-norm
(added 2026-10-28), PLUS displacement/energy/reaction (added
2026-10-01, for the accuracy-matched FEM/VINO comparison Timon asked
for once the paper was being finished -- see PROJECT_STATUS.md's
2026-10-01 entry).

REFACTORED 2026-10-28 to separate the EXPENSIVE part (re-solving a
subset of the held-out set at a fine mesh -- a real new FEM problem
per sample) from the CHEAP part (querying a given checkpoint's network
at the fine mesh's own node positions and computing QoIs against that
shared ground truth). Needed because Professor Rabczuk's own review
caught that NONE of this project's follow-up-investigation results
(region stress for items 1/2-3, H1 semi-norm for all 7 checkpoints,
the break-even sweep) had ever been evaluated against an independent
fine reference -- all of them compared the network to the SAME
6,840-element mesh it was trained on, exactly the limitation he
flagged. Re-evaluating ~10 checkpoints each by re-running the OLD,
unrefactored main() would re-solve the same expensive fine-mesh FEM
problem once per checkpoint (10 samples x ~10 checkpoints x ~177s/
sample at the 950,400-element reference = ~5 hours, almost entirely
redundant FEM solving). The new `generate_fine_reference` function
solves the fine mesh ONCE per sample and returns a reusable reference
object; `evaluate_u_only_checkpoint_against_fine_reference` then
re-uses it for as many checkpoints as needed, each costing only one
cheap forward pass. The CLI `main()` below is kept for backward
compatibility (single-checkpoint usage) and is now a thin wrapper over
these two pieces.

Why displacement/energy/reaction needed adding here, not just reusing
evaluate_B3_qois.py: that script's own accuracy numbers are the
network's error against the SAME 6,840-element FEM mesh it was trained
on -- not against an independent finer reference. This project's own
established accuracy-matched convention (see the submitted paper's own
Section 7.4) uses an INDEPENDENT 43,400-element reference for
displacement/energy/reaction specifically (close to, but distinct
from, the 950,400-element reference the paper's own mesh-convergence
study established and uses for regional stress). Both are supported
here via --fine_resolution; there is no single "correct" fine
resolution for every QoI -- pick the one matching the QoI being
checked (950,400 = (101,100,97) for stress/H1, matching the paper's
own convergence-validated reference; 43,400 = (41,36,32) for
displacement/energy/reaction, matching the paper's own accuracy-matched
analysis -- both are real, independently-established references, not
interchangeable defaults).

This script tests two separate hypotheses `region_cauchy_field_rel`'s
real result could not distinguish on its own (see evaluate_B3_qois.py's
own docstring and PROJECT_STATUS.md's 2026-09-27 entries):

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
     interpolation, not nearest-node snapping).
  2. solve a REAL new FEM problem at the finer mesh with this
     interpolated material field and the SAME phi (rocking angle) --
     reusing data_generate_B3_dataset.solve_one_sample directly, no new
     solver code.
  3. query a trained network at the finer mesh's own node positions,
     with the interpolated material field as its function input -- a
     genuine zero-shot resolution-generalization query.
  4. compute region_cauchy_field_rel / H1 semi-norm between the finer
     FEM truth and the finer network prediction, using the SAME
     resolution-agnostic functions evaluate_B3_qois.py already has.

IMPORTANT CAVEAT (present by construction, not a bug): interpolating the
coarse-grid E/nu SAMPLE onto a finer mesh does not recover a genuinely
finer-scale material realization -- it is a piecewise-multilinear
reconstruction of the SAME coarse-grid values. This is the right test
for exactly the question asked here (does refining the MESH and the
STRESS-EVALUATION sampling change the measured accuracy for a FIXED,
unchanged physical field), not a claim that this reproduces what
training on a finer physical dataset would show.

Usage (single checkpoint, CLI, backward-compatible):
  python -m omar_pfem.evaluate_B3_qois_finer \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_training_normalized/checkpoint_50000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_training_normalized/qois_finer_resolution.json \
      --fine_resolution 101 100 97 \
      --n_samples 10

Usage (multi-checkpoint, reusing one fine reference -- see
B3_FineReference_Reevaluation.ipynb for the full worked example):
  ref = generate_fine_reference(dataset_path, (101, 100, 97), 10, device, dtype)
  for ckpt_path in checkpoints:
      result = evaluate_u_only_checkpoint_against_fine_reference(ckpt_path, ref, device, dtype)
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
    compute_region_sigma_full, compute_energy, compute_reaction, compute_H1_seminorm_rel,
    cauchy_field_rel_from_full, sigma_components_rel_from_full, STRESS_COMPONENTS,
)

# The two real, independently-established fine references this project
# uses -- see this module's own docstring for which QoI each is for.
FINE_RESOLUTION_STRESS = (101, 100, 97)  # 950,400 elements -- paper's own mesh-convergence reference
FINE_RESOLUTION_DISP_ENERGY_REACTION = (41, 36, 32)  # 43,400 elements -- paper's own accuracy-matched reference


def interpolate_node_field_to_finer_mesh(field_coarse, Ntheta_c, Nr_c, Nz_c,
                                          Ntheta_f, Nr_f, Nz_f, r_grading, n_components=1):
    """field_coarse: (n_nodes_coarse,) for n_components=1, or
    (n_nodes_coarse, n_components) for n_components>1 -- node-ordered as
    index = k*(Ntheta*Nr) + j*Nr + i (matches generate_grid_hex8_bushing's
    own convention, and build_fixed_geometry's node ordering). Returns
    (n_nodes_fine,) or (n_nodes_fine, n_components) interpolated onto the
    finer mesh's own node positions, using the exact same query-point
    construction mesh_convergence_B3.py's `compare_to_reference` already
    verified for interpolating displacement fields across resolutions
    (that function calls this same `_field_interpolator` with
    n_components=3 for its own u-field comparison)."""
    thetas_c, ts_c, zs_c = _theta_t_axes(Ntheta_c, Nr_c, Nz_c, r_grading)
    interp_fn = _field_interpolator(field_coarse.reshape(-1, n_components), Ntheta_c, Nr_c, Nz_c,
                                     thetas_c, ts_c, zs_c, n_components)

    from omar_pfem.data.mesh_convergence_B3 import LZ
    thetas_f = np.linspace(0.0, np.pi, Ntheta_f)
    ts_f = np.linspace(0.0, 1.0, Nr_f) ** r_grading
    zs_f = np.linspace(0.0, LZ, Nz_f)
    TH, T, Z = np.meshgrid(thetas_f, ts_f, zs_f, indexing="ij")
    theta_q = np.transpose(TH, (2, 0, 1)).ravel()
    t_q = np.transpose(T, (2, 0, 1)).ravel()
    z_q = np.transpose(Z, (2, 0, 1)).ravel()

    field_fine = interp_fn(theta_q, t_q, z_q)
    return field_fine.reshape(-1) if n_components == 1 else field_fine.reshape(-1, n_components)


def generate_fine_reference(dataset_path, fine_resolution, n_samples, device, dtype=torch.float64,
                             verbose=True):
    """THE EXPENSIVE STEP, done ONCE regardless of how many checkpoints
    will be evaluated against it: interpolates each of the first
    n_samples held-out samples' E/nu fields onto the fine mesh and
    solves a REAL new FEM problem there. Returns a dict with the fine
    geometry (built once), the region mask, and a list of per-sample
    reference dicts (E_f_t, nu_f_t, phi_t, u_true_f, plus the FEM
    solve's own diagnostics) -- everything a cheap per-checkpoint
    evaluation needs, with no further FEM solving."""
    Ntheta_c, Nr_c, Nz_c = DEFAULT_RESOLUTION
    Ntheta_f, Nr_f, Nz_f = fine_resolution
    n_elem_f = (Ntheta_f - 1) * (Nr_f - 1) * (Nz_f - 1)
    if verbose:
        print(f"Coarse (training) resolution: {(Ntheta_c, Nr_c, Nz_c)}")
        print(f"Fine (evaluation-only) resolution: {(Ntheta_f, Nr_f, Nz_f)} -- {n_elem_f:,} elements")

    region_mask_fine_np = build_region_mask(Ntheta_f, Nr_f, Nz_f, R_GRADING)
    n_region_fine = int(region_mask_fine_np.sum())
    if verbose:
        print(f"  fine-mesh fixed region: {n_region_fine} Gauss points "
              f"(reliable p99 needs >= {MIN_RELIABLE_N_P99}) -- vs. 6 at production resolution")
    assert n_region_fine >= MIN_RELIABLE_N_P99, (
        f"chosen fine_resolution only gives {n_region_fine} region Gauss points, "
        f"below MIN_RELIABLE_N_P99={MIN_RELIABLE_N_P99} -- pick a finer resolution")

    with h5py.File(dataset_path, "r") as h5:
        E_node_all = h5["E_node"][:]
        nu_node_all = h5["nu_node"][:]
        phi_all = h5["phi"][:]
    n_available = E_node_all.shape[0]
    n_samples = min(n_samples, n_available)
    if verbose:
        print(f"  {n_available} FEM samples available, re-solving {n_samples} of them at the finer resolution")

    geom_fine = build_fixed_geometry(Ntheta_f, Nr_f, Nz_f, device, dtype=dtype)
    if verbose:
        print(f"  fine geometry: {geom_fine['n_elements']} elements, {geom_fine['n_nodes']} nodes")
    region_mask_fine_torch = torch.tensor(region_mask_fine_np, device=device)

    samples = []
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

        samples.append({
            "sample_index": idx, "phi": phi,
            "E_f_t": E_f_t, "nu_f_t": nu_f_t, "phi_t": phi_t, "u_true_f": u_true_f,
            "force_rel_residual": r["force_rel_residual"], "elapsed_s": r["elapsed_s"],
        })
        if verbose:
            print(f"  [{idx + 1}/{n_samples}] fine solve elapsed_s={r['elapsed_s']:.2f}s "
                  f"force_rel_residual={r['force_rel_residual']:.2e}")

    return {"geom_fine": geom_fine, "region_mask_fine_torch": region_mask_fine_torch,
            "fine_resolution": [Ntheta_f, Nr_f, Nz_f], "coarse_resolution": [Ntheta_c, Nr_c, Nz_c],
            "n_region_fine": n_region_fine, "samples": samples}


def evaluate_u_only_checkpoint_against_fine_reference(checkpoint_path, fine_ref, device,
                                                        dtype=torch.float64, verbose=True):
    """THE CHEAP STEP: one forward pass per sample, no FEM solving.
    Works for any u-only checkpoint (train_B3.build_model's own
    architecture) -- the PI baseline, the equilibrium-augmented
    checkpoints (items 2/3), and every data-driven checkpoint (item 4
    and the break-even sweep), since all of them share the EXACT SAME
    build_model (confirmed: train_B3_equilibrium_augmented.py and
    train_B3_data_driven.py both import it unmodified from train_B3.py,
    not their own copies). NOT for the mixed-stress checkpoints (item
    1) -- see evaluate_B3_mixed_stress_finer.py for those, which also
    need the network's own predicted P field, not just u."""
    geom_fine, region_mask_fine_torch = fine_ref["geom_fine"], fine_ref["region_mask_fine_torch"]

    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")
    install_input_norm_for_checkpoint(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    for s in fine_ref["samples"]:
        E_f_t, nu_f_t, phi_t, u_true_f = s["E_f_t"], s["nu_f_t"], s["phi_t"], s["u_true_f"]
        with torch.no_grad():
            fun_material = torch.stack(
                [E_f_t, nu_f_t, phi_t[:, None].expand(-1, geom_fine["n_nodes"])], dim=2)
            fun_material = _apply_input_norm(fun_material)
            xyz_batch = geom_fine["nodes"].unsqueeze(0)
            u_net = model(xyz_batch, fun_material)
            u_pred_f = apply_dirichlet_b3(u_net, geom_fine, phi_t)

        sigma_true_f, w_true_f = compute_region_sigma_full(u_true_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
        sigma_pred_f, _ = compute_region_sigma_full(u_pred_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
        field_rel_fine = cauchy_field_rel_from_full(sigma_true_f, w_true_f, sigma_pred_f)
        sigma_components_rel_fine = sigma_components_rel_from_full(sigma_true_f, w_true_f, sigma_pred_f)
        avg_true_f, _, _ = compute_region_sigma_xx(u_true_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
        avg_pred_f, _, _ = compute_region_sigma_xx(u_pred_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
        h1_rel_fine = compute_H1_seminorm_rel(u_true_f, u_pred_f, geom_fine)

        err = u_pred_f - u_true_f
        disp_rel_l2 = {}
        for k, comp in enumerate(("ux", "uy", "uz")):
            l2 = torch.sqrt(torch.mean(err[:, :, k] ** 2))
            ref = torch.sqrt(torch.mean(u_true_f[:, :, k] ** 2)) + 1e-12
            disp_rel_l2[comp] = float((l2 / ref).item())
        l2_all = torch.sqrt(torch.mean(torch.sum(err ** 2, dim=2)))
        ref_all = torch.sqrt(torch.mean(torch.sum(u_true_f ** 2, dim=2))) + 1e-12
        disp_rel_l2["combined"] = float((l2_all / ref_all).item())

        energy_true_f = compute_energy(u_true_f, E_f_t, nu_f_t, geom_fine)
        energy_pred_f = compute_energy(u_pred_f, E_f_t, nu_f_t, geom_fine)
        Fr_true_f, Mr_true_f, _ = compute_reaction(u_true_f, E_f_t, nu_f_t, geom_fine)
        Fr_pred_f, Mr_pred_f, _ = compute_reaction(u_pred_f, E_f_t, nu_f_t, geom_fine)

        rows.append({
            "sample_index": s["sample_index"], "phi": s["phi"],
            "force_rel_residual_fine_solve": s["force_rel_residual"],
            "fine_fem_solve_elapsed_s": s["elapsed_s"],
            "region_cauchy_field_rel_fine": float(field_rel_fine[0].item()),
            "region_avg_sigma_xx_true_fine": float(avg_true_f[0].item()),
            "region_avg_sigma_xx_pred_fine": float(avg_pred_f[0].item()),
            "H1_seminorm_rel_fine": float(h1_rel_fine[0].item()),
            "region_sigma_components_rel_fine": {
                name: float(sigma_components_rel_fine[name][0].item()) for name, _, _ in STRESS_COMPONENTS},
            "disp_rel_l2_fine": disp_rel_l2,
            "energy_true_fine": float(energy_true_f[0].item()),
            "energy_pred_fine": float(energy_pred_f[0].item()),
            "reaction_moment_y_true_fine": float(Mr_true_f[0].item()),
            "reaction_moment_y_pred_fine": float(Mr_pred_f[0].item()),
        })

    n_samples = len(rows)
    vals = np.array([row["region_cauchy_field_rel_fine"] for row in rows])
    vals_valid = vals[~np.isnan(vals)]
    h1_vals = np.array([row["H1_seminorm_rel_fine"] for row in rows])
    summary = {
        "n_samples": n_samples, "n_region_gauss_points_fine": fine_ref["n_region_fine"],
        "n_region_gauss_points_coarse": 6,
        "mean_region_cauchy_field_rel_fine": float(np.mean(vals_valid)) if len(vals_valid) else float("nan"),
        "median_region_cauchy_field_rel_fine": float(np.median(vals_valid)) if len(vals_valid) else float("nan"),
        "std_region_cauchy_field_rel_fine": float(np.std(vals_valid)) if len(vals_valid) else float("nan"),
        "mean_H1_seminorm_rel_fine": float(np.mean(h1_vals)),
        "median_H1_seminorm_rel_fine": float(np.median(h1_vals)),
        "std_H1_seminorm_rel_fine": float(np.std(h1_vals)),
    }
    for comp in ("ux", "uy", "uz", "combined"):
        vals_c = np.array([row["disp_rel_l2_fine"][comp] for row in rows])
        summary[f"mean_disp_rel_l2_fine_{comp}"] = float(np.mean(vals_c))
        summary[f"std_disp_rel_l2_fine_{comp}"] = float(np.std(vals_c))
    for name, _, _ in STRESS_COMPONENTS:
        vals_s = np.array([row["region_sigma_components_rel_fine"][name] for row in rows])
        vals_s_valid = vals_s[~np.isnan(vals_s)]
        summary[f"mean_region_sigma_{name}_rel_fine"] = float(np.mean(vals_s_valid)) if len(vals_s_valid) else float("nan")
        summary[f"median_region_sigma_{name}_rel_fine"] = float(np.median(vals_s_valid)) if len(vals_s_valid) else float("nan")

    def _rel_err_fine(key_true, key_pred):
        t = np.array([row[key_true] for row in rows])
        p = np.array([row[key_pred] for row in rows])
        pooled_rms_rel = float(np.sqrt(np.mean((p - t) ** 2)) / np.sqrt(np.mean(t ** 2)))
        mean_rel = float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-12)))
        return mean_rel, pooled_rms_rel

    mean_e, pooled_e = _rel_err_fine("energy_true_fine", "energy_pred_fine")
    summary["mean_rel_err_energy_fine"] = mean_e
    summary["pooled_rms_rel_err_energy_fine"] = pooled_e
    mean_m, pooled_m = _rel_err_fine("reaction_moment_y_true_fine", "reaction_moment_y_pred_fine")
    summary["mean_rel_err_reaction_moment_y_fine"] = mean_m
    summary["pooled_rms_rel_err_reaction_moment_y_fine"] = pooled_m
    summary["mean_fine_fem_solve_elapsed_s"] = float(np.mean(
        [row["fine_fem_solve_elapsed_s"] for row in rows]))

    if verbose:
        print(f"  region_cauchy_field_rel_fine: mean={summary['mean_region_cauchy_field_rel_fine']:.4f}  "
              f"median={summary['median_region_cauchy_field_rel_fine']:.4f}")
        print(f"  H1_seminorm_rel_fine: mean={summary['mean_H1_seminorm_rel_fine']:.4f}  "
              f"median={summary['median_H1_seminorm_rel_fine']:.4f}")
        print(f"  disp_rel_l2_fine combined: {summary['mean_disp_rel_l2_fine_combined']:.4f}")
        print(f"  energy rel err (fine): mean={mean_e:.4f}  pooled_rms={pooled_e:.4f}")
        print(f"  reaction_moment_y rel err (fine): mean={mean_m:.4f}  pooled_rms={pooled_m:.4f}")
        comp_str = "  ".join(f"{name}={summary[f'mean_region_sigma_{name}_rel_fine']:.4f}"
                              for name, _, _ in STRESS_COMPONENTS)
        print(f"  region_sigma_components_rel_fine (mean): {comp_str}")

    return {"checkpoint": checkpoint_path, "checkpoint_iter": ckpt.get("iter") or ckpt.get("step"),
            "coarse_resolution": fine_ref["coarse_resolution"], "fine_resolution": fine_ref["fine_resolution"],
            "summary": summary, "per_sample": rows}


def main():
    parser = argparse.ArgumentParser(
        "Finer-resolution region-stress/H1/displacement/energy/reaction evaluation for B3 "
        "(single checkpoint; see generate_fine_reference + "
        "evaluate_u_only_checkpoint_against_fine_reference for the multi-checkpoint, "
        "shared-reference usage).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--fine_resolution", type=int, nargs=3, default=list(FINE_RESOLUTION_STRESS),
                         metavar=("NTHETA", "NR", "NZ"))
    parser.add_argument("--n_samples", type=int, default=10,
                         help="how many of the held-out samples to re-solve at the finer resolution")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    fine_ref = generate_fine_reference(args.dataset, tuple(args.fine_resolution), args.n_samples, device, dtype)
    result = evaluate_u_only_checkpoint_against_fine_reference(args.checkpoint, fine_ref, device, dtype)

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
