"""Real physical QoIs for a trained B3 MIXED-STRESS checkpoint
(mixed_stress_B3.py / train_B3_mixed_stress.py) -- directly comparable
to evaluate_B3_qois.py's own numbers for the baseline (u-only, DEM)
checkpoint, since it reuses that script's exact QoI machinery
(compute_energy, compute_reaction, compute_region_sigma_xx,
compute_region_cauchy_field_rel -- all operate on a `u` field and don't
care how it was produced). Deliberately a separate script, not an edit
to evaluate_B3_qois.py, so the baseline evaluation path used for the
already-published numbers cannot be affected even by accident.

The one genuinely new thing this script adds, not possible for the
baseline checkpoint: the mixed-stress model also predicts P directly,
so region stress can be computed TWO ways from the same forward pass --

  (a) "implicit" stress: P = dpsi/dF via autograd on u_pred (exactly
      what evaluate_B3_qois.py already does for the baseline);
  (b) "direct" stress: the network's own predicted P_pred, interpolated
      to the region's Gauss points and converted via the same
      sigma = (1/detF) P F^T formula used for (a).

Comparing (a) vs. (b) against the true FEM stress in the same run is
the direct, apples-to-apples test of whether predicting stress
explicitly (Professor Rabczuk's item 1) actually helps the stress QoI,
not just whether the displacement field improved.

Usage:
  python -m omar_pfem.evaluate_B3_mixed_stress \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_mixed_stress_training/checkpoint_50000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_mixed_stress_training/qois_50000.json
"""
import argparse
import json

import h5py
import numpy as np
import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION, R_GRADING
from omar_pfem.data.mesh_convergence_B3 import MIN_RELIABLE_N_P99
from omar_pfem.evaluate_B3_qois import (
    build_region_mask, compute_energy, compute_reaction, compute_region_cauchy_field_rel,
    compute_region_sigma_xx,
)
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, interpolate_nodal_field_to_quadpts
from omar_pfem.train_B1 import _apply_input_norm, install_input_norm_for_checkpoint
from omar_pfem.train_B3 import apply_dirichlet_b3
from omar_pfem.train_B3_mixed_stress import build_model


def compute_region_sigma_xx_direct(p_net_raw, u_pred, E_node, nu_node, geom, region_mask_torch):
    """Same region average/p99 sigma_xx as compute_region_sigma_xx, but
    using the network's OWN predicted P (interpolated to the region's
    Gauss points via N), not an autograd-derived one -- the "direct"
    stress path unique to the mixed-stress formulation. F still comes
    from u_pred (same deformation gradient either way), so the only
    thing that differs from compute_region_sigma_xx is the SOURCE of P;
    the sigma=(1/detF) P F^T conversion is identical."""
    elements, N_sf, B_op, detJ, iweights = (
        geom["elements"], geom["N"], geom["B"], geom["detJ"], geom["iweights"])
    n_gauss = B_op.shape[0]
    Batch = u_pred.shape[0]

    ue = u_pred[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    eye = torch.eye(3, device=u_pred.device, dtype=u_pred.dtype).view(1, 1, 3, 3)
    p_flat = p_net_raw.reshape(Batch, p_net_raw.shape[1], 9)
    P_interp_all = interpolate_nodal_field_to_quadpts(p_flat, elements, N_sf)  # (g,B,e,9)

    sigma_xx_per_gauss, vol_weight_per_gauss = [], []
    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F = eye + H
        detF = torch.linalg.det(F)
        P_pred = P_interp_all[g].reshape(Batch, -1, 3, 3)
        sigma = torch.matmul(P_pred, F.transpose(-1, -2)) / detF[..., None, None]
        sigma_xx_per_gauss.append(sigma[:, :, 0, 0])
        vol_weight_per_gauss.append(iweights[g] * detJ[g].abs())

    sigma_xx = torch.stack(sigma_xx_per_gauss, dim=2)       # (B, n_elem, n_gauss)
    vol_weight = torch.stack(vol_weight_per_gauss, dim=1)   # (n_elem, n_gauss)
    sigma_xx_flat = sigma_xx[:, region_mask_torch]          # (B, n_region)
    w_flat = vol_weight[region_mask_torch]
    w_sum = w_flat.sum()
    n_region = region_mask_torch.sum().item()

    if n_region == 0 or w_sum <= 0:
        nan_col = torch.full((Batch,), float("nan"), dtype=u_pred.dtype, device=u_pred.device)
        return nan_col, nan_col.clone()

    region_avg = (sigma_xx_flat * w_flat[None, :]).sum(dim=1) / w_sum
    region_p99 = torch.full((Batch,), float("nan"), dtype=u_pred.dtype, device=u_pred.device)
    if n_region >= MIN_RELIABLE_N_P99:
        order = torch.argsort(sigma_xx_flat, dim=1)
        sorted_vals = torch.gather(sigma_xx_flat, 1, order)
        sorted_w = w_flat[order]
        cum_w = torch.cumsum(sorted_w, dim=1) / w_sum
        for b in range(Batch):
            idx99 = int(torch.searchsorted(cum_w[b], torch.tensor(0.99, dtype=u_pred.dtype,
                                                                    device=u_pred.device)).item())
            idx99 = min(idx99, sorted_vals.shape[1] - 1)
            region_p99[b] = sorted_vals[b, idx99]
    return region_avg, region_p99


def rel_err(rows, key_true, key_pred):
    """Same convention as evaluate_B3_qois.py's own (private) helper --
    duplicated here rather than imported, since that one is a closure
    over evaluate_B3_qois.main()'s own `rows` local and not meant to be
    imported; kept deliberately identical so the two scripts' numbers
    are directly comparable."""
    t = np.array([r[key_true] for r in rows])
    p = np.array([r[key_pred] for r in rows])
    return float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-12))), float(np.std(np.abs(p - t) / (np.abs(t) + 1e-12)))


def robust_stats(rows, key_true, key_pred):
    t = np.array([r[key_true] for r in rows])
    p = np.array([r[key_pred] for r in rows])
    abs_err = np.abs(p - t)
    median_rel = float(np.median(abs_err / (np.abs(t) + 1e-12)))
    pooled_rms_rel = float(np.sqrt(np.mean(abs_err ** 2)) / np.sqrt(np.mean(t ** 2)))
    corr = float(np.corrcoef(t, p)[0, 1]) if np.std(t) > 0 and np.std(p) > 0 else float("nan")
    sign_agree = float(np.mean(np.sign(t) == np.sign(p)))
    return median_rel, pooled_rms_rel, corr, sign_agree


def main():
    parser = argparse.ArgumentParser(
        "Compute real physical QoIs (energy, reaction, region Cauchy stress -- both implicit "
        "and direct) for a B3 mixed-stress checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    print(f"Building fixed geometry at resolution=({Ntheta},{Nr},{Nz})...")
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"  {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    region_mask_np = build_region_mask(Ntheta, Nr, Nz, R_GRADING)
    region_mask_torch = torch.tensor(region_mask_np, device=device)
    n_region = int(region_mask_np.sum())
    print(f"  fixed region: {n_region} Gauss points (reliable p99 needs >= {MIN_RELIABLE_N_P99})")

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    p_output_scale = float(ckpt_args.p_output_scale)

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node = torch.tensor(h5["E_node"][:], dtype=dtype)
        nu_node = torch.tensor(h5["nu_node"][:], dtype=dtype)
        phi = torch.tensor(h5["phi"][:], dtype=dtype)
        u_true_all = torch.tensor(h5["displacements"][:], dtype=dtype)
    n_samples = E_node.shape[0]
    print(f"  {n_samples} FEM samples")

    rows = []
    for start in range(0, n_samples, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n_samples)
        E_b = E_node[start:end].to(device=device, dtype=dtype)
        nu_b = nu_node[start:end].to(device=device, dtype=dtype)
        phi_b = phi[start:end].to(device=device, dtype=dtype)
        u_true = u_true_all[start:end].to(device=device, dtype=dtype)

        with torch.no_grad():
            fun_material = torch.stack([E_b, nu_b, phi_b[:, None].expand(-1, geom["n_nodes"])], dim=2)
            fun_material = _apply_input_norm(fun_material)
            xyz_batch = geom["nodes"].unsqueeze(0).expand(end - start, -1, -1)
            raw = model(xyz_batch, fun_material)
            u_net_raw = raw[..., :3]
            p_net_raw = raw[..., 3:] * p_output_scale
            u_pred = apply_dirichlet_b3(u_net_raw, geom, phi_b)

        U_true = compute_energy(u_true, E_b, nu_b, geom)
        U_pred = compute_energy(u_pred, E_b, nu_b, geom)

        Fr_true, Mr_true, eq_true = compute_reaction(u_true, E_b, nu_b, geom)
        Fr_pred, Mr_pred, eq_pred = compute_reaction(u_pred, E_b, nu_b, geom)

        avg_true, p99_true, _ = compute_region_sigma_xx(u_true, E_b, nu_b, geom, region_mask_torch)
        avg_implicit, p99_implicit, _ = compute_region_sigma_xx(u_pred, E_b, nu_b, geom, region_mask_torch)
        avg_direct, p99_direct = compute_region_sigma_xx_direct(
            p_net_raw, u_pred, E_b, nu_b, geom, region_mask_torch)

        cauchy_field_rel = compute_region_cauchy_field_rel(u_true, u_pred, E_b, nu_b, geom, region_mask_torch)

        for i in range(end - start):
            rows.append({
                "energy_true": U_true[i].item(), "energy_pred": U_pred[i].item(),
                "reaction_moment_y_true": Mr_true[i].item(), "reaction_moment_y_pred": Mr_pred[i].item(),
                "region_avg_sigma_xx_true": avg_true[i].item(),
                "region_avg_sigma_xx_implicit": avg_implicit[i].item(),
                "region_avg_sigma_xx_direct": avg_direct[i].item(),
                "region_p99_sigma_xx_true": p99_true[i].item(),
                "region_p99_sigma_xx_implicit": p99_implicit[i].item(),
                "region_p99_sigma_xx_direct": p99_direct[i].item(),
                "region_cauchy_field_rel_implicit": cauchy_field_rel[i].item(),
            })
        print(f"  [{end}/{n_samples}] done")

    summary = {}
    for name, (kt, kp) in {
        "energy": ("energy_true", "energy_pred"),
        "reaction_moment_y": ("reaction_moment_y_true", "reaction_moment_y_pred"),
        "region_avg_sigma_xx_implicit": ("region_avg_sigma_xx_true", "region_avg_sigma_xx_implicit"),
        "region_avg_sigma_xx_direct": ("region_avg_sigma_xx_true", "region_avg_sigma_xx_direct"),
        "region_p99_sigma_xx_implicit": ("region_p99_sigma_xx_true", "region_p99_sigma_xx_implicit"),
        "region_p99_sigma_xx_direct": ("region_p99_sigma_xx_true", "region_p99_sigma_xx_direct"),
    }.items():
        mean_e, std_e = rel_err(rows, kt, kp)
        median_rel, pooled_rms_rel, corr, sign_agree = robust_stats(rows, kt, kp)
        summary[f"mean_rel_err_{name}"] = mean_e
        summary[f"std_rel_err_{name}"] = std_e
        summary[f"median_rel_err_{name}"] = median_rel
        summary[f"pooled_rms_rel_err_{name}"] = pooled_rms_rel
        summary[f"corr_{name}"] = corr
        summary[f"sign_agree_{name}"] = sign_agree
        print(f"  {name}: mean rel err = {mean_e:.4f} (std={std_e:.4f})  "
              f"median={median_rel:.4f}  pooled_rms={pooled_rms_rel:.4f}  "
              f"corr={corr:.4f}  sign_agree={sign_agree:.2f}")

    cfr = np.array([r["region_cauchy_field_rel_implicit"] for r in rows])
    cfr_valid = cfr[~np.isnan(cfr)]
    summary["mean_region_cauchy_field_rel_implicit"] = float(np.mean(cfr_valid)) if len(cfr_valid) else float("nan")
    summary["median_region_cauchy_field_rel_implicit"] = float(np.median(cfr_valid)) if len(cfr_valid) else float("nan")
    summary["n_valid_region_cauchy_field_rel_implicit"] = int(len(cfr_valid))
    print(f"  region_cauchy_field_rel (implicit, full-tensor, volume-weighted): "
          f"mean={summary['mean_region_cauchy_field_rel_implicit']:.4f}  "
          f"median={summary['median_region_cauchy_field_rel_implicit']:.4f}  "
          f"(n_valid={summary['n_valid_region_cauchy_field_rel_implicit']}/{n_samples})")

    print("\nKey comparison for Professor Rabczuk's item 1 (does predicting P directly help "
          "the stress QoI?): region_avg_sigma_xx_direct's pooled_rms vs. "
          "region_avg_sigma_xx_implicit's pooled_rms above -- smaller is better, "
          "'direct' is the new capability this formulation adds.")

    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "per_sample": rows, "checkpoint": args.checkpoint,
                   "dataset": args.dataset}, f, indent=1)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
