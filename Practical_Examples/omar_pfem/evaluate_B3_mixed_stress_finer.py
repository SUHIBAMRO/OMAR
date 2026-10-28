"""Finer-resolution (independent-mesh) region-stress + H1 evaluation
for B3's MIXED-STRESS checkpoints (item 1: mixed_stress_B3.py /
train_B3_mixed_stress.py) -- the mixed-stress counterpart of
evaluate_B3_qois_finer.py, needed because that script's own
`evaluate_u_only_checkpoint_against_fine_reference` only handles
u-only models (train_B3.build_model's architecture): it has no
predicted-P field to query, so it cannot report the "direct" stress
path that is item 1's entire point.

Built 2026-10-28, after Professor Rabczuk's own review found that NONE
of this project's follow-up-investigation region-stress numbers
(item 1's included) had ever been checked against an independent fine
reference -- every one of them compared the network to the SAME
6,840-element mesh it was trained on. This module reuses
evaluate_B3_qois_finer.generate_fine_reference UNCHANGED for the
expensive part (re-solving held-out samples at a fine mesh) and adds
a mixed-stress-specific evaluation function for the cheap part
(querying the mixed-stress model and computing BOTH the implicit and
direct stress paths against that shared fine reference).

The "direct" path needs the fine mesh's own shape-function VALUES (N),
not just its gradient operator (B) -- build_fixed_geometry (used by
generate_fine_reference, shared with the u-only path) does not keep N.
This module therefore builds its OWN fine geometry via
build_fixed_geometry_mixed (same mesh, N included) rather than
re-using generate_fine_reference's own geom_fine -- the two geometries
describe the identical physical mesh (same resolution, same node
ordering), so the SAME fine_ref's per-sample E/nu/phi/u_true can be
reused directly; only B-only quantities (H1, the implicit stress path)
need geom_fine itself, and the direct path needs this module's own
geom_fine_mixed instead.

Usage:
  from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, FINE_RESOLUTION_STRESS
  from omar_pfem.evaluate_B3_mixed_stress_finer import evaluate_mixed_stress_checkpoint_against_fine_reference
  fine_ref = generate_fine_reference(dataset_path, FINE_RESOLUTION_STRESS, 10, device, dtype)
  result = evaluate_mixed_stress_checkpoint_against_fine_reference(checkpoint_path, fine_ref, device, dtype)
"""
import argparse

import numpy as np
import torch

from omar_pfem.evaluate_B3_qois import compute_H1_seminorm_rel, compute_region_sigma_full
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, interpolate_nodal_field_to_quadpts
from omar_pfem.train_B1 import _apply_input_norm, install_input_norm_for_checkpoint
from omar_pfem.train_B3 import apply_dirichlet_b3
from omar_pfem.train_B3_mixed_stress import build_model


def compute_region_sigma_full_direct(p_net_raw, u_pred, geom, region_mask_torch):
    """Same per-Gauss F/sigma construction as evaluate_B3_qois.
    compute_region_sigma_full, but sigma = (1/detF) P_pred F^T uses the
    network's OWN predicted P (interpolated to the region's Gauss
    points via N), not an autograd-derived one -- the mixed-stress
    formulation's unique "direct" capability. Returns (B, n_region, 3, 3)
    plus the region's volume weights (n_region,), matching compute_
    region_sigma_full's own return shape so both feed the same
    relative-field-error formula."""
    elements, N_sf, B_op, detJ, iweights = (
        geom["elements"], geom["N"], geom["B"], geom["detJ"], geom["iweights"])
    n_gauss = B_op.shape[0]
    Batch = u_pred.shape[0]

    ue = u_pred[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    eye = torch.eye(3, device=u_pred.device, dtype=u_pred.dtype).view(1, 1, 3, 3)
    p_flat = p_net_raw.reshape(Batch, p_net_raw.shape[1], 9)
    P_interp_all = interpolate_nodal_field_to_quadpts(p_flat, elements, N_sf)  # (g,B,e,9)

    sigma_per_gauss, vol_weight_per_gauss = [], []
    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F = eye + H
        detF = torch.linalg.det(F)
        P_pred = P_interp_all[g].reshape(Batch, -1, 3, 3)
        sigma = torch.matmul(P_pred, F.transpose(-1, -2)) / detF[..., None, None]
        sigma_per_gauss.append(sigma)
        vol_weight_per_gauss.append(iweights[g] * detJ[g].abs())

    sigma_all = torch.stack(sigma_per_gauss, dim=2)        # (B, n_elem, n_gauss, 3, 3)
    vol_weight = torch.stack(vol_weight_per_gauss, dim=1)  # (n_elem, n_gauss)
    sigma_region = sigma_all[:, region_mask_torch]  # (B, n_region, 3, 3)
    w_region = vol_weight[region_mask_torch]        # (n_region,)
    return sigma_region, w_region


def _cauchy_field_rel_from_sigma(sigma_true, w_true, sigma_pred, w_pred):
    """Shared tail of evaluate_B3_qois.compute_region_cauchy_field_rel,
    factored out so both the implicit (same geometric weights for true
    and pred, since both come from the SAME mesh) and direct paths use
    one formula. w_true and w_pred are identical here (same geom_fine_
    mixed mesh for both) -- kept as two args only for clarity."""
    Batch = sigma_true.shape[0]
    w = w_true
    w_sum = w.sum()
    if w.shape[0] == 0 or w_sum <= 0:
        return torch.full((Batch,), float("nan"), dtype=sigma_true.dtype, device=sigma_true.device)
    diff_sq = torch.sum((sigma_pred - sigma_true) ** 2, dim=(-1, -2))
    true_sq = torch.sum(sigma_true ** 2, dim=(-1, -2))
    num = torch.sqrt((diff_sq * w[None, :]).sum(dim=1) / w_sum)
    den = torch.sqrt((true_sq * w[None, :]).sum(dim=1) / w_sum)
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(num, float("nan")))


def evaluate_mixed_stress_checkpoint_against_fine_reference(checkpoint_path, fine_ref, device,
                                                              dtype=torch.float64, verbose=True):
    """THE CHEAP STEP for a mixed-stress (item 1) checkpoint: one
    forward pass per sample against fine_ref's already-solved ground
    truth (from evaluate_B3_qois_finer.generate_fine_reference, no new
    FEM solving here). Reports H1 semi-norm (same formula as the u-only
    path) plus BOTH the implicit and direct region-Cauchy-stress field
    errors, so item 1's own headline comparison (does direct beat
    implicit?) can finally be checked against an independent fine
    reference instead of the 6,840-element training mesh."""
    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    geom_fine_mixed = build_fixed_geometry_mixed(Ntheta_f, Nr_f, Nz_f, device, dtype=dtype)
    region_mask_fine_torch = fine_ref["region_mask_fine_torch"]

    if verbose:
        print(f"Loading mixed-stress checkpoint: {checkpoint_path}")
    install_input_norm_for_checkpoint(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    p_output_scale = float(ckpt_args.p_output_scale)
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    for s in fine_ref["samples"]:
        E_f_t, nu_f_t, phi_t, u_true_f = s["E_f_t"], s["nu_f_t"], s["phi_t"], s["u_true_f"]
        with torch.no_grad():
            fun_material = torch.stack(
                [E_f_t, nu_f_t, phi_t[:, None].expand(-1, geom_fine_mixed["n_nodes"])], dim=2)
            fun_material = _apply_input_norm(fun_material)
            xyz_batch = geom_fine_mixed["nodes"].unsqueeze(0)
            raw = model(xyz_batch, fun_material)
            u_pred_f = apply_dirichlet_b3(raw[..., :3], geom_fine_mixed, phi_t)
            p_net_raw_f = raw[..., 3:] * p_output_scale

        h1_rel_fine = compute_H1_seminorm_rel(u_true_f, u_pred_f, geom_fine_mixed)

        sigma_true, w_true = compute_region_sigma_full(u_true_f, E_f_t, nu_f_t, geom_fine_mixed,
                                                         region_mask_fine_torch)
        sigma_pred_implicit, w_pred_i = compute_region_sigma_full(u_pred_f, E_f_t, nu_f_t,
                                                                    geom_fine_mixed, region_mask_fine_torch)
        sigma_pred_direct, w_pred_d = compute_region_sigma_full_direct(p_net_raw_f, u_pred_f,
                                                                         geom_fine_mixed, region_mask_fine_torch)
        field_rel_implicit = _cauchy_field_rel_from_sigma(sigma_true, w_true, sigma_pred_implicit, w_pred_i)
        field_rel_direct = _cauchy_field_rel_from_sigma(sigma_true, w_true, sigma_pred_direct, w_pred_d)

        rows.append({
            "sample_index": s["sample_index"], "phi": s["phi"],
            "force_rel_residual_fine_solve": s["force_rel_residual"],
            "fine_fem_solve_elapsed_s": s["elapsed_s"],
            "H1_seminorm_rel_fine": float(h1_rel_fine[0].item()),
            "region_cauchy_field_rel_implicit_fine": float(field_rel_implicit[0].item()),
            "region_cauchy_field_rel_direct_fine": float(field_rel_direct[0].item()),
        })

    n_samples = len(rows)
    h1_vals = np.array([row["H1_seminorm_rel_fine"] for row in rows])
    imp_vals = np.array([row["region_cauchy_field_rel_implicit_fine"] for row in rows])
    dir_vals = np.array([row["region_cauchy_field_rel_direct_fine"] for row in rows])
    imp_valid, dir_valid = imp_vals[~np.isnan(imp_vals)], dir_vals[~np.isnan(dir_vals)]
    summary = {
        "n_samples": n_samples,
        "mean_H1_seminorm_rel_fine": float(np.mean(h1_vals)),
        "median_H1_seminorm_rel_fine": float(np.median(h1_vals)),
        "mean_region_cauchy_field_rel_implicit_fine": float(np.mean(imp_valid)) if len(imp_valid) else float("nan"),
        "median_region_cauchy_field_rel_implicit_fine": float(np.median(imp_valid)) if len(imp_valid) else float("nan"),
        "mean_region_cauchy_field_rel_direct_fine": float(np.mean(dir_valid)) if len(dir_valid) else float("nan"),
        "median_region_cauchy_field_rel_direct_fine": float(np.median(dir_valid)) if len(dir_valid) else float("nan"),
    }
    if verbose:
        print(f"  H1_seminorm_rel_fine: mean={summary['mean_H1_seminorm_rel_fine']:.4f}")
        print(f"  region_cauchy_field_rel_implicit_fine: mean={summary['mean_region_cauchy_field_rel_implicit_fine']:.4f}")
        print(f"  region_cauchy_field_rel_direct_fine:   mean={summary['mean_region_cauchy_field_rel_direct_fine']:.4f}")
        print("  Key comparison for item 1: direct vs. implicit, now against an independent "
              "fine reference instead of the 6,840-element training mesh.")

    return {"checkpoint": checkpoint_path, "checkpoint_iter": ckpt.get("iter") or ckpt.get("step"),
            "coarse_resolution": fine_ref["coarse_resolution"], "fine_resolution": fine_ref["fine_resolution"],
            "summary": summary, "per_sample": rows}
