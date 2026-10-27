"""Cheap, no-training-needed checkpoint inspection: has the mixed-stress
network's predicted P collapsed to a near-constant, near-zero field
(the exact failure mode attempt 4's own CPU root-cause investigation
found and confirmed, 2026-10-22 -- predicted P mean abs ~0.73-1.27 vs.
a true field whose mean abs is ~68), or is it a genuinely-scaled,
varied prediction?

Needed because pooled-RMS relative error ALONE cannot tell the two
apart: a trivial near-zero predictor naturally scores a pooled RMS
relative error close to 100% (err~=true when pred~=0, so the ratio
->1.0) regardless of how "small" that number looks next to a much
larger one from a different, non-degenerate failure mode -- exactly
the confusion this project already corrected once for attempt 1 vs.
attempt 4's own raw numbers. This script settles the question directly
by reporting the SAME statistic (predicted-P mean abs / std across
nodes) attempt 4's own diagnosis used, for ANY mixed-stress checkpoint,
against the TRUE P field's own stats computed from the held-out FEM
set itself (not assumed from the earlier toy-model measurement).

Usage:
  python -m omar_pfem.inspect_mixed_stress_P_scale \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_mixed_stress_local_refine_relative/checkpoint_50000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5
"""
import argparse

import h5py
import numpy as np
import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, neo_hookean_P_closed_form
from omar_pfem.train_B1 import _apply_input_norm, install_input_norm_for_checkpoint
from omar_pfem.train_B3 import apply_dirichlet_b3
from omar_pfem.train_B3_mixed_stress import build_model


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta, Nr, Nz = DEFAULT_RESOLUTION
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"Geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes, device={device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    p_output_scale = float(ckpt_args.p_output_scale)
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  p_output_scale={p_output_scale}, "
          f"relative_constitutive={bool(getattr(ckpt_args, 'relative_constitutive', 0))}, "
          f"local_refine_region={bool(getattr(ckpt_args, 'local_refine_region', 0))}")

    print(f"Loading held-out set: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        n = min(args.n_samples, h5["E_node"].shape[0])
        E_node = torch.tensor(h5["E_node"][:n], dtype=dtype, device=device)
        nu_node = torch.tensor(h5["nu_node"][:n], dtype=dtype, device=device)
        phi = torch.tensor(h5["phi"][:n], dtype=dtype, device=device)
        u_true = torch.tensor(h5["displacements"][:n], dtype=dtype, device=device)
    print(f"  using {n} held-out samples")

    xyz_batch = geom["nodes"].unsqueeze(0).expand(n, -1, -1)
    fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
    fun_material = _apply_input_norm(fun_material)
    with torch.no_grad():
        raw = model(xyz_batch, fun_material)
        u_pred = apply_dirichlet_b3(raw[..., :3], geom, phi)
        p_net_raw = raw[..., 3:] * p_output_scale  # (n, n_nodes, 9) -- nodal predicted P

    pred_abs = p_net_raw.abs()
    print(f"\n=== Predicted P (nodal, raw network output x p_output_scale) ===")
    print(f"  mean abs  = {pred_abs.mean().item():.4f}")
    print(f"  std       = {p_net_raw.std().item():.4f}")
    print(f"  max abs   = {pred_abs.max().item():.4f}")
    print(f"  median abs= {pred_abs.median().item():.4f}")

    # True P, computed the SAME way mixed_stress_loss_B3 does -- at every
    # Gauss point of every element, from the TRUE (FEM) displacement field.
    elements = geom["elements"]
    B_op = geom["B"]
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    ue_true = u_true[:, elements, :].permute(0, 1, 3, 2)
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    true_abs_all = []
    with torch.no_grad():
        for g in range(B_op.shape[0]):
            H = torch.einsum("bedq,ecq->bedc", ue_true, B_op[g])
            F = eye + H
            P_true = neo_hookean_P_closed_form(F, mu_elem, lam_elem)
            true_abs_all.append(P_true.abs().reshape(-1))
    true_abs_all = torch.cat(true_abs_all)
    print(f"\n=== True P (element Gauss points, from the held-out FEM displacement field) ===")
    print(f"  mean abs  = {true_abs_all.mean().item():.4f}")
    print(f"  std       = {true_abs_all.std().item():.4f}")
    print(f"  max abs   = {true_abs_all.max().item():.4f}")
    print(f"  median abs= {true_abs_all.median().item():.4f}")

    ratio = pred_abs.mean().item() / true_abs_all.mean().item()
    print(f"\n=== Verdict ===")
    print(f"  predicted-P mean-abs / true-P mean-abs = {ratio:.4f}")
    if ratio < 0.1 and p_net_raw.std().item() / pred_abs.mean().clamp_min(1e-9).item() < 1.0:
        print("  COLLAPSED: predicted P is near-constant and far smaller in magnitude than the "
              "true field -- the same signature attempt 4's own CPU diagnosis found (mean abs "
              "~0.73-1.27 vs. true ~68). The pooled-RMS 'improvement' is almost certainly the "
              "trivial-near-zero-predictor artifact, not genuine accuracy.")
    elif ratio < 0.5:
        print("  PARTIALLY COLLAPSED / UNDER-SCALED: predicted P magnitude is well below the "
              "true field's own scale -- treat any pooled-RMS improvement with real suspicion; "
              "inspect the std/mean ratio above too (a low std relative to mean, like the true "
              "field's own, would still suggest genuine variation; a near-zero std would not).")
    else:
        print("  NOT obviously collapsed: predicted P's magnitude is in the same ballpark as the "
              "true field. A pooled-RMS improvement here is more plausibly genuine -- though still "
              "worth checking correlation/sign-agreement stats from the full evaluation separately.")


if __name__ == "__main__":
    main()
