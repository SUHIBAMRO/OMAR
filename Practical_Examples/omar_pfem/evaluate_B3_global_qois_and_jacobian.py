"""Two real gaps flagged by an independent review of the paper draft,
neither requiring a new training run -- both are pure evaluation against
an already-trained, already-saved checkpoint:

1. The local-integration-refinement checkpoint
   (b3_training_local_refine/checkpoint_50000.pt) improved regional
   Cauchy-stress error from ~62% to ~29.8%, but its own displacement,
   energy, and reaction accuracy were never separately measured -- the
   paper currently reports those three quantities only for the earlier
   baseline checkpoint (b3_training_normalized/checkpoint_50000.pt).
   Running this script on BOTH checkpoints answers directly whether the
   stress improvement came at any cost to the other three QoIs.

2. No deformation-validity (J = det F > 0) check has ever been run for
   B3. The constitutive energies use ln(J), so an inverted element
   (J <= 0) is not just inaccurate, it is outside the model's domain of
   validity. This computes min/1st-percentile/median J and the fraction
   of Gauss points with J <= 0, for BOTH the true FEM field and the
   network's own prediction, over every element in the mesh (not just
   the fixed stress-region subset evaluate_B3_qois.py already checks).

Reuses this project's own existing, already-validated machinery instead
of reimplementing it: evaluate_B3.py's evaluate_accuracy (displacement),
evaluate_B3_qois.py's compute_energy/compute_reaction (energy/reaction),
and the same F = I + B_op @ u_elem construction compute_region_sigma_xx
already uses internally for stress -- applied here to every element, not
only the fixed region, since J-validity is a domain-wide question.

Usage (run once per checkpoint; run it on BOTH the baseline and the
local-refinement checkpoint to get the before/after comparison the paper
needs):
  python -m omar_pfem.evaluate_B3_global_qois_and_jacobian \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_training_local_refine/checkpoint_50000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_training_local_refine/global_qois_and_jacobian.json
"""
import argparse
import json

import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import (
    build_fixed_geometry, apply_dirichlet_b3, build_model, total_potential_energy_B3,
)
from omar_pfem.train_B1 import install_input_norm_for_checkpoint, _apply_input_norm
from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.evaluate_B3 import evaluate_accuracy
from omar_pfem.evaluate_B3_qois import compute_energy, compute_reaction


@torch.no_grad()
def jacobian_diagnostics(u, geom):
    """det(F) at EVERY (element, Gauss point) for a known displacement
    field u, (Batch, n_elements, n_gauss). Same F = I + H construction
    evaluate_B3_qois.py's compute_region_sigma_xx uses, applied to every
    element rather than only the fixed stress region, since deformation
    validity (J>0) is a domain-wide requirement of the hyperelastic
    energy (ln J appears in every constitutive model this project uses),
    not a region-local one."""
    elements = geom["elements"]
    B_op = geom["B"]
    n_gauss = B_op.shape[0]
    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)

    J_per_gauss = []
    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F = eye + H
        J_per_gauss.append(torch.linalg.det(F))  # (B, n_elem)
    return torch.stack(J_per_gauss, dim=2)  # (B, n_elem, n_gauss)


def summarize_J(J):
    """J: (B, n_elem, n_gauss) numpy array -> scalar diagnostics pooled
    over every sample, element, and Gauss point together."""
    flat = J.reshape(-1)
    return {
        "min_J": float(flat.min()),
        "p1_J": float(np.percentile(flat, 1)),
        "median_J": float(np.median(flat)),
        "fraction_J_le_0": float(np.mean(flat <= 0.0)),
        "n_points": int(flat.size),
    }


def main():
    parser = argparse.ArgumentParser(
        "B3 global displacement/energy/reaction QoIs + deformation-validity (J=det F) check, "
        "for one checkpoint against the held-out FEM dataset.")
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
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"  {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node = torch.tensor(h5["E_node"][:], dtype=dtype)
        nu_node = torch.tensor(h5["nu_node"][:], dtype=dtype)
        phi = torch.tensor(h5["phi"][:], dtype=dtype)
        u_true_all = torch.tensor(h5["displacements"][:], dtype=dtype)
    n_samples = E_node.shape[0]
    print(f"  {n_samples} FEM samples")

    # 1) Displacement accuracy (same routine/metric as the baseline checkpoint's
    #    own already-reported 1.88%/1.94%/16.16%/1.42% numbers).
    print("Computing displacement accuracy...")
    accuracy, _ = evaluate_accuracy(
        model, geom, E_node, nu_node, phi, u_true_all, device, dtype,
        eval_batch_size=args.eval_batch_size)
    print(f"  combined={accuracy['mean_rel_L2_combined']:.4f}  "
          f"ux={accuracy['mean_rel_L2_ux']:.4f}  uy={accuracy['mean_rel_L2_uy']:.4f}  "
          f"uz={accuracy['mean_rel_L2_uz']:.4f}")

    # 2) Energy / reaction / J-diagnostics, batched exactly like evaluate_B3_qois.py.
    energy_rows, reaction_rows = [], []
    J_true_all, J_pred_all = [], []
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
            u_net = model(xyz_batch, fun_material)
            u_pred = apply_dirichlet_b3(u_net, geom, phi_b)

        U_true = compute_energy(u_true, E_b, nu_b, geom)
        U_pred = compute_energy(u_pred, E_b, nu_b, geom)
        Fr_true, Mr_true, _ = compute_reaction(u_true, E_b, nu_b, geom)
        Fr_pred, Mr_pred, _ = compute_reaction(u_pred, E_b, nu_b, geom)

        for i in range(end - start):
            energy_rows.append((U_true[i].item(), U_pred[i].item()))
            reaction_rows.append((Mr_true[i].item(), Mr_pred[i].item()))

        J_true_all.append(jacobian_diagnostics(u_true, geom).cpu().numpy())
        J_pred_all.append(jacobian_diagnostics(u_pred, geom).cpu().numpy())
        print(f"  [{end}/{n_samples}] done")

    U_true_arr = np.array([r[0] for r in energy_rows])
    U_pred_arr = np.array([r[1] for r in energy_rows])
    energy_pooled_rms_rel = float(
        np.sqrt(np.mean((U_pred_arr - U_true_arr) ** 2)) / np.sqrt(np.mean(U_true_arr ** 2)))

    M_true_arr = np.array([r[0] for r in reaction_rows])
    M_pred_arr = np.array([r[1] for r in reaction_rows])
    reaction_pooled_rms_rel = float(
        np.sqrt(np.mean((M_pred_arr - M_true_arr) ** 2)) / np.sqrt(np.mean(M_true_arr ** 2)))

    J_true_cat = np.concatenate(J_true_all, axis=0)
    J_pred_cat = np.concatenate(J_pred_all, axis=0)

    summary = {
        "checkpoint": args.checkpoint,
        "displacement": accuracy,
        "energy_pooled_rms_rel": energy_pooled_rms_rel,
        "reaction_moment_y_pooled_rms_rel": reaction_pooled_rms_rel,
        "jacobian_true_FEM": summarize_J(J_true_cat),
        "jacobian_predicted": summarize_J(J_pred_cat),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
