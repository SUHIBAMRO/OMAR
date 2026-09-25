"""Evaluate a trained B3 Transolver checkpoint (train_B3.py) against the
real, held-out FEM dataset (data_generate_B3_dataset.py's dataset.h5).

This is the missing half of the B3 pipeline: train_B3.py's Deep Energy
Method loss (Pi = U) never sees a single labeled FEM displacement -- a
healthy, stable loss curve during training only proves the optimization
did not diverge, NOT that the learned field is close to the true
solution. This script is the actual accuracy check, matching B1/B2's own
methodology (train_B2.py's evaluate_dataset_hyperelastic_Q4): per-sample
relative L2 error, mean+std across the held-out set, computed against
real FEM ground truth the model never trained on.

Two things are measured:
  1. Accuracy: per-component (ux,uy,uz) and combined relative L2 error of
     the trained model's prediction vs the FEM dataset's own recorded
     displacements, evaluated at the SAME (E_node, nu_node, phi) each FEM
     sample actually used (not re-sampled), so this is a true
     apples-to-apples comparison, not a distribution match.
  2. Latency: pure forward-pass inference time (batch_size=1, no
     backward/no energy assembly), same protocol as
     measure_inference_latency.py/benchmark_inference_latency_Q4 (warmup
     + repeated timed calls, wall-clock via time.time() with
     cuda.synchronize when on GPU) -- reported alongside the dataset's
     own recorded FEM solve time (elapsed_s) for a direct per-sample
     speed comparison.

Usage:
  python -m omar_pfem.evaluate_B3 \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_training/checkpoint_2000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_training/eval_B3.json
"""
import argparse
import json
import time

import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import build_fixed_geometry, apply_dirichlet_b3, build_model
from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION


@torch.no_grad()
def evaluate_accuracy(model, geom, E_node, nu_node, phi, u_exact, device, dtype, eval_batch_size=16):
    """Batched relative-L2 accuracy check, per component and combined,
    matching train_B2.py's evaluate_dataset_hyperelastic_Q4 in spirit
    (mean rel_L2 per component + std across samples), extended to B3's
    3 displacement components."""
    model.eval()
    n = E_node.shape[0]
    xyz = geom["nodes"]
    rel_l2 = {"ux": [], "uy": [], "uz": [], "combined": []}

    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        E_batch = E_node[start:end].to(device=device, dtype=dtype)
        nu_batch = nu_node[start:end].to(device=device, dtype=dtype)
        phi_batch = phi[start:end].to(device=device, dtype=dtype)
        u_exact_batch = u_exact[start:end].to(device=device, dtype=dtype)
        b = end - start

        fun_material = torch.stack(
            [E_batch, nu_batch, phi_batch[:, None].expand(-1, geom["n_nodes"])], dim=2)
        xyz_batch = xyz.unsqueeze(0).expand(b, -1, -1)
        u_net = model(xyz_batch, fun_material)
        u_pred = apply_dirichlet_b3(u_net, geom, phi_batch)

        err = u_pred - u_exact_batch
        for k, comp in enumerate(("ux", "uy", "uz")):
            l2 = torch.sqrt(torch.mean(err[:, :, k] ** 2, dim=1))
            ref = torch.sqrt(torch.mean(u_exact_batch[:, :, k] ** 2, dim=1)) + 1e-12
            rel_l2[comp].extend((l2 / ref).tolist())

        l2_all = torch.sqrt(torch.mean(torch.sum(err ** 2, dim=2), dim=1))
        ref_all = torch.sqrt(torch.mean(torch.sum(u_exact_batch ** 2, dim=2), dim=1)) + 1e-12
        rel_l2["combined"].extend((l2_all / ref_all).tolist())

    out = {}
    for comp, vals in rel_l2.items():
        out[f"mean_rel_L2_{comp}"] = float(np.mean(vals))
        out[f"std_rel_L2_{comp}"] = float(np.std(vals))
    out["n_samples"] = n
    return out, rel_l2


@torch.no_grad()
def benchmark_inference_latency_B3(model, geom, E_node, nu_node, phi, device, dtype,
                                    n_repeats=200, n_warmup=20):
    """Pure inference latency, batch_size=1, forward pass only -- same
    protocol as benchmark_inference_latency_Q4 in train_B1.py/train_B2.py."""
    model.eval()
    xyz = geom["nodes"].unsqueeze(0)
    E1 = E_node[0:1].to(device=device, dtype=dtype)
    nu1 = nu_node[0:1].to(device=device, dtype=dtype)
    phi1 = phi[0:1].to(device=device, dtype=dtype)
    fun_material = torch.stack([E1, nu1, phi1[:, None].expand(-1, geom["n_nodes"])], dim=2)

    def _one_call():
        u_net = model(xyz, fun_material)
        return apply_dirichlet_b3(u_net, geom, phi1)

    for _ in range(n_warmup):
        _one_call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.time()
    for _ in range(n_repeats):
        _one_call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.time() - t0

    ms_per_sample = 1000.0 * elapsed_s / n_repeats
    return {
        "inference_ms_per_sample": ms_per_sample,
        "inference_n_repeats": n_repeats,
        "inference_batch_size": 1,
        "inference_device": device.type,
    }


def main():
    parser = argparse.ArgumentParser(
        "Evaluate a trained B3 Transolver checkpoint against the real FEM validation dataset.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                         help="dataset.h5 produced by data_generate_B3_dataset.py")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--n_repeats", type=int, default=200)
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"),
                         help="Override DEFAULT_RESOLUTION -- must match whatever resolution the "
                              "checkpoint was actually trained at and the dataset was solved at "
                              "(defaults to DEFAULT_RESOLUTION, the production setting).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64  # matches train_B3.py's training dtype exactly

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    print(f"  Checkpoint from iteration {ckpt['iter']}, model={ckpt_args.model}, "
          f"n_hidden={ckpt_args.n_hidden}, n_layers={ckpt_args.n_layers}")

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    print(f"Building fixed geometry at resolution=({Ntheta},{Nr},{Nz}) "
          f"(must match the resolution the checkpoint was trained at)...")
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"  {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node = torch.tensor(h5["E_node"][:], dtype=dtype)
        nu_node = torch.tensor(h5["nu_node"][:], dtype=dtype)
        phi = torch.tensor(h5["phi"][:], dtype=dtype)
        u_exact = torch.tensor(h5["displacements"][:], dtype=dtype)
        fem_elapsed_s = np.array(h5["elapsed_s"][:])

    assert E_node.shape[1] == geom["n_nodes"], (
        f"dataset n_nodes ({E_node.shape[1]}) != geometry n_nodes ({geom['n_nodes']}) -- "
        f"dataset was generated at a different resolution than DEFAULT_RESOLUTION")
    n_samples = E_node.shape[0]
    print(f"  {n_samples} FEM samples, {E_node.shape[1]} nodes each, "
          f"mean FEM solve time = {fem_elapsed_s.mean():.3f}s/sample")

    print("\nRunning accuracy evaluation (real FEM ground truth, held out from training)...")
    accuracy, rel_l2_raw = evaluate_accuracy(
        model, geom, E_node, nu_node, phi, u_exact, device, dtype,
        eval_batch_size=args.eval_batch_size)
    print(f"  mean rel L2: ux={accuracy['mean_rel_L2_ux']:.4f} (std={accuracy['std_rel_L2_ux']:.4f}), "
          f"uy={accuracy['mean_rel_L2_uy']:.4f} (std={accuracy['std_rel_L2_uy']:.4f}), "
          f"uz={accuracy['mean_rel_L2_uz']:.4f} (std={accuracy['std_rel_L2_uz']:.4f}), "
          f"combined={accuracy['mean_rel_L2_combined']:.4f} (std={accuracy['std_rel_L2_combined']:.4f})")
    worst_idx = int(np.argmax(rel_l2_raw["combined"]))
    print(f"  worst sample: index {worst_idx}, combined rel L2 = {rel_l2_raw['combined'][worst_idx]:.4f}")

    print("\nBenchmarking inference latency...")
    latency = benchmark_inference_latency_B3(
        model, geom, E_node, nu_node, phi, device, dtype,
        n_repeats=args.n_repeats, n_warmup=args.n_warmup)
    print(f"  {latency['inference_ms_per_sample']:.4f} ms/sample "
          f"(batch_size=1, {args.n_repeats} repeats, device={device.type})")
    fem_ms_per_sample = float(fem_elapsed_s.mean()) * 1000.0
    speedup = fem_ms_per_sample / latency["inference_ms_per_sample"]
    print(f"  FEM solve time: {fem_ms_per_sample:.1f} ms/sample (mean over {n_samples} samples)")
    print(f"  Speedup (FEM / Transolver inference): {speedup:.1f}x")

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_iter": ckpt["iter"],
        "dataset": args.dataset,
        "n_fem_samples": n_samples,
        "resolution": [Ntheta, Nr, Nz],
        "accuracy": accuracy,
        "latency": latency,
        "fem_mean_elapsed_s": float(fem_elapsed_s.mean()),
        "fem_ms_per_sample": fem_ms_per_sample,
        "speedup_fem_over_transolver": speedup,
        "rel_l2_per_sample": rel_l2_raw,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
