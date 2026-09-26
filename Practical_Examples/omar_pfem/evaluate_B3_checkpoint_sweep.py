"""Evaluates EVERY saved B3 checkpoint (5000, 10000, ..., 50000 from the
third training run) against a real FEM held-out set, to find the actual
best checkpoint -- not assume the final one (50000) is best.

Why this exists: Omar's methodology review (2026-09-26) pointed out that
train_B3.py has no validation/checkpoint-selection loop during training
at all (unlike train_B1.py/train_B2.py's own model_best.pt mechanism),
and that this project's own history has a real precedent (a past B2
investigation) where an early, seemingly-structural "stuck at ~1.0"
finding turned out to be caused by evaluating an undertrained/wrongly-
selected checkpoint, not a real limitation. Since checkpoint_5000.pt
through checkpoint_50000.pt are all already saved (ckpt_every=5000 in
the third run), finding the real best one costs only inference time, no
new training.

Reuses evaluate_B3.py's own evaluate_accuracy/benchmark_inference_latency_B3
functions unchanged -- this script only adds the sweep over checkpoints
and a summary table, run against a dataset argument (intended to be the
clean, genuinely disjoint held-out set from
data_generate_B3_dataset.generate_dataset(seed=99999), not the original
seed=0 dataset whose seeds overlap 92/100 with real training inputs).

Usage:
  python -m omar_pfem.evaluate_B3_checkpoint_sweep \
      --checkpoint_dir /content/drive/MyDrive/pfem_run/b3_training \
      --iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_training/checkpoint_sweep.json
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import build_fixed_geometry, build_model
from omar_pfem.train_B1 import install_input_norm_for_checkpoint
from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.evaluate_B3 import evaluate_accuracy, benchmark_inference_latency_B3


def main():
    parser = argparse.ArgumentParser(
        "Evaluate every saved B3 checkpoint against a real FEM held-out set to find the real best one.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="Directory containing checkpoint_{iter}.pt files")
    parser.add_argument("--iters", type=int, nargs="+", required=True,
                         help="Iteration numbers to evaluate, e.g. 5000 10000 ... 50000")
    parser.add_argument("--dataset", type=str, required=True,
                         help="dataset.h5 -- use the CLEAN held-out set (seed=99999), not the "
                              "original seed=0 one (its seeds overlap 92/100 with training inputs)")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--n_repeats", type=int, default=200)
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    print(f"Building fixed geometry at resolution=({Ntheta},{Nr},{Nz})...")
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"  {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node = torch.tensor(h5["E_node"][:], dtype=dtype)
        nu_node = torch.tensor(h5["nu_node"][:], dtype=dtype)
        phi = torch.tensor(h5["phi"][:], dtype=dtype)
        u_exact = torch.tensor(h5["displacements"][:], dtype=dtype)
        fem_elapsed_s = np.array(h5["elapsed_s"][:])

    assert E_node.shape[1] == geom["n_nodes"], (
        f"dataset n_nodes ({E_node.shape[1]}) != geometry n_nodes ({geom['n_nodes']}) -- "
        f"dataset was generated at a different resolution")
    n_samples = E_node.shape[0]
    print(f"  {n_samples} FEM samples, mean FEM solve time = {fem_elapsed_s.mean():.3f}s/sample\n")

    rows = []
    for it in args.iters:
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{it}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[{it}] SKIPPED -- not found: {ckpt_path}")
            continue
        install_input_norm_for_checkpoint(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_args = argparse.Namespace(**ckpt["args"])
        model = build_model(ckpt_args, device).to(dtype)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        accuracy, _ = evaluate_accuracy(model, geom, E_node, nu_node, phi, u_exact, device, dtype,
                                         eval_batch_size=args.eval_batch_size)
        row = {"iter": it, **accuracy}
        rows.append(row)
        print(f"[{it:>6}] ux={accuracy['mean_rel_L2_ux']:.4f} uy={accuracy['mean_rel_L2_uy']:.4f} "
              f"uz={accuracy['mean_rel_L2_uz']:.4f} combined={accuracy['mean_rel_L2_combined']:.4f}")

    if not rows:
        print("No checkpoints evaluated -- nothing to report.")
        return

    best = min(rows, key=lambda r: r["mean_rel_L2_combined"])
    print(f"\n{'=' * 90}")
    print(f"BEST checkpoint by combined rel L2: iteration {best['iter']} "
          f"(combined={best['mean_rel_L2_combined']:.4f}, ux={best['mean_rel_L2_ux']:.4f}, "
          f"uy={best['mean_rel_L2_uy']:.4f}, uz={best['mean_rel_L2_uz']:.4f})")
    if best["iter"] == max(r["iter"] for r in rows):
        print("(This is the final/longest-trained checkpoint -- consistent with 'train longer is better'.)")
    else:
        print("(This is NOT the final checkpoint -- later training made things worse on this metric; "
              "a real early-stopping signal train_B3.py currently has no way to catch on its own.)")

    print("\nRunning inference-latency benchmark on the best checkpoint...")
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{best['iter']}.pt")
    install_input_norm_for_checkpoint(best_ckpt_path)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    latency = benchmark_inference_latency_B3(model, geom, E_node, nu_node, phi, device, dtype,
                                              n_repeats=args.n_repeats, n_warmup=args.n_warmup)
    fem_ms_per_sample = float(fem_elapsed_s.mean()) * 1000.0
    speedup = fem_ms_per_sample / latency["inference_ms_per_sample"]
    print(f"  {latency['inference_ms_per_sample']:.4f} ms/sample, "
          f"speedup vs FEM = {speedup:.1f}x")

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "dataset": args.dataset,
        "n_fem_samples": n_samples,
        "resolution": [Ntheta, Nr, Nz],
        "sweep": rows,
        "best": best,
        "best_checkpoint_latency": latency,
        "fem_mean_elapsed_s": float(fem_elapsed_s.mean()),
        "speedup_fem_over_transolver_at_best": speedup,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
