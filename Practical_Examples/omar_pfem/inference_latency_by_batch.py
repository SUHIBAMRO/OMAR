"""
Transolver inference latency across BATCH SIZES, per the advisor's request:

  "Can you please also benchmark Transolver and GPU FEM under identical
   batch sizes"

The existing `measure_inference_latency.py` deliberately measures batch size 1
only -- the realistic deployment case, where new problem instances arrive one
at a time. That is the right number for a deployment claim, but it is NOT
comparable to the GPU-native finite-element solver, which is benchmarked at
batch sizes 1/8/32/128 (see gpu_fem_benchmark.py) because batching is the only
way a GPU solver amortises kernel-launch overhead across many small solves.
Comparing a batch-1 network against a batch-128 solver flatters whichever side
happens to benefit; this script removes that asymmetry by measuring the
network at exactly the same batch sizes the FEM solver was measured at.

Method mirrors the FEM benchmark so the two are directly comparable:
  - untimed warm-up calls first, to absorb CUDA context / cuDNN autotuning,
  - torch.cuda.synchronize() on both sides of the timed region, since kernel
    launches are asynchronous and an unsynchronised timer measures queueing,
    not execution,
  - median over repeats rather than mean, so one scheduling hiccup does not
    dominate,
  - a real batch of DISTINCT samples (different material/loading fields), not
    the same sample repeated, so nothing can be cached across the batch.

Both per-batch and per-sample timings are reported: per-sample is what enters
the throughput/Pareto comparison, per-batch is what a caller actually waits.

Usage:
  python -m omar_pfem.inference_latency_by_batch \
      --geometry B1 --material neo_hookean \
      --checkpoint .../model_best.pt --dataset .../hyperelastic_training_data_q4.npz \
      --batch_sizes 1,8,32,128 \
      --out_json .../inference_latency_by_batch_B1_neo_hookean.json
"""
import json
import time
import argparse
import statistics

import numpy as np
import torch

from omar_pfem.model_dict import get_model


def build_model(args, device):
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
    ).to(device)


def main():
    p = argparse.ArgumentParser("Transolver inference latency vs batch size")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", required=True,
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ntrain", type=int, default=800)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--batch_sizes", type=str, default="1,8,32,128",
                   help="must match the GPU-FEM benchmark's batch sizes to be comparable")
    p.add_argument("--n_repeats", type=int, default=50)
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unified_pos", type=int, default=0)
    p.add_argument("--ref", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=128)
    p.add_argument("--fun_dim", type=int, default=4)
    p.add_argument("--use_soft_dirichlet", type=int, default=1)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.out_json is None:
        args.out_json = f"inference_latency_by_batch_{args.geometry}_{args.material}.json"
        print(f"[auto-save] --out_json not given; writing to {args.out_json}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    if args.geometry == "B1":
        from omar_pfem.train_B1 import (
            predict_displacement_Q4_only as predict,
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds)
    else:
        from omar_pfem.train_B2 import (
            predict_displacement_Q4_only as predict,
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds)

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    _, test = load_ds(args.dataset, args.ntrain, args.ntest)
    print(f"Test pool: {len(test)} samples (batches are drawn from distinct samples)")

    s0 = test[0]
    xy = torch.tensor(s0["xy"], device=device, dtype=dtype)
    quad = torch.tensor(s0["quad"], device=device, dtype=torch.long)
    if args.geometry == "B1":
        bnd = (torch.tensor(s0["top_edges"], device=device, dtype=torch.long),
               torch.tensor(s0["bottom_nodes"], device=device, dtype=torch.long))
        geo_kw = {"Ly": args.Ly}
    else:
        bnd = (torch.tensor(s0["inner_edges"], device=device, dtype=torch.long),
               torch.tensor(s0["theta0_nodes"], device=device, dtype=torch.long),
               torch.tensor(s0["thetahalfpi_nodes"], device=device, dtype=torch.long))
        geo_kw = {"R_out": args.R_out}

    rows = []
    for B in batch_sizes:
        # distinct samples, cycled if the pool is smaller than the batch
        pick = [test[i % len(test)] for i in range(B)]
        E_b = torch.tensor(np.stack([s["E_node"] for s in pick]), device=device, dtype=dtype)
        nu_b = torch.tensor(np.stack([s["nu_node"] for s in pick]), device=device, dtype=dtype)
        f_b = torch.tensor(np.stack([s["node_forces"] for s in pick]), device=device, dtype=dtype)

        def one_call():
            with torch.no_grad():
                return predict(xy, quad, *bnd, model, E_b, nu_b, f_b,
                               use_soft_dirichlet=args.use_soft_dirichlet,
                               dtype=dtype, fun_dim=args.fun_dim, **geo_kw)

        for _ in range(args.n_warmup):
            one_call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times = []
        for _ in range(args.n_repeats):
            t0 = time.perf_counter()
            one_call()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

        med = statistics.median(times)
        rows.append({"batch_size": B,
                     "median_batch_time_s": med,
                     "per_sample_ms": 1000.0 * med / B,
                     "per_batch_ms": 1000.0 * med,
                     "n_repeats": args.n_repeats})
        print(f"  bs={B:>4d}: {1000.0 * med / B:>9.4f} ms/sample   "
              f"({1000.0 * med:>9.3f} ms/batch)")

    report = {"geometry": args.geometry, "material": args.material,
              "checkpoint": args.checkpoint, "device": device.type,
              "n_nodes": int(xy.shape[0]), "rows": rows}
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
