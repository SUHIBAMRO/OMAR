"""
Timing benchmark for the GPU-native FEM solver (gpu_fem_solver.py), to
produce the number the advisor asked for: a GPU-native FEM solve time per
sample. This script reports ONLY the GPU-side measurement -- for the
matching CPU-side number (with the same full breakdown the advisor asked
for: assembly/solution, iteration counts, FP64, tolerances, warm-up,
synchronization), run fem_cost_breakdown.py for the SAME --geometry
--material --N and read the CPU-vs-GPU comparison from its own output.
An earlier version of this script computed a "speedup vs CPU" ratio
against a hardcoded assumed CPU cost of 8.0 s/sample -- exactly the kind
of unverified constant the advisor explicitly rejected ("The constant
value of 8 seconds is not acceptable"). That comparison has been removed
rather than patched with a different guessed number: the only trustworthy
CPU cost is the one fem_cost_breakdown.py actually measures in the same
run it measures the GPU cost, not a number carried over from a different
script or an earlier report.

Run validate_gpu_fem_solver.py FIRST and confirm it reports PASS before
trusting any number from this script -- a fast but wrong solver is not a
useful comparison.

This benchmarks BATCHED solves (many samples' independent random material/
load fields solved simultaneously in one Newton loop, exactly analogous to
how this study's neural-operator training is batched) since that is the
natural way to use a GPU; per-sample cost is wall-clock-time-of-the-batch
divided by batch size. A batch_size=1 run is also reported as the
single-sample latency most directly comparable to "one native FE solve."

Usage:
  python -m omar_pfem.gpu_fem_benchmark --geometry B1 --material neo_hookean \
      --batch_sizes 1,8,32,128 --n_repeats 3 --out_json gpu_fem_timing_B1.json
"""
import argparse
import json
import time
import numpy as np
import torch

from omar_pfem.gpu_fem_solver import (
    precompute_element_params_B1, precompute_element_params_B2,
    solve_batch_gpu_newton_B1, solve_batch_gpu_newton_B2,
    assemble_fext_B1_numpy, assemble_fext_B2_numpy,
)


def build_batch_b1(N, batch_size, material, device, dtype, seed0):
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, generate_random_sample_spatial

    Lx = Ly = 1.0
    nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
    tol = 1e-12
    bottom_nodes = np.where(nodes[:, 1] < tol)[0]

    elem_params_list, fext_list = [], []
    for i in range(batch_size):
        E_interp, nu_interp, ty_interp, *_ = generate_random_sample_spatial(Lx, Ly, N, N, seed=seed0 + i)
        elem_params_list.append(precompute_element_params_B1(nodes, elements, E_interp, nu_interp, material))
        fext_list.append(assemble_fext_B1_numpy(nodes, elements, Ly, ty_interp))

    n_params = len(elem_params_list[0])
    elem_params_batch = tuple(
        torch.tensor(np.stack([ep[k] for ep in elem_params_list]), dtype=dtype, device=device)
        for k in range(n_params)
    )
    fext_batch = torch.tensor(np.stack(fext_list), dtype=dtype, device=device)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    bottom_nodes_t = torch.tensor(bottom_nodes, dtype=torch.long, device=device)

    def solve():
        return solve_batch_gpu_newton_B1(
            xy_t, quad_t, bottom_nodes_t, elem_params_batch, fext_batch,
            material=material, nsteps=10, newton_max=30, tol=1e-7, device=device, dtype=dtype,
        )
    return solve, len(nodes)


def build_batch_b2(N, batch_size, material, device, dtype, seed0):
    from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring, generate_random_sample_ring

    R_in, R_out = 1.0, 2.0
    nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
    tolx = 1e-9
    theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]

    elem_params_list, fext_list = [], []
    for i in range(batch_size):
        E_interp, nu_interp, p_interp, *_ = generate_random_sample_ring(R_in, R_out, N, N, seed=seed0 + i)
        elem_params_list.append(precompute_element_params_B2(nodes, elements, E_interp, nu_interp, material))
        fext_list.append(assemble_fext_B2_numpy(nodes, elements, R_in, p_interp))

    n_params = len(elem_params_list[0])
    elem_params_batch = tuple(
        torch.tensor(np.stack([ep[k] for ep in elem_params_list]), dtype=dtype, device=device)
        for k in range(n_params)
    )
    fext_batch = torch.tensor(np.stack(fext_list), dtype=dtype, device=device)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    theta0_t = torch.tensor(theta0_nodes, dtype=torch.long, device=device)
    thetahalfpi_t = torch.tensor(thetahalfpi_nodes, dtype=torch.long, device=device)

    def solve():
        return solve_batch_gpu_newton_B2(
            xy_t, quad_t, theta0_t, thetahalfpi_t, elem_params_batch, fext_batch,
            material=material, nsteps=10, newton_max=30, tol=1e-7, device=device, dtype=dtype,
        )
    return solve, len(nodes)


def main():
    parser = argparse.ArgumentParser("GPU-native FEM solver timing benchmark")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--N", type=int, default=21, help="Mesh resolution (21 = this study's mesh)")
    parser.add_argument("--batch_sizes", type=str, default="1,8,32,128")
    parser.add_argument("--n_repeats", type=int, default=3,
                         help="Timed repeats per batch size (median reported); each repeat "
                              "regenerates fresh random samples so it is not just re-solving a cached batch")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"Device: {device}, geometry={args.geometry}, material={args.material}, N={args.N}")

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    build_fn = build_batch_b1 if args.geometry == "B1" else build_batch_b2

    rows = []
    for bs in batch_sizes:
        # Untimed warm-up (separate fresh batch, not one of the timed repeats) --
        # first CUDA call in a process pays one-time kernel-compile/context costs
        # that would otherwise leak into repeat 0's measurement, matching
        # fem_cost_breakdown.py's own convention for the CPU-side comparison.
        warmup_solve, _ = build_fn(args.N, bs, args.material, device, dtype, seed0=-1)
        warmup_solve()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times_s = []
        for r in range(args.n_repeats):
            solve, n_nodes = build_fn(args.N, bs, args.material, device, dtype, seed0=1000 * r + 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            solve()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_s.append(time.perf_counter() - t0)
        median_s = float(np.median(times_s))
        per_sample_ms = 1000.0 * median_s / bs
        rows.append({
            "batch_size": bs, "n_nodes": n_nodes,
            "median_batch_time_s": median_s, "per_sample_ms": per_sample_ms,
            "all_repeat_times_s": times_s,
        })
        print(f"batch_size={bs:>4d}  median_batch_time={median_s:.4f}s  "
              f"per_sample={per_sample_ms:.3f}ms  (repeats: {[f'{t:.3f}' for t in times_s]})")

    print("\n" + "=" * 80)
    print(f"GPU-native FEM solve cost per sample (device={device}):")
    for row in rows:
        print(f"  batch_size={row['batch_size']:>4d}: {row['per_sample_ms']:.3f} ms/sample")
    print("For the CPU-vs-GPU speedup, compare this to fem_cost_breakdown.py's OWN "
          "measured CPU time for the same --geometry --material --N -- not a hardcoded "
          "assumption.")
    print("=" * 80)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"geometry": args.geometry, "material": args.material, "N": args.N,
                       "device": str(device), "rows": rows}, f, indent=2)
        print(f"Full report written to {args.out_json}")


if __name__ == "__main__":
    main()
