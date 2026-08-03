"""
Exact, fully-instrumented CPU vs. GPU FEM cost breakdown, per the advisor's
fourth-round request: a single constant "8.0 s/sample" is not an acceptable
efficiency comparison. This script replaces it with a measured, itemized
breakdown for both the CPU reference solver (data_generate_B1.py /
data_generate_B2.py) and the GPU-native solver (gpu_fem_solver.py), at the
study's mesh resolution (N=21 by default), reporting exactly what the
advisor asked for:

  - Wall-clock time split into: field generation (GRF sampling + interpolator
    setup), assembly (element loop / residual+tangent autodiff evaluation),
    and linear/nonlinear solve (spsolve on CPU, torch.linalg.solve on GPU).
    Disk I/O (dataset .h5/.npz reads/writes) is explicitly NOT part of any
    number here -- a single-sample cost benchmark never touches disk; this
    is stated in the output so nothing is ambiguous about what is timed.
  - Newton iteration count, load-increment count, Newton tolerance,
    linear-solver identity, numerical precision (dtype).
  - GPU timing is bracketed by torch.cuda.synchronize() on both sides of
    every timed block (CUDA kernel launches are asynchronous; without this,
    a "timed" block can return before the GPU has actually finished).
  - An explicit, untimed warm-up solve is run before any GPU timing to
    absorb one-time CUDA context / kernel especialization / torch.func
    tracing overhead, so it never leaks into the reported numbers.
  - Hardware (CPU model, GPU model) and software (Python/NumPy/SciPy/
    PyTorch/CUDA) versions, so the numbers are reproducible and auditable.
  - A best-effort FLOPs ESTIMATE for both solvers (explicitly labeled as an
    analytical estimate, not a hardware-profiler measurement -- see
    docstring of estimate_flops_* below for the exact counting methodology
    and its limitations). This is the one piece the advisor marked
    "ideally", i.e. ov a lower priority than the rest of this script.

Usage (run once per geometry x material; defaults to this study's N=21 mesh):
  python -m omar_pfem.fem_cost_breakdown --geometry B1 --material neo_hookean \
      --n_repeats 5 --gpu_batch_sizes 1,8,32,128 \
      --out_json fem_cost_breakdown_B1_neo_hookean.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import torch


# ============================================================
# FLOPs estimate (analytical -- see module docstring)
# ============================================================

def estimate_flops_per_element_gauss_point(material):
    """Very-rough FLOP count for ONE (element, Gauss point) evaluation of
    the Q4 kinematics (shape-function gradients, F, and the resulting P, A)
    for the Total-Lagrangian formulation shared by both solvers. This is a
    hand count of the dominant matrix/vector operations in
    fem_core.element_K_and_fint_TL and the material-model P/A evaluation,
    NOT a profiler measurement -- it is meant only to give an
    order-of-magnitude sense of arithmetic intensity, and is intentionally
    conservative (undercounts transcendental-function cost, e.g. log/exp in
    Neo-Hookean, as a small constant rather than their true relative cost)."""
    # kinematics: J0 (2x2 from 4 nodes), invJ0, dN_dX, F -- a few hundred flops
    kinematics_flops = 250
    # material model: Neo-Hookean is a closed form (~100 flops for P, ~300 for A);
    # Mooney-Rivlin / Arruda-Boyce go through JAX autodiff on the CPU reference
    # solver, whose own internal graph is materially larger -- taken here as a
    # flat ~3x multiplier on Neo-Hookean's hand-derived cost as a rough proxy,
    # not a measured ratio.
    base_material_flops = 100 + 300
    material_multiplier = {"neo_hookean": 1.0, "mooney_rivlin": 3.0, "arruda_boyce": 3.0}
    material_flops = base_material_flops * material_multiplier.get(material, 1.0)
    # internal-force / stiffness assembly contraction for this Gauss point (8x8 ke, 8 fe)
    assembly_flops = 8 * 8 * 4 * 2 + 8 * 4 * 2  # A_ijkl contraction + P.gNa contraction, rough
    return kinematics_flops + material_flops + assembly_flops


def estimate_total_flops(n_elements, n_gauss, n_newton_iters, material, include_solve_flops):
    """Total estimated FLOPs across the whole solve: elements x Gauss points
    x Newton iterations for assembly, plus (optionally) an O(n^3)-scaling
    estimate for the linear solve, using n = n_free as a stand-in for matrix
    size when the caller supplies it via include_solve_flops (a dict with
    'n_free' and 'kind' in {'sparse_direct','dense'})."""
    assembly_flops = n_elements * n_gauss * n_newton_iters * \
        estimate_flops_per_element_gauss_point(material)
    solve_flops = None
    if include_solve_flops is not None:
        n_free = include_solve_flops["n_free"]
        if include_solve_flops["kind"] == "dense":
            # dense LU solve: ~ (2/3) n^3 for factorization, per Newton iteration
            solve_flops = n_newton_iters * (2.0 / 3.0) * n_free ** 3
        else:
            # sparse direct solve on a banded/near-banded 2D Q4 mesh matrix:
            # very roughly O(n^1.5) fill-in for a 2D problem (nested dissection),
            # reported as an order-of-magnitude estimate only.
            solve_flops = n_newton_iters * n_free ** 1.5
    return assembly_flops, solve_flops


# ============================================================
# Hardware / software metadata
# ============================================================

def get_cpu_model():
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        return platform.processor() or platform.machine()
    except Exception as e:
        return f"unknown ({e})"


def get_environment_metadata(device):
    meta = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "cuda_version_torch_built_with": torch.version.cuda,
        "cpu_model": get_cpu_model(),
        "cpu_count_logical": __import__("os").cpu_count(),
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_mb": (torch.cuda.get_device_properties(device).total_memory / 1e6
                                 if device.type == "cuda" else None),
    }
    return meta


# ============================================================
# CPU reference solve, profiled, one representative sample
# ============================================================

def run_cpu_case(geometry, material, N, n_repeats):
    if geometry == "B1":
        from omar_pfem.data.data_generate_B1 import (
            generate_grid_Q4, generate_random_sample_spatial, solve_hyperelastic_TL_spatial,
        )
        Lx = Ly = 1.0
        nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
        solves = []
        for r in range(n_repeats):
            t_field0 = time.perf_counter()
            E_interp, nu_interp, ty_interp, *_ = generate_random_sample_spatial(Lx, Ly, N, N, seed=1000 * r + 1)
            t_field_s = time.perf_counter() - t_field0

            profile = {}
            t0 = time.perf_counter()
            solve_hyperelastic_TL_spatial(nodes, elements, E_interp, nu_interp, ty_interp, Ly,
                                           nsteps=10, newton_max=30, tol=1e-7,
                                           material=material, profile=profile)
            t_total_s = time.perf_counter() - t0
            profile["t_field_generation_s"] = t_field_s
            profile["t_total_s"] = t_total_s
            solves.append(profile)
    else:
        from omar_pfem.data.data_generate_B2 import (
            generate_grid_Q4_ring, generate_random_sample_ring, solve_hyperelastic_TL_ring,
        )
        R_in, R_out = 1.0, 2.0
        nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
        solves = []
        for r in range(n_repeats):
            t_field0 = time.perf_counter()
            E_interp, nu_interp, p_interp, *_ = generate_random_sample_ring(R_in, R_out, N, N, seed=1000 * r + 1)
            t_field_s = time.perf_counter() - t_field0

            profile = {}
            t0 = time.perf_counter()
            solve_hyperelastic_TL_ring(nodes, elements, E_interp, nu_interp, p_interp, R_in,
                                        nsteps=10, newton_max=30, tol=1e-7,
                                        material=material, profile=profile)
            t_total_s = time.perf_counter() - t0
            profile["t_field_generation_s"] = t_field_s
            profile["t_total_s"] = t_total_s
            solves.append(profile)

    n_free = 2 * len(nodes) - (len(np.where(np.abs(nodes[:, 1]) < 1e-9)[0]) * 2 if geometry == "B1" else 0)
    return solves, len(nodes), len(elements)


# ============================================================
# GPU solve, profiled, explicit warm-up, one representative sample
# ============================================================

def run_gpu_case(geometry, material, N, n_repeats, batch_sizes, device):
    from omar_pfem.gpu_fem_solver import (
        precompute_element_params_B1, precompute_element_params_B2,
        solve_batch_gpu_newton_B1, solve_batch_gpu_newton_B2,
        assemble_fext_B1_numpy, assemble_fext_B2_numpy,
    )
    dtype = torch.float64

    def build(bs, seed0):
        if geometry == "B1":
            from omar_pfem.data.data_generate_B1 import generate_grid_Q4, generate_random_sample_spatial
            Lx = Ly = 1.0
            nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
            bottom_nodes = np.where(nodes[:, 1] < 1e-12)[0]
            elem_params_list, fext_list = [], []
            for i in range(bs):
                E_interp, nu_interp, ty_interp, *_ = generate_random_sample_spatial(Lx, Ly, N, N, seed=seed0 + i)
                elem_params_list.append(precompute_element_params_B1(nodes, elements, E_interp, nu_interp, material))
                fext_list.append(assemble_fext_B1_numpy(nodes, elements, Ly, ty_interp))
            n_params = len(elem_params_list[0])
            elem_params_batch = tuple(torch.tensor(np.stack([ep[k] for ep in elem_params_list]),
                                                     dtype=dtype, device=device) for k in range(n_params))
            fext_batch = torch.tensor(np.stack(fext_list), dtype=dtype, device=device)
            xy_t = torch.tensor(nodes, dtype=dtype, device=device)
            quad_t = torch.tensor(elements, dtype=torch.long, device=device)
            bottom_t = torch.tensor(bottom_nodes, dtype=torch.long, device=device)

            def solve(profile=None):
                return solve_batch_gpu_newton_B1(xy_t, quad_t, bottom_t, elem_params_batch, fext_batch,
                                                  material=material, nsteps=10, newton_max=30, tol=1e-7,
                                                  device=device, dtype=dtype, profile=profile)
        else:
            from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring, generate_random_sample_ring
            R_in, R_out = 1.0, 2.0
            nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
            theta0_nodes = np.where(np.abs(nodes[:, 1]) < 1e-9)[0]
            thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < 1e-9)[0]
            elem_params_list, fext_list = [], []
            for i in range(bs):
                E_interp, nu_interp, p_interp, *_ = generate_random_sample_ring(R_in, R_out, N, N, seed=seed0 + i)
                elem_params_list.append(precompute_element_params_B2(nodes, elements, E_interp, nu_interp, material))
                fext_list.append(assemble_fext_B2_numpy(nodes, elements, R_in, p_interp))
            n_params = len(elem_params_list[0])
            elem_params_batch = tuple(torch.tensor(np.stack([ep[k] for ep in elem_params_list]),
                                                     dtype=dtype, device=device) for k in range(n_params))
            fext_batch = torch.tensor(np.stack(fext_list), dtype=dtype, device=device)
            xy_t = torch.tensor(nodes, dtype=dtype, device=device)
            quad_t = torch.tensor(elements, dtype=torch.long, device=device)
            theta0_t = torch.tensor(theta0_nodes, dtype=torch.long, device=device)
            thetahalfpi_t = torch.tensor(thetahalfpi_nodes, dtype=torch.long, device=device)

            def solve(profile=None):
                return solve_batch_gpu_newton_B2(xy_t, quad_t, theta0_t, thetahalfpi_t, elem_params_batch,
                                                  fext_batch, material=material, nsteps=10, newton_max=30,
                                                  tol=1e-7, device=device, dtype=dtype, profile=profile)
        return solve, len(nodes), len(elements)

    # Explicit, UNTIMED warm-up: absorbs CUDA context creation, kernel
    # specialization, and torch.func (grad/hessian/vmap) tracing overhead,
    # none of which should count as "solver cost" for a warmed-up service.
    warmup_solve, _, _ = build(batch_sizes[0], seed0=999_000)
    warmup_solve(profile=None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    results_by_bs = {}
    for bs in batch_sizes:
        repeats = []
        n_nodes = n_elements = None
        for r in range(n_repeats):
            solve, n_nodes, n_elements = build(bs, seed0=1000 * r + 1)
            profile = {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            solve(profile=profile)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            profile["t_total_s"] = time.perf_counter() - t0
            repeats.append(profile)
        results_by_bs[bs] = {"repeats": repeats, "n_nodes": n_nodes, "n_elements": n_elements}
    return results_by_bs


def median_profile(repeats, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in repeats if k in r]
        out[k] = float(np.median(vals)) if vals else None
    return out


def main():
    parser = argparse.ArgumentParser("Exact CPU vs. GPU FEM cost breakdown")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--N", type=int, default=21, help="Mesh resolution (21 = this study's mesh)")
    parser.add_argument("--n_repeats", type=int, default=5,
                         help="Timed repeats (median reported); each regenerates a fresh random "
                              "sample, so this is not just re-solving a cached instance")
    parser.add_argument("--gpu_batch_sizes", type=str, default="1,8,32,128")
    parser.add_argument("--skip_cpu", action="store_true")
    parser.add_argument("--skip_gpu", action="store_true")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = get_environment_metadata(device)
    print(f"Environment: {json.dumps(env, indent=2)}")

    report = {
        "geometry": args.geometry, "material": args.material, "N": args.N,
        "n_repeats": args.n_repeats, "environment": env,
        "notes": {
            "precision": "float64 (FP64) throughout, both CPU (NumPy default) and GPU (torch.float64 explicit)",
            "disk_io": "NOT included in any timed number below -- a single-sample cost benchmark "
                       "never reads/writes a dataset file; only in-memory field generation and the "
                       "solve itself are timed, and those are reported as separate buckets",
            "gpu_synchronization": "every GPU-timed block is bracketed by torch.cuda.synchronize() "
                                   "on both sides, since CUDA kernel launches are asynchronous",
            "gpu_warmup": "one full untimed solve is run before any GPU timing to absorb CUDA "
                          "context creation, kernel specialization, and torch.func "
                          "(grad/hessian/vmap) tracing overhead",
        },
    }

    if not args.skip_cpu:
        solves, n_nodes, n_elements = run_cpu_case(args.geometry, args.material, args.N, args.n_repeats)
        med = median_profile(solves, ["t_field_generation_s", "t_assembly_s", "t_solve_s", "t_total_s"])
        n_free = 2 * n_nodes  # BC elimination removes only a small edge fraction; used as FLOPs proxy only
        assembly_flops, solve_flops = estimate_total_flops(
            n_elements, n_gauss=4, n_newton_iters=solves[0]["n_newton_iters_total"],
            material=args.material, include_solve_flops={"n_free": n_free, "kind": "sparse_direct"})
        report["cpu"] = {
            "device": "cpu", "n_nodes": n_nodes, "n_elements": n_elements,
            "newton_tol": solves[0]["newton_tol"], "newton_max": solves[0]["newton_max"],
            "n_load_steps": solves[0]["n_load_steps"],
            "n_newton_iters_total_median": float(np.median([s["n_newton_iters_total"] for s in solves])),
            "linear_solver": solves[0]["linear_solver"],
            "dtype": solves[0]["dtype"],
            "median_times_s": med,
            "all_repeats_s": [{k: s[k] for k in ("t_field_generation_s", "t_assembly_s", "t_solve_s", "t_total_s")}
                              for s in solves],
            "flops_estimate": {
                "assembly_flops": assembly_flops, "solve_flops": solve_flops,
                "methodology": "analytical hand-count, see estimate_flops_per_element_gauss_point() "
                               "docstring -- an order-of-magnitude estimate, not a profiler measurement",
            },
        }
        print(f"CPU median total time: {med['t_total_s']:.4f} s "
              f"(assembly={med['t_assembly_s']:.4f}s, solve={med['t_solve_s']:.4f}s, "
              f"field_gen={med['t_field_generation_s']:.4f}s)")

    if not args.skip_gpu and device.type == "cuda":
        batch_sizes = [int(b) for b in args.gpu_batch_sizes.split(",") if b.strip()]
        results_by_bs = run_gpu_case(args.geometry, args.material, args.N, args.n_repeats, batch_sizes, device)
        report["gpu"] = {"device": str(device), "by_batch_size": {}}
        for bs, data in results_by_bs.items():
            med = median_profile(data["repeats"], ["t_assembly_s", "t_solve_s", "t_total_s"])
            n_free = 2 * data["n_nodes"]
            n_iters = data["repeats"][0]["n_newton_iters_total"]
            assembly_flops, solve_flops = estimate_total_flops(
                data["n_elements"] * bs, n_gauss=4, n_newton_iters=n_iters,
                material=args.material, include_solve_flops={"n_free": n_free, "kind": "dense"})
            per_sample_ms = 1000.0 * med["t_total_s"] / bs
            report["gpu"]["by_batch_size"][str(bs)] = {
                "n_nodes": data["n_nodes"], "n_elements": data["n_elements"],
                "newton_tol": data["repeats"][0]["newton_tol"], "newton_max": data["repeats"][0]["newton_max"],
                "n_load_steps": data["repeats"][0]["n_load_steps"],
                "n_newton_iters_total": n_iters,
                "linear_solver": data["repeats"][0]["linear_solver"],
                "dtype": data["repeats"][0]["dtype"],
                "median_times_s": med, "per_sample_ms": per_sample_ms,
                "all_repeats_s": [{k: r[k] for k in ("t_assembly_s", "t_solve_s", "t_total_s")}
                                  for r in data["repeats"]],
                "flops_estimate": {
                    "assembly_flops_whole_batch": assembly_flops, "solve_flops_whole_batch": solve_flops,
                    "methodology": "analytical hand-count (dense per-sample linear solve, O(n_free^3) "
                                   "per Newton iteration) -- an order-of-magnitude estimate",
                },
            }
            print(f"GPU batch_size={bs:>4d}: {per_sample_ms:.4f} ms/sample "
                  f"(assembly={med['t_assembly_s']*1000/bs:.4f}ms/sample, "
                  f"solve={med['t_solve_s']*1000/bs:.4f}ms/sample)")
    elif not args.skip_gpu:
        print("No CUDA device available -- skipping GPU benchmark.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Full report written to {args.out_json}")


if __name__ == "__main__":
    main()
