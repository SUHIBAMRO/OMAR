"""
Detailed inference-time profiling for the physics-informed NO checkpoint at
N=1401 (Timon, round-10, item 3): "I am a bit surprised how slow the NO is
at inference ... Could you please check this before considering the 2.29s
as the final inference number. Can you also report the pure GPU forward-pass
time after warm-up ... the precision used and the peak GPU memory. Ideally,
can you provide some profiling to find out where the most inference time is
spent."

benchmark_inference_latency_Q4 (train_B1.py) already isolates the pure
forward-pass time from data loading (tensors are built once and moved to
the device before the timed loop starts, so nothing measured here includes
data transfer or preprocessing) but does not report precision, peak memory,
or a breakdown of where the time goes inside the forward pass -- this module
adds exactly those three things on top of the same call
(predict_displacement_Q4_only), without changing the existing 2.29s
measurement or its methodology.

CUDA-only: peak-memory and profiler numbers are not meaningful on CPU, and
a bf16-autocast comparison needs a real GPU to say anything about actual
deployment latency.
"""
import time

import torch

from omar_pfem.train_B1 import predict_displacement_Q4_only


def profile_inference_detailed(sample, model, args, device, dtype,
                                n_repeats=200, n_warmup=20,
                                profiler_repeats=30, try_bf16=True):
    if device.type != "cuda":
        raise RuntimeError(
            "profile_inference_detailed needs a CUDA device -- peak-memory "
            "and profiler numbers on CPU would not be comparable to any "
            "real deployment number."
        )

    model.eval()
    xy = torch.tensor(sample["xy"], device=device, dtype=dtype)
    quad = torch.tensor(sample["quad"], device=device, dtype=torch.long)
    top_edges = torch.tensor(sample["top_edges"], device=device, dtype=torch.long)
    bottom_nodes = torch.tensor(sample["bottom_nodes"], device=device, dtype=torch.long)
    E1 = torch.tensor(sample["E_node"], device=device, dtype=dtype).unsqueeze(0)
    nu1 = torch.tensor(sample["nu_node"], device=device, dtype=dtype).unsqueeze(0)
    f1 = torch.tensor(sample["node_forces"], device=device, dtype=dtype).unsqueeze(0)

    def _call():
        return predict_displacement_Q4_only(
            xy, quad, top_edges, bottom_nodes, model, E1, nu1, f1,
            use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly,
            dtype=dtype, fun_dim=args.fun_dim,
        )

    with torch.no_grad():
        # ---- pure forward-pass time, same protocol as benchmark_inference_latency_Q4 ----
        for _ in range(n_warmup):
            _call()
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        for _ in range(n_repeats):
            _call()
        torch.cuda.synchronize(device)
        elapsed_s = time.time() - t0
        peak_bytes = torch.cuda.max_memory_allocated(device)

        # ---- where the time goes ----
        from torch.profiler import ProfilerActivity, profile

        for _ in range(5):
            _call()
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False) as prof:
            for _ in range(profiler_repeats):
                _call()
            torch.cuda.synchronize(device)
        profiler_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

        # ---- bf16 autocast: does the gap close? (diagnostic only, not an accuracy claim) ----
        bf16_ms_per_sample = None
        bf16_rel_diff_vs_fp32 = None
        if try_bf16:
            fp32_ref = _call()
            for _ in range(n_warmup):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _call()
            torch.cuda.synchronize(device)
            t0 = time.time()
            for _ in range(n_repeats):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_bf16 = _call()
            torch.cuda.synchronize(device)
            bf16_elapsed_s = time.time() - t0
            bf16_ms_per_sample = 1000.0 * bf16_elapsed_s / n_repeats
            denom = fp32_ref.float().norm().clamp_min(1e-12)
            bf16_rel_diff_vs_fp32 = ((out_bf16.float() - fp32_ref.float()).norm() / denom).item()

    ms_per_sample = 1000.0 * elapsed_s / n_repeats
    return {
        "inference_ms_per_sample": ms_per_sample,
        "inference_n_repeats": n_repeats,
        "inference_batch_size": 1,
        "inference_device": device.type,
        "gpu_name": torch.cuda.get_device_name(device),
        "precision": str(dtype),
        "peak_memory_mb": peak_bytes / (1024 ** 2),
        "profiler_table_top20_by_cuda_time": profiler_table,
        "bf16_autocast_ms_per_sample": bf16_ms_per_sample,
        "bf16_autocast_rel_diff_vs_fp32": bf16_rel_diff_vs_fp32,
    }
