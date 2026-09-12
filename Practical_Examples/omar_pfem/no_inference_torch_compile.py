"""
torch.compile and TF32 matmul precision as actual optimization attempts
for NO inference at N=1401 (Timon round-10, item 3, the gap a
second-opinion review flagged correctly): the profiling cell
(profile_no_inference_n1401.py) measured and explained WHERE the 2.29s
goes -- ~90% in `aten::bmm`/`aten::einsum`/GEMM kernels -- but never
actually TRIED to make it faster, and Timon's own wording ("check this
before considering the 2.29s as the final inference number") asks for an
attempt, not just a diagnosis.

Why torch.compile: the profiler table showed a striking amount of
kernel-launch/dispatch overhead alongside the dominant GEMM cost -- 4,800
`cudaLaunchKernel` calls and 2,893 "Command Buffer Full" events across
only 30 repeats (160 kernel launches per single forward pass), consuming
64% and 80% of CPU time respectively even though most of that time is
just waiting on the GPU queue. torch.compile's kernel fusion (via
TorchInductor) directly targets exactly this kind of overhead -- fusing
many small ops into fewer, larger kernels -- without changing the
architecture or any weight.

Why TF32 matmul precision: torch.compile's own run on an A100 printed a
warning that TensorFloat32 tensor cores are available but not enabled for
fp32 matmul, and recommended `torch.set_float32_matmul_precision('high')`.
This targets the SAME dominant cost the profiler already found (GEMM/bmm/
einsum), from a different angle -- Ampere's tensor cores can execute fp32
matmuls at TF32 (19-bit mantissa) precision several times faster than
full fp32, at a precision cost roughly comparable to (often better than)
bf16's. Tested both alone (eager+TF32) and combined with torch.compile.

Both are safe to try (no architecture or weight change) and CORRECTNESS
FIRST, same discipline as everywhere else in this project: every variant's
output is compared against the strict-fp32 eager baseline before its
speedup is trusted, and a failure in either is reported honestly rather
than assumed to have worked.
"""
import time

import torch


def profile_with_torch_compile(sample, model, args, device, dtype,
                                n_repeats=200, n_warmup=20, compile_warmup=5,
                                try_tf32=True):
    if device.type != "cuda":
        raise RuntimeError("torch.compile timing needs a CUDA device to mean anything")

    from omar_pfem.train_B1 import predict_displacement_Q4_only

    model.eval()
    xy = torch.tensor(sample["xy"], device=device, dtype=dtype)
    quad = torch.tensor(sample["quad"], device=device, dtype=torch.long)
    top_edges = torch.tensor(sample["top_edges"], device=device, dtype=torch.long)
    bottom_nodes = torch.tensor(sample["bottom_nodes"], device=device, dtype=torch.long)
    E1 = torch.tensor(sample["E_node"], device=device, dtype=dtype).unsqueeze(0)
    nu1 = torch.tensor(sample["nu_node"], device=device, dtype=dtype).unsqueeze(0)
    f1 = torch.tensor(sample["node_forces"], device=device, dtype=dtype).unsqueeze(0)

    def _call(m):
        return predict_displacement_Q4_only(
            xy, quad, top_edges, bottom_nodes, m, E1, nu1, f1,
            use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly,
            dtype=dtype, fun_dim=args.fun_dim,
        )

    def _time_it(fn, n_warmup_local, n_repeats_local):
        with torch.no_grad():
            for _ in range(n_warmup_local):
                fn()
            torch.cuda.synchronize(device)
            t0 = time.time()
            for _ in range(n_repeats_local):
                out = fn()
            torch.cuda.synchronize(device)
            elapsed = time.time() - t0
        return 1000.0 * elapsed / n_repeats_local, out

    print("Timing eager baseline (same protocol as profile_no_inference_n1401.py)...")
    eager_ms, eager_out = _time_it(lambda: _call(model), n_warmup, n_repeats)

    result = {
        "eager_ms_per_sample": eager_ms,
        "compile_succeeded": False,
        "compiled_ms_per_sample": None,
        "compiled_vs_eager_rel_diff": None,
        "speedup_vs_eager": None,
        "error": None,
    }

    print("Compiling model with torch.compile (first call pays compile-time cost, "
          "not counted in the timed result)...")
    try:
        compiled_model = torch.compile(model)
        # torch.compile's actual compilation happens lazily on first call(s) --
        # give it more untimed warm-up than the eager path to absorb that,
        # not just the usual CUDA-context warm-up.
        for _ in range(compile_warmup):
            with torch.no_grad():
                _call(compiled_model)
        torch.cuda.synchronize(device)

        compiled_ms, compiled_out = _time_it(lambda: _call(compiled_model), n_warmup, n_repeats)

        rel_diff = (torch.linalg.norm(compiled_out.float() - eager_out.float())
                    / torch.linalg.norm(eager_out.float()).clamp_min(1e-12)).item()

        result["compile_succeeded"] = True
        result["compiled_ms_per_sample"] = compiled_ms
        result["compiled_vs_eager_rel_diff"] = rel_diff
        result["speedup_vs_eager"] = eager_ms / compiled_ms
    except Exception as e:  # torch.compile can fail in many model-specific ways
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"torch.compile FAILED (reported honestly, not hidden): {result['error']}")

    result.update({
        "eager_tf32_ms_per_sample": None, "eager_tf32_vs_eager_rel_diff": None,
        "speedup_tf32_vs_eager": None,
        "compiled_tf32_ms_per_sample": None, "compiled_tf32_vs_eager_rel_diff": None,
        "speedup_compiled_tf32_vs_eager": None, "tf32_error": None,
    })
    if try_tf32:
        orig_precision = torch.get_float32_matmul_precision()
        try:
            print("\nTiming eager + TF32 matmul precision "
                  "(torch.set_float32_matmul_precision('high'))...")
            torch.set_float32_matmul_precision("high")
            eager_tf32_ms, eager_tf32_out = _time_it(lambda: _call(model), n_warmup, n_repeats)
            result["eager_tf32_ms_per_sample"] = eager_tf32_ms
            result["eager_tf32_vs_eager_rel_diff"] = (
                torch.linalg.norm(eager_tf32_out.float() - eager_out.float())
                / torch.linalg.norm(eager_out.float()).clamp_min(1e-12)).item()
            result["speedup_tf32_vs_eager"] = eager_ms / eager_tf32_ms

            if result["compile_succeeded"]:
                print("Timing torch.compile + TF32 (extra warm-up in case the precision "
                      "change triggers a guard-driven recompile)...")
                for _ in range(compile_warmup):
                    with torch.no_grad():
                        _call(compiled_model)
                torch.cuda.synchronize(device)
                compiled_tf32_ms, compiled_tf32_out = _time_it(
                    lambda: _call(compiled_model), n_warmup, n_repeats)
                result["compiled_tf32_ms_per_sample"] = compiled_tf32_ms
                result["compiled_tf32_vs_eager_rel_diff"] = (
                    torch.linalg.norm(compiled_tf32_out.float() - eager_out.float())
                    / torch.linalg.norm(eager_out.float()).clamp_min(1e-12)).item()
                result["speedup_compiled_tf32_vs_eager"] = eager_ms / compiled_tf32_ms
        except Exception as e:
            result["tf32_error"] = f"{type(e).__name__}: {e}"
            print(f"TF32 test FAILED (reported honestly, not hidden): {result['tf32_error']}")
        finally:
            torch.set_float32_matmul_precision(orig_precision)

    return result
