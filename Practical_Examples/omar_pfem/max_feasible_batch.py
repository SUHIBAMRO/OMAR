"""
Shared utility for Timon round-10 item 2: "report the maximum feasible
batch size and throughput (samples/s) for both approaches at the same GPU
memory. I expect that the NO should benefit from batching but this should
be demonstrated."

Works by doubling batch size until either a real CUDA OOM (the allocator
itself refusing, not a guessed number) or peak memory exceeds an explicit
--mem_budget_gb cap, so "max feasible" means "actually ran successfully at
this size on this GPU", not extrapolated from a smaller size.
"""
import torch


def find_max_feasible_batch(run_batch_fn, device, start_bs=1, mem_budget_gb=None,
                             max_bs_cap=100000):
    """run_batch_fn(bs) -> dict with at least "median_batch_time_s", doing
    its own warm-up + timed run entirely inside the call (this function
    resets/reads CUDA peak-memory stats around each call, so nothing about
    memory should be measured by the caller itself).

    Returns (rows, max_feasible_bs, oom_or_budget_bs): rows is one dict per
    batch size that ran successfully and stayed under mem_budget_gb (each
    with batch_size/peak_memory_mb/throughput_samples_per_s added),
    max_feasible_bs is the largest such size, and oom_or_budget_bs is the
    first size that failed (OOM) or exceeded the budget, or None if the
    search stopped at max_bs_cap without either happening.
    """
    rows = []
    bs = start_bs
    max_feasible = None
    stopped_at = None
    while bs <= max_bs_cap:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        try:
            result = dict(run_batch_fn(bs))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            stopped_at = bs
            torch.cuda.empty_cache()
            break

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result["batch_size"] = bs
        result["peak_memory_mb"] = peak_mb
        result["throughput_samples_per_s"] = bs / result["median_batch_time_s"]

        if mem_budget_gb is not None and peak_mb / 1024.0 > mem_budget_gb:
            stopped_at = bs
            break

        rows.append(result)
        max_feasible = bs
        bs *= 2

    return rows, max_feasible, stopped_at
