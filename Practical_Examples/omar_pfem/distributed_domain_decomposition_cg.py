"""Item G, first real step ("faster iterative solvers: AMG-Krylov with
multi-GPU/domain decomposition", Timon's 2026-10-11 email): the
multi-GPU/domain-decomposition half of this ask does not exist anywhere
in this project or in torch-fem (confirmed by `grep`, see
`benchmark_amg_vs_jacobi_b3.py`'s own module docstring for the AMG half's
scoping). This is a genuinely new engineering task -- not a config flag
like AMG turned out to be.

This container has NO GPU at all (`torch.cuda.is_available()` is False),
so the real multi-GPU/NCCL version was first built and proven correct
here on CPU (`backend="gloo"`, see the 2026-11-08 correctness results in
PROJECT_STATUS.md -- 5 rank/size configurations, all matching a
single-process reference to ~1e-15). `--backend` now auto-selects
"nccl" with each rank pinned to its own `cuda:{rank}` device whenever
CUDA is available (e.g. on the HPC) -- the SAME partitioning/halo-
exchange/all_reduce logic, just with every tensor moved to GPU and
point-to-point/collective ops going over NCCL instead of gloo's CPU
sockets. This real GPU path has NOT been run yet (no GPU in this
container) -- run the correctness check (small --n_side) on the HPC
FIRST, exactly as the CPU version was, before trusting any timing
number from the bigger run.

Domain decomposition specifically (not just "any parallel CG"): DOFs are
partitioned into contiguous SLABS (a 1D decomposition of a structured
3D grid, the simplest real domain decomposition -- each rank owns one
slab and only ever needs HALO data from its two immediate neighbor
slabs, never a global gather). This mirrors how a real multi-GPU FEM
solve would partition an actual mesh (there, by a graph partitioner
like METIS, not a slab -- that upgrade is a separate, later step; a slab
is the minimal case that already exercises every piece of the real
mechanism: partitioned storage, neighbor-only communication, distributed
reductions).

Test problem: a standard 3D 7-point Laplacian (finite-difference
Poisson), not this project's own B1/B3 tangent stiffness -- an honest,
well-understood SPD benchmark for testing the PARALLEL ALGORITHM itself.
The same partitioned-matvec/halo-exchange/all_reduce mechanism applies
unchanged to any SPD sparse system, including a real FEM tangent
stiffness; swapping in the real matrix is a separate, later step once
this mechanism itself is trusted.

Correctness check: the distributed CG's final solution is compared
directly against a single-process reference CG solve of the SAME full
matrix -- not merely "it ran without crashing."

Usage (CPU-only, no GPU needed, runs with 1/2/4 simulated ranks):
  python -m omar_pfem.distributed_domain_decomposition_cg --n_ranks 4 --n_side 20
"""
import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def build_3d_laplacian_coo(n_side, dtype=torch.float64):
    """Standard 7-point finite-difference Laplacian on an n_side^3 grid,
    Dirichlet boundary (boundary rows/cols removed -- interior DOFs
    only), node index = i + n_side*j + n_side*n_side*k. Returns
    (n_dof, row, col, val, b) -- a real, textbook SPD system, nothing
    project-specific.

    Vectorized (no per-node Python loop): builds all n^3 diagonal
    entries and all up-to-6*n^3 off-diagonal entries via tensor ops, so
    this stays fast enough to build at the larger n_side a real
    multi-GPU timing comparison needs (the original nested-loop version
    was fine for the small CPU correctness checks but would be
    minutes-slow just to BUILD a matrix at n_side~100+)."""
    n = n_side
    i, j, k = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij")
    i, j, k = i.reshape(-1), j.reshape(-1), k.reshape(-1)
    p = i + n * j + n * n * k  # (n^3,)

    rows = [p]
    cols = [p]
    vals = [torch.full((n ** 3,), 6.0, dtype=dtype)]
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ii, jj, kk = i + di, j + dj, k + dk
        valid = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (kk >= 0) & (kk < n)
        q = ii[valid] + n * jj[valid] + n * n * kk[valid]
        rows.append(p[valid])
        cols.append(q)
        vals.append(torch.full((int(valid.sum()),), -1.0, dtype=dtype))

    row = torch.cat(rows)
    col = torch.cat(cols)
    val = torch.cat(vals)
    b = torch.ones(n ** 3, dtype=dtype)  # constant source term -- any nonzero RHS works for a correctness check
    return n ** 3, row, col, val, b


def slab_partition(n_side, n_ranks):
    """Partition the n_side^3 grid's k-axis into n_ranks contiguous
    slabs (as equal as possible) -- the simplest real domain
    decomposition of a structured grid. Returns a list of (k_lo, k_hi)
    half-open ranges, one per rank."""
    base = n_side // n_ranks
    extra = n_side % n_ranks
    bounds, k0 = [], 0
    for r in range(n_ranks):
        width = base + (1 if r < extra else 0)
        bounds.append((k0, k0 + width))
        k0 += width
    return bounds


def local_rows_for_slab(n_side, row, col, val, k_lo, k_hi):
    """This rank's own rows (k in [k_lo,k_hi)) of the global COO matrix,
    plus which GLOBAL column indices they reference outside [k_lo,k_hi)
    -- the halo this rank needs from its neighbors every matvec."""
    plane = n_side * n_side
    row_k = row // plane
    mask = (row_k >= k_lo) & (row_k < k_hi)
    local_row, local_col, local_val = row[mask], col[mask], val[mask]
    col_k = local_col // plane
    halo_mask = (col_k < k_lo) | (col_k >= k_hi)
    halo_cols = torch.unique(local_col[halo_mask])
    return local_row, local_col, local_val, halo_cols


class DistributedCG:
    """Owns this rank's local matrix slab and runs the neighbor halo-
    exchange + all_reduce machinery a real domain-decomposed Krylov
    solve needs. One instance per rank/process."""

    def __init__(self, n_side, n_ranks, rank, dtype=torch.float64, device=None):
        self.n_side, self.n_ranks, self.rank, self.dtype = n_side, n_ranks, rank, dtype
        self.device = device or torch.device("cpu")
        self.plane = n_side * n_side
        self.n_dof = n_side ** 3
        bounds = slab_partition(n_side, n_ranks)
        self.k_lo, self.k_hi = bounds[rank]
        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < n_ranks - 1 else None

        _, row, col, val, b_full = build_3d_laplacian_coo(n_side, dtype)
        lr, lc, lv, halo_cols = local_rows_for_slab(n_side, row, col, val, self.k_lo, self.k_hi)
        # Every tensor this rank owns lives on its own GPU (or CPU) from here on --
        # NCCL send/recv/all_reduce require CUDA tensors, not just a device= kwarg
        # at creation time, so this move must happen before any collective touches them.
        self.local_row = (lr - self.k_lo * self.plane).to(self.device)  # LOCAL output row 0..n_local-1
        self.local_col = lc.to(self.device)
        self.local_val = lv.to(self.device)
        self.n_local = (self.k_hi - self.k_lo) * self.plane
        self.b_local = b_full[self.k_lo * self.plane: self.k_hi * self.plane].to(self.device)

        # Which of this rank's OWN boundary-plane DOFs the neighbors need from us,
        # and which of our halo columns belong to prev vs. next (slabs are
        # contiguous in k, so prev's halo is exactly our FIRST plane, next's is
        # exactly our LAST plane -- no search needed, a structural fact of the
        # slab decomposition, not assumed).
        self.send_to_prev = torch.arange(0, self.plane, device=self.device) if self.prev_rank is not None else None
        self.send_to_next = torch.arange(self.n_local - self.plane, self.n_local, device=self.device) if self.next_rank is not None else None

    def matvec(self, x_local):
        """x_local: this rank's own (n_local,) segment. Returns A_local @ x_full
        restricted to this rank's own output rows, using one halo exchange
        with each neighbor (point-to-point send/recv, not a global op)."""
        halo_prev = torch.zeros(self.plane, dtype=self.dtype, device=self.device) if self.prev_rank is not None else None
        halo_next = torch.zeros(self.plane, dtype=self.dtype, device=self.device) if self.next_rank is not None else None

        reqs = []
        if self.prev_rank is not None:
            reqs.append(dist.isend(x_local[self.send_to_prev].contiguous(), dst=self.prev_rank))
            reqs.append(dist.irecv(halo_prev, src=self.prev_rank))
        if self.next_rank is not None:
            reqs.append(dist.isend(x_local[self.send_to_next].contiguous(), dst=self.next_rank))
            reqs.append(dist.irecv(halo_next, src=self.next_rank))
        for r in reqs:
            r.wait()

        out = torch.zeros(self.n_local, dtype=self.dtype, device=self.device)
        local_mask = (self.local_col >= self.k_lo * self.plane) & (self.local_col < self.k_hi * self.plane)
        out.index_add_(0, self.local_row[local_mask], self.local_val[local_mask] * x_local[self.local_col[local_mask] - self.k_lo * self.plane])
        if (~local_mask).any():
            # Vectorized halo lookup (no per-element Python dict -- that was fine
            # for the tiny CPU correctness check but would be slow at any real
            # scale, and doubly so pulling GPU scalars into Python one at a time).
            # Every halo column belongs to EXACTLY ONE neighbor plane (prev's is
            # k=k_lo-1, next's is k=k_hi, a structural fact of the slab
            # decomposition), so a vectorized torch.where over the two neighbor
            # buffers, indexed by each column's offset within its plane, replaces
            # the dict entirely.
            halo_row = self.local_row[~local_mask]
            halo_col = self.local_col[~local_mask]
            halo_val = self.local_val[~local_mask]
            halo_k = halo_col // self.plane
            offset = halo_col % self.plane
            is_prev = halo_k == (self.k_lo - 1)
            halo_x = torch.where(
                is_prev,
                halo_prev[offset] if halo_prev is not None else torch.zeros_like(offset, dtype=self.dtype),
                halo_next[offset] if halo_next is not None else torch.zeros_like(offset, dtype=self.dtype))
            out.index_add_(0, halo_row, halo_val * halo_x)
        return out

    def global_dot(self, a_local, b_local):
        """Global dot product of two distributed vectors: local partial
        sum + all_reduce(SUM) -- every distributed Krylov method's other
        essential collective, besides the halo-exchanged matvec."""
        local = (a_local * b_local).sum()
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    def solve(self, max_iter=500, rtol=1e-10):
        x = torch.zeros(self.n_local, dtype=self.dtype, device=self.device)
        r = self.b_local - self.matvec(x)
        p = r.clone()
        rs_old = self.global_dot(r, r)
        b_norm = torch.sqrt(self.global_dot(self.b_local, self.b_local)) + 1e-300
        for it in range(max_iter):
            Ap = self.matvec(p)
            alpha = rs_old / (self.global_dot(p, Ap) + 1e-300)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = self.global_dot(r, r)
            if torch.sqrt(rs_new) / b_norm < rtol:
                return x, it + 1
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x, max_iter


def _worker(rank, n_ranks, n_side, backend, result_queue):
    if backend == "nccl":
        torch.cuda.set_device(rank)  # must happen BEFORE init_process_group for NCCL
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=n_ranks)
    solver = DistributedCG(n_side, n_ranks, rank, device=device)
    if backend == "nccl":
        torch.cuda.synchronize(device)
    t0 = time.time()
    x_local, n_iter = solver.solve()
    if backend == "nccl":
        torch.cuda.synchronize(device)  # NCCL ops are async -- don't time a kernel launch queue
    elapsed = time.time() - t0
    # Gather every rank's local solution segment to rank 0 for the correctness
    # check. NOT a plain dist.gather: slab_partition gives ranks DIFFERENT
    # sizes whenever n_side doesn't divide evenly by n_ranks (a real case,
    # caught by this script's own correctness check failing loudly the first
    # time it was tried at n_side=17/n_ranks=3 rather than silently
    # misassembling a wrong-but-plausible-looking answer) -- torch.distributed's
    # gather collective assumes uniform tensor size across ranks, so point-to-
    # point send/recv into correctly، independently-computed-size buffers is
    # used instead (rank 0 can compute every rank's own size itself, via the
    # same deterministic slab_partition, without being told).
    if rank == 0:
        bounds = slab_partition(n_side, n_ranks)
        plane = n_side * n_side
        segments = [None] * n_ranks
        segments[0] = x_local
        for src in range(1, n_ranks):
            k_lo, k_hi = bounds[src]
            buf = torch.zeros((k_hi - k_lo) * plane, dtype=x_local.dtype, device=device)
            dist.recv(buf, src=src)
            segments[src] = buf
        # Moved to CPU before crossing the process boundary -- torch.multiprocessing's
        # default Queue does not reliably hand off live CUDA tensors between
        # processes (needs an explicit CUDA sharing strategy), and the result is
        # only used for a correctness/timing printout on the main process anyway.
        x_full = torch.cat(segments).cpu()
        result_queue.put((x_full, n_iter, elapsed))
    else:
        dist.send(x_local.contiguous(), dst=0)
    dist.destroy_process_group()


def reference_cg(n_side, max_iter=500, rtol=1e-10, dtype=torch.float64, device=None):
    """Single-process reference solve of the SAME full system, for the
    correctness check -- plain CG, no partitioning, no distributed
    machinery at all. device=cuda:0 gives the fair "1 GPU, no domain
    decomposition" timing baseline the multi-GPU run should be compared
    against (not the CPU gloo numbers from the correctness-only pass)."""
    device = device or torch.device("cpu")
    n_dof, row, col, val, b = build_3d_laplacian_coo(n_side, dtype)
    row, col, val, b = row.to(device), col.to(device), val.to(device), b.to(device)
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n_dof, n_dof)).coalesce()
    x = torch.zeros(n_dof, dtype=dtype, device=device)
    r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    p = r.clone()
    rs_old = r @ r
    b_norm = torch.sqrt(b @ b) + 1e-300
    for it in range(max_iter):
        Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
        alpha = rs_old / (p @ Ap + 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if torch.sqrt(rs_new) / b_norm < rtol:
            return x.cpu(), it + 1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x.cpu(), max_iter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--n_ranks", type=int, default=4)
    parser.add_argument("--n_side", type=int, default=20)
    parser.add_argument("--backend", type=str, default=None, choices=["nccl", "gloo"],
                         help="default: nccl if CUDA is available (real multi-GPU run), "
                              "else gloo (CPU correctness check).")
    args = parser.parse_args()
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        n_gpus = torch.cuda.device_count()
        if n_gpus < args.n_ranks:
            raise RuntimeError(f"--n_ranks {args.n_ranks} but only {n_gpus} GPU(s) visible "
                                f"(check CUDA_VISIBLE_DEVICES / the bsub -gpu num= request)")
    print(f"backend={backend}  n_ranks={args.n_ranks}  n_side={args.n_side}  "
          f"n_dof={args.n_side ** 3:,}")

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    # Reduces (does not fully eliminate -- still seen ~1/5 runs even with this
    # set) a real FileNotFoundError/ConnectionResetError race in mp.Queue's
    # default file-descriptor-passing strategy, specific to this sandboxed
    # container and ONLY with a very-short-lived single process (--n_ranks 1).
    # Confirmed by direct repetition (not assumed): n_ranks 2/3/4/5 passed
    # every single time across many repeated runs; n_ranks=1 alone is flaky.
    # Irrelevant to the real multi-GPU goal (n_ranks=1 means no domain
    # decomposition at all -- not a case this item needs to run), so not
    # chased further; a real GPU/HPC run should just avoid --n_ranks 1.
    mp.set_sharing_strategy("file_system")
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")

    result_queue = mp.Queue()
    procs = []
    for rank in range(args.n_ranks):
        p = mp.Process(target=_worker, args=(rank, args.n_ranks, args.n_side, backend, result_queue))
        p.start()
        procs.append(p)
    x_distributed, n_iter_dist, elapsed_dist = result_queue.get()
    for p in procs:
        p.join()

    print(f"[distributed, {args.n_ranks} ranks, {backend}] converged in {n_iter_dist} CG iterations, "
          f"{elapsed_dist:.3f}s")

    ref_device = torch.device("cuda:0") if backend == "nccl" else torch.device("cpu")
    t0 = time.time()
    x_ref, n_iter_ref = reference_cg(args.n_side, device=ref_device)
    if backend == "nccl":
        torch.cuda.synchronize(ref_device)
    elapsed_ref = time.time() - t0
    print(f"[reference, 1 {'GPU' if backend == 'nccl' else 'process'}, no partitioning] "
          f"converged in {n_iter_ref} CG iterations, {elapsed_ref:.3f}s")

    diff = torch.linalg.norm(x_distributed - x_ref) / (torch.linalg.norm(x_ref) + 1e-300)
    print(f"\nrelative difference between distributed and reference solutions: {diff.item():.3e}")
    print("PASS -- domain-decomposed distributed CG matches the single-process/single-GPU reference"
          if diff.item() < 1e-8 else
          "FAIL -- distributed result does not match the reference, do not trust this mechanism yet "
          "(do NOT trust the timing numbers above either if this fails)")
    if backend == "nccl" and diff.item() < 1e-8:
        speedup = elapsed_ref / elapsed_dist if elapsed_dist > 0 else float("nan")
        print(f"\n{args.n_ranks}-GPU wall-clock vs. 1-GPU reference: "
              f"{elapsed_dist:.3f}s vs. {elapsed_ref:.3f}s -> {speedup:.2f}x "
              f"({'speedup' if speedup > 1 else 'SLOWER -- communication overhead dominates at this size'})")


if __name__ == "__main__":
    main()
