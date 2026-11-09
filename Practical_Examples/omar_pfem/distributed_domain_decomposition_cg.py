"""Item G, first real step ("faster iterative solvers: AMG-Krylov with
multi-GPU/domain decomposition", Timon's 2026-10-11 email): the
multi-GPU/domain-decomposition half of this ask does not exist anywhere
in this project or in torch-fem (confirmed by `grep`, see
`benchmark_amg_vs_jacobi_b3.py`'s own module docstring for the AMG half's
scoping). This is a genuinely new engineering task -- not a config flag
like AMG turned out to be.

This container has NO GPU at all (`torch.cuda.is_available()` is False),
so the real multi-GPU/NCCL version cannot be built or tested here. What
CAN be done, matching this project's own standing discipline of proving
a mechanism correct on a small/CPU case before spending any real GPU
time on it: implement genuine distributed domain decomposition --
partitioned sparse-matrix storage, neighbor-only halo exchange for the
matrix-vector product, and `all_reduce` for the global dot products a
Krylov method needs -- using `torch.distributed`'s "gloo" backend, which
runs over ordinary CPU processes. The SAME code, with `backend="nccl"`
and tensors moved to `cuda:{rank}`, is the real multi-GPU version; gloo
only stands in for CPU-only correctness testing here.

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
    project-specific."""
    n = n_side
    idx = lambda i, j, k: i + n * j + n * n * k
    rows, cols, vals = [], [], []
    b = torch.zeros(n ** 3, dtype=dtype)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                p = idx(i, j, k)
                rows.append(p); cols.append(p); vals.append(6.0)
                for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
                        q = idx(ii, jj, kk)
                        rows.append(p); cols.append(q); vals.append(-1.0)
                b[p] = 1.0  # constant source term -- any nonzero RHS works for a correctness check
    row = torch.tensor(rows, dtype=torch.long)
    col = torch.tensor(cols, dtype=torch.long)
    val = torch.tensor(vals, dtype=dtype)
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

    def __init__(self, n_side, n_ranks, rank, dtype=torch.float64):
        self.n_side, self.n_ranks, self.rank, self.dtype = n_side, n_ranks, rank, dtype
        self.plane = n_side * n_side
        self.n_dof = n_side ** 3
        bounds = slab_partition(n_side, n_ranks)
        self.k_lo, self.k_hi = bounds[rank]
        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < n_ranks - 1 else None

        _, row, col, val, b_full = build_3d_laplacian_coo(n_side, dtype)
        lr, lc, lv, halo_cols = local_rows_for_slab(n_side, row, col, val, self.k_lo, self.k_hi)
        self.local_row = lr - self.k_lo * self.plane  # re-index to a LOCAL output row 0..n_local-1
        self.local_col = lc
        self.local_val = lv
        self.n_local = (self.k_hi - self.k_lo) * self.plane
        self.b_local = b_full[self.k_lo * self.plane: self.k_hi * self.plane]

        # Which of this rank's OWN boundary-plane DOFs the neighbors need from us,
        # and which of our halo columns belong to prev vs. next (slabs are
        # contiguous in k, so prev's halo is exactly our FIRST plane, next's is
        # exactly our LAST plane -- no search needed, a structural fact of the
        # slab decomposition, not assumed).
        self.send_to_prev = torch.arange(0, self.plane) if self.prev_rank is not None else None
        self.send_to_next = torch.arange(self.n_local - self.plane, self.n_local) if self.next_rank is not None else None

    def matvec(self, x_local):
        """x_local: this rank's own (n_local,) segment. Returns A_local @ x_full
        restricted to this rank's own output rows, using one halo exchange
        with each neighbor (point-to-point send/recv, not a global op)."""
        halo_prev = torch.zeros(self.plane, dtype=self.dtype) if self.prev_rank is not None else None
        halo_next = torch.zeros(self.plane, dtype=self.dtype) if self.next_rank is not None else None

        reqs = []
        if self.prev_rank is not None:
            reqs.append(dist.isend(x_local[self.send_to_prev].contiguous(), dst=self.prev_rank))
            reqs.append(dist.irecv(halo_prev, src=self.prev_rank))
        if self.next_rank is not None:
            reqs.append(dist.isend(x_local[self.send_to_next].contiguous(), dst=self.next_rank))
            reqs.append(dist.irecv(halo_next, src=self.next_rank))
        for r in reqs:
            r.wait()

        x_with_halo_global = {}  # global column index -> value, for the (few) halo entries this rank needs
        if halo_prev is not None:
            for i in range(self.plane):
                x_with_halo_global[(self.k_lo - 1) * self.plane + i] = halo_prev[i]
        if halo_next is not None:
            for i in range(self.plane):
                x_with_halo_global[self.k_hi * self.plane + i] = halo_next[i]

        out = torch.zeros(self.n_local, dtype=self.dtype)
        local_mask = (self.local_col >= self.k_lo * self.plane) & (self.local_col < self.k_hi * self.plane)
        out.index_add_(0, self.local_row[local_mask], self.local_val[local_mask] * x_local[self.local_col[local_mask] - self.k_lo * self.plane])
        if (~local_mask).any():
            halo_row = self.local_row[~local_mask]
            halo_col = self.local_col[~local_mask]
            halo_val = self.local_val[~local_mask]
            halo_x = torch.tensor([x_with_halo_global[int(c)] for c in halo_col], dtype=self.dtype)
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
        x = torch.zeros(self.n_local, dtype=self.dtype)
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


def _worker(rank, n_ranks, n_side, result_queue):
    dist.init_process_group("gloo", rank=rank, world_size=n_ranks)
    solver = DistributedCG(n_side, n_ranks, rank)
    t0 = time.time()
    x_local, n_iter = solver.solve()
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
            buf = torch.zeros((k_hi - k_lo) * plane, dtype=x_local.dtype)
            dist.recv(buf, src=src)
            segments[src] = buf
        x_full = torch.cat(segments)
        result_queue.put((x_full, n_iter, elapsed))
    else:
        dist.send(x_local.contiguous(), dst=0)
    dist.destroy_process_group()


def reference_cg(n_side, max_iter=500, rtol=1e-10, dtype=torch.float64):
    """Single-process reference solve of the SAME full system, for the
    correctness check -- plain CG, no partitioning, no distributed
    machinery at all."""
    n_dof, row, col, val, b = build_3d_laplacian_coo(n_side, dtype)
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n_dof, n_dof)).coalesce()
    x = torch.zeros(n_dof, dtype=dtype)
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
            return x, it + 1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--n_ranks", type=int, default=4)
    parser.add_argument("--n_side", type=int, default=20)
    args = parser.parse_args()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")

    result_queue = mp.Queue()
    procs = []
    for rank in range(args.n_ranks):
        p = mp.Process(target=_worker, args=(rank, args.n_ranks, args.n_side, result_queue))
        p.start()
        procs.append(p)
    x_distributed, n_iter_dist, elapsed_dist = result_queue.get()
    for p in procs:
        p.join()

    print(f"[distributed, {args.n_ranks} ranks] converged in {n_iter_dist} CG iterations, {elapsed_dist:.3f}s")

    t0 = time.time()
    x_ref, n_iter_ref = reference_cg(args.n_side)
    elapsed_ref = time.time() - t0
    print(f"[reference, 1 process, no partitioning] converged in {n_iter_ref} CG iterations, {elapsed_ref:.3f}s")

    diff = torch.linalg.norm(x_distributed - x_ref) / (torch.linalg.norm(x_ref) + 1e-300)
    print(f"\nrelative difference between distributed and reference solutions: {diff.item():.3e}")
    print("PASS -- domain-decomposed distributed CG matches the single-process reference"
          if diff.item() < 1e-8 else
          "FAIL -- distributed result does not match the reference, do not trust this mechanism yet")


if __name__ == "__main__":
    main()
