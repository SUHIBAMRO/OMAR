"""Pure wall-clock/CG-iteration scaling check for the matrix-free Newton-CG
solver -- NOT a convergence study (no reference solve, no L2/H1/energy
error). Answers a narrower, more urgent question first: does this solver's
cost scale well enough with N to make a ~10M-DOF run (Point 1 of Timon's
feedback) feasible on a single Colab GPU at all, before we spend GPU-hours
on a convergence sweep at a scale we haven't confirmed is reachable.

Two small real runs so far (B1, Q4, neo_hookean) showed wall-clock scaling
close to LINEAR in N (not the cubic worst case a naive DOF-count argument
would suggest), because cost-per-DOF drops as N grows (small N is
GPU-latency-bound, not compute-bound). This script adds one or more larger,
still-affordable data points to check whether that trend holds or reverses
once the GPU starts saturating -- the actual answer determines whether ~10M
DOF is a 3-hour job or a multi-day one.

Usage:
  python -m omar_pfem.scaling_check --geometry B1 --material neo_hookean --Ns 51,101 --orders Q4
"""
import argparse
import numpy as np
import torch

from omar_pfem.high_dof_convergence_study import solve_one


def main():
    parser = argparse.ArgumentParser("Wall-clock/CG-iteration scaling check (no error computation)")
    parser.add_argument("--geometry", type=str, default="B1", choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--Ns", type=str, default="51,101")
    parser.add_argument("--orders", type=str, default="Q4")
    parser.add_argument("--cg_tol", type=float, default=1e-8)
    parser.add_argument("--newton_tol", type=float, default=1e-8)
    parser.add_argument("--cg_max_iter", type=int, default=4000)
    parser.add_argument("--no_jacobi", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    Ns = [int(n) for n in args.Ns.split(",") if n.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    for order in orders:
        print(f"\n{'='*90}\nORDER = {order}\n{'='*90}")
        rows = []
        for N in Ns:
            print(f"\nSolving N={N} ({order})...")
            r = solve_one(args.geometry, order, N, args.material, device, dtype,
                          args.cg_tol, args.newton_tol, use_jacobi=not args.no_jacobi,
                          cg_max_iter=args.cg_max_iter, verbose=False)
            ndof = r["n_dof"]
            wall = r["wall_clock_s"]
            newton_it = r["stats"]["newton_iters_total"]
            cg_it = r["stats"]["cg_iters_total"]
            cg_fail = r["stats"]["cg_failures"]
            ms_per_dof = wall / ndof * 1000.0
            cg_per_newton = cg_it / max(newton_it, 1)
            rows.append((N, ndof, wall, newton_it, cg_it, cg_fail))
            print(f"  N={N}: n_dof={ndof}, wall={wall:.1f}s, newton_iters={newton_it}, "
                  f"cg_iters={cg_it} ({cg_per_newton:.1f}/newton-iter), cg_failures={cg_fail}, "
                  f"wall/dof={ms_per_dof:.4f} ms")
            if cg_fail > 0:
                print(f"  WARNING: {cg_fail} CG failure(s) -- results below this N are not reliable.")

        if len(rows) >= 2:
            print(f"\n{'-'*90}\nPairwise empirical scaling exponents (wall ~ N^p, cg_iters ~ N^q):")
            for i in range(1, len(rows)):
                N1, ndof1, w1, nw1, cg1, _ = rows[i - 1]
                N2, ndof2, w2, nw2, cg2, _ = rows[i]
                p = np.log(w2 / w1) / np.log(N2 / N1)
                q = np.log((cg2 / max(nw2, 1)) / (cg1 / max(nw1, 1))) / np.log(N2 / N1)
                print(f"  N={N1}->N={N2}: wall exponent p={p:.2f}, cg-per-newton exponent q={q:.2f}")

            N_last, ndof_last, w_last, *_ = rows[-1]
            N1, ndof1, w1, *_ = rows[0]
            p_overall = np.log(w_last / w1) / np.log(N_last / N1)
            N_target = int(np.sqrt(5_000_000))  # ~10M total DOF, 2 dof/node in 2D
            projected_s = w_last * (N_target / N_last) ** p_overall
            print(f"\n  Overall exponent across all points: p={p_overall:.2f}")
            print(f"  Naive same-trend projection to N={N_target} (~10M DOF): "
                  f"{projected_s:.0f}s = {projected_s/3600:.1f}h = {projected_s/3600/24:.1f}d")
            print(f"  CAUTION: this assumes today's exponent holds all the way to N={N_target}, "
                  f"a large extrapolation. Treat it as a rough order-of-magnitude signal, not a promise -- "
                  f"the trend can (and likely will) get worse once the GPU saturates.")


if __name__ == "__main__":
    main()
