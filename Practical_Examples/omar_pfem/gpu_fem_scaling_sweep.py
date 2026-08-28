"""GPU-native FEM performance up to a few million DOF -- the advisor's point 8.

Point 8 asks how the GPU-native solver performs at finer discretizations, up
to a few million degrees of freedom. That is a performance question, not an
accuracy one, so this is a timing sweep and not a convergence study:
`high_dof_convergence_study.py` answers the accuracy question but has to
solve an even finer reference to do it, which at these sizes costs more than
the sweep itself and measures nothing point 8 asked about.

Which solver: the matrix-free Newton-CG one. `gpu_fem_solver.py` forms the
tangent densely and calls `torch.linalg.solve`, which cannot reach these
sizes -- a dense 2M x 2M tangent in float64 is about 32 TB. `matrix_free_solver.py`
never forms K at all, only Hessian-vector products, so its memory is O(DOF);
it is what produced the 10M and 40M DOF references already in the report.

Default resolutions target roughly 0.5M, 1M, 2M and 4M DOF (a Q4 mesh at
resolution N has 2*N^2 DOF): N = 501, 701, 1001, 1401 give 0.50M, 0.98M,
2.00M and 3.93M.

Every resolution's row is written to the output JSON as soon as it finishes,
and each solve can checkpoint internally, because a single solve at the top
of this range takes hours and a Colab runtime does not reliably last that
long.

Usage:
  python -m omar_pfem.gpu_fem_scaling_sweep \
      --geometry B1 --material neo_hookean \
      --resolutions 501,701,1001,1401 \
      --out_json .../gpu_fem_scaling_B1_neo_hookean.json
"""
import os
import json
import time
import argparse

import torch

from omar_pfem.run_manifest import write_manifest
from omar_pfem.matrix_free_solver import solve_matrix_free
from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs


def main():
    p = argparse.ArgumentParser("GPU-native FEM scaling sweep (advisor point 8)")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", default="neo_hookean",
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--order", default="Q4", choices=["Q4", "Q9"])
    p.add_argument("--resolutions", default="101,201,301,401,501,701,1001,1401",
                   help="Q4: DOF = 2*N^2. The advisor's round-6 request added "
                        "'smaller intermediate numbers' below the original "
                        "0.5M-4M range, so this now starts at 0.02M and runs "
                        "0.02, 0.08, 0.18, 0.32, 0.50, 0.98, 2.00, 3.93M DOF. "
                        "The four small ones cost minutes between them and are "
                        "what make the us/DOF trend a curve rather than four "
                        "points at one end of it.")
    p.add_argument("--nsteps", type=int, default=10)
    p.add_argument("--newton_max", type=int, default=30)
    p.add_argument("--newton_tol", type=float, default=1e-7)
    p.add_argument("--cg_tol", type=float, default=1e-6)
    p.add_argument("--cg_max_iter", type=int, default=2000)
    p.add_argument("--no_jacobi", action="store_true")
    p.add_argument("--out_json", default=None)
    p.add_argument("--checkpoint_dir", default=None,
                   help="per-resolution solver checkpoints, so a disconnect "
                        "costs at most one in-progress CG solve")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    resolutions = [int(n) for n in args.resolutions.split(",") if n.strip()]

    if args.out_json is None:
        args.out_json = (f"gpu_fem_scaling_{args.geometry}_{args.material}"
                         f"_{args.order}.json")
    out_dir = os.path.dirname(os.path.abspath(args.out_json)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Solver: matrix-free Newton-CG (never forms K)\n")

    # resume: keep resolutions already recorded, so re-running continues
    rows = []
    if os.path.exists(args.out_json):
        try:
            rows = json.load(open(args.out_json)).get("rows", [])
            if rows:
                print(f"[resume] already have N={[r['N'] for r in rows]}\n")
        except Exception:
            rows = []
    done = {r["N"] for r in rows}

    for N in resolutions:
        if N in done:
            continue
        print(f"===== N={N} =====")
        t_build = time.time()
        nodes, elements, free_dofs, fext_full, elem_params_np = build_mesh_and_bcs(
            args.geometry, args.order, N, args.material, device, dtype)
        build_s = time.time() - t_build

        xy_t = torch.tensor(nodes, dtype=dtype, device=device)
        quad_t = torch.tensor(elements, dtype=torch.long, device=device)
        free_t = torch.tensor(free_dofs, dtype=torch.long, device=device)
        params_t = tuple(torch.tensor(pp, dtype=dtype, device=device)
                         for pp in elem_params_np)
        fext_t = torch.tensor(fext_full[free_dofs], dtype=dtype, device=device)

        ndof = 2 * len(nodes)
        print(f"  {len(nodes):,} nodes, {ndof:,} DOF, {len(elements):,} elements "
              f"(mesh built in {build_s:.1f}s)")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        ckpt = (os.path.join(args.checkpoint_dir, f"solve_N{N}.pt")
                if args.checkpoint_dir else None)
        t0 = time.time()
        _, stats = solve_matrix_free(
            xy_t, quad_t, free_t, params_t, fext_t, n_free=len(free_dofs),
            material=args.material, order=args.order, nsteps=args.nsteps,
            newton_max=args.newton_max, newton_tol=args.newton_tol,
            cg_tol=args.cg_tol, cg_max_iter=args.cg_max_iter,
            use_jacobi=not args.no_jacobi, device=device, dtype=dtype,
            verbose=False, checkpoint_path=ckpt)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        solve_s = time.time() - t0

        row = {"N": N, "n_nodes": int(len(nodes)), "n_dof": int(ndof),
               "n_elements": int(len(elements)), "order": args.order,
               "mesh_build_s": build_s, "solve_s": solve_s,
               "us_per_dof": 1e6 * solve_s / ndof,
               "stats": {k: v for k, v in stats.items()
                         if isinstance(v, (int, float, str, bool))}}
        if device.type == "cuda":
            row["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated(device) / 1e6
            row["peak_gpu_reserved_MB"] = torch.cuda.max_memory_reserved(device) / 1e6

        rows.append(row)
        rows.sort(key=lambda r: r["N"])
        # cost breakdown, the advisor's round-6 request -- see the note in
        # matrix_free_solver.py on why CG time is not "solver time" cleanly
        st = row["stats"]
        if all(k in st for k in ("t_residual_s", "t_precond_s", "t_cg_s")):
            acc = st["t_residual_s"] + st["t_precond_s"] + st["t_cg_s"]
            row["cost_breakdown_pct"] = {
                k.replace("t_", "").replace("_s", ""): 100.0 * st[k] / acc
                for k in ("t_residual_s", "t_precond_s", "t_cg_s")} if acc > 0 else {}
            row["accounted_frac_of_solve"] = acc / solve_s if solve_s > 0 else 0.0

        print(f"  solved in {solve_s / 60:.1f} min  "
              f"({row['us_per_dof']:.2f} us/DOF)"
              + (f"  peak {row['peak_gpu_mem_MB']:.0f} MB"
                 if "peak_gpu_mem_MB" in row else ""))
        if row.get("cost_breakdown_pct"):
            b = row["cost_breakdown_pct"]
            print(f"    cost: residual {b['residual']:.1f}%  "
                  f"preconditioner {b['precond']:.1f}%  CG {b['cg']:.1f}%  "
                  f"({100 * row['accounted_frac_of_solve']:.1f}% of wall clock accounted)")

        report = {"geometry": args.geometry, "material": args.material,
                  "order": args.order, "device": device.type,
                  "gpu": (torch.cuda.get_device_name(0)
                          if device.type == "cuda" else None),
                  "solver": "matrix-free Newton-CG", "rows": rows}
        tmp = args.out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, args.out_json)

        del xy_t, quad_t, free_t, params_t, fext_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 66)
    print(f"{'N':>6}{'DOF':>12}{'solve (min)':>14}{'us/DOF':>10}{'peak MB':>10}")
    for r in rows:
        print(f"{r['N']:>6}{r['n_dof']:>12,}{r['solve_s'] / 60:>14.1f}"
              f"{r['us_per_dof']:>10.2f}"
              f"{r.get('peak_gpu_mem_MB', float('nan')):>10.0f}")
    print("=" * 66)
    print(f"Written to {args.out_json}")

    write_manifest(
        out_dir, kind="gpu_fem_scaling_sweep", args=args, started_at=started,
        results=report, outputs=[args.out_json],
        notes=("Advisor point 8: GPU-native FEM performance up to a few million "
               "DOF. Uses the matrix-free Newton-CG solver, not gpu_fem_solver.py, "
               "whose dense torch.linalg.solve cannot reach these sizes (a dense "
               "2M x 2M float64 tangent is ~32 TB). us/DOF is the figure that "
               "shows whether cost grows linearly in problem size or worse. "
               "Timings are hardware-specific -- the GPU is recorded in the "
               "manifest's environment block."))


if __name__ == "__main__":
    main()
