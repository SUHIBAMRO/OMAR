"""Accuracy/cost Pareto: the trained operator against GPU-native FEM.

The advisor's point 2. The difficulty is not the plot, it is making the two
axes mean the same thing on both sides.

The obvious construction is wrong. The report already contains an FEM
accuracy-vs-cost curve (Table 6a: seven resolutions, error against a ~10M-DOF
reference, with wall-clock) and an operator accuracy figure (Table 5), and
putting them on one pair of axes would produce a plausible-looking and
meaningless picture: Table 6a measures a fixed analytic field against a
10M-DOF reference, while the operator's error is measured on random GRF
fields against a same-mesh FEM solution. Different problems, different
references, different hardware.

So this script measures both sides itself, on one footing:

  * the SAME problem instances -- the parametric (E, nu, load) fields of
    data/parametric_field.py, drawn from the same seeds for both sides;
  * against the SAME reference -- one fine FEM solve at --fine_N per sample,
    which both the coarse FEM solves and the operator's predictions are
    interpolated onto and scored against, exactly as the zero-shot study
    already does;
  * on the SAME device, with the operator timed at whatever batch size the
    FEM solver is timed at.

Each coarse resolution N contributes one FEM point (its error against the
fine reference, and what that solve costs). The operator contributes one
point per resolution too, since it is resolution-invariant and can be
evaluated at any of them from a single checkpoint -- which is the whole
argument being made, so the plot should show it rather than assume it.

The fine references are cached in the same `fine_ref_cache_N{fine_N}.pt` the
zero-shot eval writes, so pointing --out_dir at a finished zero-shot case
directory costs nothing for them.

Usage:
  python -m omar_pfem.pareto_analysis \
      --geometry B1 --material neo_hookean \
      --checkpoint .../zeroshot_B1_neo_hookean/model_best.pt \
      --resolutions 13,17,21,25,29,33,37,41,49 --fine_N 101 \
      --n_samples 20 --out_dir .../zeroshot_B1_neo_hookean
"""
import os
import json
import hashlib
import time
import argparse

import numpy as np
import torch

from omar_pfem.run_manifest import write_manifest
from omar_pfem.resolution_invariance_zeroshot import (
    build_sample_b1, build_sample_b2, build_model, mesh_tensors_of,
    interpolate_to_reference)


def rel_l2(pred_on_fine, ref):
    """Combined relative L2 over the fine mesh's nodes: both displacement
    components in a single vector norm.

    This is deliberately NOT the metric of Tables 5, 11 and 12. Those report
    the per-component average 0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v)), which
    is the network-side convention used throughout the report; this is the
    convergence-study convention of Section 4.4, and it is the right one here
    because the FEM side of this plot is a convergence curve and both sides
    must be scored identically for a Pareto comparison to mean anything.

    Consequence: the operator errors this script produces cannot be laid
    alongside Table 12's. On B1 the loaded component v dominates u, so the
    combined norm is the smaller of the two measures. (An earlier version of
    this docstring claimed the two were comparable. They are not, and the
    seed base below is a second, independent reason.)
    """
    num = np.linalg.norm(pred_on_fine - ref)
    den = np.linalg.norm(ref)
    return float(num / den)


def main():
    p = argparse.ArgumentParser("Accuracy/cost Pareto: operator vs GPU FEM")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", default="neo_hookean",
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resolutions", default="13,17,21,25,29,33,37,41,49")
    p.add_argument("--fine_N", type=int, default=101)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1,
                   help="batch size BOTH sides are timed at. 1 is the "
                        "deployment case; pass 128 for the throughput case.")
    p.add_argument("--n_timing_repeats", type=int, default=20)
    p.add_argument("--out_dir", required=True,
                   help="where results and the fine-reference cache live; "
                        "point at a finished zero-shot case to reuse its cache")
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    # model hyperparameters, matching the zero-shot study's own defaults
    p.add_argument("--model", default="Transolver_Irregular_Mesh")
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
    p.add_argument("--mode", default="plane_strain")
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    args = p.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_json is None:
        args.out_json = os.path.join(
            args.out_dir, f"pareto_{args.geometry}_{args.material}.json")

    resolutions = [int(n) for n in args.resolutions.split(",") if n.strip()]
    build_fn = build_sample_b1 if args.geometry == "B1" else build_sample_b2

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}, timing both sides at batch size {args.batch_size}\n")

    # ---- the shared fine reference, cached exactly as the zero-shot eval does
    cache_path = os.path.join(args.out_dir, f"fine_ref_cache_N{args.fine_N}.pt")
    fine_cache = {}
    if os.path.exists(cache_path):
        fine_cache = torch.load(cache_path, weights_only=False)
        print(f"[fine-ref cache] {len(fine_cache)} solves already available")

    def fine_sample(seed):
        if seed not in fine_cache:
            s, _ = build_fn(args.fine_N, seed=seed, material=args.material,
                            solve_fem=True)
            fine_cache[seed] = {"xy": s["xy"], "uv_exact": s["uv_exact"]}
            torch.save(fine_cache, cache_path)
        return fine_cache[seed]

    # NOTE: a different seed base from the zero-shot eval, which uses
    # 20_000_000 + i. These are therefore different physical problems, not the
    # same ones scored differently -- a second reason the numbers here cannot
    # be read against Table 12's. Changing this to 20_000_000 would also make
    # the fine-reference cache hit instead of solving 20 fresh N=101 problems.
    seeds = [900_000 + i for i in range(args.n_samples)]

    # ---- RESUME. The JSON is rewritten after every resolution, so a run
    # that dies at N=37 leaves N=13..33 on disk -- but the loop below used to
    # start from an empty `rows` and redo all of them, overwriting what was
    # there. On this sweep that is expensive: the B1 x Mooney-Rivlin run took
    # 14 h 31 m and N=49 alone is 1.8 h of CPU, so a Colab disconnect at hour
    # ten cost everything. Completed resolutions are now read back and
    # skipped.
    #
    # Guarded by the checkpoint fingerprint: rows produced by a DIFFERENT
    # model must never be merged with new ones, which would silently mix two
    # models in one table.
    fingerprint = hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest()
    rows, done = [], set()
    if os.path.exists(args.out_json):
        try:
            prev = json.load(open(args.out_json))
        except Exception as e:
            print(f"[resume] {args.out_json} is unreadable ({e.__class__.__name__});"
                  f" starting fresh")
            prev = None
        if prev is not None:
            same = (prev.get("checkpoint_fingerprint") == fingerprint
                    and prev.get("n_samples") == args.n_samples
                    and prev.get("fine_N") == args.fine_N
                    and prev.get("material") == args.material
                    and prev.get("geometry") == args.geometry)
            if same:
                rows = [r for r in prev.get("rows", []) if r["N"] in resolutions]
                done = {r["N"] for r in rows}
                if done:
                    print(f"[resume] {args.out_json} already holds "
                          f"N={sorted(done)} from the same checkpoint and the "
                          f"same protocol -- skipping those")
            elif prev.get("checkpoint_fingerprint") is None:
                print(f"[resume] {args.out_json} predates fingerprinting, so "
                      f"its rows cannot be shown to belong to this checkpoint;"
                      f" starting fresh")
            else:
                print(f"[resume] {args.out_json} was produced by a DIFFERENT "
                      f"checkpoint or protocol; starting fresh")

    for N in resolutions:
        if N in done:
            continue
        fem_errs, op_errs, fem_times = [], [], []
        coarse0 = None
        for i, seed in enumerate(seeds):
            fine = fine_sample(seed)

            # --- FEM at this resolution: its own solve, timed
            t0 = time.perf_counter()
            coarse, _ = build_fn(N, seed=seed, material=args.material, solve_fem=True)
            fem_times.append(time.perf_counter() - t0)
            if coarse0 is None:
                coarse0 = coarse

            fem_on_fine = interpolate_to_reference(
                coarse["xy"], coarse["uv_exact"], fine["xy"])
            fem_errs.append(rel_l2(fem_on_fine, fine["uv_exact"]))

            # --- the operator on the same mesh, same sample, no retraining
            mesh_t = mesh_tensors_of(args.geometry, coarse, device, dtype)
            E = torch.tensor(coarse["E_node"][None], device=device, dtype=dtype)
            nu = torch.tensor(coarse["nu_node"][None], device=device, dtype=dtype)
            f = torch.tensor(coarse["node_forces"][None], device=device, dtype=dtype)
            with torch.no_grad():
                uv = predict(args, mesh_t, model, E, nu, f, dtype)
            op_on_fine = interpolate_to_reference(
                coarse["xy"], uv[0].cpu().numpy(), fine["xy"])
            op_errs.append(rel_l2(op_on_fine, fine["uv_exact"]))

        # --- operator inference cost at this resolution, at the shared batch size
        mesh_t = mesh_tensors_of(args.geometry, coarse0, device, dtype)
        B = args.batch_size
        E = torch.tensor(np.repeat(coarse0["E_node"][None], B, 0), device=device, dtype=dtype)
        nu = torch.tensor(np.repeat(coarse0["nu_node"][None], B, 0), device=device, dtype=dtype)
        f = torch.tensor(np.repeat(coarse0["node_forces"][None], B, 0), device=device, dtype=dtype)
        for _ in range(5):
            with torch.no_grad():
                predict(args, mesh_t, model, E, nu, f, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts = []
        for _ in range(args.n_timing_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                predict(args, mesh_t, model, E, nu, f, dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ts.append(time.perf_counter() - t0)
        op_ms = 1000.0 * float(np.median(ts)) / B

        row = {"N": N, "n_nodes": int(coarse0["xy"].shape[0]),
               "fem_rel_L2": float(np.mean(fem_errs)),
               "fem_rel_L2_std": float(np.std(fem_errs)),
               "fem_ms_per_sample": 1000.0 * float(np.median(fem_times)),
               "operator_rel_L2": float(np.mean(op_errs)),
               "operator_rel_L2_std": float(np.std(op_errs)),
               "operator_ms_per_sample": op_ms}
        rows.append(row)
        print(f"  N={N:>4}  FEM {row['fem_rel_L2']:.4e} @ {row['fem_ms_per_sample']:9.1f} ms"
              f"   |  operator {row['operator_rel_L2']:.4e} @ {op_ms:7.3f} ms")

        rows.sort(key=lambda r: r["N"])
        report = {"geometry": args.geometry, "material": args.material,
                  "checkpoint": args.checkpoint,
                  "checkpoint_fingerprint": fingerprint,
                  "fine_N": args.fine_N,
                  "n_samples": args.n_samples, "batch_size": args.batch_size,
                  "device": device.type, "rows": rows}
        tmp = args.out_json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(report, fh, indent=2)
        os.replace(tmp, args.out_json)

    print(f"\nWritten to {args.out_json}")
    write_manifest(
        args.out_dir, kind="pareto_analysis", args=args, started_at=started,
        results=report, outputs=[args.out_json],
        notes=("Advisor point 2. Both sides measured on the same problem "
               "instances, against the same fine-mesh reference, on the same "
               "device at the same batch size -- the report's existing FEM "
               "convergence curve could not be reused because it scores a "
               "different problem against a different reference. FEM cost is "
               "the CPU reference solver's own solve time, which is what "
               "generating a new solution actually costs today; the GPU "
               "solver's figures are in Table 10 and are far lower."))


def predict(args, mesh_t, model, E, nu, f, dtype):
    if args.geometry == "B1":
        from omar_pfem.train_B1 import predict_displacement_Q4_only as pred
        xy, quad, top_edges, bottom_nodes = mesh_t
        return pred(xy, quad, top_edges, bottom_nodes, model, E, nu, f,
                    use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly,
                    dtype=dtype, fun_dim=args.fun_dim)
    from omar_pfem.train_B2 import predict_displacement_Q4_only as pred
    xy, quad, inner_edges, theta0, thalf = mesh_t
    return pred(xy, quad, inner_edges, theta0, thalf, model, E, nu, f,
                use_soft_dirichlet=args.use_soft_dirichlet, R_out=args.R_out,
                dtype=dtype, fun_dim=args.fun_dim)


if __name__ == "__main__":
    main()
