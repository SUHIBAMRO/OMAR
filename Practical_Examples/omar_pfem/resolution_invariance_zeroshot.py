"""
TRUE zero-shot resolution-invariance study, replacing the earlier
resolution_invariance_study.py per the advisor's explicit correction:
"training ten independent networks on ten meshes is not a demonstration of
resolution invariance ... evaluate the SAME trained model on resolutions
not used during training. All resolutions should be compared against a
common fine-mesh FEM reference."

Design:
  1. TRAIN one Transolver model on TWO mesh resolutions at once (default
     N=21 and N=33), batches interleaved across resolutions every epoch --
     Transolver's per-node MLP + slice-attention design has no layer sized
     by the number of mesh nodes (verified directly against
     model/Transolver_Irregular_Mesh.py and model/Physics_Attention.py:
     every nn.Linear is sized by (dim, heads, slice_num), never by N, and N
     is read at runtime from the input tensor's own shape), so nothing
     about the architecture needs to change to train on two node counts.
     A single batch must still be resolution-homogeneous (all samples in
     one forward pass share one xy/quad tensor), so different mesh sizes
     alternate ACROSS batches, not within one.
  2. EVALUATE that one trained checkpoint, with NO further training or
     fine-tuning, at OTHER resolutions never seen during training (default
     N=25,29,37,41,49).
  3. Every evaluation is scored against a COMMON FINE-MESH FEM REFERENCE
     (default N=101), not each resolution's own same-N FEM solve: the same
     per-sample (E, nu, load) field (see data/parametric_field.py) is
     solved once at N_fine, and the network's prediction at whatever
     coarser N it was evaluated on is interpolated onto the fine mesh's
     own nodes before computing the relative-L2 error, exactly mirroring
     mesh_convergence.py's own node-to-node interpolation approach.

Why a NEW field generator (data/parametric_field.py) instead of the
GRF-based one used for the six main benchmark cases: the GRF sampler's
random phase array is sized by (Nx, Ny) -- the same seed at two mesh
resolutions is NOT the same continuous field at higher resolution, it is a
different realization, which would make "common reference" comparisons
meaningless. See data/parametric_field.py's own docstring for the full
rationale (identical reasoning to why mesh_convergence.py already made the
same substitution for its own, narrower purpose).

Usage:
  # Step 1: train once on two resolutions
  python -m omar_pfem.resolution_invariance_zeroshot train \
      --geometry B1 --material neo_hookean \
      --train_resolutions 21,33 --n_train_per_res 400 --n_val_per_res 100 \
      --epochs 2000 --batch_size 8 --validate_every 25 --early_stop_patience 8 \
      --out_dir ./zeroshot_B1_neo_hookean

  # Step 2: zero-shot evaluate on unseen resolutions vs. a common N=101 reference
  python -m omar_pfem.resolution_invariance_zeroshot eval \
      --geometry B1 --material neo_hookean \
      --checkpoint ./zeroshot_B1_neo_hookean/model_best.pt \
      --test_resolutions 25,29,37,41,49 --fine_N 101 --n_eval_samples 20 \
      --out_json ./zeroshot_B1_neo_hookean/zeroshot_eval.json
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator

from omar_pfem.model_dict import get_model
from omar_pfem.data.parametric_field import ParametricFieldB1, ParametricFieldB2


# ============================================================
# Sample construction: parametric field + existing (unmodified) FEM solver
# ============================================================

def build_sample_b1(N, seed, material, Lx=1.0, Ly=1.0, solve_fem=True):
    from omar_pfem.data.data_generate_B1 import generate_grid_Q4, solve_hyperelastic_TL_spatial

    nodes, elements = generate_grid_Q4(Lx, Ly, N, N)
    E_fn = ParametricFieldB1("E", seed)
    nu_fn = ParametricFieldB1("nu", seed)
    ty_fn = ParametricFieldB1("ty", seed)

    tolx = 1e-12
    bottom_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    top_edges = np.array([e for e in elements
                           if abs(nodes[e[2], 1] - Ly) < tolx and abs(nodes[e[3], 1] - Ly) < tolx])

    E_node = E_fn(nodes).astype(np.float32)
    nu_node = nu_fn(nodes).astype(np.float32)

    node_forces = np.zeros((len(nodes), 2), dtype=np.float32)
    if len(top_edges) > 0:
        top_nodes = np.unique(top_edges[:, 2:4].reshape(-1))
        node_forces[top_nodes, 1] = ty_fn(nodes[top_nodes]).astype(np.float32)

    uv_exact = None
    if solve_fem:
        uv_exact = solve_hyperelastic_TL_spatial(nodes, elements, E_fn, nu_fn, ty_fn, Ly,
                                                  nsteps=10, newton_max=30, tol=1e-7,
                                                  material=material).astype(np.float32)

    return {
        "xy": nodes.astype(np.float32), "quad": elements.astype(np.int64),
        "top_edges": top_edges.astype(np.int64) if len(top_edges) else np.zeros((0, 4), dtype=np.int64),
        "bottom_nodes": bottom_nodes.astype(np.int64),
        "E_node": E_node, "nu_node": nu_node, "node_forces": node_forces,
        "uv_exact": uv_exact,
    }, (E_fn, nu_fn, ty_fn)


def build_sample_b2(N, seed, material, R_in=1.0, R_out=2.0, solve_fem=True):
    from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring, solve_hyperelastic_TL_ring

    nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
    E_fn = ParametricFieldB2("E", seed)
    nu_fn = ParametricFieldB2("nu", seed)
    p_fn = ParametricFieldB2("p", seed)

    tolx = 1e-9
    theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]
    inner_edges = []
    for e in elements:
        r1 = np.linalg.norm(nodes[e[0]])
        r4 = np.linalg.norm(nodes[e[3]])
        if abs(r1 - R_in) < tolx and abs(r4 - R_in) < tolx:
            inner_edges.append(e)
    inner_edges = np.array(inner_edges) if inner_edges else np.zeros((0, 4), dtype=np.int64)

    def to_polar(pts):
        r = np.linalg.norm(pts, axis=1)
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        return np.stack([theta, r], axis=1)

    polar = to_polar(nodes)
    E_node = E_fn(polar).astype(np.float32)
    nu_node = nu_fn(polar).astype(np.float32)

    node_forces = np.zeros((len(nodes), 2), dtype=np.float32)
    if len(inner_edges) > 0:
        inner_nodes = np.unique(inner_edges[:, [0, 3]].reshape(-1))
        p_vals = p_fn(polar[inner_nodes])
        normals = nodes[inner_nodes] / np.linalg.norm(nodes[inner_nodes], axis=1, keepdims=True)
        node_forces[inner_nodes] = (p_vals[:, None] * normals).astype(np.float32)

    uv_exact = None
    if solve_fem:
        uv_exact = solve_hyperelastic_TL_ring(nodes, elements, E_fn, nu_fn, p_fn, R_in,
                                               nsteps=10, newton_max=30, tol=1e-7,
                                               material=material).astype(np.float32)

    return {
        "xy": nodes.astype(np.float32), "quad": elements.astype(np.int64),
        "inner_edges": inner_edges.astype(np.int64),
        "theta0_nodes": theta0_nodes.astype(np.int64), "thetahalfpi_nodes": thetahalfpi_nodes.astype(np.int64),
        "E_node": E_node, "nu_node": nu_node, "node_forces": node_forces,
        "uv_exact": uv_exact,
    }, (E_fn, nu_fn, p_fn)


# ============================================================
# Model + loss (thin re-exports so this module has one obvious source per
# geometry, matching train_B1.py / train_B2.py's own signatures exactly)
# ============================================================

def build_model(args, device):
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden, dropout=args.dropout,
        n_head=args.n_heads, Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim,
        out_dim=2, slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
    ).to(device)


def loss_and_pred(geometry, mesh_tensors, model, E_batch, nu_batch, force_batch, args, dtype):
    if geometry == "B1":
        from omar_pfem.train_B1 import total_potential_energy_Q4_hyperelastic
        xy, quad, top_edges, bottom_nodes = mesh_tensors
        return total_potential_energy_Q4_hyperelastic(
            xy, quad, top_edges, bottom_nodes, model, E_batch, nu_batch, force_batch,
            use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly, mode=args.mode,
            dtype=dtype, fun_dim=args.fun_dim, material=args.material)
    else:
        from omar_pfem.train_B2 import total_potential_energy_Q4_hyperelastic
        xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes = mesh_tensors
        return total_potential_energy_Q4_hyperelastic(
            xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes, model, E_batch, nu_batch, force_batch,
            use_soft_dirichlet=args.use_soft_dirichlet, R_out=args.R_out, mode=args.mode,
            dtype=dtype, fun_dim=args.fun_dim, material=args.material)


def mesh_tensors_of(geometry, sample, device, dtype):
    if geometry == "B1":
        return (torch.tensor(sample["xy"], device=device, dtype=dtype),
                torch.tensor(sample["quad"], device=device, dtype=torch.long),
                torch.tensor(sample["top_edges"], device=device, dtype=torch.long),
                torch.tensor(sample["bottom_nodes"], device=device, dtype=torch.long))
    else:
        return (torch.tensor(sample["xy"], device=device, dtype=dtype),
                torch.tensor(sample["quad"], device=device, dtype=torch.long),
                torch.tensor(sample["inner_edges"], device=device, dtype=torch.long),
                torch.tensor(sample["theta0_nodes"], device=device, dtype=torch.long),
                torch.tensor(sample["thetahalfpi_nodes"], device=device, dtype=torch.long))


@torch.no_grad()
def evaluate_resolution(geometry, samples, mesh_tensors, model, args, device, dtype):
    if len(samples) == 0:
        return None
    model.eval()
    E_batch = torch.tensor(np.stack([s["E_node"] for s in samples]), device=device, dtype=dtype)
    nu_batch = torch.tensor(np.stack([s["nu_node"] for s in samples]), device=device, dtype=dtype)
    force_batch = torch.tensor(np.stack([s["node_forces"] for s in samples]), device=device, dtype=dtype)
    uv_exact = torch.tensor(np.stack([s["uv_exact"] for s in samples]), device=device, dtype=dtype)

    _, _, _, uv_pred, _ = loss_and_pred(geometry, mesh_tensors, model, E_batch, nu_batch, force_batch, args, dtype)
    err = uv_pred - uv_exact
    l2_u = torch.sqrt(torch.mean(err[:, :, 0] ** 2, dim=1))
    l2_v = torch.sqrt(torch.mean(err[:, :, 1] ** 2, dim=1))
    ref_u = torch.sqrt(torch.mean(uv_exact[:, :, 0] ** 2, dim=1)) + 1e-12
    ref_v = torch.sqrt(torch.mean(uv_exact[:, :, 1] ** 2, dim=1)) + 1e-12
    rel_l2 = 0.5 * ((l2_u / ref_u) + (l2_v / ref_v))
    return float(rel_l2.mean().item())


# ============================================================
# Training: interleaved batches across the training resolutions
# ============================================================

def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    train_resolutions = [int(n) for n in args.train_resolutions.split(",") if n.strip()]
    build_fn = build_sample_b1 if args.geometry == "B1" else build_sample_b2

    print(f"Generating training/validation samples at resolutions {train_resolutions} "
          f"({args.n_train_per_res} train + {args.n_val_per_res} val each, real FEM solves)...")
    train_samples, val_samples, mesh_tensors = {}, {}, {}
    for N in train_resolutions:
        t0 = time.time()
        train_samples[N] = [build_fn(N, seed=10_000 * N + i, material=args.material)[0]
                             for i in range(args.n_train_per_res)]
        val_samples[N] = [build_fn(N, seed=10_000 * N + 500_000 + i, material=args.material)[0]
                           for i in range(args.n_val_per_res)]
        mesh_tensors[N] = mesh_tensors_of(args.geometry, train_samples[N][0], device, dtype)
        print(f"  N={N}: {len(train_samples[N])} train + {len(val_samples[N])} val samples "
              f"generated in {time.time()-t0:.1f}s")

    model = build_model(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    bs = max(int(args.batch_size), 1)
    metrics_history = []
    best_combined_val, best_epoch, patience_counter = None, None, 0
    train_wall_clock = 0.0
    opt_steps = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()

        # Build one shuffled list of (N, batch_indices) spanning ALL training
        # resolutions -- each individual batch is resolution-homogeneous
        # (required: one forward pass shares one xy/quad tensor), but
        # consecutive batches alternate resolution, so no single resolution
        # dominates a contiguous stretch of optimizer steps.
        batch_plan = []
        for N in train_resolutions:
            order = np.random.permutation(len(train_samples[N]))
            for start in range(0, len(order), bs):
                batch_plan.append((N, order[start:start + bs]))
        random.shuffle(batch_plan)

        for N, idx in batch_plan:
            samples = [train_samples[N][i] for i in idx]
            E_batch = torch.tensor(np.stack([s["E_node"] for s in samples]), device=device, dtype=dtype)
            nu_batch = torch.tensor(np.stack([s["nu_node"] for s in samples]), device=device, dtype=dtype)
            force_batch = torch.tensor(np.stack([s["node_forces"] for s in samples]), device=device, dtype=dtype)

            Pi, _, _, _, _ = loss_and_pred(args.geometry, mesh_tensors[N], model,
                                           E_batch, nu_batch, force_batch, args, dtype)
            loss = Pi.mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt_steps += 1

        epoch_time = time.time() - epoch_t0
        train_wall_clock += epoch_time

        if epoch % args.validate_every == 0:
            per_res_val = {}
            for N in train_resolutions:
                per_res_val[N] = evaluate_resolution(args.geometry, val_samples[N], mesh_tensors[N],
                                                      model, args, device, dtype)
            combined_val = float(np.mean(list(per_res_val.values())))
            print(f"[epoch {epoch}] per-resolution val_error={per_res_val}  "
                  f"combined(mean)={combined_val:.4e}  opt_steps={opt_steps}  "
                  f"cum_wall_clock={train_wall_clock:.1f}s")

            is_best = (best_combined_val is None) or (combined_val < best_combined_val - args.early_stop_min_delta)
            if is_best:
                best_combined_val, best_epoch, patience_counter = combined_val, epoch, 0
                torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pt"))
            else:
                patience_counter += 1

            metrics_history.append({
                "epoch": epoch, "per_resolution_val_error": per_res_val,
                "combined_val_error": combined_val, "is_best": is_best,
                "opt_steps": opt_steps, "cumulative_wall_clock_s": train_wall_clock,
            })
            with open(os.path.join(args.out_dir, "metrics_history.json"), "w") as f:
                json.dump(metrics_history, f, indent=2)

            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"[early stop] combined val_error hasn't improved for {patience_counter} "
                      f"validation events -- stopping at epoch {epoch}, best was epoch {best_epoch} "
                      f"(combined_val_error={best_combined_val:.4e})")
                with open(os.path.join(args.out_dir, "EARLY_STOPPED"), "w") as f:
                    f.write(f"stopped_at_epoch={epoch}\nbest_epoch={best_epoch}\n"
                            f"best_combined_val_error={best_combined_val}\n")
                break

        if epoch % 1000 == 0 and epoch < args.epochs:
            for pg in opt.param_groups:
                pg["lr"] *= 0.9
    else:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))

    print(f"\nTraining complete. Trained jointly on resolutions {train_resolutions}. "
          f"Best combined val_error={best_combined_val:.4e} at epoch={best_epoch}. "
          f"Checkpoint: {args.out_dir}/model_best.pt")


# ============================================================
# Zero-shot evaluation: SAME checkpoint, unseen resolutions, common fine reference
# ============================================================

def interpolate_to_reference(coarse_nodes, coarse_field, fine_nodes):
    """Interpolates a (n_coarse, 2) vector field onto fine_nodes' locations
    via linear interpolation over the coarse mesh's own triangulation --
    the same node-to-node interpolation approach mesh_convergence.py uses
    to compare different-resolution nodal solutions at shared query points."""
    interp_u = LinearNDInterpolator(coarse_nodes, coarse_field[:, 0])
    interp_v = LinearNDInterpolator(coarse_nodes, coarse_field[:, 1])
    u_at_fine = interp_u(fine_nodes)
    v_at_fine = interp_v(fine_nodes)
    # Points exactly on/outside the coarse mesh's convex hull boundary can
    # come back NaN from LinearNDInterpolator; nearest-neighbor fallback
    # for those few boundary points only.
    if np.any(np.isnan(u_at_fine)) or np.any(np.isnan(v_at_fine)):
        from scipy.interpolate import NearestNDInterpolator
        nn_u = NearestNDInterpolator(coarse_nodes, coarse_field[:, 0])
        nn_v = NearestNDInterpolator(coarse_nodes, coarse_field[:, 1])
        bad = np.isnan(u_at_fine)
        u_at_fine[bad] = nn_u(fine_nodes[bad])
        bad = np.isnan(v_at_fine)
        v_at_fine[bad] = nn_v(fine_nodes[bad])
    return np.stack([u_at_fine, v_at_fine], axis=1)


def cmd_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_resolutions = [int(n) for n in args.test_resolutions.split(",") if n.strip()]
    build_fn = build_sample_b1 if args.geometry == "B1" else build_sample_b2

    rows = []
    for N in test_resolutions:
        rel_l2_errors = []
        for i in range(args.n_eval_samples):
            seed = 20_000_000 + 10_000 * N + i  # disjoint from every training/val seed range
            # Coarse sample at N (network's own input resolution) -- no FEM
            # solve needed here, only the mesh + material/load fields the
            # network actually consumes.
            coarse_sample, _ = build_fn(N, seed=seed, material=args.material, solve_fem=False)
            # SAME seed, at the common fine reference resolution -- this
            # DOES need a real (expensive) nonlinear FEM solve.
            fine_sample, _ = build_fn(args.fine_N, seed=seed, material=args.material, solve_fem=True)

            mesh_t = mesh_tensors_of(args.geometry, coarse_sample, device, dtype)
            E_b = torch.tensor(coarse_sample["E_node"][None], device=device, dtype=dtype)
            nu_b = torch.tensor(coarse_sample["nu_node"][None], device=device, dtype=dtype)
            f_b = torch.tensor(coarse_sample["node_forces"][None], device=device, dtype=dtype)
            with torch.no_grad():
                _, _, _, uv_pred, _ = loss_and_pred(args.geometry, mesh_t, model, E_b, nu_b, f_b, args, dtype)
            uv_pred_np = uv_pred[0].cpu().numpy()

            uv_pred_on_fine = interpolate_to_reference(coarse_sample["xy"], uv_pred_np, fine_sample["xy"])
            uv_exact_fine = fine_sample["uv_exact"]

            err = uv_pred_on_fine - uv_exact_fine
            l2_u = np.sqrt(np.mean(err[:, 0] ** 2))
            l2_v = np.sqrt(np.mean(err[:, 1] ** 2))
            ref_u = np.sqrt(np.mean(uv_exact_fine[:, 0] ** 2)) + 1e-12
            ref_v = np.sqrt(np.mean(uv_exact_fine[:, 1] ** 2)) + 1e-12
            rel_l2_errors.append(0.5 * (l2_u / ref_u + l2_v / ref_v))

        row = {"N": N, "n_eval_samples": args.n_eval_samples,
               "mean_rel_L2_vs_fine_reference": float(np.mean(rel_l2_errors)),
               "std_rel_L2_vs_fine_reference": float(np.std(rel_l2_errors))}
        rows.append(row)
        print(f"N={N:>4d}: mean rel-L2 vs. N={args.fine_N} reference = "
              f"{row['mean_rel_L2_vs_fine_reference']:.4e} (+/- {row['std_rel_L2_vs_fine_reference']:.4e}, "
              f"{args.n_eval_samples} samples, NO retraining)")

    print("\n" + "=" * 90)
    print(f"ZERO-SHOT RESOLUTION-INVARIANCE EVAL (single checkpoint: {args.checkpoint})")
    print(f"Common fine-mesh reference: N={args.fine_N}")
    print("=" * 90)
    for row in rows:
        print(f"  N={row['N']:>4d}: {row['mean_rel_L2_vs_fine_reference']:.4e}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"geometry": args.geometry, "material": args.material,
                       "checkpoint": args.checkpoint, "fine_N": args.fine_N,
                       "test_resolutions": test_resolutions, "rows": rows}, f, indent=2)
        print(f"\nFull report written to {args.out_json}")


# ============================================================
# CLI
# ============================================================

def add_common_args(p):
    p.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    p.add_argument("--material", type=str, default="neo_hookean",
                    choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
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
    p.add_argument("--mode", type=str, default="plane_strain")
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--cpu", action="store_true")


def main():
    parser = argparse.ArgumentParser("True zero-shot resolution-invariance study")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    add_common_args(p_train)
    p_train.add_argument("--train_resolutions", type=str, default="21,33")
    p_train.add_argument("--n_train_per_res", type=int, default=400)
    p_train.add_argument("--n_val_per_res", type=int, default=100)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--epochs", type=int, default=2000)
    p_train.add_argument("--validate_every", type=int, default=25)
    p_train.add_argument("--early_stop_patience", type=int, default=8)
    p_train.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p_train.add_argument("--lr", type=float, default=2e-3)
    p_train.add_argument("--weight_decay", type=float, default=0.0)
    p_train.add_argument("--grad_clip", type=float, default=1.0)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--out_dir", type=str, default="./zeroshot_study")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--test_resolutions", type=str, default="25,29,37,41,49")
    p_eval.add_argument("--fine_N", type=int, default=101,
                         help="Common fine-mesh FEM reference resolution, per the advisor's request")
    p_eval.add_argument("--n_eval_samples", type=int, default=20)
    p_eval.add_argument("--out_json", type=str, default=None)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
