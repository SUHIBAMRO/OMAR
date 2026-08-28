"""The data-driven counterpart of the physics-informed operator.

The advisor's round-5 point 7b: compare the physics-informed model against a
data-driven one trained on finite-element solutions. Round 6 then said to
start with one problem, preferably B1 x Neo-Hookean.

What is held fixed, and why
---------------------------
The comparison is only meaningful if the ONLY thing that differs is the
training principle. So this script does not reimplement anything: it imports
the same dataset loader, the same forward pass including the soft-Dirichlet
construction, and the same evaluation routine that train_B1/train_B2 use,
and changes exactly one line -- the loss.

  physics-informed :  loss = mean over the batch of  Pi = U - W
  data-driven      :  loss = mean over the batch of  ||u_pred - u_fem|| / ||u_fem||

Same architecture, same mesh, same 800/200 split, same optimizer, same
learning-rate schedule, and -- the part that is easy to get wrong -- the
same OPTIMIZER-STEP budget rather than the same number of epochs, which is
the lesson Section 8.2 already had to learn once.

The cost that does not appear in the training log
-------------------------------------------------
The data-driven model needs `uv_exact`, a finite-element solution per
training sample. Those already exist in the dataset, so this run needs no new
FEM solves, but they were not free: at Table 4a's measured cost they
represent ntrain x (that case's seconds per sample) of CPU time that the
physics-informed model never spends. The script computes and reports that
number, because an accuracy comparison that omits it is not a comparison of
methods, only of outcomes.

Loss choice
-----------
The default is the per-sample relative L2, which is the standard operator-
learning loss and is also the metric this study reports, so the data-driven
model is trained on exactly what it is graded on -- the most favourable
honest setting for it, which is the right way round for a comparison whose
conclusion may be that the physics-informed model loses. `--loss mse` gives
the plain alternative; it weights large-displacement samples more heavily
and generally scores worse on the relative metric.

Usage:
  python -m omar_pfem.train_data_driven \
      --geometry B1 --material neo_hookean \
      --path .../hyperelastic_training_data_q4.npz \
      --ntrain 800 --ntest 200 --batch_size 8 --opt_steps 75000 \
      --out_dir .../data_driven_B1_neo_hookean
"""
import os
import json
import time
import random
import argparse

import numpy as np
import torch

from omar_pfem.model_dict import get_model
from omar_pfem.run_manifest import write_manifest

# Table 4a's measured native CPU FEM cost per sample, which is what the
# labels this model trains on actually cost to produce.
FEM_COST_S = {("B1", "neo_hookean"): 25.432, ("B1", "mooney_rivlin"): 53.735,
              ("B1", "arruda_boyce"): 52.542, ("B2", "neo_hookean"): 25.909,
              ("B2", "mooney_rivlin"): 61.712, ("B2", "arruda_boyce"): 60.285}


def geometry_api(geometry):
    """The same functions the physics-informed trainer uses, selected by
    geometry rather than reimplemented."""
    if geometry == "B1":
        from omar_pfem.train_B1 import (
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds,
            predict_displacement_Q4_only as predict,
            evaluate_dataset_hyperelastic_Q4 as evaluate)
    else:
        from omar_pfem.train_B2 import (
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds,
            predict_displacement_Q4_only as predict,
            evaluate_dataset_hyperelastic_Q4 as evaluate)
    # `predict_displacement_Q4_only` is decorated @torch.no_grad(), which is
    # right for its own callers (the latency benchmark, pareto_analysis) and
    # fatal here: the loss would arrive with no grad_fn and backward() would
    # raise "element 0 of tensors does not require grad". Unwrap it rather
    # than either removing the decorator -- which would silently change what
    # the latency benchmark measures -- or copying the forward pass and the
    # soft-Dirichlet ramp into this file, which would defeat the whole point
    # of the comparison by giving the two models two code paths.
    train_predict = getattr(predict, "__wrapped__", None)
    assert train_predict is not None, (
        "predict_displacement_Q4_only is no longer wrapped in @torch.no_grad(); "
        "check whether it is now differentiable and use it directly")
    return load_ds, train_predict, evaluate


def mesh_tensors(geometry, sample, device, dtype):
    t = lambda a, d: torch.tensor(a, device=device, dtype=d)
    if geometry == "B1":
        return (t(sample["xy"], dtype), t(sample["quad"], torch.long),
                t(sample["top_edges"], torch.long), t(sample["bottom_nodes"], torch.long))
    return (t(sample["xy"], dtype), t(sample["quad"], torch.long),
            t(sample["inner_edges"], torch.long), t(sample["theta0_nodes"], torch.long),
            t(sample["thetahalfpi_nodes"], torch.long))


def forward(geometry, predict, mesh_t, model, E, nu, f, args, dtype):
    if geometry == "B1":
        xy, quad, top_edges, bottom_nodes = mesh_t
        return predict(xy, quad, top_edges, bottom_nodes, model, E, nu, f,
                       use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly,
                       dtype=dtype, fun_dim=args.fun_dim)
    xy, quad, inner_edges, theta0, thalf = mesh_t
    return predict(xy, quad, inner_edges, theta0, thalf, model, E, nu, f,
                   use_soft_dirichlet=args.use_soft_dirichlet, R_out=args.R_out,
                   dtype=dtype, fun_dim=args.fun_dim)


def data_loss(uv_pred, uv_exact, kind):
    if kind == "mse":
        return torch.mean((uv_pred - uv_exact) ** 2)
    # per-sample relative L2, averaged over the batch: the same shape as the
    # reported metric, so training and grading agree
    num = torch.sqrt(torch.sum((uv_pred - uv_exact) ** 2, dim=(1, 2)))
    den = torch.sqrt(torch.sum(uv_exact ** 2, dim=(1, 2))) + 1e-12
    return torch.mean(num / den)


def main():
    p = argparse.ArgumentParser("Data-driven operator, for comparison with the "
                                "physics-informed one (advisor point 7b)")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", required=True,
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--path", required=True, help="the SAME .npz the PI model trained on")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ntrain", type=int, default=800)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--opt_steps", type=int, default=75_000,
                   help="matched to the physics-informed run's own step count "
                        "(Table 7), NOT to its epoch count")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="rel_l2", choices=["rel_l2", "mse"])
    p.add_argument("--eval_every", type=int, default=2000, help="in optimizer steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    # architecture -- identical to the physics-informed runs
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
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    args = p.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_ds, predict, evaluate = geometry_api(args.geometry)
    train, test = load_ds(args.path, args.ntrain, args.ntest)
    print(f"Loaded {len(train)} train / {len(test)} test from {args.path}")

    label_cost_s = len(train) * FEM_COST_S.get((args.geometry, args.material), float("nan"))
    print(f"Label cost NOT shown in this run's wall clock: {len(train)} FEM solves "
          f"= {label_cost_s / 3600:.2f} h of CPU (Table 4a's measured per-sample cost).\n"
          f"The physics-informed model spends none of it.\n")

    model = get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos).to(device)
    n_par = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"Model: {n_par:,} trainable parameters, loss = {args.loss}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.opt_steps, pct_start=0.1)

    mesh_t = mesh_tensors(args.geometry, train[0], device, dtype)
    uv_all = torch.tensor(np.stack([s["uv_exact"] for s in train]), device=device, dtype=dtype)
    E_all = torch.tensor(np.stack([s["E_node"] for s in train]), device=device, dtype=dtype)
    nu_all = torch.tensor(np.stack([s["nu_node"] for s in train]), device=device, dtype=dtype)
    f_all = torch.tensor(np.stack([s["node_forces"] for s in train]), device=device, dtype=dtype)

    ckpt_path = os.path.join(args.out_dir, "model_best.pt")
    hist_path = os.path.join(args.out_dir, "history.json")
    best = {"step": -1, "val_rel_L2": float("inf")}
    history = []
    n = len(train)
    step = 0
    t0 = time.time()

    while step < args.opt_steps:
        order = np.random.permutation(n)
        for s0 in range(0, n, args.batch_size):
            if step >= args.opt_steps:
                break
            idx = torch.as_tensor(order[s0:s0 + args.batch_size], device=device)
            model.train()
            uv = forward(args.geometry, predict, mesh_t, model,
                         E_all[idx], nu_all[idx], f_all[idx], args, dtype)
            loss = data_loss(uv, uv_all[idx], args.loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1

            if step % args.eval_every == 0 or step == args.opt_steps:
                m = evaluate(test, model, args, device, dtype)
                val = 0.5 * (m["mean_rel_L2_u"] + m["mean_rel_L2_v"])
                history.append({"step": step, "train_loss": float(loss.item()),
                                "val_rel_L2": val, "elapsed_s": time.time() - t0})
                flag = ""
                if val < best["val_rel_L2"]:
                    best = {"step": step, "val_rel_L2": val}
                    torch.save(model.state_dict(), ckpt_path)
                    flag = "  <- best"
                print(f"  step {step:>7,}/{args.opt_steps:,}  loss={loss.item():.5f}  "
                      f"val rel-L2={val:.4f}{flag}")
                with open(hist_path, "w") as fh:
                    json.dump(history, fh, indent=2)

    wall = time.time() - t0
    report = {
        "geometry": args.geometry, "material": args.material,
        "training_principle": "data-driven (supervised on FEM solutions)",
        "loss": args.loss, "path": args.path,
        "ntrain": len(train), "ntest": len(test),
        "batch_size": args.batch_size, "opt_steps": args.opt_steps,
        "n_parameters": n_par,
        "best_val_rel_L2": best["val_rel_L2"], "best_step": best["step"],
        "train_wall_clock_s": wall,
        "label_generation_cost_s": label_cost_s,
        "label_generation_cost_h": label_cost_s / 3600.0,
        "total_cost_including_labels_s": wall + label_cost_s,
        "checkpoint": ckpt_path, "device": device.type,
    }
    out_json = os.path.join(args.out_dir, f"data_driven_{args.geometry}_{args.material}.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"DATA-DRIVEN  {args.geometry} x {args.material}")
    print(f"  best validation rel. L2 : {best['val_rel_L2']:.4f} (step {best['step']:,})")
    print(f"  training wall clock     : {wall:.1f} s")
    print(f"  label generation        : {label_cost_s:.0f} s ({label_cost_s / 3600:.2f} h)")
    print(f"  total, labels included  : {(wall + label_cost_s) / 3600:.2f} h")
    print("=" * 70)
    print(f"Written to {out_json}")

    write_manifest(
        args.out_dir, kind="train_data_driven", args=args, started_at=started,
        results=report, outputs=[out_json, ckpt_path, hist_path],
        notes=("Advisor point 7b. Identical architecture, dataset, split, "
               "optimizer and optimizer-step budget to the physics-informed "
               "run; the ONLY difference is the loss. The label-generation "
               "cost is reported alongside accuracy because the data-driven "
               "model requires a finite-element solution per training sample "
               "and the physics-informed one requires none -- an accuracy "
               "comparison that omits it compares outcomes, not methods."))


if __name__ == "__main__":
    main()
