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

Same architecture, same mesh, same 800/200 split, the same OPTIMIZER-STEP
budget rather than the same number of epochs (the lesson Section 8.2 already
had to learn once), and -- by default, and this had to be fixed once -- the
same optimizer and learning-rate schedule: plain Adam at lr=2e-3 with
weight_decay=0, decayed by 0.9 every 1000 epochs, exactly as train_B1/B2 do.

The first version of this script used AdamW with OneCycleLR, which is very
likely a better recipe and gave a better number. That made the experiment a
comparison of optimizers wearing the clothes of a comparison of losses, so
the default now matches; `--match_pi_optimizer 0` restores the other recipe
for anyone who wants it, clearly labelled in the output JSON.

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

Two further baselines, added per the advisor's follow-up review asking for
"a Sobolev-trained DD baseline and possibly a hybrid physics+data baseline
on the representative case":

  `--loss sobolev`: adds a gradient-matching term to the plain data loss --
  (1-w)*rel_L2(u_pred, u_fem) + w*rel_L2(grad_u_pred, grad_u_fem), weight
  `--sobolev_grad_weight` (default 0.5). grad_u here is exactly the
  deformation-gradient contribution (I + du/dX) at every Gauss point that
  compute_hyperelastic_energy_Q4 already computes for the energy integral --
  reused via a differentiable twin of that function (differentiable_grad_u_Q4
  below; the original stays untouched and keeps returning a detached Fg,
  since every other caller only ever used it for post-hoc inspection). This
  is a genuine Sobolev / H1-type loss: matching the field's own derivative,
  not just its value, to the FEM reference -- the same H1 semi-norm this
  study already reports as a QoI, now used as a training signal instead of
  only an evaluation metric.

  `--loss hybrid`: combines the physics-informed objective (Pi = U - W,
  identical to train_B1/B2's own loss) with the data loss above, both
  computed from the SAME forward pass (one call to
  total_potential_energy_Q4_hyperelastic, so both terms see the same
  dropout mask -- calling the data-only forward and the physics forward
  separately would silently blend two different stochastic forward passes
  as if they were one). Pi and the relative-L2 data loss live on
  incomparable scales (an energy functional vs. a value in roughly [0,1]),
  so Pi is rescaled by a running estimate of its own magnitude
  (`--hybrid_phys_norm running`, an exponential moving average of
  |Pi.mean()|, momentum `--hybrid_phys_norm_momentum`, default 0.99) before
  being combined at weight `--hybrid_phys_weight` (default 0.5); `--loss
  hybrid --hybrid_phys_norm none` combines the raw, unnormalized terms
  instead, for anyone who wants to see what that looks like. Both the raw
  and the normalized components are logged to history.json at every
  evaluation step, so the run is auditable even if a different weighting
  would have been more balanced.

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
from omar_pfem.train_B1 import shape_Q4_torch

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
            evaluate_dataset_hyperelastic_Q4 as evaluate,
            total_potential_energy_Q4_hyperelastic as energy)
    else:
        from omar_pfem.train_B2 import (
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds,
            predict_displacement_Q4_only as predict,
            evaluate_dataset_hyperelastic_Q4 as evaluate,
            total_potential_energy_Q4_hyperelastic as energy)
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
    return load_ds, train_predict, evaluate, energy


def differentiable_grad_u_Q4(xy, quad, uv, dtype):
    """A differentiable twin of compute_hyperelastic_energy_Q4's own
    deformation-gradient computation (train_B1.py/train_B2.py), stripped of
    the material/energy-density evaluation and, critically, WITHOUT the
    `.detach()` that function applies to its own Fg output (correct for its
    actual callers -- the energy loss never needs Fg's gradient, only the
    network's -- but fatal for a Sobolev loss, which needs gradients to flow
    back into the network through the deformation gradient itself). Uses the
    exact same reference-element shape functions and 2x2 Gauss rule, so the
    quantity being matched here is identical to what compute_hyperelastic_
    energy_Q4 would have computed, not an approximation of it.

    uv: (B,N,2). Returns (B,Q,2,2), the per-sample, per-Gauss-point
    deformation gradient F = I + du/dX, differentiable w.r.t. uv."""
    device = xy.device
    B = uv.shape[0]
    Q = quad.shape[0]
    Xe = xy[quad]        # (Q,4,2)
    ue = uv[:, quad]     # (B,Q,4,2)

    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    F_list = []
    for (xi, eta) in gps:
        xi_t = torch.tensor(float(xi), device=device, dtype=dtype)
        eta_t = torch.tensor(float(eta), device=device, dtype=dtype)
        _, dN_dxi = shape_Q4_torch(xi_t, eta_t, device, dtype)

        J0 = torch.einsum("qai,aj->qij", Xe, dN_dxi)
        detJ0 = torch.clamp(J0[:, 0, 0] * J0[:, 1, 1] - J0[:, 0, 1] * J0[:, 1, 0], min=1e-12)
        invJ0 = torch.zeros_like(J0)
        invJ0[:, 0, 0] = J0[:, 1, 1] / detJ0
        invJ0[:, 1, 1] = J0[:, 0, 0] / detJ0
        invJ0[:, 0, 1] = -J0[:, 0, 1] / detJ0
        invJ0[:, 1, 0] = -J0[:, 1, 0] / detJ0
        dN_dX = torch.einsum("aj,qjk->qak", dN_dxi, invJ0)

        grad_u = torch.einsum("bqai,qaj->bqij", ue, dN_dX)
        I = torch.eye(2, device=device, dtype=dtype).reshape(1, 1, 2, 2).expand(B, Q, 2, 2)
        F_list.append((I + grad_u).unsqueeze(1))  # (B,1,Q,2,2)
    return torch.cat(F_list, dim=1).reshape(B, 4 * Q, 2, 2)


def grad_rel_l2(F_pred, F_exact):
    """Same relative-L2 shape as data_loss()'s rel_l2 branch, applied to the
    flattened (B, 4*Q, 2, 2) deformation-gradient tensor instead of (B,N,2)
    displacement -- the Sobolev loss's derivative term."""
    num = torch.sqrt(torch.sum((F_pred - F_exact) ** 2, dim=(1, 2, 3)))
    den = torch.sqrt(torch.sum(F_exact ** 2, dim=(1, 2, 3))) + 1e-12
    return torch.mean(num / den)


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


def forward_energy(geometry, energy, mesh_t, model, E, nu, f, args, dtype):
    """Same dispatch as forward(), but through the differentiable energy
    function (Pi, U, W, uv, Fg) instead of the no-grad-only predict wrapper
    -- used by `--loss hybrid`, which needs Pi and uv from a SINGLE forward
    pass (see the module docstring on why two separate forward calls would
    be wrong whenever dropout is active)."""
    if geometry == "B1":
        xy, quad, top_edges, bottom_nodes = mesh_t
        return energy(xy, quad, top_edges, bottom_nodes, model, E, nu, f,
                     use_soft_dirichlet=args.use_soft_dirichlet, Ly=args.Ly,
                     dtype=dtype, fun_dim=args.fun_dim, material=args.material)
    xy, quad, inner_edges, theta0, thalf = mesh_t
    return energy(xy, quad, inner_edges, theta0, thalf, model, E, nu, f,
                 use_soft_dirichlet=args.use_soft_dirichlet, R_out=args.R_out,
                 dtype=dtype, fun_dim=args.fun_dim, material=args.material)


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
    p.add_argument("--lr", type=float, default=2e-3,
                   help="train_B1/B2's own default, so the optimizer matches")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="train_B1/B2's own default (Adam, no decay)")
    p.add_argument("--match_pi_optimizer", type=int, default=1,
                   help="1 = Adam with the physics-informed run's own LR schedule "
                        "(x0.9 every 1000 epochs). 0 = AdamW + OneCycleLR, which is "
                        "a better recipe but makes the comparison about the recipe "
                        "rather than the loss.")
    p.add_argument("--steps_per_epoch", type=int, default=100,
                   help="ntrain/batch_size, so 'epoch' means the same thing in both "
                        "runs and the x0.9-every-1000-epochs decay lands identically")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="rel_l2",
                   choices=["rel_l2", "mse", "sobolev", "hybrid"])
    p.add_argument("--sobolev_grad_weight", type=float, default=0.5,
                   help="--loss sobolev: weight on the deformation-gradient "
                        "term, (1-w)*rel_L2(u) + w*rel_L2(grad_u)")
    p.add_argument("--hybrid_phys_weight", type=float, default=0.5,
                   help="--loss hybrid: weight on the (optionally normalized) "
                        "physics term, w*Pi_term + (1-w)*rel_L2(u)")
    p.add_argument("--hybrid_phys_norm", default="running", choices=["running", "none"],
                   help="--loss hybrid: rescale Pi by a running EMA of its own "
                        "|mean| before combining with the data loss (default), "
                        "or combine the raw, unnormalized terms ('none')")
    p.add_argument("--hybrid_phys_norm_momentum", type=float, default=0.99)
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

    load_ds, predict, evaluate, energy = geometry_api(args.geometry)
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

    # The comparison is only about the training principle if everything else
    # matches. train_B1/B2 use plain Adam at lr=2e-3 with weight_decay=0 and a
    # hand-rolled decay -- multiply the LR by 0.9 every 1000 epochs -- and NOT
    # AdamW with OneCycleLR. An earlier version of this script used the latter,
    # which is very likely the better recipe but turns the experiment into a
    # comparison of optimizers rather than of losses.
    if args.match_pi_optimizer:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        sched = None
        print(f"Optimizer: Adam lr={args.lr}, wd={args.weight_decay}, "
              f"x0.9 every {1000 * args.steps_per_epoch:,} steps "
              f"-- matched to train_{args.geometry}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.opt_steps, pct_start=0.1)
        print("Optimizer: AdamW + OneCycleLR -- NOT matched to the "
              "physics-informed run; the comparison is then about the recipe")

    mesh_t = mesh_tensors(args.geometry, train[0], device, dtype)
    uv_all = torch.tensor(np.stack([s["uv_exact"] for s in train]), device=device, dtype=dtype)
    E_all = torch.tensor(np.stack([s["E_node"] for s in train]), device=device, dtype=dtype)
    nu_all = torch.tensor(np.stack([s["nu_node"] for s in train]), device=device, dtype=dtype)
    f_all = torch.tensor(np.stack([s["node_forces"] for s in train]), device=device, dtype=dtype)

    graduv_all = None
    if args.loss == "sobolev":
        xy_mesh, quad_mesh = mesh_t[0], mesh_t[1]
        with torch.no_grad():
            graduv_all = differentiable_grad_u_Q4(xy_mesh, quad_mesh, uv_all, dtype)
        print(f"Sobolev loss: precomputed exact deformation gradient for all "
              f"{len(train)} training samples, grad weight={args.sobolev_grad_weight}")

    phys_scale_ema = None  # --loss hybrid, --hybrid_phys_norm running

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
            log_extra = {}
            if args.loss == "hybrid":
                Pi, _U, _W, uv, _Fg = forward_energy(
                    args.geometry, energy, mesh_t, model,
                    E_all[idx], nu_all[idx], f_all[idx], args, dtype)
                phys_raw = Pi.mean()
                data_term = data_loss(uv, uv_all[idx], "rel_l2")
                if args.hybrid_phys_norm == "running":
                    cur = float(phys_raw.detach().abs().item()) + 1e-12
                    phys_scale_ema = (cur if phys_scale_ema is None else
                                      args.hybrid_phys_norm_momentum * phys_scale_ema +
                                      (1 - args.hybrid_phys_norm_momentum) * cur)
                    phys_term = phys_raw / phys_scale_ema
                else:
                    phys_term = phys_raw
                w = args.hybrid_phys_weight
                loss = w * phys_term + (1 - w) * data_term
                log_extra = {"phys_raw": float(phys_raw.item()),
                             "phys_term": float(phys_term.item()),
                             "data_term": float(data_term.item())}
            elif args.loss == "sobolev":
                uv = forward(args.geometry, predict, mesh_t, model,
                             E_all[idx], nu_all[idx], f_all[idx], args, dtype)
                val_term = data_loss(uv, uv_all[idx], "rel_l2")
                xy_mesh, quad_mesh = mesh_t[0], mesh_t[1]
                F_pred = differentiable_grad_u_Q4(xy_mesh, quad_mesh, uv, dtype)
                grad_term = grad_rel_l2(F_pred, graduv_all[idx])
                w = args.sobolev_grad_weight
                loss = (1 - w) * val_term + w * grad_term
                log_extra = {"val_term": float(val_term.item()),
                             "grad_term": float(grad_term.item())}
            else:
                uv = forward(args.geometry, predict, mesh_t, model,
                             E_all[idx], nu_all[idx], f_all[idx], args, dtype)
                loss = data_loss(uv, uv_all[idx], args.loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            elif step > 0 and step % (1000 * args.steps_per_epoch) == 0:
                for pg in opt.param_groups:
                    pg["lr"] *= 0.9
                print(f"  step {step:,}: lr -> {opt.param_groups[0]['lr']:.2e}")
            step += 1

            if step % args.eval_every == 0 or step == args.opt_steps:
                m = evaluate(test, model, args, device, dtype)
                val = 0.5 * (m["mean_rel_L2_u"] + m["mean_rel_L2_v"])
                history.append({"step": step, "train_loss": float(loss.item()),
                                "val_rel_L2": val, "elapsed_s": time.time() - t0,
                                **log_extra})
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
        "optimizer": ("Adam + x0.9/1000 epochs, matched to the physics-informed run"
                      if args.match_pi_optimizer else "AdamW + OneCycleLR, NOT matched"),
        "lr": args.lr, "weight_decay": args.weight_decay,
        "n_parameters": n_par,
        "best_val_rel_L2": best["val_rel_L2"], "best_step": best["step"],
        "train_wall_clock_s": wall,
        "label_generation_cost_s": label_cost_s,
        "label_generation_cost_h": label_cost_s / 3600.0,
        "total_cost_including_labels_s": wall + label_cost_s,
        "checkpoint": ckpt_path, "device": device.type,
        "sobolev_grad_weight": args.sobolev_grad_weight if args.loss == "sobolev" else None,
        "hybrid_phys_weight": args.hybrid_phys_weight if args.loss == "hybrid" else None,
        "hybrid_phys_norm": args.hybrid_phys_norm if args.loss == "hybrid" else None,
        "last_logged_components": history[-1] if history else None,
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
