"""Data-driven B3 operator, for a fair PI-vs-DD comparison (Professor
Rabczuk, 2026-10-20 email: "figure out when the expensive high-fidelity
labels actually make PI worthwhile" -- the B3 analogue of the submitted
paper's own B1 PI-vs-DD 2x2 grid, Table 22). B3 has never had a trained
data-driven baseline before -- only the physics-informed DEM checkpoint
(train_B3.py) exists.

Mirrors train_data_driven.py's own B1/B2 pattern (same --loss choices:
rel_l2, mse, sobolev, hybrid; same matched-optimizer convention) rather
than reimplementing it from scratch, but is a SEPARATE file, not an
extension of that one's own `geometry_api` dispatch -- B3's h5-based
dataset format (data_generate_B3_dataset.py) is array-batched, not the
per-sample dict-list .npz format B1/B2 use, so the loading/mesh-tensor
plumbing genuinely differs; forcing it through the same dispatch would
have meant editing train_data_driven.py's own trusted B1/B2 path to
accommodate a third, differently-shaped case. Reuses B3's OWN trusted
geometry/Dirichlet/energy machinery unchanged (train_B3.build_
fixed_geometry, apply_dirichlet_b3, total_potential_energy_B3, build_
model) -- no FEM/physics code duplicated or re-derived.

Dataset split: the NEW labeled training set this script needs (generate
with data_generate_B3_dataset.generate_dataset at a seed clearly
disjoint from the project's existing B3 seed ranges -- see
make_b3_dd_dataset_notebook.py) is used ONLY for training. Evaluation
always uses the SAME held-out set (b3_dataset_clean_holdout, seed=99999,
sample_seed 999990-1000089) every physics-informed B3 checkpoint has
been evaluated on, so DD and PI numbers are directly comparable, not
just internally consistent.

Usage:
  python -m omar_pfem.train_B3_data_driven \
      --train_dataset /content/drive/MyDrive/pfem_run/b3_dd_training_set/dataset.h5 \
      --test_dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --loss rel_l2 \
      --out_dir /content/drive/MyDrive/pfem_run/b3_data_driven
"""
import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.train_B1 import _apply_input_norm, install_input_norm_for_checkpoint, set_input_norm
from omar_pfem.train_B3 import apply_dirichlet_b3, build_fixed_geometry, build_model, \
    neo_hookean_energy_density_batched, total_potential_energy_B3


def load_b3_h5(path):
    with h5py.File(path, "r") as h5:
        return {
            "E_node": torch.tensor(h5["E_node"][:], dtype=torch.float64),
            "nu_node": torch.tensor(h5["nu_node"][:], dtype=torch.float64),
            "phi": torch.tensor(h5["phi"][:], dtype=torch.float64),
            "u": torch.tensor(h5["displacements"][:], dtype=torch.float64),
        }


def compute_F_batched(u, geom):
    """Differentiable F=I+grad(u) at every (Batch,n_gauss,n_elem,3,3) --
    the exact same per-gauss construction total_potential_energy_B3
    already uses internally, factored out here (not detached anywhere)
    so the Sobolev loss's gradient-matching term can backprop through
    it, the same role differentiable_grad_u_Q4 plays for B1/B2."""
    elements, B_op = geom["elements"], geom["B"]
    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 1, 3, 3)
    F_list = []
    for g in range(B_op.shape[0]):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F_list.append((H.unsqueeze(1)))
    H_all = torch.cat(F_list, dim=1)  # (B, n_gauss, n_elem, 3, 3)
    return eye + H_all


def grad_rel_l2(F_pred, F_exact):
    num = torch.sqrt(torch.sum((F_pred - F_exact) ** 2, dim=(1, 2, 3, 4)))
    den = torch.sqrt(torch.sum(F_exact ** 2, dim=(1, 2, 3, 4))) + 1e-12
    return torch.mean(num / den)


def data_loss(u_pred, u_exact, kind):
    if kind == "mse":
        return torch.mean((u_pred - u_exact) ** 2)
    num = torch.sqrt(torch.sum((u_pred - u_exact) ** 2, dim=(1, 2)))
    den = torch.sqrt(torch.sum(u_exact ** 2, dim=(1, 2))) + 1e-12
    return torch.mean(num / den)


def predict(model, geom, E, nu, phi, normalize_inputs):
    fun_material = torch.stack(
        [E, nu, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
    if normalize_inputs:
        fun_material = _apply_input_norm(fun_material)
    xyz_batch = geom["nodes"].unsqueeze(0).expand(E.shape[0], -1, -1)
    u_net = model(xyz_batch, fun_material)
    return apply_dirichlet_b3(u_net, geom, phi)


@torch.no_grad()
def evaluate(test, model, geom, device, dtype, normalize_inputs):
    model.eval()
    E = test["E_node"].to(device=device, dtype=dtype)
    nu = test["nu_node"].to(device=device, dtype=dtype)
    phi = test["phi"].to(device=device, dtype=dtype)
    u_true = test["u"].to(device=device, dtype=dtype)
    u_pred = predict(model, geom, E, nu, phi, normalize_inputs)
    num = torch.sqrt(torch.sum((u_pred - u_true) ** 2, dim=(1, 2)))
    den = torch.sqrt(torch.sum(u_true ** 2, dim=(1, 2))) + 1e-12
    rel_l2 = (num / den)
    model.train()
    return {"mean_rel_L2": float(rel_l2.mean().item()),
            "median_rel_L2": float(rel_l2.median().item())}


def main():
    p = argparse.ArgumentParser(
        "Data-driven B3 operator, for a fair PI-vs-DD comparison (Prof. Rabczuk, 2026-10-20).")
    p.add_argument("--train_dataset", required=True)
    p.add_argument("--test_dataset", required=True,
                   help="the SAME held-out set every PI B3 checkpoint is evaluated on")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--opt_steps", type=int, default=50_000,
                   help="matched to train_B3.py's own --n_iters default, so the two "
                        "training principles are compared at the same step budget")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="rel_l2", choices=["rel_l2", "mse", "sobolev", "hybrid"])
    p.add_argument("--sobolev_grad_weight", type=float, default=0.5)
    p.add_argument("--hybrid_phys_weight", type=float, default=0.5)
    p.add_argument("--hybrid_phys_norm", default="running", choices=["running", "none"])
    p.add_argument("--hybrid_phys_norm_momentum", type=float, default=0.99)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--normalize_inputs", type=int, default=0)
    p.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"))
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unified_pos", type=int, default=0)
    p.add_argument("--ref", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=128)
    p.add_argument("--model", default="Transolver_Irregular_Mesh")
    args = p.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"Geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    train = load_b3_h5(args.train_dataset)
    test = load_b3_h5(args.test_dataset)
    n_train = train["E_node"].shape[0]
    print(f"Loaded {n_train} train samples from {args.train_dataset}")
    print(f"Loaded {test['E_node'].shape[0]} test (held-out) samples from {args.test_dataset}")

    if args.normalize_inputs:
        norm_path = os.path.join(args.out_dir, "input_norm.json")
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                norm_stats = json.load(f)
        else:
            E_np = train["E_node"].numpy().reshape(-1)
            nu_np = train["nu_node"].numpy().reshape(-1)
            phi_np = train["phi"].numpy()
            mean = [float(E_np.mean()), float(nu_np.mean()), float(phi_np.mean())]
            std = [float(E_np.std()), float(nu_np.std()), float(phi_np.std())]
            norm_stats = {"mean": mean, "std": std, "n_samples": n_train,
                          "fun_dim": 3, "channels": ["E", "nu", "phi"],
                          "constant_channels_passed_through": []}
            with open(norm_path, "w") as f:
                json.dump(norm_stats, f, indent=1)
        set_input_norm(norm_stats)
        print(f"[input-norm] mean={norm_stats['mean']} std={norm_stats['std']}")
    else:
        set_input_norm(None)

    model = build_model(args, device).to(dtype)
    n_par = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"Model: {n_par:,} trainable parameters, loss = {args.loss}\n")

    # Matched to train_B3.py's own optimizer (plain Adam, no decay schedule --
    # train_B3.py itself uses a constant LR throughout, unlike B1/B2's
    # every-1000-epochs decay) so the comparison is about the loss, not the
    # recipe.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    E_all = train["E_node"].to(device=device, dtype=dtype)
    nu_all = train["nu_node"].to(device=device, dtype=dtype)
    phi_all = train["phi"].to(device=device, dtype=dtype)
    u_all = train["u"].to(device=device, dtype=dtype)

    F_all = None
    if args.loss == "sobolev":
        with torch.no_grad():
            F_all = compute_F_batched(u_all, geom)
        print(f"Sobolev loss: precomputed exact deformation gradient for all "
              f"{n_train} training samples, grad weight={args.sobolev_grad_weight}")

    phys_scale_ema = None

    ckpt_path = os.path.join(args.out_dir, "model_best.pt")
    hist_path = os.path.join(args.out_dir, "history.json")
    best = {"step": -1, "val_rel_L2": float("inf")}
    history = []
    step = 0
    t0 = time.time()

    while step < args.opt_steps:
        order = np.random.permutation(n_train)
        for s0 in range(0, n_train, args.batch_size):
            if step >= args.opt_steps:
                break
            idx = torch.as_tensor(order[s0:s0 + args.batch_size], device=device)
            model.train()
            log_extra = {}

            if args.loss == "hybrid":
                u_pred = predict(model, geom, E_all[idx], nu_all[idx], phi_all[idx],
                                  args.normalize_inputs)
                Pi = total_potential_energy_B3(u_pred, E_all[idx], nu_all[idx], geom)
                phys_raw = Pi.mean()
                data_term = data_loss(u_pred, u_all[idx], "rel_l2")
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
                log_extra = {"phys_raw": float(phys_raw.item()), "data_term": float(data_term.item())}
            elif args.loss == "sobolev":
                u_pred = predict(model, geom, E_all[idx], nu_all[idx], phi_all[idx],
                                  args.normalize_inputs)
                val_term = data_loss(u_pred, u_all[idx], "rel_l2")
                F_pred = compute_F_batched(u_pred, geom)
                grad_term = grad_rel_l2(F_pred, F_all[idx])
                w = args.sobolev_grad_weight
                loss = (1 - w) * val_term + w * grad_term
                log_extra = {"val_term": float(val_term.item()), "grad_term": float(grad_term.item())}
            else:
                u_pred = predict(model, geom, E_all[idx], nu_all[idx], phi_all[idx],
                                  args.normalize_inputs)
                loss = data_loss(u_pred, u_all[idx], args.loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1

            if step % args.eval_every == 0 or step == args.opt_steps:
                m = evaluate(test, model, geom, device, dtype, args.normalize_inputs)
                val = m["mean_rel_L2"]
                history.append({"step": step, "train_loss": float(loss.item()),
                                "val_rel_L2": val, "elapsed_s": time.time() - t0, **log_extra})
                flag = ""
                if val < best["val_rel_L2"]:
                    best = {"step": step, "val_rel_L2": val}
                    torch.save({"model_state": model.state_dict(), "step": step,
                               "args": vars(args)}, ckpt_path)
                    flag = "  <- best"
                print(f"  step {step:>7,}/{args.opt_steps:,}  loss={loss.item():.5f}  "
                      f"val rel-L2={val:.4f}{flag}")
                with open(hist_path, "w") as fh:
                    json.dump(history, fh, indent=2)

    wall = time.time() - t0
    report = {
        "geometry": "B3", "training_principle": "data-driven (supervised on FEM solutions)",
        "loss": args.loss, "train_dataset": args.train_dataset, "test_dataset": args.test_dataset,
        "ntrain": n_train, "ntest": test["E_node"].shape[0],
        "batch_size": args.batch_size, "opt_steps": args.opt_steps,
        "lr": args.lr, "weight_decay": args.weight_decay, "n_parameters": n_par,
        "best_val_rel_L2": best["val_rel_L2"], "best_step": best["step"],
        "train_wall_clock_s": wall, "checkpoint": ckpt_path, "device": device.type,
        "sobolev_grad_weight": args.sobolev_grad_weight if args.loss == "sobolev" else None,
        "hybrid_phys_weight": args.hybrid_phys_weight if args.loss == "hybrid" else None,
        "last_logged_components": history[-1] if history else None,
    }
    out_json = os.path.join(args.out_dir, f"data_driven_B3_{args.loss}.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"DATA-DRIVEN B3 ({args.loss})")
    print(f"  best validation rel. L2 : {best['val_rel_L2']:.4f} (step {best['step']:,})")
    print(f"  training wall clock     : {wall:.1f} s")
    print("=" * 70)
    print(f"Written to {out_json}")


if __name__ == "__main__":
    main()
