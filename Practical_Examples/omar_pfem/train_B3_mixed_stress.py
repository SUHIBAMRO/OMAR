"""Mixed displacement-stress training for B3 (Professor Rabczuk's item 1
of 4, 2026-10-16 follow-up-paper email -- see PROJECT_STATUS.md). The
network predicts both u and the first Piola-Kirchhoff stress P; the
loss is the two physics residuals mixed_stress_B3.mixed_stress_loss_B3
computes (constitutive consistency P=dpsi/dF, and equilibrium
Div_0(P)=0), NOT the total potential energy train_B3.py minimizes --
this is a different training paradigm, not a modification of that
script (see mixed_stress_B3.py's own module docstring for why it is
kept entirely separate from train_B3.py).

Mirrors train_B3.py's own CLI/training-loop structure as closely as
possible so the two are easy to compare side by side; every piece that
does not need to change (sampling, geometry construction, input
normalization, the Dirichlet-BC ansatz for u) is imported from
train_B3.py rather than copy-pasted.

STATUS: the two underlying operators (constitutive, equilibrium) and
the loss function's own plumbing are verified on CPU to 1e-10 or better
(omar_pfem/test_mixed_stress_B3.py, all 5 checks passing) -- but this
training script itself has NOT yet been run, not even briefly. Two
things should be checked on a short run (a few hundred iterations,
small batch) before committing to any long training run:

1. P_OUTPUT_SCALE below (the stress-channel analogue of train_B3.py's
   own OUTPUT_SCALE for displacement) is a first-cut estimate, not yet
   empirically confirmed. train_B3.py's OUTPUT_SCALE=0.02 was only
   found after a real production run showed the loss climbing without
   bound (see that constant's own docstring) -- the untrained network's
   raw output (~0.3 magnitude) was 5-50x too large relative to the true
   displacement scale (~0.01-0.06). The SAME kind of mismatch is
   expected here, in the opposite direction: true P values are O(mu),
   i.e. O(100-1000) for this project's E~1000, nu~0.45 sampling range,
   while the untrained network's raw per-channel output starts at the
   same ~0.3 magnitude regardless of which channel it is -- so without
   some compensating scale, the constitutive residual will start
   enormous (P_pred under-scaled by ~1000x relative to P_true) even
   though nothing is actually unstable. P_OUTPUT_SCALE multiplies the
   network's raw stress channels before they enter the loss, mirroring
   OUTPUT_SCALE's own role, with an initial guess of 500.0 (mid-range
   of this project's own mu = E/(2(1+nu)) for E~1000, nu~0.45, i.e.
   ~345) -- confirm or correct this the same way OUTPUT_SCALE was
   confirmed: watch the loss trajectory over a short run on a FIXED
   batch (no new sampling noise each step) and check it settles to a
   stable plateau rather than diverging, before trusting any longer run.
2. The constitutive and equilibrium residuals are NOT the same order of
   magnitude -- CONFIRMED empirically, not just a priori, via a 200-
   iteration fixed-batch diagnostic (same sample every iteration, no
   sampling noise, production mesh, a small toy model) at the naive
   w_constitutive=w_equilibrium=1.0: the loss is stable (no divergence,
   settles to a plateau) but is 99.995%+ dominated by the constitutive
   term from iteration 1 (62,047 vs. 2.18, a ~28,000x gap), so
   equilibrium's gradient contribution is noise-level and is not
   actually being optimized at those weights -- it drifts slightly
   WORSE over the run (2.18 -> 2.61) while constitutive slowly improves,
   exactly the signature of one term's gradient swamping the other's.
   Re-running the same diagnostic with w_equilibrium=28000 (the
   measured initial ratio) makes equilibrium converge to ~0 within ~20
   iterations while constitutive still improves and the total loss
   remains stable -- confirming the problem is pure scale imbalance,
   not an optimization pathology, and that w_equilibrium=28000 (the
   default below) is a reasonable starting point, not a guess. This
   ratio was measured on a tiny, undertrained toy model, so it should
   still be re-checked on the first real production-sized run (see
   diagnose_mixed_stress_B3_scale.py), the same way OUTPUT_SCALE's own
   value was confirmed at production scale before being trusted.
"""
import argparse
import os
import time

import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, mixed_stress_loss_B3
from omar_pfem.model_dict import get_model
from omar_pfem.train_B3 import (
    _apply_input_norm, compute_input_norm_b3, sample_batch, set_input_norm,
)

P_OUTPUT_SCALE = 500.0
"""First-cut estimate, NOT yet empirically confirmed -- see this
module's own docstring, point 1, before trusting a full training run."""


def build_model(args, device):
    return get_model(args).Model(
        space_dim=3,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=3,  # E, nu, phi -- same as train_B3.py
        out_dim=12,  # 3 (u) + 9 (P, flattened row-major i,I), vs. train_B3.py's 3
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=args.unified_pos,
    ).to(device)


def train(args, device, dtype=torch.float64, resolution=None):
    Ntheta, Nr, Nz = resolution or DEFAULT_RESOLUTION
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"Fixed geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    model = build_model(args, device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    xyz = geom["nodes"]
    os.makedirs(args.output_dir, exist_ok=True)

    norm_path = os.path.join(args.output_dir, "input_norm.json")
    if int(getattr(args, "normalize_inputs", 0)):
        import json
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                norm_stats = json.load(f)
            print(f"[input-norm] reusing {norm_path} from an earlier run")
        else:
            norm_stats = compute_input_norm_b3(Ntheta, Nr, Nz, n_reference_samples=200,
                                                dtype=dtype, device=device)
            with open(norm_path, "w") as f:
                json.dump(norm_stats, f, indent=1)
            print(f"[input-norm] computed from {norm_stats['n_samples']} reference "
                  f"samples and written to {norm_path}")
        set_input_norm(norm_stats)
    else:
        set_input_norm(None)

    t0 = time.time()
    for it in range(1, args.n_iters + 1):
        E_node, nu_node, phi = sample_batch(args.batch_size, Ntheta, Nr, Nz,
                                             seed_offset=it * args.batch_size,
                                             dtype=dtype, device=device)
        fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
        fun_material = _apply_input_norm(fun_material)

        xyz_batch = xyz.unsqueeze(0).expand(args.batch_size, -1, -1)
        raw = model(xyz_batch, fun_material)  # (Batch, N, 12)
        u_net_raw = raw[..., :3]
        p_net_raw = raw[..., 3:] * args.p_output_scale

        loss, parts = mixed_stress_loss_B3(
            u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
            w_constitutive=args.w_constitutive, w_equilibrium=args.w_equilibrium)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"[{it}/{args.n_iters}] loss={loss.item():.6f} "
                  f"constitutive={parts['constitutive']:.6f} "
                  f"equilibrium={parts['equilibrium']:.6f} elapsed={elapsed:.1f}s")

        if it % args.ckpt_every == 0 or it == args.n_iters:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{it}.pt")
            torch.save({"model_state": model.state_dict(), "iter": it, "args": vars(args)}, ckpt_path)

    return model


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_iters", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unified_pos", type=int, default=0)
    p.add_argument("--ref", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=128)
    p.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="b3_mixed_stress_output")
    p.add_argument("--normalize_inputs", type=int, default=0)
    p.add_argument("--p_output_scale", type=float, default=P_OUTPUT_SCALE,
                    help="Multiplies the network's raw stress-channel output before it enters "
                         "the loss -- see this module's own docstring, point 1. NOT yet "
                         "empirically confirmed; check the loss trajectory on a short run first.")
    p.add_argument("--w_constitutive", type=float, default=1.0)
    p.add_argument("--w_equilibrium", type=float, default=28000.0,
                    help="Relative weight of the two residuals -- see this module's own "
                         "docstring, point 2. Measured on a tiny toy model via a fixed-batch "
                         "diagnostic (28000 = the initial constitutive/equilibrium ratio found "
                         "there); re-check at production model scale before a long run.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
