"""Equilibrium-augmented training for B3 (Professor Rabczuk's items 2
and 3, 2026-10-16 email -- see equilibrium_augmented_B3.py's own module
docstring for why these two collapse to the same weak-form-residual
mechanism for a u-only network). Mirrors train_B3.py's own training
loop structure as closely as possible -- SAME model (out_dim=3, no
separate stress channel), SAME Dirichlet ansatz, SAME DEM energy term
-- adding only the weak-form equilibrium residual as an extra loss
term. Kept as a separate file, not an edit to train_B3.py, for the same
reason as every other item on this follow-up-paper roadmap: train_B3.py
produced the real, already-published B3 numbers in the submitted paper
and must not be touched.

Usage:
  python -m omar_pfem.train_B3_equilibrium_augmented \
      --n_iters 50000 --batch_size 8 --output_dir .../b3_equilibrium_augmented
"""
import json
import os
import time

import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.equilibrium_augmented_B3 import weak_form_equilibrium_residual
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed
from omar_pfem.train_B3 import (
    _apply_input_norm, apply_dirichlet_b3, build_model, compute_input_norm_b3,
    sample_batch, set_input_norm, total_potential_energy_B3,
)


def train(args, device, dtype=torch.float64, resolution=None):
    Ntheta, Nr, Nz = resolution or DEFAULT_RESOLUTION
    # build_fixed_geometry_mixed, not train_B3's own build_fixed_geometry --
    # identical mesh/BCs/energy machinery, this one also keeps N (shape
    # function values), unused by this script directly but kept for
    # consistency with the rest of the follow-up-paper code; no behavior
    # difference for anything this script actually computes.
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"Fixed geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    model = build_model(args, device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    xyz = geom["nodes"]
    os.makedirs(args.output_dir, exist_ok=True)

    norm_path = os.path.join(args.output_dir, "input_norm.json")
    if int(getattr(args, "normalize_inputs", 0)):
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
        u_net = model(xyz_batch, fun_material)
        u = apply_dirichlet_b3(u_net, geom, phi)

        U = total_potential_energy_B3(u, E_node, nu_node, geom)
        eq_resid = weak_form_equilibrium_residual(u, E_node, nu_node, geom,
                                                   n_virtual=args.n_virtual)
        loss = U.mean() + args.w_equilibrium * eq_resid

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"[{it}/{args.n_iters}] loss={loss.item():.6f} "
                  f"energy={U.mean().item():.6f} weak_form_residual={eq_resid.item():.6f} "
                  f"elapsed={elapsed:.1f}s")

        if it % args.ckpt_every == 0 or it == args.n_iters:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{it}.pt")
            torch.save({"model_state": model.state_dict(), "iter": it, "args": vars(args)}, ckpt_path)

    return model


def get_eq_args(argv=None):
    """Same argument set as train_B3.get_args() (model architecture,
    optimizer, logging, normalization -- unchanged defaults, so this
    script's CLI matches train_B3.py's exactly for everything they
    share), plus the two new equilibrium-specific flags."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_iters", type=int, default=50000)
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
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="b3_equilibrium_augmented_output")
    p.add_argument("--normalize_inputs", type=int, default=0)
    p.add_argument("--w_equilibrium", type=float, default=1.0,
                   help="weight on the weak-form equilibrium residual, relative to the "
                        "energy loss. NOT yet confirmed at production scale -- watch the "
                        "printed energy vs. weak_form_residual magnitudes early in a real "
                        "run the same way mixed-stress's own w_equilibrium=28000 was "
                        "calibrated via a fixed-batch diagnostic before trusting this "
                        "default.")
    p.add_argument("--n_virtual", type=int, default=8,
                   help="number of random virtual test fields per iteration -- a fixed-batch "
                        "diagnostic (diagnose_equilibrium_augmented_B3_scale.py) showed the "
                        "residual is noisy run-to-run at n_virtual=4 purely from the random "
                        "test-function draw, not a sign of an actual problem; 8 trades a "
                        "little more compute for a steadier per-iteration estimate.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = get_eq_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
