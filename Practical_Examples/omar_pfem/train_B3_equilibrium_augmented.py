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

STATUS (updated 2026-10-22): the first real 50,000-iteration run (at
the original w_equilibrium=1.0 default) was a genuine failure --
roughly 100x worse than the DEM baseline on every QoI (energy 181.8%,
reaction 216.9%, region Cauchy field 1013.8% vs. the baseline's
0.3%/3.0%). Root-caused via a CPU-only measurement, not a guess: at
PRODUCTION model size, with REAL sampled material/load batches, the raw
(unweighted) weak-form residual is roughly 6.5-10x LARGER in magnitude
than the energy term at initialization (measured across 3 seeds: energy
2.6-5.2, residual 22-51, ratio energy/residual = 0.10-0.15) -- i.e.
w_equilibrium=1.0 was never actually "balanced," it was only ever
sanity-checked at a smaller/undertrained scale. With w=1.0, the
equilibrium term (which is ALSO inherently noisy -- resampled with
fresh random, unseeded virtual test fields every iteration; measured
swinging +/-40% between back-to-back evaluations of the SAME u at
n_virtual=8) completely dominated the total loss, so the optimizer
mostly chased a noisy, overweighted target instead of the energy term
that actually drives a good displacement field, explaining the ~100x
failure without needing to invoke the "unseeded resampling" noise as a
separate, additional villain -- the weighting alone was already a 6.5-
10x miscalibration.

W_EQUILIBRIUM's default below was corrected from 1.0 to 0.12 (the
measured ratio, rounded) following this measurement. Verified on a CPU
fixed-batch diagnostic (`diagnose_equilibrium_augmented_B3_scale.py
--tiny --w_equilibrium 0.12`, 600 steps): `energy` settles to a stable
plateau (~2.30) that is NEARLY IDENTICAL to a pure-energy-only
(w_equilibrium=0) reference run on the exact same fixed batch (~2.25-
2.30) -- i.e. the corrected weight no longer visibly perturbs the
energy term's own healthy convergence, unlike w=1.0's complete
domination. `weak_form_residual` itself remains noisy run-to-run (still
the same unseeded-resampling behavior, now not investigated further
since it no longer appears to be the dominant problem) but this no
longer destabilizes the term that actually matters. This is the same
kind of real, measured calibration this project has now done four times
(train_B3.py's OUTPUT_SCALE, mixed_stress's P_OUTPUT_SCALE and
w_equilibrium, now this) -- the fixed-batch CPU check is a necessary
sanity gate, not a guarantee; the real test is a full GPU run.
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
    p.add_argument("--w_equilibrium", type=float, default=0.12,
                   help="weight on the weak-form equilibrium residual, relative to the "
                        "energy loss. Corrected 2026-10-22 from 1.0 (which caused a real, "
                        "measured ~100x-worse-than-baseline training failure) to 0.12, the "
                        "measured production-scale ratio of energy/residual magnitude "
                        "(0.10-0.15 across 3 seeds) -- see this module's own docstring, "
                        "'STATUS (updated 2026-10-22)'.")
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
