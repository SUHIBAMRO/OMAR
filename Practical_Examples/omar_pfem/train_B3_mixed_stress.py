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

STATUS (updated 2026-10-21, after the first REAL 50,000-iteration run):
the two underlying operators (constitutive, equilibrium) and the loss
function's own plumbing are verified on CPU to 1e-10 or better
(omar_pfem/test_mixed_stress_B3.py, all 5 checks passing). The first
real training run completed (PROJECT_STATUS.md has the full account)
but the resulting checkpoint was a genuine failure -- combined
displacement/energy/reaction errors against the held-out set were
roughly 100x worse than the DEM baseline (1.88%/0.3%/3.0%), and the
"constitutive" loss term showed no clear downward trend across the
whole run. Two things identified since:

1. P_OUTPUT_SCALE was a hand-picked guess (500.0), never rigorously
   checked the way train_B3.py's own OUTPUT_SCALE=0.02 was (that one
   was found only after measuring the untrained network's real raw
   output magnitude against the true displacement scale from real FEM
   data -- see that constant's own docstring). `calibrate_p_output_
   scale.py` does the analogous measurement for P: a "realistic true P"
   (from real sampled E/nu/phi run through the same Dirichlet-ansatz
   construction, raw network contribution zeroed) has RMS ~74, while
   the untrained network's own raw stress-channel output has RMS ~0.30
   at production model size -- giving a DERIVED scale of ~246.6, found
   to be a real, ~2x over-estimate versus the 500.0 that was actually
   used (500/246.6 ~ 2.03x). Updated the default below to 250.0 (the
   derived value is 246.6; rounded). **This is a real, measured
   correction, but almost certainly not the WHOLE explanation** -- it
   is a much smaller mismatch than OUTPUT_SCALE's original 5-50x, and
   the observed per-component constitutive residual during the failed
   run (~65) was already comparable in magnitude to the true RMS P
   itself (~74), i.e. close to 100% relative error throughout training,
   which a mere 2x initial-scale fix would not fully resolve on its
   own. Treat 250.0 as a real improvement worth re-testing, not a
   confirmed fix -- the next real run's own loss trajectory is the
   actual test, the same way OUTPUT_SCALE's correctness was ultimately
   confirmed by a real run settling to a stable plateau, not by the
   calibration arithmetic alone.
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

UPDATE 2026-10-21 (second real run, p_output_scale=250.0): interrupted
by Omar at iteration 14,000/50,000 after a full re-audit of this
module's own code found no further implementation bugs (loss weights,
P-index ordering, gradient flow from u into the constitutive residual
via F -- all re-checked, all correct). The `constitutive` term
oscillated between ~27,000 and ~103,000 with NO visible downward trend
across all 14,000 iterations -- the identical qualitative signature the
first (500.0-scale) run showed across its full 50,000 iterations.
`equilibrium` again converges to ~0 within the first 500 iterations and
stays there. Conclusion: the 500->250 scale correction was real (see
calibrate_p_output_scale.py) but made NO visible difference to the
actual training dynamics -- the scale was never the dominant problem.

New working hypothesis, not yet disproven: jointly learning u AND P
from scratch, under a loss with no energy term at all, is a
substantially harder optimization landscape than train_B3.py's own
single, well-behaved energy objective -- and once `equilibrium` is
driven to ~0 (trivially achievable by outputting a sufficiently smooth P
field, since many such fields have near-zero divergence), essentially
ALL subsequent gradient signal is about `constitutive` alone, yet it
still does not consistently decrease even with 14,000+ iterations
focused on nothing else. PRETRAIN_ITERS below implements the resulting
curriculum: spend the first `--pretrain_iters` iterations training u
ALONE via train_B3.py's own already-proven total_potential_energy_B3
objective (p_net_raw computed but excluded from the loss, so it gets no
gradient and stays at its random init -- harmless, since stage 2
retrains it from there anyway), then switch to the full mixed_stress_
loss_B3 objective for the remainder, so the constitutive/equilibrium
optimization starts from an already-good displacement field instead of
a random one. Default 0 (old behavior, single-stage) for backward
compatibility with every existing caller/checkpoint.
"""
import argparse
import os
import time

import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, mixed_stress_loss_B3
from omar_pfem.model_dict import get_model
from omar_pfem.train_B3 import (
    _apply_input_norm, apply_dirichlet_b3, compute_input_norm_b3, sample_batch, set_input_norm,
    total_potential_energy_B3,
)

P_OUTPUT_SCALE = 250.0
"""Derived via calibrate_p_output_scale.py (~246.6, rounded) after the
first real 500.0-scale run failed -- see this module's own docstring,
point 1. A real, measured correction (2x smaller), not yet confirmed to
fully fix the failure on its own; the next real run is the actual test."""


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

    xyz = geom["nodes"]
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args, device).to(dtype)

    # Resume from the latest checkpoint already in output_dir, if any --
    # added after a real Colab disconnection mid-run (2026-10-21) lost no
    # progress only because the interruption happened to land exactly on a
    # checkpoint boundary; without this, any other interruption point would
    # have silently discarded real GPU time on the next re-run. Loads INTO
    # the already-constructed model (not a new one) so the optimizer built
    # right after this block starts from the resumed weights, not a fresh
    # model's parameter objects. Optimizer state itself is not saved/
    # restored (matching this project's own established checkpoint
    # convention everywhere else -- train_B3.py, train_B3_equilibrium_
    # augmented.py -- none of them save optimizer state either), so Adam's
    # own momentum restarts fresh; still far better than re-initializing
    # the model weights themselves from scratch.
    start_it = 1
    existing = [f for f in os.listdir(args.output_dir)
                if f.startswith("checkpoint_") and f.endswith(".pt")]
    if existing:
        latest = max(existing, key=lambda f: int(f[len("checkpoint_"):-len(".pt")]))
        ckpt = torch.load(os.path.join(args.output_dir, latest), map_location=device)
        # Real near-miss, 2026-10-21: P_OUTPUT_SCALE was corrected (500.0 ->
        # 250.0) after the first run failed. Blindly resuming an old
        # checkpoint under a NEW scale would silently reinterpret its
        # already-learned stress-channel weights at the wrong magnitude,
        # mixing two inconsistent trainings together with no warning --
        # checked explicitly so this can never happen silently again.
        ckpt_scale = ckpt.get("args", {}).get("p_output_scale")
        if ckpt_scale is not None and ckpt_scale != args.p_output_scale:
            raise ValueError(
                f"{latest} was trained with p_output_scale={ckpt_scale}, but this run "
                f"is configured with p_output_scale={args.p_output_scale} -- resuming "
                f"would silently mix two inconsistent trainings. Use a different "
                f"--output_dir for the new scale, or pass the matching p_output_scale.")
        model.load_state_dict(ckpt["model_state"])
        start_it = ckpt["iter"] + 1
        print(f"[resume] found {latest}, resuming from iteration {start_it} "
              f"(optimizer state not saved/restored -- Adam's own momentum restarts fresh)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    if start_it > args.n_iters:
        print(f"[resume] already at iteration {start_it - 1} >= n_iters={args.n_iters}, nothing to do")
        return model

    t0 = time.time()
    for it in range(start_it, args.n_iters + 1):
        E_node, nu_node, phi = sample_batch(args.batch_size, Ntheta, Nr, Nz,
                                             seed_offset=it * args.batch_size,
                                             dtype=dtype, device=device)
        fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
        fun_material = _apply_input_norm(fun_material)

        xyz_batch = xyz.unsqueeze(0).expand(args.batch_size, -1, -1)
        raw = model(xyz_batch, fun_material)  # (Batch, N, 12)
        u_net_raw = raw[..., :3]

        pretraining = it <= args.pretrain_iters
        if pretraining:
            # Stage 1: u alone, via the same proven energy objective
            # train_B3.py uses. p_net_raw is deliberately NOT computed
            # from raw here -- it would get no gradient anyway (unused
            # in this loss) and stage 2 retrains it from scratch, so
            # skipping it is a real (if small) compute saving, not a
            # correctness shortcut.
            u = apply_dirichlet_b3(u_net_raw, geom, phi)
            loss = total_potential_energy_B3(u, E_node, nu_node, geom).mean()
            parts = {"constitutive": float("nan"), "equilibrium": float("nan")}
        else:
            p_net_raw = raw[..., 3:] * args.p_output_scale
            loss, parts = mixed_stress_loss_B3(
                u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
                w_constitutive=args.w_constitutive, w_equilibrium=args.w_equilibrium)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.log_every == 0 or it == start_it or it == args.pretrain_iters + 1:
            elapsed = time.time() - t0
            if pretraining:
                print(f"[{it}/{args.n_iters}] [stage1:pretrain-u] loss(mean U)={loss.item():.6f} "
                      f"elapsed={elapsed:.1f}s")
            else:
                print(f"[{it}/{args.n_iters}] [stage2:mixed-stress] loss={loss.item():.6f} "
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
    p.add_argument("--pretrain_iters", type=int, default=0,
                    help="Two-stage curriculum, added 2026-10-21 after the second real run "
                         "(p_output_scale=250.0) showed the identical non-converging "
                         "'constitutive' pattern as the first (500.0-scale) run even after "
                         "14,000 iterations -- see this module's own docstring, 'UPDATE "
                         "2026-10-21'. For iterations 1..pretrain_iters, train u ALONE via "
                         "train_B3.py's own proven total_potential_energy_B3 objective (p_net "
                         "gets no gradient and stays at its random init); for every iteration "
                         "after that, switch to the normal mixed_stress_loss_B3 objective, "
                         "continuing from the same model weights. Default 0 reproduces the "
                         "old single-stage behavior exactly.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
