"""Fixed-batch diagnostic for train_B3_mixed_stress.py's loss-term
balance, mirroring the exact methodology that found train_B3.py's own
OUTPUT_SCALE=0.02 (see that constant's docstring): train on the SAME
material/load sample every iteration (no new sampling noise each step),
watch whether the loss settles to a stable plateau, and -- specific to
the mixed formulation -- whether BOTH the constitutive and equilibrium
residuals are actually being driven down, not just whichever one
happens to start larger.

Run with --n_hidden/--n_layers at their defaults (production size) on
a GPU before trusting any real training run; the --tiny flags below
reproduce the CPU-only diagnostic already run once this session (see
PROJECT_STATUS.md, 2026-10-17 entry) at toy model size, which is what
first surfaced the ~28,000x constitutive/equilibrium imbalance that
train_B3_mixed_stress.py's own P_OUTPUT_SCALE/w_equilibrium defaults
are based on.

    python -m omar_pfem.diagnose_mixed_stress_B3_scale
    python -m omar_pfem.diagnose_mixed_stress_B3_scale --tiny   # fast CPU check
"""
import argparse

import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, mixed_stress_loss_B3
from omar_pfem.train_B3 import _apply_input_norm, sample_batch, set_input_norm
from omar_pfem.train_B3_mixed_stress import build_model, get_args


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true",
                    help="Toy model size for a fast CPU sanity check, not a real training run.")
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--p_output_scale", type=float, default=500.0)
    p.add_argument("--w_constitutive", type=float, default=1.0)
    p.add_argument("--w_equilibrium", type=float, default=28000.0)
    diag_args = p.parse_args(argv)

    torch.manual_seed(0)
    set_input_norm(None)
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Ntheta, Nr, Nz = DEFAULT_RESOLUTION
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes, device={device}")

    model_argv = []
    if diag_args.tiny:
        model_argv += ["--n_hidden", "32", "--n_layers", "2", "--n_heads", "2",
                        "--mlp_ratio", "1", "--slice_num", "16", "--ref", "4"]
    model_args = get_args(model_argv)
    model = build_model(model_args, device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=model_args.lr)

    E_node, nu_node, phi = sample_batch(diag_args.batch_size, Ntheta, Nr, Nz, seed_offset=0,
                                         dtype=dtype, device=device)
    xyz = geom["nodes"]
    xyz_batch = xyz.unsqueeze(0).expand(diag_args.batch_size, -1, -1)
    fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
    fun_material = _apply_input_norm(fun_material)

    for it in range(1, diag_args.n_steps + 1):
        raw = model(xyz_batch, fun_material)
        u_net_raw = raw[..., :3]
        p_net_raw = raw[..., 3:] * diag_args.p_output_scale
        loss, parts = mixed_stress_loss_B3(
            u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
            w_constitutive=diag_args.w_constitutive, w_equilibrium=diag_args.w_equilibrium)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % max(1, diag_args.n_steps // 10) == 0 or it == 1:
            print(f"[{it}/{diag_args.n_steps}] loss={loss.item():.3f} "
                  f"constitutive={parts['constitutive']:.3f} equilibrium={parts['equilibrium']:.6f}")

    print("\nCheck: did the loss settle to a stable plateau (no divergence)? "
          "Did BOTH constitutive and equilibrium actually decrease (not just the larger one)? "
          "If equilibrium is still being ignored, increase --w_equilibrium further and re-run.")


if __name__ == "__main__":
    main()
