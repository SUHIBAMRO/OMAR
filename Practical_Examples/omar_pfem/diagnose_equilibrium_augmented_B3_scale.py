"""Fixed-batch diagnostic for train_B3_equilibrium_augmented.py's loss
balance, same methodology as diagnose_mixed_stress_B3_scale.py and the
OUTPUT_SCALE story this project already established: train on the SAME
material/load sample every iteration, watch whether the loss settles to
a stable plateau, and whether the weak-form residual actually responds
to the chosen w_equilibrium rather than being swamped by (or swamping)
the energy term.

    python -m omar_pfem.diagnose_equilibrium_augmented_B3_scale
    python -m omar_pfem.diagnose_equilibrium_augmented_B3_scale --tiny   # fast CPU check
"""
import argparse

import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.equilibrium_augmented_B3 import weak_form_equilibrium_residual
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed
from omar_pfem.train_B3 import (
    _apply_input_norm, apply_dirichlet_b3, build_model, sample_batch, set_input_norm,
    total_potential_energy_B3,
)
from omar_pfem.train_B3_equilibrium_augmented import get_eq_args


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--w_equilibrium", type=float, default=1.0)
    p.add_argument("--n_virtual", type=int, default=4)
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
    model_args = get_eq_args(model_argv)
    model = build_model(model_args, device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=model_args.lr)

    E_node, nu_node, phi = sample_batch(diag_args.batch_size, Ntheta, Nr, Nz, seed_offset=0,
                                         dtype=dtype, device=device)
    xyz = geom["nodes"]
    xyz_batch = xyz.unsqueeze(0).expand(diag_args.batch_size, -1, -1)
    fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
    fun_material = _apply_input_norm(fun_material)

    for it in range(1, diag_args.n_steps + 1):
        u_net = model(xyz_batch, fun_material)
        u = apply_dirichlet_b3(u_net, geom, phi)
        U = total_potential_energy_B3(u, E_node, nu_node, geom)
        eq_resid = weak_form_equilibrium_residual(u, E_node, nu_node, geom,
                                                   n_virtual=diag_args.n_virtual)
        loss = U.mean() + diag_args.w_equilibrium * eq_resid
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % max(1, diag_args.n_steps // 10) == 0 or it == 1:
            print(f"[{it}/{diag_args.n_steps}] loss={loss.item():.4f} "
                  f"energy={U.mean().item():.4f} weak_form_residual={eq_resid.item():.4f}")

    print("\nCheck: stable (no divergence)? Does BOTH energy and weak_form_residual actually "
          "move, not just one dominating? If residual is ignored, increase --w_equilibrium.")


if __name__ == "__main__":
    main()
