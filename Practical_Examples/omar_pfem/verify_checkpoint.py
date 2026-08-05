"""Independent sanity check on a high_dof_convergence_study.py checkpoint
file -- does NOT trust the stored 'final residual' number from the solve
that produced it. Instead, it rebuilds the mesh/material/tangent-residual
function fresh from the current code and RE-EVALUATES the residual at the
saved displacement field, at the load level the checkpoint claims to have
reached. If that independently-recomputed residual is small, the saved
state is a genuine equilibrium solution, not a fluke or a corrupted file.

Also checks for the cheap, obvious failure modes: NaN/Inf in the solution,
and a displacement magnitude that is a sane fraction of the domain size
(not zero, not absurdly large) -- either of those would be visible even
without any FEM re-evaluation.

Usage:
  python -m omar_pfem.verify_checkpoint \
      --geometry B1 --material neo_hookean --order Q4 --fine_N 2236 \
      --checkpoint /content/drive/MyDrive/pfem_ckpt/fine_B1_neo_hookean_Q4_N2236.pt
"""
import argparse

import numpy as np
import torch
from torch.func import grad as torch_grad

from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs
from omar_pfem.matrix_free_solver import _make_energy_fn


def main():
    parser = argparse.ArgumentParser("Independently verify a checkpoint's saved state is a real equilibrium")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--order", type=str, required=True, choices=["Q4", "Q9"])
    parser.add_argument("--fine_N", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nsteps", type=int, default=10,
                         help="Must match the --resolutions run's own load-step count (default 10, "
                              "matching high_dof_convergence_study.py's own hardcoded nsteps=10).")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    u_free = ckpt["u_free"].to(device=device, dtype=dtype)
    stats = ckpt["stats"]
    next_step = ckpt["next_step"]
    steps_done = next_step - 1
    print(f"  Checkpoint claims: {steps_done}/{args.nsteps} load steps done, "
          f"stats={stats}")

    # ---- Cheap checks first: no NaN/Inf, sane magnitude ----
    n_nan = int(torch.isnan(u_free).sum())
    n_inf = int(torch.isinf(u_free).sum())
    print(f"\nNaN count: {n_nan}, Inf count: {n_inf}  -> {'FAIL' if n_nan or n_inf else 'OK'}")

    max_abs_disp = float(torch.max(torch.abs(u_free)))
    domain_scale = 1.0 if args.geometry == "B1" else 1.0  # both B1 (unit square) and B2 (R_out-R_in=1) have O(1) scale
    print(f"Max |displacement| = {max_abs_disp:.4e} (domain scale ~{domain_scale}) "
          f"-> {'SUSPICIOUS (zero or huge)' if max_abs_disp < 1e-10 or max_abs_disp > 10 * domain_scale else 'OK'}")

    if steps_done <= 0:
        print("\nNo completed load step yet -- nothing further to verify (residual check needs a "
              "completed step's load level).")
        return

    # ---- Independent residual recomputation at the claimed load level ----
    print(f"\nRebuilding mesh/material fresh from current code (order={args.order}, "
          f"N={args.fine_N}, material={args.material})...")
    nodes, elements, free_dofs_np, fext_full, elem_params_np = build_mesh_and_bcs(
        args.geometry, args.order, args.fine_N, args.material, device, dtype)

    xy_t = torch.tensor(nodes, dtype=dtype, device=device)
    quad_t = torch.tensor(elements, dtype=torch.long, device=device)
    free_dofs_t = torch.tensor(free_dofs_np, dtype=torch.long, device=device)
    elem_params_t = tuple(torch.tensor(p, dtype=dtype, device=device) for p in elem_params_np)
    fext_free_t = torch.tensor(fext_full[free_dofs_np], dtype=dtype, device=device)

    n_nodes = nodes.shape[0]
    energy_fn = _make_energy_fn(xy_t, quad_t, n_nodes, args.material, args.order, dtype)
    residual_fn = torch_grad(energy_fn, argnums=0)

    alpha = steps_done / args.nsteps
    fext_at_step = alpha * fext_free_t
    print(f"Load level of last completed step: alpha={alpha:.3f} (step {steps_done}/{args.nsteps})")

    print("Recomputing residual R = dPi/du - fext at the saved u_free (this evaluates the energy "
          "gradient over the whole mesh once -- seconds, not a new solve)...")
    R = residual_fn(u_free, elem_params_t, free_dofs_t) - fext_at_step
    res_norm = float(torch.linalg.norm(R))
    res_norm_rel = res_norm / (float(torch.linalg.norm(fext_at_step)) + 1e-30)

    print(f"\n{'='*70}\nIndependently recomputed residual norm: {res_norm:.6e}")
    print(f"Relative to external load norm: {res_norm_rel:.6e}")
    verdict = "PASS (genuine equilibrium)" if res_norm_rel < 1e-4 else "FAIL (residual too large)"
    print(f"Verdict: {verdict}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
