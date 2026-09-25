"""Deep Energy Method training of a 3D Transolver operator for B3 (the
sharper-groove rocking rubber-mount bushing), per the scope decided
2026-09-25 (Prof. Rabczuk confirmed "VINO" means this project's own
Transolver -- no architecture or training-paradigm change needed).

Mirrors train_B1.py/train_B2.py's own Deep Energy Method training in
spirit -- minimizes the discretized internal strain energy of the
network's own predicted displacement field, with NO labeled FEM
displacement data anywhere in the loss (data_generate_B3_dataset.py
solves real FEM samples only for held-out validation later, matching
B1/B2's own reference-dataset-for-validation-only convention, NOT as
training labels).

Real difference from B1/B2, not a copy-paste: B3's loading is entirely
prescribed-DISPLACEMENT (a rigid core rotation), not prescribed-FORCE
(B2's pressure), so there is no external-work term W in the loss
(Pi = U, not U - W). All three of B3's Dirichlet conditions (inner:
exact rigid rotation; outer: exactly fixed; symmetry: u_y=0 at
theta=0,pi) are enforced EXACTLY by construction on the network's raw
output (see `apply_dirichlet_b3`), not by a soft penalty term -- the
network only ever has to get the INTERIOR field right.

Deformation-gradient/energy assembly reuses torch-fem's own
Solid.eval_shape_functions (the shape-function-gradient/Jacobian
machinery already used and trusted throughout this project's B3 FEM
solver code) for the fixed reference geometry, computed ONCE since the
mesh does not vary across samples -- confirmed directly against a real
small mesh in this session (not assumed): eval_shape_functions returns
N (8_gauss, 8_node), B (8_gauss, n_elem, 3_dof, 8_node), detJ
(8_gauss, n_elem). This module's own small, batched Neo-Hookean
energy-density function then integrates it -- the exact formula
torchfem_comparison.neo_hookean_psi_3d uses (mu/2*(I1-3-2lnJ) +
lam/2*lnJ^2), vectorized over batch*element via torch.linalg.slogdet's
native batching, rather than that function's own per-element/vmap form.
No custom shape-function or Jacobian code was written from scratch.
"""
import argparse
import os
import time

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import generate_grid_hex8_bushing, boundary_node_sets
from omar_pfem.data.data_generate_B3_dataset import (
    GROOVE_DEPTH, GROOVE_HALF_WIDTH, R_GRADING, DEFAULT_RESOLUTION,
    sample_material_and_load,
)
from omar_pfem.data.mesh_convergence_B3 import R_IN0, R_OUT, LZ
from omar_pfem.model_dict import get_model


def build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=torch.float64):
    """Everything that depends only on the mesh (fixed across every
    sample): nodes, elements, boundary masks, per-node parametric (t,
    theta) coordinates (needed for the Dirichlet-BC construction), and
    the reference-configuration shape-function gradients/Jacobians
    (needed for the energy integral) -- computed ONCE and reused for
    every training batch."""
    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=R_GRADING)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)

    # Per-node (theta, t) parametric coordinates, matching
    # generate_grid_hex8_bushing's own node ordering exactly (verified
    # earlier in this session against the same convention
    # data_generate_B3_dataset.py relies on): node index =
    # k*(Ntheta*Nr) + j*Nr + i, with theta = thetas[j], t = ts[i].
    thetas = np.linspace(0.0, np.pi, Ntheta)
    ts = np.linspace(0.0, 1.0, Nr) ** R_GRADING
    j_idx = (np.arange(Ntheta * Nr * Nz) // Nr) % Ntheta
    i_idx = np.arange(Ntheta * Nr * Nz) % Nr
    theta_node = thetas[j_idx]
    t_node = ts[i_idx]

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        from torchfem import Solid
        from torchfem.materials import Hyperelastic3D
        from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

        nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
        elements_t = torch.tensor(elements, dtype=torch.long, device=device)
        # Dummy material: only used to construct a Solid so its own
        # etype/eval_shape_functions can be reused -- these do not depend
        # on the material at all, only on nodes/elements.
        dummy_params = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=dummy_params)
        with torch.device(device):
            model = Solid(nodes_t, elements_t, material)
        N, B, detJ = model.eval_shape_functions(model.etype.ipoints)
        iweights = model.etype.iweights.to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(old_default_dtype)

    return {
        "nodes": nodes_t, "elements": elements_t,
        "inner": torch.tensor(inner, device=device), "outer": torch.tensor(outer, device=device),
        "sym": torch.tensor(sym, device=device),
        "theta_node": torch.tensor(theta_node, dtype=dtype, device=device),
        "t_node": torch.tensor(t_node, dtype=dtype, device=device),
        "B": B, "detJ": detJ, "iweights": iweights,
        "n_nodes": nodes.shape[0], "n_elements": elements.shape[0],
    }


def rigid_rotation_displacement_torch(x0, z0, phi):
    """Batched torch version of data_generate_B3.rigid_rotation_displacement
    -- x0,z0: (N,) node reference coords (shared across the batch); phi:
    (B,) one rocking angle per sample. Returns (ux,uy,uz), each (B,N),
    with uy identically 0 (a rotation about the y-axis never touches y --
    the exact same fact the FEM solver's own BC construction relies on)."""
    dz = z0[None, :] - LZ / 2.0
    c = torch.cos(phi)[:, None]
    s = torch.sin(phi)[:, None]
    ux = x0[None, :] * (c - 1.0) + dz * s
    uz = -x0[None, :] * s + dz * (c - 1.0)
    uy = torch.zeros_like(ux)
    return ux, uy, uz


def apply_dirichlet_b3(u_net, geom, phi):
    """Hard, exact enforcement of all three B3 Dirichlet conditions on the
    network's raw per-node output u_net (B,N,3):

    - outer (t=1): u = 0 exactly, via the radial ramp t*(1-t) which
      vanishes there.
    - inner (t=0): u = the real rigid-rotation displacement at that
      node's own reference (x,z) -- exact because the ramp also vanishes
      at t=0, leaving only the additive particular term, which is
      already the exact right value there by construction.
    - symmetry (theta=0 or pi): u_y = 0 exactly, via an extra sin(theta)
      factor on the y-component only (theta=0,pi are exactly the domain's
      own two ends, so sin(theta)=0 there and nowhere else in [0,pi]).

    No soft penalty term is needed anywhere -- the network only has to
    get the free interior field right."""
    t = geom["t_node"]
    theta = geom["theta_node"]
    ramp = (t * (1.0 - t))[None, :]

    x0, z0 = geom["nodes"][:, 0], geom["nodes"][:, 2]
    ux_d, uy_d, uz_d = rigid_rotation_displacement_torch(x0, z0, phi)
    particular = (1.0 - t)[None, :]

    ux = ux_d * particular + ramp * u_net[:, :, 0]
    uy = ramp * torch.sin(theta)[None, :] * u_net[:, :, 1]
    uz = uz_d * particular + ramp * u_net[:, :, 2]
    return torch.stack([ux, uy, uz], dim=2)


def neo_hookean_energy_density_batched(F, mu, lam):
    """Batched Neo-Hookean strain-energy density -- the exact formula
    torchfem_comparison.neo_hookean_psi_3d uses (mu/2*(I1-3-2lnJ) +
    lam/2*lnJ^2), vectorized directly over F's leading batch dims via
    torch.linalg.slogdet's own native batching, instead of that
    function's per-element/vmap form. F: (...,3,3); mu,lam: (...)."""
    _sign, lnJ = torch.linalg.slogdet(F)
    I1 = torch.sum(F ** 2, dim=(-2, -1))
    return (mu / 2.0) * (I1 - 3.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)


def total_potential_energy_B3(u, E_node, nu_node, geom):
    """Pi = U (no external-work term: B3's loading is entirely
    prescribed-displacement, not prescribed-force). u: (B,N,3) already
    satisfies every Dirichlet condition exactly (apply_dirichlet_b3's
    output). Returns U, shape (B,)."""
    elements = geom["elements"]
    B_op, detJ, iweights = geom["B"], geom["detJ"], geom["iweights"]
    n_elem = geom["n_elements"]
    Batch = u.shape[0]

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, 3_dof, 8_node)
    E_elem = E_node[:, elements].mean(dim=2)    # (Batch, n_elem)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))

    U = torch.zeros(Batch, device=u.device, dtype=u.dtype)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    for g in range(B_op.shape[0]):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])  # (Batch, n_elem, 3, 3)
        F = eye + H
        psi = neo_hookean_energy_density_batched(F, mu_elem, lam_elem)  # (Batch, n_elem)
        U = U + torch.sum(psi * detJ[g][None, :] * iweights[g], dim=1)
    return U


def build_model(args, device):
    return get_model(args).Model(
        space_dim=3,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=3,  # E, nu, phi (phi broadcast per node -- see sample_batch)
        out_dim=3,
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=args.unified_pos,
    ).to(device)


def sample_batch(batch_size, Ntheta, Nr, Nz, seed_offset, dtype, device,
                  E_mean=1000.0, E_std=200.0, nu_mean=0.45, nu_std=0.02, nu_clip=(0.40, 0.49),
                  phi_mean=0.05, phi_std=0.02, phi_clip=(0.01, 0.15)):
    """Live material/load sampling for training -- reuses the exact same
    3D-GRF/scalar-phi sampler data_generate_B3_dataset.py uses for the
    (separate, validation-only) FEM dataset, so training and the held-out
    FEM reference dataset are drawn from the identical distribution.
    (Ntheta,Nr,Nz) must match the geometry training is actually using --
    passed explicitly rather than assumed, so a caller testing at a
    different (smaller) resolution than DEFAULT_RESOLUTION cannot end up
    with mismatched field/mesh sizes silently."""
    E_list, nu_list, phi_list = [], [], []
    for b in range(batch_size):
        seed = seed_offset + b
        E_node, nu_node, phi, _, _ = sample_material_and_load(
            Ntheta, Nr, Nz, seed,
            E_mean=E_mean, E_std=E_std, nu_mean=nu_mean, nu_std=nu_std, nu_clip=nu_clip,
            phi_mean=phi_mean, phi_std=phi_std, phi_clip=phi_clip)
        E_list.append(E_node)
        nu_list.append(nu_node)
        phi_list.append(phi)
    E_node = torch.tensor(np.stack(E_list), dtype=dtype, device=device)
    nu_node = torch.tensor(np.stack(nu_list), dtype=dtype, device=device)
    phi = torch.tensor(phi_list, dtype=dtype, device=device)
    return E_node, nu_node, phi


def train(args, device, dtype=torch.float64, resolution=None):
    Ntheta, Nr, Nz = resolution or DEFAULT_RESOLUTION
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"Fixed geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    model = build_model(args, device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    xyz = geom["nodes"]  # (N,3), reference coordinates, shared across the batch

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    for it in range(1, args.n_iters + 1):
        E_node, nu_node, phi = sample_batch(args.batch_size, Ntheta, Nr, Nz,
                                             seed_offset=it * args.batch_size,
                                             dtype=dtype, device=device)
        fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)

        xyz_batch = xyz.unsqueeze(0).expand(args.batch_size, -1, -1)
        u_net = model(xyz_batch, fun_material)
        u = apply_dirichlet_b3(u_net, geom, phi)

        U = total_potential_energy_B3(u, E_node, nu_node, geom)
        loss = U.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"[{it}/{args.n_iters}] loss(mean U)={loss.item():.6f} "
                  f"elapsed={elapsed:.1f}s")

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
    p.add_argument("--output_dir", type=str, default="b3_training_output")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
