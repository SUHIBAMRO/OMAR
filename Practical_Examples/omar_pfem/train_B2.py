"""
Training script for the B2 benchmark (quarter ring, R_in=1, R_out=2,
internal pressure on the inner arc, symmetry BCs on the two straight
edges) using the PFEM/Transolver pipeline -- mirrors train_B1.py's
structure (Q4 isoparametric shape functions, 2x2-Gauss "explicit
differentiation" energy computation, no-autodiff-for-spatial-derivatives,
the total-potential-energy Pi = U - W loss, the Transolver
architecture/training-loop), generalized where B2's geometry/BCs differ
from B1's:

  - Two symmetry Dirichlet constraints instead of one full fixation: at
    theta=0 (the y=0 edge) u_y=0 and at theta=pi/2 (the x=0 edge) u_x=0.
    Generalizes B1's single scalar ramp (xy[:,1]/Ly) into two independent
    per-component ramps, each based on the OTHER coordinate's proportional
    distance from its own zero edge: free_u = clamp(x/R_out, 0, 1) (zero
    exactly on the x=0 edge, nonzero on the y=0 edge since x>=R_in>0
    there) and free_v = clamp(y/R_out, 0, 1) (the mirror image).
  - The loaded boundary is the inner arc (r=R_in) instead of a flat top
    edge; the per-node force direction is each inner node's own radial
    direction (cos theta, sin theta) instead of a fixed (0,1) (see
    omar_pfem/data/convert_B2_quad.py).
  - Material dispatch (Neo-Hookean / Mooney-Rivlin / Arruda-Boyce) via the
    same omar_pfem/materials_torch.py registry as train_B1.py.
"""
import os
import re
import glob
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import random

from omar_pfem.model_dict import get_model
from omar_pfem.materials_torch import get_material_fns


def find_latest_checkpoint(out_dir):
    """Epoch-level resume -- see train_B1.py's find_latest_checkpoint for
    the rationale (a case at 10000 epochs can run for hours, so a mid-case
    disconnect shouldn't discard all of it)."""
    ckpts = glob.glob(os.path.join(out_dir, "model_epoch*.pt"))
    if not ckpts:
        return None, 0

    def _epoch_num(p):
        m = re.search(r"model_epoch(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=_epoch_num)
    return latest, _epoch_num(latest)


# ============================================================
# 0) Utility: quad -> tri (for plotting only)
# ============================================================
def quad_to_tri(quad):
    q = quad
    t1 = np.stack([q[:, 0], q[:, 1], q[:, 2]], axis=1)
    t2 = np.stack([q[:, 0], q[:, 2], q[:, 3]], axis=1)
    tri = np.concatenate([t1, t2], axis=0)
    return tri


# ============================================================
# 1) Q4 shape functions on [-1,1]^2 -- unchanged from PFEM/train_B1
# ============================================================
def shape_Q4_torch(xi, eta, device, dtype):
    N = 0.25 * torch.tensor([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ], device=device, dtype=dtype)

    dN_dxi = 0.25 * torch.tensor([
        [-(1 - eta), -(1 - xi)],
        [+(1 - eta), -(1 + xi)],
        [+(1 + eta), +(1 + xi)],
        [-(1 + eta), +(1 - xi)],
    ], device=device, dtype=dtype)
    return N, dN_dxi


# ============================================================
# 2) Hyperelastic energy on Q4 mesh (TL, 2x2 Gauss) -- geometry-agnostic,
#    identical to train_B1.py's version (the isoparametric element routine
#    only ever sees each element's own corner reference coordinates, curved
#    ring or flat square makes no difference here)
# ============================================================
def compute_hyperelastic_energy_Q4(xy, quad, uv, param_nodes, energy_density_fn, dtype=torch.float32):
    """Batched version -- see train_B1.py's compute_hyperelastic_energy_Q4
    for the full rationale. uv is (B,N,2), each entry of param_nodes is
    (B,N); reference-configuration quantities (J0, detJ0, dN_dX) depend
    only on the shared mesh, so they're computed once per Gauss point and
    reused across the batch."""
    device = xy.device
    B = uv.shape[0]
    Q = quad.shape[0]

    Xe = xy[quad]           # (Q,4,2) -- shared across the batch
    ue = uv[:, quad]        # (B,Q,4,2)

    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    U = torch.zeros(B, device=device, dtype=dtype)
    F_list = []

    for (xi, eta) in gps:
        xi_t = torch.tensor(float(xi), device=device, dtype=dtype)
        eta_t = torch.tensor(float(eta), device=device, dtype=dtype)
        N, dN_dxi = shape_Q4_torch(xi_t, eta_t, device, dtype)

        J0 = torch.einsum("qai,aj->qij", Xe, dN_dxi)

        detJ0 = J0[:, 0, 0] * J0[:, 1, 1] - J0[:, 0, 1] * J0[:, 1, 0]
        detJ0 = torch.clamp(detJ0, min=1e-12)

        invJ0 = torch.zeros_like(J0)
        invJ0[:, 0, 0] = J0[:, 1, 1] / detJ0
        invJ0[:, 1, 1] = J0[:, 0, 0] / detJ0
        invJ0[:, 0, 1] = -J0[:, 0, 1] / detJ0
        invJ0[:, 1, 0] = -J0[:, 1, 0] / detJ0

        dN_dX = torch.einsum("aj,qjk->qak", dN_dxi, invJ0)

        grad_u = torch.einsum("bqai,qaj->bqij", ue, dN_dX)

        I = torch.eye(2, device=device, dtype=dtype).reshape(1, 1, 2, 2).expand(B, Q, 2, 2)
        F = I + grad_u  # (B,Q,2,2)

        params_e = tuple(torch.mean(p[:, quad], dim=2) for p in param_nodes)  # each (B,Q)

        F_flat = F.reshape(B * Q, 2, 2)
        params_e_flat = tuple(p.reshape(B * Q) for p in params_e)
        psi = energy_density_fn(F_flat, *params_e_flat, dtype=dtype).reshape(B, Q)
        U = U + torch.sum(psi * detJ0[None, :] * w, dim=1)

        F_list.append(F.detach().reshape(B * Q, 2, 2))

    F_g = torch.cat(F_list, dim=0)
    return U, F_g


# ============================================================
# 3) Total potential energy Pi = U - W  (Q4 version, B2 ring geometry)
# ============================================================
def total_potential_energy_Q4_hyperelastic(
    xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
    model, E_nodes, nu_nodes, node_forces,
    use_soft_dirichlet=True,
    R_out=2.0,
    mode="plane_strain",
    dtype=torch.float32,
    fun_dim=4,
    material="neo_hookean",
):
    """Batched: E_nodes, nu_nodes are (B,N); node_forces is (B,N,2). All
    batch items share the same mesh (fixed per case), only material/force
    fields vary. Returns per-sample Pi, U, W of shape (B,) and uv (B,N,2)
    -- see train_B1.py's version of this function for the full rationale."""
    device = xy.device
    B = E_nodes.shape[0]
    N = xy.shape[0]

    xy_domain = xy.unsqueeze(0).expand(B, N, 2)  # (B,N,2)

    if fun_dim == 3:
        fun_material = torch.stack([
            E_nodes, nu_nodes, node_forces[:, :, 1],
        ], dim=2)
    else:
        fun_material = torch.stack([
            E_nodes, nu_nodes, node_forces[:, :, 0], node_forces[:, :, 1],
        ], dim=2)

    uv_raw = model(xy_domain, fun_material)  # (B,N,2)

    # ---- Symmetry Dirichlet: u_y=0 on theta=0 (y=0), u_x=0 on
    # theta=pi/2 (x=0). Each ramp vanishes exactly on its own zero edge and
    # is nonzero on the *other* symmetry edge (since x,y >= R_in > 0 there).
    if use_soft_dirichlet:
        free_u = (xy[:, 0] / R_out).clamp(0.0, 1.0)
        free_v = (xy[:, 1] / R_out).clamp(0.0, 1.0)
    else:
        free_u = torch.ones(N, device=device, dtype=dtype)
        free_v = torch.ones(N, device=device, dtype=dtype)
        if thetahalfpi_nodes.numel() > 0:
            free_u[thetahalfpi_nodes] = 0.0
        if theta0_nodes.numel() > 0:
            free_v[theta0_nodes] = 0.0
    uv = torch.stack([uv_raw[:, :, 0] * free_u[None, :], uv_raw[:, :, 1] * free_v[None, :]], dim=2)

    energy_density_fn, E_nu_to_params_fn = get_material_fns(material)
    if material == "neo_hookean":
        param_nodes = E_nu_to_params_fn(E_nodes, nu_nodes, mode=mode)
    else:
        param_nodes = E_nu_to_params_fn(E_nodes, nu_nodes)

    U, Fg = compute_hyperelastic_energy_Q4(xy, quad, uv, param_nodes, energy_density_fn, dtype=dtype)

    # No /len(inner_edges) here: node_forces is now the FEM-consistent
    # nodal force (see data_generate_B2.py's assemble_traction_inner_curved
    # and convert_B2_quad.py) -- the exact same Fext the ground-truth
    # solve used directly, unaveraged. Dividing by the edge count was a
    # holdover from train_B1.py that happened to compensate for the OLD,
    # unweighted "raw pressure x direction" B2 force being inflated by
    # ~edge-count-many x; with the consistent force it instead shrinks W
    # to near-zero relative to U, collapsing training toward the
    # near-undeformed (U-minimizing) solution -- see
    # check_boundary_force_consistency.py and the force_fixed ablation's
    # ~3x error regression vs OLD_BASELINE_pre_fix.
    W = torch.sum(node_forces * uv, dim=(1, 2))

    Pi = U - W
    return Pi, U.detach(), W.detach(), uv, Fg


@torch.no_grad()
def predict_displacement_Q4_only(
    xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
    model, E_nodes, nu_nodes, node_forces,
    use_soft_dirichlet=True,
    R_out=2.0,
    dtype=torch.float32,
    fun_dim=4,
):
    """Deployment-time inference: the network forward pass plus the
    symmetry ramps, with none of the Gauss-quadrature energy assembly of
    total_potential_energy_Q4_hyperelastic -- see train_B1.py's version
    for the full rationale. Used by benchmark_inference_latency_Q4."""
    device = xy.device
    B = E_nodes.shape[0]
    N = xy.shape[0]
    xy_domain = xy.unsqueeze(0).expand(B, N, 2)

    if fun_dim == 3:
        fun_material = torch.stack([E_nodes, nu_nodes, node_forces[:, :, 1]], dim=2)
    else:
        fun_material = torch.stack([E_nodes, nu_nodes, node_forces[:, :, 0], node_forces[:, :, 1]], dim=2)

    uv_raw = model(xy_domain, fun_material)

    if use_soft_dirichlet:
        free_u = (xy[:, 0] / R_out).clamp(0.0, 1.0)
        free_v = (xy[:, 1] / R_out).clamp(0.0, 1.0)
    else:
        free_u = torch.ones(N, device=device, dtype=dtype)
        free_v = torch.ones(N, device=device, dtype=dtype)
        if thetahalfpi_nodes.numel() > 0:
            free_u[thetahalfpi_nodes] = 0.0
        if theta0_nodes.numel() > 0:
            free_v[theta0_nodes] = 0.0
    return torch.stack([uv_raw[:, :, 0] * free_u[None, :], uv_raw[:, :, 1] * free_v[None, :]], dim=2)


def benchmark_inference_latency_Q4(samples, model, args, device, dtype, n_repeats=200, n_warmup=20):
    """Pure inference latency: one new sample at a time (batch_size=1),
    forward pass only, no backward pass, no optimizer step, no energy
    assembly -- see train_B1.py's version for the full rationale."""
    model.eval()
    s0 = samples[0]
    xy = torch.tensor(s0["xy"], device=device, dtype=dtype)
    quad = torch.tensor(s0["quad"], device=device, dtype=torch.long)
    inner_edges = torch.tensor(s0["inner_edges"], device=device, dtype=torch.long)
    theta0_nodes = torch.tensor(s0["theta0_nodes"], device=device, dtype=torch.long)
    thetahalfpi_nodes = torch.tensor(s0["thetahalfpi_nodes"], device=device, dtype=torch.long)
    E1 = torch.tensor(s0["E_node"], device=device, dtype=dtype).unsqueeze(0)
    nu1 = torch.tensor(s0["nu_node"], device=device, dtype=dtype).unsqueeze(0)
    f1 = torch.tensor(s0["node_forces"], device=device, dtype=dtype).unsqueeze(0)

    def _one_call():
        return predict_displacement_Q4_only(
            xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes, model, E1, nu1, f1,
            use_soft_dirichlet=args.use_soft_dirichlet, R_out=args.R_out,
            dtype=dtype, fun_dim=args.fun_dim,
        )

    for _ in range(n_warmup):
        _one_call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.time()
    for _ in range(n_repeats):
        _one_call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.time() - t0

    ms_per_sample = 1000.0 * elapsed_s / n_repeats
    return {
        "inference_ms_per_sample": ms_per_sample,
        "inference_n_repeats": n_repeats,
        "inference_batch_size": 1,
        "inference_device": device.type,
    }


def total_potential_energy_Q4_hyperelastic_single(
    xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
    model, E_nodes, nu_nodes, node_forces,
    use_soft_dirichlet=True,
    R_out=2.0,
    mode="plane_strain",
    dtype=torch.float32,
    fun_dim=4,
    material="neo_hookean",
):
    """Single-sample convenience wrapper (eval/visualization/dataset
    preview) -- see train_B1.py's version for the rationale."""
    Pi, U, W, uv, Fg = total_potential_energy_Q4_hyperelastic(
        xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
        model, E_nodes.unsqueeze(0), nu_nodes.unsqueeze(0), node_forces.unsqueeze(0),
        use_soft_dirichlet=use_soft_dirichlet, R_out=R_out, mode=mode, dtype=dtype,
        fun_dim=fun_dim, material=material,
    )
    return Pi[0], U[0], W[0], uv[0], Fg


# ============================================================
# 4) Visualization helpers (Q4)
# ============================================================
def plot_mesh_with_materials_and_forces_q4(samples, args, sid=0, tag="mesh_material_force_check"):
    os.makedirs(args.out_dir, exist_ok=True)

    s = samples[sid]
    xy = s["xy"]
    quad = s["quad"]
    theta0_nodes = s["theta0_nodes"]
    thetahalfpi_nodes = s["thetahalfpi_nodes"]
    inner_edges = s["inner_edges"]
    E_node = s.get("E_node", None)
    nu_node = s.get("nu_node", None)
    node_forces = s.get("node_forces", None)

    x = xy[:, 0]
    y = xy[:, 1]

    sym_mask = np.zeros(x.shape[0], dtype=bool)
    if theta0_nodes is not None and len(theta0_nodes) > 0:
        sym_mask[theta0_nodes] = True
    if thetahalfpi_nodes is not None and len(thetahalfpi_nodes) > 0:
        sym_mask[thetahalfpi_nodes] = True

    tri = quad_to_tri(quad)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    ax.triplot(x, y, tri, linewidth=0.3, color='gray', alpha=0.5)
    ax.scatter(x, y, s=6, c="k", alpha=0.7)
    if sym_mask.any():
        ax.scatter(x[sym_mask], y[sym_mask], s=18, c="lime", label="Symmetry BC")
    if inner_edges is not None and inner_edges.shape[0] > 0:
        for (i, j) in inner_edges:
            ax.plot([x[i], x[j]], [y[i], y[j]], "r-", linewidth=1.5)
        ax.plot([], [], "r-", linewidth=1.5, label="Pressure edges")
    ax.set_aspect("equal")
    ax.set_title(f"Q4 Ring Mesh - sid={sid}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="best")

    if E_node is not None:
        ax = axes[0, 1]
        sc = ax.scatter(x, y, c=E_node, s=10, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(f"E Distribution\nMean={E_node.mean():.1f}, Std={E_node.std():.1f}")
        plt.colorbar(sc, ax=ax)

    if node_forces is not None:
        ax = axes[0, 2]
        fm = np.sqrt(node_forces[:, 0]**2 + node_forces[:, 1]**2)
        mask = fm > 0
        if np.any(mask):
            sc = ax.scatter(x[mask], y[mask], c=fm[mask], s=30, cmap="hot", alpha=0.8)
            ax.set_aspect("equal")
            ax.set_title(f"Force |f|\nMean={fm[mask].mean():.3f}")
            plt.colorbar(sc, ax=ax)
        else:
            ax.set_aspect("equal")
            ax.set_title("No force applied")

    if E_node is not None:
        ax = axes[1, 0]
        ax.hist(E_node, bins=30, alpha=0.7, edgecolor="black")
        ax.set_title("E Histogram")

    if nu_node is not None:
        ax = axes[1, 1]
        ax.hist(nu_node, bins=30, alpha=0.7, edgecolor="black")
        ax.set_title("nu Histogram")

    ax = axes[1, 2]
    if node_forces is not None:
        fm = np.sqrt(node_forces[:, 0]**2 + node_forces[:, 1]**2)
        nz = fm[fm > 0]
        if len(nz) > 0:
            ax.hist(nz, bins=30, alpha=0.7, edgecolor="black")
            ax.set_title(f"Force |f| Histogram\nMean={nz.mean():.3f}")
        else:
            ax.set_title("No force applied")
    else:
        ax.set_title("No force field")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"mesh_materials_forces_{tag}_sid{sid}.png"), dpi=300)
    plt.close()
    print(f"[Saved] mesh_materials_forces_{tag}_sid{sid}.png in {args.out_dir}")


# ============================================================
# 5) Dataset loader (Q4 NPZ, B2 fields)
# ============================================================
def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def load_fem_dataset_Q4_with_materials_and_random_force(path, ntrain, ntest):
    """
    Expect NPZ fields (Q4, B2 ring geometry):
      coord, quad, disp2D, inner_edges, theta0_nodes, thetahalfpi_nodes,
      E_node, nu_node, boundary_info
    boundary_info: dict with keys node_indices, force_vectors, coordinates
    """
    data = np.load(path, allow_pickle=True)

    coords = data["coord"]
    quads = data["quad"]
    disp2Ds = data["disp2D"]
    inner_edges_list = data["inner_edges"]
    theta0_nodes_list = data["theta0_nodes"]
    thetahalfpi_nodes_list = data["thetahalfpi_nodes"]
    E_nodes_list = data["E_node"]
    nu_nodes_list = data["nu_node"]
    boundary_info_list = data["boundary_info"]

    N = len(coords)
    assert ntrain + ntest <= N, f"ntrain+ntest={ntrain+ntest} exceeds dataset size N={N}"

    def build_sample(i):
        binfo = _as_dict(boundary_info_list[i])
        boundary_nodes = np.array(binfo["node_indices"], dtype=np.int64)
        boundary_forces = np.array(binfo["force_vectors"], dtype=np.float32)
        boundary_coords = np.array(binfo["coordinates"], dtype=np.float32)

        n_nodes = len(coords[i])
        node_forces = np.zeros((n_nodes, 2), dtype=np.float32)
        if len(boundary_nodes) > 0:
            node_forces[boundary_nodes] = boundary_forces

        return {
            "gid": int(i),
            "xy": coords[i].astype(np.float32),
            "quad": quads[i].astype(np.int64),
            "uv_exact": disp2Ds[i].astype(np.float32),
            "inner_edges": inner_edges_list[i].astype(np.int64),
            "theta0_nodes": theta0_nodes_list[i].astype(np.int64),
            "thetahalfpi_nodes": thetahalfpi_nodes_list[i].astype(np.int64),
            "E_node": E_nodes_list[i].astype(np.float32),
            "nu_node": nu_nodes_list[i].astype(np.float32),
            "node_forces": node_forces,
            "boundary_nodes": boundary_nodes,
            "boundary_forces": boundary_forces,
            "boundary_coords": boundary_coords,
        }

    Train_samples = [build_sample(i) for i in range(ntrain)]
    Test_samples = [build_sample(ntrain + i) for i in range(ntest)]

    # ntrain=0 is a legitimate call pattern for evaluation-only use (e.g.
    # evaluate_ood.py) -- see train_B1.py's version of this function for
    # the rationale.
    if Train_samples:
        print("\nMaterial statistics in training set:")
        all_E = np.concatenate([s["E_node"] for s in Train_samples], axis=0)
        all_nu = np.concatenate([s["nu_node"] for s in Train_samples], axis=0)
        print(f"  E: mean={all_E.mean():.2f}, std={all_E.std():.2f}, min={all_E.min():.2f}, max={all_E.max():.2f}")
        print(f"  nu: mean={all_nu.mean():.4f}, std={all_nu.std():.4f}, min={all_nu.min():.4f}, max={all_nu.max():.4f}")

        print("\nForce statistics in training set:")
        all_forces = np.concatenate([s["node_forces"] for s in Train_samples], axis=0)
        fm = np.sqrt(all_forces[:, 0]**2 + all_forces[:, 1]**2)
        nz = fm[fm > 0]
        if len(nz) > 0:
            print(f"  Force: mean={nz.mean():.3f}, std={nz.std():.3f}, min={nz.min():.3f}, max={nz.max():.3f}")
            print(f"  Nodes with force: {len(nz)} / {len(fm)}")
        else:
            print("  No nonzero forces found.")

    return Train_samples, Test_samples


# ============================================================
# 6) Eval + Viz (Q4)
# ============================================================
@torch.no_grad()
def evaluate_dataset_hyperelastic_Q4(samples, model, args, device, dtype, eval_batch_size=32):
    """Batched evaluation -- see train_B1.py's version for the rationale
    (validation now runs every 25-50 epochs instead of every 250-500, so
    its own cost matters)."""
    model.eval()
    rel_l2_u_list, rel_l2_v_list = [], []
    E_mean_list, nu_mean_list, f_mean_list = [], [], []

    if len(samples) == 0:
        return {
            "mean_rel_L2_u": 0.0, "mean_rel_L2_v": 0.0,
            "std_rel_L2_u": 0.0, "std_rel_L2_v": 0.0,
            "mean_E": 0.0, "mean_nu": 0.0, "mean_force": 0.0,
        }

    s0 = samples[0]
    xy = torch.tensor(s0["xy"], device=device, dtype=dtype)
    quad = torch.tensor(s0["quad"], device=device, dtype=torch.long)
    inner_edges = torch.tensor(s0["inner_edges"], device=device, dtype=torch.long)
    theta0_nodes = torch.tensor(s0["theta0_nodes"], device=device, dtype=torch.long)
    thetahalfpi_nodes = torch.tensor(s0["thetahalfpi_nodes"], device=device, dtype=torch.long)

    n = len(samples)
    for start in range(0, n, eval_batch_size):
        chunk = samples[start:start + eval_batch_size]
        E_batch = torch.tensor(np.stack([c["E_node"] for c in chunk]), device=device, dtype=dtype)
        nu_batch = torch.tensor(np.stack([c["nu_node"] for c in chunk]), device=device, dtype=dtype)
        force_batch = torch.tensor(np.stack([c["node_forces"] for c in chunk]), device=device, dtype=dtype)
        uv_exact_batch = torch.tensor(np.stack([c["uv_exact"] for c in chunk]), device=device, dtype=dtype)

        _, _, _, uv_pred_batch, _ = total_potential_energy_Q4_hyperelastic(
            xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
            model, E_batch, nu_batch, force_batch,
            use_soft_dirichlet=args.use_soft_dirichlet,
            R_out=args.R_out,
            mode=args.mode,
            dtype=dtype,
            fun_dim=args.fun_dim,
            material=args.material,
        )

        err = uv_pred_batch - uv_exact_batch
        l2_u = torch.sqrt(torch.mean(err[:, :, 0] ** 2, dim=1))
        l2_v = torch.sqrt(torch.mean(err[:, :, 1] ** 2, dim=1))
        ref_u = torch.sqrt(torch.mean(uv_exact_batch[:, :, 0] ** 2, dim=1)) + 1e-12
        ref_v = torch.sqrt(torch.mean(uv_exact_batch[:, :, 1] ** 2, dim=1)) + 1e-12

        rel_l2_u_list.extend((l2_u / ref_u).tolist())
        rel_l2_v_list.extend((l2_v / ref_v).tolist())
        E_mean_list.extend(E_batch.mean(dim=1).tolist())
        nu_mean_list.extend(nu_batch.mean(dim=1).tolist())

        fm = torch.sqrt(torch.sum(force_batch ** 2, dim=2))
        for i in range(fm.shape[0]):
            nz = fm[i][fm[i] > 0]
            if len(nz) > 0:
                f_mean_list.append(nz.mean().item())

    return {
        "mean_rel_L2_u": float(np.mean(rel_l2_u_list)) if rel_l2_u_list else 0.0,
        "mean_rel_L2_v": float(np.mean(rel_l2_v_list)) if rel_l2_v_list else 0.0,
        "std_rel_L2_u": float(np.std(rel_l2_u_list)) if rel_l2_u_list else 0.0,
        "std_rel_L2_v": float(np.std(rel_l2_v_list)) if rel_l2_v_list else 0.0,
        "mean_E": float(np.mean(E_mean_list)) if E_mean_list else 0.0,
        "mean_nu": float(np.mean(nu_mean_list)) if nu_mean_list else 0.0,
        "mean_force": float(np.mean(f_mean_list)) if f_mean_list else 0.0,
    }


def dump_bad_case_q4(samples, sid, epoch, it, args, reason="nonfinite"):
    os.makedirs(args.out_dir, exist_ok=True)
    s = samples[sid]
    gid = s.get("gid", sid)

    save_path = os.path.join(args.out_dir, f"bad_case_{reason}_epoch{epoch}_it{it}_sid{sid}_gid{gid}.npz")
    np.savez(
        save_path,
        gid=gid,
        xy=s["xy"],
        quad=s["quad"],
        uv_exact=s.get("uv_exact", None),
        inner_edges=s.get("inner_edges", None),
        theta0_nodes=s.get("theta0_nodes", None),
        thetahalfpi_nodes=s.get("thetahalfpi_nodes", None),
        E_node=s.get("E_node", None),
        nu_node=s.get("nu_node", None),
        node_forces=s.get("node_forces", None),
    )

    plot_mesh_with_materials_and_forces_q4(samples, args, sid=sid, tag=f"bad_{reason}_epoch{epoch}_it{it}")

    print("\n" + "="*80)
    print(f"[NaN/Inf DETECTED] reason={reason}")
    print(f"  epoch={epoch}, it={it}, sid(in split)={sid}, gid(in original npz)={gid}")
    print(f"  saved bad sample: {save_path}")
    print("="*80 + "\n")


@torch.no_grad()
def visualize_one_hyperelastic_Q4(samples, model, args, device, dtype, sid=0, tag="final"):
    model.eval()
    s = samples[sid]
    xy = torch.tensor(s["xy"], device=device, dtype=dtype)
    quad = torch.tensor(s["quad"], device=device, dtype=torch.long)
    inner_edges = torch.tensor(s["inner_edges"], device=device, dtype=torch.long)
    theta0_nodes = torch.tensor(s["theta0_nodes"], device=device, dtype=torch.long)
    thetahalfpi_nodes = torch.tensor(s["thetahalfpi_nodes"], device=device, dtype=torch.long)
    uv_exact = torch.tensor(s["uv_exact"], device=device, dtype=dtype)
    E_node = torch.tensor(s["E_node"], device=device, dtype=dtype)
    nu_node = torch.tensor(s["nu_node"], device=device, dtype=dtype)
    node_forces = torch.tensor(s["node_forces"], device=device, dtype=dtype)

    Pi, U, W, uv_pred, Fg = total_potential_energy_Q4_hyperelastic_single(
        xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
        model, E_node, nu_node, node_forces,
        use_soft_dirichlet=args.use_soft_dirichlet,
        R_out=args.R_out,
        mode=args.mode,
        dtype=dtype,
        fun_dim=args.fun_dim,
        material=args.material,
    )

    err = uv_pred - uv_exact
    l2_u = torch.sqrt(torch.mean(err[:, 0] ** 2)).item()
    l2_v = torch.sqrt(torch.mean(err[:, 1] ** 2)).item()
    ref_u = torch.sqrt(torch.mean(uv_exact[:, 0] ** 2)).item() + 1e-12
    ref_v = torch.sqrt(torch.mean(uv_exact[:, 1] ** 2)).item() + 1e-12
    rel_l2_u = l2_u / ref_u
    rel_l2_v = l2_v / ref_v
    linf = torch.max(torch.linalg.norm(err, dim=1)).item()

    fm = torch.sqrt(torch.sum(node_forces ** 2, dim=1))
    nz = fm[fm > 0]
    force_mean = nz.mean().item() if len(nz) > 0 else 0.0

    Jg = Fg[:, 0, 0] * Fg[:, 1, 1] - Fg[:, 0, 1] * Fg[:, 1, 0]
    J_mean, J_min, J_max = Jg.mean().item(), Jg.min().item(), Jg.max().item()

    print(f"[viz sid={sid} tag={tag}] Pi={Pi.item():.3e}  U={U.item():.3e}  W={W.item():.3e}")
    print(f"                L2(u)={rel_l2_u:.3e}  L2(v)={rel_l2_v:.3e}  Linf(|e|)={linf:.3e}")
    print(f"                E_mean={E_node.mean().item():.1f}  nu_mean={nu_node.mean().item():.3f}  Force_mean={force_mean:.3f}")
    print(f"                det(F)@GP: mean={J_mean:.3f}, min={J_min:.3f}, max={J_max:.3f}")

    x_np = s["xy"][:, 0]
    y_np = s["xy"][:, 1]
    uvp = uv_pred.detach().cpu().numpy()
    uve = uv_exact.detach().cpu().numpy()
    ep = uvp - uve

    u_pred, v_pred = uvp[:, 0], uvp[:, 1]
    u_exact, v_exact = uve[:, 0], uve[:, 1]
    u_err, v_err = ep[:, 0], ep[:, 1]

    umag_pred = np.sqrt(u_pred*u_pred + v_pred*v_pred)
    umag_exact = np.sqrt(u_exact*u_exact + v_exact*v_exact)
    umag_err = np.sqrt(u_err*u_err + v_err*v_err)

    def save_combined_scatter(pred, exact, err, name, err_label, cmap="coolwarm"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmin = min(pred.min(), exact.min())
        vmax = max(pred.max(), exact.max())

        sc1 = axes[0].scatter(x_np, y_np, c=pred, s=8, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_aspect("equal"); axes[0].set_title("Prediction")
        plt.colorbar(sc1, ax=axes[0])

        sc2 = axes[1].scatter(x_np, y_np, c=exact, s=8, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_aspect("equal"); axes[1].set_title("Exact")
        plt.colorbar(sc2, ax=axes[1])

        sc3 = axes[2].scatter(x_np, y_np, c=err, s=8, cmap=cmap)
        axes[2].set_aspect("equal"); axes[2].set_title(f"Error ({err_label})")
        plt.colorbar(sc3, ax=axes[2])

        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{name}_combined_{tag}_sid{sid}.png"), dpi=300)
        plt.close()

    save_combined_scatter(u_pred, u_exact, u_err, "ux", f"L2={rel_l2_u:.3e}")
    save_combined_scatter(v_pred, v_exact, v_err, "uy", f"L2={rel_l2_v:.3e}")
    save_combined_scatter(umag_pred, umag_exact, umag_err, "umag", f"Linf={linf:.3e}")


# ============================================================
# 7) Train (Q4)
# ============================================================
def train_hyperelastic_Q4(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    Train_samples, Test_samples = load_fem_dataset_Q4_with_materials_and_random_force(
        args.path, args.ntrain, args.ntest
    )
    print(f"Loaded Q4 dataset: ntrain={len(Train_samples)}, ntest={len(Test_samples)}")

    for i in range(min(3, len(Train_samples))):
        s = Train_samples[i]
        fm = np.sqrt(np.sum(s["node_forces"]**2, axis=1))
        nz = fm[fm > 0]
        fmean = nz.mean() if len(nz) > 0 else 0.0
        print(f"Sample {i}: nodes={s['xy'].shape[0]}, quads={s['quad'].shape[0]}, "
              f"E_mean={s['E_node'].mean():.1f}, nu_mean={s['nu_node'].mean():.3f}, "
              f"Force_mean={fmean:.3f}, Force_nodes={len(nz)}")

    plot_mesh_with_materials_and_forces_q4(Train_samples, args, sid=0, tag="train_sample0")
    if len(Test_samples) > 0:
        plot_mesh_with_materials_and_forces_q4(Test_samples, args, sid=0, tag="test_sample0")

    model = get_model(args).Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=args.fun_dim,
        out_dim=2,
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=args.unified_pos
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ntrain = len(Train_samples)
    bs = max(int(args.batch_size), 1)
    validate_every = max(int(args.validate_every if getattr(args, "validate_every", None) else args.save_every), 1)
    metrics_history = []
    start_epoch = 1
    opt_steps = 0
    train_wall_clock = 0.0
    best_val_error = None
    best_epoch = None
    patience_counter = 0

    if os.path.exists(os.path.join(args.out_dir, "model_final.pt")):
        print(f"[resume] {args.out_dir}/model_final.pt already exists -- this case is already fully trained, skipping.")
        return
    if os.path.exists(os.path.join(args.out_dir, "EARLY_STOPPED")):
        print(f"[resume] {args.out_dir}/EARLY_STOPPED marker found -- this case already stopped early, skipping.")
        return

    ckpt_path, last_epoch = find_latest_checkpoint(args.out_dir)
    if ckpt_path is not None:
        print(f"[resume] Found checkpoint {ckpt_path} (epoch {last_epoch}) -- resuming from epoch {last_epoch + 1}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        start_epoch = last_epoch + 1

        metrics_history_path = os.path.join(args.out_dir, "metrics_history.json")
        if os.path.exists(metrics_history_path):
            with open(metrics_history_path) as f:
                metrics_history = json.load(f)
            for rec in metrics_history:
                train_wall_clock = max(train_wall_clock, rec.get("cumulative_wall_clock_s", 0.0))
                opt_steps = max(opt_steps, rec.get("opt_steps", 0))
                if "val_error" in rec and (best_val_error is None or rec["val_error"] < best_val_error - 1e-12):
                    best_val_error = rec["val_error"]
                    best_epoch = rec["epoch"]

        best_ckpt_meta_path = os.path.join(args.out_dir, "best_checkpoint_meta.json")
        if os.path.exists(best_ckpt_meta_path):
            with open(best_ckpt_meta_path) as f:
                best_meta = json.load(f)
            best_val_error = best_meta.get("best_val_error", best_val_error)
            best_epoch = best_meta.get("best_epoch", best_epoch)
            patience_counter = best_meta.get("patience_counter", 0)

        # Replay the LR-decay schedule so resuming doesn't jump back to the
        # full initial LR (see the epoch%1000==0 decay below)
        e = 1000
        while e < args.epochs and e < start_epoch:
            for pg in opt.param_groups:
                pg["lr"] *= 0.9
            e += 1000
        print(f"[resume] LR after replaying decay schedule: {opt.param_groups[0]['lr']:.2e}")

    early_stop_patience = int(getattr(args, "early_stop_patience", 0))
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 1e-4))
    stopped_early = False

    # See train_B1.py's version for the full rationale: mem_get_info()
    # gives the whole-device (nvidia-smi-like) usage, distinct from
    # PyTorch's own allocated/reserved bookkeeping below.
    device_total_mb = (torch.cuda.mem_get_info(device)[1] / 1e6) if device.type == "cuda" else 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        order = np.random.permutation(ntrain) if getattr(args, "shuffle", 1) else np.arange(ntrain)

        model.train()
        epoch_t0 = time.time()
        skipped = 0
        epoch_peak_device_used_mb = 0.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for bi, start in enumerate(range(0, ntrain, bs)):
            batch_idx = order[start:start + bs]
            batch_samples = [Train_samples[i] for i in batch_idx]

            xy = torch.tensor(batch_samples[0]["xy"], device=device, dtype=dtype)
            quad = torch.tensor(batch_samples[0]["quad"], device=device, dtype=torch.long)
            inner_edges = torch.tensor(batch_samples[0]["inner_edges"], device=device, dtype=torch.long)
            theta0_nodes = torch.tensor(batch_samples[0]["theta0_nodes"], device=device, dtype=torch.long)
            thetahalfpi_nodes = torch.tensor(batch_samples[0]["thetahalfpi_nodes"], device=device, dtype=torch.long)
            E_batch = torch.tensor(np.stack([s["E_node"] for s in batch_samples]), device=device, dtype=dtype)
            nu_batch = torch.tensor(np.stack([s["nu_node"] for s in batch_samples]), device=device, dtype=dtype)
            force_batch = torch.tensor(np.stack([s["node_forces"] for s in batch_samples]), device=device, dtype=dtype)

            Pi, U, W, uv_pred, Fg = total_potential_energy_Q4_hyperelastic(
                xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes,
                model, E_batch, nu_batch, force_batch,
                use_soft_dirichlet=args.use_soft_dirichlet,
                R_out=args.R_out,
                mode=args.mode,
                dtype=dtype,
                fun_dim=args.fun_dim,
                material=args.material,
            )

            finite_mask = torch.isfinite(Pi) & torch.isfinite(uv_pred).all(dim=(1, 2))
            if not finite_mask.all():
                bad_local = torch.nonzero(~finite_mask).flatten().tolist()
                skipped += len(bad_local)
                for li in bad_local:
                    dump_bad_case_q4(Train_samples, int(batch_idx[li]), epoch, start, args, reason="nonfinite")
                if not finite_mask.any():
                    opt.zero_grad(set_to_none=True)
                    continue
                loss = Pi[finite_mask].mean()
            else:
                loss = Pi.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()

            bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
            if bad_grad:
                skipped += len(batch_idx)
                opt.zero_grad(set_to_none=True)
                continue

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            opt_steps += 1

            if device.type == "cuda":
                free_b, total_b = torch.cuda.mem_get_info(device)
                epoch_peak_device_used_mb = max(epoch_peak_device_used_mb, (total_b - free_b) / 1e6)

            if bi % args.print_every == 0:
                fm = torch.sqrt(torch.sum(force_batch ** 2, dim=2))
                nz = fm[fm > 0]
                fmean = nz.mean().item() if nz.numel() > 0 else 0.0
                Jg = Fg[:, 0, 0] * Fg[:, 1, 1] - Fg[:, 0, 1] * Fg[:, 1, 0]
                Jmean = Jg.mean().item()
                print(f"[batch {bi:5d}] (epoch={epoch}) Pi_mean={loss.item():.6e}  "
                      f"E={E_batch.mean().item():.1f}  nu={nu_batch.mean().item():.3f}  "
                      f"Force={fmean:.3f}  detF@GP={Jmean:.3f}  opt_steps={opt_steps}")

        epoch_time = time.time() - epoch_t0
        train_wall_clock += epoch_time

        if skipped > 0:
            print(f"[epoch {epoch}] skipped={skipped} bad samples")

        do_checkpoint = (epoch % args.save_every == 0)
        do_validate = (epoch % validate_every == 0)

        if do_checkpoint:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pt"))

        if do_validate:
            viz_sid = random.randint(0, ntrain - 1)
            visualize_one_hyperelastic_Q4(
                Train_samples, model, args, device, dtype,
                sid=viz_sid, tag=f"epoch{epoch}"
            )

            gpu_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0
            gpu_mem_reserved_mb = (torch.cuda.max_memory_reserved(device) / 1e6) if device.type == "cuda" else 0.0
            gpu_mem_device_mb = epoch_peak_device_used_mb

            metrics = evaluate_dataset_hyperelastic_Q4(Test_samples, model, args, device, dtype)
            val_error = 0.5 * (metrics["mean_rel_L2_u"] + metrics["mean_rel_L2_v"])
            print("===== Test-set Evaluation =====")
            print(f"Mean Rel L2(u): {metrics['mean_rel_L2_u']:.3e}")
            print(f"Mean Rel L2(v): {metrics['mean_rel_L2_v']:.3e}")
            print(f"Std  Rel L2(u): {metrics['std_rel_L2_u']:.3e}")
            print(f"Std  Rel L2(v): {metrics['std_rel_L2_v']:.3e}")
            print(f"Avg E: {metrics['mean_E']:.1f}")
            print(f"Avg nu: {metrics['mean_nu']:.3f}")
            print(f"Avg Force: {metrics['mean_force']:.3f}")
            print(f"Val error (mean of Rel L2 u,v): {val_error:.3e}  "
                  f"epoch_time={epoch_time:.2f}s  cum_wall_clock={train_wall_clock:.1f}s  "
                  f"gpu_peak_mem_allocated={gpu_mem_mb:.1f}MB  gpu_peak_mem_reserved={gpu_mem_reserved_mb:.1f}MB  "
                  f"gpu_peak_mem_device={gpu_mem_device_mb:.1f}MB/{device_total_mb:.1f}MB  "
                  f"opt_steps={opt_steps}")

            is_best = (best_val_error is None) or (val_error < best_val_error - early_stop_min_delta)
            if is_best:
                best_val_error = val_error
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pt"))
                with open(os.path.join(args.out_dir, "best_checkpoint_meta.json"), "w") as f:
                    json.dump({"best_val_error": best_val_error, "best_epoch": best_epoch,
                               "patience_counter": patience_counter}, f, indent=2)
                print(f"[best] New best val_error={best_val_error:.3e} at epoch={best_epoch} -> model_best.pt")
            else:
                patience_counter += 1
                with open(os.path.join(args.out_dir, "best_checkpoint_meta.json"), "w") as f:
                    json.dump({"best_val_error": best_val_error, "best_epoch": best_epoch,
                               "patience_counter": patience_counter}, f, indent=2)

            metrics_record = {
                "epoch": int(epoch), **metrics,
                "val_error": val_error,
                "is_best": bool(is_best),
                "epoch_time_s": epoch_time,
                "cumulative_wall_clock_s": train_wall_clock,
                "gpu_peak_mem_mb": gpu_mem_mb,
                "gpu_peak_mem_reserved_mb": gpu_mem_reserved_mb,
                "gpu_peak_mem_device_mb": gpu_mem_device_mb,
                "gpu_device_total_mb": device_total_mb,
                "opt_steps": opt_steps,
                "batch_size": bs,
            }
            metrics_history.append(metrics_record)
            with open(os.path.join(args.out_dir, "metrics_history.json"), "w") as f:
                json.dump(metrics_history, f, indent=2)

            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"[early stop] val_error hasn't improved for {patience_counter} validation "
                      f"events (patience={early_stop_patience}) -- stopping at epoch {epoch}, "
                      f"best was epoch {best_epoch} (val_error={best_val_error:.3e})")
                stopped_early = True
                break

        if epoch % 1000 == 0 and epoch < args.epochs:
            for pg in opt.param_groups:
                pg["lr"] *= 0.9
            print(f"[epoch {epoch}] lr -> {opt.param_groups[0]['lr']:.2e}")

    if not stopped_early:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    else:
        with open(os.path.join(args.out_dir, "EARLY_STOPPED"), "w") as f:
            f.write(f"stopped_at_epoch={epoch}\nbest_epoch={best_epoch}\nbest_val_error={best_val_error}\n")

    final_metrics = evaluate_dataset_hyperelastic_Q4(Test_samples, model, args, device, dtype)
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print(f"Mean Rel L2(u): {final_metrics['mean_rel_L2_u']:.3e}")
    print(f"Mean Rel L2(v): {final_metrics['mean_rel_L2_v']:.3e}")
    print(f"Avg E: {final_metrics['mean_E']:.1f}")
    print(f"Avg nu: {final_metrics['mean_nu']:.3f}")
    print(f"Avg Force: {final_metrics['mean_force']:.3f}")
    if best_epoch is not None:
        print(f"Best val_error: {best_val_error:.3e} at epoch {best_epoch} (see model_best.pt)")
    print(f"Total wall clock: {train_wall_clock:.1f}s over {opt_steps} optimizer steps (batch_size={bs})")
    print("="*80)

    inference_stats = benchmark_inference_latency_Q4(Test_samples, model, args, device, dtype)
    print(f"Inference latency (batch_size=1, forward pass only, no training): "
          f"{inference_stats['inference_ms_per_sample']:.4f} ms/sample "
          f"(averaged over {inference_stats['inference_n_repeats']} calls, {device.type})")
    with open(os.path.join(args.out_dir, "inference_latency.json"), "w") as f:
        json.dump(inference_stats, f, indent=2)

    print("\nTraining completed!")
    print(f"Results saved to: {args.out_dir}")


# ============================================================
# 8) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        "PFEM/Transolver on FEM Q4 meshes -- B2 (quarter ring, internal pressure, symmetry BCs)"
    )

    parser.add_argument("--path", type=str, default="./omar_pfem/data/training_data_B2_q4/hyperelastic_training_data_q4.npz")
    parser.add_argument("--R_in", type=float, default=1.0)
    parser.add_argument("--R_out", type=float, default=2.0)

    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])

    parser.add_argument("--mode", type=str, default="plane_strain", choices=["plane_stress", "plane_strain"])

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10,
                         help="Epoch cadence for full model_epoch{N}.pt checkpoints (resume points)")
    parser.add_argument("--validate_every", type=int, default=None,
                         help="Epoch cadence for test-set evaluation + best-checkpoint tracking; "
                              "defaults to --save_every if not set.")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                         help="Stop if val_error hasn't improved for this many validation events "
                              "(0 disables early stopping)")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                         help="Minimum val_error improvement to reset the early-stopping patience counter")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./results_B2_transolver")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1,
                         help="Real mini-batch size (parallel forward pass over this many samples "
                              "at once, all sharing B2's fixed mesh) -- not gradient accumulation.")
    parser.add_argument("--shuffle", type=int, default=0)

    parser.add_argument("--ntrain", type=int, default=35)
    parser.add_argument("--ntest", type=int, default=10)

    parser.add_argument("--use_soft_dirichlet", type=int, default=1)

    parser.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unified_pos", type=int, default=0)
    parser.add_argument("--ref", type=int, default=16)
    parser.add_argument("--slice_num", type=int, default=128)

    parser.add_argument("--fun_dim", type=int, default=4)

    args = parser.parse_args()
    args.use_soft_dirichlet = bool(args.use_soft_dirichlet)

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Dataset file not found: {args.path}")

    print(f"Using dataset: {args.path}")
    print(f"Output dir: {args.out_dir}")
    print(f"Material: {args.material}")
    print(f"fun_dim={args.fun_dim}  (3->[E,nu,fy], 4->[E,nu,fx,fy])")

    train_hyperelastic_Q4(args)


if __name__ == "__main__":
    main()
