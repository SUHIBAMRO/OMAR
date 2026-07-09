"""
Training script for the B1 benchmark (unit square, top-edge traction, fixed
bottom edge) using the PFEM/Transolver pipeline -- adapted from PFEM-main's
practical_problems/Hyper_beam/exp_hyper_PINO_transolver_quad.py (Yizheng
Wang et al., "Pretrain finite element method", JMPS 2026).

Unchanged from the beam example: the Q4 isoparametric shape functions, the
2x2-Gauss "explicit differentiation" energy computation (element strain
gradients via shape-function derivatives contracted with the network's
predicted nodal displacements, not autodiff), the Neo-Hookean strain energy
density, the total-potential-energy (Pi = U - W) loss with no other loss
term, and the Transolver architecture/training-loop structure.

Swapped for B1's geometry: the beam fixes the LEFT edge (x=0) and loads the
RIGHT edge (x=Lx) with a y-varying traction (cantilever); B1 fixes the
BOTTOM edge (y=0) and loads the TOP edge (y=Ly) with an x-varying traction.
The hard-Dirichlet ramp and the boundary dataset fields (top_edges /
bottom_nodes vs. right_edges / left_nodes) are updated accordingly.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import random

from omar_pfem.model_dict import get_model


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
# 1) Neo-Hookean energy density (vectorized) -- unchanged from PFEM
# ============================================================
def neo_hookean_energy_density_vectorized(F, mu, lam, dtype=torch.float32):
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    J = torch.clamp(J, min=1e-8)
    I1 = torch.sum(F ** 2, dim=(1, 2))
    lnJ = torch.log(J)
    psi = (mu / 2.0) * (I1 - 2.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)
    return psi


def E_nu_to_mu_lam(E, nu, mode="plane_strain"):
    if mode == "plane_strain":
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        mu = E / (2.0 * (1.0 + nu))
        nu_star = nu / (1.0 + nu)
        lam = E * nu_star / ((1.0 + nu_star) * (1.0 - 2.0 * nu_star))
    return mu, lam


# ============================================================
# 2) Q4 shape functions on [-1,1]^2 -- unchanged from PFEM
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
# 3) Hyperelastic energy on Q4 mesh (TL, 2x2 Gauss) -- unchanged from PFEM
# ============================================================
def compute_hyperelastic_energy_Q4(xy, quad, uv, mu_nodes, lam_nodes, mode="plane_strain", dtype=torch.float32):
    device = xy.device
    Q = quad.shape[0]

    Xe = xy[quad]
    ue = uv[quad]

    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    U = torch.zeros((), device=device, dtype=dtype)
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

        grad_u = torch.einsum("qai,qaj->qij", ue, dN_dX)

        I = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(Q, 2, 2)
        F = I + grad_u

        mu_e = torch.mean(mu_nodes[quad], dim=1)
        lam_e = torch.mean(lam_nodes[quad], dim=1)

        psi = neo_hookean_energy_density_vectorized(F, mu_e, lam_e, dtype=dtype)
        U = U + torch.sum(psi * detJ0 * w)

        F_list.append(F.detach())

    F_g = torch.cat(F_list, dim=0)
    return U, F_g


# ============================================================
# 4) Total potential energy Pi = U - W  (Q4 version, B1 geometry)
# ============================================================
def total_potential_energy_Q4_hyperelastic(
    xy, quad, top_edges, bottom_nodes,
    model, E_nodes, nu_nodes, node_forces,
    use_soft_dirichlet=True,
    Ly=1.0,
    mode="plane_strain",
    dtype=torch.float32,
    fun_dim=4,
):
    device = xy.device

    xy_domain = xy.unsqueeze(0)  # (1,N,2)

    if fun_dim == 3:
        fun_material = torch.cat([
            E_nodes.unsqueeze(1),
            nu_nodes.unsqueeze(1),
            node_forces[:, 1].unsqueeze(1),
        ], dim=1).unsqueeze(0)
    else:
        fun_material = torch.cat([
            E_nodes.unsqueeze(1),
            nu_nodes.unsqueeze(1),
            node_forces[:, 0].unsqueeze(1),
            node_forces[:, 1].unsqueeze(1),
        ], dim=1).unsqueeze(0)

    uv_raw = model(xy_domain, fun_material)
    if uv_raw.dim() == 3:
        uv_raw = uv_raw[0]

    # ---- Dirichlet: bottom edge (y=0) fixed -- ramp grows with y instead of x
    if use_soft_dirichlet:
        free = (xy[:, 1] / Ly).clamp(0.0, 1.0)
    else:
        free = torch.ones(xy.shape[0], device=device, dtype=dtype)
        if bottom_nodes.numel() > 0:
            free[bottom_nodes] = 0.0
    uv = uv_raw * free[:, None]

    mu_nodes, lam_nodes = E_nu_to_mu_lam(E_nodes, nu_nodes, mode=mode)

    U, Fg = compute_hyperelastic_energy_Q4(xy, quad, uv, mu_nodes, lam_nodes, mode=mode, dtype=dtype)

    W = torch.sum(node_forces * uv) / len(top_edges)

    Pi = U - W
    return Pi, U.detach(), W.detach(), uv, Fg


# ============================================================
# 5) Visualization helpers (Q4)
# ============================================================
def plot_mesh_with_materials_and_forces_q4(samples, args, sid=0, tag="mesh_material_force_check"):
    os.makedirs(args.out_dir, exist_ok=True)

    s = samples[sid]
    xy = s["xy"]
    quad = s["quad"]
    bottom_nodes = s["bottom_nodes"]
    top_edges = s["top_edges"]
    E_node = s.get("E_node", None)
    nu_node = s.get("nu_node", None)
    node_forces = s.get("node_forces", None)

    x = xy[:, 0]
    y = xy[:, 1]

    bottom_mask = np.zeros(x.shape[0], dtype=bool)
    if bottom_nodes is not None and len(bottom_nodes) > 0:
        bottom_mask[bottom_nodes] = True

    tri = quad_to_tri(quad)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    ax.triplot(x, y, tri, linewidth=0.3, color='gray', alpha=0.5)
    ax.scatter(x, y, s=6, c="k", alpha=0.7)
    if bottom_mask.any():
        ax.scatter(x[bottom_mask], y[bottom_mask], s=18, c="lime", label="Fixed (y=0)")
    if top_edges is not None and top_edges.shape[0] > 0:
        for (i, j) in top_edges:
            ax.plot([x[i], x[j]], [y[i], y[j]], "r-", linewidth=1.5)
        ax.plot([], [], "r-", linewidth=1.5, label="Traction edges")
    ax.set_aspect("equal")
    ax.set_title(f"Q4 Mesh - sid={sid}")
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
# 6) Dataset loader (Q4 NPZ, B1 fields)
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
    Expect NPZ fields (Q4, B1 geometry):
      coord, quad, disp2D, top_edges, bottom_nodes, E_node, nu_node, boundary_info
    boundary_info: dict with keys node_indices, force_vectors, coordinates
    """
    data = np.load(path, allow_pickle=True)

    coords = data["coord"]
    quads = data["quad"]
    disp2Ds = data["disp2D"]
    top_edges_list = data["top_edges"]
    bottom_nodes_list = data["bottom_nodes"]
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
            "top_edges": top_edges_list[i].astype(np.int64),
            "bottom_nodes": bottom_nodes_list[i].astype(np.int64),
            "E_node": E_nodes_list[i].astype(np.float32),
            "nu_node": nu_nodes_list[i].astype(np.float32),
            "node_forces": node_forces,
            "boundary_nodes": boundary_nodes,
            "boundary_forces": boundary_forces,
            "boundary_coords": boundary_coords,
        }

    Train_samples = [build_sample(i) for i in range(ntrain)]
    Test_samples = [build_sample(ntrain + i) for i in range(ntest)]

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
# 7) Eval + Viz (Q4)
# ============================================================
@torch.no_grad()
def evaluate_dataset_hyperelastic_Q4(samples, model, args, device, dtype):
    model.eval()
    rel_l2_u_list, rel_l2_v_list = [], []
    E_mean_list, nu_mean_list, f_mean_list = [], [], []

    for sid in range(len(samples)):
        s = samples[sid]
        xy = torch.tensor(s["xy"], device=device, dtype=dtype)
        quad = torch.tensor(s["quad"], device=device, dtype=torch.long)
        top_edges = torch.tensor(s["top_edges"], device=device, dtype=torch.long)
        bottom_nodes = torch.tensor(s["bottom_nodes"], device=device, dtype=torch.long)
        uv_exact = torch.tensor(s["uv_exact"], device=device, dtype=dtype)
        E_node = torch.tensor(s["E_node"], device=device, dtype=dtype)
        nu_node = torch.tensor(s["nu_node"], device=device, dtype=dtype)
        node_forces = torch.tensor(s["node_forces"], device=device, dtype=dtype)

        _, _, _, uv_pred, _ = total_potential_energy_Q4_hyperelastic(
            xy, quad, top_edges, bottom_nodes,
            model, E_node, nu_node, node_forces,
            use_soft_dirichlet=args.use_soft_dirichlet,
            Ly=args.Ly,
            mode=args.mode,
            dtype=dtype,
            fun_dim=args.fun_dim
        )

        err = uv_pred - uv_exact
        l2_u = torch.sqrt(torch.mean(err[:, 0] ** 2))
        l2_v = torch.sqrt(torch.mean(err[:, 1] ** 2))
        ref_u = torch.sqrt(torch.mean(uv_exact[:, 0] ** 2)) + 1e-12
        ref_v = torch.sqrt(torch.mean(uv_exact[:, 1] ** 2)) + 1e-12

        rel_l2_u_list.append((l2_u / ref_u).item())
        rel_l2_v_list.append((l2_v / ref_v).item())

        E_mean_list.append(E_node.mean().item())
        nu_mean_list.append(nu_node.mean().item())

        fm = torch.sqrt(torch.sum(node_forces ** 2, dim=1))
        nz = fm[fm > 0]
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
        top_edges=s.get("top_edges", None),
        bottom_nodes=s.get("bottom_nodes", None),
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
    top_edges = torch.tensor(s["top_edges"], device=device, dtype=torch.long)
    bottom_nodes = torch.tensor(s["bottom_nodes"], device=device, dtype=torch.long)
    uv_exact = torch.tensor(s["uv_exact"], device=device, dtype=dtype)
    E_node = torch.tensor(s["E_node"], device=device, dtype=dtype)
    nu_node = torch.tensor(s["nu_node"], device=device, dtype=dtype)
    node_forces = torch.tensor(s["node_forces"], device=device, dtype=dtype)

    Pi, U, W, uv_pred, Fg = total_potential_energy_Q4_hyperelastic(
        xy, quad, top_edges, bottom_nodes,
        model, E_node, nu_node, node_forces,
        use_soft_dirichlet=args.use_soft_dirichlet,
        Ly=args.Ly,
        mode=args.mode,
        dtype=dtype,
        fun_dim=args.fun_dim
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
# 8) Train (Q4)
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
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(ntrain) if getattr(args, "shuffle", 1) else np.arange(ntrain)

        model.train()
        opt.zero_grad(set_to_none=True)

        accum = 0
        skipped = 0

        for it, sid in enumerate(order):
            s = Train_samples[sid]

            xy = torch.tensor(s["xy"], device=device, dtype=dtype)
            quad = torch.tensor(s["quad"], device=device, dtype=torch.long)
            top_edges = torch.tensor(s["top_edges"], device=device, dtype=torch.long)
            bottom_nodes = torch.tensor(s["bottom_nodes"], device=device, dtype=torch.long)
            E_node = torch.tensor(s["E_node"], device=device, dtype=dtype)
            nu_node = torch.tensor(s["nu_node"], device=device, dtype=dtype)
            node_forces = torch.tensor(s["node_forces"], device=device, dtype=dtype)

            Pi, U, W, uv_pred, Fg = total_potential_energy_Q4_hyperelastic(
                xy, quad, top_edges, bottom_nodes,
                model, E_node, nu_node, node_forces,
                use_soft_dirichlet=args.use_soft_dirichlet,
                Ly=args.Ly,
                mode=args.mode,
                dtype=dtype,
                fun_dim=args.fun_dim
            )

            if (not torch.isfinite(Pi)) or (not torch.isfinite(uv_pred).all()):
                skipped += 1
                dump_bad_case_q4(Train_samples, sid, epoch, it, args, reason="nonfinite")
                opt.zero_grad(set_to_none=True)
                accum = 0
                continue

            loss = Pi / float(bs)
            loss.backward()
            accum += 1

            do_step = (accum == bs) or (it == ntrain - 1)
            if do_step:
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all()):
                        bad_grad = True
                        break
                if bad_grad:
                    skipped += accum
                    opt.zero_grad(set_to_none=True)
                    accum = 0
                    continue

                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0

            if it % args.print_every == 0:
                fm = torch.sqrt(torch.sum(node_forces ** 2, dim=1))
                nz = fm[fm > 0]
                fmean = nz.mean().item() if len(nz) > 0 else 0.0
                Jg = Fg[:, 0, 0] * Fg[:, 1, 1] - Fg[:, 0, 1] * Fg[:, 1, 0]
                Jmean = Jg.mean().item()
                print(f"[{it:6d}] (epoch={epoch}) Pi={Pi.item():.6e}  U={U.item():.6e}  W={W.item():.6e}  "
                      f"E={E_node.mean().item():.1f}  nu={nu_node.mean().item():.3f}  "
                      f"Force={fmean:.3f}  detF@GP={Jmean:.3f}")

        if skipped > 0:
            print(f"[epoch {epoch}] skipped={skipped} bad samples")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pt"))

            viz_sid = random.randint(0, ntrain - 1)
            visualize_one_hyperelastic_Q4(
                Train_samples, model, args, device, dtype,
                sid=viz_sid, tag=f"epoch{epoch}"
            )

            metrics = evaluate_dataset_hyperelastic_Q4(Test_samples, model, args, device, dtype)
            print("===== Test-set Evaluation =====")
            print(f"Mean Rel L2(u): {metrics['mean_rel_L2_u']:.3e}")
            print(f"Mean Rel L2(v): {metrics['mean_rel_L2_v']:.3e}")
            print(f"Std  Rel L2(u): {metrics['std_rel_L2_u']:.3e}")
            print(f"Std  Rel L2(v): {metrics['std_rel_L2_v']:.3e}")
            print(f"Avg E: {metrics['mean_E']:.1f}")
            print(f"Avg nu: {metrics['mean_nu']:.3f}")
            print(f"Avg Force: {metrics['mean_force']:.3f}")

            metrics_record = {"epoch": int(epoch), **metrics}
            metrics_history.append(metrics_record)
            with open(os.path.join(args.out_dir, "metrics_history.json"), "w") as f:
                json.dump(metrics_history, f, indent=2)

        if epoch % 1000 == 0 and epoch < args.epochs:
            for pg in opt.param_groups:
                pg["lr"] *= 0.9
            print(f"[epoch {epoch}] lr -> {opt.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))

    final_metrics = evaluate_dataset_hyperelastic_Q4(Test_samples, model, args, device, dtype)
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print(f"Mean Rel L2(u): {final_metrics['mean_rel_L2_u']:.3e}")
    print(f"Mean Rel L2(v): {final_metrics['mean_rel_L2_v']:.3e}")
    print(f"Avg E: {final_metrics['mean_E']:.1f}")
    print(f"Avg nu: {final_metrics['mean_nu']:.3f}")
    print(f"Avg Force: {final_metrics['mean_force']:.3f}")
    print("="*80)
    print("\nTraining completed!")
    print(f"Results saved to: {args.out_dir}")


# ============================================================
# 9) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        "PFEM/Transolver on FEM Q4 meshes -- B1 (unit square, top-edge traction, fixed bottom)"
    )

    parser.add_argument("--path", type=str, default="./omar_pfem/data/training_data_B1_q4/hyperelastic_training_data_q4.npz")
    parser.add_argument("--Lx", type=float, default=1.0)
    parser.add_argument("--Ly", type=float, default=1.0)

    parser.add_argument("--mode", type=str, default="plane_strain", choices=["plane_stress", "plane_strain"])

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./results_B1_transolver")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
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
    print(f"fun_dim={args.fun_dim}  (3->[E,nu,fy], 4->[E,nu,fx,fy])")

    train_hyperelastic_Q4(args)


if __name__ == "__main__":
    main()
