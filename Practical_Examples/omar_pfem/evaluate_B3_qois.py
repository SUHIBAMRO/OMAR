"""Real physical QoIs (total strain energy, reaction force/moment,
fixed-region Cauchy stress) for a trained B3 Transolver checkpoint,
comparing its predicted displacement field against the true FEM field --
not just raw displacement L2 error.

Why this exists: Timon named regional Cauchy stress, reaction
force/moment, and total strain energy as the real success criteria for
B3 (matching this project's own established B1/B2/mesh_convergence_B3
convention), not pointwise displacement error alone. checkpoint_50000.pt
(run 4, normalized inputs) reaches 1.88% combined displacement error --
this script checks whether that translates into comparably good
stress/reaction/energy accuracy.

No new FEM solve is needed: every QoI below is computed directly from a
KNOWN displacement field (either the true FEM one already in
dataset_clean_holdout's dataset.h5, or the network's own prediction) via
the exact same total_potential_energy_B3 functional train_B3.py's loss
already uses, and the exact same Gauss-point/region-mask machinery
mesh_convergence_B3.py already established for B3's FEM-side QoI study:

  - Total strain energy: total_potential_energy_B3(u, E, nu, geom)
    directly (Pi = U for B3, no external-work term).
  - Reaction force/moment on the fixed outer housing: since B3's whole
    load is prescribed-displacement (no external nodal force anywhere),
    the reaction at a Dirichlet-constrained node is exactly
    d(total_potential_energy_B3)/d(u) there, by the same variational
    argument any FEM solver itself uses internally to report reactions --
    computed here via autograd on a KNOWN field, not a new nonlinear
    solve. A translation-invariance identity (sum of this quantity over
    ALL nodes ~ 0 for ANY field, equilibrium or not -- hyperelastic
    energy cannot see a uniform rigid translation of u) is checked as an
    implementation-correctness sanity check, not a proof that a field is
    physically valid -- confirmed directly (2026-09-27) to hold at
    machine precision (~1e-15) for both fields tested.
  - Fixed-region Cauchy stress (volume-weighted average + 99th
    percentile of sigma_xx, AND a volume-weighted full-tensor relative
    FIELD error): same physical region (near the groove's deepest
    point, continuously bonded, no BC-transition contamination) and
    Gauss-point parametric mapping mesh_convergence_B3.py uses, built
    ONCE (purely geometric, independent of material/displacement) and
    reused for every sample. Deformation gradient F at each Gauss point
    reuses the same B_op/detJ construction total_potential_energy_B3
    already assembles; first Piola-Kirchhoff stress P = d(psi)/d(F) via
    autograd (exact for Neo-Hookean, no closed-form transcription
    risk), then sigma = (1/det F) P F^T.

    IMPORTANT (found 2026-09-27, on the real GPU result):
    region_avg_sigma_xx's naive relative error (|pred-true|/(|true|+eps)
    on a per-sample SIGNED average) can blow up arbitrarily when
    tension/compression partially cancel within the region, making the
    true signed average pass near zero for some samples -- exactly the
    failure mode mesh_convergence_B3.py's own `cauchy_field_rel` was
    already introduced to fix for the FEM-vs-FEM case (see that file's
    2026-09-21 note). `compute_region_cauchy_field_rel` applies the
    same fix here, for the network-vs-FEM case: a volume-weighted
    full-TENSOR relative field error, squaring each Gauss point's
    contribution before summing, so a near-zero SIGNED average cannot
    collapse the denominator. Verified against a real true=0 identity
    check (u_pred=u_true gives exactly 0.0, not just "small").

Usage:
  python -m omar_pfem.evaluate_B3_qois \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_training_normalized/checkpoint_50000.pt \
      --dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_training_normalized/qois_50000.json
"""
import argparse
import json

import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import (
    build_fixed_geometry, apply_dirichlet_b3, build_model, total_potential_energy_B3,
    neo_hookean_energy_density_batched,
)
from omar_pfem.train_B1 import install_input_norm_for_checkpoint
from omar_pfem.data.data_generate_B3_dataset import (
    DEFAULT_RESOLUTION, GROOVE_DEPTH, GROOVE_HALF_WIDTH, R_GRADING,
)
from omar_pfem.data.mesh_convergence_B3 import (
    R_IN0, R_OUT, LZ, MIN_RELIABLE_N_P99, HEXA1_IPOINTS, _gauss_point_parametric,
)
from omar_pfem.data.data_generate_B3 import groove_R_in, groove_radius_of_curvature


def build_region_mask(Ntheta, Nr, Nz, r_grading):
    """Purely geometric (elem,gauss) boolean mask -- same fixed physical
    region mesh_convergence_B3.py uses, built with THIS project's real
    production groove parameters (GROOVE_DEPTH=0.20, GROOVE_HALF_WIDTH=0.15
    from data_generate_B3_dataset.py, NOT mesh_convergence_B3.py's own
    module-level 0.05/0.15 defaults from an earlier, different groove
    candidate)."""
    theta_g, t_g, z_g = _gauss_point_parametric(Ntheta, Nr, Nz, r_grading, HEXA1_IPOINTS)
    R_in_eff_g = groove_R_in(z_g, LZ, R_IN0, GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    Rr_g = R_in_eff_g + (R_OUT - R_in_eff_g) * t_g
    x_g = Rr_g * np.cos(theta_g)
    y_g = Rr_g * np.sin(theta_g)
    pos_g = np.stack([x_g, y_g, z_g], axis=-1)

    ref_point = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    dist = np.linalg.norm(pos_g - ref_point[None, None, :], axis=-1)
    return dist < region_radius  # (n_elem, n_gauss)


@torch.no_grad()
def compute_energy(u, E_node, nu_node, geom):
    return total_potential_energy_B3(u, E_node, nu_node, geom)


def compute_reaction(u, E_node, nu_node, geom):
    """Reaction force (B,3), reaction moment_y (B,), and a global
    IMPLEMENTATION-correctness check (B,) -- sum of d(U)/d(u) over EVERY
    node. This is NOT a test of whether u is a genuine equilibrium field:
    hyperelastic energy is invariant under a uniform rigid translation of
    the whole displacement field (F = I + grad(u) is unchanged by adding
    any constant to u), so this sum is ~0 by that symmetry alone for ANY
    u, equilibrium or not -- confirmed directly (2026-09-27): it comes
    out at machine precision (~1e-15) for BOTH the true FEM field and an
    untrained network's prediction. What it DOES verify is that this
    function's own indexing/broadcasting has no bug (a real error here,
    e.g. a wrong axis or a missed node, would generally break the
    translation-invariance identity and show up as a large residual)."""
    u_leaf = u.detach().clone().requires_grad_(True)
    U = total_potential_energy_B3(u_leaf, E_node, nu_node, geom)
    grad_u, = torch.autograd.grad(U.sum(), u_leaf, create_graph=False)

    outer = geom["outer"]
    f_outer = grad_u[:, outer, :]
    reaction_force = f_outer.sum(dim=1)

    center = torch.tensor([0.0, 0.0, LZ / 2.0], dtype=u.dtype, device=u.device)
    pos_outer = geom["nodes"][outer] - center
    B = u.shape[0]
    torque = torch.cross(pos_outer[None, :, :].expand(B, -1, -1), f_outer, dim=-1)
    reaction_moment_y = torque[:, :, 1].sum(dim=1)

    total_force = grad_u.sum(dim=1)  # (B,3) -- should be ~0 (no external load anywhere)
    equilibrium_residual = torch.linalg.norm(total_force, dim=1)
    return reaction_force.detach(), reaction_moment_y.detach(), equilibrium_residual.detach()


def compute_region_sigma_xx(u, E_node, nu_node, geom, region_mask_torch):
    """Volume-weighted region average + 99th percentile of sigma_xx
    (Batch,) at the fixed groove-neighborhood region, from a KNOWN
    displacement field -- reuses total_potential_energy_B3's own
    per-Gauss F construction, then derives P=d(psi)/d(F) via autograd
    (exact, not a hand-transcribed closed form) and
    sigma=(1/detF) P F^T."""
    elements = geom["elements"]
    B_op, detJ, iweights = geom["B"], geom["detJ"], geom["iweights"]
    n_gauss = B_op.shape[0]
    Batch = u.shape[0]

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)

    sigma_xx_per_gauss = []
    vol_weight_per_gauss = []
    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F = (eye + H).detach().requires_grad_(True)  # (B, n_elem, 3, 3)
        psi = neo_hookean_energy_density_batched(F, mu_elem, lam_elem)  # (B, n_elem)
        P, = torch.autograd.grad(psi.sum(), F, create_graph=False)     # (B, n_elem, 3, 3)
        detF = torch.linalg.det(F).detach()
        sigma = torch.matmul(P.detach(), F.detach().transpose(-1, -2)) / detF[..., None, None]
        sigma_xx_per_gauss.append(sigma[:, :, 0, 0])         # (B, n_elem)
        vol_weight_per_gauss.append(iweights[g] * detJ[g].abs())  # (n_elem,)

    sigma_xx = torch.stack(sigma_xx_per_gauss, dim=2)   # (B, n_elem, n_gauss)
    vol_weight = torch.stack(vol_weight_per_gauss, dim=1)  # (n_elem, n_gauss)

    sigma_xx_flat = sigma_xx[:, region_mask_torch]      # (B, n_region)
    w_flat = vol_weight[region_mask_torch]              # (n_region,)
    w_sum = w_flat.sum()
    n_region = region_mask_torch.sum().item()

    if n_region == 0 or w_sum <= 0:
        nan_col = torch.full((Batch,), float("nan"), dtype=u.dtype, device=u.device)
        return nan_col, nan_col.clone(), n_region

    region_avg = (sigma_xx_flat * w_flat[None, :]).sum(dim=1) / w_sum  # (B,)

    region_p99 = torch.full((Batch,), float("nan"), dtype=u.dtype, device=u.device)
    if n_region >= MIN_RELIABLE_N_P99:
        order = torch.argsort(sigma_xx_flat, dim=1)
        sorted_vals = torch.gather(sigma_xx_flat, 1, order)
        sorted_w = w_flat[order]
        cum_w = torch.cumsum(sorted_w, dim=1) / w_sum
        for b in range(Batch):
            idx99 = int(torch.searchsorted(cum_w[b], torch.tensor(0.99, dtype=u.dtype)).item())
            idx99 = min(idx99, sorted_vals.shape[1] - 1)
            region_p99[b] = sorted_vals[b, idx99]

    return region_avg, region_p99, n_region


def compute_region_sigma_full(u, E_node, nu_node, geom, region_mask_torch):
    """Same per-Gauss F/P/sigma construction as compute_region_sigma_xx,
    but keeps the FULL Cauchy tensor (not just the xx component),
    restricted to the fixed region: returns (B, n_region, 3, 3) plus the
    region's volume weights (n_region,). Needed for the volume-weighted,
    full-tensor relative FIELD error below -- a purely geometric
    (material-independent) weight, so it is identical for every sample."""
    elements = geom["elements"]
    B_op, detJ, iweights = geom["B"], geom["detJ"], geom["iweights"]
    n_gauss = B_op.shape[0]

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)

    sigma_per_gauss = []
    vol_weight_per_gauss = []
    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F = (eye + H).detach().requires_grad_(True)  # (B, n_elem, 3, 3)
        psi = neo_hookean_energy_density_batched(F, mu_elem, lam_elem)
        P, = torch.autograd.grad(psi.sum(), F, create_graph=False)
        detF = torch.linalg.det(F).detach()
        sigma = torch.matmul(P.detach(), F.detach().transpose(-1, -2)) / detF[..., None, None]
        sigma_per_gauss.append(sigma)  # (B, n_elem, 3, 3)
        vol_weight_per_gauss.append(iweights[g] * detJ[g].abs())  # (n_elem,)

    sigma_all = torch.stack(sigma_per_gauss, dim=2)      # (B, n_elem, n_gauss, 3, 3)
    vol_weight = torch.stack(vol_weight_per_gauss, dim=1)  # (n_elem, n_gauss)

    sigma_region = sigma_all[:, region_mask_torch]  # (B, n_region, 3, 3)
    w_region = vol_weight[region_mask_torch]        # (n_region,)
    return sigma_region, w_region


def compute_region_cauchy_field_rel(u_true, u_pred, E_node, nu_node, geom, region_mask_torch):
    """Volume-weighted, FULL-TENSOR relative field error of Cauchy
    stress within the fixed region:
      sqrt(sum(w * ||sigma_pred - sigma_true||_F^2) / sum(w))
      / sqrt(sum(w * ||sigma_true||_F^2) / sum(w))
    per sample -- the same convention mesh_convergence_B3.py's own
    `cauchy_field_rel` already established (after Omar's review there
    caught an earlier signed-scalar-ratio version producing a
    misleading, non-converging number: comparing a per-sample SIGNED
    average against itself lets tension/compression cancel in the
    denominator, which region_avg_sigma_xx's naive relative error above
    is vulnerable to in exactly the same way). Squaring each Gauss
    point's contribution before summing means a near-zero SIGNED
    average cannot make the denominator collapse -- the denominator
    here is a sum of squared norms, not a signed sum."""
    sigma_true, w = compute_region_sigma_full(u_true, E_node, nu_node, geom, region_mask_torch)
    sigma_pred, _ = compute_region_sigma_full(u_pred, E_node, nu_node, geom, region_mask_torch)
    Batch = u_true.shape[0]
    w_sum = w.sum()
    if w.shape[0] == 0 or w_sum <= 0:
        return torch.full((Batch,), float("nan"), dtype=u_true.dtype, device=u_true.device)

    diff_sq = torch.sum((sigma_pred - sigma_true) ** 2, dim=(-1, -2))  # (B, n_region)
    true_sq = torch.sum(sigma_true ** 2, dim=(-1, -2))                  # (B, n_region)
    num = torch.sqrt((diff_sq * w[None, :]).sum(dim=1) / w_sum)
    den = torch.sqrt((true_sq * w[None, :]).sum(dim=1) / w_sum)
    rel = torch.where(den > 0, num / den.clamp_min(1e-300),
                       torch.full_like(num, float("nan")))
    return rel


def main():
    parser = argparse.ArgumentParser(
        "Compute real physical QoIs (energy, reaction, region Cauchy stress) for a B3 checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    print(f"Building fixed geometry at resolution=({Ntheta},{Nr},{Nz})...")
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"  {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    region_mask_np = build_region_mask(Ntheta, Nr, Nz, R_GRADING)
    region_mask_torch = torch.tensor(region_mask_np, device=device)
    n_region = int(region_mask_np.sum())
    print(f"  fixed region: {n_region} Gauss points (reliable p99 needs >= {MIN_RELIABLE_N_P99})")

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node = torch.tensor(h5["E_node"][:], dtype=dtype)
        nu_node = torch.tensor(h5["nu_node"][:], dtype=dtype)
        phi = torch.tensor(h5["phi"][:], dtype=dtype)
        u_true_all = torch.tensor(h5["displacements"][:], dtype=dtype)
    n_samples = E_node.shape[0]
    print(f"  {n_samples} FEM samples")

    from omar_pfem.train_B1 import _apply_input_norm

    rows = []
    for start in range(0, n_samples, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n_samples)
        E_b = E_node[start:end].to(device=device, dtype=dtype)
        nu_b = nu_node[start:end].to(device=device, dtype=dtype)
        phi_b = phi[start:end].to(device=device, dtype=dtype)
        u_true = u_true_all[start:end].to(device=device, dtype=dtype)

        with torch.no_grad():
            fun_material = torch.stack([E_b, nu_b, phi_b[:, None].expand(-1, geom["n_nodes"])], dim=2)
            fun_material = _apply_input_norm(fun_material)
            xyz_batch = geom["nodes"].unsqueeze(0).expand(end - start, -1, -1)
            u_net = model(xyz_batch, fun_material)
            u_pred = apply_dirichlet_b3(u_net, geom, phi_b)

        U_true = compute_energy(u_true, E_b, nu_b, geom)
        U_pred = compute_energy(u_pred, E_b, nu_b, geom)

        Fr_true, Mr_true, eq_true = compute_reaction(u_true, E_b, nu_b, geom)
        Fr_pred, Mr_pred, eq_pred = compute_reaction(u_pred, E_b, nu_b, geom)

        avg_true, p99_true, _ = compute_region_sigma_xx(u_true, E_b, nu_b, geom, region_mask_torch)
        avg_pred, p99_pred, _ = compute_region_sigma_xx(u_pred, E_b, nu_b, geom, region_mask_torch)

        cauchy_field_rel = compute_region_cauchy_field_rel(u_true, u_pred, E_b, nu_b, geom, region_mask_torch)

        for i in range(end - start):
            rows.append({
                "energy_true": U_true[i].item(), "energy_pred": U_pred[i].item(),
                "reaction_moment_y_true": Mr_true[i].item(), "reaction_moment_y_pred": Mr_pred[i].item(),
                "reaction_force_true": Fr_true[i].tolist(), "reaction_force_pred": Fr_pred[i].tolist(),
                "equilibrium_residual_true": eq_true[i].item(), "equilibrium_residual_pred": eq_pred[i].item(),
                "region_avg_sigma_xx_true": avg_true[i].item(), "region_avg_sigma_xx_pred": avg_pred[i].item(),
                "region_p99_sigma_xx_true": p99_true[i].item(), "region_p99_sigma_xx_pred": p99_pred[i].item(),
                "region_cauchy_field_rel": cauchy_field_rel[i].item(),
            })
        print(f"  [{end}/{n_samples}] done")

    def rel_err(key_true, key_pred):
        """Naive mean-of-ratios relative error. Kept for continuity with
        B1/B2's own established convention, but this metric blows up
        arbitrarily when |true| is near zero for some samples (a single
        sample with true~0.004 and pred~-2.4 can dominate the mean even
        if every other sample is accurate) -- exactly the failure mode
        this project already documented for B2's checkpoint-selection
        metric. See the pooled/median/correlation stats below, which are
        NOT vulnerable to this and should be read together with this
        one, not instead of it."""
        t = np.array([r[key_true] for r in rows])
        p = np.array([r[key_pred] for r in rows])
        return float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-12))), float(np.std(np.abs(p - t) / (np.abs(t) + 1e-12)))

    def robust_stats(key_true, key_pred):
        """Metrics that stay meaningful even when some true values are
        near zero: median of the per-sample ratio (robust to a few huge
        outliers), a pooled RMS-based relative error
        sqrt(mean(err^2))/sqrt(mean(true^2)) (one global ratio, no
        per-sample division by a near-zero value), Pearson correlation,
        and sign-agreement fraction (whether the network even gets the
        sign of the quantity right)."""
        t = np.array([r[key_true] for r in rows])
        p = np.array([r[key_pred] for r in rows])
        abs_err = np.abs(p - t)
        median_rel = float(np.median(abs_err / (np.abs(t) + 1e-12)))
        pooled_rms_rel = float(np.sqrt(np.mean(abs_err ** 2)) / np.sqrt(np.mean(t ** 2)))
        corr = float(np.corrcoef(t, p)[0, 1]) if np.std(t) > 0 and np.std(p) > 0 else float("nan")
        sign_agree = float(np.mean(np.sign(t) == np.sign(p)))
        return median_rel, pooled_rms_rel, corr, sign_agree

    summary = {}
    for name, (kt, kp) in {
        "energy": ("energy_true", "energy_pred"),
        "reaction_moment_y": ("reaction_moment_y_true", "reaction_moment_y_pred"),
        "region_avg_sigma_xx": ("region_avg_sigma_xx_true", "region_avg_sigma_xx_pred"),
        "region_p99_sigma_xx": ("region_p99_sigma_xx_true", "region_p99_sigma_xx_pred"),
    }.items():
        mean_e, std_e = rel_err(kt, kp)
        median_rel, pooled_rms_rel, corr, sign_agree = robust_stats(kt, kp)
        summary[f"mean_rel_err_{name}"] = mean_e
        summary[f"std_rel_err_{name}"] = std_e
        summary[f"median_rel_err_{name}"] = median_rel
        summary[f"pooled_rms_rel_err_{name}"] = pooled_rms_rel
        summary[f"corr_{name}"] = corr
        summary[f"sign_agree_{name}"] = sign_agree
        print(f"  {name}: mean rel err = {mean_e:.4f} (std={std_e:.4f})  "
              f"median={median_rel:.4f}  pooled_rms={pooled_rms_rel:.4f}  "
              f"corr={corr:.4f}  sign_agree={sign_agree:.2f}")

    # region_cauchy_field_rel is already a per-sample normalized ratio
    # (volume-weighted full-tensor field error, NOT a signed-average
    # ratio -- see compute_region_cauchy_field_rel's docstring for why
    # this one is not vulnerable to the near-zero-denominator blowup
    # region_avg_sigma_xx's naive relative error above is), so it is
    # summarized directly rather than through rel_err/robust_stats.
    cfr = np.array([r["region_cauchy_field_rel"] for r in rows])
    cfr_valid = cfr[~np.isnan(cfr)]
    summary["mean_region_cauchy_field_rel"] = float(np.mean(cfr_valid)) if len(cfr_valid) else float("nan")
    summary["median_region_cauchy_field_rel"] = float(np.median(cfr_valid)) if len(cfr_valid) else float("nan")
    summary["std_region_cauchy_field_rel"] = float(np.std(cfr_valid)) if len(cfr_valid) else float("nan")
    summary["n_valid_region_cauchy_field_rel"] = int(len(cfr_valid))
    print(f"  region_cauchy_field_rel (full-tensor, volume-weighted, "
          f"NOT vulnerable to near-zero signed-average denominators): "
          f"mean={summary['mean_region_cauchy_field_rel']:.4f}  "
          f"median={summary['median_region_cauchy_field_rel']:.4f}  "
          f"std={summary['std_region_cauchy_field_rel']:.4f}  "
          f"(n_valid={summary['n_valid_region_cauchy_field_rel']}/{n_samples})")

    eq_true_mean = float(np.mean([r["equilibrium_residual_true"] for r in rows]))
    eq_pred_mean = float(np.mean([r["equilibrium_residual_pred"] for r in rows]))
    print(f"  translation-invariance identity check (~0 for ANY field, equilibrium "
          f"or not -- confirms no indexing bug, not a measure of solution quality): "
          f"true field={eq_true_mean:.2e}, predicted field={eq_pred_mean:.2e}")

    result = {
        "checkpoint": args.checkpoint, "checkpoint_iter": ckpt["iter"],
        "dataset": args.dataset, "n_samples": n_samples,
        "n_region_gauss_points": n_region,
        "summary": summary,
        "equilibrium_residual_true_mean": eq_true_mean,
        "equilibrium_residual_pred_mean": eq_pred_mean,
        "per_sample": rows,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
