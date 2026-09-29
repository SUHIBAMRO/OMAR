"""Local integration refinement for B3's region-stress QoI, per Timon's
explicit request (2026-09-28 reply): before any resolution-changing
retrain, test whether just integrating the ALREADY-COMPUTED coarse-mesh
displacement field (production resolution, 6,840 elements -- the same
mesh the operator itself was trained/evaluated on, UNCHANGED) more
finely near the groove region changes the measured region-stress
accuracy.

Why this is different from evaluate_B3_qois_finer.py: that script
refines the MESH (re-solves a new FEM problem, queries the network
zero-shot at new node positions) -- this script refines only the
QUADRATURE used to INTEGRATE/SAMPLE the region, reusing the SAME
coarse element's own already-known 8 nodal displacement values (no new
FEM solve, no new network query at all). This isolates "is 6 points
enough to reliably SAMPLE the region" from "is the operator's own
discretization too coarse" -- exactly the distinction Timon asked to
separate.

How it works: a hex8 element's displacement field is defined entirely
by its own 8 corner nodal values via trilinear (isoparametric) shape
functions. torch-fem's own `Solid.eval_shape_functions(xi)` accepts
ANY local coordinates `xi` (not just its default 2-point-per-axis
quadrature) and returns the shape functions N, the physical-space
gradient operator B, and the Jacobian determinant detJ at exactly those
points -- so passing a much finer Gauss-Legendre grid (n_sub points per
axis instead of 2, i.e. n_sub^3 points per element instead of 8) gives
a denser sample of the SAME already-known trilinear displacement field,
with no new solve needed. Physical position of each fine point is
N @ (element's own node coordinates); it is then tested against the
same physical region membership test the coarse-quadrature region mask
already uses (distance from the groove's deepest point < region_radius).

Reported metrics, addressing Timon's explicit requests:
  - n_region_fine_local: how many fine quadrature points now fall in
    the region (should be far above the 6,840-element case's 6, since
    this refines WITHIN the same coarse elements near the region).
  - All SIX independent Cauchy stress components (sigma_xx, sigma_yy,
    sigma_zz, sigma_xy, sigma_yz, sigma_xz) reported SEPARATELY, with
    their own typical magnitude (volume-weighted RMS) in the region --
    so a small individual component cannot silently dominate a combined
    error without being visible.
  - A full-tensor field-relative error per sample (same formula as
    evaluate_B3_qois.compute_region_cauchy_field_rel, reused
    unmodified), now computed at the locally-refined quadrature.
  - A SINGLE, dataset-wide Frobenius-norm relative error -- ONE
    normalization constant (sqrt(mean over ALL samples and ALL region
    points of ||sigma_true||_F^2)), not a per-sample denominator -- so
    an individual sample's small region-stress magnitude cannot inflate
    the reported number, directly addressing Timon's normalization
    concern.

Verified locally before any GPU run: (1) an identity/consistency check
-- refining to the SAME degree as the original 2-point rule reproduces
the original coarse computation; (2) a total-element-volume check (the
fine quadrature must integrate each element's own volume to the same
value the coarse quadrature does, confirming the weights/detJ are
correctly combined, not just plausible-looking); (3) a full toy-scale
end-to-end run with a real FEM solve and a real (if undertrained) model.
"""
import argparse
import json

import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import (
    build_fixed_geometry, apply_dirichlet_b3, build_model, neo_hookean_energy_density_batched,
)
from omar_pfem.train_B1 import install_input_norm_for_checkpoint, _apply_input_norm
from omar_pfem.data.data_generate_B3_dataset import (
    DEFAULT_RESOLUTION, R_GRADING, GROOVE_DEPTH, GROOVE_HALF_WIDTH,
)
# NOTE: GROOVE_DEPTH/GROOVE_HALF_WIDTH deliberately imported from
# data_generate_B3_dataset (0.20/0.15, the real production groove), NOT
# from mesh_convergence_B3 (which has its own STALE module-level
# defaults, 0.05/0.15, from an earlier groove candidate -- a documented
# pitfall elsewhere in this project; caught here directly by a failing
# volume-consistency check before it silently produced a wrong-geometry
# region, not assumed safe).
from omar_pfem.data.mesh_convergence_B3 import R_IN0, R_OUT, LZ
from omar_pfem.data.data_generate_B3 import groove_R_in, groove_radius_of_curvature

STRESS_COMPONENT_NAMES = ["xx", "yy", "zz", "xy", "yz", "xz"]

# Cauchy stress is symmetric: the full 3x3 tensor's Frobenius norm sums
# ALL 9 entries, which counts each off-diagonal (shear) entry TWICE
# (sigma_xy appears at both [0,1] and [1,0], equal by symmetry). When
# combining the SIX INDEPENDENT components into a tensor-norm-style
# quantity, the shear components must be weighted by 2 to reproduce the
# true ||sigma||_F^2 = xx^2+yy^2+zz^2 + 2*(xy^2+yz^2+xz^2) -- omitting
# this (summing the six components unweighted, as an earlier version of
# this file did) understates the true tensor norm and was caught as a
# real bug during review (2026-09-29), before it reached any reported
# number: the OLD full-3x3-tensor code elsewhere in this project
# (evaluate_B3_qois.compute_region_cauchy_field_rel) already gets this
# right automatically, since summing over a full 3x3 tensor's 9 entries
# naturally double-counts the symmetric off-diagonal pair -- this
# weight vector reproduces that same behavior from the reduced
# 6-component representation.
FROB_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def fine_quadrature_3d(n_sub):
    """n_sub^3-point tensor-product Gauss-Legendre rule on [-1,1]^3,
    via numpy's own leggauss (standard library implementation, exact
    for polynomials up to degree 2*n_sub-1 per axis -- not hand-derived).
    Returns (points: (n_sub**3, 3), weights: (n_sub**3,))."""
    xi1d, w1d = np.polynomial.legendre.leggauss(n_sub)
    X1, X2, X3 = np.meshgrid(xi1d, xi1d, xi1d, indexing="ij")
    W1, W2, W3 = np.meshgrid(w1d, w1d, w1d, indexing="ij")
    points = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=-1)
    weights = (W1 * W2 * W3).ravel()
    return points, weights


def build_shape_function_evaluator(Ntheta, Nr, Nz, device, dtype):
    """A throwaway torch-fem Solid model, built ONLY so its
    eval_shape_functions can be queried at arbitrary (non-default) local
    coordinates -- same construction build_fixed_geometry already uses
    for the DEFAULT quadrature, exposed here for a custom one. No solve
    is ever run on this model."""
    from omar_pfem.data.data_generate_B3 import generate_grid_hex8_bushing
    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=R_GRADING)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
        elements_t = torch.tensor(elements, dtype=torch.long, device=device)
        dummy_params = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=dummy_params)
        with torch.device(device):
            model = Solid(nodes_t, elements_t, material)
    finally:
        torch.set_default_dtype(old_default_dtype)
    return model


def region_membership(pos, region_ref_point, region_radius):
    """pos: (..., 3) physical positions. Returns a boolean mask of the
    same leading shape -- SAME physical-region test (distance from the
    groove's deepest point) the coarse-quadrature region mask already
    uses, just applied to however many points are given."""
    dist = torch.linalg.norm(pos - region_ref_point[None, None, :], dim=-1)
    return dist < region_radius


def identify_region_elements(model_sf, elements, fine_xi, region_ref_point, region_radius):
    """Which COARSE elements have at least one FINE-quadrature point
    within the physical region -- purely geometric (independent of any
    material/displacement), computed ONCE and reused for every training
    batch. These elements get their energy contribution computed via
    the FINE quadrature during training (see
    total_potential_energy_B3_locally_refined); every other element
    keeps the standard coarse 8-point rule, completely unchanged. Uses
    the SAME region-membership test/fine grid as the evaluation-side
    convergence sweep, so "which elements count as region elements" is
    consistent between training and evaluation."""
    dtype, device = model_sf.nodes.dtype, model_sf.nodes.device
    fine_xi_t = torch.tensor(fine_xi, dtype=dtype, device=device)
    N, _, _ = model_sf.eval_shape_functions(fine_xi_t)
    nodes_elem = model_sf.nodes[elements]  # (n_elem, 8, 3)
    pos = torch.einsum("pn,enk->epk", N, nodes_elem).permute(1, 0, 2)  # (nf, n_elem, 3)
    in_region = region_membership(pos, region_ref_point, region_radius)  # (nf, n_elem)
    return in_region.any(dim=0)  # (n_elem,) bool


def total_potential_energy_B3_locally_refined(u, E_node, nu_node, geom, model_sf,
                                               region_elem_mask, fine_xi, fine_w):
    """Same physics and same total_potential_energy_B3 formula, but
    elements in region_elem_mask (touching the groove) have their own
    energy contribution integrated with a FINER quadrature (fine_xi/
    fine_w, n_sub^3 points) instead of the standard 8-point rule --
    every other element is completely unchanged. This is Timon's actual
    request (2026-09-28/29): decouple the OPERATOR's own discretization
    (geom['nodes']/geom['elements'], untouched here) from the DEM
    background-integration mesh, refining ONLY the latter, and ONLY
    where it matters physically. No element's volume is double-counted
    or dropped: each element contributes via EXACTLY ONE of the two
    quadrature rules (fine XOR coarse), verified directly by a
    volume-consistency check (see the module's own test suite) against
    the standard total_potential_energy_B3 at n_sub=2.

    Differentiable end-to-end through ordinary autograd (u already
    requires_grad from the network during training) -- no special
    autograd.grad handling needed here, unlike the stress-extraction
    functions elsewhere in this file, since this returns the energy
    itself, not a derived stress."""
    elements = geom["elements"]
    non_region_mask = ~region_elem_mask
    Batch = u.shape[0]

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)

    U = torch.zeros(Batch, device=u.device, dtype=u.dtype)

    # Standard coarse quadrature, restricted to non-region elements only.
    B_op, detJ, iweights = geom["B"], geom["detJ"], geom["iweights"]
    ue_nr = ue[:, non_region_mask]
    mu_nr = mu_elem[:, non_region_mask]
    lam_nr = lam_elem[:, non_region_mask]
    for g in range(B_op.shape[0]):
        H = torch.einsum("bedq,ecq->bedc", ue_nr, B_op[g, non_region_mask])
        F = eye + H
        psi = neo_hookean_energy_density_batched(F, mu_nr, lam_nr)
        U = U + torch.sum(psi * detJ[g, non_region_mask][None, :] * iweights[g], dim=1)

    # Finer quadrature, restricted to region elements only.
    if region_elem_mask.any():
        fine_xi_t = torch.tensor(fine_xi, dtype=u.dtype, device=u.device)
        fine_w_t = torch.tensor(fine_w, dtype=u.dtype, device=u.device)
        N_f, B_f, detJ_f = model_sf.eval_shape_functions(fine_xi_t)
        ue_r = ue[:, region_elem_mask]
        mu_r = mu_elem[:, region_elem_mask]
        lam_r = lam_elem[:, region_elem_mask]
        for p in range(fine_xi.shape[0]):
            H = torch.einsum("bedq,ecq->bedc", ue_r, B_f[p, region_elem_mask])
            F = eye + H
            psi = neo_hookean_energy_density_batched(F, mu_r, lam_r)
            U = U + torch.sum(psi * detJ_f[p, region_elem_mask][None, :] * fine_w_t[p], dim=1)

    return U


def compute_region_local_refined(u, E_node, nu_node, model_sf, elements, fine_xi, fine_w,
                                  region_ref_point, region_radius):
    """Full six-component Cauchy stress at a LOCALLY REFINED quadrature
    (fine_xi/fine_w, n_sub^3 points per element) within the region,
    reusing the SAME coarse element's own already-known nodal
    displacement field u (no new FEM solve, no new network query).

    Returns: sigma (B, n_region_fine, 6) [xx,yy,zz,xy,yz,xz],
             weight (n_region_fine,) volume weight, n_region_fine (int).
    """
    dtype, device = u.dtype, u.device
    fine_xi_t = torch.tensor(fine_xi, dtype=dtype, device=device)
    N, B_op, detJ = model_sf.eval_shape_functions(fine_xi_t)  # N:(nf,8) B:(nf,n_elem,3,8) detJ:(nf,n_elem)

    nodes_elem = model_sf.nodes[elements]  # (n_elem, 8, 3)
    pos = torch.einsum("pn,enk->epk", N, nodes_elem)  # (n_elem, nf, 3)
    pos = pos.permute(1, 0, 2)  # (nf, n_elem, 3)
    in_region = region_membership(pos, region_ref_point, region_radius)  # (nf, n_elem)
    n_region_fine = int(in_region.sum().item())

    Batch = u.shape[0]
    if n_region_fine == 0:
        return (torch.full((Batch, 0, 6), float("nan"), dtype=dtype, device=device),
                torch.zeros(0, dtype=dtype, device=device), 0)

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (B, n_elem, 3, 8)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)

    fine_w_t = torch.tensor(fine_w, dtype=dtype, device=device)  # (nf,)
    vol_weight = fine_w_t[:, None] * detJ.abs()  # (nf, n_elem)

    sigma_all = []
    w_all = []
    for p in range(fine_xi.shape[0]):
        mask_p = in_region[p]  # (n_elem,)
        if not mask_p.any():
            continue
        H = torch.einsum("bedq,ecq->bedc", ue[:, mask_p], B_op[p, mask_p])  # (B, n_masked, 3, 3)
        F = (eye + H).detach().requires_grad_(True)
        psi = neo_hookean_energy_density_batched(F, mu_elem[:, mask_p], lam_elem[:, mask_p])
        P, = torch.autograd.grad(psi.sum(), F, create_graph=False)
        detF = torch.linalg.det(F).detach()
        sigma = torch.matmul(P.detach(), F.detach().transpose(-1, -2)) / detF[..., None, None]  # (B, n_masked, 3, 3)
        sigma6 = torch.stack([
            sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2],
            sigma[..., 0, 1], sigma[..., 1, 2], sigma[..., 0, 2],
        ], dim=-1)  # (B, n_masked, 6)
        sigma_all.append(sigma6)
        w_all.append(vol_weight[p, mask_p])

    sigma_cat = torch.cat(sigma_all, dim=1)  # (B, n_region_fine, 6)
    w_cat = torch.cat(w_all, dim=0)  # (n_region_fine,)
    return sigma_cat, w_cat, n_region_fine


def total_element_volume_check(model_sf, elements, fine_xi, fine_w, coarse_B_op, coarse_detJ, coarse_iweights):
    """Sanity check: integrating '1' over each element with the FINE
    quadrature must reproduce the SAME per-element volume the coarse
    (default, 2-point) quadrature already gives -- confirms the fine
    weights/detJ are combined correctly, not just plausible-looking."""
    fine_xi_t = torch.tensor(fine_xi, dtype=coarse_detJ.dtype, device=coarse_detJ.device)
    _, _, detJ_fine = model_sf.eval_shape_functions(fine_xi_t)
    fine_w_t = torch.tensor(fine_w, dtype=coarse_detJ.dtype, device=coarse_detJ.device)
    vol_fine = (fine_w_t[:, None] * detJ_fine.abs()).sum(dim=0)  # (n_elem,)
    vol_coarse = (coarse_iweights[:, None] * coarse_detJ.abs()).sum(dim=0)  # (n_elem,)
    return vol_fine, vol_coarse


def evaluate_at_n_sub(n_sub, u_true_all, u_pred_all, E_node_all, nu_node_all,
                       model_sf, elements, region_ref_point, region_radius,
                       eval_batch_size, device, dtype):
    """One local-refinement level: returns n_region_fine, the per-
    component report (RMS + median-abs magnitude, pooled relative
    error, CORRECTLY Frobenius-weighted -- shear components count
    double, matching the true symmetric-tensor norm), the single
    dataset-wide pooled Frobenius-norm relative error, and the
    per-sample field-relative error list."""
    fine_xi, fine_w = fine_quadrature_3d(n_sub)
    n_samples = u_true_all.shape[0]
    frob_w = FROB_WEIGHTS.to(device=device, dtype=dtype)

    per_sample_field_rel = []
    sigma_true_sum_c = torch.zeros(6, dtype=dtype)
    sigma_pred_sum_c = torch.zeros(6, dtype=dtype)
    diff_sum_c = torch.zeros(6, dtype=dtype)
    true_abs_all_c = [[] for _ in range(6)]  # for median|true| per component
    pooled_w_sum = 0.0
    n_region_fine_reported = None

    for start in range(0, n_samples, eval_batch_size):
        end = min(start + eval_batch_size, n_samples)
        E_b = E_node_all[start:end].to(device=device, dtype=dtype)
        nu_b = nu_node_all[start:end].to(device=device, dtype=dtype)
        u_true = u_true_all[start:end].to(device=device, dtype=dtype)
        u_pred = u_pred_all[start:end].to(device=device, dtype=dtype)

        sigma_true, w, n_region_fine = compute_region_local_refined(
            u_true, E_b, nu_b, model_sf, elements, fine_xi, fine_w,
            region_ref_point, region_radius)
        sigma_pred, _, _ = compute_region_local_refined(
            u_pred, E_b, nu_b, model_sf, elements, fine_xi, fine_w,
            region_ref_point, region_radius)
        n_region_fine_reported = n_region_fine

        w_sum = w.sum().item()
        diff = sigma_pred - sigma_true  # (b, n_region_fine, 6)
        # Frobenius-norm-style combination: shear components (indices
        # 3,4,5 = xy,yz,xz) weighted by 2, per the true symmetric-tensor
        # norm -- see FROB_WEIGHTS's own docstring.
        diff_sq_sample = (diff ** 2 * frob_w).sum(dim=-1)          # (b, n_region_fine)
        true_sq_sample = (sigma_true ** 2 * frob_w).sum(dim=-1)    # (b, n_region_fine)
        num_sample = torch.sqrt((diff_sq_sample * w[None, :]).sum(dim=1) / w_sum)
        den_sample = torch.sqrt((true_sq_sample * w[None, :]).sum(dim=1) / w_sum)
        per_sample_field_rel.extend((num_sample / den_sample.clamp_min(1e-300)).tolist())

        for c in range(6):
            sigma_true_sum_c[c] += (sigma_true[..., c] ** 2 * w[None, :]).sum().item()
            sigma_pred_sum_c[c] += (sigma_pred[..., c] ** 2 * w[None, :]).sum().item()
            diff_sum_c[c] += ((sigma_pred[..., c] - sigma_true[..., c]) ** 2 * w[None, :]).sum().item()
            true_abs_all_c[c].extend(sigma_true[..., c].abs().reshape(-1).tolist())
        pooled_w_sum += w_sum * (end - start)

    component_report = {}
    for c, name in enumerate(STRESS_COMPONENT_NAMES):
        mag_true_rms = float(np.sqrt(sigma_true_sum_c[c].item() / pooled_w_sum))
        mag_pred_rms = float(np.sqrt(sigma_pred_sum_c[c].item() / pooled_w_sum))
        mag_true_median_abs = float(np.median(true_abs_all_c[c]))
        err = float(np.sqrt(diff_sum_c[c].item() / pooled_w_sum) / max(mag_true_rms, 1e-300))
        component_report[f"sigma_{name}"] = {
            "true_rms_magnitude": mag_true_rms, "pred_rms_magnitude": mag_pred_rms,
            "true_median_abs_magnitude": mag_true_median_abs,
            "pooled_rel_error": err,
        }

    # ONE common Frobenius-norm normalization across the WHOLE dataset
    # (not per-sample) -- Timon's explicit request, so an individual
    # sample's small region-stress magnitude cannot inflate the number.
    # Correctly Frobenius-weighted (shear counted twice), unlike an
    # earlier version of this file that summed the six components
    # unweighted (caught in review before it reached any real number).
    diff_sum_frob = (diff_sum_c * FROB_WEIGHTS.to(dtype=dtype)).sum().item()
    true_sum_frob = (sigma_true_sum_c * FROB_WEIGHTS.to(dtype=dtype)).sum().item()
    pooled_num = float(np.sqrt(diff_sum_frob / pooled_w_sum))
    pooled_den = float(np.sqrt(true_sum_frob / pooled_w_sum))
    pooled_frobenius_rel = pooled_num / max(pooled_den, 1e-300)

    field_rel_arr = np.array(per_sample_field_rel)
    return {
        "n_region_fine": n_region_fine_reported,
        "component_report": component_report,
        "pooled_frobenius_rel_error": pooled_frobenius_rel,
        "mean_field_rel_error": float(field_rel_arr.mean()),
        "median_field_rel_error": float(np.median(field_rel_arr)),
        "std_field_rel_error": float(field_rel_arr.std()),
        "per_sample_field_rel_error": per_sample_field_rel,
    }


def main():
    parser = argparse.ArgumentParser(
        "Local integration refinement CONVERGENCE study for B3's region-stress QoI "
        "(Timon's request, 2026-09-28/29): SAME coarse mesh/operator discretization, "
        "only the QUADRATURE used to sample/integrate the groove region is refined -- "
        "no new FEM solve, no new network query. Sweeps n_sub to check whether the "
        "tensor error STABILIZES (converges) as more region points are added, rather "
        "than reporting one arbitrary refinement level.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--n_sub_sweep", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12],
                         help="Gauss-Legendre points per axis (n_sub**3 per element) at each "
                              "step of the convergence sweep. Default gives region point "
                              "counts of roughly 6, 36, 124, 292, 566, 994 at production "
                              "resolution (n_sub=2 matches the original default 2-point rule).")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    print(f"Coarse mesh (operator's own discretization, UNCHANGED): ({Ntheta},{Nr},{Nz})")
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"  {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    model_sf = build_shape_function_evaluator(Ntheta, Nr, Nz, device, dtype)

    region_ref_point = torch.tensor(
        [R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0], dtype=dtype, device=device)
    region_radius = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loading FEM dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as h5:
        E_node_all = torch.tensor(h5["E_node"][:], dtype=dtype)
        nu_node_all = torch.tensor(h5["nu_node"][:], dtype=dtype)
        phi_all = torch.tensor(h5["phi"][:], dtype=dtype)
        u_true_all = torch.tensor(h5["displacements"][:], dtype=dtype)
    n_samples = E_node_all.shape[0]
    print(f"  {n_samples} FEM samples")

    # Network inference does NOT depend on n_sub (same coarse mesh,
    # same nodes, every time) -- computed ONCE here and reused for
    # every level of the convergence sweep, not re-run per level.
    print("Running network inference once (reused for every sweep level)...")
    u_pred_chunks = []
    for start in range(0, n_samples, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n_samples)
        E_b = E_node_all[start:end].to(device=device, dtype=dtype)
        nu_b = nu_node_all[start:end].to(device=device, dtype=dtype)
        phi_b = phi_all[start:end].to(device=device, dtype=dtype)
        with torch.no_grad():
            fun_material = torch.stack([E_b, nu_b, phi_b[:, None].expand(-1, geom["n_nodes"])], dim=2)
            fun_material = _apply_input_norm(fun_material)
            xyz_batch = geom["nodes"].unsqueeze(0).expand(end - start, -1, -1)
            u_net = model(xyz_batch, fun_material)
            u_pred = apply_dirichlet_b3(u_net, geom, phi_b)
        u_pred_chunks.append(u_pred.cpu())
    u_pred_all = torch.cat(u_pred_chunks, dim=0)

    sweep_results = []
    prev_pooled = None
    for n_sub in args.n_sub_sweep:
        print(f"\n=== n_sub={n_sub} ({n_sub ** 3} points/element) ===")
        r = evaluate_at_n_sub(n_sub, u_true_all, u_pred_all, E_node_all, nu_node_all,
                               model_sf, geom["elements"], region_ref_point, region_radius,
                               args.eval_batch_size, device, dtype)
        rel_change = (abs(r["pooled_frobenius_rel_error"] - prev_pooled) / prev_pooled
                      if prev_pooled is not None else None)
        prev_pooled = r["pooled_frobenius_rel_error"]
        r["n_sub"] = n_sub
        r["rel_change_from_previous_level"] = rel_change
        sweep_results.append(r)

        print(f"  n_region_fine={r['n_region_fine']}  "
              f"pooled_frobenius_rel_error={r['pooled_frobenius_rel_error']:.4f}"
              + (f"  (change from previous level: {rel_change:.2%})" if rel_change is not None else ""))
        for name, d in r["component_report"].items():
            print(f"    {name}: true_RMS={d['true_rms_magnitude']:.4f}  "
                  f"true_median_abs={d['true_median_abs_magnitude']:.4f}  "
                  f"pred_RMS={d['pred_rms_magnitude']:.4f}  "
                  f"pooled_rel_error={d['pooled_rel_error']:.4f}")

    print(f"\n{'=' * 90}")
    print("CONVERGENCE SUMMARY (pooled Frobenius-norm relative error vs. n_sub):")
    for r in sweep_results:
        change_str = f"{r['rel_change_from_previous_level']:.2%}" if r["rel_change_from_previous_level"] is not None else "--"
        print(f"  n_sub={r['n_sub']:3d}  n_region_fine={r['n_region_fine']:5d}  "
              f"pooled_frobenius_rel_error={r['pooled_frobenius_rel_error']:.4f}  "
              f"change={change_str}")
    print("If this stabilizes (small relative change between the last few levels) and stays "
          "elevated, the region-stress gap is real, not a local-integration-count artifact.")

    result = {
        "checkpoint": args.checkpoint, "checkpoint_iter": ckpt["iter"],
        "dataset": args.dataset, "n_samples": n_samples,
        "coarse_resolution": [Ntheta, Nr, Nz],
        "n_region_coarse_default": 6,
        "sweep": sweep_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
