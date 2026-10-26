"""Mixed displacement-stress formulation for B3 (Professor Rabczuk's
item 1 of 4 in his 2026-10-16 follow-up-paper email, see
PROJECT_STATUS.md): the network predicts both u and the first
Piola-Kirchhoff stress P, trained against two physics residuals instead
of total potential energy --

  (a) constitutive consistency: P_pred should equal the P the material
      model itself implies for the network's own F = I + grad(u), i.e.
      P_true = dpsi/dF(F);
  (b) equilibrium: Div_0(P_pred) = 0 (no body force) in the interior.

This is a genuinely different training objective from train_B3.py's
Deep Energy Method (which only ever minimizes total potential energy
and never represents P as its own field) -- not a drop-in modification
of it. Deliberately kept as a SEPARATE module rather than edited into
train_B3.py: train_B3.py is already used for real, already-trained,
already-published production checkpoints (the B3 numbers in the
submitted paper), and this follow-up work must not risk changing its
behavior even by accident.

Reuses, rather than re-derives, every piece of machinery this project
already trusts: `generate_grid_hex8_bushing`/`boundary_node_sets` for
the mesh, `apply_dirichlet_b3` for the exact hard-constrained
displacement boundary conditions (completely unchanged -- the mixed
formulation only changes how the INTERIOR field and the new stress
field are trained, not the boundary treatment), and torch-fem's own
`Solid.eval_shape_functions` for N/B/detJ (same call train_B3.py's
`build_fixed_geometry` makes; duplicated here, rather than editing that
function, specifically so this file cannot alter train_B3.py's own
already-verified behavior).

The two new pieces of machinery this formulation needs beyond what
train_B3.py already has -- a closed-form Neo-Hookean P (cross-checked
against autograd dpsi/dF to 1e-14) and a generic per-node-field
gradient/divergence operator (verified against a uniform stress field,
Div=0 to 1e-13, AND a linear-in-X stress field with a known nonzero
closed-form Div, matched to 1e-13) -- were verified standalone, on CPU,
before this module was written; see
`verify_mixed_stress_ops.py` in this session's scratchpad for the
record of that check. Both are re-implemented here in their
production form (batched over quad points, no Python loop over quad
points in the hot path where avoidable).
"""
import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import generate_grid_hex8_bushing, boundary_node_sets
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH, R_GRADING
from omar_pfem.data.mesh_convergence_B3 import R_IN0, R_OUT, LZ
from omar_pfem.train_B3 import apply_dirichlet_b3  # noqa: F401 (re-exported for callers)


def build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=torch.float64):
    """Same mesh/geometry as train_B3.build_fixed_geometry, PLUS the
    shape-function VALUES N (train_B3.py discards these -- it only
    needs gradients B for the energy integral; the mixed formulation
    also needs N itself, to interpolate the network's own nodal P
    prediction to the quadrature points for the constitutive residual)."""
    nodes, elements = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=R_GRADING)
    inner, outer, sym = boundary_node_sets(nodes, R_IN0, R_OUT, LZ, GROOVE_DEPTH, GROOVE_HALF_WIDTH)

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
        dummy_params = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=dummy_params)
        with torch.device(device):
            model = Solid(nodes_t, elements_t, material)
        N_sf, B, detJ = model.eval_shape_functions(model.etype.ipoints)
        iweights = model.etype.iweights.to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(old_default_dtype)

    return {
        "nodes": nodes_t, "elements": elements_t,
        "inner": torch.tensor(inner, device=device), "outer": torch.tensor(outer, device=device),
        "sym": torch.tensor(sym, device=device),
        "theta_node": torch.tensor(theta_node, dtype=dtype, device=device),
        "t_node": torch.tensor(t_node, dtype=dtype, device=device),
        "N": N_sf, "B": B, "detJ": detJ, "iweights": iweights,
        "n_nodes": nodes.shape[0], "n_elements": elements.shape[0],
    }


def neo_hookean_P_closed_form(F, mu, lam):
    """Closed-form first Piola-Kirchhoff stress for the SAME Neo-Hookean
    potential train_B3.neo_hookean_energy_density_batched uses (mu/2*
    (I1-3-2lnJ) + lam/2*lnJ^2): P = mu*(F - F^-T) + lam*ln(J)*F^-T
    (textbook result, e.g. Bonet & Wood eq. 6.28). Verified against
    autograd dpsi/dF directly (max abs diff 2.8e-14) before being used
    here -- used in its closed form (not nested autograd) so the
    constitutive loss needs only a single backward pass through the
    network, not a double backward.
    F: (...,3,3); mu, lam: (...)."""
    Finv = torch.linalg.inv(F)
    FinvT = Finv.transpose(-1, -2)
    _sign, lnJ = torch.linalg.slogdet(F)
    return mu[..., None, None] * (F - FinvT) + lam[..., None, None] * lnJ[..., None, None] * FinvT


def interpolate_nodal_field_to_quadpts(field_node, elements, N_sf):
    """field_node: (Batch, n_nodes, k). Returns (n_gauss, Batch, n_elem,
    k): plain shape-function interpolation (no B/Jacobian involved --
    N_sf does not depend on element geometry, only on the reference
    element), the same operation train_B1/B2/B3's own FEM-comparison
    code already relies on for any nodal-to-quadrature-point
    interpolation."""
    fe = field_node[:, elements, :]  # (Batch, n_elem, 8_node, k)
    return torch.einsum("gq,beqk->gbek", N_sf, fe)


def gradient_of_nodal_field(field_node, elements, B_op):
    """field_node: (Batch, n_nodes, k). Returns (n_gauss, Batch, n_elem,
    k, 3) = d(field_k)/dX_c at each quadrature point -- the exact same
    einsum pattern train_B3.total_potential_energy_B3 already uses for
    H = grad(u) (there with k=3 fixed; here k is generic, e.g. 9 for
    the flattened P field), applied with an extra leading g loop since
    B_op's n_gauss axis cannot be vectorized away without changing the
    einsum's contracted index (kept this way to mirror train_B3.py's
    own, already-trusted pattern as closely as possible)."""
    fe = field_node[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, k, 8_node)
    n_gauss = B_op.shape[0]
    out = []
    for g in range(n_gauss):
        grad_g = torch.einsum("bedq,ecq->bedc", fe, B_op[g])  # (Batch, n_elem, k, 3)
        out.append(grad_g)
    return torch.stack(out, dim=0)  # (n_gauss, Batch, n_elem, k, 3)


def divergence_from_grad(gradP):
    """gradP: (..., 9, 3), the 9 ordered (i,I) row-major (component
    index = 3*i + I), last axis d/dX_J. Returns Div_i = sum_I
    dP_iI/dX_I, shape (..., 3). Verified: returns exactly zero (to
    1e-13) for a reconstructed spatially-uniform stress field, AND
    matches a known nonzero closed-form divergence (to 1e-13) for a
    field that varies linearly in X -- the second check specifically
    rules out a trivial always-zero bug passing the first."""
    g = gradP.reshape(*gradP.shape[:-2], 3, 3, 3)  # (..., i, I, J)
    return torch.diagonal(g, dim1=-2, dim2=-1).sum(-1)


def _mixed_stress_residual_pass(p_flat, ue, mu_elem, lam_elem, elements, B_op, N_sf, w,
                                 eye, relative_constitutive, constitutive_eps, Batch):
    """One pass of the constitutive+equilibrium residual accumulation over a
    SET of elements at a GIVEN quadrature rule (B_op/N_sf/w already restricted
    to that element subset and that rule -- coarse for non-region elements,
    fine for region elements). Shared by mixed_stress_loss_B3 (one coarse pass
    over every element) and mixed_stress_loss_B3_locally_refined (one coarse
    pass over non-region elements, one fine pass over region elements) so the
    per-quadrature-point math is written and tested exactly once.

    w: (n_gauss, n_elem) already-combined volume weight (detJ*iweight for the
    coarse rule, detJ_fine.abs()*fine_w for the fine rule -- the fine rule's
    detJ can be negative for a left-handed local parametrization, same as any
    other quadrature point, so callers must pass .abs() if their own detJ is
    unsigned-unsafe; the coarse geom['detJ'] used elsewhere in this project is
    passed through as-is, matching mixed_stress_loss_B3's own prior behavior).

    Returns (total_weight, L_constitutive, L_equilibrium), each (Batch,),
    UNNORMALIZED (not yet divided by total_weight) so a caller can add a
    second pass's own totals before normalizing once at the end."""
    n_gauss = B_op.shape[0]
    total_weight = torch.zeros(Batch, device=ue.device, dtype=ue.dtype)
    L_constitutive = torch.zeros(Batch, device=ue.device, dtype=ue.dtype)
    L_equilibrium = torch.zeros(Batch, device=ue.device, dtype=ue.dtype)
    if ue.shape[1] == 0:  # no elements in this pass (e.g. an empty region)
        return total_weight, L_constitutive, L_equilibrium

    P_interp_all = interpolate_nodal_field_to_quadpts(p_flat, elements, N_sf)  # (g,B,e,9)
    pe = p_flat[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, 9, 8_node)

    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])  # (Batch, n_elem, 3, 3)
        F = eye + H
        P_true = neo_hookean_P_closed_form(F, mu_elem, lam_elem)
        P_pred = P_interp_all[g].reshape(Batch, -1, 3, 3)

        wg = w[g]  # (n_elem,)
        total_weight = total_weight + wg.sum().expand(Batch)

        resid_c = ((P_pred - P_true) ** 2).sum(dim=(-1, -2))  # (Batch, n_elem)
        if relative_constitutive:
            P_true_sq = (P_true ** 2).sum(dim=(-1, -2))
            resid_c = resid_c / (constitutive_eps + P_true_sq)
        L_constitutive = L_constitutive + torch.sum(resid_c * wg[None, :], dim=1)

        gradP_g = torch.einsum("bedq,ecq->bedc", pe, B_op[g])  # (Batch, n_elem, 9, 3)
        divP = divergence_from_grad(gradP_g)  # (Batch, n_elem, 3)
        resid_e = (divP ** 2).sum(dim=-1)  # (Batch, n_elem)
        L_equilibrium = L_equilibrium + torch.sum(resid_e * wg[None, :], dim=1)

    return total_weight, L_constitutive, L_equilibrium


def _mixed_stress_residual_pass_vectorized(p_flat, ue, mu_elem, lam_elem, elements, B_op, N_sf, w,
                                            eye, relative_constitutive, constitutive_eps, Batch):
    """Same math as _mixed_stress_residual_pass, but fully VECTORIZED over
    the quadrature-point (g) axis in a single batched einsum, instead of a
    Python loop over g -- required whenever n_gauss is large (the fine
    region quadrature can have 1000+ points per element at n_sub=10).

    REAL BUG FOUND ON GPU, 2026-10-26: the first version of
    mixed_stress_loss_B3_locally_refined used _mixed_stress_residual_pass
    (the coarse pass's own Python-loop-over-g implementation, fine for its
    8 points) for the FINE region pass too -- looping 1000 times per
    training iteration in Python, each a tiny einsum over only ~4 region
    elements. Measured on an A100: 2.52s/iteration vs. the ~0.18s/iteration
    every other mixed-stress run gets -- a ~14x slowdown, matching almost
    exactly the energy-only baseline's own already-documented ~12x
    slowdown bug (evaluate_B3_region_local_refine.precompute_region_fine_
    quadrature's own docstring) from the SAME root cause (an unvectorized
    loop over the fine-point dimension) -- this implementation failed to
    replicate that fix the first time. Fixed here by computing H/gradP for
    ALL gauss points in one batched einsum (adding a leading g axis
    throughout) instead of looping. The coarse pass (_mixed_stress_
    residual_pass, 8 points) is untouched -- already cheap, already proven
    correct and fast on real GPU runs, not the source of this bug."""
    n_elem = ue.shape[1]
    if n_elem == 0:
        z = torch.zeros(Batch, device=ue.device, dtype=ue.dtype)
        return z, z.clone(), z.clone()

    P_interp_all = interpolate_nodal_field_to_quadpts(p_flat, elements, N_sf)  # (g,B,e,9)
    P_pred_all = P_interp_all.reshape(*P_interp_all.shape[:-1], 3, 3)  # (g,B,e,3,3)
    pe = p_flat[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, 9, 8_node)

    H_all = torch.einsum("bedq,gecq->gbedc", ue, B_op)  # (g, Batch, n_elem, 3, 3)
    F_all = eye.unsqueeze(0) + H_all
    P_true_all = neo_hookean_P_closed_form(F_all, mu_elem.unsqueeze(0), lam_elem.unsqueeze(0))

    resid_c_all = ((P_pred_all - P_true_all) ** 2).sum(dim=(-1, -2))  # (g, Batch, n_elem)
    if relative_constitutive:
        P_true_sq_all = (P_true_all ** 2).sum(dim=(-1, -2))
        resid_c_all = resid_c_all / (constitutive_eps + P_true_sq_all)
    L_constitutive = torch.sum(resid_c_all * w[:, None, :], dim=(0, 2))  # (Batch,)

    gradP_all = torch.einsum("bedq,gecq->gbedc", pe, B_op)  # (g, Batch, n_elem, 9, 3)
    divP_all = divergence_from_grad(gradP_all)  # (g, Batch, n_elem, 3)
    resid_e_all = (divP_all ** 2).sum(dim=-1)  # (g, Batch, n_elem)
    L_equilibrium = torch.sum(resid_e_all * w[:, None, :], dim=(0, 2))  # (Batch,)

    total_weight = w.sum().expand(Batch).clone()
    return total_weight, L_constitutive, L_equilibrium


def mixed_stress_loss_B3_locally_refined(
        u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
        region_elem_mask, N_f_region, B_f_region, detJ_f_region, fine_w_t,
        w_constitutive=1.0, w_equilibrium=1.0,
        relative_constitutive=False, constitutive_eps=400.0):
    """Same two residuals as mixed_stress_loss_B3 (constitutive consistency,
    equilibrium), but elements touching the groove (region_elem_mask) have
    BOTH residuals integrated with a FINER quadrature (N_f_region/B_f_region/
    detJ_f_region/fine_w_t, precomputed once by
    precompute_region_fine_quadrature_mixed) instead of the standard 8-point
    rule -- every other element is completely unchanged.

    Mirrors evaluate_B3_region_local_refine.total_potential_energy_B3_
    locally_refined's own pattern, already proven on real GPU training for
    the energy-only baseline (roughly HALVED that formulation's own
    region-stress error, 62.2%->29.7%, 2026-09-30) -- extended here to the
    constitutive AND equilibrium residuals this mixed-stress formulation
    actually trains on, since that formulation has no single "energy" term
    to refine. Each element contributes via EXACTLY ONE of the two
    quadrature rules (fine XOR coarse): verified by
    test_mixed_stress_B3_local_refine.py's identity check (an EMPTY region
    mask must exactly reproduce plain mixed_stress_loss_B3's own result,
    since then every element takes the coarse pass alone)."""
    elements = geom["elements"]
    B_op, N_sf, detJ, iweights = geom["B"], geom["N"], geom["detJ"], geom["iweights"]
    non_region_mask = ~region_elem_mask
    Batch = u_net_raw.shape[0]

    u = apply_dirichlet_b3(u_net_raw, geom, phi)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, 3, 8_node)
    p_flat = p_net_raw.reshape(Batch, p_net_raw.shape[1], 9)

    # Standard coarse quadrature, restricted to non-region elements.
    coarse_w = detJ * iweights[:, None]  # (n_gauss, n_elem), matches mixed_stress_loss_B3's own w
    tw_c, Lc_c, Le_c = _mixed_stress_residual_pass(
        p_flat, ue[:, non_region_mask], mu_elem[:, non_region_mask], lam_elem[:, non_region_mask],
        elements[non_region_mask], B_op[:, non_region_mask], N_sf, coarse_w[:, non_region_mask],
        eye, relative_constitutive, constitutive_eps, Batch)

    # Finer quadrature, restricted to region elements -- VECTORIZED over
    # the (potentially 1000+) fine quadrature points, not a Python loop
    # (see _mixed_stress_residual_pass_vectorized's own docstring for the
    # real ~14x slowdown bug this fixes).
    fine_w = detJ_f_region.abs() * fine_w_t[:, None]  # (n_fine_gauss, n_region_elem)
    tw_f, Lc_f, Le_f = _mixed_stress_residual_pass_vectorized(
        p_flat, ue[:, region_elem_mask], mu_elem[:, region_elem_mask], lam_elem[:, region_elem_mask],
        elements[region_elem_mask], B_f_region, N_f_region, fine_w,
        eye, relative_constitutive, constitutive_eps, Batch)

    total_weight = tw_c + tw_f
    L_constitutive = (Lc_c + Lc_f) / total_weight
    L_equilibrium = (Le_c + Le_f) / total_weight
    loss = w_constitutive * L_constitutive.mean() + w_equilibrium * L_equilibrium.mean()
    return loss, {"constitutive": L_constitutive.mean().item(),
                  "equilibrium": L_equilibrium.mean().item()}


def precompute_region_fine_quadrature_mixed(nodes_t, elements_t, region_elem_mask, fine_xi, fine_w,
                                             device, dtype):
    """Like evaluate_B3_region_local_refine.precompute_region_fine_quadrature,
    but also returns the fine shape-function VALUES N_f_region: the
    energy-only baseline's own precompute only ever needs B/detJ (it
    differentiates u, never interpolates a field), while the mixed-stress
    formulation also needs N to interpolate the network's own predicted
    nodal P field to the fine quadrature points for the constitutive
    residual. Purely geometric (independent of material/displacement/
    network) -- computed ONCE before training starts and reused every
    iteration, never recomputed inside the training loop (the baseline's own
    docstring has the full story of a real ~12x slowdown bug found and fixed
    by this same discipline)."""
    from torchfem import Solid
    from torchfem.materials import Hyperelastic3D
    from omar_pfem.torchfem_comparison import neo_hookean_psi_3d

    region_elements = elements_t[region_elem_mask]  # (n_region_elem, 8)
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        dummy_params = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        material = Hyperelastic3D(psi=neo_hookean_psi_3d, params=dummy_params)
        with torch.device(device):
            model_region = Solid(nodes_t, region_elements, material)
        fine_xi_t = torch.tensor(fine_xi, dtype=dtype, device=device)
        N_f_region, B_f_region, detJ_f_region = model_region.eval_shape_functions(fine_xi_t)
    finally:
        torch.set_default_dtype(old_default_dtype)
    fine_w_t = torch.tensor(fine_w, dtype=dtype, device=device)
    return N_f_region, B_f_region, detJ_f_region, fine_w_t


def mixed_stress_loss_B3(u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
                          w_constitutive=1.0, w_equilibrium=1.0,
                          relative_constitutive=False, constitutive_eps=400.0):
    """u_net_raw: (Batch,N,3) network's raw displacement output (BEFORE
    the Dirichlet ansatz -- apply_dirichlet_b3 is called inside here,
    exactly as train_B3.py's own training loop calls it, so every
    caller gets the same hard-constrained BCs for free).
    p_net_raw: (Batch,N,9) network's raw stress output, one P per node,
    flattened row-major (i,I); UNCONSTRAINED -- B3 has no Neumann
    (traction-free) boundary term in this first version (every boundary
    here is already Dirichlet-constrained on u; a traction-free-surface
    residual, if the groove/lateral surfaces need one, is a candidate
    follow-up extension, not added here to keep this first version's
    correctness easy to verify term-by-term).

    relative_constitutive (added 2026-10-22, see train_B3_mixed_stress.
    py's own docstring, "UPDATE 2026-10-22" for the full diagnosis):
    if True, each quadrature point's constitutive residual is divided
    by (constitutive_eps + |P_true|_F^2) before being volume-weighted
    and summed, instead of being used as a raw absolute squared error.
    Found necessary because the TRUE |P| distribution across B3's
    domain is extremely heavy-tailed (measured directly from real
    sampled E/nu/phi: median ~99, p99 ~801 (8x median), max ~2599 (26x
    median) -- almost certainly the groove region, where strain
    concentrates) -- with a plain absolute L2 residual, the handful of
    high-|P| elements dominate the loss/gradient quadratically more
    than the bulk of the domain, which is a real, measured candidate
    explanation for why `constitutive` oscillated with no visible
    downward trend across three separate real GPU runs (500/250 scale,
    and the two-stage curriculum) despite no implementation bug being
    found in any of them. Default False preserves the exact old
    behavior for every existing caller/checkpoint/test.
    constitutive_eps: a floor on the per-point |P_true|_F^2 used as the
    relative-error denominator, so very-low-stress quadrature points
    (measured min |P|_F ~0.21, far below the 400=20^2 floor) are not
    divided by a near-zero number and do not themselves come to
    dominate the loss the opposite way. 400.0 = (~20)^2, close to the
    measured 10th-percentile |P|_F (~20.5) -- i.e. quadrature points at
    or below the lower ~10% of the real |P| distribution are treated
    with an effectively-absolute (not blown-up) residual, while the
    heavy right tail that caused the original problem is properly
    relative.

    Returns (loss, parts) where parts is a dict of the two un-weighted
    residual means, for logging/diagnosis independent of the chosen
    weights."""
    elements = geom["elements"]
    B_op, N_sf, detJ, iweights = geom["B"], geom["N"], geom["detJ"], geom["iweights"]
    Batch = u_net_raw.shape[0]

    u = apply_dirichlet_b3(u_net_raw, geom, phi)

    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, 3, 8_node)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).view(1, 1, 3, 3)

    p_flat = p_net_raw.reshape(Batch, p_net_raw.shape[1], 9)
    P_interp_all = interpolate_nodal_field_to_quadpts(p_flat, elements, N_sf)  # (g,B,e,9)
    gradP_all = gradient_of_nodal_field(p_flat, elements, B_op)  # (g,B,e,9,3)

    n_gauss = B_op.shape[0]
    total_weight = torch.zeros(Batch, device=u.device, dtype=u.dtype)
    L_constitutive = torch.zeros(Batch, device=u.device, dtype=u.dtype)
    L_equilibrium = torch.zeros(Batch, device=u.device, dtype=u.dtype)
    for g in range(n_gauss):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])  # (Batch, n_elem, 3, 3)
        F = eye + H
        P_true = neo_hookean_P_closed_form(F, mu_elem, lam_elem)  # (Batch, n_elem, 3, 3)
        P_pred = P_interp_all[g].reshape(Batch, -1, 3, 3)

        w = detJ[g] * iweights[g]  # (n_elem,), broadcasts over the batch dim below
        total_weight = total_weight + w.sum().expand(Batch)

        resid_c = ((P_pred - P_true) ** 2).sum(dim=(-1, -2))  # (Batch, n_elem)
        if relative_constitutive:
            P_true_sq = (P_true ** 2).sum(dim=(-1, -2))  # (Batch, n_elem)
            resid_c = resid_c / (constitutive_eps + P_true_sq)
        L_constitutive = L_constitutive + torch.sum(resid_c * w[None, :], dim=1)

        divP = divergence_from_grad(gradP_all[g])  # (Batch, n_elem, 3)
        resid_e = (divP ** 2).sum(dim=-1)  # (Batch, n_elem)
        L_equilibrium = L_equilibrium + torch.sum(resid_e * w[None, :], dim=1)

    L_constitutive = L_constitutive / total_weight
    L_equilibrium = L_equilibrium / total_weight
    loss = w_constitutive * L_constitutive.mean() + w_equilibrium * L_equilibrium.mean()
    return loss, {"constitutive": L_constitutive.mean().item(),
                  "equilibrium": L_equilibrium.mean().item()}
