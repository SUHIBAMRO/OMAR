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


def mixed_stress_loss_B3(u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
                          w_constitutive=1.0, w_equilibrium=1.0):
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
        L_constitutive = L_constitutive + torch.sum(resid_c * w[None, :], dim=1)

        divP = divergence_from_grad(gradP_all[g])  # (Batch, n_elem, 3)
        resid_e = (divP ** 2).sum(dim=-1)  # (Batch, n_elem)
        L_equilibrium = L_equilibrium + torch.sum(resid_e * w[None, :], dim=1)

    L_constitutive = L_constitutive / total_weight
    L_equilibrium = L_equilibrium / total_weight
    loss = w_constitutive * L_constitutive.mean() + w_equilibrium * L_equilibrium.mean()
    return loss, {"constitutive": L_constitutive.mean().item(),
                  "equilibrium": L_equilibrium.mean().item()}
