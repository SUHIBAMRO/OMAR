"""Weak-form equilibrium residual for B3, added to the existing u-only
Deep Energy Method loss (train_B3.py) -- Professor Rabczuk's items 2
and 3 (2026-10-16 email): "an equilibrium-augmented displacement
formulation, adding equilibrium and traction residuals to the present
energy loss" and "a more general H1/gradient or weak-form term that
constrains displacement gradients more directly."

A real technical finding, not a guess, worked out before writing any
code: for a u-ONLY network (no separate predicted stress field, unlike
mixed_stress_B3.py's item 1), a STRONG-form pointwise residual
Div_0(P)=0 is not meaningful to add as written. B3's mesh uses
trilinear (Hexa1) elements, whose shape-function gradients are CONSTANT
within each element -- so F, and hence P=dpsi/dF, are themselves
constant within each element, and the within-element divergence of ANY
piecewise-constant field is identically zero regardless of whether the
field is physically correct (confirmed directly: this is exactly the
same "uniform field -> zero divergence" identity mixed_stress_B3.py's
own test suite uses as a correctness check for an INDEPENDENTLY
predicted P field -- here it is a triviality, not a check, because P is
not independent). This is the standard reason finite-element practice
uses the WEAK form (virtual work) rather than a strong-form residual
for low-order elements.

The weak form is therefore the one well-posed way to realize BOTH of
Professor Rabczuk's items 2 and 3 for a u-only network -- they collapse
to the same mechanism here, which is itself worth recording rather than
building two superficially different but mathematically coincident
losses:

  integral_Omega0  P : grad_0(v)  dV  =  0   for every admissible
  virtual displacement v (v=0 wherever u itself is Dirichlet-
  constrained; B3's groove/lateral surfaces are genuinely traction-free,
  so the right-hand side is exactly zero there too, not merely assumed).

Random virtual fields v are constructed by reusing train_B3.apply_
dirichlet_b3's OWN boundary-vanishing shape functions (the radial ramp
t*(1-t), zero at both t=0/1; the sin(theta) factor zeroing v_y at the
symmetry planes) multiplied by fresh per-node random noise each call --
so every v is automatically admissible (zero on every Dirichlet/
symmetry boundary) without any new BC-handling code, reusing the exact
mechanism already trusted for u itself.

This residual is NOT already implied to be zero merely because the
network minimizes total potential energy: that minimization is only
exact in the infinite-capacity/fully-converged limit; at any real,
finite checkpoint there is a genuine nonzero gap, and explicitly
penalizing it is a legitimate auxiliary-loss practice (extra, more
direct gradient signal toward the same stationarity condition the
energy loss's own gradient already points to, not a different target).
"""
import torch

from omar_pfem.mixed_stress_B3 import neo_hookean_P_closed_form


def sample_virtual_fields(n_virtual, geom, dtype, device, generator=None):
    """n_virtual independent admissible virtual displacement fields,
    shape (n_virtual, n_nodes, 3) -- zero on every Dirichlet/symmetry
    boundary of B3 by construction, reusing apply_dirichlet_b3's own
    boundary-vanishing shape functions (ramp, sin(theta))."""
    t = geom["t_node"]
    theta = geom["theta_node"]
    ramp = (t * (1.0 - t))[None, :]  # (1, n_nodes), zero at t=0 and t=1

    noise = torch.randn(n_virtual, geom["n_nodes"], 3, dtype=dtype, device=device,
                        generator=generator)
    vx = ramp * noise[:, :, 0]
    vy = ramp * torch.sin(theta)[None, :] * noise[:, :, 1]
    vz = ramp * noise[:, :, 2]
    return torch.stack([vx, vy, vz], dim=2)  # (n_virtual, n_nodes, 3)


def weak_form_equilibrium_residual(u, E_node, nu_node, geom, n_virtual=4, generator=None):
    """Mean squared weak-form residual, averaged over n_virtual random
    admissible test fields and over the batch. u: (Batch,N,3), already
    Dirichlet-satisfying (apply_dirichlet_b3's own output).

    integral P:grad(v) dV is assembled the same per-Gauss way
    total_potential_energy_B3 assembles the energy integral -- same
    B_op/detJ/iweights, same mu/lam-from-E/nu construction, no new
    geometric machinery."""
    elements = geom["elements"]
    B_op, detJ, iweights = geom["B"], geom["detJ"], geom["iweights"]
    Batch = u.shape[0]
    dtype, device = u.dtype, u.device

    v = sample_virtual_fields(n_virtual, geom, dtype, device, generator=generator)  # (K,N,3)
    ve = v[:, elements, :].permute(0, 1, 3, 2)  # (K, n_elem, 3, 8)

    ue = u[:, elements, :].permute(0, 1, 3, 2)  # (Batch, n_elem, 3, 8)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)

    K = v.shape[0]
    total = torch.zeros(Batch, K, device=device, dtype=dtype)
    for g in range(B_op.shape[0]):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])        # (Batch, n_elem, 3, 3)
        F = eye + H
        P = neo_hookean_P_closed_form(F, mu_elem, lam_elem)     # (Batch, n_elem, 3, 3)
        grad_v = torch.einsum("kedq,ecq->kedc", ve, B_op[g])    # (K, n_elem, 3, 3)
        w = (detJ[g] * iweights[g])[None, None, :]              # (1, 1, n_elem)
        contrib = torch.einsum("bexy,kexy->bke", P, grad_v) * w  # (Batch, K, n_elem)
        total = total + contrib.sum(dim=2)
    return (total ** 2).mean()


def total_potential_energy_B3_equilibrium_augmented(u, E_node, nu_node, geom,
                                                      total_potential_energy_B3,
                                                      w_equilibrium=1.0, n_virtual=4,
                                                      generator=None):
    """Augmented loss: the SAME total_potential_energy_B3 (imported by
    the caller, not re-derived here, to avoid two copies of the energy
    formula) plus w_equilibrium times the weak-form residual above.
    Returns (loss, parts) for logging, same convention as
    mixed_stress_B3.mixed_stress_loss_B3."""
    U = total_potential_energy_B3(u, E_node, nu_node, geom)
    eq_resid = weak_form_equilibrium_residual(u, E_node, nu_node, geom,
                                               n_virtual=n_virtual, generator=generator)
    loss = U.mean() + w_equilibrium * eq_resid
    return loss, {"energy": U.mean().item(), "weak_form_residual": eq_resid.item()}
