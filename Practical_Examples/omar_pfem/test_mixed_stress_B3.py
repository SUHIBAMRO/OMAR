"""Correctness checks for mixed_stress_B3.py (Professor Rabczuk's item 1
of the 2026-10-16 follow-up-paper email: a mixed displacement-stress
formulation for B3). All run on CPU, no GPU needed -- these only have
to be right once, since the operators they check are reused for every
future mixed-stress training run.

    python -m omar_pfem.test_mixed_stress_B3
"""
import torch

from omar_pfem.mixed_stress_B3 import (
    build_fixed_geometry_mixed, divergence_from_grad, gradient_of_nodal_field,
    interpolate_nodal_field_to_quadpts, mixed_stress_loss_B3, neo_hookean_P_closed_form,
)
from omar_pfem.train_B3 import apply_dirichlet_b3

torch.set_default_dtype(torch.float64)


def _neo_hookean_psi_scalar(F, mu, lam):
    _sign, lnJ = torch.linalg.slogdet(F)
    I1 = torch.sum(F ** 2)
    return (mu / 2.0) * (I1 - 3.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)


def test_constitutive_matches_autograd():
    """Closed-form Neo-Hookean P = mu*(F-F^-T) + lam*ln(J)*F^-T against
    autograd dpsi/dF, at a generic (non-degenerate) F."""
    torch.manual_seed(0)
    mu, lam = torch.tensor(400.0), torch.tensor(1800.0)
    F0 = (torch.eye(3) + 0.15 * torch.randn(3, 3)).requires_grad_(True)
    psi0 = _neo_hookean_psi_scalar(F0, mu, lam)
    (P_autograd,) = torch.autograd.grad(psi0, F0)
    P_closed = neo_hookean_P_closed_form(F0.detach().unsqueeze(0), mu.unsqueeze(0), lam.unsqueeze(0))[0]
    err = (P_autograd - P_closed).abs().max().item()
    assert err < 1e-10, f'closed-form P does not match autograd dpsi/dF (err={err})'
    print(f'PASS  closed-form P matches autograd dpsi/dF (max abs diff {err:.2e})')


def _small_geom():
    return build_fixed_geometry_mixed(Ntheta=5, Nr=4, Nz=4, device='cpu', dtype=torch.float64)


def test_divergence_zero_for_uniform_field():
    """A nodal stress field reconstructed from a single constant P0
    (no spatial variation) must have exactly zero divergence."""
    geom = _small_geom()
    mu, lam = torch.tensor(400.0), torch.tensor(1800.0)
    F0 = torch.eye(3) + 0.15 * torch.randn(3, 3)
    P0 = neo_hookean_P_closed_form(F0.unsqueeze(0), mu.unsqueeze(0), lam.unsqueeze(0))[0]
    P0_nodal = P0.reshape(-1).unsqueeze(0).expand(geom['n_nodes'], 9).contiguous().unsqueeze(0)
    gradP = gradient_of_nodal_field(P0_nodal, geom['elements'], geom['B'])
    divP = divergence_from_grad(gradP)
    err = divP.abs().max().item()
    assert err < 1e-8, f'Div of a uniform stress field is not ~0 (err={err})'
    print(f'PASS  Div(uniform P) ~ 0 (max abs {err:.2e})')


def test_divergence_matches_known_nonzero_value():
    """A stress field that varies LINEARLY in X (exactly representable
    by trilinear shape functions) has a known, generally nonzero,
    constant divergence -- rules out a trivial always-zero bug."""
    torch.manual_seed(1)
    geom = _small_geom()
    nodes_t = geom['nodes']
    mu, lam = torch.tensor(400.0), torch.tensor(1800.0)
    F0 = torch.eye(3) + 0.15 * torch.randn(3, 3)
    P0 = neo_hookean_P_closed_form(F0.unsqueeze(0), mu.unsqueeze(0), lam.unsqueeze(0))[0]
    c = 0.1 * torch.randn(3, 3, 3)  # c[i,I,J]
    true_div = torch.einsum('iII->i', c)
    P_lin_nodal = torch.stack(
        [P0 + torch.einsum('iIJ,J->iI', c, nodes_t[n]) for n in range(geom['n_nodes'])], dim=0
    ).reshape(1, geom['n_nodes'], 9)
    gradP = gradient_of_nodal_field(P_lin_nodal, geom['elements'], geom['B'])
    divP = divergence_from_grad(gradP)
    err = (divP - true_div.reshape(1, 1, 1, 3)).abs().max().item()
    assert err < 1e-8, f'Div does not match the known nonzero closed-form value (err={err})'
    print(f'PASS  Div(linear-in-X P) matches the known nonzero closed form (max abs diff {err:.2e})')


def test_affine_field_end_to_end():
    """A globally affine displacement field u(X)=G@X makes F (hence the
    true P) constant everywhere; setting every node's predicted P to
    that same constant must make the shape-function-interpolated P
    match the gradient-derived true P exactly, confirming the gradient
    (B) and interpolation (N) operators are consistent with each
    other on the same mesh."""
    torch.manual_seed(2)
    geom = _small_geom()
    nodes_t, elements, B_op, N_sf = geom['nodes'], geom['elements'], geom['B'], geom['N']
    n_elem = geom['n_elements']
    mu, lam = torch.tensor(400.0), torch.tensor(1800.0)

    G = 0.08 * torch.randn(3, 3)
    u_nodal = torch.einsum('ij,nj->ni', G, nodes_t).unsqueeze(0)
    H = gradient_of_nodal_field(u_nodal, elements, B_op)  # (8,1,n_elem,3,3)... flattened k=3
    F = torch.eye(3).reshape(1, 1, 1, 3, 3) + H.reshape(8, 1, n_elem, 3, 3)

    mu_t = torch.full((8, 1, n_elem), float(mu))
    lam_t = torch.full((8, 1, n_elem), float(lam))
    P_true = neo_hookean_P_closed_form(F, mu_t, lam_t)

    P0 = neo_hookean_P_closed_form((torch.eye(3) + G).unsqueeze(0), mu.unsqueeze(0), lam.unsqueeze(0))[0]
    P0_nodal = P0.reshape(-1).unsqueeze(0).expand(geom['n_nodes'], 9).unsqueeze(0).contiguous()
    P_pred = interpolate_nodal_field_to_quadpts(P0_nodal, elements, N_sf).reshape(8, 1, n_elem, 3, 3)

    err_F = (F - (torch.eye(3) + G).reshape(1, 1, 1, 3, 3)).abs().max().item()
    err_P = (P_pred - P_true).abs().max().item()
    assert err_F < 1e-10, f'grad operator does not recover the constant F=I+G (err={err_F})'
    assert err_P < 1e-10, f'interpolation does not match the gradient-derived P (err={err_P})'
    print(f'PASS  affine field: grad op exact ({err_F:.2e}), interpolation matches ({err_P:.2e})')


def test_loss_is_finite_and_differentiable():
    """Full mixed_stress_loss_B3 wrapper (real B3 Dirichlet BCs via
    apply_dirichlet_b3): finite, positive loss; gradients reach both
    the raw displacement and raw stress network outputs."""
    torch.manual_seed(3)
    geom = _small_geom()
    n_nodes = geom['n_nodes']
    Batch = 2
    E_node = (1000.0 + 50.0 * torch.randn(Batch, n_nodes)).abs() + 500.0
    nu_node = torch.full((Batch, n_nodes), 0.45)
    phi = torch.tensor([0.05, 0.07])
    u_net_raw = (0.1 * torch.randn(Batch, n_nodes, 3)).requires_grad_(True)
    p_net_raw = (10.0 * torch.randn(Batch, n_nodes, 9)).requires_grad_(True)

    loss, parts = mixed_stress_loss_B3(u_net_raw, p_net_raw, E_node, nu_node, phi, geom)
    assert torch.isfinite(loss) and loss.item() > 0, 'loss is not finite/positive'
    loss.backward()
    assert u_net_raw.grad is not None and torch.isfinite(u_net_raw.grad).all() \
        and u_net_raw.grad.abs().sum() > 0, 'no/invalid/zero grad on u_net_raw'
    assert p_net_raw.grad is not None and torch.isfinite(p_net_raw.grad).all() \
        and p_net_raw.grad.abs().sum() > 0, 'no/invalid/zero grad on p_net_raw'
    print(f'PASS  loss={loss.item():.3f} parts={parts}, gradients reach both outputs')


if __name__ == '__main__':
    test_constitutive_matches_autograd()
    test_divergence_zero_for_uniform_field()
    test_divergence_matches_known_nonzero_value()
    test_affine_field_end_to_end()
    test_loss_is_finite_and_differentiable()
    print('\nall mixed-stress-B3 checks passed')
