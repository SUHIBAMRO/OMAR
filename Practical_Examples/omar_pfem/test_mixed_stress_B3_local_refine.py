"""Correctness checks for mixed_stress_B3.mixed_stress_loss_B3_locally_
refined -- item 1 attempt 5 (2026-10-26): reusing the already-GPU-proven
local-integration-refinement technique (it roughly HALVED the energy-only
baseline's own region-stress error, 62.2% -> 29.7%, see
evaluate_B3_region_local_refine.py and PROJECT_STATUS.md's 2026-09-30
entry) for the mixed-stress formulation's own constitutive and
equilibrium residuals, which the energy-only version never needed to
refine (it only ever integrates a single scalar energy density, not a
per-node stress field that must also be gradient'd and interpolated).

All run on CPU, no GPU needed:

    python -m omar_pfem.test_mixed_stress_B3_local_refine
"""
import torch

from omar_pfem.mixed_stress_B3 import (
    build_fixed_geometry_mixed, mixed_stress_loss_B3, mixed_stress_loss_B3_locally_refined,
    precompute_region_fine_quadrature_mixed,
)
from omar_pfem.evaluate_B3_region_local_refine import fine_quadrature_3d

torch.set_default_dtype(torch.float64)


def _small_geom():
    return build_fixed_geometry_mixed(Ntheta=5, Nr=4, Nz=4, device='cpu', dtype=torch.float64)


def _random_batch(geom, Batch=2, seed=0):
    torch.manual_seed(seed)
    n_nodes = geom['n_nodes']
    E_node = (1000.0 + 50.0 * torch.randn(Batch, n_nodes)).abs() + 500.0
    nu_node = torch.full((Batch, n_nodes), 0.45)
    phi = torch.tensor([0.05, 0.07][:Batch])
    u_net_raw = 0.1 * torch.randn(Batch, n_nodes, 3)
    p_net_raw = 10.0 * torch.randn(Batch, n_nodes, 9)
    return E_node, nu_node, phi, u_net_raw, p_net_raw


def test_empty_region_matches_plain_loss_exactly():
    """An EMPTY region mask must exactly reproduce mixed_stress_loss_B3's
    own plain result, since then every element takes the coarse pass alone
    and the fine-pass code path contributes nothing -- confirms the
    fine/coarse split introduces no contamination when there is nothing to
    refine. Dummy zero-element fine-quadrature tensors are built directly
    (not via precompute_region_fine_quadrature_mixed, which would need to
    construct a 0-element torch-fem Solid -- an unrelated edge case this
    test is not trying to exercise)."""
    geom = _small_geom()
    E_node, nu_node, phi, u_net_raw, p_net_raw = _random_batch(geom)
    n_elements, dtype = geom['n_elements'], geom['nodes'].dtype
    region_elem_mask = torch.zeros(n_elements, dtype=torch.bool)

    n_fine = 8  # arbitrary, unused since n_region_elem=0
    N_f_region = torch.zeros(n_fine, 8, dtype=dtype)
    B_f_region = torch.zeros(n_fine, 0, 3, 8, dtype=dtype)
    detJ_f_region = torch.zeros(n_fine, 0, dtype=dtype)
    fine_w_t = torch.ones(n_fine, dtype=dtype)

    loss_plain, parts_plain = mixed_stress_loss_B3(u_net_raw, p_net_raw, E_node, nu_node, phi, geom)
    loss_refined, parts_refined = mixed_stress_loss_B3_locally_refined(
        u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
        region_elem_mask, N_f_region, B_f_region, detJ_f_region, fine_w_t)

    err_loss = abs(loss_plain.item() - loss_refined.item())
    err_c = abs(parts_plain['constitutive'] - parts_refined['constitutive'])
    err_e = abs(parts_plain['equilibrium'] - parts_refined['equilibrium'])
    assert err_loss < 1e-10, f'empty-region refined loss does not match plain loss (err={err_loss})'
    assert err_c < 1e-10 and err_e < 1e-10, \
        f'empty-region refined parts do not match plain parts (err_c={err_c}, err_e={err_e})'
    print(f'PASS  empty region: locally-refined loss exactly matches plain mixed_stress_loss_B3 '
          f'(loss err={err_loss:.2e}, constitutive err={err_c:.2e}, equilibrium err={err_e:.2e})')


def test_nonempty_region_at_matching_degree_matches_plain_loss():
    """A NON-EMPTY region, refined at n_sub=2 (the SAME degree as the
    standard coarse 2-point-per-axis rule -- fine_quadrature_3d's own
    docstring confirms n_sub=2 reproduces the default rule exactly), must
    still exactly reproduce mixed_stress_loss_B3's own plain result: this
    is the real test that the SPLIT (non-region coarse + region fine) and
    RECOMBINE (single total_weight normalization) are both done correctly,
    not just that an empty region is a no-op. Mirrors the energy-only
    baseline's own identity check in evaluate_B3_region_local_refine.py's
    module docstring ("refining to the SAME degree as the original 2-point
    rule reproduces the original coarse computation")."""
    geom = _small_geom()
    E_node, nu_node, phi, u_net_raw, p_net_raw = _random_batch(geom, seed=7)
    n_elements = geom['n_elements']
    region_elem_mask = torch.zeros(n_elements, dtype=torch.bool)
    region_elem_mask[:3] = True  # arbitrary handful of "region" elements

    fine_xi, fine_w = fine_quadrature_3d(n_sub=2)
    N_f_region, B_f_region, detJ_f_region, fine_w_t = precompute_region_fine_quadrature_mixed(
        geom['nodes'], geom['elements'], region_elem_mask, fine_xi, fine_w,
        device='cpu', dtype=geom['nodes'].dtype)

    loss_plain, parts_plain = mixed_stress_loss_B3(u_net_raw, p_net_raw, E_node, nu_node, phi, geom)
    loss_refined, parts_refined = mixed_stress_loss_B3_locally_refined(
        u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
        region_elem_mask, N_f_region, B_f_region, detJ_f_region, fine_w_t)

    err_loss = abs(loss_plain.item() - loss_refined.item())
    err_c = abs(parts_plain['constitutive'] - parts_refined['constitutive'])
    err_e = abs(parts_plain['equilibrium'] - parts_refined['equilibrium'])
    assert err_loss < 1e-8, f'matching-degree refined loss does not match plain loss (err={err_loss})'
    assert err_c < 1e-8 and err_e < 1e-8, \
        f'matching-degree refined parts do not match plain parts (err_c={err_c}, err_e={err_e})'
    print(f'PASS  non-empty region (3 elements) at matching n_sub=2: locally-refined loss exactly '
          f'matches plain mixed_stress_loss_B3 (loss err={err_loss:.2e}, constitutive '
          f'err={err_c:.2e}, equilibrium err={err_e:.2e})')


def test_finer_region_quadrature_changes_the_loss_and_stays_differentiable():
    """A genuinely FINER quadrature (n_sub=6, far above the coarse 2-point
    rule) in the region must (a) produce a DIFFERENT loss value than the
    plain coarse-everywhere computation (confirming the refinement actually
    does something, not a silent no-op) and (b) remain finite and
    differentiable through both network outputs -- the two properties
    train_B3_mixed_stress.py's training loop actually depends on."""
    geom = _small_geom()
    E_node, nu_node, phi, u_net_raw, p_net_raw = _random_batch(geom, seed=11)
    u_net_raw = u_net_raw.requires_grad_(True)
    p_net_raw = p_net_raw.requires_grad_(True)
    n_elements = geom['n_elements']
    region_elem_mask = torch.zeros(n_elements, dtype=torch.bool)
    region_elem_mask[:3] = True

    fine_xi, fine_w = fine_quadrature_3d(n_sub=6)
    N_f_region, B_f_region, detJ_f_region, fine_w_t = precompute_region_fine_quadrature_mixed(
        geom['nodes'], geom['elements'], region_elem_mask, fine_xi, fine_w,
        device='cpu', dtype=geom['nodes'].dtype)

    loss_plain, _ = mixed_stress_loss_B3(
        u_net_raw.detach(), p_net_raw.detach(), E_node, nu_node, phi, geom)
    loss_refined, parts_refined = mixed_stress_loss_B3_locally_refined(
        u_net_raw, p_net_raw, E_node, nu_node, phi, geom,
        region_elem_mask, N_f_region, B_f_region, detJ_f_region, fine_w_t)

    assert torch.isfinite(loss_refined) and loss_refined.item() > 0
    assert abs(loss_plain.item() - loss_refined.item()) > 1e-6, \
        'a genuinely finer region quadrature (n_sub=6) should change the loss value'
    loss_refined.backward()
    assert u_net_raw.grad is not None and torch.isfinite(u_net_raw.grad).all() \
        and u_net_raw.grad.abs().sum() > 0, 'no/invalid/zero grad on u_net_raw'
    assert p_net_raw.grad is not None and torch.isfinite(p_net_raw.grad).all() \
        and p_net_raw.grad.abs().sum() > 0, 'no/invalid/zero grad on p_net_raw'
    print(f'PASS  n_sub=6 region refinement changes the loss (plain={loss_plain.item():.6f}, '
          f'refined={loss_refined.item():.6f}) and stays finite/differentiable '
          f'(parts={parts_refined})')


if __name__ == '__main__':
    test_empty_region_matches_plain_loss_exactly()
    test_nonempty_region_at_matching_degree_matches_plain_loss()
    test_finer_region_quadrature_changes_the_loss_and_stays_differentiable()
    print('\nall mixed-stress-B3 local-refine checks passed')
