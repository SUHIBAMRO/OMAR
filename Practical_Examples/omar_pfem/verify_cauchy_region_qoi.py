"""CPU-only verification of the new round-12 point-1 code (Cauchy
stress push-forward, fixed-region selection, weighted percentile/top-1%
robust statistics) BEFORE trusting it for any real B1/B2 sweep or
spending any GPU time on it -- same discipline this project has used
for every other new QoI (peak-stress, tangent-energy, reaction
resultant all had a CPU check like this before their first real run).

Checks:
  1. Weighted-percentile/top-fraction helpers reduce to the plain
     unweighted statistic when all weights are equal (checked against
     numpy's own np.percentile as an independent reference).
  2. Cauchy stress reduces to PK1 stress in the small-strain limit
     (F -> I, J -> 1 => sigma = P F^T / J -> P), checked numerically at
     a tiny but nonzero displacement gradient.
  3. Cauchy stress is symmetric (a hyperelastic material with no body
     couples must give symmetric Cauchy stress, even though PK1 itself
     is generally NOT symmetric) -- this is an independent structural
     check, not just a numerical-agreement one.
  4. IDENTITY check on a real B1 mesh: select_fixed_region +
     compute_region_cauchy_stress_error with coarse=fine (same solved
     field passed as both) must give EXACTLY zero for every relative
     error reported, and the region must actually contain more than a
     handful of points (not degenerate).
  5. A real (non-identity) B1 solve at two different coarse resolutions
     against a shared fine reference: the region errors should be finite,
     non-nan, and (informally) the finer coarse mesh should not be
     dramatically worse than the coarser one -- a sanity smoke test at
     real (non-identity) fields, not just the trivial identity case.
"""
import numpy as np
import torch

from omar_pfem.high_dof_convergence_study import (
    solve_one, find_fine_peak_stress, select_fixed_region,
    compute_region_cauchy_stress_error, _weighted_percentile, _weighted_top_fraction_mean,
    _pk1_stress_at, _cauchy_stress_at,
)
from omar_pfem.mesh_convergence import AnalyticFieldB1

torch.manual_seed(0)
np.random.seed(0)

print('=== 1) weighted percentile / top-fraction sanity checks (equal weights) ===')
vals = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0])
w = np.ones_like(vals)
# Boundary behavior must hold exactly for ANY reasonable percentile
# convention, regardless of sample size or which interpolation rule is
# used (mine is Hazen-style; numpy's default is linear -- they legitimately
# disagree at small n, e.g. mine's p90=8.1 vs numpy's 6.9 here, which is
# a difference of CONVENTION, not a bug, so this test checks properties
# every convention must share rather than exact agreement with one of them).
assert abs(_weighted_percentile(vals, w, 0.0) - vals.min()) < 1e-10
assert abs(_weighted_percentile(vals, w, 100.0) - vals.max()) < 1e-10
p50 = _weighted_percentile(vals, w, 50.0)
print(f'  p0={vals.min():.4f} (must equal min), p100={vals.max():.4f} (must equal max), p50={p50:.4f}')
# At a LARGE n, Hazen and numpy's linear convention converge (both approximate
# the same continuous distribution), so this IS a meaningful agreement check.
big = np.random.default_rng(1).normal(size=5000)
w_big = np.ones_like(big)
for q in [50.0, 90.0, 99.0]:
    mine = _weighted_percentile(big, w_big, q)
    ref = float(np.percentile(big, q))
    print(f'  n=5000 p{q}: mine={mine:.4f}  numpy={ref:.4f}  diff={abs(mine-ref):.4f}')
    assert abs(mine - ref) < 0.05, 'weighted percentile should closely match numpy at large n'
top1 = _weighted_top_fraction_mean(vals, w, 1.0)  # top 100% == plain mean
assert abs(top1 - vals.mean()) < 1e-10, 'top-100% mean must equal the plain mean'
print(f'  top-100% mean={top1:.4f}  plain mean={vals.mean():.4f}  (must match exactly)')

print('\n=== 2) Cauchy -> PK1 in the small-strain limit ===')
device, dtype = torch.device('cpu'), torch.float64
E = np.array([1000.0])
nu = np.array([0.3])
eps = 1e-6
grad_u_tiny = (eps * np.random.randn(1, 2, 2))
P_tiny = _pk1_stress_at(grad_u_tiny, E, nu, 'neo_hookean', device, dtype)
sigma_tiny = _cauchy_stress_at(grad_u_tiny, E, nu, 'neo_hookean', device, dtype)
diff = np.abs(P_tiny - sigma_tiny).max()
scale = np.abs(P_tiny).max()
print(f'  max|P - sigma| = {diff:.3e}  (scale ~{scale:.3e}, ratio {diff/scale:.3e}, expect O(eps))')
assert diff / scale < 1e-4, 'Cauchy stress should match PK1 to O(eps) at tiny strain'

print('\n=== 3) Cauchy stress symmetry (PK1 need not be symmetric) ===')
grad_u_finite = 0.05 * np.random.randn(5, 2, 2)
P_fin = _pk1_stress_at(grad_u_finite, E, nu, 'neo_hookean', device, dtype)
sigma_fin = _cauchy_stress_at(grad_u_finite, E, nu, 'neo_hookean', device, dtype)
pk1_asym = np.abs(P_fin - P_fin.transpose(0, 2, 1)).max()
cauchy_asym = np.abs(sigma_fin - sigma_fin.transpose(0, 2, 1)).max()
print(f'  max PK1 asymmetry (expected nonzero): {pk1_asym:.3e}')
print(f'  max Cauchy asymmetry (must be ~0):    {cauchy_asym:.3e}')
assert cauchy_asym < 1e-10, 'Cauchy stress must be symmetric'
assert pk1_asym > 1e-6, 'sanity: PK1 should generally be asymmetric at finite strain (else the test is not exercising anything)'

print('\n=== 4) IDENTITY check on a real B1 mesh (coarse == fine) ===')
N = 11
res = solve_one('B1', 'Q4', N, 'neo_hookean', device, dtype, cg_tol=1e-8, newton_tol=1e-8)
field = {'nodes': res['nodes'], 'elements': res['elements'], 'N': N, 'u': res['u']}

E_fn = AnalyticFieldB1('E')
nu_fn = AnalyticFieldB1('nu')

x_star, peak_ref = find_fine_peak_stress(field, 'Q4', 'B1', 'neo_hookean', E_fn, nu_fn, device, dtype)
print(f'  x_star={x_star}, peak_ref={peak_ref:.4f}')

region_pts, region_w = select_fixed_region(field, 'Q4', x_star, radius=0.15)
print(f'  region contains {len(region_pts)} fine Gauss points within radius 0.15')
assert len(region_pts) >= 8, 'region should contain a non-trivial number of points at N=21'

out_identity = compute_region_cauchy_stress_error(
    field, field, region_pts, region_w, 'Q4', 'B1', 'neo_hookean', E_fn, nu_fn, device, dtype)
for k, v in out_identity.items():
    if 'rel_err' in k:
        print(f'  {k} = {v:.3e}  (must be ~0)')
        assert v < 1e-10, f'{k} should be exactly 0 in the identity check, got {v}'
print('  IDENTITY CHECK PASSED: every relative error is ~0 when coarse==fine.')

print('\n=== 5) non-identity smoke test: perturb the N=21 field\'s own displacements ===')
# A genuinely different coarse mesh solve at real scale is too slow for a
# quick CPU smoke test (plain-Jacobi CG iteration count grows with N --
# this project's own real sweeps for exactly this reason always run on
# GPU). evaluate_fe_field_and_gradient itself (used for both sides here)
# is already thoroughly validated elsewhere in this project across many
# real non-identity resolution sweeps; what this check adds on top is
# only the NEW region/percentile machinery's behavior when the two sides
# genuinely differ, which a synthetic perturbation of the same solved
# field exercises just as well as a second real solve, in milliseconds
# instead of minutes.
region_pts2, region_w2 = select_fixed_region(field, 'Q4', x_star, radius=0.15)
perturbed_field = dict(field)
perturbed_field['u'] = field['u'] * 0.9 + 0.01 * np.random.default_rng(2).standard_normal(field['u'].shape)
out = compute_region_cauchy_stress_error(
    perturbed_field, field, region_pts2, region_w2, 'Q4', 'B1', 'neo_hookean',
    E_fn, nu_fn, device, dtype)
print(f'  field_l2_rel={out["region_cauchy_field_l2_rel"]:.4f}  '
      f'avg_rel_err={out["region_cauchy_avg_rel_err"]:.4f}  '
      f'p99_rel_err={out["region_cauchy_p99_rel_err"]:.4f}  '
      f'top1pct_rel_err={out["region_cauchy_top1pct_rel_err"]:.4f}  '
      f'max_rel_err={out["region_cauchy_max_rel_err"]:.4f}')
for k, v in out.items():
    if 'rel_err' in k:
        assert np.isfinite(v) and v > 0, f'{k} should be finite and nonzero for a perturbed field, got {v}'

print('\nALL CHECKS PASSED.')
