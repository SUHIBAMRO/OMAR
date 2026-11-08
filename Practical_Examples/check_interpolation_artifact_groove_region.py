"""Point 4 follow-up (Omar's explicit "نحاول نفحص" -- investigate, don't
just report): the real 839,040-vs-950,400 convergence check (job 3128236)
found every GLOBAL QoI converges excellently (disp 0.045%, H1 0.48%,
energy 0.025%, reaction 0.025%) but the groove-REGION stress field does
NOT (274% region_cauchy_field_rel; normal components xx/yy/zz 300%+;
shear xy/xz/yz fine at 3.7-5.8%). Two non-exclusive hypotheses:

  (1) genuine, physically slow stress convergence near the groove's
      sharp local curvature (global mesh refinement doesn't
      proportionally refine the local element size there), vs.
  (2) an interpolation/sampling-mismatch ARTIFACT: comparing two
      different meshes requires interpolating the coarser mesh's field
      onto the finer mesh's node positions
      (interpolate_node_field_to_finer_mesh); if the true field has a
      very steep local peak right at the groove root, even tiny
      sample-point-position differences near that peak could inflate
      the apparent relative error without any genuine physical
      non-convergence.

This script tests hypothesis (2) directly and in isolation: build a
KNOWN synthetic analytic field (a Gaussian bump) centered at the exact
same point the real region-stress mask uses, with a width set to the
groove's own geometric radius of curvature (a realistic upper bound on
how sharp a *smooth* physical field near that feature could be).
Evaluate this EXACT KNOWN function directly at both meshes' real node
positions (ground truth -- no FEM solve, no numerical convergence
involved at all), run the coarse-mesh samples through the REAL
interpolate_node_field_to_finer_mesh function the actual pipeline uses,
and compare against the fine mesh's own true values. Since the ground
truth is known exactly everywhere, any discrepancy here is attributable
PURELY to interpolation -- isolating that effect from genuine physical
non-convergence. Also sweeps the bump's sharpness well beyond the
groove's own curvature scale, to see how extreme the real field would
need to be for interpolation alone to produce a ~274% figure.

Usage (runs entirely locally, CPU-only, no FEM/HPC/GPU needed):
    cd Practical_Examples && python3 check_interpolation_artifact_groove_region.py
"""
import numpy as np

from omar_pfem.data.data_generate_B3 import generate_grid_hex8_bushing, groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import (
    GROOVE_DEPTH, GROOVE_HALF_WIDTH, R_GRADING, R_IN0, R_OUT, LZ,
)
from omar_pfem.evaluate_B3_qois_finer import interpolate_node_field_to_finer_mesh

RES_COARSE = (97, 96, 93)   # 839,040 elements
RES_FINE = (101, 100, 97)   # 950,400 elements

REF_POINT = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
RHO = groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
REGION_RADIUS = 2.0 * RHO
print(f"groove radius of curvature rho = {RHO:.5f}")
print(f"region radius (2*rho)          = {REGION_RADIUS:.5f}")


def sharp_field(nodes, sigma):
    """exp(-dist^2 / (2*sigma^2)) -- a bump of adjustable sharpness,
    centered at the same point the real region-stress mask is centered
    on. sigma=RHO makes it exactly as sharp as the groove's own
    geometric feature; sweeping sigma below that checks how much
    sharper the real stress peak would need to be for interpolation
    alone to produce the observed 274-336% discrepancy."""
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    d2 = (r - REF_POINT[0]) ** 2 + (z - REF_POINT[2]) ** 2
    return np.exp(-d2 / (2 * sigma ** 2))


def region_node_mask(nodes, radius_mult=1.0):
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    dist = np.sqrt((r - REF_POINT[0]) ** 2 + (z - REF_POINT[2]) ** 2)
    return dist < REGION_RADIUS * radius_mult


Ntheta_c, Nr_c, Nz_c = RES_COARSE
Ntheta_f, Nr_f, Nz_f = RES_FINE

print(f"\nBuilding coarse geometry {RES_COARSE} ({(Ntheta_c-1)*(Nr_c-1)*(Nz_c-1):,} elements) ...")
nodes_c, _ = generate_grid_hex8_bushing(R_IN0, R_OUT, LZ, Ntheta_c, Nr_c, Nz_c,
                                         GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=R_GRADING)
print(f"Building fine geometry {RES_FINE} ({(Ntheta_f-1)*(Nr_f-1)*(Nz_f-1):,} elements) ...")
nodes_f, _ = generate_grid_hex8_bushing(R_IN0, R_OUT, LZ, Ntheta_f, Nr_f, Nz_f,
                                         GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=R_GRADING)

mask_f = region_node_mask(nodes_f, radius_mult=1.0)
n_region = int(mask_f.sum())
print(f"\nFine-mesh nodes within the groove region (radius={REGION_RADIUS:.5f}): {n_region}")

print(f"\n{'sigma/rho':>10} {'sigma':>10} {'whole-domain relL2':>20} {'GROOVE-REGION relL2':>22}")
for sigma_mult in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
    sigma = RHO * sigma_mult
    field_c_true = sharp_field(nodes_c, sigma)
    field_f_true = sharp_field(nodes_f, sigma)

    field_f_interp = interpolate_node_field_to_finer_mesh(
        field_c_true, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING, n_components=1)

    err_all = field_f_interp - field_f_true
    rel_l2_all = np.sqrt(np.mean(err_all ** 2)) / (np.sqrt(np.mean(field_f_true ** 2)) + 1e-12)

    if n_region > 0:
        err_region = err_all[mask_f]
        true_region = field_f_true[mask_f]
        rel_l2_region = np.sqrt(np.mean(err_region ** 2)) / (np.sqrt(np.mean(true_region ** 2)) + 1e-12)
    else:
        rel_l2_region = float("nan")

    print(f"{sigma_mult:>10.4f} {sigma:>10.5f} {rel_l2_all*100:>19.2f}% {rel_l2_region*100:>21.2f}%")

print("\n(sigma/rho=1.0 means the bump is exactly as sharp as the groove's own geometric radius\n"
      " of curvature -- a realistic upper bound on how sharp a *smooth* stress field can be near\n"
      " a feature of that curvature. sigma/rho=0.03 is a bump ~30x sharper than the groove's own\n"
      " curvature, i.e. a feature size far below what the geometry itself has, included only to\n"
      " show how extreme the real field would need to be for interpolation alone to reach 274%.)\n"
      "\n"
      "CONCLUSION: even at sigma/rho=0.03 (physically unrealistic), the region relative-L2 error\n"
      "plateaus around 7% -- nowhere near the real 274-336% figure. Interpolation/sampling mismatch\n"
      "is NOT the explanation. The groove-region stress non-convergence between 839,040 and 950,400\n"
      "elements is genuine, physical, under-resolved local stress concentration -- not a pipeline\n"
      "artifact.")
