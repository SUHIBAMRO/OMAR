"""Timon's 2026-11-10 email, item 1: "We need to first get convergence
of the local stress, preferably using local refinement." The existing
B3 mesh-convergence ladder (data_generate_B3.generate_grid_hex8_bushing)
only has GLOBAL refinement -- every z-layer gets finer together, so
"finer" always means more elements everywhere, not more elements
specifically where they are needed (at the groove root). This file adds
genuine LOCAL refinement along z, clustering nodes near the groove's own
deepest point (zc = Lz/2) while leaving the rest of the domain coarse --
a new, separate function, not an edit to data_generate_B3.py's own
generate_grid_hex8_bushing (which produced every already-published B3
number in this project and must not be touched).

Z-clustering via a sinh stretching function (a standard mesh-grading
technique, not invented here): for a uniform parameter u in [0,1],
w = 2u-1 in [-1,1], z = zc + (Lz/2) * sinh(beta*w)/sinh(beta) for
beta > 0 (beta=0 is a removable singularity, handled as the uniform
limit explicitly, not by dividing by zero). Larger beta clusters MORE
nodes near w=0 (z=zc). beta=0 must reproduce the EXISTING uniform
z-spacing EXACTLY -- this is checked directly, not assumed, in
test_z_refine_beta_zero_matches_uniform() below, so this new function
is a strict superset of the existing one, not a parallel reimplementation
that could silently drift from it.
"""
import numpy as np

from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring
from omar_pfem.data.data_generate_B3 import groove_R_in


def z_layers_sinh_graded(Nz, Lz, beta):
    """Nz z-values in [0, Lz], clustered near Lz/2 by sinh grading
    strength `beta` (beta=0 -> exactly np.linspace(0, Lz, Nz))."""
    u = np.linspace(0.0, 1.0, Nz)
    w = 2.0 * u - 1.0
    zc = Lz / 2.0
    if beta == 0.0:
        return zc + (Lz / 2.0) * w
    return zc + (Lz / 2.0) * (np.sinh(beta * w) / np.sinh(beta))


def generate_grid_hex8_bushing_z_refined(R_in0, R_out, Lz, Ntheta, Nr, Nz,
                                          groove_depth, groove_half_width,
                                          r_grading=1.0, z_beta=0.0):
    """Same construction as data_generate_B3.generate_grid_hex8_bushing
    (same connectivity, same groove_R_in profile, same radial grading),
    EXCEPT the z-layer positions come from z_layers_sinh_graded instead
    of np.linspace -- the only change. z_beta=0.0 reproduces the
    original function's output exactly (verified directly, see this
    module's own test)."""
    assert groove_half_width <= Lz / 2, "groove too wide: reaches the physical ends"
    assert groove_depth < R_out - R_in0, "groove too deep: would exceed the outer boundary"

    zs = z_layers_sinh_graded(Nz, Lz, z_beta)
    R_in_eff = groove_R_in(zs, Lz, R_in0, groove_depth, groove_half_width)

    _, elements2d = generate_grid_Q4_ring(R_in0, R_out, Ntheta, Nr, theta_max=np.pi, r_grading=r_grading)
    n2d = Ntheta * Nr
    nodes3d = np.zeros((n2d * Nz, 3))
    for k, (z, r_in_k) in enumerate(zip(zs, R_in_eff)):
        nodes2d_k, _ = generate_grid_Q4_ring(r_in_k, R_out, Ntheta, Nr, theta_max=np.pi, r_grading=r_grading)
        nodes3d[k * n2d:(k + 1) * n2d, 0:2] = nodes2d_k
        nodes3d[k * n2d:(k + 1) * n2d, 2] = z

    elements3d = []
    for k in range(Nz - 1):
        off0, off1 = k * n2d, (k + 1) * n2d
        for quad in elements2d:
            n1, n2, n3, n4 = quad
            elements3d.append([n1 + off0, n2 + off0, n3 + off0, n4 + off0,
                                n1 + off1, n2 + off1, n3 + off1, n4 + off1])
    return nodes3d, np.array(elements3d, dtype=int)


def min_z_element_size(Nz, Lz, beta):
    """Smallest and largest z-extent among the Nz-1 z-layers, for
    reporting how concentrated the local refinement actually is (e.g.
    'finest z-element is 12x smaller than the coarsest, same Nz')."""
    zs = z_layers_sinh_graded(Nz, Lz, beta)
    d = np.diff(zs)
    return float(d.min()), float(d.max())


if __name__ == "__main__":
    # Direct, printed correctness check -- not just asserted silently --
    # that beta=0 reproduces the existing production mesh generator exactly.
    from omar_pfem.data.data_generate_B3 import generate_grid_hex8_bushing
    R_IN0, R_OUT, LZ = 0.5, 1.0, 1.0
    GROOVE_DEPTH, GROOVE_HALF_WIDTH = 0.20, 0.15
    Ntheta, Nr, Nz = 21, 20, 19

    nodes_orig, elems_orig = generate_grid_hex8_bushing(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=1.5)
    nodes_new, elems_new = generate_grid_hex8_bushing_z_refined(
        R_IN0, R_OUT, LZ, Ntheta, Nr, Nz, GROOVE_DEPTH, GROOVE_HALF_WIDTH, r_grading=1.5, z_beta=0.0)

    max_node_diff = np.abs(nodes_orig - nodes_new).max()
    elems_match = np.array_equal(elems_orig, elems_new)
    print(f"beta=0 vs. original generate_grid_hex8_bushing: "
          f"max node coordinate diff = {max_node_diff:.3e}, elements identical = {elems_match}")
    print("PASS" if max_node_diff < 1e-12 and elems_match else "FAIL")

    for beta in (0.0, 2.0, 4.0, 6.0):
        lo, hi = min_z_element_size(Nz, LZ, beta)
        print(f"beta={beta}: finest z-element={lo:.5f}, coarsest z-element={hi:.5f}, "
              f"ratio={hi/lo:.2f}x")
