"""Second, LIGHTWEIGHT 3D candidate (Omar's own instruction, 2026-09-21):
Timon separately mentioned a tire example in an earlier round as a
candidate HARDER problem (not a formal round-13 requirement like the
rubber mount) -- prepared here in parallel, to the SAME single-material,
no-contact scope, but deliberately kept simpler than B3's own final
rigor (geometry+BC+one material+smoke solve+preliminary convergence
only, NO full QoI/threshold table, NO training) so both candidates can
be shown to Timon before any expensive commitment.

REVISED 2026-09-21 (Omar's own technical review, before any email to
Timon): four real corrections, not smoothed over:
  1. Terminology: the loading region is renamed from "contact patch" to
     "localized tread loading region" throughout -- there is no actual
     ground-contact formulation here (no contact mechanics, no rigid
     ground surface), so "contact patch" overstated what this is.
  2. Sector cuts: VERIFIED (not just assumed) that the two
     circumferential cut faces (Phi=0, Phi=Phi_max) are never
     constrained anywhere in this module -- they are genuinely FREE,
     not "artificially clamped." Genuine PERIODIC boundary conditions
     (tying the two cut faces' displacements together, the physically
     correct treatment for "a segment of a full ring") are NOT
     implemented -- torch-fem's own Dirichlet-only constraint API has no
     built-in multi-point/periodic-tie mechanism, and adding one is a
     real, nontrivial piece of new solver infrastructure, explicitly out
     of scope for a LIGHTWEIGHT preliminary candidate. This is a real,
     disclosed limitation: the free-cut sector is only a reasonable
     approximation to a periodic ring near the sector's own middle
     (which is where the tread loading and the region-Cauchy QoI are
     both centered, deliberately, per point 4 below), not at or near the
     cuts themselves.
  3. An internal inflation pressure is now applied FIRST, over the
     WHOLE tread surface (not just the localized window), superposed
     with the localized tread load (which adds extra, inward pressure
     only within its own window) -- both scaled together through the
     same incremental solve (a real simplification for this lightweight
     candidate: two genuinely SEQUENTIAL load stages, inflate-then-load,
     would need either two separate .solve() calls or a per-load
     increment schedule torch-fem's own API does not expose directly;
     superposing both loads under one incremental ramp still represents
     "an inflated tire under an added local tread load," just not as two
     strictly separate physical stages). This makes the geometry
     meaningfully closer to an actual tire than a plain compressed
     rubber torus, per Omar's own request.
  4. The region-Cauchy-stress QoI stays fixed in physical space, at the
     tread groove's own deepest point (theta=pi/2) and mid-sector
     (Phi=Phi_max/2) -- away from BOTH sector cuts and the rim's own
     bonded/free transition. It sits at the CENTER of the localized
     load window (matching the groove's own location, by design, since
     both represent the same physical tread centerline), not at that
     window's own edge (where the pressure boundary condition itself is
     discontinuous and would contaminate the measurement the same way
     B3's original bonded/free edge did).

Geometry: a TIRE SECTOR -- a genuine torus segment, not a straight
extrusion (that would just be B3 again). A meridian cross-section (the
same (theta,r) half-ring parametrization B2/B3 already use, reused
unchanged for the geometry math) is swept through a limited
circumferential sector angle Phi in [0,Phi_max] around a big wheel axis,
instead of extruded along a straight line -- genuinely toroidal, the
correct topology for a tire (B3's own bushing is a cylinder; a tire is
a torus; these are not the same shape and one is not a stand-in for the
other).

In the meridian cross-section: r_local=R_bead (theta=0..pi) is bonded to
a RIGID RIM (fixed); r_local=R_tread_eff(theta) is the tread surface,
with a smooth (C1, raised-cosine, same construction as B3's own groove)
GROOVE dipping at theta=pi/2 (the tread's own centerline) -- a disclosed,
simplified stand-in for a real tire's circumferential tread groove
(constant around the tire's circumference by construction, since it only
depends on theta, not on the sweep angle Phi). Its own radius of
curvature is computed and disclosed exactly as B3's groove is.

Single material model (Neo-Hookean), reusing `neo_hookean_psi_3d`
unchanged, exactly as B3 does.
"""
import numpy as np

from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring


def tread_groove_R_out(theta, theta_max, R_tread0, groove_depth, groove_half_theta):
    """Smooth (C1) tread-radius profile: R_tread0 everywhere except a
    raised-cosine dip of depth `groove_depth` within `groove_half_theta`
    of the tread centerline (theta=theta_max/2) -- same construction as
    B3's own `groove_R_in`, applied to the OUTER boundary instead."""
    theta = np.asarray(theta, dtype=float)
    theta_c = theta_max / 2.0
    r = np.full_like(theta, R_tread0)
    band = np.abs(theta - theta_c) < groove_half_theta
    r[band] = R_tread0 - (groove_depth / 2.0) * (1.0 + np.cos(np.pi * (theta[band] - theta_c) / groove_half_theta))
    return r


def tread_groove_radius_of_curvature(groove_depth, groove_half_theta, R_tread0):
    """Radius of curvature at the groove's own deepest point -- same
    formula as B3's `groove_radius_of_curvature`, but the groove here is
    parametrized by an ANGLE (theta) rather than a length (z), so an
    extra factor of R_tread0 (the local arc-length-per-radian scale)
    converts the angular curvature to a genuine physical length."""
    d, w = groove_depth, groove_half_theta
    r_pp_theta = (d / 2.0) * (np.pi / w) ** 2  # curvature w.r.t. the angular parameter
    return (R_tread0 ** 2) / r_pp_theta if r_pp_theta > 0 else float("inf")


def generate_grid_hex8_tire_sector(R_bead, R_tread0, R_big, Phi_max, Ntheta, Nr, Nphi,
                                    groove_depth, groove_half_theta, r_grading=1.0):
    """Meridian half-ring cross-section (B2's OWN generate_grid_Q4_ring,
    reused unchanged for the theta/r math) with a tread groove in its
    outer radius, swept through sector angle Phi_max around a big wheel
    axis of radius R_big -- a genuine torus segment. theta=0 and
    theta=pi rows are the tire's two flat sidewall-facing lines (see
    module docstring); theta=pi/2 is the tread centerline. The two
    circumferential cut faces (Phi=0, Phi=Phi_max) are NEVER constrained
    by this module (see boundary_node_sets) -- genuinely free, not
    artificially clamped, though also not periodic (a real, disclosed
    limitation for a lightweight candidate -- see module docstring)."""
    assert groove_half_theta <= np.pi / 2, "groove too wide: reaches theta=0 or pi"
    assert groove_depth < R_tread0 - R_bead, "groove too deep: would exceed the bead radius"

    thetas = np.linspace(0.0, np.pi, Ntheta)
    R_tread_eff = tread_groove_R_out(thetas, np.pi, R_tread0, groove_depth, groove_half_theta)
    t = np.linspace(0.0, 1.0, Nr) ** r_grading
    Rr = R_bead + np.outer(R_tread_eff - R_bead, t)  # (Ntheta, Nr)
    TH = np.outer(thetas, np.ones(Nr))
    x_local = (Rr * np.cos(TH)).ravel()
    y_local = (Rr * np.sin(TH)).ravel()

    # NOTE: node order is (n1,n4,n3,n2), the REVERSE winding of B2/B3's
    # own ring convention -- the (radius*cos(phi), x_local,
    # radius*sin(phi)) axis assignment below is a parity-flipping
    # permutation relative to B3's own (radius*cos, radius*sin, z)
    # placement (which keeps the local ring's own (x,y) as two of the
    # three global axes and extrudes the third) -- checked directly
    # (every element had NEGATIVE signed volume with the un-reversed
    # order), so the winding is reversed here to compensate and restore
    # positive volumes, verified by the assertion in __main__ below.
    elements2d = []
    for j in range(Ntheta - 1):
        for i in range(Nr - 1):
            n1 = j * Nr + i
            n2 = n1 + 1
            n3 = (j + 1) * Nr + i + 1
            n4 = (j + 1) * Nr + i
            elements2d.append([n1, n4, n3, n2])

    n2d = Ntheta * Nr
    phis = np.linspace(0.0, Phi_max, Nphi)
    nodes3d = np.zeros((n2d * Nphi, 3))
    for k, phi in enumerate(phis):
        radius = R_big + y_local
        nodes3d[k * n2d:(k + 1) * n2d, 0] = radius * np.cos(phi)
        nodes3d[k * n2d:(k + 1) * n2d, 1] = x_local
        nodes3d[k * n2d:(k + 1) * n2d, 2] = radius * np.sin(phi)

    elements3d = []
    for k in range(Nphi - 1):
        off0, off1 = k * n2d, (k + 1) * n2d
        for quad in elements2d:
            n1, n2, n3, n4 = quad
            elements3d.append([n1 + off0, n2 + off0, n3 + off0, n4 + off0,
                                n1 + off1, n2 + off1, n3 + off1, n4 + off1])
    return nodes3d, np.array(elements3d, dtype=int)


def boundary_node_sets(nodes, R_bead, R_tread0, R_big, groove_depth, groove_half_theta,
                        tread_load_half_theta, tol=1e-9):
    """Returns (rim, full_tread, tread_load_region) boolean masks,
    computed from each node's own LOCAL (theta, r_local) recovered from
    its global position (inverting the sweep).
    rim: r_local=R_bead (bonded to the fixed rigid rim, every Phi).
    full_tread: the ENTIRE tread surface r_local=R_tread_eff(theta), all
    theta and Phi -- target of the internal inflation pressure.
    tread_load_region: the SAME tread surface but restricted to theta
    within tread_load_half_theta of the tread centerline (theta=pi/2) --
    the additional LOCALIZED TREAD LOADING REGION (renamed from the
    earlier, inaccurate "contact patch" -- there is no ground-contact
    formulation here, see module docstring point 1)."""
    # Invert the forward sweep mapping (generate_grid_hex8_tire_sector):
    #   X_global = (R_big + y_local) * cos(phi), Y_global = x_local,
    #   Z_global = (R_big + y_local) * sin(phi)
    # so x_local is the global Y directly, and y_local follows from the
    # radial distance from the wheel axis (X,Z plane) minus R_big.
    x_local = nodes[:, 1]
    radial_from_wheel_axis = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 2] ** 2)
    y_local = radial_from_wheel_axis - R_big
    r_local = np.sqrt(x_local ** 2 + y_local ** 2)
    theta_local = np.arctan2(y_local, x_local)  # in [0, pi] since y_local = Rr*sin(theta) >= 0
    theta_local = np.clip(theta_local, 0.0, np.pi)

    rim = np.abs(r_local - R_bead) < tol
    r_tread_eff = tread_groove_R_out(theta_local, np.pi, R_tread0, groove_depth, groove_half_theta)
    full_tread = np.abs(r_local - r_tread_eff) < 1e-6
    theta_c = np.pi / 2.0
    tread_load_region = full_tread & (np.abs(theta_local - theta_c) < tread_load_half_theta)
    return rim, full_tread, tread_load_region


if __name__ == "__main__":
    # Cheap structural smoke test (mesh only, no solve).
    R_bead, R_tread0, R_big = 0.5, 1.0, 3.0
    Phi_max = np.pi / 6  # 30-degree sector
    groove_depth, groove_half_theta = 0.05, 0.15
    tread_load_half_theta = 0.3
    Ntheta, Nr, Nphi = 13, 6, 9
    nodes, elements = generate_grid_hex8_tire_sector(
        R_bead, R_tread0, R_big, Phi_max, Ntheta, Nr, Nphi, groove_depth, groove_half_theta)
    print(f"nodes: {nodes.shape}, elements: {elements.shape}")
    print(f"tread groove radius of curvature: "
          f"{tread_groove_radius_of_curvature(groove_depth, groove_half_theta, R_tread0):.4f}")
    assert nodes.shape == (Ntheta * Nr * Nphi, 3)
    assert elements.shape == ((Ntheta - 1) * (Nr - 1) * (Nphi - 1), 8)

    rim, full_tread, tread_load_region = boundary_node_sets(
        nodes, R_bead, R_tread0, R_big, groove_depth, groove_half_theta, tread_load_half_theta)
    print(f"rim: {rim.sum()}, full_tread: {full_tread.sum()}, "
          f"tread_load_region: {tread_load_region.sum()}")
    assert rim.sum() == Ntheta * Nphi
    assert full_tread.sum() == Ntheta * Nphi
    assert tread_load_region.sum() > 0, \
        "tread load region is empty -- widen tread_load_half_theta or check theta indexing"

    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        return sum(np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
                   for a, b, c, d in tets)

    bad = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    print(f"inverted/degenerate elements: {bad} / {len(elements)}")
    assert bad == 0, "mesh has inverted elements -- fix node ordering before any solve"
    print("Mesh structural checks OK.")
