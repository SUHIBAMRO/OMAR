"""Second, LIGHTWEIGHT 3D candidate (Omar's own instruction, 2026-09-21):
Timon separately mentioned a tire example in an earlier round as a
candidate HARDER problem (not a formal round-13 requirement like the
rubber mount) -- prepared here in parallel, to the SAME single-material,
no-contact scope, but deliberately kept simpler than B3's own final
rigor (geometry+BC+one material+smoke solve+preliminary convergence
only, NO full QoI/threshold table, NO training) so both candidates can
be shown to Timon before any expensive commitment.

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

Load: a normal PRESSURE over a limited angular "contact patch" window
around the tread centerline (theta near pi/2), applied via torch-fem's
own `integrate_surface_load` (a scalar pressure acting along the
outward normal) -- a simplified stand-in for a real ground-contact
load, explicitly NOT a solved contact problem (Timon's own call whether
real contact modeling is worth the extra scope if this candidate is the
one chosen).

Explicit, disclosed simplifications for this LIGHTWEIGHT candidate
(would need revisiting if Timon picks the tire over the bushing):
  - The two circumferential cut faces (Phi=0, Phi=Phi_max) are left
    FREE (not periodic) -- a genuine approximation, valid only for a
    small sector "far" from needing a fully periodic solution.
  - The bead sits on a rounded (half-circle) profile rather than a
    flat ring, and no separate belt/ply reinforcement layers are
    modeled -- single homogeneous hyperelastic material throughout,
    per the same "one material model" instruction as B3.

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
    module docstring); theta=pi/2 is the tread centerline."""
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
                        contact_half_theta, tol=1e-9):
    """Returns (rim, contact_patch) boolean masks, computed from each
    node's own LOCAL (theta, r_local) recovered from its global
    position (inverting the sweep -- y_local=x_global's own axial
    coordinate is stored directly as the global Y, and r_local follows
    from the global radial distance minus R_big).
    rim: r_local=R_bead (bonded to the fixed rigid rim, every Phi).
    contact_patch: r_local=R_tread_eff(theta) AND theta within
    contact_half_theta of the tread centerline (theta=pi/2) -- the
    normal-pressure load region, every Phi (the load spans the whole
    modeled sector, a simplification -- see module docstring)."""
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
    on_tread = np.abs(r_local - r_tread_eff) < 1e-6
    theta_c = np.pi / 2.0
    contact_patch = on_tread & (np.abs(theta_local - theta_c) < contact_half_theta)
    return rim, contact_patch


if __name__ == "__main__":
    # Cheap structural smoke test (mesh only, no solve).
    R_bead, R_tread0, R_big = 0.5, 1.0, 3.0
    Phi_max = np.pi / 6  # 30-degree sector
    groove_depth, groove_half_theta = 0.05, 0.15
    contact_half_theta = 0.3
    Ntheta, Nr, Nphi = 13, 6, 9
    nodes, elements = generate_grid_hex8_tire_sector(
        R_bead, R_tread0, R_big, Phi_max, Ntheta, Nr, Nphi, groove_depth, groove_half_theta)
    print(f"nodes: {nodes.shape}, elements: {elements.shape}")
    print(f"tread groove radius of curvature: "
          f"{tread_groove_radius_of_curvature(groove_depth, groove_half_theta, R_tread0):.4f}")
    assert nodes.shape == (Ntheta * Nr * Nphi, 3)
    assert elements.shape == ((Ntheta - 1) * (Nr - 1) * (Nphi - 1), 8)

    rim, contact_patch = boundary_node_sets(
        nodes, R_bead, R_tread0, R_big, groove_depth, groove_half_theta, contact_half_theta)
    print(f"rim: {rim.sum()}, contact_patch: {contact_patch.sum()}")
    assert rim.sum() == Ntheta * Nphi
    assert contact_patch.sum() > 0, "contact patch is empty -- widen contact_half_theta or check theta indexing"

    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        return sum(np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
                   for a, b, c, d in tets)

    bad = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    print(f"inverted/degenerate elements: {bad} / {len(elements)}")
    assert bad == 0, "mesh has inverted elements -- fix node ordering before any solve"
    print("Mesh structural checks OK.")
