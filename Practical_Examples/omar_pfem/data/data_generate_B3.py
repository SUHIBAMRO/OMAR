"""B3: the new 3D "realistic case" (Timon's newest email, item 4) --
DESIGN HISTORY (all corrections real, all driven by Omar's own review,
recorded honestly per this project's standing discipline):

1. First draft: a finite-thickness PLATE with a circular through-hole.
   Rejected -- wrong shape category ("لوح", not the "كتلة" originally
   agreed), and too close to a textbook 2D benchmark to match what
   Timon actually asked for ("a 3D hyperelastic rubber mount").
2. Second draft: a rocking rubber bushing (rigid core + rigid housing,
   core given a z-linear tilting displacement), with a quarter-circle
   FILLET terminating the bond short of each end to avoid a sharp
   bonded/free corner. This got the overall concept right, but Omar
   then asked directly: is the stress concentration produced by an
   explicit smooth feature, or just by the sharp BC jump (bonded ->
   free) at the start of the fillet? It was the latter -- geometry
   smooth, but the BOUNDARY CONDITION itself still jumped discontinuously
   at one line, which can itself reproduce a free-edge singularity.
3. THIS version (current): removes the free/bonded transition entirely.
   The core is now bonded to the rubber continuously over the ENTIRE
   height (z in [0,Lz], same as the outer housing) -- no BC jump
   anywhere on the inner surface. The one explicit, disclosed,
   finite-radius feature is now a smooth circumferential GROOVE (a
   waist in the inner radius profile, R_in(z), C1-continuous, i.e. zero
   slope at both edges of the groove -- no kink, no discontinuity in
   the boundary condition, only a smooth geometric dip in an otherwise
   uniformly-bonded surface). The stress-concentration QoI region is
   centered on this groove, far from every real bonded/free transition
   (which still exist only at the physical ends z=0/Lz, exactly as any
   real bonded rubber part has, but which are now nowhere near what is
   being measured).

Geometry: a hollow rubber cylinder (annulus in cross-section) between a
RIGID INNER CORE (radius R_in(z), continuously bonded) and a RIGID OUTER
HOUSING (radius R_out, constant, continuously bonded) -- an elastomeric
bushing/engine mount, height Lz. The core's own radius profile has a
smooth groove at mid-height: R_in(z) = R_in0 everywhere except within
+/-groove_half_width of z=Lz/2, where it dips smoothly (a raised-cosine
profile) to R_in0-groove_depth at the very center. The groove's own
radius of curvature at its deepest point (`groove_radius_of_curvature`,
below) is the explicit, disclosed "finite radius" this case's stress
concentration is attributed to.

Loading: a TRUE RIGID-BODY ROTATION of the core (not just a linear
u_x(z) profile) by a small angle `phi` about the y-axis through the
point (0,0,Lz/2) -- a pure rock/tilt, no net translation. This is more
physically correct for a large-deformation (hyperelastic) rocking
bushing than a linear displacement profile: a genuinely rigid rotation
couples u_x AND u_z on the core surface (a linear-u_x-only profile is
merely its own small-angle limit, missing the z-motion entirely).

WHY THIS IS GENUINELY 3D, NOT A TRIVIAL EXTRUSION OF B2: B2's own
loading (uniform internal pressure) is z-independent and axisymmetric,
so its solution is identical at every z and every theta -- a real
extrusion in every meaningful sense. Here BOTH the rotation (a
z-dependent, non-axisymmetric boundary condition) AND the groove (a
z-dependent geometry) break that: the rotation forces genuine
out-of-plane shear (du_x/dz, du_z/dr) that cannot exist in any
z-independent model, and the groove's own stress state depends on all
three mesh directions (theta, since the rotation is not axisymmetric;
r, since the groove's own curvature must be resolved radially; z, since
the groove is a localized feature). Checked empirically, not assumed,
by mesh_convergence_B3.py.

Single material model throughout (Neo-Hookean), reusing
`neo_hookean_psi_3d` from torchfem_comparison.py completely unchanged.

Mesh: B2's OWN ring generator (`generate_grid_Q4_ring`), called once per
z-layer with that layer's own effective R_in(z) (same Ntheta/Nr/theta
connectivity every layer -- only coordinates change), theta_max=pi (a
HALF ring, exploiting the one real mirror symmetry a y-axis rotation
has about the xz-plane -- there is no second symmetry plane).
"""
import numpy as np

from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring


def groove_R_in(z, Lz, R_in0, groove_depth, groove_half_width):
    """Smooth (C1-continuous) inner-radius profile: R_in0 everywhere
    except a raised-cosine dip of depth `groove_depth` within
    `groove_half_width` of mid-height (z=Lz/2). Zero slope at the edges
    of the dip (matches the straight R_in0 wall's own zero slope there)
    -- no kink, no BC discontinuity, purely a smooth geometric feature."""
    z = np.asarray(z, dtype=float)
    zc = Lz / 2.0
    r = np.full_like(z, R_in0)
    band = np.abs(z - zc) < groove_half_width
    r[band] = R_in0 - (groove_depth / 2.0) * (1.0 + np.cos(np.pi * (z[band] - zc) / groove_half_width))
    return r


def groove_radius_of_curvature(groove_depth, groove_half_width):
    """Radius of curvature at the groove's own deepest point (z=Lz/2),
    from r(z)=R_in0-(d/2)(1+cos(pi(z-zc)/w)): r'(zc)=0,
    r''(zc)=(d/2)(pi/w)^2, so rho=1/r''(zc) there (curvature=|r''|
    since r'=0). A genuine, disclosed, finite physical length -- this is
    the "finite radius" the module docstring refers to."""
    d, w = groove_depth, groove_half_width
    r_pp = (d / 2.0) * (np.pi / w) ** 2
    return 1.0 / r_pp


def generate_grid_hex8_bushing(R_in0, R_out, Lz, Ntheta, Nr, Nz,
                                groove_depth, groove_half_width, r_grading=1.0):
    """Half-ring (theta in [0,pi]) cross-section, B2's own generator,
    stacked along z with the inner radius following `groove_R_in`
    instead of a constant value -- each z-layer gets its own ring
    geometry (same connectivity every layer, from `generate_grid_Q4_ring`
    -- only coordinates differ). theta=0 and theta=pi rows lie exactly
    on y=0 (the mirror-symmetry plane for a y-axis rotation).
    Requires groove_half_width <= Lz/2 (the dip must not reach the ends)
    and groove_depth < R_out-R_in0 (must stay inside the outer boundary)."""
    assert groove_half_width <= Lz / 2, "groove too wide: reaches the physical ends"
    assert groove_depth < R_out - R_in0, "groove too deep: would exceed the outer boundary"

    zs = np.linspace(0.0, Lz, Nz)
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


def boundary_node_sets(nodes, R_in0, R_out, Lz, groove_depth, groove_half_width, tol=1e-9):
    """Returns (inner_core, outer_housing, symmetry_y0) boolean masks.
    inner_core: the FULL groove-profile inner surface, r=R_in_eff(z) for
    EVERY z (continuously bonded -- no free/bonded transition anywhere
    on this surface, unlike the rejected fillet draft). outer_housing:
    r=R_out surface (all z). symmetry_y0: y=0 plane (theta=0 and pi
    rows). The flat z=0/z=Lz end faces are the rubber's own real free
    surfaces (need no constraint) -- same as every earlier draft; the
    bonded/free transitions that exist there are now far from the
    groove feature being studied."""
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    r_in_eff = groove_R_in(z, Lz, R_in0, groove_depth, groove_half_width)
    inner_core = np.abs(r - r_in_eff) < tol
    outer_housing = np.abs(r - R_out) < tol
    symmetry_y0 = np.abs(y) < tol
    return inner_core, outer_housing, symmetry_y0


def rigid_rotation_displacement(nodes, Lz, phi):
    """TRUE rigid-body rotation displacement for points on the core
    surface: rotation by angle `phi` (radians) about the y-axis through
    (0,0,Lz/2). For reference position (x0,y0,z0):
        dz = z0 - Lz/2
        x' = x0*cos(phi) + dz*sin(phi)
        z' = -x0*sin(phi) + dz*cos(phi) + Lz/2
        y' = y0   (rotation about y leaves y untouched)
    Returns (ux, uy, uz) = (x'-x0, 0, z'-z0). Small-angle limit
    (phi->0): ux -> dz*phi (the OLD linear-in-z profile), uz -> -x0*phi
    (a genuinely new z-motion the old model never had) -- the old model
    was this rotation's own small-angle, x0-independent approximation of
    ux, with uz assumed (wrongly) to be exactly zero.

    Preserves the y -> -y mirror symmetry exactly (ux, uz depend on x0,
    z0 only; uy stays 0 for every point), so the half-ring (theta in
    [0,pi]) domain remains valid."""
    x0, z0 = nodes[:, 0], nodes[:, 2]
    dz = z0 - Lz / 2.0
    c, s = np.cos(phi), np.sin(phi)
    ux = x0 * (c - 1.0) + dz * s
    uz = -x0 * s - dz * (1.0 - c)
    uy = np.zeros_like(ux)
    return ux, uy, uz


if __name__ == "__main__":
    # Cheap structural smoke test (mesh only, no solve).
    R_in0, R_out, Lz = 0.5, 1.0, 1.0
    groove_depth, groove_half_width = 0.05, 0.15
    Ntheta, Nr, Nz = 13, 6, 15
    nodes, elements = generate_grid_hex8_bushing(
        R_in0, R_out, Lz, Ntheta, Nr, Nz, groove_depth, groove_half_width)
    print(f"nodes: {nodes.shape}, elements: {elements.shape}")
    print(f"groove radius of curvature: {groove_radius_of_curvature(groove_depth, groove_half_width):.4f}")
    assert nodes.shape == (Ntheta * Nr * Nz, 3)
    assert elements.shape == ((Ntheta - 1) * (Nr - 1) * (Nz - 1), 8)

    inner, outer, sym = boundary_node_sets(nodes, R_in0, R_out, Lz, groove_depth, groove_half_width)
    print(f"inner_core: {inner.sum()}, outer_housing: {outer.sum()}, symmetry_y0: {sym.sum()}")
    assert inner.sum() == Ntheta * Nz  # bonded at EVERY z-layer now
    assert outer.sum() == Ntheta * Nz
    assert sym.sum() == 2 * Nr * Nz

    def hex_signed_volume(pts):
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6),
                (3, 4, 6, 7), (1, 4, 5, 6)]
        vol = 0.0
        for a, b, c, d in tets:
            vol += np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
        return vol

    bad = sum(1 for el in elements if hex_signed_volume(nodes[el]) <= 0)
    print(f"inverted/degenerate elements: {bad} / {len(elements)}")
    assert bad == 0, "mesh has inverted elements -- fix node ordering before any solve"

    # Small-angle sanity check for the rotation kinematics: at phi small,
    # ux should closely match the OLD linear-in-z profile at x0~0
    # (theta=pi/2-ish points), and uz should be small but NONZERO near
    # x0=R_in (theta=0), confirming the new z-motion the old model omitted.
    phi_test = 0.01
    core_pts = nodes[inner]
    ux, uy, uz = rigid_rotation_displacement(core_pts, Lz, phi_test)
    near_x0 = np.abs(core_pts[:, 0]) < 1e-6
    if near_x0.any():
        expected_ux = (core_pts[near_x0, 2] - Lz / 2.0) * phi_test
        assert np.allclose(ux[near_x0], expected_ux, atol=1e-6)
    near_theta0 = np.abs(core_pts[:, 1]) < 1e-6  # y=0, theta=0 line (x0=R_in there)
    if near_theta0.any():
        assert np.abs(uz[near_theta0]).max() > 1e-6, "expected nonzero uz from the true rotation at theta=0"
    print("Mesh + kinematics structural checks OK.")
