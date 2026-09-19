"""B3: the new 3D "realistic case" (Timon's newest email, item 4) --
CORRECTED DESIGN (2026-09-19, after Omar's explicit rejection of an
earlier "plate with a hole" draft: "شو لوح! احنا ما اتفقنا لوح!" -- wrong
shape category, and not what Timon asked for).

Geometry, following Timon's own wording as literally as possible ("a 3D
hyperelastic rubber mount with a finite-radius stress concentration"),
kept as simple as the physics allows: a hollow rubber cylinder (annulus
in cross-section, radius R_in to R_out, height Lz) bonded to a RIGID
INNER CORE at r=R_in and a RIGID OUTER HOUSING at r=R_out -- exactly an
elastomeric bushing/engine-mount, the standard real-world meaning of
"rubber mount". The outer housing is fixed; the inner core is given a
prescribed ROCKING (tilting) displacement in x that varies LINEARLY
along z, from -delta0 at z=0 to +delta0 at z=Lz, with zero net
translation (a pure tilt about its own mid-height) -- a completely
standard duty cycle for a real bushing (shafts routinely rock/tilt
relative to their housing, not just translate).

WHY THIS IS GENUINELY 3D, NOT A TRIVIAL EXTRUSION OF B2 (Omar's second
explicit requirement): B2's own loading (uniform internal pressure) is
z-independent, so its solution is the SAME at every z -- an extrusion of
a 2D solution in every meaningful sense. Here the boundary displacement
itself varies with z, which forces nonzero out-of-plane shear strain
components (du_x/dz, du_z/dr etc.) that CANNOT exist in any 2D or
z-independent model at all -- the deformation gradient F genuinely
couples different z-slices through the material's own stiffness, not
just through matching end conditions. This is a property of the PHYSICS
(the boundary condition), not an assumption -- checked empirically by
the mesh-convergence study below (mesh_convergence_B3.py), which must
show real z-resolution sensitivity for this claim to be more than a
plausible-sounding argument.

WHERE THE STRESS CONCENTRATION IS: at the bonded inner surface (r=R_in,
a SMOOTH circular surface, no sharp corner -- satisfying Timon's own
"smooth finite-radius" wording literally, the radius of curvature R_in
itself being the "finite radius"), concentrated near the locations
(theta, z) where the core's local motion compresses the rubber gap the
most -- near theta=0/pi (the rocking direction) and near whichever end
(z=0 or Lz) the core is displaced away from the housing at that side.

Single material model throughout (Neo-Hookean), reusing
`neo_hookean_psi_3d` from torchfem_comparison.py completely unchanged
(a genuine 3x3-F energy function, already GPU-validated at N=1401 for
this project's 2D cases -- no new physics code needed for B3 either).

Mesh: B2's OWN ring generator (`generate_grid_Q4_ring`), reused
unchanged with theta_max=pi (a HALF ring, not a quarter -- exploiting
the one mirror symmetry this loading actually has, about the xz-plane,
since a pure x-direction rocking is symmetric under y -> -y; there is
no second symmetry plane here, unlike B2's own fully axisymmetric
pressure loading, since the core's rocking is NOT axisymmetric).
Extruded to HEX8 exactly as before (`extrude_to_hex8`, unchanged from
the rejected draft -- the EXTRUSION MECHANICS were never the problem,
only the 2D cross-section and the loading were).
"""
import numpy as np

from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring


def extrude_to_hex8(nodes2d, elements2d, Lz, Nz):
    """Stacks Nz-1 layers of the 2D quad mesh along z in [0,Lz] into
    HEX8 elements. Node i at layer k gets global index i + k*n2d;
    element [n1,n2,n3,n4] at layer k becomes hex
    [n1,n2,n3,n4, n1',n2',n3',n4'] (k+1 layer), matching Hexa1's own
    bottom-then-top node order (torchfem.elements.Hexa1 docstring)."""
    n2d = nodes2d.shape[0]
    zs = np.linspace(0.0, Lz, Nz)
    nodes3d = np.zeros((n2d * Nz, 3))
    for k, z in enumerate(zs):
        nodes3d[k * n2d:(k + 1) * n2d, 0:2] = nodes2d
        nodes3d[k * n2d:(k + 1) * n2d, 2] = z

    elements3d = []
    for k in range(Nz - 1):
        off0, off1 = k * n2d, (k + 1) * n2d
        for quad in elements2d:
            n1, n2, n3, n4 = quad
            elements3d.append([n1 + off0, n2 + off0, n3 + off0, n4 + off0,
                                n1 + off1, n2 + off1, n3 + off1, n4 + off1])
    return nodes3d, np.array(elements3d, dtype=int)


def generate_grid_hex8_bushing(R_in, R_out, Lz, Ntheta, Nr, Nz, r_grading=1.0):
    """Half-ring (theta in [0,pi]) cross-section, B2's own generator,
    extruded along z. theta=0 and theta=pi rows both lie exactly on
    y=0 (the mirror-symmetry plane for x-direction rocking)."""
    nodes2d, elements2d = generate_grid_Q4_ring(R_in, R_out, Ntheta, Nr,
                                                 theta_max=np.pi, r_grading=r_grading)
    return extrude_to_hex8(nodes2d, elements2d, Lz, Nz)


def boundary_node_sets(nodes, R_in, R_out, tol=1e-9):
    """Returns (inner_core, outer_housing, symmetry_y0) boolean masks.
    inner_core: r=R_in surface, bonded to the rigid rocking core.
    outer_housing: r=R_out surface, bonded to the fixed rigid housing.
    symmetry_y0: y=0 plane (both theta=0 and theta=pi rows), u_y=0.
    z=0/z=Lz faces need NO constraint -- they are the rubber's own real,
    physically free (unbonded) end faces."""
    x, y = nodes[:, 0], nodes[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    inner_core = np.abs(r - R_in) < tol
    outer_housing = np.abs(r - R_out) < tol
    symmetry_y0 = np.abs(y) < tol
    return inner_core, outer_housing, symmetry_y0


def rocking_displacement_x(nodes, Lz, delta0):
    """Prescribed x-displacement for the inner core: linear in z, from
    -delta0 at z=0 to +delta0 at z=Lz (zero at mid-height) -- a pure
    tilt, no net translation. Same value for every theta at a given z
    (the core is rigid: every point on it at height z moves together)."""
    z = nodes[:, 2]
    return delta0 * (2.0 * z / Lz - 1.0)


if __name__ == "__main__":
    # Cheap structural smoke test (mesh only, no solve).
    R_in, R_out, Lz = 0.5, 1.0, 1.0
    Ntheta, Nr, Nz = 13, 6, 7
    nodes, elements = generate_grid_hex8_bushing(R_in, R_out, Lz, Ntheta, Nr, Nz)
    print(f"nodes: {nodes.shape}, elements: {elements.shape}")
    assert nodes.shape == (Ntheta * Nr * Nz, 3)
    assert elements.shape == ((Ntheta - 1) * (Nr - 1) * (Nz - 1), 8)

    inner, outer, sym = boundary_node_sets(nodes, R_in, R_out)
    print(f"inner_core: {inner.sum()}, outer_housing: {outer.sum()}, symmetry_y0: {sym.sum()}")
    assert inner.sum() == Ntheta * Nz
    assert outer.sum() == Ntheta * Nz
    assert sym.sum() == 2 * Nr * Nz  # theta=0 row + theta=pi row, each a full (r,z) grid

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
    print("Mesh structural checks OK.")
