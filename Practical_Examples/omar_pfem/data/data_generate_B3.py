"""B3: the new 3D "realistic case" (Timon's newest email, item 4).

Geometry: a quarter-symmetry model of a finite-thickness square plate of
side 2*Lx (occupying x,y in [0,Lx]x[0,Ly], Lx==Ly by convention) and
thickness Lz, with a circular through-hole of radius R centered on the
z-axis (hole axis parallel to z), loaded by a face traction in +x on the
x=Lx face. This is the classic "plate with a circular hole under
uniaxial tension" stress-concentration benchmark, extended to finite
thickness -- unlike B7 (this project's earlier 2D ring+notch "realistic
case," see PROJECT_STATUS.md), the z=0/z=Lz faces are genuinely
traction-free surfaces, so the through-thickness stress state is NOT a
simple extrusion of a 2D solution (2D plane-strain has nonzero sigma_zz
everywhere; here sigma_zz must vanish at the free surfaces and only
approaches the plane-strain value deep in the interior) -- a real,
well-known 3D effect that only a genuine 3D solve captures, which is
also why this case naturally needs a finer mesh (through-thickness
resolution, not just in-plane resolution) than B1/B2 ever did.

Single material model (Neo-Hookean, per the advisor's own request to
keep this one case to one material) -- reuses `neo_hookean_psi_3d` from
torchfem_comparison.py UNCHANGED (already GPU-validated at N=1401 for
this project's own 2D cases; it is written for a genuine 3x3 F all
along, so it needs no modification for real 3D use).

Mesh: a "mapped" (transfinite) quarter-annulus, structured in
(theta, r-like parameter t) exactly like B2's own generate_grid_Q4_ring
(reused pattern, not a new mesh-generation idea), except the OUTER
boundary at t=1 is not a fixed radius but the point where the ray at
angle theta meets the square's own two straight edges (x=Lx or y=Ly) --
this makes the 2D cross-section a proper "plate with hole", not a ring.
Extruded along z into HEX8 (Hexa1) elements by stacking Nz layers; the
extrusion adds no new mesh-generation risk since Hexa1's own bottom-face
node order (torchfem.elements.Hexa1) is identical to Q4's own
(-1,-1),(1,-1),(1,1),(-1,1) convention, so B2's already-CCW quad
connectivity is reused unchanged as each hex's bottom face, with the
top face being the same 2D connectivity at the next z-layer.
"""
import numpy as np


def generate_grid_Q4_plate_with_hole(Lx, Ly, R, Ntheta, Nr, r_grading=1.0, theta_max=np.pi / 2):
    """Quarter-plane (x,y >= 0) square domain [0,Lx]x[0,Ly] minus the
    quarter-disk of radius R at the origin. theta=0 row lies exactly on
    y=0 (the symmetry-y0 boundary); theta=theta_max row lies exactly on
    x=0 (the symmetry-x0 boundary); the i=0 (t=0) column is the hole
    boundary (r=R, traction-free); the i=Nr-1 (t=1) column is the
    outer boundary, lying on x=Lx for theta below the corner angle
    atan2(Ly,Lx) and on y=Ly above it (identify the loaded x=Lx face by
    node COORDINATE, not by this parametrization, since it is robust to
    any Ntheta)."""
    thetas = np.linspace(0.0, theta_max, Ntheta)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    eps = 1e-12
    s_x = np.where(cos_t > eps, Lx / np.maximum(cos_t, eps), np.inf)
    s_y = np.where(sin_t > eps, Ly / np.maximum(sin_t, eps), np.inf)
    s = np.minimum(s_x, s_y)  # ray-to-square-boundary distance, per theta

    t = np.linspace(0.0, 1.0, Nr)
    tt = t ** r_grading
    Rr = R + np.outer(s - R, tt)  # (Ntheta, Nr): radius(theta_j, t_i)
    TH = np.outer(thetas, np.ones(Nr))
    X = Rr * np.cos(TH)
    Y = Rr * np.sin(TH)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T

    elements = []
    for j in range(Ntheta - 1):
        for i in range(Nr - 1):
            n1 = j * Nr + i
            n2 = n1 + 1
            n3 = (j + 1) * Nr + i + 1
            n4 = (j + 1) * Nr + i
            elements.append([n1, n2, n3, n4])
    return nodes, np.array(elements, dtype=int)


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


def generate_grid_hex8_plate_with_hole(Lx, Ly, Lz, R, Ntheta, Nr, Nz, r_grading=1.0):
    nodes2d, elements2d = generate_grid_Q4_plate_with_hole(Lx, Ly, R, Ntheta, Nr, r_grading)
    return extrude_to_hex8(nodes2d, elements2d, Lz, Nz)


def boundary_node_sets(nodes, Lx, Ly, tol=1e-9):
    """Returns (symmetry_y0, symmetry_x0, loaded_x_face, free_y_face,
    hole_nodes) -- boolean masks over `nodes` (N,3). symmetry_y0: y=0
    plane (u_y=0). symmetry_x0: x=0 plane (u_x=0). loaded_x_face: x=Lx
    face (traction applied here, +x direction). free_y_face: y=Ly face
    (left traction-free, NOT a symmetry plane -- it is the plate's own
    real, physical free edge, far from the hole). hole_nodes: r=R
    surface (traction-free), returned for reference/plotting only, no
    BC needed there beyond the natural (already traction-free) condition."""
    x, y = nodes[:, 0], nodes[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    symmetry_y0 = np.abs(y) < tol
    symmetry_x0 = np.abs(x) < tol
    loaded_x_face = np.abs(x - Lx) < tol
    free_y_face = np.abs(y - Ly) < tol
    R_guess = r[np.abs(r - r.min()) < tol].min() if len(nodes) else None
    hole_nodes = np.abs(r - R_guess) < tol if R_guess is not None else np.zeros(len(nodes), dtype=bool)
    return symmetry_y0, symmetry_x0, loaded_x_face, free_y_face, hole_nodes


if __name__ == "__main__":
    # Cheap structural smoke test (mesh only, no solve) -- run this file
    # directly to sanity-check node/element counts and BC-set sizes
    # before spending any solver time.
    Lx = Ly = 1.0
    Lz = 0.3
    R = 0.2
    Ntheta, Nr, Nz = 9, 7, 4
    nodes, elements = generate_grid_hex8_plate_with_hole(Lx, Ly, Lz, R, Ntheta, Nr, Nz)
    print(f"nodes: {nodes.shape}, elements: {elements.shape}")
    assert nodes.shape == (Ntheta * Nr * Nz, 3)
    assert elements.shape == ((Ntheta - 1) * (Nr - 1) * (Nz - 1), 8)

    sym_y0, sym_x0, loaded, free_y, hole = boundary_node_sets(nodes, Lx, Ly)
    print(f"symmetry_y0: {sym_y0.sum()}, symmetry_x0: {sym_x0.sum()}, "
          f"loaded_x_face: {loaded.sum()}, free_y_face: {free_y.sum()}, hole: {hole.sum()}")
    assert sym_y0.sum() == Nr * Nz  # one full (r,z) grid on the y=0 plane
    assert sym_x0.sum() == Nr * Nz  # one full (r,z) grid on the x=0 plane
    assert loaded.sum() > 0 and free_y.sum() > 0
    assert hole.sum() == Ntheta * Nz  # one full (theta,z) grid on the hole surface

    # Jacobian sign check on every hex, at the element centroid, via the
    # standard trilinear shape-function derivative -- catches an inverted
    # (negative-volume) element from a node-ordering bug BEFORE any GPU
    # time is spent on a real solve.
    def hex_signed_volume(pts):
        # Split into 6 tets from node 0 (valid for a convex-ish hex; a
        # near-degenerate/inverted element will still show up as a sign
        # flip in at least one tet).
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6),
                (3, 4, 6, 7), (1, 4, 5, 6)]
        vol = 0.0
        for a, b, c, d in tets:
            vol += np.linalg.det(np.array([pts[b] - pts[a], pts[c] - pts[a], pts[d] - pts[a]])) / 6.0
        return vol

    bad = 0
    for el in elements:
        v = hex_signed_volume(nodes[el])
        if v <= 0:
            bad += 1
    print(f"inverted/degenerate elements: {bad} / {len(elements)}")
    assert bad == 0, "mesh has inverted elements -- fix node ordering before any solve"
    print("Mesh structural checks OK.")
