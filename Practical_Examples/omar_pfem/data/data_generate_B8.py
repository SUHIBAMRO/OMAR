"""FEM geometry/BC generator for B8: a 3D laminated annular elastomeric
seismic bearing -- Option B from the 2026-09-22 candidate-direction
discussion, proposed as a genuinely harder 3D benchmark than B3 for
Timon's stated target (10^5-10^6 elements, 5-10% QoI error).

PHYSICAL DESIGN: a real seismic isolation bearing is a stack of thin
rubber layers bonded between steel shim plates, with a central hole
(often housing a lead core for energy dissipation, not modeled here).
The shims force the rubber's near-incompressible bulging into a thin,
laterally-confined shape under vertical load, while the whole stack
stays flexible in horizontal shear -- both effects are well documented
in the literature as genuinely hard for FEM: near-incompressibility
(this project already uses NU=0.45 for rubber, mild but present) and a
FREE-EDGE stress concentration at the outer radius, where each shim
ends abruptly and the rubber is free to bulge -- a much sharper,
naturally-occurring feature than B3's deliberately smoothed groove.

MODELING CHOICE FOR THE SHIMS (real constraint, not a shortcut): the
originally-discussed idea was "shims as a rigid boundary condition, not
a second deformable material," to keep a single-material-model scope.
Checked directly against torch-fem's own API before writing this code:
`Hyperelastic3D.__init__` accepts `params: list | Tensor` and sets
`is_vectorized = params.dim() > 1` (torchfem/materials.py) -- i.e.
torch-fem supports genuinely PER-ELEMENT material parameters within the
SAME Neo-Hookean model, but has no general rigid multi-point-constraint
mechanism in its Dirichlet-BC-based `Solid.constraints`/`displacements`
API (only per-node prescribed values, not "these N nodes share one
unknown rigid motion"). A true floating rigid MPC for the internal
shims would need Lagrange multipliers or constraint elimination this
project's FEM stack does not have. `torchfem/laminate.py`'s `Laminate`
class was also checked directly and is NOT applicable: it implements
classical lamination theory for SHELL elements (thin plate/shell
plane-stress plies), not through-thickness 3D solid layers -- it cannot
represent the real 3D bulging/near-incompressibility behavior this
benchmark is specifically about.

REVISED, IMPLEMENTABLE CHOICE: mesh the shims as hex8 elements in the
SAME connected mesh as the rubber (no interface treatment needed --
node-sharing already enforces displacement/traction continuity), using
the SAME Neo-Hookean psi function with per-element (vectorized) params:
a much higher modulus for shim elements than rubber elements. This is a
standard, realistic FEM idealization of "near-rigid steel shim" (a
large but FINITE stiffness contrast, not an artificial infinite-rigid
constraint) -- and it is honestly still "one material model throughout"
in the sense Omar's scope constraint cared about (same psi, same
constitutive law), just with spatially-varying parameters, which
Hyperelastic3D already supports natively. The stiffness ratio is a
tunable, documented modeling choice (see SHIM_STIFFNESS_RATIO below),
not the full real steel/rubber ratio (~1e5), which would very likely
reproduce the same catastrophic linear-solver ill-conditioning already
confirmed directly in mesh_convergence_B3_groove_sharpness.py's r_grading
experiments -- start moderate, verified to actually solve, and revisit
only if a harder problem is later needed.

GEOMETRY: annular ring cross-section (R_in > 0, a real central hole,
matching "annular" in the design name) extruded along z through
alternating bands: rubber, shim, rubber, shim, ..., rubber
(n_rubber_layers rubber bands, n_rubber_layers-1 internal shim bands).
Reuses data_generate_B2.generate_grid_Q4_ring exactly (same node/element
index convention as B2/B3) for the ring cross-section, stacked per
z-layer exactly as data_generate_B3.generate_grid_hex8_bushing does.
theta in [0, pi] (half-cylinder): valid by the SAME mirror-symmetry
argument B3 uses -- the combined top-plate load below (shear along x,
compression along z, optional rocking about the y-axis) has ux, uz
depending only on (x0, z0) and uy=0 identically, so the y=0 plane is a
genuine symmetry plane (checked by the same reasoning as B3's
rigid_rotation_displacement, not re-derived from scratch: a rotation
about y and a translation confined to the x-z plane both leave y=0
invariant).

BOUNDARY CONDITIONS: bottom face (z=0, bottom of the first rubber
layer) fully fixed (bonded to the foundation) -- like B3's outer
housing. Top face (z=Lz, top of the last rubber layer) gets a
PRESCRIBED RIGID-BODY displacement (vertical compression + horizontal
shear + optional small rocking about y) -- like B3's inner core, i.e.
the top mounting plate itself is not meshed as a deformable body, its
motion is a boundary condition, exactly the same simplification B3
already uses for its own rigid core. Central hole (r=R_in) and outer
surface (r=R_out) are both genuine free surfaces -- the outer surface
at each internal rubber-shim interface is where the free-edge stress
concentration this benchmark is about actually occurs.
"""
import numpy as np

from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring

# ---- default physical parameters (nondimensional, same convention as
# B1/B2/B3) ----
R_IN, R_OUT = 0.3, 1.0
N_RUBBER_LAYERS = 4
T_RUBBER = 0.10     # thickness of each rubber layer
T_SHIM = 0.02       # thickness of each internal shim layer (5:1 ratio)
NZ_PER_RUBBER = 3   # element layers through each rubber band's thickness
NZ_PER_SHIM = 2      # element layers through each shim band's thickness (thin)

# Rubber: SAME Neo-Hookean parameters used throughout this project's B1/B2/B3.
E_RUBBER, NU_RUBBER = 1000.0, 0.45
# Shim: same Neo-Hookean psi, much higher modulus -- a documented, moderate
# stiffness-contrast idealization of near-rigid steel, NOT the full real
# steel/rubber ratio (~1e5), which is expected to reproduce the same
# CG ill-conditioning already found and documented for aggressive mesh
# grading in mesh_convergence_B3_groove_sharpness.py. Start here; only
# increase if solver behavior at this ratio is confirmed acceptable AND a
# harder problem is later wanted.
SHIM_STIFFNESS_RATIO = 100.0
E_SHIM = E_RUBBER * SHIM_STIFFNESS_RATIO
NU_SHIM = 0.30  # generic near-rigid-solid Poisson ratio, not tuned to real steel


def layer_bands(n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim):
    """Returns (z_boundaries, band_is_shim, band_nz): the z-coordinate of
    every band boundary (band 0 = [z_boundaries[0], z_boundaries[1]],
    etc.), whether each band is a shim band, and how many z-element-layers
    each band gets. Band order: rubber, shim, rubber, shim, ..., rubber
    (n_rubber_layers rubber bands, n_rubber_layers-1 shim bands)."""
    assert n_rubber_layers >= 2, "need at least 2 rubber layers to have an internal shim"
    z = 0.0
    z_boundaries = [z]
    band_is_shim = []
    band_nz = []
    for layer in range(n_rubber_layers):
        z += t_rubber
        z_boundaries.append(z)
        band_is_shim.append(False)
        band_nz.append(nz_per_rubber)
        if layer < n_rubber_layers - 1:
            z += t_shim
            z_boundaries.append(z)
            band_is_shim.append(True)
            band_nz.append(nz_per_shim)
    return np.array(z_boundaries), band_is_shim, band_nz


def generate_grid_hex8_laminated_bearing(R_in, R_out, Ntheta, Nr,
                                          n_rubber_layers=N_RUBBER_LAYERS,
                                          nz_per_rubber=NZ_PER_RUBBER,
                                          nz_per_shim=NZ_PER_SHIM,
                                          t_rubber=T_RUBBER, t_shim=T_SHIM,
                                          r_grading=1.0, theta_max=np.pi):
    """Half-cylinder (theta in [0, theta_max]) annular ring cross-section
    stacked along z through alternating rubber/shim bands. Returns
    (nodes, elements, element_is_shim, Lz): nodes (n_nodes,3), elements
    (n_elem,8) hex8 connectivity (same winding/ordering convention as
    data_generate_B3.generate_grid_hex8_bushing), element_is_shim (n_elem,)
    bool mask, Lz total height."""
    z_boundaries, band_is_shim, band_nz = layer_bands(
        n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim)
    Lz = float(z_boundaries[-1])

    nodes2d, elements2d = generate_grid_Q4_ring(R_in, R_out, Ntheta, Nr,
                                                 theta_max=theta_max, r_grading=r_grading)
    n2d = Ntheta * Nr

    zs = [z_boundaries[0]]
    for b in range(len(band_nz)):
        z0, z1 = z_boundaries[b], z_boundaries[b + 1]
        sub = np.linspace(z0, z1, band_nz[b] + 1)[1:]  # skip z0 (already added)
        zs.extend(sub.tolist())
    zs = np.array(zs)
    Nz = len(zs)

    nodes3d = np.zeros((n2d * Nz, 3))
    for k, z in enumerate(zs):
        nodes3d[k * n2d:(k + 1) * n2d, 0:2] = nodes2d
        nodes3d[k * n2d:(k + 1) * n2d, 2] = z

    # Per-z-LAYER (element layer index k, 0..Nz-2) shim flag, expanded from
    # per-band flags via each band's own element-layer count.
    layer_is_shim = []
    for b in range(len(band_nz)):
        layer_is_shim.extend([band_is_shim[b]] * band_nz[b])
    assert len(layer_is_shim) == Nz - 1

    elements3d = []
    element_is_shim = []
    for k in range(Nz - 1):
        off0, off1 = k * n2d, (k + 1) * n2d
        for quad in elements2d:
            n1, n2, n3, n4 = quad
            elements3d.append([n1 + off0, n2 + off0, n3 + off0, n4 + off0,
                                n1 + off1, n2 + off1, n3 + off1, n4 + off1])
            element_is_shim.append(layer_is_shim[k])
    return nodes3d, np.array(elements3d, dtype=int), np.array(element_is_shim, dtype=bool), Lz


def boundary_node_sets(nodes, R_in, R_out, Lz, tol=1e-9):
    """Returns (bottom_face, top_face, outer_surface, inner_hole,
    symmetry_y0) boolean masks. bottom_face (z=0): bonded to the fixed
    foundation. top_face (z=Lz): gets the prescribed rigid top-plate
    motion. outer_surface (r=R_out) and inner_hole (r=R_in) are both free
    surfaces -- the outer one is where the free-edge stress concentration
    at each internal rubber-shim interface occurs."""
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    bottom_face = np.abs(z - 0.0) < tol
    top_face = np.abs(z - Lz) < tol
    outer_surface = np.abs(r - R_out) < tol
    inner_hole = np.abs(r - R_in) < tol
    symmetry_y0 = np.abs(y) < tol
    return bottom_face, top_face, outer_surface, inner_hole, symmetry_y0


def rigid_top_plate_displacement(nodes_top, Lz, compression, shear_x, phi_y=0.0):
    """Prescribed RIGID displacement for the top mounting plate (a boundary
    condition, not a meshed deformable body -- same simplification
    data_generate_B3.rigid_rotation_displacement uses for its own rigid
    core): uniform vertical compression (uz = -compression, a fraction of
    Lz, e.g. compression=0.02*Lz for 2% nominal compressive strain),
    uniform horizontal shear (ux = shear_x, representing lateral seismic
    drift), and an optional small rigid rotation phi_y about the y-axis
    (combined shear + rocking, matching the seismic-bearing design brief).
    Since the rotation is about y and uy stays 0 identically (checked
    directly, same reasoning as B3's own rigid_rotation_displacement:
    x' = x*cos(phi)+z*sin(phi), z' = -x*sin(phi)+z*cos(phi), y'=y), the
    y=0 plane remains a genuine mirror-symmetry plane for this combined
    load, exactly as it does for B3's pure rocking."""
    x0, y0, z0 = nodes_top[:, 0], nodes_top[:, 1], nodes_top[:, 2]
    x_rot = x0 * np.cos(phi_y) + z0 * np.sin(phi_y) - x0
    z_rot = -x0 * np.sin(phi_y) + z0 * np.cos(phi_y) - z0
    ux = x_rot + shear_x
    uy = np.zeros_like(y0)
    uz = z_rot - compression
    return ux, uy, uz


def build_vectorized_neo_hookean_params(element_is_shim,
                                         E_rubber=E_RUBBER, NU_rubber=NU_RUBBER,
                                         E_shim=E_SHIM, NU_shim=NU_SHIM):
    """Per-element [mu, lambda] pairs (n_elem, 2), for Hyperelastic3D's own
    vectorized-material path (params.dim() > 1 -> is_vectorized=True,
    confirmed directly against torchfem/materials.py before writing this
    module). SAME psi (neo_hookean_psi_3d) is used for every element
    regardless of rubber/shim -- only the two scalar parameters differ."""
    mu_r = E_rubber / (2 * (1 + NU_rubber))
    lam_r = E_rubber * NU_rubber / ((1 + NU_rubber) * (1 - 2 * NU_rubber))
    mu_s = E_shim / (2 * (1 + NU_shim))
    lam_s = E_shim * NU_shim / ((1 + NU_shim) * (1 - 2 * NU_shim))
    params = np.where(element_is_shim[:, None], np.array([mu_s, lam_s]), np.array([mu_r, lam_r]))
    return params
