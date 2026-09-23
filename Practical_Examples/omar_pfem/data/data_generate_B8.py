"""FEM geometry/BC/material generator for B8-FINAL: a 3D laminated
annular elastomeric seismic bearing -- Option B, rebuilt 2026-09-23 from
a real published bearing design after real limitations in the earlier
prototype (data_generate_B8_prototype.py, archived unmodified) were
caught in review:

1. GEOMETRY was nondimensional/made-up, not a real published design.
   Fixed: real bearing dimensions from Kalantari & Rofooei, 10th
   Canadian Conference on Earthquake Engineering, 2010 (caee.ca/
   10CCEEpdf/2010EQConf-000137.pdf) -- outer diameter 152 mm, inner
   (central hole) diameter 30 mm, 20 rubber layers x 3 mm each, 19
   steel shims x 3 mm each (20 rubber bands, 19 internal shim bands --
   this project's own existing rubber/shim/rubber/.../rubber band
   pattern, from layer_bands() below, already matches this exactly).
   All lengths here are in millimeters (a real, intentional departure
   from B1/B2/B3's own nondimensional convention -- justified because
   this design is now matched to a specific real, cited structure, not
   an arbitrary nondimensional benchmark).

2. MATERIALS were an arbitrary 100x rubber-stiffness shim proxy, not
   real materials. Fixed: real Neo-Hookean rubber (same source) --
   G=0.68 MPa, C10=0.34 MPa, D1=0.00099 MPa^-1, K=2000 MPa (checked
   directly: C10=G/2 and K=2/D1=2020 MPa both consistent with the
   quoted G, K to within the source's own rounding) -- and REAL
   linear-elastic steel, E=200 GPa, nu=0.3 (this source's own value;
   standard structural steel). Corrects an earlier, wrongly-remembered
   G=0.86 MPa figure that was never actually checked against a source.

3. The SHIM MATERIAL MODEL was itself wrong in kind, not just in
   magnitude: real steel is linear-elastic, not "very stiff Neo-
   Hookean." Fixed: checked directly against torch-fem's own API
   (Hyperelastic3D supports per-element VECTORIZED PARAMETERS of one
   psi function, not per-element different psi functions/material
   classes; Solid takes exactly one material object) before deciding
   how to represent this -- mixing torchfem.materials.
   IsotropicElasticity3D (real linear elasticity) with Hyperelastic3D
   rubber in one Solid model is not directly supported. Real,
   implementable fix: omar_pfem.torchfem_comparison.
   neo_hookean_or_stvk_psi_3d dispatches PER ELEMENT (via a flag baked
   into the same per-element vectorized params Hyperelastic3D already
   supports) between real compressible Neo-Hookean (rubber) and real
   St. Venant-Kirchhoff (steel) -- St. Venant-Kirchhoff is the standard
   way to get genuinely linear-elastic material behavior (same mu,
   lambda as classical linear elasticity, verified directly to match a
   small-strain reference to 8e-5 relative) inside a large-rotation-
   capable hyperelastic framework, which is exactly what a shim
   undergoing large rigid-body-like rotation but negligible internal
   strain needs. See that function's own docstring for the full
   derivation and the finite-Hessian-at-F=I check.

MODELING CHOICE KEPT FROM THE PROTOTYPE (still correct, not revisited):
shims are meshed as real hex8 elements in the SAME connected mesh as
the rubber (node-sharing enforces displacement/traction continuity
automatically) rather than a rigid multi-point constraint -- confirmed
in the prototype's own review that torch-fem has no general rigid MPC
mechanism, and this remains the right, implementable choice regardless
of which material law the shim elements use.

GEOMETRY STRUCTURE (unchanged from the prototype): annular ring
cross-section (R_in > 0, a real central hole) extruded along z through
alternating bands: rubber, shim, rubber, shim, ..., rubber
(n_rubber_layers rubber bands, n_rubber_layers-1 internal shim bands).
Reuses data_generate_B2.generate_grid_Q4_ring exactly (same node/element
index convention as B2/B3) for the ring cross-section, stacked per
z-layer exactly as data_generate_B3.generate_grid_hex8_bushing does.
theta in [0, pi] (half-cylinder): valid by the SAME mirror-symmetry
argument B3 uses.

BOUNDARY CONDITIONS (unchanged): bottom face (z=0) fully fixed (bonded
to the foundation). Top face (z=Lz) gets a PRESCRIBED RIGID-BODY
displacement (vertical compression + horizontal shear + optional small
rocking about y) -- a boundary condition, not a meshed body, matching
the source's own 25mm-thick end plates being effectively rigid compared
to the rubber (not meshed here, same simplification B3 uses for its own
rigid core).

STRESS-QOI REGION (REVISED, real fix, not the prototype's): the
prototype centered its region at the shim's own mid-height, spanning
BOTH rubber and steel, and its cross-mesh comparison interpolated a
field ACROSS that material discontinuity -- not a clean QoI definition.
`first_rubber_layer_interface_z` below instead gives the z-coordinate
of the rubber/shim-1 interface itself; mesh_convergence_B8.py's own
region mask is built to sample ONLY rubber elements strictly inside
rubber layer 1 (near that interface, at the outer free edge), with the
cross-mesh comparison interpolator scoped to rubber layer 1's own
element range only -- never crossing into shim elements.
"""
import numpy as np

from omar_pfem.data.data_generate_B2 import generate_grid_Q4_ring

# ---- real bearing geometry (mm), Kalantari & Rofooei 2010 ----
R_IN, R_OUT = 15.0, 76.0             # inner hole / outer radius, mm (30mm/152mm diameters)
N_RUBBER_LAYERS = 20
T_RUBBER = 3.0                       # mm, per rubber layer
T_SHIM = 3.0                         # mm, per steel shim
NZ_PER_RUBBER = 2                    # element layers through each rubber band's thickness
NZ_PER_SHIM = 2                      # element layers through each shim band's thickness

# ---- real materials, same source ----
# Rubber: compressible Neo-Hookean. G=0.68 MPa (=2*C10=2*0.34, checked
# directly against the source), K=2000 MPa (=2/D1=2/0.00099=2020 MPa,
# consistent with the source's own rounding). mu=G exactly; lam from the
# standard small-strain relation K=lam+(2/3)*mu (the same relation this
# project's own E,NU->mu,lam helper elsewhere already relies on for how
# this exact hyperelastic form behaves near the reference state).
G_RUBBER = 0.68     # MPa
K_RUBBER = 2000.0   # MPa
MU_RUBBER = G_RUBBER
LAM_RUBBER = K_RUBBER - (2.0 / 3.0) * MU_RUBBER

# Steel: real linear-elastic (via St. Venant-Kirchhoff, see
# omar_pfem.torchfem_comparison.neo_hookean_or_stvk_psi_3d), standard
# Lame relations from E, nu.
E_STEEL = 200_000.0   # MPa (200 GPa)
NU_STEEL = 0.30
MU_STEEL = E_STEEL / (2 * (1 + NU_STEEL))
LAM_STEEL = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))


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


def build_z_axis(n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim):
    """Single source of truth for the z-grid: node z-coordinates (zs) and,
    per z-element-layer (index k, 0..Nz-2), whether that layer is a shim
    layer (layer_is_shim). Used by BOTH the geometry generator below and
    mesh_convergence_B8.py's cross-mesh interpolation, so the two can never
    silently drift apart."""
    z_boundaries, band_is_shim, band_nz = layer_bands(
        n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim)
    zs = [z_boundaries[0]]
    for b in range(len(band_nz)):
        z0, z1 = z_boundaries[b], z_boundaries[b + 1]
        sub = np.linspace(z0, z1, band_nz[b] + 1)[1:]  # skip z0 (already added)
        zs.extend(sub.tolist())
    zs = np.array(zs)
    layer_is_shim = []
    for b in range(len(band_nz)):
        layer_is_shim.extend([band_is_shim[b]] * band_nz[b])
    assert len(layer_is_shim) == len(zs) - 1
    return zs, layer_is_shim


def first_rubber_layer_z_range(n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim):
    """(z0, z1, k0, k1): the z-range and element-layer-index range of the
    FIRST rubber band (z0=0 at the fixed foundation, z1=t_rubber at the
    rubber/shim-1 interface), and [k0,k1) the z-element-layer indices
    belonging to it. Used to scope BOTH the region-Cauchy QoI (a point
    near this band's own outer free edge, close to its z1 interface) and
    the cross-mesh comparison interpolator to rubber elements ONLY,
    never crossing into the adjacent shim band's own different material
    response."""
    zs, layer_is_shim = build_z_axis(n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim)
    k0 = 0
    k1 = layer_is_shim.index(True)  # first shim layer index == end of first rubber band
    return float(zs[k0]), float(zs[k1]), k0, k1


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
    zs, layer_is_shim = build_z_axis(n_rubber_layers, nz_per_rubber, nz_per_shim, t_rubber, t_shim)
    Lz = float(zs[-1])
    Nz = len(zs)

    nodes2d, elements2d = generate_grid_Q4_ring(R_in, R_out, Ntheta, Nr,
                                                 theta_max=theta_max, r_grading=r_grading)
    n2d = Ntheta * Nr

    nodes3d = np.zeros((n2d * Nz, 3))
    for k, z in enumerate(zs):
        nodes3d[k * n2d:(k + 1) * n2d, 0:2] = nodes2d
        nodes3d[k * n2d:(k + 1) * n2d, 2] = z

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
    core, and consistent with the source's own 25mm end plates being
    effectively rigid next to the rubber): uniform vertical compression
    (uz = -compression, e.g. compression=0.02*Lz for 2% nominal
    compressive strain), uniform horizontal shear (ux = shear_x,
    representing lateral seismic drift), and an optional small rigid
    rotation phi_y about the y-axis. Since the rotation is about y and uy
    stays 0 identically (checked directly, same reasoning as B3's own
    rigid_rotation_displacement), the y=0 plane remains a genuine
    mirror-symmetry plane for this combined load."""
    x0, y0, z0 = nodes_top[:, 0], nodes_top[:, 1], nodes_top[:, 2]
    x_rot = x0 * np.cos(phi_y) + z0 * np.sin(phi_y) - x0
    z_rot = -x0 * np.sin(phi_y) + z0 * np.cos(phi_y) - z0
    ux = x_rot + shear_x
    uy = np.zeros_like(y0)
    uz = z_rot - compression
    return ux, uy, uz


def build_vectorized_material_params(element_is_shim,
                                      mu_rubber=MU_RUBBER, lam_rubber=LAM_RUBBER,
                                      mu_steel=MU_STEEL, lam_steel=LAM_STEEL):
    """Per-element [mu, lambda, is_shim] triples (n_elem, 3), for
    omar_pfem.torchfem_comparison.neo_hookean_or_stvk_psi_3d's own
    per-element MATERIAL dispatch (real Neo-Hookean rubber vs. real
    St. Venant-Kirchhoff steel -- see that function's own docstring),
    via Hyperelastic3D's existing vectorized-parameter support
    (params.dim() > 1 -> is_vectorized=True, confirmed directly against
    torchfem/materials.py). Real material constants throughout -- no
    stiffness-ratio shortcut."""
    params = np.where(
        element_is_shim[:, None],
        np.array([mu_steel, lam_steel, 1.0]),
        np.array([mu_rubber, lam_rubber, 0.0]))
    return params
