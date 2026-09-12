"""
FEM ground-truth generator for a NEW benchmark case, B7: a quarter ring
under internal pressure (same base geometry/BCs as B2) but with a single
smooth LOCAL notch (a local reduction in wall thickness) cut into the
inner arc at one angular position -- Timon round-10, item 4: "the fine
spatial resolution is actually required by the geometry ... e.g. due to
local stress concentrations ... could be a tire, pressure vessels with
local details."

This is a genuine, non-simplified instance of "a pressure vessel with a
local detail," not an approximation of one: B2 is already a pressure
vessel cross-section (a ring under internal pressure); this adds the one
thing B2's own uniform geometry lacks -- a local geometric feature whose
own stress concentration should genuinely require finer resolution to
capture than the smooth baseline does, which is the actual property Timon
asked for, independent of whether the shape is literally called a "tire."

Reuses B2's own solver machinery unchanged (materials, Q4 shape
functions, element internal-force/tangent, Dirichlet elimination,
Newton-Raphson loop) -- the ONLY two things that need to change for a
notched inner boundary are (1) the mesh generator, since B2's own inner
radius is a single constant, and (2) the inner-arc traction assembler,
since B2's own version detects the loaded boundary by checking each
node's distance from that SAME constant -- both fail silently (not
loudly) on a notched boundary, so both are reimplemented here rather than
monkeypatched.

Geometry: same structured (r fast index i, theta slow index j) grid B2
uses, but the inner radius is now a smooth function of theta,
    R_in(theta) = R_in_base - notch_depth * exp(-0.5*((theta-notch_theta0)/notch_width)**2),
a single Gaussian dimple centered at notch_theta0, so the boundary stays
smooth (C-infinity, no re-entrant corner) but locally curves inward --
still simply connected, no change to element connectivity/topology, so
every downstream routine (materials, solver) needs no awareness of the
notch at all beyond correct node positions and correct traction loading.
"""
import numpy as np

from omar_pfem.data.fem_core import apply_dirichlet, element_K_and_fint_TL, shape_Q4
from omar_pfem.data.materials import get_material_fns


def r_in_of_theta(theta, R_in_base, notch_depth, notch_theta0, notch_width):
    return R_in_base - notch_depth * np.exp(-0.5 * ((theta - notch_theta0) / notch_width) ** 2)


def generate_grid_Q4_ring_notch(R_in_base, R_out, Ntheta, Nr, theta_max=np.pi / 2,
                                 notch_depth=0.3, notch_theta0=None, notch_width=0.15,
                                 r_grading=1.0):
    """Same index convention as data_generate_B2.generate_grid_Q4_ring
    (r fast index i, theta slow index j) -- only the per-column inner
    radius differs, so every downstream index-based routine (element
    connectivity, boundary-by-index detection) is unaffected by the
    notch. notch_theta0 defaults to the ARC MIDPOINT (theta_max/2), away
    from both symmetry edges, so the notch never touches either straight
    boundary's own BCs."""
    if notch_theta0 is None:
        notch_theta0 = theta_max / 2.0

    thetas = np.linspace(0.0, theta_max, Ntheta)
    t = np.linspace(0.0, 1.0, Nr)

    nodes = np.zeros((Ntheta * Nr, 2), dtype=float)
    r_in_theta = r_in_of_theta(thetas, R_in_base, notch_depth, notch_theta0, notch_width)
    for j, (theta, r_in_j) in enumerate(zip(thetas, r_in_theta)):
        rs = r_in_j + (R_out - r_in_j) * (t ** r_grading)
        nodes[j * Nr:(j + 1) * Nr, 0] = rs * np.cos(theta)
        nodes[j * Nr:(j + 1) * Nr, 1] = rs * np.sin(theta)

    elements = []
    for j in range(Ntheta - 1):
        for i in range(Nr - 1):
            n1 = j * Nr + i
            n2 = n1 + 1
            n3 = (j + 1) * Nr + i + 1
            n4 = (j + 1) * Nr + i
            elements.append([n1, n2, n3, n4])
    return nodes, np.array(elements, dtype=int), r_in_theta


def assemble_traction_inner_indexed(nodes, elements, Nr, p_interp):
    """Same physics/quadrature as data_generate_B2.assemble_traction_inner_
    curved (per-edge chord outward normal, force-consistent curved-boundary
    loading), but detects the loaded boundary by INDEX (i=0, i.e.
    node_index % Nr == 0) instead of by distance from a constant R_in --
    the only change needed for a notched (theta-dependent) inner radius.
    Evaluates p_interp at the edge's own ACTUAL local (theta, r), not a
    hardcoded R_in, so the pressure field is sampled at the true physical
    boundary location even where the notch has moved it."""
    n_nodes = nodes.shape[0]
    Fext = np.zeros(2 * n_nodes, dtype=float)

    g = 1.0 / np.sqrt(3.0)
    gps = [-g, g]
    ws = [1.0, 1.0]

    for e in elements:
        if e[0] % Nr != 0 or e[3] % Nr != 0:
            continue
        Xe = nodes[e]
        X1, X4 = Xe[0], Xe[3]
        edge_vec = X4 - X1
        edge_len = np.linalg.norm(edge_vec)
        tangent_hat = edge_vec / edge_len

        normal = np.array([tangent_hat[1], -tangent_hat[0]])
        midpoint = 0.5 * (X1 + X4)
        if np.dot(normal, midpoint) < 0:
            normal = -normal

        for (eta, w1d) in zip(gps, ws):
            xi = -1.0
            N, _ = shape_Q4(xi, eta)

            pos = N @ Xe
            r_pos = np.linalg.norm(pos)
            theta_pos = np.arctan2(pos[1], pos[0])

            p_value = p_interp(np.array([[theta_pos, r_pos]]))[0]
            t = p_value * normal

            Jedge = edge_len / 2.0
            for a in range(4):
                fa = N[a] * t * (w1d * Jedge)
                node_idx = e[a]
                Fext[2 * node_idx:2 * node_idx + 2] += fa
    return Fext


def solve_hyperelastic_TL_ring_notch(nodes, elements, Nr, E_grid, nu_grid, p_grid,
                                      nsteps=10, newton_max=25, tol=1e-8,
                                      material="neo_hookean", verbose=True):
    """Thin copy of data_generate_B2.solve_hyperelastic_TL_ring, differing
    only in which traction assembler it calls (assemble_traction_inner_
    indexed, notch-aware, instead of the constant-R_in one) -- symmetry
    BCs (theta=0 -> u_y=0, theta=theta_max -> u_x=0) are geometry checks
    on y=0/x=0 that hold regardless of the notch, since the notch sits at
    an interior theta away from both straight edges."""
    PK1_and_tangent_fn, E_nu_to_params_fn = get_material_fns(material)

    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes
    u = np.zeros(ndof, dtype=float)

    tolx = 1e-9
    theta0_nodes = np.where(np.abs(nodes[:, 1]) < tolx)[0]
    thetahalfpi_nodes = np.where(np.abs(nodes[:, 0]) < tolx)[0]
    fixed_dofs = []
    for n in theta0_nodes:
        fixed_dofs.append(2 * n + 1)
    for n in thetahalfpi_nodes:
        fixed_dofs.append(2 * n)

    Fext_full = assemble_traction_inner_indexed(nodes, elements, Nr, p_grid)

    def _material_fn(X_gp):
        r_gp = np.linalg.norm(X_gp)
        theta_gp = np.arctan2(X_gp[1], X_gp[0])
        E_val = E_grid(np.array([[theta_gp, r_gp]]))[0]
        nu_val = nu_grid(np.array([[theta_gp, r_gp]]))[0]
        return E_nu_to_params_fn(E_val, nu_val)

    for step in range(1, nsteps + 1):
        alpha = step / nsteps
        Fext = alpha * Fext_full

        for it in range(1, newton_max + 1):
            rows, cols, vals = [], [], []
            fint = np.zeros(ndof, dtype=float)

            for e in elements:
                Xe = nodes[e]
                ue = u.reshape(n_nodes, 2)[e].reshape(-1)

                ke, fe_int = element_K_and_fint_TL(
                    Xe, ue, None, PK1_and_tangent_fn, material_fn=_material_fn)

                edofs = []
                for a in e:
                    edofs += [2 * a, 2 * a + 1]
                for i_local, I in enumerate(edofs):
                    fint[I] += fe_int[i_local]
                    for j_local, J in enumerate(edofs):
                        rows.append(I)
                        cols.append(J)
                        vals.append(ke[i_local, j_local])

            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import spsolve

            K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
            R = fint - Fext

            free, Kff, Rf = apply_dirichlet(K, R, fixed_dofs)

            res_norm = np.linalg.norm(Rf)
            if res_norm < tol:
                if verbose:
                    print(f"[step {step}/{nsteps}] converged in {it - 1} iters, "
                          f"||R||={res_norm:.3e}")
                break

            du_free = spsolve(Kff, -Rf)
            u[free] += du_free

            if it == newton_max and verbose:
                print(f"[step {step}/{nsteps}] NOT converged, last ||R||={res_norm:.3e}")

    return u.reshape(n_nodes, 2)
