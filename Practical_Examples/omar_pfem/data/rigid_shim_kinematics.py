"""Exact finite-rotation rigid-body kinematics for B8's internal steel
shims, used by rigid_shim_solver.py.

Each shim is represented by its own 6 unknowns q = [t (3,), theta (3,)]:
a translation and a rotation VECTOR (axis-angle, |theta| = rotation
angle). The exact rotation matrix is the matrix exponential of the
skew-symmetric generator of theta (R = exp(skew(theta))), computed via
torch.linalg.matrix_exp -- not a linearized/small-angle approximation,
and not Rodrigues' formula typed out by hand (a common source of subtle
sign/convention bugs in finite-rotation FEM). A node originally at
position X (in the shim's own undeformed frame, relative to a fixed
reference point X_ref, itself the centroid of the shim's own undeformed
node set) moves to:

    x_new = X_ref + t + R(theta) @ (X - X_ref)
    u = x_new - X = t + R(theta) @ (X - X_ref) - (X - X_ref)

This is differentiated automatically (torch.func.jacrev), not by a
hand-derived closed-form rotation Jacobian, for the same reason this
project already prefers autodiff over hand-derived material tangents
elsewhere (omar_pfem.torchfem_comparison's own psi functions): it is
exact for whatever parameterization is coded, and removes an entire
class of possible sign/convention errors in a hand-derived formula.
"""
import numpy as np
import torch
from torch.func import jacrev


def skew(v):
    """Skew-symmetric (cross-product) matrix from a length-3 vector."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def rotation_matrix(theta):
    """Exact SO(3) rotation matrix from an axis-angle vector theta (3,),
    via the matrix exponential of its skew-symmetric generator -- valid
    for any rotation magnitude, not a small-angle approximation."""
    return torch.linalg.matrix_exp(skew(theta))


def rigid_body_displacement(q, X_rel):
    """q: (6,) = [t(3,), theta(3,)]. X_rel: (n_nodes, 3), undeformed node
    positions relative to the shim's own reference point (X - X_ref).
    Returns u: (n_nodes, 3) displacement of every node under this exact
    rigid transformation."""
    t, theta = q[:3], q[3:]
    R = rotation_matrix(theta)
    return t + (R @ X_rel.T).T - X_rel


def rigid_body_displacement_and_jacobian(q, X_rel):
    """Returns (u, J): u as in rigid_body_displacement, and J = du/dq
    with shape (n_nodes, 3, 6), via automatic differentiation (exact,
    not a finite-difference or hand-derived approximation)."""
    u = rigid_body_displacement(q, X_rel)
    J = jacrev(rigid_body_displacement, argnums=0)(q, X_rel)
    return u, J


def _embed_symmetric(q3):
    """q3 = (t_x, t_z, theta_y) -> full 6-vector (t_x, 0, t_z, 0, theta_y, 0),
    i.e. translation confined to the x-z plane and rotation confined to
    the y-axis -- the same restriction B8/B3's own rigid top-plate BC
    already uses (rigid_top_plate_displacement: uy=0 identically, only
    rotation about y), required here for the SAME reason: this mesh is
    a half-cylinder (theta in [0, pi]) with a genuine mirror-symmetry
    plane at y=0, and only this subspace of rigid motions keeps every
    point on that plane exactly on it (uy=0) for ANY point on the shim,
    not just a specific overridden node -- an unrestricted 6-dof rigid
    motion would only satisfy uy=0 at isolated nodes by coincidence,
    tearing the shim's own rigidity apart at the symmetry boundary."""
    tx, tz, theta_y = q3[0], q3[1], q3[2]
    z = torch.zeros_like(tx)
    return torch.stack([tx, z, tz, z, theta_y, z])


def rigid_body_displacement_symmetric(q3, X_rel):
    """Exact rigid-body displacement restricted to the mirror-symmetric
    subspace (see _embed_symmetric) -- for shims in a half-cylinder
    mesh with a y=0 symmetry plane."""
    return rigid_body_displacement(_embed_symmetric(q3), X_rel)


def rigid_body_displacement_and_jacobian_symmetric(q3, X_rel):
    """Returns (u, J): u as in rigid_body_displacement_symmetric, and
    J = du/dq3 with shape (n_nodes, 3, 3), via automatic
    differentiation."""
    u = rigid_body_displacement_symmetric(q3, X_rel)
    J = jacrev(rigid_body_displacement_symmetric, argnums=0)(q3, X_rel)
    return u, J


def shim_node_groups(nodes, zs, band_is_shim, band_z_boundaries, tol=1e-9):
    """Groups mesh node indices by which physical shim band they belong
    to, using the z-coordinate bands already established by
    data_generate_B8.layer_bands (the SAME z_boundaries/band_is_shim
    used to build the mesh in the first place, so grouping can never
    silently drift from the actual geometry).

    A shim band's own node group = every mesh node whose z lies in
    [z0, z1] (inclusive of both faces) -- this deliberately INCLUDES the
    interface nodes shared with the adjacent rubber layers above/below.
    Those shared nodes ARE the bonded-interface coupling mechanism: once
    they are driven by the shim's rigid map, the touching rubber
    elements see a rigid boundary condition there, exactly modeling a
    bonded rigid inclusion, with no separate contact/tie constraint
    needed.

    Returns a list of (node_indices, z0, z1) tuples, one per shim band,
    in bottom-to-top order.
    """
    z = nodes[:, 2]
    groups = []
    for b, is_shim in enumerate(band_is_shim):
        if not is_shim:
            continue
        z0, z1 = band_z_boundaries[b], band_z_boundaries[b + 1]
        mask = (z >= z0 - tol) & (z <= z1 + tol)
        idx = np.nonzero(mask)[0]
        groups.append((idx, float(z0), float(z1)))
    return groups
