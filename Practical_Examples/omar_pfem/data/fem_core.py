"""
Geometry-agnostic Q4 (bilinear quadrilateral) Total-Lagrangian FEM building
blocks shared by data_generate_B1.py and data_generate_B2.py: the reference
shape functions/derivatives, the per-element internal-force/tangent-stiffness
routine (material-agnostic -- dispatches through any PK1_and_tangent_fn(F,
*mat_params) callable from omar_pfem/data/materials.py), and Dirichlet BC
elimination. None of this depends on B1's flat-square or B2's curved-ring
geometry: the isoparametric Q4 formulation only ever sees each element's own
corner-node reference coordinates.

Unchanged from PFEM-main's data/hyper/data_generate_beam.py except
element_K_and_fint_TL takes a material-dispatch callable instead of being
hardcoded to Neo-Hookean.
"""
import numpy as np


def shape_Q4(xi, eta):
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    dN_dxi = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [+(1 - eta), -(1 + xi)],
        [+(1 + eta), +(1 + xi)],
        [-(1 + eta), +(1 - xi)]
    ])
    return N, dN_dxi


def element_K_and_fint_TL(Xe, ue, mat_params, PK1_and_tangent_fn):
    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    ke = np.zeros((8, 8), dtype=float)
    fe = np.zeros(8, dtype=float)

    xe = Xe + ue.reshape(4, 2)

    for (xi, eta) in gauss:
        N, dN_dxi = shape_Q4(xi, eta)

        J0 = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J0[0, 0] += Xe[a, 0] * dN_dxi[a, 0]
            J0[1, 0] += Xe[a, 1] * dN_dxi[a, 0]
            J0[0, 1] += Xe[a, 0] * dN_dxi[a, 1]
            J0[1, 1] += Xe[a, 1] * dN_dxi[a, 1]

        detJ0 = np.linalg.det(J0)
        if detJ0 <= 0:
            raise ValueError("Reference element has non-positive det(J0).")

        invJ0 = np.linalg.inv(J0)
        dN_dX = dN_dxi @ invJ0

        F = np.zeros((2, 2), dtype=float)
        for a in range(4):
            F[0, 0] += xe[a, 0] * dN_dX[a, 0]
            F[0, 1] += xe[a, 0] * dN_dX[a, 1]
            F[1, 0] += xe[a, 1] * dN_dX[a, 0]
            F[1, 1] += xe[a, 1] * dN_dX[a, 1]

        P, A = PK1_and_tangent_fn(F, *mat_params)

        for a in range(4):
            gNa = dN_dX[a]
            fi = P @ gNa
            fe[2*a:2*a+2] += fi * (w * detJ0)

        for a in range(4):
            for b in range(4):
                gNa = dN_dX[a]
                gNb = dN_dX[b]
                Kab = np.zeros((2, 2), dtype=float)
                for i in range(2):
                    for J_ in range(2):
                        for k in range(2):
                            for L in range(2):
                                Kab[i, k] += gNa[J_] * A[i, J_, k, L] * gNb[L]
                ke[2*a:2*a+2, 2*b:2*b+2] += Kab * (w * detJ0)

    return ke, fe


def apply_dirichlet(K, R, fixed_dofs):
    fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
    all_dofs = np.arange(K.shape[0], dtype=int)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    Kff = K[free][:, free]
    Rf = R[free]
    return free, Kff, Rf
