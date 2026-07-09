"""
Shared material-model registry for the FEM ground-truth generators
(data_generate_B1.py, data_generate_B2.py): PK1 stress + tangent for all
three energy densities, plus (E, nu) -> material-parameter conversions so
every material can reuse the exact same GRF-sampled (E, nu) random fields
as PFEM-main's own Neo-Hookean convention (only the final constitutive
mapping differs).

Neo-Hookean's PK1/tangent are hand-derived (from PFEM-main, unchanged).
Mooney-Rivlin and Arruda-Boyce use JAX autodiff (material_models_jax.py)
from the same energy densities already validated in omar/losses.py.
"""
import numpy as np
from omar_pfem.data.material_models_jax import (
    mooney_rivlin_PK1_and_tangent,
    arruda_boyce_PK1_and_tangent,
)


def neo_hookean_PK1_and_tangent(F, mu, lam):
    Finv = np.linalg.inv(F)
    FinvT = Finv.T
    J = np.linalg.det(F)
    if J <= 0:
        raise ValueError(f"Nonphysical J={J} (det(F) <= 0).")

    lnJ = np.log(J)
    P = mu * (F - FinvT) + lam * lnJ * FinvT

    A = np.zeros((2, 2, 2, 2), dtype=float)
    delta = np.eye(2)
    for i in range(2):
        for J_ in range(2):
            for k in range(2):
                for L in range(2):
                    term1 = mu * delta[i, k] * delta[J_, L]
                    term2 = (mu - lam * lnJ) * FinvT[i, L] * FinvT[k, J_]
                    term3 = lam * FinvT[k, L] * FinvT[i, J_]
                    A[i, J_, k, L] = term1 + term2 + term3
    return P, A


def E_nu_to_neo_hookean(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (mu, lam)


def E_nu_to_mooney_rivlin(E, nu):
    """
    c1, c2 split the classical MR-to-linear-elasticity shear modulus
    mu = 2*(c1+c2) in a fixed 3:1 ratio (a common convention when no
    further data constrains the split); c penalizes volume change at a
    magnitude set by the Lame parameter.

    d is fixed to zero the PK1 stress at the undeformed state F=I (a
    stress-free reference configuration is required for a well-posed
    hyperelastic material -- otherwise the FEM/PFEM solve carries a
    spurious constant pre-stress that biases every sample). With this
    module's 2D invariants (I1 = tr(C), I2 = J^2 exactly, both from
    _invariants()/psi_mooney_rivlin() in material_models_jax.py),
    dpsi/dF at F=I works out to (-d + 2*c1 + 2*c2)*I, so d = 2*(c1+c2)
    is the value that zeros it (unlike the general-3D convention
    d = 2*(c1+2*c2) used in omar/losses.py's MooneyRivlin2D, which
    assumes I2 is an independent invariant with reference value 3 --
    not valid in 2D, where I2 = J^2 collapses to a function of J alone
    and leaves a residual stress of -2*c2 = -0.5*mu if reused here).
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c1 = 0.75 * mu
    c2 = 0.25 * mu
    c = lam / 2.0
    d = 2.0 * (c1 + c2)
    return (c, c1, c2, d)


def E_nu_to_arruda_boyce(E, nu, N_ab=5.0):
    """mu_ab = initial shear modulus (standard AB convention); kappa_ab
    set to the standard 2D bulk-modulus-like combination lam + 2/3*mu_ab.
    N_ab (chain locking parameter) held fixed -- it has no direct (E, nu)
    analog, matching the fixed value used throughout omar/losses.py."""
    mu_ab = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    kappa_ab = lam + (2.0 / 3.0) * mu_ab
    return (mu_ab, N_ab, kappa_ab)


MATERIALS = {
    "neo_hookean": (neo_hookean_PK1_and_tangent, E_nu_to_neo_hookean),
    "mooney_rivlin": (mooney_rivlin_PK1_and_tangent, E_nu_to_mooney_rivlin),
    "arruda_boyce": (arruda_boyce_PK1_and_tangent, E_nu_to_arruda_boyce),
}


def get_material_fns(name):
    if name not in MATERIALS:
        raise ValueError(f"Unknown material '{name}', expected one of {list(MATERIALS)}")
    return MATERIALS[name]
