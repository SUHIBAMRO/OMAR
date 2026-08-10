"""
PyTorch strain-energy densities (vectorized over Gauss points) for the
Transolver training scripts, plus the same (E, nu) -> material-parameter
conversions used by the FEM ground-truth generators
(omar_pfem/data/materials.py), so training and data generation always use
the exact same constitutive mapping from the shared random E, nu fields.

Neo-Hookean is PFEM-main's original formula, unchanged. Mooney-Rivlin and
Arruda-Boyce match the energy densities already validated in
omar/losses.py (the VINO project) and omar_pfem/data/material_models_jax.py.
"""
import torch


def neo_hookean_energy_density_vectorized(F, mu, lam, dtype=torch.float32):
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    J = torch.clamp(J, min=1e-8)
    I1 = torch.sum(F ** 2, dim=(1, 2))
    lnJ = torch.log(J)
    psi = (mu / 2.0) * (I1 - 2.0 - 2.0 * lnJ) + (lam / 2.0) * (lnJ ** 2)
    return psi


def mooney_rivlin_energy_density_vectorized(F, c, c1, c2, d, dtype=torch.float32):
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    J = torch.clamp(J, min=1e-8)
    I1 = torch.sum(F ** 2, dim=(1, 2))
    I2 = J * J  # 2D identity: 0.5*(tr(C)^2 - tr(C@C)) == det(C) == J^2
    psi = c * (J - 1) ** 2 - d * torch.log(J) + c1 * (I1 - 2) + c2 * (I2 - 1)
    return psi


def arruda_boyce_energy_density_vectorized(F, mu_ab, N_ab, kappa_ab, dtype=torch.float32):
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    J = torch.clamp(J, min=1e-8)
    I1 = torch.sum(F ** 2, dim=(1, 2))
    I1_bar = I1 / J
    # The 5-term 8-chain series is only a valid approximation below the
    # chain-locking limit I1_bar = 3*N_ab; it has no physical meaning past
    # that point and, being 5th-order in I1_bar, its value and gradient
    # overflow float32 well before that for the wild, untrained-network F
    # seen early in training (neo-Hookean/Mooney-Rivlin stay at most
    # quadratic in I1 and don't hit this). Clamp to the model's own valid
    # domain instead of letting it diverge.
    I1_bar = torch.clamp(I1_bar, max=3.0 * N_ab - 1e-3)
    alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    psi = torch.zeros_like(I1_bar)
    i1_bar_pow = I1_bar
    for k, a in enumerate(alpha, start=1):
        psi = psi + mu_ab * a / (N_ab ** (k - 1)) * (i1_bar_pow - 2.0 ** k)
        i1_bar_pow = i1_bar_pow * I1_bar
    psi = psi + kappa_ab / 2 * (J - 1) ** 2
    return psi


def E_nu_to_neo_hookean(E, nu, mode="plane_strain"):
    if mode == "plane_strain":
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        mu = E / (2.0 * (1.0 + nu))
        nu_star = nu / (1.0 + nu)
        lam = E * nu_star / ((1.0 + nu_star) * (1.0 - 2.0 * nu_star))
    return (mu, lam)


def E_nu_to_mooney_rivlin(E, nu):
    """Same calibration as omar_pfem/data/materials.py's
    E_nu_to_mooney_rivlin -- see that module's docstring for the rationale
    (3:1 c1/c2 split of the linearized shear modulus; d=2*(c1+c2) is the
    2D-correct value that zeros the PK1 stress at F=I, given this
    module's I2=J^2 invariant)."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c1 = 0.75 * mu
    c2 = 0.25 * mu
    c = lam / 2.0
    d = 2.0 * (c1 + c2)
    return (c, c1, c2, d)


def E_nu_to_arruda_boyce(E, nu, N_ab=5.0):
    """Same calibration as omar_pfem/data/materials.py's
    E_nu_to_arruda_boyce."""
    mu_ab = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    kappa_ab = lam + (2.0 / 3.0) * mu_ab
    N_ab_t = torch.full_like(E, float(N_ab))
    return (mu_ab, N_ab_t, kappa_ab)


MATERIALS = {
    "neo_hookean": (neo_hookean_energy_density_vectorized, E_nu_to_neo_hookean),
    "mooney_rivlin": (mooney_rivlin_energy_density_vectorized, E_nu_to_mooney_rivlin),
    "arruda_boyce": (arruda_boyce_energy_density_vectorized, E_nu_to_arruda_boyce),
}


def get_material_fns(name):
    if name not in MATERIALS:
        raise ValueError(f"Unknown material '{name}', expected one of {list(MATERIALS)}")
    return MATERIALS[name]
