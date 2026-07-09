"""
Mooney-Rivlin and Arruda-Boyce 1st Piola-Kirchhoff stress (P = dpsi/dF) and
material tangent (A = dP/dF) for the numpy/scipy Total-Lagrangian
Newton-Raphson FEM solver in data_generate_B1.py / data_generate_B2.py.

PFEM-main only ever hand-derives P and A for Neo-Hookean. Rather than
hand-deriving the (much messier) Mooney-Rivlin and Arruda-Boyce tangents --
a good way to introduce a silent sign/index error that only shows up as
slow/non-converging Newton iterations -- both are obtained here via JAX
autodiff (jax.grad for P, jax.jacfwd(jax.grad(...)) for A) from the exact
same strain-energy density formulas already validated in omar/losses.py
(the VINO project's DEM/VinoHyperelasticityLoss materials), so correctness
only depends on the (much simpler, already cross-checked) energy density
expression, not on a hand-propagated tangent.

2D identity used below: for a 2x2 right Cauchy-Green tensor C = F^T F,
the "second invariant" I2 = 0.5*(tr(C)^2 - tr(C^2)) equals det(C) = J^2
exactly (standard 2x2 linear-algebra identity, verified via eigenvalues:
0.5*((l1+l2)^2 - (l1^2+l2^2)) = l1*l2 = det(C)) -- this is the same I2
used in omar/losses.py's MooneyRivlin2D.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _invariants(F):
    C = F.T @ F
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    I1 = jnp.trace(C)
    I2 = J * J  # = 0.5*(tr(C)^2 - tr(C@C)) in 2D, see module docstring
    return I1, I2, J


def psi_mooney_rivlin(F, c, c1, c2, d):
    I1, I2, J = _invariants(F)
    return c * (J - 1) ** 2 - d * jnp.log(J) + c1 * (I1 - 2) + c2 * (I2 - 1)


def psi_arruda_boyce(F, mu_ab, N_ab, kappa_ab):
    I1, I2, J = _invariants(F)
    I1_bar = I1 / J
    alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    strain_energy = 0.0
    i1_bar_pow = I1_bar
    for k, a in enumerate(alpha, start=1):
        strain_energy = strain_energy + mu_ab * a / (N_ab ** (k - 1)) * (i1_bar_pow - 2.0 ** k)
        i1_bar_pow = i1_bar_pow * I1_bar
    strain_energy = strain_energy + kappa_ab / 2 * (J - 1) ** 2
    return strain_energy


def _make_P_and_A(psi_fn, n_params):
    """psi_fn(F, *params) -> scalar. Returns a numpy-in/numpy-out function
    computing (P, A) = (dpsi/dF, d^2psi/dF^2), JIT-compiled once (params
    are traced JAX values, not static, so different mu/lam/F per element
    do not trigger retracing)."""
    grad_psi = jax.grad(psi_fn, argnums=0)
    tangent_psi = jax.jacfwd(grad_psi, argnums=0)

    grad_psi_jit = jax.jit(grad_psi)
    tangent_psi_jit = jax.jit(tangent_psi)

    def P_and_A(F_np, *params_np):
        F_j = jnp.asarray(F_np, dtype=jnp.float64)
        params_j = [jnp.asarray(p, dtype=jnp.float64) for p in params_np]
        P_j = grad_psi_jit(F_j, *params_j)
        A_j = tangent_psi_jit(F_j, *params_j)
        return np.asarray(P_j), np.asarray(A_j)

    return P_and_A


_mr_P_and_A = _make_P_and_A(psi_mooney_rivlin, 4)
_ab_P_and_A = _make_P_and_A(psi_arruda_boyce, 3)


def mooney_rivlin_PK1_and_tangent(F, c, c1, c2, d):
    J = np.linalg.det(F)
    if J <= 0:
        raise ValueError(f"Nonphysical J={J} (det(F) <= 0).")
    return _mr_P_and_A(F, c, c1, c2, d)


def arruda_boyce_PK1_and_tangent(F, mu_ab, N_ab, kappa_ab):
    J = np.linalg.det(F)
    if J <= 0:
        raise ValueError(f"Nonphysical J={J} (det(F) <= 0).")
    return _ab_P_and_A(F, mu_ab, N_ab, kappa_ab)
