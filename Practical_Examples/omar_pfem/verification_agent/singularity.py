"""Corner singular exponents of plane linear elasticity (Williams 1952).

Near a corner of opening angle alpha the leading displacement term is
u ~ r^lam f(theta), stress ~ r^(lam-1). With Kolosov-Muskhelishvili
potentials phi = A z^lam, psi = B z^lam (A, B complex), on a ray theta:

  traction-free:  A e^{i lam th} + lam conj(A) e^{i(2-lam) th} + conj(B) e^{-i lam th} = 0
  clamped:   kappa A e^{i lam th} - lam conj(A) e^{i(2-lam) th} - conj(B) e^{-i lam th} = 0

kappa = 3 - 4 nu (plane strain) or (3 - nu)/(1 + nu) (plane stress). The
two edge conditions give a real 4x4 system M(lam) [Re A, Im A, Re B, Im B]
= 0; the exponents are the real roots of det M(lam) in (0, 1].

Consequences for uniform h-refinement with degree-1 elements (Babuska &
Kellogg; Strang & Fix): energy-norm error ~ h^lam, so the strain energy
(and, through pollution, point values anywhere in the domain) converge at
order ~2 lam, and the pointwise stress peak grows like h^(lam - 1).
"""
import numpy as np
from scipy.optimize import brentq


def _rows(lam, theta, bc, kappa):
    """Two real rows (Re, Im) of the edge condition at angle theta."""
    e1 = np.exp(1j * lam * theta)            # multiplies A
    e2 = lam * np.exp(1j * (2 - lam) * theta)  # multiplies conj(A)
    e3 = np.exp(-1j * lam * theta)           # multiplies conj(B)
    if bc == "free":
        cA, cAb, cBb = e1, e2, e3
    elif bc == "clamped":
        cA, cAb, cBb = kappa * e1, -e2, -e3
    else:
        raise ValueError(bc)
    # A = a1 + i a2, conj(A) = a1 - i a2, conj(B) = b1 - i b2
    col = [cA + cAb, 1j * cA - 1j * cAb, cBb, -1j * cBb]
    return [np.real(col), np.imag(col)]


def char_det(lam, alpha, bc0, bc1, nu, plane="strain"):
    kappa = 3 - 4 * nu if plane == "strain" else (3 - nu) / (1 + nu)
    M = np.array(_rows(lam, 0.0, bc0, kappa) + _rows(lam, alpha, bc1, kappa))
    return float(np.linalg.det(M))


def smallest_exponent(alpha, bc0, bc1, nu, plane="strain", lo=0.05, hi=1.0, n=4000):
    """Smallest real root of det M(lam) in (lo, hi]; None if there is none
    (no real singular term: the solution is at least as smooth as r^1)."""
    grid = np.linspace(lo, hi, n)
    vals = [char_det(l, alpha, bc0, bc1, nu, plane) for l in grid]
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            return float(grid[i])
        if vals[i] * vals[i + 1] < 0:
            return float(brentq(char_det, grid[i], grid[i + 1],
                                args=(alpha, bc0, bc1, nu, plane), xtol=1e-12))
    return None


def b1_corner_exponent(nu=0.3):
    """B1: the clamped bottom edge (theta = 0) meets a traction-free
    vertical edge (theta = pi/2) at both bottom corners, plane strain."""
    return smallest_exponent(np.pi / 2, "clamped", "free", nu, "strain")
