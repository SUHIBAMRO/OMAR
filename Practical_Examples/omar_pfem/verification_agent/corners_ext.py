"""Corner exponents with symmetry (roller) edges, for the autonomous 2D pipeline (round O).

singularity.py knows traction-free and clamped edges. Literature problems often use a symmetry
or roller edge: normal displacement zero, shear traction zero. With the same Kolosov-
Muskhelishvili potentials phi = A z^lam, psi = B z^lam, on a ray z = r e^{i th}:

  displacement   2 mu (u_x + i u_y) ~ r^lam Dc,  Dc = kappa A e1 - lam conj(A) e2 - conj(B) e3
  resultant      F = phi + z conj(phi') + conj(psi) ~ r^lam Fc,  Fc = A e1 + lam conj(A) e2 + conj(B) e3
  (e1 = e^{i lam th}, e2 = e^{i(2-lam) th}, e3 = e^{-i lam th}; the rows of singularity._rows)

the traction on the ray is proportional to -i dF/dr, so after rotating into the ray's frame
  normal displacement u_th = 0      <=>  Im(Dc e^{-i th}) = 0
  shear traction     t_r  = 0      <=>  Im(Fc e^{-i th}) = 0
which gives two real rows, like the other conditions. Checked against the mirror argument
(test_corners_ext): a symmetry edge at angle alpha has the exponents of the symmetric modes of
the mirrored corner of angle 2 alpha, e.g. free/symmetry at 180 deg -> 0.5 (a crack, mode I),
free/symmetry at 90 deg -> no singular term (a straight free edge).
"""
import math

import numpy as np
from scipy.optimize import brentq


def _rows(lam, theta, bc, kappa):
    from omar_pfem.verification_agent import singularity
    if bc in ("free", "clamped"):
        return singularity._rows(lam, theta, bc, kappa)
    if bc != "symmetry":
        raise ValueError(bc)
    e1 = np.exp(1j * lam * theta)
    e2 = lam * np.exp(1j * (2 - lam) * theta)
    e3 = np.exp(-1j * lam * theta)
    rot = np.exp(-1j * theta)
    # columns for [Re A, Im A, Re B, Im B]; conj(A) = a1 - i a2, conj(B) = b1 - i b2
    D = [kappa * e1 - e2, 1j * kappa * e1 + 1j * e2, -e3, 1j * e3]
    F = [e1 + e2, 1j * e1 - 1j * e2, e3, -1j * e3]
    return [np.imag(np.array(D) * rot), np.imag(np.array(F) * rot)]


def char_det(lam, alpha, bc0, bc1, nu, plane="strain"):
    kappa = 3 - 4 * nu if plane == "strain" else (3 - nu) / (1 + nu)
    M = np.array(list(_rows(lam, 0.0, bc0, kappa)) + list(_rows(lam, alpha, bc1, kappa)))
    return float(np.linalg.det(M))


def exponents(angle, bc_in, bc_out, nu, plane="strain", pmax=2.2, n=4000):
    """Real exponents in (0, pmax] (sign changes of det M; double roots of a free/free crack
    are handled by remedy.corner_exponents, which this delegates to without symmetry edges)."""
    if "symmetry" not in (bc_in, bc_out):
        from omar_pfem.verification_agent.remedy import corner_exponents
        return corner_exponents(angle, bc_in, bc_out, nu, plane, pmax)
    grid = np.linspace(0.02, pmax, n)
    f = [char_det(l, angle, bc_in, bc_out, nu, plane) for l in grid]
    roots = []
    for i in range(len(grid) - 1):
        if f[i] == 0.0:
            roots.append(float(grid[i]))
        elif f[i] * f[i + 1] < 0:
            roots.append(float(brentq(char_det, grid[i], grid[i + 1],
                                      args=(angle, bc_in, bc_out, nu, plane), xtol=1e-12)))
    return roots


def _closed_form(pair, a, kappa):
    """Williams' characteristic equations (analytic in lam) for the three classical pairs;
    their real roots coincide with singularity.char_det's (checked in test_corners_ext)."""
    import cmath
    s2 = math.sin(a) ** 2
    if pair == ("free", "free"):
        return lambda l: cmath.sin(l * a) ** 2 - l * l * s2
    if pair == ("clamped", "clamped"):
        return lambda l: kappa ** 2 * cmath.sin(l * a) ** 2 - l * l * s2
    return lambda l: 4 * kappa * cmath.sin(l * a) ** 2 - ((kappa + 1) ** 2 - 4 * l * l * s2)


def complex_exponents(angle, bc_in, bc_out, nu, plane="strain", pmax=2.2):
    """Complex exponents (Re in (0, pmax), Im > 0) found by Newton's method from a grid of
    seeds on the analytic characteristic equation. With a symmetry edge the mirrored corner
    (angle 2 alpha, the other edge's condition on both sides) is searched: a superset of the
    symmetric modes, so a reported real part can only be too SMALL (conservative)."""
    kappa = 3 - 4 * nu if plane == "strain" else (3 - nu) / (1 + nu)
    bcs = tuple(sorted((bc_in, bc_out)))
    a = angle
    if "symmetry" in bcs:
        other = [b for b in bcs if b != "symmetry"]
        if not other:
            return []
        bcs, a = (other[0], other[0]), 2 * angle
    g = _closed_form(bcs, a, kappa)
    h = 1e-7
    found = []
    for re in np.linspace(0.05, pmax, 44):
        for im in np.linspace(0.01, 0.8, 10):
            z = complex(re, im)
            for _ in range(80):
                d = (g(z + h) - g(z - h)) / (2 * h)
                if d == 0:
                    break
                z -= g(z) / d
            if abs(g(z)) < 1e-9 and 0 < z.real < pmax and z.imag > 1e-6:
                if not any(abs(z.real - r) < 1e-6 and abs(z.imag - i) < 1e-6 for r, i in found):
                    found.append((float(z.real), float(z.imag)))
    return sorted(found)


def corner_analysis(polygon, nu, degree=2, plane="strain"):
    """As remedy.corner_analysis, with symmetry edges allowed, and complex exponents: a
    complex pair r^(Re +- i Im) limits the rate like a real exponent Re, so its real part
    joins rate_limiting (and is marked in complex_exponents)."""
    out = []
    n = len(polygon)
    for i in range(n):
        x0, y0, _ = polygon[i - 1]
        x1, y1, bc_out = polygon[i]
        x2, y2, _ = polygon[(i + 1) % n]
        bc_in = polygon[i - 1][2]
        a_in = np.arctan2(y0 - y1, x0 - x1)
        a_out = np.arctan2(y2 - y1, x2 - x1)
        ang = (a_in - a_out) % (2 * np.pi)
        if ang < 1e-9:
            ang = 2 * np.pi
        lams = exponents(ang, bc_in, bc_out, nu, plane)
        limiting = [l for l in lams if l < degree - 1e-9 and abs(l - round(l)) > 1e-6]
        cpx = complex_exponents(ang, bc_in, bc_out, nu, plane) if abs(ang - np.pi) > 1e-9 \
            or bc_in != bc_out else []
        limiting += [r for r, _ in cpx if r < degree - 1e-9]
        out.append({"x": x1, "y": y1, "angle_deg": float(np.degrees(ang)),
                    "edges": [bc_in, bc_out], "exponents": [float(l) for l in lams],
                    "complex_exponents": [list(c) for c in cpx],
                    "rate_limiting": sorted(float(l) for l in limiting)})
    return out
