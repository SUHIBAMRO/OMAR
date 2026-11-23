"""Automatic remedy for a corner singularity (round K, 2026-11-23).

The diagnosis names a singular point (hypothesis M1) and the singular-order check
knows its exponent lam. Until now the agent stopped there: it reported that a pointwise
peak is not verifiable and that global quantities converge at the reduced order 2 lam.
This module turns the diagnosis into an action: it designs a mesh family graded toward
the singular corner with the exponent theory prescribes, so that degree-p elements
recover their optimal order.

Theory (Babuska, Kellogg and Pitkaranta 1979; Apel 1999, tensor-product grading): for a
corner singularity r^lam and elements of degree p, node coordinates graded as
x = s^g toward the corner lines give the optimal energy-norm rate N^(-p/2) if
g > p / lam. The strain energy then converges at order 2p in h = 1/n (instead of 2 lam).

    from omar_pfem.verification_agent.remedy import plan_grading
"""
import math


def plan_grading(lam, degree, step=0.1):
    """Smallest grading exponent on a grid of `step` that exceeds degree / lam, and the
    orders it predicts (h = 1/n, n elements per unit length before grading)."""
    g_min = degree / lam
    g = math.floor(g_min / step + 1 + 1e-9) * step
    return {"lam": lam, "degree": degree, "g_min": g_min, "g": round(g, 10),
            "predicted_energy_order_uniform": 2 * lam,
            "predicted_energy_order_graded": 2 * degree,
            "rule": "g > p / lam (tensor-product grading toward the corner lines)"}


# ---------------------------------------------------------------- automatic corner analysis
import numpy as np


def corner_exponents(angle, bc_in, bc_out, nu, plane="strain", pmax=2.2):
    """All real exponents in (0, pmax] of a corner of opening `angle` between an edge with
    condition bc_in and one with bc_out ("free" covers loaded edges too: a traction does
    not change the homogeneous eigenproblem). A crack (angle 2 pi, free/free) has double
    roots k/2 that a sign-change search cannot find; they are checked directly."""
    from omar_pfem.verification_agent.singularity import char_det, smallest_exponent
    if abs(angle - 2 * np.pi) < 1e-9 and bc_in == bc_out == "free":
        return [k / 2 for k in range(1, int(2 * pmax) + 1)
                if abs(char_det(k / 2, angle, "free", "free", nu, plane)) < 1e-10]
    roots, lo = [], 0.02
    while True:
        r = smallest_exponent(angle, bc_in, bc_out, nu, plane, lo=lo, hi=pmax)
        if r is None:
            return roots
        roots.append(r)
        lo = r + 0.01


def corner_analysis(polygon, nu, degree=2, plane="strain"):
    """polygon: list of (x, y, bc of the edge that LEAVES this vertex), counter-clockwise
    around the material. For every vertex: interior angle, the edge conditions on both
    sides, the exponents, and whether it limits the rate of degree-`degree` elements (a
    non-integer exponent below the degree)."""
    out = []
    n = len(polygon)
    for i in range(n):
        x0, y0, _ = polygon[i - 1]
        x1, y1, bc_out = polygon[i]
        x2, y2, _ = polygon[(i + 1) % n]
        bc_in = polygon[i - 1][2]
        a_in = np.arctan2(y0 - y1, x0 - x1)            # direction back along the incoming edge
        a_out = np.arctan2(y2 - y1, x2 - x1)           # direction along the outgoing edge
        ang = (a_in - a_out) % (2 * np.pi)             # interior angle (CCW polygon)
        if ang < 1e-9:
            ang = 2 * np.pi                            # crack tip: the two faces coincide
        lams = corner_exponents(ang, bc_out, bc_in, nu, plane)
        limiting = [l for l in lams if l < degree - 1e-9 and abs(l - round(l)) > 1e-6]
        out.append({"x": x1, "y": y1, "angle_deg": float(np.degrees(ang)),
                    "edges": [bc_in, bc_out], "exponents": [float(l) for l in lams],
                    "rate_limiting": [float(l) for l in limiting]})
    return out


def plan_problem(polygon, nu, degree=2, step=0.1):
    """Grading plan from the corner analysis: grade toward the lines through every
    rate-limiting corner, with one exponent g above degree / min(lam)."""
    corners = corner_analysis(polygon, nu, degree)
    sing = [c for c in corners if c["rate_limiting"]]
    lam_min = min(min(c["rate_limiting"]) for c in sing)
    base = plan_grading(lam_min, degree, step)
    base.update(singular_corners=[{k: c[k] for k in ("x", "y", "angle_deg", "edges",
                                                     "rate_limiting")} for c in sing],
                x_targets=sorted({c["x"] for c in sing}), y_targets=sorted({c["y"] for c in sing}),
                predicted_energy_order_uniform=2 * lam_min,
                predicted_energy_order_graded_strongest_only=2 * min(
                    [min(c["rate_limiting"]) for c in sing if min(c["rate_limiting"]) > lam_min]
                    or [degree]))
    return base


def graded_coords(n, g, targets, lo=0.0, hi=2.0):
    """Node coordinates on [lo, hi] with n (even) elements per unit length. Each unit
    [k, k+1] is split at its midpoint; a half next to an integer point in `targets` is
    graded toward it (spacing ~ s ** g), the others are uniform."""
    t = np.linspace(0.0, 1.0, n // 2 + 1)[1:]
    pts = [float(lo)]
    for k in range(int(round(lo)), int(round(hi))):
        left = k + 0.5 * (t ** g if k in targets else t)
        right = (k + 1) - 0.5 * ((1 - t) ** g if (k + 1) in targets else (1 - t))
        pts.extend(left.tolist())
        pts.extend(right.tolist())
    return np.array(pts)


# the singular-point facts of the two 2D problems (written by hand from their set-up; the
# exponents and the plan are computed): (x, y, condition of the edge leaving the vertex)
POLYGONS = {
    "l_shape": [(0, 0, "clamped"), (2, 0, "free"), (2, 1, "free"), (1, 1, "free"),
                (1, 2, "free"), (0, 2, "free")],
    # the crack faces run from (0, 0) to the tip (1, 0) and back; the two (0, 0) copies are
    # the upper and lower crack-face corners
    "edge_crack": [(0, -1, "clamped"), (2, -1, "free"), (2, 1, "free"), (0, 1, "free"),
                   (0, 0, "free"), (1, 0, "free"), (0, 0, "free")],
}
