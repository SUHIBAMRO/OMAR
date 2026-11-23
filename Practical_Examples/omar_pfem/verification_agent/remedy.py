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
