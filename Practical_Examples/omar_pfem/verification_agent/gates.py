"""Deterministic verification gates.

Each gate returns a dict with a verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
and the numbers it was decided on, so a stored record can be re-audited
without re-running anything.
"""
import math

import numpy as np


def fit_rate(hs, errors):
    """Least-squares slope of log(error) vs log(h)."""
    lh, le = np.log(np.asarray(hs, float)), np.log(np.asarray(errors, float))
    return float(np.polyfit(lh, le, 1)[0])


def mms_rate_gate(hs, errors_by_norm, expected_by_norm, tol=0.2):
    """PASS iff every fitted rate is within `tol` of its theoretical value."""
    rates, ok = {}, True
    for norm, expected in expected_by_norm.items():
        r = fit_rate(hs, errors_by_norm[norm])
        rates[norm] = {"observed": r, "expected": expected,
                       "pass": abs(r - expected) <= tol}
        ok &= rates[norm]["pass"]
    return {"gate": "mms_rates", "verdict": "PASS" if ok else "FAIL",
            "tol": tol, "rates": rates}


def _celik_order(f1, f2, f3, r21, r32, iters=100):
    """Observed order p from three solutions, fine (f1) to coarse (f3),
    Celik et al. 2008 (ASME V&V), fixed-point iteration."""
    e21, e32 = f2 - f1, f3 - f2
    s = math.copysign(1.0, e32 / e21)
    p = 1.0
    for _ in range(iters):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    return p


def qoi_convergence_gate(hs, values, band_tol=0.01, gci_tol=0.05, scale=None):
    """Decide whether a QoI sequence under refinement is converged.

    hs must be ordered coarse to fine (decreasing). Only the three finest
    points are judged:
      - PASS if their relative spread is within `band_tol` (stagnant within
        mesh-to-mesh noise, whatever the sign pattern);
      - FAIL if they oscillate (sign change in successive differences)
        outside that band;
      - otherwise monotone: PASS iff the fine-grid GCI <= gci_tol.
    Spread and GCI are relative to `scale` (default |finest value|); pass a
    characteristic magnitude for components that are near zero, where a
    self-relative measure only amplifies noise.
    """
    hs, values = list(map(float, hs)), list(map(float, values))
    if len(values) < 3:
        return {"gate": "qoi_convergence", "verdict": "INCONCLUSIVE",
                "reason": "fewer than three refinement levels"}
    if any(hs[i] <= hs[i + 1] for i in range(len(hs) - 1)):
        raise ValueError("hs must be strictly decreasing (coarse to fine)")
    h3, h2, h1 = hs[-3:]
    f3, f2, f1 = values[-3:]
    scale = abs(f1) if scale is None else float(scale)
    spread = (max(f1, f2, f3) - min(f1, f2, f3)) / scale
    out = {"gate": "qoi_convergence", "finest_three": [f3, f2, f1], "scale": scale,
           "relative_spread": spread, "band_tol": band_tol, "gci_tol": gci_tol}
    if spread <= band_tol:
        return {**out, "verdict": "PASS",
                "reason": f"finest three within {band_tol:.2%} band (spread {spread:.3%})"}
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0:
        return {**out, "verdict": "PASS", "reason": "finest two identical"}
    if e32 / e21 < 0:
        return {**out, "verdict": "FAIL",
                "reason": "oscillatory: successive differences change sign"}
    if abs(e21) >= abs(e32):
        # successive differences not shrinking: the sequence is not in the
        # asymptotic convergent range (e.g. a singular peak growing with
        # refinement); Celik's |.| would report a spurious positive order
        return {**out, "verdict": "FAIL", "increments_fine_to_coarse": [e21, e32],
                "reason": "divergent: successive differences do not decrease"}
    r21, r32 = h2 / h1, h3 / h2
    p = _celik_order(f1, f2, f3, r21, r32)
    gci = 1.25 * abs((f1 - f2) / scale) / (r21 ** p - 1)
    extrap = f1 + (f1 - f2) / (r21 ** p - 1)
    verdict = "PASS" if gci <= gci_tol else "FAIL"
    return {**out, "verdict": verdict, "observed_order": p, "gci_fine": gci,
            "richardson_extrapolate": extrap,
            "reason": f"monotone, GCI_fine={gci:.2%} vs tol {gci_tol:.2%}"}


def strategy_robustness_gate(values, tol=0.02, scale=None):
    """PASS iff a QoI agrees across independent refinement strategies at
    comparable resolution (spread relative to `scale` within `tol`;
    default scale |median|)."""
    v = np.asarray(values, float)
    scale = abs(float(np.median(v))) if scale is None else float(scale)
    spread = float((v.max() - v.min()) / scale)
    return {"gate": "strategy_robustness", "verdict": "PASS" if spread <= tol else "FAIL",
            "reason": f"spread across strategies {spread:.3%} vs tol {tol:.2%}",
            "relative_spread": spread, "scale": scale, "tol": tol, "values": v.tolist()}
