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


def qoi_convergence_gate(hs, values, band_tol=0.01, gci_tol=0.05, scale=None,
                         order_window=None):
    """Decide whether a QoI sequence under refinement is converged.

    hs must be ordered coarse to fine (decreasing). Only the three finest
    points are judged:
      - monotone with shrinking differences: Celik order and GCI; PASS iff
        the order is in `order_window` and GCI <= gci_tol (evidence "rate");
      - otherwise PASS if the relative spread is within `band_tol`
        (evidence "band"), else FAIL (oscillatory or divergent).
    Spread and GCI are relative to `scale` (default |finest value|); pass a
    characteristic magnitude for components that are near zero, where a
    self-relative measure only amplifies noise.

    order_window=(p_lo, p_hi): the GCI is only meaningful in the asymptotic
    range (Celik et al. 2008), so a monotone sequence whose observed order
    lies outside the window FAILs as "not asymptotic" whatever its GCI.
    Every result carries `evidence`: "rate" (monotone with shrinking
    differences, order in the window, GCI checked), "band" (not a clean
    monotone sequence, or no admissible order, but stable within band_tol:
    a weaker claim), or "none". Rate evidence is tried first; the band is
    only the fallback.
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
    out["order_window"] = order_window
    e21, e32 = f2 - f1, f3 - f2
    r21, r32 = h2 / h1, h3 / h2
    in_band = spread <= band_tol
    band_pass = {**out, "verdict": "PASS", "evidence": "band",
                 "reason": f"finest three within {band_tol:.2%} band (spread {spread:.3%}); "
                           "no convergence rate demonstrated"}
    if e21 == 0:
        return {**band_pass, "reason": "finest two identical; no rate identifiable"}
    if e32 / e21 > 0 and abs(e21) < abs(e32):
        # monotone with shrinking differences: rate evidence takes precedence
        p = _celik_order(f1, f2, f3, r21, r32)
        gci = 1.25 * abs((f1 - f2) / scale) / (r21 ** p - 1)
        extrap = f1 + (f1 - f2) / (r21 ** p - 1)
        res = {**out, "observed_order": p, "gci_fine": gci, "richardson_extrapolate": extrap}
        if order_window is None or order_window[0] <= p <= order_window[1]:
            return {**res, "evidence": "rate", "verdict": "PASS" if gci <= gci_tol else "FAIL",
                    "reason": f"monotone, p={p:.2f}, GCI_fine={gci:.2%} vs tol {gci_tol:.2%}"}
        if in_band:
            return {**band_pass, "observed_order_diagnostic": p,
                    "reason": band_pass["reason"] + f" (monotone but p={p:.2f} outside "
                                                    f"[{order_window[0]}, {order_window[1]}])"}
        return {**res, "evidence": "none", "verdict": "FAIL",
                "reason": f"monotone but not asymptotic: observed order {p:.2f} outside "
                          f"[{order_window[0]}, {order_window[1]}] (GCI {gci:.2%} not trusted)"}
    if in_band:
        return band_pass
    if e32 / e21 < 0:
        return {**out, "verdict": "FAIL", "evidence": "none",
                "reason": "oscillatory: successive differences change sign"}
    # successive differences not shrinking: not in the asymptotic convergent
    # range (e.g. a singular peak growing with refinement); Celik's |.| would
    # report a spurious positive order
    return {**out, "verdict": "FAIL", "evidence": "none", "increments_fine_to_coarse": [e21, e32],
            "reason": "divergent: successive differences do not decrease"}


def _power_fit(hs, values, p_bounds=(0.05, 20.0)):
    """Least-squares fit v = a + b * h^p (three parameters).

    h is normalised by its smallest value (conditioning only; a and p are
    unchanged). p is found by a log-spaced grid search over p_bounds and a
    bounded 1-D refinement of the residual. Returns a, b (for normalised h),
    p, the residuals, and whether p ended on a search bound.
    """
    from scipy.optimize import minimize_scalar
    h = np.asarray(hs, float)
    h = h / h.min()
    v = np.asarray(values, float)

    def solve(p):
        A = np.c_[np.ones_like(h), h ** p]
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return c, v - A @ c

    ssr = lambda p: float(np.sum(solve(p)[1] ** 2))
    grid = np.geomspace(p_bounds[0], p_bounds[1], 2001)
    k = int(np.argmin([ssr(p) for p in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    p = float(minimize_scalar(ssr, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10}).x) if hi > lo else float(grid[k])
    (a, b), res = solve(p)
    at_bound = p <= p_bounds[0] * 1.001 or p >= p_bounds[1] * 0.999
    return {"a": float(a), "b": float(b), "p": p, "residuals": res.tolist(),
            "p_at_search_bound": bool(at_bound)}


def multilevel_convergence_gate(hs, values, band_tol=0.01, gci_tol=0.05, scale=None,
                                order_window=None, n_levels=5, fit_quality=0.1, fs=1.25):
    """Convergence from the finest n_levels (default 5) instead of three.

    Least-squares variant in the spirit of Eca & Hoekstra (2014, J. Comput.
    Phys. 262:104-130), simplified. Criteria, fixed (and unit-tested on
    synthetic sequences) before this gate was run on project data -- the
    region_study diagnostics had already been seen, so this is not blind:
      1. at least 4 levels are used (the finest min(n_levels, available));
      2. monotone: all successive differences non-zero and of one sign;
      3. fit v = a + b h^p (p searched on [0.05, 20], so the window test is
         not decided by the optimiser's bounds);
      4. fit quality: sigma = sqrt(SSR / (n - 3)) <= fit_quality * (range
         of the n values);
      5. p inside order_window;
      6. leave-one-out stability (only when n >= 5, so that every refit is
         still over-determined): refitting without each level in turn gives
         p inside order_window every time;
    if all hold, evidence is "rate" and the uncertainty of the finest value
      U = fs * |v_fine - a| + sigma + |residual at the finest level|
    relative to `scale` must be <= gci_tol for PASS. Otherwise it falls back
    to the same stability band as qoi_convergence_gate (finest three within
    band_tol: PASS with evidence "band", else FAIL). `failed_criteria` lists
    which of 2-6 failed, so a band verdict says why no rate was certified.
    """
    hs, values = list(map(float, hs)), list(map(float, values))
    if any(hs[i] <= hs[i + 1] for i in range(len(hs) - 1)):
        raise ValueError("hs must be strictly decreasing (coarse to fine)")
    n = min(n_levels, len(values))
    base = {"gate": "qoi_convergence_multilevel", "n_levels": n, "band_tol": band_tol,
            "gci_tol": gci_tol, "order_window": order_window}
    if n < 4:
        return {**base, "verdict": "INCONCLUSIVE", "evidence": "none",
                "reason": f"multilevel test needs >= 4 levels, have {len(values)}"}
    h, v = hs[-n:], values[-n:]
    scale = abs(v[-1]) if scale is None else float(scale)
    f3, f2, f1 = v[-3:]
    spread = (max(f1, f2, f3) - min(f1, f2, f3)) / scale
    base.update(levels_used=v, scale=scale, relative_spread=spread)
    in_window = lambda p: order_window is None or order_window[0] <= p <= order_window[1]

    failed = []
    d = np.diff(v)
    if not (np.all(d > 0) or np.all(d < 0)):
        failed.append("monotone")
    fit = _power_fit(h, v)
    sigma = float(np.sqrt(np.sum(np.square(fit["residuals"])) / (n - 3))) if n > 3 else 0.0
    rng = max(v) - min(v)
    base.update(fit_order=fit["p"], fit_limit=fit["a"], fit_sigma=sigma,
                fit_p_at_search_bound=fit["p_at_search_bound"])
    if rng == 0 or sigma > fit_quality * rng:
        failed.append("fit_quality")
    if not in_window(fit["p"]) or fit["p_at_search_bound"]:
        failed.append("order_window")
    if n >= 5:
        loo = [_power_fit(h[:i] + h[i + 1:], v[:i] + v[i + 1:])["p"] for i in range(n)]
        base["leave_one_out_orders"] = loo
        if not all(in_window(p) for p in loo):
            failed.append("leave_one_out")
    base["failed_criteria"] = failed

    if not failed:
        U = (fs * abs(v[-1] - fit["a"]) + sigma + abs(fit["residuals"][-1])) / scale
        loo = base.get("leave_one_out_orders")
        loo_txt = (f"leave-one-out {min(loo):.2f}-{max(loo):.2f}" if loo else
                   f"no leave-one-out check with {n} levels")
        return {**base, "evidence": "rate", "uncertainty_fine": U,
                "verdict": "PASS" if U <= gci_tol else "FAIL",
                "reason": f"{n} levels monotone, fitted p={fit['p']:.2f} ({loo_txt}), "
                          f"U_fine={U:.2%} vs tol {gci_tol:.2%}"}
    why = ", ".join(failed)
    if spread <= band_tol:
        return {**base, "verdict": "PASS", "evidence": "band",
                "reason": f"finest three within {band_tol:.2%} band (spread {spread:.3%}); "
                          f"no rate from {n} levels (failed: {why}; fitted p={fit['p']:.2f})"}
    return {**base, "verdict": "FAIL", "evidence": "none",
            "reason": f"no rate from {n} levels (failed: {why}; fitted p={fit['p']:.2f}) and "
                      f"finest-three spread {spread:.3%} exceeds {band_tol:.2%}"}


def singular_order_gate(hs, values, lam, qoi_class, tol=0.15, n_levels=5):
    """Is the observed convergence consistent with the corner singularity?

    lam: smallest singular exponent at the corner (singularity.py). For
    degree-1 elements under uniform refinement theory gives
      global (energy)             order = 2 lam          ("equal": |p - 2 lam| <= tol)
      displacement, region avg    order >= 2 lam (pollution bound; may be faster)
      pointwise peak              peak ~ h^(lam - 1)      (log-log slope, "equal")
    p is the multilevel least-squares order over the finest n_levels (the
    peak: log-log slope over the same levels). tol = 0.15 was fixed after
    the B1 orders had been seen (not blind)."""
    if lam is None:
        return {"gate": "singular_order", "verdict": "INCONCLUSIVE",
                "reason": "no singular exponent available for this problem"}
    h, v = np.asarray(hs[-n_levels:], float), np.asarray(values[-n_levels:], float)
    if qoi_class == "pointwise_peak":
        p = float(np.polyfit(np.log(h), np.log(np.abs(v)), 1)[0])
        expected, mode = lam - 1.0, "equal"
        what = "log-log slope of the peak"
    else:
        p = _power_fit(list(h), list(v))["p"]
        expected = 2.0 * lam
        mode = "equal" if qoi_class == "global" else "lower_bound"
        what = "fitted order"
    ok = abs(p - expected) <= tol if mode == "equal" else p >= expected - tol
    rel = "=" if mode == "equal" else ">="
    return {"gate": "singular_order", "verdict": "PASS" if ok else "FAIL",
            "observed": p, "expected": expected, "mode": mode, "tol": tol, "lam": lam,
            "reason": f"{what} {p:.3f} vs theory {rel} {expected:.3f} (lam = {lam:.4f}, "
                      f"tol {tol})"}


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


def anchored_robustness_gate(values, anchor, tol=0.02, scale=None, anchor_kind=""):
    """PASS iff every refinement strategy lies within `tol` (relative to
    `scale`, default |anchor|) of the refinement family's anchor value: its
    Richardson extrapolate when one exists, else its finest value. Unlike
    strategy_robustness_gate, agreement among strategies that share the same
    discretization error does not pass this gate."""
    v = np.asarray(values, float)
    scale = abs(float(anchor)) if scale is None else float(scale)
    dev = np.abs(v - anchor) / scale
    return {"gate": "anchored_robustness", "verdict": "PASS" if dev.max() <= tol else "FAIL",
            "reason": f"max deviation from {anchor_kind or 'anchor'} {dev.max():.3%} vs tol {tol:.2%}",
            "anchor": float(anchor), "anchor_kind": anchor_kind, "deviations": dev.tolist(),
            "values": v.tolist(), "scale": scale, "tol": tol}


def family_anchor(conv_result):
    """Anchor for anchored_robustness_gate from a qoi_convergence or
    qoi_convergence_multilevel result."""
    if conv_result.get("evidence") == "rate" and "richardson_extrapolate" in conv_result:
        return conv_result["richardson_extrapolate"], "Richardson extrapolate"
    if conv_result.get("evidence") == "rate" and "fit_limit" in conv_result:
        return conv_result["fit_limit"], "multilevel fit limit"
    if "levels_used" in conv_result:
        return conv_result["levels_used"][-1], "finest family value"
    return conv_result["finest_three"][-1], "finest family value"


CONV_CHECKS = ("qoi_convergence", "qoi_convergence_multilevel")


def combine_verdicts(verdicts):
    """Overall verdict of one QoI (FAIL if any gate failed, PASS if all
    passed, else INCONCLUSIVE) and the strongest convergence evidence among
    the convergence gates that PASSED ("rate" > "band" > "none"), with the
    gate it came from. Several convergence gates never relax each other: a
    FAIL from either one fails the QoI.

    Contradiction rule: a rate from the three finest levels does not count
    when the multilevel gate found the finest five levels NOT MONOTONE --
    that is evidence against an asymptotic regime, not merely a lack of
    evidence for one -- and is downgraded to "band". (The reverse case, a
    3-level gate that only saw a band because the finest changes are tiny,
    does not contradict a multilevel rate.)"""
    vs = [v["verdict"] for v in verdicts.values()]
    overall = "FAIL" if "FAIL" in vs else ("PASS" if all(v == "PASS" for v in vs)
                                           else "INCONCLUSIVE")
    ml = verdicts.get("qoi_convergence_multilevel", {})
    non_monotone = "monotone" in (ml.get("failed_criteria") or [])
    passed = [(k, "band" if (non_monotone and v.get("evidence") == "rate"
                             and k != "qoi_convergence_multilevel") else v.get("evidence"))
              for k, v in verdicts.items()
              if k in CONV_CHECKS and v["verdict"] == "PASS"]
    for level in ("rate", "band"):
        src = [k for k, e in passed if e == level]
        if src:
            return overall, level, src
    return overall, "none", []
