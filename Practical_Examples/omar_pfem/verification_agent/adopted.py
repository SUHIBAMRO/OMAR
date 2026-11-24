"""The rule set adopted on 24 November 2026 after Professor Rabczuk's answers.

Not blind: these decisions were taken after the held-out results of rounds F-N had been seen.
The frozen modules (gates.py, trees.py, orchestrate.py) are NOT changed, so every stored
pre-registered score stays reproducible; the adopted rules are a layer on top:

  Q1  the singular-order extension for quadratic elements is adopted (the former post hoc
      switch VERIF_POSTHOC_SINGULAR=1 is on), and pre-asymptotic convergence from EITHER
      direction is INCONCLUSIVE until it is independently verified:
        from above  observed order above 2 lam, within the element optimum (gates.py, as
                    before);
        from below  NEW: observed order below the theoretical one (more than tol), but the
                    sequence is monotone and its pairwise orders are still rising over the
                    finest levels, i.e. it is moving toward the theory. Below the theory
                    WITHOUT a rising trend stays FAIL.
      "Independently verified" is done by the existing cross-check (crosscheck.py) and by new
      unseen benchmarks; the label says so.
  Q2  absolute accuracy requirements, QoI-dependent: 1% for global quantities and
      displacements, 5% for region-averaged stresses (requirements.ABSOLUTE_TARGETS_ADOPTED;
      a check tolerance is only ever tightened), and U <= half-width / 3 where a constitutive
      range exists (requirements.constitutive_requirement).
  Q3  a further targeted refinement of the groove peak only through experiment.py: the agent
      states the justification, the hypothesis and the predicted outcome before it runs, and
      scores them after.

    with adopted.ruleset():
        ...run the rule plan as usual...
"""
import contextlib
import math
import os

import numpy as np

RULESET = "adopted-2026-11-24"
RISE_MIN = 0.02        # a rise smaller than this is round-off, not a trend (fixed in the unit test)
LABEL_BELOW = ("pre-asymptotic from below: order still rising toward theory; INCONCLUSIVE "
               "until independently verified (adopted 24 Nov 2026, advisor decision on Q1)")


def pairwise_orders(hs, values):
    """Observed orders from three consecutive levels: log(|d_i-1| / |d_i|) / log(h ratio),
    d_i = v_i+1 - v_i. Returns [] for fewer than three levels."""
    h, v = np.asarray(hs, float), np.asarray(values, float)
    d = np.diff(v)
    out = []
    for i in range(1, len(d)):
        if d[i] == 0 or d[i - 1] == 0 or np.sign(d[i]) != np.sign(d[i - 1]):
            out.append(math.nan)
            continue
        r = math.sqrt((h[i - 1] / h[i]) * (h[i] / h[i + 1]))
        out.append(math.log(abs(d[i - 1]) / abs(d[i])) / math.log(r))
    return out


def below_rule(gate, hs, values, n_levels=5):
    """Q1 'from below'. gate: a singular_order result. Returns the gate unchanged or relabelled
    INCONCLUSIVE."""
    if gate.get("verdict") != "FAIL" or gate.get("expected") is None:
        return gate
    if gate.get("mode") == "equal" and gate.get("observed", math.inf) > gate["expected"]:
        return gate                                     # above: gates.py's own rule decides
    if gate["expected"] < 0:                            # pointwise peak slope: never relabelled
        return gate
    hs, values = list(hs)[-n_levels:], list(values)[-n_levels:]
    d = np.diff(values)
    if len(values) < 4 or not (np.all(d > 0) or np.all(d < 0)):
        return gate
    po = pairwise_orders(hs, values)
    last = [p for p in po[-2:] if math.isfinite(p)]
    if len(last) == 2 and last[1] - last[0] > RISE_MIN and gate["observed"] < gate["expected"] - gate["tol"]:
        return {**gate, "verdict": "INCONCLUSIVE", "label": LABEL_BELOW, "pairwise_orders": po,
                "reason": gate["reason"] + f"; pairwise orders {', '.join(f'{p:.2f}' for p in po)} "
                          "still rising toward theory: pre-asymptotic from below (adopted rule)"}
    return {**gate, "pairwise_orders": po}


def _wrap(original):
    def run_check(adapter, check, qoi, params, **kw):
        r = original(adapter, check, qoi, params, **kw)
        if check == "singular_order":
            hs, vals, _ = adapter.series(qoi)
            r = below_rule(r, hs, vals)
        return r
    return run_check


@contextlib.contextmanager
def ruleset():
    """Run anything inside with the adopted rules; restores the defaults afterwards."""
    from omar_pfem.verification_agent import orchestrate
    saved_env = {k: os.environ.get(k) for k in ("VERIF_POSTHOC_SINGULAR", "VERIF_ABS_TARGETS")}
    original = orchestrate.run_check
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    os.environ["VERIF_ABS_TARGETS"] = "adopted"
    orchestrate.run_check = _wrap(original)
    try:
        yield RULESET
    finally:
        orchestrate.run_check = original
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
