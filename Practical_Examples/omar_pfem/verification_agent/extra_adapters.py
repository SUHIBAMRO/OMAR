"""Data adapters added after the pre-registrations froze orchestrate.py (round K, 2026-11-23).

orchestrate.py is part of the frozen file set of earlier pre-registrations; a new
adapter written there would change its hash. Adapters for new problems therefore live
here and are registered into orchestrate.ADAPTERS / ADAPTER_VOCAB at import time
(agent._default_spec imports this module). The adapter interface is the same.
"""
import json
import os

from omar_pfem.verification_agent import orchestrate


class LShapeExactAdapter:
    """HELD-OUT singular benchmark WITH an exact solution (lshape_exact.py); spec,
    description and predictions committed before any loaded solve
    (lshape_exact_preregistration.json); rules unchanged; the exact values are not
    available to the agent."""
    geometry = "l_shape_exact"
    description = (
        "L-shaped panel: union of the unit squares [0,1]x[0,1], [1,2]x[0,1], [0,1]x[1,2]; "
        "plane strain, small-strain linear elasticity, E = 1000, nu = 0.3. Prescribed "
        "displacements on the four outer edges x = 0, x = 2, y = 0, y = 2; the two edges "
        "through the re-entrant corner (1,1) are traction-free (270-degree free/free corner, "
        "lam = 0.5445). Q2 (9-node) elements, n x n per unit square (n = 2, 4, ..., 128, "
        "h = 1/n); strategies at n = 16: node coordinates graded toward x = 1 and y = 1 with "
        "g = 1.5, 2.0. QoIs: strain_energy = 1/2 u.K u; u_mid_left = u_x at (0.5,1.5); "
        "v_mid_right = u_y at (1.5,0.5); corner_gk_avg_von_mises = von Mises averaged with "
        "exp(-(d/0.1)^2), d the distance to (1,1); corner_disc_avg_von_mises = its mean over "
        "the disc d < 0.2; far_gk_avg_syy = sigma_yy averaged with exp(-(d/0.1)^2) about "
        "(0.5,0.5); peak_von_mises = max von Mises over all Gauss points.")
    qois = ("strain_energy", "u_mid_left", "v_mid_right", "corner_gk_avg_von_mises",
            "corner_disc_avg_von_mises", "far_gk_avg_syy", "peak_von_mises")

    def __init__(self, strategy_n=16):
        from omar_pfem.verification_agent.lshape_exact import CACHE
        rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
        self.rows = sorted((r for r in rows if r["g"] == 1.0), key=lambda r: r["n"])
        self.hs = [r["h"] for r in self.rows]
        self.strategy_rows = sorted((r for r in rows if r["n"] == strategy_n),
                                    key=lambda r: r["g"])

    @property
    def family(self):
        return [{"n_elem": r["n_elem"], "min_detF": None} for r in self.rows]

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def singular_exponent(self):
        from omar_pfem.verification_agent.lshape_exact import exponents
        return exponents()[0]

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


NEW = {"l_shape_exact": LShapeExactAdapter}
for _g, _A in NEW.items():
    orchestrate.ADAPTERS.setdefault(_g, _A)
    orchestrate.ADAPTER_VOCAB.setdefault(_g, (_A.description, _A.qois))
