"""Diagnosis of non-convergence: WHY did a quantity not converge, and what
should be investigated next?

Prof. Rabczuk (2026-11-19): "the key question the agent should figure out is
why we did not obtain convergence", with a classification into
mathematical/physical, numerical discretization, solver/implementation,
QoI/post-processing and surrogate-related causes; hypotheses generated from
regularity, geometry, constitutive behaviour and formulation; diagnostic
tests selected; and an autonomous decision on the next investigation,
including recognising when the evidence is insufficient.

Design
------
* PROBLEM FACTS (FACTS below) describe what the agent knows about each
  problem: corners/edges with their boundary conditions (-> singular
  exponents), finite-radius features, Poisson ratio, element, quadrature,
  and which evidence exists (solver-tolerance checks, linear-solver
  cross-check, MMS, full-integration re-solves, smooth-kernel and surface
  counterparts, strategies).
* HYPOTHESES (HYPOTHESES below): each has a cause class, a statement, an
  applicability condition on (facts, QoI, verdict) and a test. A test uses
  stored data only and returns SUPPORTED, REFUTED, or NEEDS_DATA together
  with the investigation that would decide it and a cost estimate.
* DECISION: for a QoI that failed or passed without a demonstrated rate,
  every applicable hypothesis is tested. The diagnosis is the set of
  supported hypotheses. The evidence is SUFFICIENT only when at least one
  hypothesis is supported and no applicable hypothesis is left untested;
  otherwise the agent names the cheapest open test as the next
  investigation. Supported hypotheses come with the remedy they imply.
* Quantities that converged WITH a rate are also checked for a REDUCED order
  (e.g. B3's global quantities at p ~ 1.2-1.6 with a degree-2 element): an
  order below the element's optimum is explained, or flagged, too.

The hypotheses encode diagnostic knowledge gathered on B1 and B3, so their
agreement with the manual findings there is a consistency check, not an
independent test; held-out problems are the real test.

    python -m omar_pfem.verification_agent.diagnose [--geometry ...]
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

CLASSES = ("mathematical/physical", "numerical discretization", "solver/implementation",
           "QoI/post-processing", "surrogate-related")
SUPPORTED, REFUTED, NEEDS_DATA, NOT_APPLICABLE = "supported", "refuted", "needs data", "n/a"


# --------------------------------------------------------------------------
# Problem facts
# --------------------------------------------------------------------------
def _b3_h_root():
    """Largest edge of the Q20 elements touching the groove-root node, per
    family level (corner-to-corner lengths): the local resolution of the
    groove curvature radius rho."""
    from omar_pfem.verification_agent import b3_global as b
    from omar_pfem.data.solve_quadratic_hex_groove_stress import (
        build_quadratic_mesh_with_curved_surfaces)
    cache = os.path.join(HERE, "diagnose_b3_h_root.json")
    if os.path.exists(cache):
        return {int(k): v for k, v in json.load(open(cache)).items()}
    res = list(b._family_resolutions())
    fine = os.path.join(HERE, "b3_fine_levels.json")
    if os.path.exists(fine):
        res += [tuple(r) for r in json.load(open(fine))]
    ref = np.array([b.R_IN0 - b.GROOVE_DEPTH, 0.0, b.LZ / 2.0])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
             (2, 6), (3, 7)]
    out = {}
    for r in res:
        nodes, el, *_ = build_quadratic_mesh_with_curved_surfaces(*r, groove_depth=b.GROOVE_DEPTH)
        root = int(np.argmin(np.linalg.norm(nodes - ref, axis=1)))
        adj = el[(el[:, :8] == root).any(axis=1)]
        if len(adj) == 0:                          # root on a midside node
            adj = el[(el == root).any(axis=1)]
        lens = [np.linalg.norm(nodes[e[a]] - nodes[e[c]]) for e in adj for a, c in edges]
        out[int(el.shape[0])] = {"h_max": float(max(lens)), "h_min": float(min(lens))}
    json.dump(out, open(cache, "w"), indent=1)
    return out


def _b1_mms_ok():
    d = json.load(open(os.path.join(PKG, "point9_results", "mms_B1_neo_hookean.json")))
    return d["rate_check"].get("Q4") == "as expected", "2D MMS (Q4): rates " + ", ".join(
        f"{k} {v['rate']:.2f} (theory {v['expected']})" for k, v in d["convergence_rates"]["Q4"].items()
        if k in ("L2", "H1_semi"))


def _cook_code_ok():
    """Cook is solved with scikit-fem (cook_ladder.py) using the residual and
    tangent of independent_solver.py; those reproduce the project's B1/B2
    solvers to round-off, whose Q4 element passes the 2D MMS."""
    path = os.path.join(HERE, "independent_solver_result.json")
    if not os.path.exists(path):
        return None, "no independent-solver comparison stored"
    cmp_ = json.load(open(path)).get("comparison", {})
    worst = max((r["rel_diff"] for p in ("b1", "b2") for r in cmp_.get(p, {}).get("same", [])),
                default=None)
    mms_ok, mms_txt = _b1_mms_ok()
    if worst is None:
        return None, "no same-discretisation comparison stored"
    return bool(worst < 1e-9 and mms_ok), (
        f"scikit-fem residual/tangent reproduce the project's B1/B2 solvers to {worst:.1e}; "
        + mms_txt)


def _q20_mms_ok():
    path = os.path.join(HERE, "mms_q20_result.json")
    if not os.path.exists(path):
        return None, "no Q20 MMS result"
    from omar_pfem.verification_agent.mms_q20 import rates
    rows = sorted((r for r in json.load(open(path)) if r["quadrature"] == "reduced"
                   and r["nu"] == 0.45), key=lambda r: r["n"])
    l2, h1 = rates(rows, "L2_rel")[-1], rates(rows, "H1_rel")[-1]
    return abs(l2 - 3) < 0.25 and abs(h1 - 2) < 0.25, (
        f"Q20 MMS (reduced rule, nu = 0.45): finest rates L2 {l2:.2f} (theory 3), "
        f"H1 {h1:.2f} (theory 2)")


def _b1_tolerance_change():
    rows = json.load(open(os.path.join(HERE, "b1_solver_check.json")))
    a, b_ = rows[0], rows[1]
    keys = [k for k in a if isinstance(a[k], float) and k not in ("h", "wall_clock_s",
                                                                  "newton_tol", "min_elem_size")]
    return max(abs(a[k] - b_[k]) / max(abs(a[k]), 1e-30) for k in keys), (
        f"N=41 solved at Newton tol {a['newton_tol']:g} and {b_['newton_tol']:g}")


def _b2_tolerance_change():
    from omar_pfem.verification_agent.b2_ladder import CACHE
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    tight = [r for r in rows if r["newton_tol"] < 1e-8 and r["r_grading"] == 1.0]
    if not tight:
        return None, "no B2 solve at a tighter Newton tolerance"
    t = tight[0]
    a = next(r for r in rows if r["N"] == t["N"] and r["newton_tol"] == 1e-8
             and r["r_grading"] == 1.0)
    keys = ("max_disp_mag", "strain_energy", "peak_PK1_frob", "interior_avg_PK1_frob",
            "interior_gk_PK1_frob", "u_inner", "v_mid")
    return max(abs(a[k] - t[k]) / abs(a[k]) for k in keys), (
        f"N={t['N']} solved at Newton tol 1e-8 and {t['newton_tol']:g}")


COOK_TOL = os.path.join(HERE, "cook_tolerance_check.json")


def _cook_tolerance_change():
    if not os.path.exists(COOK_TOL):
        return None, "no Cook solve at a tighter Newton tolerance"
    from omar_pfem.verification_agent.cook_ladder import CACHE
    t = json.load(open(COOK_TOL))
    a = next(r for r in json.load(open(CACHE)) if r["N"] == t["N"] and r["grading"] == 0.0)
    keys = ("u_tip", "v_tip", "max_disp_mag", "strain_energy", "interior_avg_PK1_frob",
            "interior_gk_PK1_frob", "peak_PK1_frob")
    return max(abs(a[k] - t[k]) / abs(a[k]) for k in keys), (
        f"N={t['N']} solved at Newton tol 1e-9 and {t['newton_tol']:g} "
        f"(final residuals {a['newton'][-1][1]:.1e} / {t['newton'][-1][1]:.1e})")


def _b3_tolerance_change():
    rows = json.load(open(os.path.join(HERE, "b3_extended.json")))
    out = 0.0
    for n in (2376, 10200):
        a = next(r for r in rows if r["n_elem"] == n and r["z_beta"] is None and r["rtol"] == 1e-8)
        b_ = next(r for r in rows if r["n_elem"] == n and r["z_beta"] is None and r["rtol"] == 1e-11)
        for k in ("reaction_moment_y", "strain_energy", "region_avg_sigma_frob",
                  "region_avg_pressure", "region_max_von_mises"):
            out = max(out, abs(a[k] - b_[k]) / abs(a[k]))
    return out, "2,376 and 10,200 el solved at rtol 1e-8 and 1e-11"


def _b3_linear_solver_change():
    path = os.path.join(HERE, "b3_solver_crosscheck.json")
    if not os.path.exists(path):
        return None, "no direct-solver re-solve stored"
    d = json.load(open(path))[0]
    rows = json.load(open(os.path.join(HERE, "b3_extended.json")))
    a = next(r for r in rows if r["n_elem"] == d["n_elem"] and r["z_beta"] is None
             and r["rtol"] == 1e-8)
    ch = max(abs(a[k] - d[k]) / abs(a[k]) for k in ("reaction_moment_y", "strain_energy",
                                                     "region_avg_sigma_frob", "region_avg_pressure"))
    return ch, f"{d['n_elem']} el re-solved with a direct solver instead of CG"


_PS_CACHE = {}


def _b3_pressure_scatter():
    if "v" not in _PS_CACHE:
        _PS_CACHE["v"] = _b3_pressure_scatter_uncached()
    return _PS_CACHE["v"]


def _b3_pressure_scatter_uncached():
    """Gauss-point pressure scatter within 2 R0 of the groove root, as a
    fraction of the region stress scale, per family level, and the same for
    the full-integration re-solves."""
    import glob
    out = {"reduced": {}, "full": {}}
    for kind, pat in (("reduced", "b3_gauss_near_groove/*_zbNone.npz"),
                      ("full", "b3_gauss_near_groove_fullint/*_full.npz")):
        for f in glob.glob(os.path.join(HERE, pat)):
            d = np.load(f)
            dist = np.linalg.norm(d["pos"] - d["ref"][None, :], axis=1)
            near = dist < 2 * float(d["radius"])
            p = -np.trace(d["sigma"][near], axis1=1, axis2=2) / 3.0
            out[kind][int(d["n_elem"])] = float(np.std(p))
    return out


FACTS = {
    "unit_square": {
        "name": "B1", "element": "Q4", "degree": 1, "quadrature": "full", "nu": 0.3,
        "corners": [{"where": "clamped bottom edge meets the free vertical edges",
                     "angle": np.pi / 2, "bc": ("clamped", "free")}],
        "corners_in_qoi_region": True,      # the peak is a maximum over the whole body
        "feature_radius": None,
        "code_check": _b1_mms_ok,
        "tolerance_check": _b1_tolerance_change,
        "linear_solver": ("direct sparse LU", None),
        "fullint": None,
    },
    "quarter_ring": {
        "name": "B2", "element": "Q4", "degree": 1, "quadrature": "full", "nu": 0.3,
        "corners": [], "corners_in_qoi_region": False, "feature_radius": None,
        "code_check": _b1_mms_ok,            # same Q4 element and energy as B1
        "tolerance_check": _b2_tolerance_change,
        "linear_solver": ("direct sparse LU", None),
        "fullint": None,
    },
    # held-out singular problem; facts are geometry/discretisation only
    "cook_membrane": {
        "name": "Cook", "element": "Q4", "degree": 1, "quadrature": "full", "nu": 0.3,
        "corners": [{"where": "clamped left edge meets the free top edge at (0,44)",
                     "angle": np.pi / 2 + np.arctan(16 / 48), "bc": ("clamped", "free")},
                    {"where": "clamped left edge meets the free bottom edge at (0,0)",
                     "angle": np.pi / 2 - np.arctan(44 / 48), "bc": ("clamped", "free")}],
        "corners_in_qoi_region": True,      # the peak is a maximum over the whole body
        "feature_radius": None,
        "code_check": _cook_code_ok,
        "tolerance_check": _cook_tolerance_change,
        "linear_solver": ("direct sparse LU", None),
        "fullint": None,
    },
    "bushing_groove": {
        "name": "B3", "element": "Q20", "degree": 2, "quadrature": "reduced", "nu": 0.45,
        "corners": [{"where": "Dirichlet inner/outer surfaces meet the free end faces "
                              "(four circular edges)", "angle": np.pi / 2,
                     "bc": ("clamped", "free")}],
        # the stress QoIs are taken in a sphere at the groove root (z = 0.5),
        # the singular edges lie on the end faces z = 0 and z = 1
        "corners_in_qoi_region": False,
        "feature_radius": {"name": "groove root curvature radius", "rho": 0.0228,
                           "h_root": _b3_h_root},
        "code_check": _q20_mms_ok,
        "tolerance_check": _b3_tolerance_change,
        "linear_solver": ("Jacobi-preconditioned CG", _b3_linear_solver_change),
        "fullint": "b3_fullint.json",
        "pressure_scatter": _b3_pressure_scatter,
    },
}


def corner_lambda(facts):
    """Smallest singular exponent over the problem's corners/edges (plane
    strain eigenproblem), or None."""
    from omar_pfem.verification_agent import singularity as S
    lams = [S.smallest_exponent(c["angle"], c["bc"][0], c["bc"][1], facts["nu"], "strain")
            for c in facts["corners"]]
    lams = [l for l in lams if l is not None]      # regular corners have no exponent < 1
    return min(lams) if lams else None


# --------------------------------------------------------------------------
# Context: what the agent knows about one QoI
# --------------------------------------------------------------------------
@dataclass
class Ctx:
    geometry: str
    facts: dict
    qoi: str
    qoi_class: str
    verdict: str
    evidence: str
    gates: dict
    all_verdicts: dict            # {qoi: (overall, evidence)} for the whole problem
    adapter: object
    all_gates: dict = field(default_factory=dict)   # {qoi: {check: result}}
    cache: dict = field(default_factory=dict)

    def lam(self):
        if "lam" not in self.cache:
            self.cache["lam"] = corner_lambda(self.facts)
        return self.cache["lam"]


def _r(status, evidence, next_action=None, cost=None, remedy=None):
    return {"status": status, "evidence": evidence, "next_action": next_action,
            "cost": cost, "remedy": remedy}


# --------------------------------------------------------------------------
# Hypotheses and their tests
# --------------------------------------------------------------------------
def t_point_singularity(c):
    lam = c.lam()
    g = c.gates.get("singular_order")
    hs, vals, _ = c.adapter.series(c.qoi)
    slope = float(np.polyfit(np.log(hs[-5:]), np.log(np.abs(vals[-5:])), 1)[0])
    if lam is None:
        return _r(REFUTED, "no corner or edge with a singular exponent below 1 in the geometry")
    if g is not None and g.get("verdict") == "PASS":
        return _r(SUPPORTED, f"peak grows like h^{slope:.3f}; theory h^(lambda-1) = "
                             f"h^{lam - 1:.3f} (lambda = {lam:.4f})",
                  remedy="not a usable reference value: no finite limit exists; use energy, "
                         "displacements or region averages, or a singular-field quantity "
                         "(stress-intensity factor)")
    if slope < -0.05 and abs(slope - (lam - 1)) < 0.15:
        return _r(SUPPORTED, f"log-log slope of the peak {slope:.3f} matches lambda-1 = "
                             f"{lam - 1:.3f}",
                  remedy="not a usable reference value: no finite limit exists")
    return _r(REFUTED, f"peak slope {slope:.3f} does not match a growth h^{lam - 1:.3f}")


def t_finite_radius_underresolved(c):
    fr = c.facts["feature_radius"]
    if c.qoi_class == "region_average":
        sm = _smooth_verdict(c)
        kern, kv = sm if sm else (None, None)
        if c.qoi.startswith("gk2_"):
            kern, kv = c.qoi, c.all_verdicts[c.qoi]
        if kern and kv == ("PASS", "rate"):
            return _r(REFUTED, f"averaged over the region, the field converges with a rate on "
                               f"the same meshes ({kern}): resolution of the feature is "
                               "sufficient for averaged quantities")
    h = fr["h_root"]()
    fam = [g["n_elem"] for g in getattr(c.adapter, "family", [])]
    if not fam:
        return _r(NEEDS_DATA, "no family element counts available",
                  next_action="measure the element size at the feature", cost="seconds")
    finest = h[max(fam)]
    ratio = finest["h_max"] / fr["rho"]
    coarse = h[min(fam)]["h_max"] / fr["rho"]
    target = fr["rho"] / 4
    # cost of reaching h <= rho/4 by uniform refinement: elements ~ (h/target)^3
    n_needed = max(fam) * (finest["h_max"] / target) ** 3
    if ratio > 0.5:
        return _r(SUPPORTED,
                  f"the {fr['name']} rho = {fr['rho']} is resolved by elements of size "
                  f"{finest['h_max']:.4f} at the finest level (h/rho = {ratio:.2f}; coarsest "
                  f"{coarse:.2f}): the finite-radius concentration is not yet resolved, so its "
                  "peak is pre-asymptotic, not divergent",
                  next_action=f"local refinement around the feature to h <= rho/4 = {target:.4f} "
                              f"(uniform refinement would need about {n_needed:,.0f} elements)",
                  cost="hours (local), infeasible uniformly on this machine",
                  remedy="local refinement at the feature, or a higher-order/IGA discretization "
                         "of the curved boundary; the peak must converge once h << rho")
    return _r(REFUTED, f"feature resolved: h/rho = {ratio:.2f} at the finest level")


def _conv_only(gates_of_qoi):
    """(overall, evidence) from the two convergence checks alone (no
    strategy checks): what refinement alone shows."""
    from omar_pfem.verification_agent.gates import combine_verdicts
    conv = {k: v for k, v in gates_of_qoi.items()
            if k in ("qoi_convergence", "qoi_convergence_multilevel")}
    return combine_verdicts(conv)[:2]


_RANK = {("FAIL", "none"): 0, ("FAIL", "band"): 0, ("INCONCLUSIVE", "none"): 0,
         ("PASS", "band"): 1, ("PASS", "rate"): 2}


def _smooth_counterpart(c):
    q = c.qoi
    if q.startswith("interior_avg"):
        k = q.replace("interior_avg", "interior_gk")
    elif not q.startswith("gk2_") and q.endswith("_avg"):
        k = "gk2_" + q
    else:
        return None
    return k if k in c.all_verdicts else None


_B3_COMP = {"sigma_xy_avg": ("cart", 0, 1), "sigma_yz_avg": ("cart", 1, 2),
            "cyl_rt_avg": ("cyl", 0, 1), "cyl_tz_avg": ("cyl", 1, 2)}


def _b3_kernel_on_the_fly(q, adapter):
    """Gaussian-weighted (l = 2 R0) family series of a B3 stress component
    that has no stored counterpart, from the Gauss-point dumps, and its
    convergence-only verdict. This IS an investigation the agent runs itself
    (seconds, no solve)."""
    from omar_pfem.verification_agent import region_study as RS, gates, trees
    kind, i, j = _B3_COMP[q]
    fam, _ = RS.load()
    levels = [int(g["n_elem"]) for g in adapter.family]       # the same solutions
    fam = sorted((r for r in fam if r["n_elem"] in levels), key=lambda r: r["n_elem"])
    assert [r["n_elem"] for r in fam] == levels, "Gauss-point dumps missing for a level"
    R0 = float(fam[0]["radius"])
    hs, vals = [], []
    for rec in fam:
        w = RS.weights(rec, "gauss", 2 * R0)[0]
        s = rec["sigma"]
        if kind == "cyl":
            th = np.arctan2(rec["pos"][:, 1], rec["pos"][:, 0])
            co, sn = np.cos(th), np.sin(th)
            Q = np.zeros((len(th), 3, 3))
            Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1], Q[:, 2, 2] = co, sn, -sn, co, 1.0
            s = np.einsum("nai,nij,nbj->nab", Q, s, Q)
        vals.append(float((s[:, i, j] * w).sum() / w.sum()))
        hs.append(rec["n_elem"] ** (-1 / 3))
    cp = trees.class_params("qoi_convergence", "region_average")
    win = RS.ORDER_WINDOW
    scale = adapter.kernel_scale
    g3 = gates.qoi_convergence_gate(hs, vals, scale=scale, order_window=win, **cp)
    g5 = gates.multilevel_convergence_gate(hs, vals, scale=scale, order_window=win, **cp)
    return gates.combine_verdicts({"qoi_convergence": g3, "qoi_convergence_multilevel": g5})[:2]


_KF_CACHE = {}


def _smooth_verdict(c):
    """(name, (overall, evidence)) of the smooth-weight counterpart of a
    region average: the stored one, or, for B3 components without one, the
    Gaussian-weighted average computed here from the Gauss-point dumps."""
    kern = _smooth_counterpart(c)
    if kern is not None:
        return kern, c.all_verdicts[kern]
    if c.geometry == "bushing_groove" and c.qoi in _B3_COMP:
        if c.qoi not in _KF_CACHE:
            _KF_CACHE[c.qoi] = _b3_kernel_on_the_fly(c.qoi, c.adapter)
        return f"Gaussian-weighted {c.qoi} (computed here)", _KF_CACHE[c.qoi]
    return None


def t_sharp_sampling(c):
    kern = _smooth_counterpart(c)
    if kern is None and c.geometry == "bushing_groove" and c.qoi in _B3_COMP:
        kv = _smooth_verdict(c)[1]
        sv = _conv_only(c.gates)
        better = _RANK.get(kv, 0) > _RANK.get(sv, 0)
        return _r(SUPPORTED if better else REFUTED,
                  f"computed here from the stored Gauss-point data: the Gaussian-weighted "
                  f"average of this component gives {kv[0]}/{kv[1]} under refinement, the sharp "
                  f"one {sv[0]}/{sv[1]}" + ("" if better else ": no better, so the sharp cut-off "
                                                              "is not the cause"),
                  remedy="add the Gaussian-weighted average of this component as a QoI"
                  if better else None)
    if kern is None:
        return _r(NEEDS_DATA, "no smooth-weight counterpart of this region average",
                  next_action="dump Gauss-point data near the region and evaluate a "
                              "Gaussian-weighted average on the same solutions",
                  cost="one re-solve per level with a dump, then seconds")
    ov, ev = c.all_verdicts[kern]
    if _RANK.get((ov, ev), 0) > _RANK.get((c.verdict, c.evidence), 0):
        return _r(SUPPORTED, f"the same field averaged with a smooth weight ({kern}: {ov}, "
                             f"{ev}) does better on the same solutions ({c.verdict}, "
                             f"{c.evidence}); the sharp cut-off samples a changing set of Gauss "
                             "points from mesh to mesh",
                  remedy=f"report {kern} as a separate QoI next to the sharp one")
    return _r(REFUTED, f"the smooth-weight counterpart {kern} does no better ({ov}, {ev}): "
                       "the sharp cut-off is not the cause")


def t_norm_accumulation(c):
    if not any(k in c.qoi for k in ("frob", "von_mises")):
        return _r(NOT_APPLICABLE, "not a magnitude measure")
    kern = _smooth_counterpart(c)
    if kern and c.all_verdicts[kern] == ("PASS", "rate"):
        return _r(REFUTED, f"the smooth average of the same magnitude ({kern}) converges with a "
                           "rate: no point-to-point oscillation accumulates")
    prefix = "gk2_" if c.qoi.startswith("gk2_") else ""
    signed = [prefix + s for s in ("sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg",
                                   "pressure_avg")]
    signed = [s for s in signed if s in c.all_verdicts]
    if not signed:
        return _r(NEEDS_DATA, "no signed components of the same region evaluated",
                  next_action="evaluate the signed stress components over the same region",
                  cost="seconds")
    ok = [s for s in signed if _conv_only(c.all_gates[s])[0] == "PASS"]
    if len(ok) == len(signed) and _conv_only(c.gates) != ("PASS", "rate"):
        return _r(SUPPORTED, f"under refinement the signed components of the same region pass "
                             f"({', '.join(ok)}) while the magnitude measure does not reach a "
                             "rate: point-to-point oscillation cancels in signed averages but "
                             "accumulates in magnitudes",
                  remedy="use signed (normal) components; find the source of the oscillation")
    return _r(REFUTED, "the signed components of the same region do not pass either")


def t_gauss_location(c):
    if not c.qoi.endswith("_max"):
        return _r(NOT_APPLICABLE, "not a maximum")
    path = os.path.join(HERE, "surface_study_result.json")
    if not os.path.exists(path):
        return _r(NEEDS_DATA, "maxima are taken over Gauss points inside the body, not on the "
                              "surface where the peak of a notch lies",
                  next_action="extrapolate Gauss stresses to the surface nodes and re-test",
                  cost="seconds from stored data")
    s = json.load(open(path))["gates"]
    root = {k: v["overall"] + "/" + v["evidence"] for k, v in s.items() if k.startswith("root_")}
    surf = {k: v["overall"] for k, v in s.items() if k.startswith("surface_max")}
    stable = [k for k, v in root.items() if v.startswith("PASS")]
    return _r(SUPPORTED if stable else REFUTED,
              f"surface values at the groove root: {', '.join(f'{k} {v}' for k, v in root.items())}"
              f"; surface maxima: {sorted(set(surf.values()))}. The Gauss-point maximum mixes "
              "location (points 0.2h inside) with resolution",
              remedy="evaluate notch stresses on the surface (extrapolated nodal values) at a "
                     "fixed point, not as a maximum over Gauss points")


def t_reduced_integration(c):
    if c.facts["quadrature"] != "reduced":
        return _r(NOT_APPLICABLE, "full integration is used")
    sc = c.facts.get("pressure_scatter")
    if c.facts.get("fullint") is None or sc is None:
        return _r(NEEDS_DATA, "reduced integration may admit spurious (hourglass) modes",
                  next_action="re-solve two levels with the full rule and compare",
                  cost="minutes")
    d = sc()
    common = sorted(set(d["reduced"]) & set(d["full"]))
    txt = "; ".join(f"{n} el: pressure std {d['reduced'][n]:.2f} (reduced) vs "
                    f"{d['full'][n]:.2f} (full)" for n in common)
    if all(d["full"][n] < 0.5 * d["reduced"][n] for n in common):
        return _r(SUPPORTED, "full integration removes the Gauss-point oscillation: " + txt,
                  remedy="use the full rule or hourglass control")
    return _r(REFUTED, "full integration does not remove the Gauss-point oscillation: " + txt)


def t_volumetric(c):
    nu = c.facts["nu"]
    if nu < 0.4:
        return _r(REFUTED, f"nu = {nu}: far from incompressibility")
    sc = c.facts.get("pressure_scatter")
    code_ok, code_txt = c.facts["code_check"]()
    if sc is None:
        return _r(NEEDS_DATA, f"nu = {nu}: near-incompressible; pressure oscillation possible",
                  next_action="measure the Gauss-point pressure scatter near the feature",
                  cost="seconds from stored data")
    d = sc()["reduced"]
    lv = sorted(d)
    scale = c.adapter.scale
    frac = d[lv[-1]] / scale
    txt = (f"nu = {nu}; Gauss-point pressure std within 2R0 falls "
           + " -> ".join(f"{d[n]:.2f}" for n in lv) + f" (levels {lv[0]}-{lv[-1]} el), "
           f"{frac:.0%} of the region stress scale at the finest; "
           f"displacements are not locked ({code_txt})")
    if frac < 0.10:
        return _r(REFUTED, txt + ": too small to matter")
    return _r(SUPPORTED, txt,
              next_action="cross-check with a mixed u/p or F-bar formulation",
              cost="needs a formulation not available in torchfem",
              remedy="mixed u/p or F-bar formulation; until then use signed or smoothly "
                     "weighted stress measures")


def t_nonlinear_solver(c):
    f = c.facts.get("tolerance_check")
    if f is None:
        return _r(NEEDS_DATA, "no solve at a tighter Newton tolerance stored",
                  next_action="re-solve one level at a 1000x tighter tolerance and compare",
                  cost="minutes")
    ch, where = f()
    if ch is None:
        return _r(NEEDS_DATA, where, next_action="re-solve one level at a 100x tighter "
                                                 "Newton tolerance and compare", cost="minutes")
    if ch < 1e-8:
        return _r(REFUTED, f"{where}: quantities change by at most {ch:.1e} (relative)")
    return _r(SUPPORTED, f"{where}: quantities change by {ch:.1e}",
              remedy="tighten the Newton tolerance")


def t_linear_solver(c):
    name, f = c.facts["linear_solver"]
    if f is None:
        return _r(REFUTED, f"{name}: exact up to round-off")
    ch, where = f()
    if ch is None:
        return _r(NEEDS_DATA, f"{name} not cross-checked",
                  next_action="re-solve one level with a direct solver", cost="minutes")
    if ch < 1e-8:
        return _r(REFUTED, f"{where}: quantities change by at most {ch:.1e}")
    return _r(SUPPORTED, f"{where}: quantities change by {ch:.1e}")


def t_implementation(c):
    ok, txt = c.facts["code_check"]()
    if ok is None:
        return _r(NEEDS_DATA, txt, next_action="run a manufactured-solution test for the element",
                  cost="minutes")
    return _r(REFUTED if ok else SUPPORTED, txt + ("" if ok else ": rates off theory"))


def t_preasymptotic(c):
    hs, vals, _ = c.adapter.series(c.qoi)
    if len(vals) < 5:
        return _r(NEEDS_DATA, "fewer than five levels", next_action="add a finer level",
                  cost="see the adaptive controller's cost model")
    d = np.diff(np.asarray(vals[-5:], float))
    shrinking = abs(d[-1]) < abs(d[0])
    sign_changes = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
    if shrinking and sign_changes <= 1:
        return _r(SUPPORTED, f"increments over the finest five levels shrink "
                             f"({abs(d[0]):.3g} -> {abs(d[-1]):.3g}) with {sign_changes} sign "
                             "change(s): the sequence may still enter the asymptotic range",
                  next_action="add a finer level", cost="see the adaptive controller")
    return _r(REFUTED, f"increments over the finest five levels do not settle "
                       f"({sign_changes} sign changes, last/first {abs(d[-1]) / abs(d[0]):.2f})")


def t_physical_model(c):
    rows = getattr(c.adapter, "family", None) or getattr(c.adapter, "rows", [])
    dets = [r.get("min_detF") for r in rows if r.get("min_detF") is not None]
    if not dets:
        return _r(NEEDS_DATA, "no deformation measures stored",
                  next_action="record min det F and the largest principal stretch",
                  cost="seconds per level")
    lo = min(dets)
    if 0.5 < lo < 2.0:
        return _r(REFUTED, f"min det F >= {lo:.3f} on every level: moderate deformation, within "
                           "the range of the compressible neo-Hookean model; no sign that a "
                           "different model or scale is needed",
                  remedy=None)
    return _r(SUPPORTED, f"min det F reaches {lo:.3f}: extreme volume change; check whether "
                         "the constitutive model (and its compressibility) is adequate")


def t_surrogate(c):
    return _r(NOT_APPLICABLE, "this is a reference (FE) quantity; surrogate-related causes "
                              "apply when a surrogate's error, not the reference, fails to "
                              "converge (needs the surrogate evaluations)")


def t_reduced_order(c):
    """For a rate-converged QoI: is its order below the element's optimum,
    and do corner/edge singularities explain it?"""
    p = c.gates.get("qoi_convergence_multilevel", {}).get("fit_order")
    if p is None:
        return _r(NOT_APPLICABLE, "no fitted order")
    k = c.facts["degree"]
    optimal = {"global": 2 * k, "displacement": k + 1}.get(c.qoi_class)
    if optimal is None or p > optimal - 0.3:
        return _r(NOT_APPLICABLE, f"order {p:.2f}, element optimum {optimal}")
    lam = c.lam()
    if lam is None:
        return _r(NEEDS_DATA, f"order {p:.2f} below the element optimum {optimal}, no "
                              "singular feature known",
                  next_action="look for corners, edges or material interfaces")
    pred = 2 * lam
    txt =(f"observed order {p:.2f} below the element optimum {optimal}; the corners/edges "
           f"({'; '.join(x['where'] for x in c.facts['corners'])}) have lambda = {lam:.4f}, "
           f"predicting order 2 lambda = {pred:.2f} for energy-type quantities")
    if abs(p - pred) < 0.4:
        return _r(SUPPORTED, txt, remedy="expected for this geometry; to recover the element "
                                         "order, grade the mesh toward the singular edges")
    return _r(REFUTED, txt + f"; the observed order differs by {abs(p - pred):.2f}")


@dataclass
class Hyp:
    id: str
    cls: str
    statement: str
    applies: object
    test: object
    order: int = 0              # tie-break for the next investigation (cheap first)


def _is_peak(c):
    return c.qoi_class == "pointwise_peak"


HYPOTHESES = [
    Hyp("M1", "mathematical/physical",
        "a geometric singularity makes the true peak unbounded",
        lambda c: _is_peak(c) and bool(c.facts["corners"]) and c.facts["corners_in_qoi_region"],
        t_point_singularity),
    Hyp("M2", "mathematical/physical",
        "corner/edge singularities reduce the convergence order of global quantities",
        lambda c: c.qoi_class in ("global", "displacement") and c.verdict == "PASS",
        t_reduced_order),
    Hyp("P1", "mathematical/physical",
        "the physical/constitutive model is inadequate for the deformation reached",
        lambda c: True, t_physical_model),
    Hyp("D1", "numerical discretization",
        "a finite-radius stress concentration is not yet resolved (pre-asymptotic)",
        lambda c: c.facts["feature_radius"] is not None and c.qoi_class in
        ("pointwise_peak", "region_average"), t_finite_radius_underresolved),
    Hyp("D2", "numerical discretization",
        "reduced integration admits spurious modes",
        lambda c: c.qoi_class in ("pointwise_peak", "region_average"), t_reduced_integration),
    Hyp("D3", "numerical discretization",
        "near-incompressibility causes volumetric locking / pressure oscillation",
        lambda c: c.qoi_class == "pointwise_peak" or any(
            k in c.qoi for k in ("frob", "von_mises", "pressure")), t_volumetric),
    Hyp("D4", "numerical discretization",
        "the sequence is still pre-asymptotic",
        lambda c: c.verdict != "PASS" or c.evidence != "rate", t_preasymptotic),
    Hyp("S1", "solver/implementation", "the Newton tolerance is too loose",
        lambda c: True, t_nonlinear_solver),
    Hyp("S2", "solver/implementation", "the linear solver is inaccurate",
        lambda c: True, t_linear_solver),
    Hyp("S3", "solver/implementation", "the element implementation is wrong",
        lambda c: True, t_implementation),
    Hyp("Q1", "QoI/post-processing",
        "a sharp-edged region samples a changing set of Gauss points",
        lambda c: c.qoi_class == "region_average" and not c.qoi.startswith("gk2_")
        and "gk_" not in c.qoi, t_sharp_sampling),
    Hyp("Q2", "QoI/post-processing",
        "a magnitude measure accumulates point-to-point oscillation",
        lambda c: c.qoi_class == "region_average", t_norm_accumulation),
    Hyp("Q3", "QoI/post-processing",
        "the maximum is taken at Gauss points inside the body, not on the surface",
        lambda c: c.qoi.endswith("_max"), t_gauss_location),
    Hyp("U1", "surrogate-related", "the surrogate, not the reference, fails",
        lambda c: True, t_surrogate),
]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _escalation(supported):
    from omar_pfem.verification_agent.model_hierarchy import escalate
    return escalate(supported)


def diagnose_problem(geometry, qois=None):
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.verify_report import rules
    R = rules(geometry)
    facts = FACTS[geometry]
    out = {}
    targets = qois or list(R["combine"])
    for q in targets:
        overall, ev = R["combine"][q][:2]
        needs = overall != "PASS" or ev != "rate"
        c = Ctx(geometry, facts, q, trees.classify_qoi(q), overall, ev, R["v"][q],
                {k: v[:2] for k, v in R["combine"].items()}, R["adapter"], R["v"])
        results = []
        for h in HYPOTHESES:
            if not needs and h.id != "M2":
                continue
            if not h.applies(c):
                continue
            r = h.test(c)
            if r["status"] == NOT_APPLICABLE:
                continue
            results.append({"id": h.id, "class": h.cls, "hypothesis": h.statement, **r})
        if not results:
            continue
        supported = [r for r in results if r["status"] == SUPPORTED]
        open_ = [r for r in results if r["status"] == NEEDS_DATA]
        out[q] = {
            "verdict": overall, "evidence": ev,
            "diagnosis": sorted({r["class"] for r in supported}),
            "supported": [r["id"] for r in supported],
            "refuted": [r["id"] for r in results if r["status"] == REFUTED],
            "open": [r["id"] for r in open_],
            "evidence_sufficient": bool(supported) and not open_,
            "next_investigation": (open_[0]["next_action"] if open_ else
                                   next((r["next_action"] for r in supported
                                         if r.get("next_action")), None)),
            "remedies": [r["remedy"] for r in supported if r.get("remedy")],
            "escalation": _escalation([r["id"] for r in supported]),
            "tests": results,
        }
    return out


# --------------------------------------------------------------------------
# Autonomous investigation: run the cheap tests the diagnosis asks for
# --------------------------------------------------------------------------
def _act_b2_tolerance():
    from omar_pfem.verification_agent.b2_ladder import run_ladder
    run_ladder([49], tol=1e-10)


def _act_cook_tolerance():
    from omar_pfem.verification_agent.cook_ladder import solve_level
    r = solve_level(64, tol=1e-12)
    r["newton_tol"] = 1e-12
    json.dump(r, open(COOK_TOL, "w"), indent=1)


def _act_b3_direct_solver():
    from omar_pfem.verification_agent import b3_global as b
    r = b.solve_extended(*b._family_resolutions()[0], linear_solver="direct")
    keys = ("n_elem", "linear_solver", "elapsed_s", "reaction_moment_y", "strain_energy",
            "region_avg_sigma_frob", "region_avg_pressure", "region_max_von_mises", "min_detF")
    json.dump([{k: r[k] for k in keys}],
              open(os.path.join(HERE, "b3_solver_crosscheck.json"), "w"), indent=1)


# (geometry, hypothesis) -> (action, estimated seconds, what it does)
ACTIONS = {
    ("quarter_ring", "S1"): (_act_b2_tolerance, 90,
                             "re-solve B2 at N=49 with Newton tol 1e-10"),
    ("cook_membrane", "S1"): (_act_cook_tolerance, 10,
                              "re-solve Cook at N=64 with Newton tol 1e-12"),
    ("bushing_groove", "S2"): (_act_b3_direct_solver, 160,
                               "re-solve B3 at 2,376 el with a direct linear solver"),
}


# ---- confirmatory (costly) investigations ---------------------------------------------------
# A SUPPORTED hypothesis predicts something that a further computation can
# check. CONFIRM maps (geometry, hypothesis) -> (estimate() -> (seconds, GB,
# what), run(), check() -> (confirmed: bool, evidence)). The agent runs one
# only if the estimate fits the time budget and the memory limit.
MEM_LIMIT_GB = 12.0


def _power_cost(ns, ts, n_new):
    """time ~ a n^alpha fitted on the two largest previous solves."""
    (n1, t1), (n2, t2) = sorted(zip(ns, ts))[-2:]
    alpha = np.log(t2 / t1) / np.log(n2 / n1)
    return t2 * (n_new / n2) ** alpha, alpha


def _cook_rows():
    from omar_pfem.verification_agent.cook_ladder import CACHE
    return sorted((r for r in json.load(open(CACHE)) if r["grading"] == 0.0),
                  key=lambda r: r["N"])


def _cook_finer_estimate():
    rows = _cook_rows()
    N = 2 * rows[-1]["N"]
    t, a = _power_cost([r["n_dof"] for r in rows], [r["wall_clock_s"] for r in rows],
                       rows[-1]["n_dof"] * 4)
    gb = 3.3 * (N / 512) ** 2                       # measured 3.3 GB at N = 512
    return t, gb, f"solve Cook at N = {N} (est. {t:,.0f} s from time ~ n^{a:.2f}, {gb:.1f} GB)"


def _cook_finer_run():
    from omar_pfem.verification_agent.cook_ladder import run_ladder
    run_ladder([2 * _cook_rows()[-1]["N"]])


def _local_order_check(values, theory):
    v = np.asarray(values, float)
    p = np.log2(np.abs(np.diff(v)[:-1] / np.diff(v)[1:]))
    moved = abs(p[-1] - theory) < abs(p[-2] - theory)
    return bool(moved), (f"local orders {p[-2]:.3f} -> {p[-1]:.3f} (theory {theory:.3f}): "
                         + ("moving toward theory, as a pre-asymptotic sequence should"
                            if moved else "NOT moving toward theory"))


def _cook_finer_check():
    lam = corner_lambda(FACTS["cook_membrane"])
    return _local_order_check([r["strain_energy"] for r in _cook_rows()], 2 * lam)


def _b3_fbar_estimate():
    return 300.0, 3.0, "re-solve B3 at 2,376 el with the mean-dilatation (F-bar, Q20/P0) " \
                       "formulation (mixed_b3.py)"


def _b3_fbar_run():
    from omar_pfem.verification_agent import mixed_b3
    cache = json.load(open(mixed_b3.OUT)) if os.path.exists(mixed_b3.OUT) else {}
    mixed_b3.run(2376, "reduced", "disp", cache)
    mixed_b3.run(2376, "reduced", "fbar", cache)


def _b3_fbar_check():
    from omar_pfem.verification_agent import mixed_b3
    m = json.load(open(mixed_b3.OUT))
    d, f = m["2376:reduced:disp"], m["2376:reduced:fbar"]
    ratio = d["pressure_std_2R0"] / f["pressure_std_2R0"]
    return bool(ratio >= 2.0), (f"Gauss-point pressure std within 2R0: displacement "
                                f"{d['pressure_std_2R0']:.3f}, F-bar {f['pressure_std_2R0']:.3f} "
                                f"(factor {ratio:.1f}; prediction: >= 2)")


CONFIRM = {
    ("cook_membrane", "D4"): (_cook_finer_estimate, _cook_finer_run, _cook_finer_check),
    ("bushing_groove", "D3"): (_b3_fbar_estimate, _b3_fbar_run, _b3_fbar_check),
}


def confirm(geometries, budget_s, res=None, log=print):
    """Run the affordable confirmatory investigations for supported
    hypotheses; record skipped ones with the reason."""
    import time
    res = res or {g: diagnose_problem(g) for g in geometries}
    out, spent = [], 0.0
    for (g, h), (est, run, check) in CONFIRM.items():
        if g not in res or not any(h in d["supported"] for d in res[g].values()):
            continue
        t, gb, what = est()
        rec = {"geometry": g, "hypothesis": h, "action": what, "est_seconds": t, "est_GB": gb}
        if gb > MEM_LIMIT_GB:
            rec["status"] = f"skipped: needs ~{gb:.0f} GB > {MEM_LIMIT_GB:g} GB"
        elif spent + t > budget_s:
            rec["status"] = f"skipped: ~{t:,.0f} s exceeds the remaining budget"
        else:
            log(f"[confirm] {FACTS[g]['name']} {h}: {what}")
            t0 = time.time()
            run()
            spent += time.time() - t0
            ok, ev = check()
            rec.update(status="run", seconds=time.time() - t0, confirmed=ok, evidence=ev)
        if rec["status"] != "run":
            try:                                    # the check on data already present
                ok, ev = check()
                rec.update(existing_data_confirms=ok, evidence=ev)
            except Exception as e:                  # no data yet
                rec["evidence"] = f"no data yet ({type(e).__name__})"
        log(f"[confirm] {g} {h}: {rec['status']}; {rec.get('evidence', '')}")
        out.append(rec)
    return out


def investigate(geometries, budget_s, log=print):
    """Diagnose, run the cheapest executable open test within the budget,
    re-diagnose; repeat until the evidence is sufficient, nothing executable
    is left, or the budget is spent. Returns the final diagnoses and the
    log of what was run and why."""
    import time
    spent, done, history = 0.0, set(), []
    while True:
        res = {g: diagnose_problem(g) for g in geometries}
        todo = {}
        for g, d in res.items():
            for q, r in d.items():
                for h in r["open"]:
                    if (g, h) in ACTIONS and (g, h) not in done:
                        todo.setdefault((g, h), []).append(q)
        if not todo:
            return res, history
        (g, h), qs = min(todo.items(), key=lambda kv: ACTIONS[kv[0]][1])
        fn, est, what = ACTIONS[(g, h)]
        if spent + est > budget_s:
            log(f"[investigate] next test '{what}' (~{est}s) exceeds the budget; stopping")
            return res, history
        log(f"[investigate] {FACTS[g]['name']}: hypothesis {h} untested for {len(qs)} QoI(s) "
            f"-> {what} (~{est}s)")
        t0 = time.time()
        fn()
        dt = time.time() - t0
        spent += dt
        done.add((g, h))
        history.append({"geometry": g, "hypothesis": h, "action": what, "seconds": dt,
                        "qois_waiting": qs})


# What the manual investigations (PROJECT_STATUS, 2026-11-16..18) established.
# A consistency check only: the hypotheses were written knowing these results.
MANUAL = {
    ("unit_square", "peak_PK1_frob"): ({"M1"}, set()),
    ("unit_square", "strain_energy"): ({"M2"}, set()),
    ("quarter_ring", "interior_avg_PK1_frob"): ({"Q1"}, set()),
    **{("bushing_groove", q): ({"Q1"}, set()) for q in (
        "sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg", "cyl_rr_avg",
        "cyl_tt_avg", "cyl_zz_avg")},
    ("bushing_groove", "sigma_frob_avg"): ({"D3", "Q2"}, {"D2"}),
    ("bushing_groove", "gk2_sigma_frob_avg"): ({"D3", "Q2"}, {"D2"}),
    **{("bushing_groove", q): ({"Q3"}, {"D2"}) for q in (
        "sigma_xx_max", "sigma_zz_max", "von_mises_max", "cyl_rr_max")},
}


def consistency(res):
    rows = []
    for (g, q), (exp_sup, exp_ref) in MANUAL.items():
        if g not in res:                  # geometry not diagnosed in this run
            continue
        d = res.get(g, {}).get(q)
        if d is None:
            rows.append((g, q, False, "not diagnosed"))
            continue
        ok = exp_sup <= set(d["supported"]) and exp_ref <= set(d["refuted"])
        rows.append((g, q, ok, f"expected supported {sorted(exp_sup)} refuted "
                               f"{sorted(exp_ref)}; got supported {d['supported']}"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="unit_square,quarter_ring,bushing_groove,cook_membrane")
    ap.add_argument("--out", default=os.path.join(HERE, "diagnose_result.json"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--investigate", action="store_true",
                    help="run the cheap open tests the diagnosis asks for, then re-diagnose")
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--confirm", action="store_true",
                    help="also run affordable confirmatory investigations of SUPPORTED "
                         "hypotheses (costly; same budget)")
    args = ap.parse_args(argv)
    geos = args.geometry.split(",")
    history = []
    if args.investigate:
        res, history = investigate(geos, args.budget_s)
    else:
        res = {g: diagnose_problem(g) for g in geos}
    confirmations = confirm(geos, args.budget_s, res) if args.confirm else []
    for geo in geos:
        print(f"\n=== {FACTS[geo]['name']} ({geo})")
        for q, d in res[geo].items():
            tag = "sufficient" if d["evidence_sufficient"] else "INSUFFICIENT"
            print(f"{q:<24} {d['verdict']}/{d['evidence']:<5} -> {', '.join(d['diagnosis']) or '-'}"
                  f"  [supported {d['supported']}, refuted {d['refuted']}, open {d['open']}; "
                  f"evidence {tag}]")
            if d["next_investigation"]:
                print(f"{'':<26}next: {d['next_investigation']}")
            if args.verbose:
                for t in d["tests"]:
                    print(f"{'':<26}{t['id']} {t['status']:<10} {t['evidence']}")
    rows = consistency(res)
    print(f"\nconsistency with the manual findings: {sum(r[2] for r in rows)} of {len(rows)}")
    for g, q, ok, txt in rows:
        if not ok:
            print(f"  MISMATCH {g} {q}: {txt}")
    json.dump({"diagnoses": res, "investigations": history, "confirmations": confirmations,
               "consistency": [{"geometry": g, "qoi": q, "ok": ok, "detail": t}
                               for g, q, ok, t in rows]},
              open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
