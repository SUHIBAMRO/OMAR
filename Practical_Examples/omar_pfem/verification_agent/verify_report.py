"""Recompute the numbers of the technical report (2026-11-18, updated 2026-11-19,
advisor_feedback/2026-11-18_Verification_Framework_Technical_Report.pdf)
from the stored data and compare them with the printed values.

Every claim is (section, label, printed value, tolerance, getter). The
tolerance is half a unit of the last printed digit, so a claim passes when
the recomputed value rounds to what the report prints. Nothing is solved
again: the checks, studies and the adaptive replay run on the stored
results (a few minutes). `--mms` also re-runs the 2D MMS study (minutes).

    python -m omar_pfem.verification_agent.verify_report [--mms]

Exit status 0 if every claim matches, 1 otherwise; a JSON record of every
claim is written to verify_report_result.json.
"""
import argparse
import contextlib
import io
import json
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


@lru_cache(maxsize=None)
def rules(geometry):
    """Verdicts of the rule-based plan, exactly as `agent rules` computes them."""
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.orchestrate import (ADAPTERS, run_check, combine,
                                                          convergence_params)
    spec = _default_spec(geometry)
    sd = spec.to_dict()
    plan, analysis = trees.rule_plan(sd), trees.apply_rules(sd)
    adapter = ADAPTERS[geometry]()
    cp = convergence_params(plan)
    v = {}
    for c in plan["checks"]:
        for q in c["qois"]:
            v.setdefault(q, {})[c["check"]] = run_check(adapter, c["check"], q, c["params"],
                                                         element=spec.element,
                                                         conv_params=cp.get(q))
    hits = sum(analysis[q]["predicted"] == combine(v[q])[0]
               for q in spec.qois if analysis[q]["predicted"])
    n_pred = sum(1 for q in spec.qois if analysis[q]["predicted"])
    return {"v": v, "combine": {q: combine(v[q]) for q in spec.qois},
            "hits": hits, "n_pred": n_pred, "adapter": adapter}


def g(geometry, q, check, key):
    return rules(geometry)["v"][q][check][key]


def p3(geo, q):
    return g(geo, q, "qoi_convergence", "observed_order")


def gci(geo, q):
    return 100 * g(geo, q, "qoi_convergence", "gci_fine")


def p5(geo, q):
    return g(geo, q, "qoi_convergence_multilevel", "fit_order")


def loo(geo, q, which):
    o = g(geo, q, "qoi_convergence_multilevel", "leave_one_out_orders")
    return min(o) if which == "min" else max(o)


def U(geo, q):
    return 100 * g(geo, q, "qoi_convergence_multilevel", "uncertainty_fine")


def spread(geo, q):
    return 100 * g(geo, q, "strategy_robustness", "relative_spread")


def anchored(geo, q):
    r = rules(geo)["v"][q]["anchored_robustness"]
    return 100 * max(abs(d) for d in r["deviations"])


def evidence(geo, q):
    return rules(geo)["combine"][q][1]


def overall(geo, q):
    return rules(geo)["combine"][q][0]


# ---- B1 data ---------------------------------------------------------------
@lru_cache(maxsize=None)
def b1_rows():
    return json.load(open(os.path.join(HERE, "b1_ladder_neo_hookean.json")))


def b1_uniform(qoi):
    return [r[qoi] for r in sorted((r for r in b1_rows() if r["grading"] == 0.0),
                                   key=lambda r: r["N"])]


def b1_peak_law(exclude_N161=False):
    """peak = C h_min^p by a log-log least-squares fit. h_min = the smallest
    element edge as stored with the graded solves (`min_elem_size`), 1/(N-1)
    for the uniform ones -- the definition used when the prediction was made."""
    rows = [r for r in b1_rows()
            if not (exclude_N161 and r["N"] == 161 and r["grading"] == 0.0)]
    hmin = np.array([r.get("min_elem_size") or 1.0 / (r["N"] - 1) for r in rows])
    pk = np.array([r["peak_PK1_frob"] for r in rows])
    p, logc = np.polyfit(np.log(hmin), np.log(pk), 1)
    misfit = np.max(np.abs(np.exp(logc) * hmin ** p / pk - 1))
    return p, np.exp(logc), 100 * misfit


def b1_peak(N, a):
    return next(r["peak_PK1_frob"] for r in b1_rows() if r["N"] == N and r["grading"] == a)


def b1_strategy_spread(qoi):
    v = [r[qoi] for r in b1_rows() if r["N"] == 81]
    ref = next(r[qoi] for r in b1_rows() if r["N"] == 81 and r["grading"] == 0.0)
    return 100 * (max(v) - min(v)) / abs(ref)


# ---- B3 data ---------------------------------------------------------------
@lru_cache(maxsize=None)
def b3_family():
    rows = [r for r in json.load(open(os.path.join(HERE, "b3_extended.json")))
            if r["z_beta"] is None and r["rtol"] == 1e-8]
    return sorted(rows, key=lambda r: r["n_elem"])


def fam(i, key):
    return b3_family()[i][key]


def ratio(i):
    f = b3_family()
    return (f[i]["n_elem"] / f[i - 1]["n_elem"]) ** (1 / 3)


def moment_change(i):
    f = b3_family()
    return f[i - 1]["reaction_moment_y"] - f[i]["reaction_moment_y"]


@lru_cache(maxsize=None)
def radial():
    return {r["r_grading"]: r for r in json.load(open(os.path.join(HERE, "b3_radial.json")))}


def radial_moment(rg):
    u = next(r for r in b3_family() if r["n_elem"] == 10200)["reaction_moment_y"]
    return 100 * (radial()[rg]["reaction_moment_y"] - u) / u


@lru_cache(maxsize=None)
def fullint():
    return {r["n_elem"]: r for r in json.load(open(os.path.join(HERE, "b3_fullint.json")))}


def fullint_change(n, key):
    red = next(r for r in b3_family() if r["n_elem"] == n)[key]
    return 100 * (fullint()[n][key] - red) / red


@lru_cache(maxsize=None)
def fullint_fields():
    """Gauss-point pressure std within 2 R0 and the gk2 averages, reduced vs
    full rule, from the stored Gauss-point dumps of the same two levels."""
    from omar_pfem.verification_agent import region_study as R
    out = {}
    for n, res in ((2376, "13_12_19"), (10200, "21_18_31")):
        rec = {}
        for kind, path in (("reduced", f"b3_gauss_near_groove/{res}_zbNone.npz"),
                           ("full", f"b3_gauss_near_groove_fullint/{res}_full.npz")):
            d = dict(np.load(os.path.join(HERE, path)))
            dist = np.linalg.norm(d["pos"] - d["ref"][None, :], axis=1)
            p = -np.trace(d["sigma"], axis1=1, axis2=2) / 3.0
            R0 = float(d["radius"])
            q = R.qois(d, R.weights(d, "gauss", 2 * R0)[0])
            rec[kind] = {"p_std": float(np.std(p[dist < 2 * R0])), "q": q}
        out[n] = rec
    return out


def fullint_pstd(n, kind):
    return fullint_fields()[n][kind]["p_std"]


def fullint_gk_signed_max(n, scale):
    r, f = fullint_fields()[n]["reduced"]["q"], fullint_fields()[n]["full"]["q"]
    keys = ("sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg", "cyl_rr_avg",
            "cyl_tt_avg", "cyl_zz_avg", "cyl_rz_avg", "sigma_xz_avg")
    return 100 * max(abs(f[k] - r[k]) for k in keys) / scale


def fullint_gk_frob_change(n):
    r, f = fullint_fields()[n]["reduced"]["q"], fullint_fields()[n]["full"]["q"]
    return 100 * (f["frob_avg"] - r["frob_avg"]) / r["frob_avg"]


@lru_cache(maxsize=None)
def region():
    from omar_pfem.verification_agent import region_study
    _quiet(region_study.main)
    return json.load(open(os.path.join(HERE, "region_study_result.json")))


def region_stat(name, what):
    d = region()[name]
    if what == "points":
        return d["points"][-1]
    if what == "sign":
        return sum(q["diagnostics"]["sign_changes"] for q in d["qois"].values())
    if what == "rms":
        return 100 * float(np.median([q["diagnostics"]["fit_rms"] for q in d["qois"].values()]))
    if what == "rate5":
        return sum(q["overall_ml"] == "PASS" and q["multilevel"]["evidence"] == "rate"
                   for q in d["qois"].values())
    raise KeyError(what)


@lru_cache(maxsize=None)
def surface():
    from omar_pfem.verification_agent import surface_study
    _quiet(surface_study.main)
    return json.load(open(os.path.join(HERE, "surface_study_result.json")))


def surface_spread(k):
    return 100 * surface()["gates"][k]["3level"]["relative_spread"]


@lru_cache(maxsize=None)
def adaptive_b3():
    from omar_pfem.verification_agent import adaptive
    from omar_pfem.verification_agent.orchestrate import B3Adapter
    a = B3Adapter()
    qois = [q for q in a.qois if q in a.GLOBAL or q.endswith("_avg")]
    n = len(a.family)

    def solve(i):
        f = a.family[i]
        return {"N": i, "h": a.hs[i], "n_dof": 3 * f["n_nodes"], "wall_clock_s": f["elapsed_s"],
                **{q: a.get[q](f) for q in qois}}

    res = _quiet(adaptive.adaptive_refine, solve, qois, N0=(0, 1, 2), budget_s=1e9,
                 max_levels=n, element="Q20", geometry="bushing_groove",
                 next_level=lambda Ns: Ns[-1] + 1 if Ns[-1] + 1 < n else None,
                 n_dof_of=lambda i: 3 * a.family[i]["n_nodes"], scale_of=a._scale)
    total = sum(f["elapsed_s"] for f in a.family)
    cum = np.cumsum([f["elapsed_s"] for f in a.family])
    return res, total, cum, a


def adaptive_level(q):
    res, _, _, a = adaptive_b3()
    return a.family[max(res["decisions"][q]["levels"])]["n_elem"]


def adaptive_share(n_elem):
    _, total, cum, a = adaptive_b3()
    i = [f["n_elem"] for f in a.family].index(n_elem)
    return 100 * cum[i] / total


def adaptive_matches_rules():
    """Adaptive verdict + evidence equal the full-family convergence verdicts
    (both convergence gates combined) for every QoI it handles."""
    from omar_pfem.verification_agent import gates
    res, _, _, _ = adaptive_b3()
    bad = []
    for q, d in res["decisions"].items():
        v = rules("bushing_groove")["v"][q]
        conv = {k: v[k] for k in ("qoi_convergence", "qoi_convergence_multilevel") if k in v}
        o, e, _ = gates.combine_verdicts(conv)
        if (d["verdict"], d["evidence"]) != (o, e):
            bad.append(q)
    return len(res["decisions"]) - len(bad)


@lru_cache(maxsize=None)
def mms():
    return json.load(open(os.path.join(HERE, "mms_q20_result.json")))


def mms_rates(quadrature, nu, key):
    from omar_pfem.verification_agent.mms_q20 import rates
    rows = sorted((r for r in mms() if r["quadrature"] == quadrature and r["nu"] == nu),
                  key=lambda r: r["n"])
    return rates(rows, key)


@lru_cache(maxsize=None)
def b2_rows():
    from omar_pfem.verification_agent.orchestrate import B2Adapter
    return B2Adapter().rows


def b2_spread(q):
    from omar_pfem.verification_agent.orchestrate import B2Adapter
    from omar_pfem.verification_agent import gates
    v, _ = B2Adapter().strategies(q)
    return 100 * gates.strategy_robustness_gate(v, tol=1.0)["relative_spread"]


def singular(*args, **kw):
    from omar_pfem.verification_agent import singularity as S
    return S.smallest_exponent(*args, **kw)


def lam_b1():
    from omar_pfem.verification_agent import singularity as S
    return S.b1_corner_exponent(0.3)


# ---- the claims ------------------------------------------------------------
B1, B3 = "unit_square", "bushing_groove"
GK = "gk2_"


# ---- Sections 12-16 (added after the advisor's reply) -----------------------------------------
def _j(name):
    return json.load(open(os.path.join(HERE, name)))


def indep(prob):
    return _j("independent_solver_result.json")["comparison"][prob]


def indep_q2(prob, q, key):
    return next(r for r in indep(prob)["q2"] if r["qoi"] == q)[key]


def mixed(key, case):
    return _j("mixed_b3_result.json")[case][key]


def cook_rows():
    return sorted((r for r in _j("cook_ladder.json") if r["grading"] == 0.0), key=lambda r: r["N"])


def cook_local_orders(q):
    v = np.array([r[q] for r in cook_rows()], float)
    d = np.diff(v)
    return list(np.log2(np.abs(d[:-1] / d[1:])))


def cook_res(n_levels):
    return _j(f"cook_result_{n_levels}levels.json")


def impact(q, which):
    return _j("stress_fix_impact.json")["rows"][q][which]["finest"]


def hi_counts():
    from omar_pfem.verification_agent import human_intervention as H
    return H.E, H.summarise()


def claims():
    c = []
    add = lambda sec, label, val, tol, fn: c.append((sec, label, val, tol, fn))
    h2, h3, h1 = 0.005, 0.0005, 0.05          # half a unit of 2, 3, 1 decimals
    # Section 4/7: singular exponents
    add("4.6", "L-corner exponent 1", 0.5445, h3 / 1, lambda: singular(1.5 * np.pi, "free", "free", 0.3))
    add("4.6", "L-corner exponent 2", 0.9085, h3, lambda: singular(
        1.5 * np.pi, "free", "free", 0.3, lo=singular(1.5 * np.pi, "free", "free", 0.3) + 1e-3))
    add("7", "B1 corner exponent lambda", 0.7112, 0.00005, lam_b1)
    add("7", "2 lambda", 1.42, h2, lambda: 2 * lam_b1())
    add("7", "peak exponent lambda-1", -0.289, h3, lambda: lam_b1() - 1)
    # Section 7: B1 table
    for q, a3, a5, lo, hi, u in (
            ("strain_energy", 1.42, 1.43, 1.42, 1.43, 0.01),
            ("max_disp_mag", 1.49, 1.54, 1.51, 1.54, 0.01),
            ("u_top_off_center", 1.97, 1.92, 1.92, 1.96, 0.02),
            ("v_top_off_center", 1.45, 1.48, 1.46, 1.48, 0.01),
            ("u_mid_domain", 1.42, 1.54, 1.47, 1.56, 0.01),
            ("v_mid_domain", 1.44, 1.47, 1.45, 1.47, 0.01),
            ("interior_avg_PK1_frob", 0.76, 2.34, 1.49, 2.44, 0.01),
            ("interior_gk_PK1_frob", 1.61, 1.79, 1.69, 1.80, 0.00)):
        add("7", f"B1 {q} 3-level p", a3, h2, lambda q=q: p3(B1, q))
        add("7", f"B1 {q} 5-level p", a5, h2, lambda q=q: p5(B1, q))
        add("7", f"B1 {q} LOO min", lo, h2, lambda q=q: loo(B1, q, "min"))
        add("7", f"B1 {q} LOO max", hi, h2, lambda q=q: loo(B1, q, "max"))
        add("7", f"B1 {q} U_fine %", u, h2, lambda q=q: U(B1, q))
        add("7", f"B1 {q} PASS rate", 1, 0, lambda q=q: float(
            overall(B1, q) == "PASS" and evidence(B1, q) == "rate"))
    add("7", "B1 peak FAIL", 1, 0, lambda: float(overall(B1, "peak_PK1_frob") == "FAIL"))
    add("7", "B1 peak slope, finest 5", -0.275, h3,
        lambda: g(B1, "peak_PK1_frob", "singular_order", "observed"))
    add("7", "B1 strategy spread energy %", 0.010, h3, lambda: b1_strategy_spread("strain_energy"))
    add("7", "B1 strategy spread max disp %", 0.008, h3, lambda: b1_strategy_spread("max_disp_mag"))
    add("7", "B1 strategy spread interior sharp %", 0.004, h3,
        lambda: spread(B1, "interior_avg_PK1_frob"))
    add("7", "B1 strategy spread interior Gaussian %", 0.001, h3,
        lambda: spread(B1, "interior_gk_PK1_frob"))
    add("7", "B1 anchored interior sharp %", 0.018, h3, lambda: anchored(B1, "interior_avg_PK1_frob"))
    add("7", "B1 anchored interior Gaussian %", 0.007, h3, lambda: anchored(B1, "interior_gk_PK1_frob"))
    for i, val in enumerate((9.34, 11.04, 13.34, 16.28, 19.94)):
        add("7", f"B1 peak level {i}", val, h2, lambda i=i: b1_uniform("peak_PK1_frob")[i])
    add("7", "B1 peak law exponent (11 meshes)", -0.281, h3, lambda: b1_peak_law()[0])
    add("7", "B1 peak law max misfit %", 2.3, h1, lambda: b1_peak_law()[2])
    add("7", "B1 N=161 peak predicted from 10 meshes", 19.859, h3, lambda: (
        lambda p, C, _: C * (1 / 160) ** p)(*b1_peak_law(exclude_N161=True)))
    add("7", "B1 N=161 peak", 19.938, h3, lambda: b1_peak(161, 0.0))
    add("7", "B1 N=81 a=0.5 peak", 19.932, h3, lambda: b1_peak(81, 0.5))
    add("7", "B1 rule predictions correct", 7, 0, lambda: rules(B1)["hits"])
    # Section 6: B3 family table
    for i, (n, gp, t, dj) in enumerate(((2376, 2, 35, 0.942), (5760, 4, 131, 0.935),
                                        (10200, 8, 299, 0.929), (15960, 12, 534, 0.925),
                                        (23520, 20, 931, 0.921), (33120, 26, 1366, 0.918),
                                        (45000, 36, 2081, 0.914), (59400, 50, 2739, 0.911))):
        add("6", f"B3 level {i} elements", n, 0, lambda i=i: fam(i, "n_elem"))
        add("6", f"B3 level {i} Gauss points in sphere", gp, 0, lambda i=i: fam(i, "n_region_gauss"))
        add("6", f"B3 level {i} solve time s", t, 0.5, lambda i=i: fam(i, "elapsed_s"))
        add("6", f"B3 level {i} min det F", dj, h3, lambda i=i: fam(i, "min_detF"))
    for i, r in zip(range(1, 8), (1.34, 1.21, 1.16, 1.14, 1.12, 1.11, 1.10)):
        add("6", f"B3 ratio level {i}", r, h2, lambda i=i: ratio(i))
    # Section 8.1
    add("8.1", "moment finest", 34.520, h3, lambda: fam(7, "reaction_moment_y"))
    add("8.1", "moment coarsest", 34.669, h3, lambda: fam(0, "reaction_moment_y"))
    add("8.1", "energy finest", 0.86361, 0.000005, lambda: fam(7, "strain_energy"))
    for q, a3, gc, a5, lo, hi, u in (("reaction_moment_y", 1.35, 0.18, 1.56, 1.33, 1.65, 0.15),
                                     ("strain_energy", 1.41, 0.16, 1.62, 1.39, 1.72, 0.14)):
        add("8.1", f"B3 {q} 3-level p", a3, h2, lambda q=q: p3(B3, q))
        add("8.1", f"B3 {q} GCI %", gc, h2, lambda q=q: gci(B3, q))
        add("8.1", f"B3 {q} 5-level p", a5, h2, lambda q=q: p5(B3, q))
        add("8.1", f"B3 {q} LOO min", lo, h2, lambda q=q: loo(B3, q, "min"))
        add("8.1", f"B3 {q} LOO max", hi, h2, lambda q=q: loo(B3, q, "max"))
        add("8.1", f"B3 {q} U %", u, h2, lambda q=q: U(B3, q))
    for i, d in zip(range(1, 8), (0.063, 0.030, 0.017, 0.015, 0.011, 0.008, 0.007)):
        add("8.1", f"moment change {i}", d, h3, lambda i=i: moment_change(i))
    add("8.1", "W vs M phi/2 at 2,376 el %", 0.06, h2, lambda: 100 * abs(
        fam(0, "strain_energy") / (0.5 * 0.05 * fam(0, "reaction_moment_y")) - 1))
    # Section 8.2: region study
    for name, pts, sg, rms, r5 in (("sharp 1R0", 50, 24, 0.250, 0), ("sharp 2R0", 314, 18, 0.159, 0),
                                   ("sharp 3R0", 910, 13, 0.047, 0), ("sharp 4R0", 1938, 12, 0.025, 0),
                                   ("gauss l=1R0", 148, 1, 0.012, 5), ("gauss l=2R0", 916, 5, 0.004, 7)):
        add("8.2", f"{name} effective points", pts, 0.5, lambda n=name: region_stat(n, "points"))
        add("8.2", f"{name} sign changes", sg, 0, lambda n=name: region_stat(n, "sign"))
        add("8.2", f"{name} median fit residual %", rms, h3, lambda n=name: region_stat(n, "rms"))
        add("8.2", f"{name} 5-level rate count", r5, 0, lambda n=name: region_stat(n, "rate5"))
    # Section 8.3: Gaussian table
    for q, a5, lo, hi, u, sp, an in (("sigma_xx_avg", 2.27, 1.43, 3.21, 0.18, 0.47, 0.52),
                                     ("sigma_yy_avg", 2.48, 1.80, 3.41, 0.12, 0.33, 0.39),
                                     ("sigma_zz_avg", 2.72, 2.11, 3.70, 0.11, 0.31, 0.39),
                                     ("pressure_avg", 2.48, 1.75, 3.43, 0.14, 0.36, 0.43),
                                     ("cyl_rr_avg", 2.27, 1.44, 3.20, 0.18, 0.47, 0.52),
                                     ("cyl_tt_avg", 2.49, 1.79, 3.43, 0.12, 0.33, 0.39)):
        k = GK + q
        add("8.3", f"{k} 5-level p", a5, h2, lambda k=k: p5(B3, k))
        add("8.3", f"{k} LOO min", lo, h2, lambda k=k: loo(B3, k, "min"))
        add("8.3", f"{k} LOO max", hi, h2, lambda k=k: loo(B3, k, "max"))
        add("8.3", f"{k} U %", u, h2, lambda k=k: U(B3, k))
        add("8.3", f"{k} strategy spread %", sp, h2, lambda k=k: spread(B3, k))
        add("8.3", f"{k} anchored %", an, h2, lambda k=k: anchored(B3, k))
        add("8.3", f"{k} PASS rate", 1, 0, lambda k=k: float(
            overall(B3, k) == "PASS" and evidence(B3, k) == "rate"))
    add("8.3", "gk2 cyl_zz equals sigma_zz", 0, 1e-12, lambda: abs(
        p5(B3, GK + "cyl_zz_avg") - p5(B3, GK + "sigma_zz_avg")))
    add("8.3", "gk2 xz spread %", 0.47, h2, lambda: spread(B3, GK + "sigma_xz_avg"))
    add("8.3", "gk2 rz spread %", 0.48, h2, lambda: spread(B3, GK + "cyl_rz_avg"))
    add("8.3", "gk2 xz anchored %", 0.31, h2, lambda: anchored(B3, GK + "sigma_xz_avg"))
    add("8.3", "gk2 rz anchored %", 0.31, h2, lambda: anchored(B3, GK + "cyl_rz_avg"))
    add("8.3", "gk2 von Mises spread %", 0.75, h2, lambda: spread(B3, GK + "von_mises_avg"))
    add("8.3", "gk2 von Mises anchored %", 0.81, h2, lambda: anchored(B3, GK + "von_mises_avg"))
    add("8.3", "gk2 Frobenius spread %", 6.1, h1, lambda: spread(B3, GK + "sigma_frob_avg"))
    add("8.3", "gk2 Frobenius anchored %", 5.6, h1, lambda: anchored(B3, GK + "sigma_frob_avg"))
    add("8.3", "gk2 Frobenius FAIL", 1, 0, lambda: float(overall(B3, GK + "sigma_frob_avg") == "FAIL"))
    # Section 8.4
    add("8.4", "sharp sigma_zz strategy spread %", 1.96, h2, lambda: spread(B3, "sigma_zz_avg"))
    add("8.4", "sharp sigma_zz PASS (narrow)", 1, 0, lambda: float(overall(B3, "sigma_zz_avg") == "PASS"))
    add("8.4", "sharp normal anchored min %", 1.2, h1, lambda: min(anchored(B3, q) for q in (
        "sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg", "cyl_rr_avg", "cyl_tt_avg")))
    add("8.4", "sharp normal anchored max %", 1.6, h1, lambda: max(anchored(B3, q) for q in (
        "sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg", "cyl_rr_avg", "cyl_tt_avg")))
    add("8.4", "B3 rule predictions correct", 12, 0, lambda: rules(B3)["hits"])
    add("8.4", "B3 rule predictions made", 15, 0, lambda: rules(B3)["n_pred"])
    add("8.4", "uniform 10,200 sphere |sigma| avg", 1.030, h3,
        lambda: fam(2, "region_avg_sigma_frob"))
    for rg, dm, fr, dj in ((1.1, -0.07, 0.980, 5.03), (1.2, -0.13, 0.911, 7.81),
                           (1.3, -0.18, 1.061, 12.44)):
        add("8.4", f"radial {rg} moment change %", dm, h2, lambda rg=rg: radial_moment(rg))
        add("8.4", f"radial {rg} sphere |sigma| avg", fr, h3,
            lambda rg=rg: radial()[rg]["region_avg_sigma_frob"])
        add("8.4", f"radial {rg} det J contrast", dj, h2, lambda rg=rg: radial()[rg]["detJ_max_over_min"])
    # Section 8.5: surface
    for k, val in (("root_sxx", 0.59), ("root_szz", 0.36), ("root_pressure", 0.43),
                   ("root_von_mises", 0.24), ("root_sxz", 0.81)):
        add("8.5", f"{k} finest-three spread %", val, h2, lambda k=k: surface_spread(k))
        add("8.5", f"{k} PASS band", 1, 0, lambda k=k: float(
            (surface()["gates"][k]["overall"], surface()["gates"][k]["evidence"]) == ("PASS", "band")))
    add("8.5", "root node at the groove root on every level", 0, 1e-12, lambda: max(
        r["root_node_distance"] for r in surface()["levels"]))
    for k in ("surface_max_abs_sxx", "surface_max_abs_szz", "surface_max_abs_sxz",
              "surface_max_abs_von_mises", "surface_max_abs_pressure"):
        add("8.5", f"{k} FAIL", 1, 0, lambda k=k: float(surface()["gates"][k]["overall"] == "FAIL"))
    # Section 8.6: integration rule
    add("8.6", "full vs reduced moment 2,376 %", 0.22, h2, lambda: fullint_change(2376, "reaction_moment_y"))
    add("8.6", "full vs reduced energy 2,376 %", 0.21, h2, lambda: fullint_change(2376, "strain_energy"))
    add("8.6", "full vs reduced moment 10,200 %", 0.085, h3, lambda: fullint_change(10200, "reaction_moment_y"))
    add("8.6", "full vs reduced energy 10,200 %", 0.079, h3, lambda: fullint_change(10200, "strain_energy"))
    add("8.6", "pressure std reduced 2,376", 4.05, h2, lambda: fullint_pstd(2376, "reduced"))
    add("8.6", "pressure std full 2,376", 3.98, h2, lambda: fullint_pstd(2376, "full"))
    add("8.6", "pressure std reduced 10,200", 0.43, h2, lambda: fullint_pstd(10200, "reduced"))
    add("8.6", "pressure std full 10,200", 0.61, h2, lambda: fullint_pstd(10200, "full"))
    add("8.6", "gk2 signed averages full-reduced 2,376 (max, % of scale) <=", 1.3, h1,
        lambda: fullint_gk_signed_max(2376, rules(B3)["adapter"].kernel_scale))
    add("8.6", "gk2 signed averages full-reduced 10,200 (max, % of scale) <=", 0.29, h2,
        lambda: fullint_gk_signed_max(10200, rules(B3)["adapter"].kernel_scale))
    add("8.6", "gk2 Frobenius full vs reduced 10,200 %", 9.2, h1, lambda: fullint_gk_frob_change(10200))
    # Section 9: adaptive
    add("9", "B3 family solve time total s", 8116, 0.5, lambda: adaptive_b3()[1])
    for q, n in (("reaction_moment_y", 15960), ("strain_energy", 15960),
                 (GK + "sigma_yy_avg", 23520), (GK + "sigma_zz_avg", 15960),
                 (GK + "pressure_avg", 23520), (GK + "cyl_tt_avg", 23520), (GK + "cyl_zz_avg", 15960),
                 ("sigma_frob_avg", 23520), ("von_mises_avg", 23520), ("sigma_xz_avg", 23520),
                 ("cyl_rz_avg", 23520), (GK + "sigma_xx_avg", 45000), (GK + "cyl_rr_avg", 45000)):
        add("9", f"adaptive {q} decided at", n, 0, lambda q=q: adaptive_level(q))
    for n, s in ((15960, 12), (23520, 24), (45000, 66)):
        add("9", f"adaptive share up to {n} %", s, 0.5, lambda n=n: adaptive_share(n))
    add("9", "adaptive verdicts equal full-family convergence verdicts (of 28)", 28, 0,
        adaptive_matches_rules)
    # Section 4.7: Q20 MMS
    for (quad, nu), l2s, h1s in ((("reduced", 0.3), (3.21, 3.06, 3.02), (2.11, 2.03, 2.01)),
                                 (("full", 0.3), (3.28, 3.10, 3.03), (2.11, 2.02, 2.01)),
                                 (("reduced", 0.45), (3.32, 3.32, 3.14), (2.11, 2.13, 2.06)),
                                 (("full", 0.45), (3.31, 3.34, 3.16), (2.13, 2.10, 2.04))):
        for i, (l2, h1) in enumerate(zip(l2s, h1s)):
            add("4.7", f"Q20 MMS {quad} nu={nu} L2 rate {i}", l2, h2,
                lambda q=quad, n=nu, i=i: mms_rates(q, n, "L2_rel")[i])
            add("4.7", f"Q20 MMS {quad} nu={nu} H1 rate {i}", h1, h2,
                lambda q=quad, n=nu, i=i: mms_rates(q, n, "H1_rel")[i])
    add("4.7", "Q20 MMS largest dof count", 56355, 0, lambda: max(r["n_dof"] for r in mms()))
    # Section 11: held-out B2
    B2 = "quarter_ring"
    for q, a3, a5, lo, hi in (("max_disp_mag", 2.00, 1.98, 1.98, 2.00),
                              ("u_inner", 2.00, 1.98, 1.97, 1.99), ("v_inner", 2.00, 1.99, 1.99, 2.00),
                              ("u_mid", 2.00, 1.99, 1.98, 2.00), ("v_mid", 2.00, 1.99, 1.99, 2.00),
                              ("strain_energy", 2.00, 2.00, 2.00, 2.00),
                              ("interior_gk_PK1_frob", 2.00, 2.02, 2.00, 2.03),
                              ("peak_PK1_frob", 1.01, 1.03, 0.99, 1.04)):
        add("11", f"B2 {q} 3-level p", a3, h2, lambda q=q: p3(B2, q))
        add("11", f"B2 {q} 5-level p", a5, h2, lambda q=q: p5(B2, q))
        add("11", f"B2 {q} LOO min", lo, h2, lambda q=q: loo(B2, q, "min"))
        add("11", f"B2 {q} LOO max", hi, h2, lambda q=q: loo(B2, q, "max"))
        add("11", f"B2 {q} PASS rate", 1, 0, lambda q=q: float(
            overall(B2, q) == "PASS" and evidence(B2, q) == "rate"))
        if q != "peak_PK1_frob":
            add("11", f"B2 {q} U_fine below 0.01 %", 0, 0.01, lambda q=q: U(B2, q))
    add("11", "B2 peak U_fine %", 0.20, h2, lambda: U(B2, "peak_PK1_frob"))
    add("11", "B2 sharp average PASS band", 1, 0, lambda: float(
        (overall(B2, "interior_avg_PK1_frob"), evidence(B2, "interior_avg_PK1_frob")) == ("PASS", "band")))
    add("11", "B2 sharp average finest-three spread %", 0.04, h2, lambda: 100 * g(
        B2, "interior_avg_PK1_frob", "qoi_convergence", "relative_spread"))
    add("11", "B2 rule predictions correct", 6, 0, lambda: rules(B2)["hits"])
    add("11", "B2 rule predictions made", 6, 0, lambda: rules(B2)["n_pred"])
    for i, (n, ne) in enumerate(((7, 36), (13, 144), (25, 576), (49, 2304), (97, 9216), (193, 36864))):
        add("11", f"B2 level {i} N", n, 0, lambda i=i: b2_rows()[i]["N"])
        add("11", f"B2 level {i} elements", ne, 0, lambda i=i: b2_rows()[i]["n_elem"])
        add("11", f"B2 level {i} query points are nodes", 0, 1e-12, lambda i=i: max(
            b2_rows()[i]["query_inner_node_distance"], b2_rows()[i]["query_mid_node_distance"]))
    add("11", "B2 N=193 solve time s", 1268, 0.5, lambda: b2_rows()[5]["wall_clock_s"])
    add("11", "B2 sharp disc points coarsest", 6, 0, lambda: b2_rows()[0]["interior_region_gauss_points"])
    add("11", "B2 sharp disc points finest", 4426, 0, lambda: b2_rows()[5]["interior_region_gauss_points"])
    add("11", "B2 strategy spread displacements/energy max %", 0.005, h3, lambda: max(
        b2_spread(q) for q in ("max_disp_mag", "u_inner", "v_inner", "u_mid", "v_mid", "strain_energy")))
    add("11", "B2 strategy spread Gaussian %", 0.007, h3, lambda: b2_spread("interior_gk_PK1_frob"))
    add("11", "B2 strategy spread sharp %", 0.19, h2, lambda: b2_spread("interior_avg_PK1_frob"))
    add("11", "B2 strategy spread peak %", 0.95, h2, lambda: b2_spread("peak_PK1_frob"))
    # Section 12: diagnosis
    add("12", "consistency with hand-established causes (of 16)", 16, 0, lambda: sum(
        r["ok"] for r in _j("diagnose_result.json")["consistency"]))
    add("12", "hand-established causes", 16, 0, lambda: len(_j("diagnose_result.json")["consistency"]))
    add("12", "B2 Newton-tolerance re-solve s", 88, 0.5, lambda: next(
        r["wall_clock_s"] for r in _j("b2_ladder_neo_hookean.json") if r["newton_tol"] == 1e-10))
    add("12", "B3 h/rho at the finest level", 1.13, h2, lambda: _j("diagnose_b3_h_root.json")["59400"][
        "h_max"] / 0.0228)
    mh = lambda k: 100 * _j("model_hierarchy_result.json")["b1_linear_vs_neo_hookean"][k]["rel_diff"]
    big = ("max_disp_mag", "v_top_off_center", "u_top_off_center", "v_mid_domain", "strain_energy")
    add("12", "B1 linear vs neo-Hookean min of five %", 0.43, h2, lambda: min(mh(k) for k in big))
    add("12", "B1 linear vs neo-Hookean max of five %", 0.65, h2, lambda: max(mh(k) for k in big))
    add("12", "B1 linear vs neo-Hookean sixth %", 0.02, h2, lambda: mh("u_mid_domain"))
    add("12", "B1 max |grad u| (linear)", 0.011, h3, lambda: _j("model_hierarchy_result.json")[
        "b1_linear_vs_neo_hookean"]["max_grad_u_linear"])
    add("12", "F-bar pressure-scatter factor 2,376", 2.9, h1, lambda: mixed(
        "pressure_std_2R0", "2376:reduced:disp") / mixed("pressure_std_2R0", "2376:reduced:fbar"))
    # Section 13.1: second FE code
    add("13", "B1 same-discretisation max rel diff (x1e-12)", 2.1, h1, lambda: 1e12 * max(
        r["rel_diff"] for r in indep("b1")["same"]))
    add("13", "B1 same-discretisation values", 40, 0, lambda: len(indep("b1")["same"]))
    add("13", "B2 same-discretisation max rel diff (x1e-13)", 2.2, h1, lambda: 1e13 * max(
        r["rel_diff"] for r in indep("b2")["same"]))
    add("13", "B2 same-discretisation values", 54, 0, lambda: len(indep("b2")["same"]))
    add("13", "B1 Q2 within uncertainty (of 7)", 7, 0, lambda: sum(
        r.get("within_uncertainty", False) for r in indep("b1")["q2"]))
    add("13", "B2 Q2 finest within uncertainty (of 8)", 7, 0, lambda: sum(
        r.get("within_uncertainty", False) for r in indep("b2")["q2"]))
    add("13", "B2 Q2 extrapolated within uncertainty (of 8)", 8, 0, lambda: sum(
        r.get("extrapolated_within_uncertainty", False) for r in indep("b2")["q2"]))
    add("13", "B2 peak finest Q2 vs limit %", 0.23, h2, lambda: 100 * indep_q2(
        "b2", "peak_PK1_frob", "q2_finest_vs_limit_rel"))
    add("13", "B2 peak agent U %", 0.196, h3, lambda: 100 * indep_q2(
        "b2", "peak_PK1_frob", "agent_uncertainty_rel"))
    add("13", "B2 peak extrapolated Q2 vs limit %", 0.020, h3, lambda: 100 * indep_q2(
        "b2", "peak_PK1_frob", "q2_extrapolated_vs_limit_rel"))
    for i, val in enumerate((0.118, 0.060, 0.031)):
        add("13", f"B2 Q2 peak increment {i}", val, h3, lambda i=i: float(np.diff(
            indep_q2("b2", "peak_PK1_frob", "q2_values"))[1 + i]))
    add("13", "B1 Q2 peak coarsest", 11.1, h1, lambda: indep_q2("b1", "peak_PK1_frob", "q2_values")[0])
    add("13", "B1 Q2 peak finest", 24.9, h1, lambda: indep_q2("b1", "peak_PK1_frob", "q2_values")[-1])
    # Section 13.2: F-bar table
    for key, fmt, vals in (("reaction_moment_y", h3, ((34.669, 34.577, 34.560), (34.206, 34.332, 34.354))),
                           ("pressure_std_2R0", h2, ((4.05, 0.43, 0.35), (1.38, 0.53, 0.63))),
                           ("gk2_sigma_zz_avg", h3, ((0.407, 0.421, 0.424), (0.488, 0.469, 0.466))),
                           ("gk2_pressure_avg", h3, ((-0.443, -0.454, -0.457), (-0.490, -0.488, -0.488))),
                           ("gk2_frob_avg", h2, ((6.83, 3.26, 3.15), (5.20, 4.04, 3.83)))):
        for form, row in zip(("disp", "fbar"), vals):
            for lv, val in zip((2376, 10200, 15960), row):
                add("13", f"{key} {form} {lv}", val, fmt,
                    lambda key=key, form=form, lv=lv: mixed(key, f"{lv}:reduced:{form}"))
    for lv, gap in ((2376, 1.33), (10200, 0.71), (15960, 0.60)):   # relative to displacement
        add("13", f"moment gap disp-F-bar {lv} %", gap, h2, lambda lv=lv: 100 * (
            1 - mixed("reaction_moment_y", f"{lv}:reduced:fbar") / mixed(
                "reaction_moment_y", f"{lv}:reduced:disp")))
    # Section 13.3: stress-correction impact (finest level)
    for q, old, new in (("sigma_xx_avg", 0.4386, 0.4525), ("sigma_zz_avg", 0.4284, 0.4151),
                        ("gk2_sigma_xx_avg", 0.5397, 0.5361), ("gk2_sigma_zz_avg", 0.4382, 0.4318)):
        add("13", f"impact {q} before", old, 0.00005, lambda q=q: impact(q, "old"))
        add("13", f"impact {q} after", new, 0.00005, lambda q=q: impact(q, "new"))
    add("13", "impact sigma_xz before", -0.292, h3, lambda: impact("sigma_xz_avg", "old"))
    add("13", "impact sigma_xz after", -0.270, h3, lambda: impact("sigma_xz_avg", "new"))
    add("13", "impact verdict changes", 2, 0, lambda: len(_j("stress_fix_impact.json")["verdict_changes"]))
    add("13", "impact quantities", 41, 0, lambda: len(_j("stress_fix_impact.json")["rows"]))
    # Section 14: Cook
    add("14", "Cook rule predictions correct (7 levels)", 4, 0, lambda: cook_res(7)["rule_hits"])
    add("14", "Cook rule predictions made", 5, 0, lambda: cook_res(7)["rule_predictions"])
    add("14", "Cook expectations met (of 7)", 6, 0, lambda: sum(
        cook_res(7)["developer_expectations_met"].values()))
    for q, val in (("u_tip", 1.32), ("v_tip", 1.43), ("max_disp_mag", 1.38), ("strain_energy", 1.75),
                   ("interior_gk_PK1_frob", 1.94), ("peak_PK1_frob", -0.328)):
        add("14", f"Cook {q} order/slope (7 levels)", val, h2,
            lambda q=q: cook_res(7)["qois"][q]["singular_order"]["observed"])
    add("14", "Cook energy order (8 levels)", 1.71, h2,
        lambda: cook_res(8)["qois"]["strain_energy"]["singular_order"]["observed"])
    add("14", "Cook lambda", 0.6442, 0.00005, lambda: _j("cook_preregistration.json")["corner_exponent_lam"])
    lo = lambda: cook_local_orders("strain_energy")
    for i, val in zip((2, 3, 4, 5), (1.76, 1.74, 1.67, 1.59)):
        add("14", f"Cook energy local order {i}", val, h2, lambda i=i: lo()[i])
    add("14", "Cook u_tip local order finest", 1.11, h2, lambda: cook_local_orders("u_tip")[-1])
    add("14", "Cook peak local slope finest", -0.361, h3, lambda: -np.log2(
        cook_rows()[-1]["peak_PK1_frob"] / cook_rows()[-2]["peak_PK1_frob"]))
    add("14", "Cook N=512 solve s", 1207, 0.5, lambda: cook_rows()[-1]["wall_clock_s"])
    add("14", "Cook N=512 elements", 262144, 0, lambda: cook_rows()[-1]["n_elem"])
    # Section 15: local refinement
    def lr(case_idx, quad, form, key):
        from omar_pfem.verification_agent.b3_local_refine import SEQUENCE
        c = list(SEQUENCE[case_idx])
        return next(r[key] for r in _j("b3_local_refine.json") if r["case"] == c
                    and r.get("quadrature", "reduced") == quad and r.get("formulation", "disp") == form)
    for i, hr in ((0, 2.50), (1, 0.77), (2, 0.44), (3, 0.25), (4, 0.15)):
        add("15", f"local refinement h/rho mesh {i}", hr, h2, lambda i=i: lr(i, "reduced", "disp", "h_over_rho"))
    for (i, quad, form), vals in {
            (1, "reduced", "disp"): (34.531, 0.82, 1.09, 0.538),
            (3, "reduced", "disp"): (34.482, 10.1, 10.5, 0.541),
            (1, "full", "disp"): (34.664, 0.833, 1.034, 0.497),
            (3, "full", "disp"): (34.780, 0.839, 1.110, 0.471),
            (1, "full", "fbar"): (34.294, 1.044, 1.141, 0.565),
            (3, "full", "fbar"): (34.212, 1.014, 1.277, 0.569)}.items():
        for key, val in zip(("reaction_moment_y", "max_sigma_xx_R0", "max_sigma_zz_R0",
                             "gk2_sigma_xx_avg"), vals):
            tol = {2: h2, 1: h1, 3: h3}[min(3, len(str(val).split(".")[1]))]
            add("15", f"local {quad}/{form} mesh {i} {key}", val, tol,
                lambda i=i, quad=quad, form=form, key=key: lr(i, quad, form, key))
    add("15", "reduced max at h/rho 0.15", 29.0, h1, lambda: lr(4, "reduced", "disp", "max_sigma_xx_R0"))
    add("15", "full max at h/rho 0.15", 0.856, h3, lambda: lr(4, "full", "disp", "max_sigma_xx_R0"))
    add("15", "probe reduced max", 10.08, h2, lambda: _j("b3_local_refine_probe_reduced.json")[0]["sxx"])
    add("15", "probe full max", 0.84, h2, lambda: _j("b3_local_refine_probe_full.json")[0]["sxx"])
    # Section 16: human intervention
    add("16", "interventions", 59, 0, lambda: len(hi_counts()[0]))
    add("16", "found by developer", 36, 0, lambda: hi_counts()[1][1]["developer"])
    add("16", "found by LLM critic", 15, 0, lambda: hi_counts()[1][1]["LLM critic"])
    add("16", "found by automated checks", 2, 0, lambda: hi_counts()[1][1]["automated check"])
    add("16", "Omar decisions", 4, 0, lambda: hi_counts()[1][1]["Omar"])
    add("16", "now fully automated", 6, 0, lambda: hi_counts()[1][2]["yes"])
    add("16", "framework errors fixed", 22, 0, lambda: hi_counts()[1][0]["framework bug fixed"])
    # ---------------- Sections 18-22 (after the OpenPINN paper) ----------------
    add_f_round(add, h1, h2, h3)
    return c


def _plate6(q, check, key):
    """A gate of the plate with a hole on the six pre-registered levels (m <= 64)."""
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.orchestrate import (PlateHoleAdapter, run_check,
                                                          convergence_params)
    ad = PlateHoleAdapter()
    ad.rows = [r for r in ad.rows if r["m"] <= 64]
    ad.hs = [r["h"] for r in ad.rows]
    sd = _default_spec("plate_hole").to_dict()
    plan = trees.rule_plan(sd)
    c = next(c for c in plan["checks"] if c["check"] == check and q in c["qois"])
    return run_check(ad, check, q, c["params"], element="Q9",
                     conv_params=convergence_params(plan).get(q))[key]


def add_f_round(add, h1, h2, h3):
    lad = lambda: {(r["m"], r["g"]): r for r in _j("plate_hole_ladder.json")}
    # Section 18
    r6 = lambda: _j("plate_hole_result_6levels.json")
    add("18", "plate rule hits (6 levels)", 1, 0, lambda: r6()["rule_hits"])
    add("18", "plate rule predictions", 6, 0, lambda: r6()["rule_predictions"])
    add("18", "plate expectations met (6 levels)", 1, 0,
        lambda: sum(r6()["developer_expectations_met"].values()))
    add("18", "plate expectations (count)", 9, 0, lambda: len(r6()["developer_expectations_met"]))
    add("18", "plate lambda", 0.7583, 0.00005, lambda: _j("plate_hole_preregistration.json")[
        "corner_exponent_lam"])
    add("18", "plate 2 lambda", 1.52, h2, lambda: 2 * _j("plate_hole_preregistration.json")[
        "corner_exponent_lam"])
    add("18", "plate dof m=2", 832, 0, lambda: lad()[(2, 1.0)]["n_dof"])
    add("18", "plate dof m=64", 788480, 0, lambda: lad()[(64, 1.0)]["n_dof"])
    for q, a, b in (("u_right_mid", 1.58, 1.92), ("v_top_right", 1.37, 2.98),
                    ("max_disp_mag", 1.58, 1.92), ("strain_energy", 1.55, 1.76),
                    ("hole_peak_sigma_xx", 0.96, 0.86), ("peak_von_mises", 0.99, 0.90)):
        add("18", f"plate {q} 3-level order (6 levels)", a, h2,
            lambda q=q: _plate6(q, "qoi_convergence", "observed_order"))
        add("18", f"plate {q} 5-level order (6 levels)", b, h2,
            lambda q=q: _plate6(q, "qoi_convergence_multilevel", "fit_order"))
    add("18", "plate von Mises strategy spread %", 2.04, h2, lambda: 100 * float(
        _plate6("peak_von_mises", "strategy_robustness", "reason").split("strategies ")[1]
        .split("%")[0]) / 100)
    add("18", "corner vM m=2 MPa", 1.44, h2, lambda: lad()[(2, 1.0)][
        "posthoc_corner_peak_von_mises"] / 1e6)
    add("18", "corner vM m=64 MPa", 3.01, h2, lambda: lad()[(64, 1.0)][
        "posthoc_corner_peak_von_mises"] / 1e6)
    add("18", "corner local slope m=32-64", -0.240, h3, lambda: np.log2(
        lad()[(32, 1.0)]["posthoc_corner_peak_von_mises"]
        / lad()[(64, 1.0)]["posthoc_corner_peak_von_mises"]))
    m128 = lambda: _j("plate_hole_m128_prediction.json")["outcome_after_solving_m128"]
    add("18", "m128 predictions met", 5, 0, lambda: sum(
        v["met"] for v in m128().values() if isinstance(v, dict)))
    add("18", "m128 corner vM MPa", 3.562, h3, lambda: lad()[(128, 1.0)][
        "posthoc_corner_peak_von_mises"] / 1e6)
    add("18", "m128 local slope", -0.241, h3, lambda: m128()["local corner slope -0.242 within 0.03"][
        "value"])
    add("18", "m128 hole peak MPa", 3.382, h3, lambda: lad()[(128, 1.0)]["hole_peak_sigma_xx"] / 1e6)
    add("18", "m128 dof", 3149824, 0, lambda: lad()[(128, 1.0)]["n_dof"])
    add("18", "m128 max change (5e-6)", 5, 0.5, lambda: m128()[
        "displacements and energy change < 1e-4"]["max_change"] * 1e6)
    ph = lambda q: _j("plate_hole_posthoc_rules.json")["verdicts"][q]["singular_order"]["observed"]
    add("18", "post hoc energy order 7 levels", 1.59, h2, lambda: ph("strain_energy"))
    add("18", "post hoc Gaussian false-alarm order", 0.38, h2, lambda: ph("hole_gk_avg_sigma_xx"))
    add("18", "corner slope finest five (diagnosis)", -0.236, h3, lambda: float(np.polyfit(
        np.log([1 / m for m in (8, 16, 32, 64, 128)]),
        np.log([lad()[(m, 1.0)]["posthoc_corner_peak_von_mises"] for m in (8, 16, 32, 64, 128)]),
        1)[0]))
    mms = lambda: _j("plate_hole_mms.json")["rates"]
    add("18", "plate MMS L2", 3.00, h2, lambda: mms()["L2"])
    add("18", "plate MMS H1", 2.00, h2, lambda: mms()["H1_semi"])
    re = lambda: _j("plate_hole_reference_error.json")
    q1 = lambda m: next(r for r in re()["q1"] if r["m"] == m)
    q2 = lambda m: next(r for r in re()["q2"] if r["m"] == m)
    add("18", "Q4 m=40 energy-norm %", 1.07, h2, lambda: 100 * q1(40)["energy_norm_rel_error"])
    add("18", "Q4 m=40 u error %", 0.015, h3, lambda: 100 * q1(40)["u_right_mid_rel_error"])
    add("18", "Q4 m=40 dof", 77440, 0, lambda: q1(40)["n_dof"])
    add("18", "Q9 m=64 energy-norm %", 0.21, h2, lambda: 100 * q2(64)["energy_norm_rel_error"])
    add("18", "Q9 m=64 u error %", 0.0005, 0.00005, lambda: 100 * q2(64)["u_right_mid_rel_error"])
    add("18", "Q9 m=128 energy-norm %", 0.13, h2, lambda: 100 * q2(128)["energy_norm_rel_error"])
    add("18", "Q9 m=128 u error %", 0.0002, 0.00005, lambda: 100 * q2(128)["u_right_mid_rel_error"])
    # Section 19: Q20/P1
    mx = lambda k: _j("mixed_b3_result.json")[k]
    for n, (md, mp, mf), (xd, xp, xf), (zd, zp, zf), dif in (
            (2376, (34.669, 34.618, 34.206), (0.518, 0.531, 0.532), (0.407, 0.428, 0.488), 2.4),
            (10200, (34.577, 34.542, 34.332), (0.527, 0.538, 0.549), (0.421, 0.429, 0.469), 2.1),
            (15960, (34.560, 34.529, 34.354), (0.531, 0.539, 0.552), (0.424, 0.431, 0.466), 1.6)):
        for form, (m_, x_, z_) in (("reduced:disp", (md, xd, zd)), ("full:fbar_p1", (mp, xp, zp)),
                                   ("reduced:fbar", (mf, xf, zf))):
            k = f"{n}:{form}"
            add("19", f"{k} moment", m_, h3, lambda k=k: mx(k)["reaction_moment_y"])
            add("19", f"{k} gk2 xx", x_, h3, lambda k=k: mx(k)["gk2_sigma_xx_avg"])
            add("19", f"{k} gk2 zz", z_, h3, lambda k=k: mx(k)["gk2_sigma_zz_avg"])
        add("19", f"{n} P1 vs disp xx %", dif, h1, lambda n=n: 100 * (
            mx(f"{n}:full:fbar_p1")["gk2_sigma_xx_avg"] / mx(f"{n}:reduced:disp")["gk2_sigma_xx_avg"] - 1))
    lr = lambda i, quad, form: next(r for r in _j("b3_local_refine.json") if r["case"] == list(
        __import__("omar_pfem.verification_agent.b3_local_refine", fromlist=["SEQUENCE"]).SEQUENCE[i])
        and r.get("quadrature", "reduced") == quad and r.get("formulation", "disp") == form)
    for i, h, (m_, x_, p_) in ((1, 0.77, (34.57, 0.526, 1.73)), (3, 0.25, (34.63, 0.510, 1.67)),
                              (4, 0.15, (34.63, 0.510, 2.20))):
        add("19", f"P1 local h/rho {h} moment", m_, h2, lambda i=i: lr(i, "full", "fbar_p1")[
            "reaction_moment_y"])
        add("19", f"P1 local h/rho {h} gk2 xx", x_, h3, lambda i=i: lr(i, "full", "fbar_p1")[
            "gk2_sigma_xx_avg"])
        add("19", f"P1 local h/rho {h} max xx", p_, h2, lambda i=i: lr(i, "full", "fbar_p1")[
            "max_sigma_xx_R0"])
    add("19", "P1 local peak rise %", 31, 0.5, lambda: 100 * (
        lr(4, "full", "fbar_p1")["max_sigma_xx_R0"] / lr(3, "full", "fbar_p1")["max_sigma_xx_R0"] - 1))
    add("19", "graded spread low (full disp)", 0.471, h3, lambda: lr(3, "full", "disp")[
        "gk2_sigma_xx_avg"])
    add("19", "graded spread high (F-bar)", 0.569, h3, lambda: lr(3, "full", "fbar")[
        "gk2_sigma_xx_avg"])
    # Section 20: Gaussian surrogate errors (cluster job 3271674, transcribed output)
    f4 = lambda: _j("f4_gaussian_result_job3271674.json")["mean_self_relative_error_percent"]
    add("20", "best variant gk2 xx %", 173, 0.5, lambda: f4()["local-integration-refinement seed 3"][
        "gk2_xx"])
    add("20", "PI baseline gk2 xx %", 417, 0.5, lambda: f4()["PI baseline"]["gk2_xx"])
    add("20", "equilibrium 2 gk2 xx %", 4290, 0.5, lambda: f4()["equilibrium attempt 2 (w=0.12)"][
        "gk2_xx"])
    add("20", "checkpoints where Gaussian < sharp (xx)", 10, 0, lambda: sum(
        v["gk2_xx"] < v["sharp_xx"] for v in f4().values()))
    add("20", "best normal gk2 min %", 146, 0.5, lambda: min(
        v for k, v in _j("f4_gaussian_result_job3271674.json")["best_gk2_mean_self_relative_percent"]
        .items() if k in ("sigma_xx", "sigma_yy", "sigma_zz", "pressure", "cyl_rr", "cyl_tt", "cyl_zz")))
    # Section 21: LLM against rules
    ab = lambda p: _j("ablation_result_live.json")["problems"][p]
    for p, cost in (("b1", 2.06), ("b2", 2.57), ("b3", 3.03)):
        add("21", f"ablation {p} cost USD", cost, h2, lambda p=p: sum(
            r["llm_cost_usd"] for r in ab(p)["runs"]))
    add("21", "ablation total USD", 7.66, h2, lambda: sum(
        r["llm_cost_usd"] for p in ("b1", "b2", "b3") for r in ab(p)["runs"]))
    add("21", "B3 rules missed", 5, 0, lambda: len(ab("b3")["rule_only"]["missed"]))
    add("21", "B3 rules false alarms", 3, 0, lambda: len(ab("b3")["rule_only"]["false_alarm"]))
    add("21", "B3 LLM missed (max)", 1, 0, lambda: max(r["llm_missed_n"] for r in ab("b3")["runs"]))
    add("21", "B3 LLM false alarms max", 10, 0, lambda: max(
        r["llm_false_alarm_n"] for r in ab("b3")["runs"]))
    add("21", "spec recovered (of 9)", 9, 0, lambda: sum(all(r["spec_match"].values())
                                                          for p in ("b1", "b2", "b3")
                                                          for r in ab(p)["runs"]))
    # Section 22: manuscript numbers
    f1 = lambda: _j(os.path.join("manuscript_recompute", "f1_values.json"))["manuscript_vs_new"]
    k = "avg range over eight resolutions 2,376-47,040 ('0.38-0.44')"
    add("22", "avg range low", 0.42, h2, lambda: f1()[k]["new"][0])
    add("22", "avg range high", 0.47, h2, lambda: f1()[k]["new"][1])
    k2 = "max first six levels ('4.13 to 0.706')"
    add("22", "max first level", 4.16, h2, lambda: f1()[k2]["new"][0])
    add("22", "max sixth level", 0.729, h3, lambda: f1()[k2]["new"][1])
    k3 = "rise at the seventh level ('26%', '0.891 at 45,000')"
    add("22", "rise %", 25, 0.5, lambda: 100 * f1()[k3]["new"][0])
    add("22", "max at 45,000", 0.910, h3, lambda: f1()[k3]["new"][1])
    k4 = "z_beta 0-3 at 10,200 ('0.615-0.826')"
    add("22", "z_beta low", 0.621, h3, lambda: f1()[k4]["new"][0])
    add("22", "z_beta high", 0.836, h3, lambda: f1()[k4]["new"][1])
    # Section 23: CFM open-hole lamina
    lm = lambda: _j("lamina_result_6levels.json")
    lr_ = lambda: {(r["m"], r["g"]): r for r in _j("lamina_ladder.json")}
    add("23", "lamina compliance mm/N x1e5", 6.981, h3, lambda: 1e5 * lr_()[(32, 1.5)]["compliance"])
    add("23", "lamina dof m=32", 1050624, 0, lambda: lr_()[(32, 1.5)]["n_dof"])
    add("23", "lamina QoIs PASS", 8, 0, lambda: sum(r["verdict"] == "PASS" for r in lm()["qois"].values()))
    add("23", "lamina rule predictions", 1, 0, lambda: lm()["rule_predictions"])
    add("23", "lamina expectations met", 5, 0, lambda: sum(lm()["developer_expectations_met"].values()))
    add("23", "lamina u_tip 5-level order", 2.36, h2, lambda: lm()["qois"]["u_tip"]["order_5level"])
    add("23", "lamina u_tip 3-level order", 3.40, h2, lambda: lm()["qois"]["u_tip"]["order_3level"])
    cf = lambda q: lm()["cfm_comparison"][q]
    num = lambda q: max(cf(q)["error_at_m4_vs_m32_pct"], cf(q)["strategy_spread_m8_pct"])
    for q, e, mg in (("compliance", 0.012, 132), ("hole_gamma12_avg", 0.10, 29),
                     ("far_gamma12_avg", 0.05, 50), ("hole_e22_avg", 0.37, 143)):
        add("23", f"lamina {q} error %", e, 0.0005 if e < 0.05 else h2, lambda q=q: num(q))
        add("23", f"lamina {q} margin", mg, 0.5, lambda q=q: cf(q)["margin_half_width_over_numerical"])
    # Section 24: active learning
    al = lambda: _j("al_lamina_result.json")
    st = lambda n: al()["strategies"][n]
    med = lambda n: float(np.median(st(n)["final_rmse_rel"])) * 100
    add("24", "random final RMS %", 0.97, h2, lambda: med("random"))
    add("24", "al_fine final RMS %", 0.26, h2, lambda: med("al_fine"))
    add("24", "mf_naive final RMS %", 1.12, h2, lambda: med("mf_naive"))
    add("24", "mf_verified final RMS %", 0.013, 0.0005, lambda: med("mf_verified"))
    rch = lambda n, t: [x for x in st(n)["cost_to_reach"][t] if x is not None]
    add("24", "mf_verified to 1% min s", 1.6, h1, lambda: min(rch("mf_verified", "0.01")))
    add("24", "mf_verified to 1% max s", 2.1, h1, lambda: max(rch("mf_verified", "0.01")))
    add("24", "mf_verified to 0.1% min s", 19, 0.5, lambda: min(rch("mf_verified", "0.001")))
    add("24", "mf_verified to 0.1% max s", 20, 0.5, lambda: max(rch("mf_verified", "0.001")))
    add("24", "al_fine to 1% min s", 25, 0.5, lambda: min(rch("al_fine", "0.01")))
    add("24", "al_fine to 1% max s", 31, 0.5, lambda: max(rch("al_fine", "0.01")))
    add("24", "random to 1% min s", 33, 0.5, lambda: min(rch("random", "0.01")))
    add("24", "random to 1% max s", 41, 0.5, lambda: max(rch("random", "0.01")))
    add("24", "random seeds never reaching 1%", 1, 0, lambda: sum(
        x is None for x in st("random")["cost_to_reach"]["0.01"]))
    add("24", "speed-up to 1% vs al_fine", 16, 0.5, lambda: _j("al_lamina_scored.json")[
        "speedup_to_1pct_vs_al_fine"])
    add("24", "expectations met", 3, 0, lambda: _j("al_lamina_scored.json")["met"])
    m = lambda: json.load(open(os.path.join(HERE, "lamina_mms.json")))["rows"]
    add("23", "lamina MMS L2 rate", 3.00, h2, lambda: np.log2(m()[-2]["L2_rel"] / m()[-1]["L2_rel"]))
    add("23", "lamina MMS H1 rate", 2.00, h2, lambda: np.log2(m()[-2]["H1_semi_rel"] / m()[-1]["H1_semi_rel"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mms", action="store_true", help="also re-run the 2D MMS study")
    args = ap.parse_args(argv)
    results, n_bad = [], 0
    for sec, label, val, tol, fn in claims():
        try:
            got = float(fn())
            ok = abs(got - val) <= tol + 1e-12
        except Exception as e:                      # a missing key is a failed claim
            got, ok = f"error: {e!r}", False
        n_bad += not ok
        results.append({"section": sec, "claim": label, "printed": val, "recomputed": got,
                        "tolerance": tol, "ok": ok})
        if not ok:
            print(f"MISMATCH  [{sec}] {label}: printed {val}, recomputed {got}")
    if args.mms:
        from omar_pfem.verification_agent import agent
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            agent.main(["mms", "--orders", "Q4,Q9", "--Ns", "5,9,17", "--cpu"])
        print(out.getvalue())
    json.dump(results, open(os.path.join(HERE, "verify_report_result.json"), "w"), indent=1,
              default=str)
    print(f"{len(results) - n_bad} of {len(results)} claims match the stored data"
          + ("" if n_bad == 0 else f"; {n_bad} MISMATCH"))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
