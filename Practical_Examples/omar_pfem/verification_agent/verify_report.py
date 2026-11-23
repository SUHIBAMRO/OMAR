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
    # Section 24: literature agent, billed evaluation (run 1 as designed, run 2 post hoc)
    lit = lambda r: _j("literature_eval_run1.json" if r == 1 else
                       "literature_eval_run2_posthoc.json")
    for r, want in ((1, 7), (2, 8)):
        for p in ("openpinn_plate", "cfm_lamina"):
            add("24", f"literature run {r} {p} checks passed", want, 0,
                lambda r=r, p=p: sum(lit(r)[p][0]["score"].values()))
            add("24", f"literature run {r} {p} checks total", 8, 0,
                lambda r=r, p=p: len(lit(r)[p][0]["score"]))
            add("24", f"literature run {r} {p} quotes found %", 100, 0,
                lambda r=r, p=p: 100 * lit(r)[p][0]["quote_rate"])
    add("24", "literature run 1 plate miss is discretisation", 1, 0,
        lambda: int(not lit(1)["openpinn_plate"][0]["score"]["discretisation"]))
    add("24", "literature run 1 lamina miss is singular features", 1, 0,
        lambda: int(not lit(1)["cfm_lamina"][0]["score"]["singular features as expected"]))
    add("24", "literature cost USD, two runs", 0.67, 0.005,
        lambda: lit(1)["_cost_usd"] + lit(2)["_cost_usd"])
    add("24", "literature run 2 lamina features flagged singular", 0, 0,
        lambda: sum(s["singular"] for s in lit(2)["cfm_lamina"][0]["card"]["singular_features"]))
    add_g_round(add, h1, h2)
    add_i_round(add, h1, h2)
    add_j_round(add, h1, h2, h3)
    add_k_round(add, h1, h2, h3)
    add_l_round(add, h1, h2, h3)
    m = lambda: json.load(open(os.path.join(HERE, "lamina_mms.json")))["rows"]
    add("23", "lamina MMS L2 rate", 3.00, h2, lambda: np.log2(m()[-2]["L2_rel"] / m()[-1]["L2_rel"]))
    add("23", "lamina MMS H1 rate", 2.00, h2, lambda: np.log2(m()[-2]["H1_semi_rel"] / m()[-1]["H1_semi_rel"]))



def add_g_round(add, h1, h2):
    """Section 25 (20 Nov, second round)."""
    from omar_pfem.verification_agent import gates
    # 25.2 literature agent on unseen sections
    un = lambda: _j("literature_eval_unseen.json")
    add("25", "unseen truss checks passed", 6, 0, lambda: sum(un()["csm_truss"][0]["score"].values()))
    add("25", "unseen truss checks total", 6, 0, lambda: len(un()["csm_truss"][0]["score"]))
    add("25", "unseen shell checks passed", 8, 0, lambda: sum(un()["csm_shell"][0]["score"].values()))
    add("25", "unseen shell checks total", 8, 0, lambda: len(un()["csm_shell"][0]["score"]))
    add("25", "unseen shell quotes found", 19, 0,
        lambda: sum(un()["csm_shell"][0]["quotes_found"].values()))
    add("25", "unseen shell quotes total", 20, 0,
        lambda: len(un()["csm_shell"][0]["quotes_found"]))
    add("25", "unseen shell quote rate after NFC fix", 1.0, 0,
        lambda: un()["csm_shell"][0]["quote_rate_after_nfc_fix_posthoc"])
    add("25", "unseen truss reference type analytical", 1, 0,
        lambda: int(un()["csm_truss"][0]["card"]["reference"]["type"] == "analytical"))
    add("25", "unseen cost USD", 0.43, 0.005, lambda: un()["_cost_usd"])
    add("25", "literature agent total cost USD", 1.10, 0.005,
        lambda: un()["_cost_usd"] + _j("literature_eval_run1.json")["_cost_usd"]
        + _j("literature_eval_run2_posthoc.json")["_cost_usd"])
    # 25.3 L-shaped panel
    ls = lambda: _j("lshape_result_7levels.json")
    P = lambda q: ls()["modes"]["posthoc"]["qois"][q]["singular_order"]
    add("25", "L-shape lam", 0.5445, 0.00005, lambda: ls()["lam"])
    add("25", "L-shape rules default hits", 1, 0, lambda: ls()["modes"]["default"]["rule_hits"])
    add("25", "L-shape rules posthoc hits", 3, 0, lambda: ls()["modes"]["posthoc"]["rule_hits"])
    add("25", "L-shape rule predictions", 4, 0, lambda: ls()["modes"]["posthoc"]["rule_predictions"])
    add("25", "L-shape expectations met", 10, 0,
        lambda: sum(ls()["developer_expectations_met"].values()))
    add("25", "L-shape expectations total", 11, 0, lambda: len(ls()["developer_expectations_met"]))
    for q, v in (("strain_energy", 1.113), ("u_top_left", 1.087), ("far_gk_avg_syy", 1.097),
                 ("corner_gk_avg_von_mises", 1.273), ("v_right_top", 0.84),
                 ("peak_von_mises", -0.447)):
        add("25", f"L-shape {q} observed order", v, 0.0005 if q != "v_right_top" else 0.005,
            lambda q=q: P(q)["observed"])
    lad = lambda: sorted((r for r in _j("lshape_ladder.json") if r["g"] == 1.0),
                         key=lambda r: r["n"])
    add("25", "L-shape finest elements", 49152, 0, lambda: lad()[-1]["n_elem"])
    pair = lambda i: float(np.log2(abs((lad()[i]["v_right_top"] - lad()[i + 1]["v_right_top"])
                                       / (lad()[i + 1]["v_right_top"] - lad()[i + 2]["v_right_top"]))))
    for i, v in ((2, 0.53), (3, 0.76), (4, 0.90), (5, 0.97)):
        add("25", f"L-shape v_right_top pairwise order {i}", v, 0.005, lambda i=i: pair(i - 1))
    # 25.4 IGA-type pilot
    ig = lambda: _j("iga_lamina_result.json")
    row = lambda p, m, b: next(r for r in ig()["iga"] if r["p"] == p and r["m"] == m
                               and r["beta"] == b)
    fe = lambda m: next(r for r in ig()["fem"] if r["m"] == m)
    add("25", "IGA p3 m16 dof", 67072, 0, lambda: row(3, 16, 2.8)["n_dof"])
    add("25", "FEM m32 dof", 1050624, 0, lambda: fe(32)["n_dof"])
    add("25", "IGA p3 m16 compliance diff", 1.6e-9, 0.05e-9,
        lambda: abs(row(3, 16, 2.8)["rel_diff_vs_fem32"]["compliance"]))
    add("25", "IGA p3 m16 gk gamma diff", 2.7e-7, 0.05e-7,
        lambda: abs(row(3, 16, 2.8)["rel_diff_vs_fem32"]["hole_gk_gamma12_avg"]))
    add("25", "IGA p3 m16 gk e22 diff", 1.7e-6, 0.05e-6,
        lambda: abs(row(3, 16, 2.8)["rel_diff_vs_fem32"]["hole_gk_e22_avg"]))
    add("25", "FEM m8 dof", 66048, 0, lambda: fe(8)["n_dof"])
    add("25", "FEM m8 compliance err", 7.8e-6, 0.05e-6, lambda: abs(fe(8)["rel_err"]["compliance"]))
    add("25", "FEM m8 gk gamma err", 1.2e-6, 0.05e-6,
        lambda: abs(fe(8)["rel_err"]["hole_gk_gamma12_avg"]))
    add("25", "FEM m8 gk e22 err", 3.5e-5, 0.05e-5, lambda: abs(fe(8)["rel_err"]["hole_gk_e22_avg"]))
    add("25", "IGA p2 m4 dof", 4352, 0, lambda: row(2, 4, 2.8)["n_dof"])
    add("25", "IGA p2 m4 gk e22 err %", 9.2, h1,
        lambda: 100 * abs(row(2, 4, 2.8)["rel_diff_vs_fem32"]["hole_gk_e22_avg"]))
    add("25", "FEM m2 dof", 4224, 0, lambda: fe(2)["n_dof"])
    add("25", "FEM m2 gk e22 err %", 0.85, h2, lambda: 100 * abs(fe(2)["rel_err"]["hole_gk_e22_avg"]))
    # 25.5 goal-oriented estimate
    go = lambda: _j("goal_oriented_result.json")["rows"]
    eff = lambda q: [r["qois"][q]["effectivity"] for r in go() if r["m"] >= 2]
    add("25", "goal eff min (u_tip, gk) m>=2", 0.83, h2,
        lambda: min(min(eff(q)) for q in ("u_tip", "hole_gk_gamma12_avg", "hole_gk_e22_avg")))
    add("25", "goal eff max (u_tip, gk) m>=2", 1.01, h2,
        lambda: max(max(eff(q)) for q in ("u_tip", "hole_gk_gamma12_avg", "hole_gk_e22_avg")))
    add("25", "goal eff min gk m>=2", 0.90, h2,
        lambda: min(min(eff(q)) for q in ("hole_gk_gamma12_avg", "hole_gk_e22_avg")))
    add("25", "goal eff max gk m>=2", 0.97, h2,
        lambda: max(max(eff(q)) for q in ("hole_gk_gamma12_avg", "hole_gk_e22_avg")))
    add("25", "goal compliance eff min", 0.03, h2,
        lambda: min(r["qois"]["compliance"]["effectivity"] for r in go()))
    add("25", "goal compliance eff max", 0.08, h2,
        lambda: max(r["qois"]["compliance"]["effectivity"] for r in go()))
    g1 = lambda m: next(r for r in go() if r["m"] == m)
    add("25", "m1 compliance rel err %", -2.37, h2,
        lambda: -100 * g1(1)["qois"]["compliance"]["true_rel"])
    add("25", "m1 area rel err %", -2.50, h2, lambda: 100 * g1(1)["area_rel_error"])
    add("25", "m4 compliance rel err %", -0.0121, 0.00005,
        lambda: -100 * g1(4)["qois"]["compliance"]["true_rel"])
    add("25", "m4 area rel err %", -0.0122, 0.00005, lambda: 100 * g1(4)["area_rel_error"])
    # 25.6 AMG against the direct solver
    # timed alone (the first pair ran beside other jobs: 240 s / 502 s, kept as files)
    ap = lambda: _j("amg_check_2376_pardiso_alone.json")
    aa = lambda: _j("amg_check_2376_amg_alone.json")
    add("25", "contended Pardiso seconds", 240, 0.5, lambda: _j("amg_check_2376_pardiso.json")["seconds"])
    add("25", "AMG vs Pardiso max rel diff below 1e-12", 1, 0, lambda: int(max(
        abs(aa()["qois"][k] - ap()["qois"][k]) / max(abs(ap()["qois"][k]), 1e-30)
        for k in ap()["qois"] if isinstance(ap()["qois"][k], (int, float))) < 1e-12))
    add("25", "AMG seconds", 318, 0.5, lambda: aa()["seconds"])
    add("25", "Pardiso seconds", 146, 0.5, lambda: ap()["seconds"])
    add("25", "AMG peak GB", 3.5, h1, lambda: aa()["peak_rss_gb"])
    add("25", "Pardiso peak GB", 4.2, h1, lambda: ap()["peak_rss_gb"])
    add("25", "AMG CG its min", 128, 0, lambda: min(aa()["krylov_its"]))
    add("25", "AMG CG its max", 142, 0, lambda: max(aa()["krylov_its"]))
    # 23,520-element uniform level with AMG
    mb = lambda: _j("mixed_b3_result.json")
    add("25", "P1 23520 seconds", 6040, 0.5, lambda: mb()["23520:full:fbar_p1"]["seconds"])
    add("25", "disp 23520 seconds", 4678, 0.5, lambda: mb()["23520:reduced:disp"]["seconds"])
    from omar_pfem.verification_agent.orchestrate import B3Adapter
    fam = lambda k: dict(zip([r["n_elem"] for r in B3Adapter().family], B3Adapter().series(k)[1]))
    add("25", "disp 23520 AMG = stored family (moment) rel below 1e-12", 1, 0, lambda: int(abs(
        mb()["23520:reduced:disp"]["reaction_moment_y"] / fam("reaction_moment_y")[23520] - 1) < 1e-12))
    gap = lambda l, k: 100 * (mb()[f"{l}:full:fbar_p1"][k] - mb()[f"{l}:reduced:disp"][k]) / abs(
        mb()[f"{l}:reduced:disp"][k])
    add("25", "gap sigma_xx 15960 %", 1.6, h1, lambda: gap(15960, "gk2_sigma_xx_avg"))
    add("25", "gap sigma_xx 23520 %", 1.4, h1, lambda: gap(23520, "gk2_sigma_xx_avg"))
    for k in ("gk2_sigma_yy_avg", "gk2_sigma_zz_avg", "gk2_pressure_avg"):
        add("25", f"gap {k} 23520 % (abs)", 1.4, h1, lambda k=k: abs(gap(23520, k)))
    def lim(k):
        hs, v, _ = B3Adapter().series(k)
        f = fam(k)
        off = np.mean([mb()[f"{l}:reduced:disp"][k] / f[l] - 1 for l in (2376, 10200, 15960, 23520)])
        return gates._power_fit(list(hs[-5:]), list(v[-5:]))["a"] * (1 + off), off
    add("25", "truncation offset sigma_xx %", 0.35, h2, lambda: 100 * lim("gk2_sigma_xx_avg")[1])
    for l, k, v in ((15960, "gk2_sigma_xx_avg", -0.55), (23520, "gk2_sigma_xx_avg", -0.19),
                    (23520, "gk2_pressure_avg", -0.08)):
        add("25", f"P1 {l} {k} vs limit %", v, h2,
            lambda l=l, k=k: 100 * (mb()[f"{l}:full:fbar_p1"][k] / lim(k)[0] - 1))
    add("25", "disp 23520 sigma_xx vs limit %", -1.56, h2,
        lambda: 100 * (mb()["23520:reduced:disp"]["gk2_sigma_xx_avg"] / lim("gk2_sigma_xx_avg")[0] - 1))
    lv = [2376, 10200, 15960, 23520]
    p1 = lambda k: [mb()[f"{l}:full:fbar_p1"][k] for l in lv]
    hsl = [1.0 / n ** (1 / 3) for l, n in zip(lv, lv)]
    g4 = lambda k: gates.multilevel_convergence_gate(hsl, p1(k), band_tol=0.01, gci_tol=0.02,
                                                     order_window=(0.5, 5))
    # 3D IGA-type splines on the groove
    ib = lambda: [_j("iga_b3_result.json")[f"L{l}:p2"] for l in (1, 2, 3, 4)]
    ihs = lambda: [1.0 / r["spans"][0] for r in ib()]
    add("25", "IGA3D L4 dof", 319800, 0, lambda: ib()[3]["n_dof"])
    add("25", "IGA3D L4 hours", 2.2, h1, lambda: ib()[3]["seconds"] / 3600)
    for k, vals, tol in (("reaction_moment_y", (35.112, 34.633, 34.555, 34.524), 0.0005),
                         ("strain_energy", (0.8782, 0.8664, 0.8645, 0.8637), 0.00005),
                         ("gk2_sigma_xx_avg", (0.504, 0.525, 0.534, 0.539), 0.0005),
                         ("max_sigma_xx_R0", (14.4, 3.18, 1.68, 1.28), None),
                         ("root_sigma_xx", (-0.800, 0.505, 0.441, 0.4445), 0.0005)):
        for i, v in enumerate(vals):
            t = tol if tol is not None else (0.05 if v > 10 else 0.005)
            add("25", f"IGA3D {k} L{i + 1}", v, t, lambda k=k, i=i: ib()[i][k])
    ig4 = lambda k: gates.multilevel_convergence_gate(ihs(), [r[k] for r in ib()], band_tol=0.01,
                                                      gci_tol=0.02, order_window=(0.5, 5))
    add("25", "IGA3D moment p", 2.1, h1, lambda: ig4("reaction_moment_y")["fit_order"])
    add("25", "IGA3D moment U %", 0.12, h2, lambda: 100 * ig4("reaction_moment_y")["uncertainty_fine"])
    add("25", "IGA3D energy U %", 0.12, h2, lambda: 100 * ig4("strain_energy")["uncertainty_fine"])
    add("25", "IGA3D gk2 xx p", 0.5, h1, lambda: ig4("gk2_sigma_xx_avg")["fit_order"])
    add("25", "IGA3D gk2 xx U %", 8.0, h1, lambda: 100 * ig4("gk2_sigma_xx_avg")["uncertainty_fine"])
    add("25", "IGA3D max R0 U %", 41, 0.5, lambda: 100 * ig4("max_sigma_xx_R0")["uncertainty_fine"])
    add("25", "IGA3D root xx no rate", 1, 0, lambda: int(ig4("root_sigma_xx")["evidence"] != "rate"))
    ilim = lambda k: gates._power_fit(ihs(), [r[k] for r in ib()])["a"]
    from omar_pfem.verification_agent.orchestrate import B3Adapter as _B3
    flim = lambda k: gates._power_fit(list(_B3().series(k)[0][-5:]), list(_B3().series(k)[1][-5:]))["a"]
    add("25", "IGA3D moment limit", 34.493, 0.0005, lambda: ilim("reaction_moment_y"))
    add("25", "FEM moment limit", 34.479, 0.0005, lambda: flim("reaction_moment_y"))
    add("25", "IGA3D energy limit", 0.86300, 0.000005, lambda: ilim("strain_energy"))
    add("25", "FEM energy limit", 0.86268, 0.000005, lambda: flim("strain_energy"))
    add("25", "IGA vs FEM limits moment %", 0.04, h2,
        lambda: 100 * abs(ilim("reaction_moment_y") / flim("reaction_moment_y") - 1))
    add("25", "IGA vs FEM limits energy %", 0.04, h2,
        lambda: 100 * abs(ilim("strain_energy") / flim("strain_energy") - 1))
    add("25", "IGA3D L4 vs FEM limit moment %", 0.13, h2,
        lambda: 100 * (ib()[3]["reaction_moment_y"] / flim("reaction_moment_y") - 1))
    # spline P1 (projected dilatation) on the groove
    sp = lambda: [_j("iga_b3_result.json")[f"L{l}:p2:fbar_p1"] for l in (1, 2, 3, 4)]
    for k, vals, tol in (("reaction_moment_y", (34.848, 34.545, 34.504, 34.487), 0.0005),
                         ("strain_energy", (0.8719, 0.8644, 0.8633, 0.8629), 0.00005),
                         ("gk2_sigma_xx_avg", (0.547, 0.548, 0.550, 0.552), 0.0005),
                         ("gk2_pressure_avg", (-0.483, -0.474, -0.475, -0.476), 0.0005),
                         ("max_sigma_xx_R0", (17.2, 1.71, 1.06, 1.00), None)):
        for i, v in enumerate(vals):
            t = tol if tol is not None else (0.05 if abs(v) > 10 else 0.005)
            add("25", f"splineP1 {k} L{i + 1}", v, t, lambda k=k, i=i: sp()[i][k])
    add("25", "splineP1 L4 hours", 2.9, h1, lambda: sp()[3]["seconds"] / 3600)
    sg4 = lambda k: gates.multilevel_convergence_gate([1.0 / r["spans"][0] for r in sp()],
                                                      [r[k] for r in sp()], band_tol=0.01,
                                                      gci_tol=0.02, order_window=(0.5, 5))
    add("25", "splineP1 moment p", 2.4, h1, lambda: sg4("reaction_moment_y")["fit_order"])
    add("25", "splineP1 moment U %", 0.06, h2, lambda: 100 * sg4("reaction_moment_y")["uncertainty_fine"])
    add("25", "splineP1 energy U %", 0.05, h2, lambda: 100 * sg4("strain_energy")["uncertainty_fine"])
    add("25", "splineP1 L4 moment vs FEM limit %", 0.02, h2,
        lambda: 100 * (sp()[3]["reaction_moment_y"] / flim("reaction_moment_y") - 1))
    pk = lambda: gates.qoi_convergence_gate([1.0 / r["spans"][0] for r in sp()[1:]],
                                            [r["max_sigma_xx_R0"] for r in sp()[1:]],
                                            band_tol=0.01, gci_tol=0.05, order_window=(0.5, 5))
    add("25", "splineP1 peak 3-level order", 5.88, 0.005, lambda: pk()["observed_order"])
    add("25", "splineP1 peak GCI %", 1.52, 0.005, lambda: 100 * pk()["gci_fine"])
    add("25", "splineP1 peak not certified", 1, 0, lambda: int(pk()["verdict"] == "FAIL"))
    add("25", "splineP1 peak last change %", 5.1, h1,
        lambda: 100 * abs(sp()[3]["max_sigma_xx_R0"] / sp()[2]["max_sigma_xx_R0"] - 1))
    ftl = lambda k: gates._power_fit(list(_B3().series(k)[0][-5:]), list(_B3().series(k)[1][-5:]))["a"]
    add("25", "splineP1 trunc sxx vs Q20 limit %", 1.9, h1,
        lambda: 100 * (sp()[3]["gk2trunc_sigma_xx_avg"] / ftl("gk2_sigma_xx_avg") - 1))
    add("25", "splineP1 pressure vs Q20 limit %", 2.0, h1,
        lambda: 100 * (sp()[3]["gk2trunc_pressure_avg"] / ftl("gk2_pressure_avg") - 1))
    # agent against conventional verification
    cv = lambda: _j("conventional_result.json")
    add("25", "conventional quantities", 89, 0, lambda: len(cv()["rows"]))
    add("25", "verifiable quantities", 68, 0, lambda: cv()["conventional"]["verifiable"])
    add("25", "non-verifiable quantities", 14, 0, lambda: cv()["conventional"]["not_verifiable"])
    for k, key, v in (("conventional", "certified", 46), ("conventional", "wrongly_failed", 22),
                      ("agent", "certified", 51), ("agent", "wrongly_failed", 4),
                      ("agent", "inconclusive", 13), ("agent_posthoc", "certified", 62),
                      ("agent_posthoc", "wrongly_failed", 5), ("agent_posthoc", "inconclusive", 1),
                      ("conventional", "missed_failures", 0), ("agent", "missed_failures", 0),
                      ("agent_posthoc", "missed_failures", 0)):
        add("25", f"{k} {key}", v, 0, lambda k=k, key=key: cv()[k][key])
    # edge crack
    ck = lambda: _j("crack_result_7levels.json")
    add("25", "crack rules default", 1, 0, lambda: ck()["modes"]["default"]["rule_hits"])
    add("25", "crack rules posthoc", 4, 0, lambda: ck()["modes"]["posthoc"]["rule_hits"])
    add("25", "crack expectations met", 11, 0, lambda: sum(ck()["developer_expectations_met"].values()))
    add("25", "crack expectations total", 12, 0, lambda: len(ck()["developer_expectations_met"]))
    CP = lambda q: ck()["modes"]["posthoc"]["qois"][q]["singular_order"]["observed"]
    for q, v in (("strain_energy", 0.983), ("u_cod_mouth", 0.983), ("v_top_right", 0.982),
                 ("far_gk_avg_syy", 0.968), ("peak_von_mises", -0.511)):
        add("25", f"crack {q} order", v, 0.0005, lambda q=q: CP(q))
    add("25", "crack finest elements", 65536, 0,
        lambda: max(r["n_elem"] for r in _j("crack_ladder.json")))
    # final-uncertainty rule
    # Sections 25 and 30 are checked against the round-I snapshots (committed outputs of
    # 4c06d90); round J (Section 31) changes the live files and is checked there
    # v1 of the rule (2026-11-22), on the problems it covered (the 3D benchmark came later)
    v1 = lambda st: sum(r["v1"]["status"] == st for r in _j("crosscheck_result_round_I.json")["rows"]
                        if r["problem"] != "thick_cylinder_3d")
    add("25", "crosscheck cross_checked", 21, 0, lambda: v1("cross_checked"))
    add("25", "crosscheck contradicted", 5, 0, lambda: v1("contradicted"))
    add("25", "crosscheck single_family", 8, 0, lambda: v1("single_family"))
    add("25", "crosscheck rate-certified total", 34, 0,
        lambda: sum(v1(s) for s in ("cross_checked", "contradicted", "single_family")))
    # v2 (2026-11-23), all nine problems
    xc = lambda: _j("crosscheck_result_round_I.json")["summary"]
    add("25", "crosscheck v2 cross_checked", 27, 0, lambda: xc()["cross_checked"])
    add("25", "crosscheck v2 contradicted", 0, 0, lambda: xc()["contradicted"])
    add("25", "crosscheck v2 independent unconverged", 9, 0, lambda: xc()["independent_unconverged"])
    add("25", "crosscheck v2 single_family", 2, 0, lambda: xc()["single_family"])
    add("25", "crosscheck v2 rate-certified total", 38, 0, lambda: sum(xc().values()))
    # the B3 Gaussian uncertainty and gap on one basis (2026-11-22): U is relative to the
    # stress scale, the gap relative to the averages; both stated in Sections 1, 20, 25, 26
    gk = lambda: [r for r in _j("crosscheck_result_round_I.json")["rows"]
                  if r["problem"] == "bushing_groove" and r["qoi"] in
                  ("gk2_sigma_xx_avg", "gk2_sigma_yy_avg", "gk2_sigma_zz_avg", "gk2_pressure_avg")]
    u_val = lambda: [100 * r["U_rel"] * r["scale"] / abs(r["limit"]) for r in gk()]
    gap_val = lambda: [100 * abs(r["v1"]["independent"] - r["limit"]) / abs(r["limit"]) for r in gk()]
    add("25", "Gaussian U, % of the averages, min", 0.8, h1, lambda: min(u_val()))
    add("25", "Gaussian U, % of the averages, max", 1.0, h1, lambda: max(u_val()))
    add("25", "Gaussian gap to spline P1, % of the averages, min", 1.9, h1,
        lambda: min(gap_val()))
    add("25", "Gaussian gap to spline P1, % of the averages, max", 2.1, h1,
        lambda: max(gap_val()))
    add("25", "Fig 25.1 sigma_xx U absolute", 0.0054, 0.00005,
        lambda: next(r["U_rel"] * r["scale"] for r in gk() if r["qoi"] == "gk2_sigma_xx_avg"))
    # MMS of the new solvers
    mm = lambda: _j("mms_new_solvers.json")
    def mrow(name, lv):
        return next(r for r in mm() if r["solver"] == name and r["level"] == lv)
    def mord(name, l1, l2, key):
        a, b = mrow(name, l1), mrow(name, l2)
        return float(np.log(a[key] / b[key]) / np.log(a["h"] / b["h"]))
    for name, (l1, l2, l3), vals in (
            ("spline disp", (1, 2, 3), (3.38, 2.05)), ("spline fbar_p1", (1, 2, 3), (3.26, 2.05)),
            ("Q20/P1 (mixed_b3), cartesian field", (2376, 10200, 15960), (4.14, 2.84))):
        add("25", f"MMS {name} L2 order (finest pair)", vals[0], 0.005,
            lambda name=name, l2=l2, l3=l3: mord(name, l2, l3, "L2_rel"))
        add("25", f"MMS {name} H1 order (finest pair)", vals[1], 0.005,
            lambda name=name, l2=l2, l3=l3: mord(name, l2, l3, "H1_rel"))
    add("25", "MMS Q20/P1 cart L2 order 1st pair", 3.59, 0.005,
        lambda: mord("Q20/P1 (mixed_b3), cartesian field", 2376, 10200, "L2_rel"))
    add("25", "MMS Q20/P1 cart H1 order 1st pair", 2.51, 0.005,
        lambda: mord("Q20/P1 (mixed_b3), cartesian field", 2376, 10200, "H1_rel"))
    add("25", "MMS Q20 groove field 2376 L2 %", 4.0, h1,
        lambda: 100 * mrow("Q20/P1 (mixed_b3)", 2376)["L2_rel"])
    add("25", "MMS Q20 groove field 10200 L2 %", 8.0, h1,
        lambda: 100 * mrow("Q20/P1 (mixed_b3)", 10200)["L2_rel"])
    add("25", "P1 moment 4-level rate", 1, 0, lambda: int(g4("reaction_moment_y")["evidence"] == "rate"))
    add("25", "P1 sigma_xx 4-level no rate", 1, 0, lambda: int(g4("gk2_sigma_xx_avg")["evidence"] != "rate"))

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



def add_i_round(add, h1, h2):
    """Section 30 (23 November): audit, cross-check v2, 3D benchmark, three-way, follow-up."""
    mc = lambda: _j("manuscript_claims_result.json")
    add("30", "manuscript claims matching", 28, 0, lambda: sum(r["ok"] is True for r in mc()))
    xr = lambda: _j("crosscheck_result_round_I.json")["rows"]
    old = lambda st: sum(r["status"] == st for r in xr() if r["problem"] != "thick_cylinder_3d")
    add("30", "v2 on the 34: cross-checked", 23, 0, lambda: old("cross_checked"))
    add("30", "v2 on the 34: unconverged", 9, 0, lambda: old("independent_unconverged"))
    add("30", "v2 on the 34: single family", 2, 0, lambda: old("single_family"))
    add("30", "v2 on the 34: contradicted", 0, 0, lambda: old("contradicted"))
    src = lambda q, s: next(x for r in xr() if r["problem"] == "bushing_groove" and r["qoi"] == q
                            for x in r["independent_sources"] if x["source"].startswith(s))
    for q, v in (("gk2_sigma_yy_avg", 1.5), ("gk2_pressure_avg", 3.0)):
        add("30", f"{q} U_ind % of value", v, h1, lambda q=q: next(
            100 * x["U_ind_rel"] * r["scale"] / abs(r["limit"]) for r in xr()
            if r["problem"] == "bushing_groove" and r["qoi"] == q
            for x in r["independent_sources"] if x["status"] == "cross_checked"))
    iga = lambda: _j("iga_b3_result.json")
    def three(key):
        from omar_pfem.verification_agent import gates
        d = iga()
        v = [d[f"L{l}:p2{key}"]["gk2trunc_sigma_xx_avg"] for l in (2, 3, 4)]
        h = [d[f"L{l}:p2{key}"]["n_elem"] ** (-1 / 3) for l in (2, 3, 4)]
        return gates.qoi_convergence_gate(h, v)["observed_order"]
    add("30", "spline disp 3-level order", 0.47, h2, lambda: three(""))
    add("30", "spline P1 3-level order", 0.75, h2, lambda: three(":fbar_p1"))
    zs = lambda: _j("lamina_zone_study.json")
    add("30", "zone area error at compared levels, max (x1e-4)", 4, 0.05, lambda: 1e4 * max(
        [abs(r["zone_area_error_rel"]) for r in zs()["fem"] if r["m"] >= 8]
        + [abs(r["zone_area_error_rel"]) for r in zs()["spline"] if r["m"] == 16]))
    lam_src = lambda: next(x for r in xr() if r["problem"] == "open_hole_lamina"
                           and r["qoi"] == "hole_e22_avg" for x in r["independent_sources"])
    add("30", "lamina spline U %", 9.5, h1, lambda: 100 * lam_src()["assessment"]["uncertainty_fine"])
    cy = lambda: _j("cylinder3d_result.json")
    add("30", "3D rule predictions correct", 3, 0, lambda: cy()["rule_hits"])
    add("30", "3D T2 covered", 4, 0, lambda: cy()["T2_covered"])
    add("30", "3D expectations met", 7, 0, lambda: sum(cy()["developer_expectations_met"].values()))
    add("30", "3D finest unknowns", 57555, 0, lambda: cy()["n_dof"][-1])
    add("30", "3D peak finest true error %", 2.03, h2,
        lambda: 100 * cy()["qois"]["peak_hoop"]["true_error_finest_rel"])
    add("30", "3D peak extrapolated", 85.18, h2, lambda: cy()["peak_hoop_interval_check"]["fit_limit"])
    add("30", "3D peak U", 2.39, h2, lambda: cy()["peak_hoop_interval_check"]["U_abs"])
    add("30", "3D gk order", 4.94, h2, lambda: cy()["qois"]["gk_avg_hoop"]["fit_order"])
    tw = lambda: _j("three_way_result_round_I.json")
    for k, v in (("A", 56), ("B", 56), ("Bp", 72), ("Bstar", 47)):
        add("30", f"three-way {k} certified of 79", v, 0,
            lambda k=k: tw()["scores"][k]["certified_correctly"])
        add("30", f"three-way {k} missed failures", 0, 0,
            lambda k=k: tw()["scores"][k]["missed_failures"])
    add("30", "three-way B1-B3 GCI certified", 22, 0,
        lambda: tw()["scores_on_B1_B3"]["A"]["certified_correctly"])
    add("30", "three-way B1-B3 B+ certified", 38, 0,
        lambda: tw()["scores_on_B1_B3"]["B+"]["certified_correctly"])
    for k, v in (("A", 37), ("Bp", 9), ("C", 19)):
        add("30", f"human proxy B1-B3 {k}", v, 0, lambda k=k: tw()["human"][k]["total_on_B1_B3"])
    for k, v in (("A", 52), ("Bp", 33), ("Bstar", 53)):
        add("30", f"human proxy all {k}", v, 0, lambda k=k: tw()["human"][k]["total"])
    add("30", "B3 unknowns GCI (M)", 1.76, h2, lambda: tw()["per_problem"]["bushing_groove"]["unknowns_A"] / 1e6)
    add("30", "B3 unknowns B (M)", 3.88, h2, lambda: tw()["per_problem"]["bushing_groove"]["unknowns_B"] / 1e6)
    add("30", "B3 unknowns B* (M)", 4.91, h2, lambda: tw()["per_problem"]["bushing_groove"]["unknowns_Bstar"] / 1e6)
    add("30", "diagnosis independent correct", 12, 0, lambda: tw()["diagnosis"]["correct_excluding_C1"])
    add("30", "diagnosis independent of", 12, 0, lambda: tw()["diagnosis"]["of_excluding_C1"])
    add("30", "diagnosis C1 by construction", 19, 0, lambda: tw()["diagnosis"]["C1_cases_by_construction"])
    add("30", "diagnosis before P5 correct", 9, 0, lambda: tw()["diagnosis_before_P5"]["correct"])
    fu = lambda: _j("followup_l_shape_v_right_top.json")
    add("30", "follow-up new pairwise order", 1.01, h2, lambda: fu()["pairwise_orders_after"][-1])
    add("30", "follow-up five-level order", 0.938, 0.0005, lambda: fu()["fit_after"]["v_right_top"])
    add("30", "follow-up seconds", 41, 0.5, lambda: fu()["measured_seconds"])
    add("30", "follow-up estimate seconds", 29, 0.5, lambda: fu()["plan"]["seconds_estimate"])
    add("30", "follow-up estimate GB", 4.7, h1, lambda: fu()["plan"]["gb_estimate"])


def add_j_round(add, h1, h2, h3):
    """Section 31 (23 November, round J): graded failure, block, bug fix, hooks, normalisation."""
    gq = lambda g, q, n: next(r for r in _j("graded_q20_study.json")
                              if r["grading"] == g and r["quadrature"] == q
                              and r.get("nsteps", 10) == n)
    add("31", "g=2 reduced 10 min det F", -1.026, h3, lambda: gq(2.0, "reduced", 10)["min_detF"])
    add("31", "g=2 full 10 min det F", -0.736, h3, lambda: gq(2.0, "full", 10)["min_detF"])
    add("31", "g=2 reduced 40 min det F", 1.004, h3, lambda: gq(2.0, "reduced", 40)["min_detF"])
    add("31", "g=2 full 40 min det F", 1.004, h3, lambda: gq(2.0, "full", 40)["min_detF"])
    add("31", "g=2 reduced 10 gk hoop error %", 27, 0.5,
        lambda: -100 * gq(2.0, "reduced", 10)["errors"]["gk_avg_hoop"])
    add("31", "g=2 full 10 gk hoop error %", 23, 0.5,
        lambda: -100 * gq(2.0, "full", 10)["errors"]["gk_avg_hoop"])
    add("31", "g=2 full 40 global error max", 1.1e-5, 0.05e-5, lambda: max(
        abs(gq(2.0, "full", 40)["errors"][q]) for q in ("reaction_inner_radial", "u_outer_radial")))
    dg = lambda: _j("detf_guard.json")
    add("31", "det F solves recorded", 64, 0, lambda: dg()["solves_with_min_detF_recorded"])
    add("31", "det F inadmissible", 1, 0, lambda: len(dg()["inadmissible"]))
    add("31", "det F inadmissible used by a verdict", 0, 0,
        lambda: sum(r["consumed_by_a_verdict"] for r in dg()["inadmissible"]))
    bk = lambda: _j("block3d_result.json")
    add("31", "block default rule hits", 1, 0, lambda: bk()["modes"]["default"]["rule_hits"])
    add("31", "block posthoc rule hits", 2, 0, lambda: bk()["modes"]["posthoc"]["rule_hits"])
    add("31", "block expectations met", 8, 0, lambda: sum(bk()["developer_expectations_met"].values()))
    add("31", "block peak slope", -0.281, h3, lambda: bk()["peak_slope_finest4"])
    add("31", "block lam", 0.7112, 0.00005, lambda: bk()["lam"])
    P = lambda q: bk()["modes"]["posthoc"]["qois"][q]
    add("31", "block energy fit order", 1.68, h2, lambda: P("strain_energy")["fit_order"])
    add("31", "block mid fit order", 4.41, h2, lambda: P("mid_gk_avg_sxx")["fit_order"])
    add("31", "block root fit order", 2.11, h2, lambda: P("root_gk_avg_von_mises")["fit_order"])
    from omar_pfem.verification_agent.diagnose import pairwise_orders
    bl = lambda: sorted((r for r in _j("block3d_ladder.json") if r["grading"] == 1.0),
                        key=lambda r: r["k"])
    for i, v in enumerate((2.56, 1.84, 1.53, 1.38)):
        add("31", f"block energy pairwise order {i}", v, h2, lambda i=i: pairwise_orders(
            [r["h"] for r in bl()], [r["strain_energy"] for r in bl()])[i])
    fb = lambda: _j("followup_clamped_block_3d_strain_energy.json")["plan"]
    add("31", "block follow-up estimate s", 983, 0.5, lambda: fb()["seconds_estimate"])
    add("31", "block follow-up estimate GB", 14.8, h1, lambda: fb()["gb_estimate"])
    add("31", "block follow-up next k", 12, 0, lambda: fb()["n_next"])
    hk = lambda: _j("followup_hooks_result.json")
    add("31", "MMS L2 rate", 2.98, h2, lambda: hk()["mms:l_shape"]["rate_L2"])
    add("31", "MMS H1 rate", 2.00, h2, lambda: hk()["mms:l_shape"]["rate_H1"])
    add("31", "tol change cylinder", 2.6e-15, 0.05e-15,
        lambda: hk()["tol:thick_cylinder_3d"]["max_rel_change"])
    add("31", "tol change block", 9.6e-12, 0.05e-12,
        lambda: hk()["tol:clamped_block_3d"]["max_rel_change"])
    ns = lambda: _j("normalisation_study.json")
    add("31", "normalisation rate-certified", 54, 0, lambda: ns()["rate_certified"])
    add("31", "normalisation on larger scale", 7, 0,
        lambda: ns()["judged_on_a_scale_larger_than_the_value"])
    add("31", "normalisation max ratio", 7.3, h1, lambda: ns()["max_scale_over_value"])
    add("31", "normalisation failures on own value", 0, 0, lambda: len(ns()["fail_tol_on_own_value"]))
    sc = [r for r in _j("normalisation_study.json").get("rows", []) if r["scale_over_value"] > 1.01]
    add("31", "normalisation min U of value %", 0.79, h2, lambda: 100 * min(r["U_of_value"] for r in sc))
    add("31", "normalisation max U of value %", 1.01, h2, lambda: 100 * max(r["U_of_value"] for r in sc))
    tw = lambda: _j("three_way_result.json")
    add("31", "three-way quantities", 107, 0, lambda: tw()["scores"]["A"]["quantities"])
    add("31", "three-way verifiable", 83, 0, lambda: tw()["scores"]["A"]["verifiable"])
    add("31", "three-way must not certify", 17, 0, lambda: tw()["scores"]["A"]["must_not_certify"])
    for k, v in (("A", 60), ("B", 56), ("Bp", 74), ("Bstar", 60)):
        add("31", f"three-way {k} certified of 83", v, 0,
            lambda k=k: tw()["scores"][k]["certified_correctly"])
        add("31", f"three-way {k} missed failures", 0, 0,
            lambda k=k: tw()["scores"][k]["missed_failures"])
    for k, v in (("A", 54), ("Bp", 38), ("Bstar", 51)):
        add("31", f"human proxy all {k} (ten problems)", v, 0, lambda k=k: tw()["human"][k]["total"])
    add("31", "diagnosis independent correct", 13, 0, lambda: tw()["diagnosis"]["correct_excluding_C1"])
    add("31", "diagnosis independent of", 13, 0, lambda: tw()["diagnosis"]["of_excluding_C1"])
    add("31", "diagnosis C1 by construction", 22, 0, lambda: tw()["diagnosis"]["C1_cases_by_construction"])
    # 31.10 further checks on the block
    k6 = lambda: _j("block3d_strategy_k6.json")["comparison"]["root_gk_avg_von_mises"]
    add("31", "root avg k=6 spread %", 1.82, h2, lambda: 100 * k6()["k6_spread"])
    add("31", "root avg k=4 spread %", 3.48, h2, lambda: 100 * k6()["k4_spread"])
    add("31", "root avg k=6 dev from limit %", 0.95, h2, lambda: 100 * k6()["k6_max_dev_from_limit"])
    add("31", "other quantities k=6 spread max %", 0.14, h2, lambda: 100 * max(
        v["k6_spread"] for q, v in _j("block3d_strategy_k6.json")["comparison"].items()
        if q != "root_gk_avg_von_mises"))
    q8 = lambda: max(r["n_dof"] for r in _j("block3d_q8_ladder.json"))
    add("31", "Q8 finest unknowns", 109395, 0, q8)
    xp = lambda q: next(r for r in _j("crosscheck_result.json")["rows_posthoc_only"]
                        if r["problem"] == "clamped_block_3d" and r["qoi"] == q)
    for q, p_, u, gap in (("w_tip", 1.76, 0.46, 0.022), ("mid_gk_avg_sxx", 1.86, 0.23, 0.024)):
        add("31", f"Q8 {q} order", p_, h2, lambda q=q: xp(q)["independent_sources"][0]["assessment"]["fit_order"])
        add("31", f"Q8 {q} U %", u, h2, lambda q=q: 100 * xp(q)["independent_sources"][0]["U_ind_rel"])
        add("31", f"Q8 {q} gap %", gap, h3, lambda q=q: 100 * xp(q)["independent_sources"][0]["gap_rel"])
        add("31", f"Q8 {q} cross-checked", 1, 0, lambda q=q: int(xp(q)["status"] == "cross_checked"))
    # Section 31 covers the ten problems of round J; l_shape_exact (round K) is excluded
    sp = lambda: {st: sum(r["status"] == st for r in _j("crosscheck_result.json")
                          ["rows_posthoc_only"] if r["problem"] != "l_shape_exact")
                  for st in ("cross_checked", "single_family", "independent_unconverged")}
    add("31", "posthoc-only cross-checked", 13, 0, lambda: sp()["cross_checked"])
    add("31", "posthoc-only single family", 0, 0, lambda: sp()["single_family"])
    add("31", "posthoc-only independent unconverged", 3, 0, lambda: sp()["independent_unconverged"])
    xq = lambda g, q: next(r for r in _j("crosscheck_result.json")["rows_posthoc_only"]
                           if r["problem"] == g and r["qoi"] == q)
    add("31", "P2 cross-checked L-shape", 3, 0, lambda: sum(
        r["status"] == "cross_checked" for r in _j("crosscheck_result.json")["rows_posthoc_only"]
        if r["problem"] == "l_shape"))
    add("31", "P2 cross-checked crack", 4, 0, lambda: sum(
        r["status"] == "cross_checked" for r in _j("crosscheck_result.json")["rows_posthoc_only"]
        if r["problem"] == "edge_crack"))
    add("31", "P2 cross-checked plate", 4, 0, lambda: sum(
        r["status"] == "cross_checked" for r in _j("crosscheck_result.json")["rows_posthoc_only"]
        if r["problem"] == "plate_hole"))
    add("31", "P2 crack v_top_right U %", 0.70, h2,
        lambda: 100 * xq("edge_crack", "v_top_right")["independent_sources"][0]["U_ind_rel"])
    add("31", "P2 L-shape energy order", 1.13, h2,
        lambda: xq("l_shape", "strain_energy")["independent_sources"][0]["assessment"]["fit_order"])
    add("31", "P2 crack energy order", 0.98, h2,
        lambda: xq("edge_crack", "strain_energy")["independent_sources"][0]["assessment"]["fit_order"])
    add("31", "P2 plate hole gk spread %", 0.003, 0.0005, lambda: 100 * float(
        xq("plate_hole", "hole_gk_avg_sigma_xx")["independent_sources"][0]["assessment"]["reason"]
        .split("spread ")[1].split("%")[0]) / 100)


def add_k_round(add, h1, h2, h3):
    """Section 32 (round K): exact singular benchmark, remedy, calibration, B3 Jacobian."""
    import math
    sig2 = lambda v: 0.5 * 10 ** (math.floor(math.log10(abs(v))) - 1)   # half a unit, 2 digits
    ev = lambda: _j("lshape_exact_values.json")
    add("32", "exact lam1", 0.5445, 0.00005, lambda: ev()["lam"][0])
    add("32", "exact lam2", 0.9085, 0.00005, lambda: ev()["lam"][1])
    add("32", "exact energy", 2661.7219, 0.00005, lambda: ev()["strain_energy"])
    add("32", "energy two ways", 1.7e-14, 0.05e-14, lambda: ev()["energy_two_ways_rel"])
    add("32", "corner traction", 1.7e-14, 0.05e-14, lambda: ev()["checks"]["traction_on_free_edges_rel"])
    lr = lambda: _j("lshape_exact_result.json")
    add("32", "default rule hits", 1, 0, lambda: lr()["modes"]["default"]["rule_hits"])
    add("32", "default rule predictions", 4, 0, lambda: lr()["modes"]["default"]["rule_predictions"])
    add("32", "posthoc rule hits", 4, 0, lambda: lr()["modes"]["posthoc"]["rule_hits"])
    add("32", "posthoc T2 covered", 6, 0, lambda: lr()["modes"]["posthoc"]["T2_covered"])
    add("32", "posthoc T2 of", 6, 0, lambda: lr()["modes"]["posthoc"]["T2_of"])
    add("32", "expectations met", 9, 0, lambda: sum(lr()["developer_expectations_met"].values()))
    Q = lambda q: lr()["modes"]["posthoc"]["qois"][q]
    for q, p_, err, U, ratio in (("strain_energy", 1.09, 2.8e-6, 4.6e-4, 0.006),
                                 ("u_mid_left", 1.12, 9.4e-6, 1.7e-4, 0.054),
                                 ("v_mid_right", 1.11, 5.5e-6, 1.1e-4, 0.052),
                                 ("corner_gk_avg_von_mises", 1.37, 4.9e-6, 4.7e-4, 0.010),
                                 ("corner_disc_avg_von_mises", 3.85, 4.3e-4, 6.2e-4, 0.70),
                                 ("far_gk_avg_syy", 1.14, 6.6e-6, 9.7e-5, 0.068)):
        add("32", f"{q} order", p_, h2, lambda q=q: Q(q)["fit_order"])
        add("32", f"{q} true error", err, sig2(err), lambda q=q: Q(q)["true_error_of_limit_rel"])
        add("32", f"{q} U", U, sig2(U), lambda q=q: Q(q)["U_rel"])
        tol = 0.0005 if ratio < 0.1 else 0.005
        add("32", f"{q} ratio", ratio, tol, lambda q=q: Q(q)["effectivity"])
    xq = lambda q: next(r for r in _j("crosscheck_result.json")["rows_posthoc_only"]
                        if r["problem"] == "l_shape_exact" and r["qoi"] == q)
    add("32", "exact benchmark P2 cross-checked", 4, 0, lambda: sum(
        r["status"] == "cross_checked" for r in _j("crosscheck_result.json")["rows_posthoc_only"]
        if r["problem"] == "l_shape_exact"))
    rm = lambda: lr()["remedy"]
    add("32", "remedy g", 3.7, 0, lambda: rm()["g"])
    add("32", "remedy energy order", 3.34, h2, lambda: rm()["qois"]["strain_energy"]["fit_order"])
    add("32", "remedy energy error graded", 1.5e-7, 0.05e-7,
        lambda: rm()["qois"]["strain_energy"]["graded_true_error_finest_rel"])
    add("32", "remedy energy error uniform", 3.7e-4, 0.05e-4,
        lambda: rm()["qois"]["strain_energy"]["uniform_true_error_finest_rel"])
    add("32", "remedy energy gain", 2400, 50, lambda: rm()["qois"]["strain_energy"]
        ["uniform_true_error_finest_rel"] / rm()["qois"]["strain_energy"]["graded_true_error_finest_rel"])
    add("32", "remedy graded dof", 99330, 0, lambda: rm()["qois"]["strain_energy"]["graded_finest_dof"])
    add("32", "remedy uniform dof", 395266, 0, lambda: rm()["qois"]["strain_energy"]["uniform_finest_dof"])
    add("32", "remedy corner gk order", 3.93, h2,
        lambda: rm()["qois"]["corner_gk_avg_von_mises"]["fit_order"])
    add("32", "remedy corner gk error graded", 2.6e-7, 0.05e-7,
        lambda: rm()["qois"]["corner_gk_avg_von_mises"]["graded_true_error_finest_rel"])
    add("32", "remedy corner gk error uniform", 3.4e-4, 0.05e-4,
        lambda: rm()["qois"]["corner_gk_avg_von_mises"]["uniform_true_error_finest_rel"])
    add("32", "remedy disc error graded", 2.9e-4, 0.05e-4,
        lambda: rm()["qois"]["corner_disc_avg_von_mises"]["graded_true_error_finest_rel"])
    add("32", "remedy disc error uniform", 2.7e-4, 0.05e-4,
        lambda: rm()["qois"]["corner_disc_avg_von_mises"]["uniform_true_error_finest_rel"])
    rr = lambda: _j("remedy_result.json")
    add("32", "remedy predictions met", 4, 0, lambda: sum(rr()["predictions_met"].values()))
    for geo, vals in (("l_shape", ((1.11, 5.2e-4), (1.91, 1.2e-4), (2.92, 1.0e-4))),
                      ("edge_crack", ((0.98, 2.4e-3), (2.93, 3.1e-5), (2.77, 4.0e-4)))):
        for v, (p_, U) in zip(("uniform", "strongest", "all"), vals):
            add("32", f"remedy {geo} {v} order", p_, h2,
                lambda geo=geo, v=v: rr()[geo]["strain_energy"][v]["fit_order"])
            add("32", f"remedy {geo} {v} U", U, sig2(U),
                lambda geo=geo, v=v: rr()[geo]["strain_energy"][v]["uncertainty_fine"])
        add("32", f"remedy {geo} P2 cross-checked", 1, 0, lambda geo=geo: int(
            rr()[geo]["strain_energy"]["p2_crosscheck"]["status"] == "cross_checked"))
    add("32", "crack strongest U reduction", 79, 0.5, lambda: rr()["edge_crack"]["strain_energy"]
        ["uniform"]["uncertainty_fine"] / rr()["edge_crack"]["strain_energy"]["strongest"]["uncertainty_fine"])
    from omar_pfem.verification_agent.remedy import POLYGONS, plan_problem
    add("32", "plan g L-shape", 3.7, 0, lambda: plan_problem(POLYGONS["l_shape"], 0.3)["g"])
    add("32", "plan g crack", 4.1, 0, lambda: plan_problem(POLYGONS["edge_crack"], 0.3)["g"])
    add("32", "p over lam", 3.673, h3, lambda: plan_problem(POLYGONS["l_shape"], 0.3)["g_min"])
    cs = lambda: _j("calibration_result.json")["summary"]
    add("32", "calibration framework certified", 12, 0, lambda: cs()["framework_certified"])
    add("32", "calibration framework covered", 12, 0, lambda: cs()["framework_covered"])
    add("32", "calibration framework median ratio", 0.07, h2, lambda: cs()["framework_effectivity_median"])
    add("32", "calibration framework max ratio", 0.70, h2, lambda: cs()["framework_effectivity_max"])
    add("32", "calibration GCI certified", 11, 0, lambda: cs()["gci_certified"])
    add("32", "calibration GCI covered", 9, 0, lambda: cs()["gci_covered"])
    add("32", "calibration GCI max ratio", 17.1, h1, lambda: cs()["gci_effectivity_max"])
    cr = lambda g, q: next(r for r in _j("calibration_result.json")["rows"]
                           if r["problem"] == g and r["qoi"] == q and r["family"] == "uniform")
    add("32", "GCI cylinder disc ratio", 2.6, h1, lambda: cr("thick_cylinder_3d", "disc_avg_hoop")["gci"]["effectivity"])
    add("32", "framework median width factor", 15, 0.5,
        lambda: 1 / cs()["framework_effectivity_median"])
    # 32.6 the second code for the block
    sc2 = lambda q: next(s for r in _j("crosscheck_result.json")["rows_posthoc_only"]
                         if r["problem"] == "clamped_block_3d" and r["qoi"] == q
                         for s in r["independent_sources"] if s["source"].startswith("second code"))
    add("32", "second code w_tip order", 1.90, h2, lambda: sc2("w_tip")["assessment"]["fit_order"])
    add("32", "second code w_tip U %", 0.10, h2, lambda: 100 * sc2("w_tip")["U_ind_rel"])
    add("32", "second code w_tip gap %", 0.021, h3, lambda: 100 * sc2("w_tip")["gap_rel"])
    add("32", "second code w_tip cross-checked", 1, 0, lambda: int(sc2("w_tip")["status"] == "cross_checked"))
    add("32", "second code mid spread %", 0.013, h3, lambda: float(
        sc2("mid_gk_avg_sxx")["assessment"]["reason"].split("spread ")[1].split("%")[0]))
    add("32", "second code mid gap", 1e-6, 0.5e-6, lambda: sc2("mid_gk_avg_sxx")["gap_finest_rel"])
    hx = lambda: _j("block3d_hex27_ladder.json")
    add("32", "second code finest unknowns", 47775, 0, lambda: max(r["n_dof"] for r in hx()))
    add("32", "second code finest seconds", 3730, 0.5, lambda: max(r["seconds"] for r in hx()))
    add("32", "second code levels", 5, 0, lambda: len(hx()))
    bj = lambda rg: next(r for r in _j("b3_mesh_jacobian.json")["rows"] if r["r_grading"] == rg)
    add("32", "B3 1.75 negative points", 480, 0, lambda: bj(1.75)["negative_points"])
    add("32", "B3 1.75 elements", 80, 0, lambda: bj(1.75)["elements_with_negative_points"])
    add("32", "B3 2.0 negative points", 840, 0, lambda: bj(2.0)["negative_points"])
    add("32", "B3 2.0 elements", 120, 0, lambda: bj(2.0)["elements_with_negative_points"])
    add("32", "B3 1.5 negative points", 0, 0, lambda: bj(1.5)["negative_points"])


def add_l_round(add, h1, h2, h3):
    """Section 33 (round L): intensity-aware remedy on the held-out U-shape."""
    import math
    sig2 = lambda v: 0.5 * 10 ** (math.floor(math.log10(abs(v))) - 1)
    ur = lambda: _j("ushape_result.json")
    cor = lambda x, y: next(c for c in ur()["plan"]["corners"] if (c["x"], c["y"]) == (x, y))
    for (x, y), share, g in (((1, 1), 0.69, 3.7), ((0, 0), 0.17, 2.9), ((2, 1), 0.14, 3.7)):
        add("33", f"share {x},{y}", share, h2, lambda x=x, y=y: cor(x, y)["share"])
        add("33", f"g {x},{y}", g, 0, lambda x=x, y=y: cor(x, y)["g"])
    add("33", "share 0,2", 0.003, h3, lambda: cor(0, 2)["share"])
    add("33", "0,2 not graded", 0, 0, lambda: int(cor(0, 2)["graded"]))
    Q = lambda q, v: ur()["qois"][q][v]
    for q, vals in (("strain_energy", ((1.12, 9.5e-4), (2.83, 3.8e-5), (2.91, 7.7e-5), (2.90, 6.5e-5))),
                    ("v_tip", ((1.12, 9.8e-4), (2.69, 4.1e-5), (2.79, 1.7e-5), (2.78, 1.4e-5)))):
        for v, (p_, U) in zip(("uniform", "strongest", "all", "intensity"), vals):
            add("33", f"{q} {v} order", p_, h2, lambda q=q, v=v: Q(q, v)["fit_order"])
            add("33", f"{q} {v} U", U, sig2(U), lambda q=q, v=v: Q(q, v)["uncertainty_fine"])
    add("33", "unknowns n=64", 165378, 0, lambda: Q("strain_energy", "intensity")["finest_dof"])
    add("33", "predictions met", 5, 0, lambda: sum(ur()["predictions_met"].values()))
    add("33", "energy P2 gap %", 0.004, h3,
        lambda: 100 * ur()["qois"]["strain_energy"]["p2_crosscheck"]["gap_rel"])
    add("33", "v_tip P2 cross-checked", 1, 0,
        lambda: int(ur()["qois"]["v_tip"]["p2_crosscheck"]["status"] == "cross_checked"))
    add("33", "energy U gain vs uniform", 15, 0.5, lambda: Q("strain_energy", "uniform")
        ["uncertainty_fine"] / Q("strain_energy", "intensity")["uncertainty_fine"])
    add("33", "v_tip U gain vs uniform", 71, 0.5, lambda: Q("v_tip", "uniform")
        ["uncertainty_fine"] / Q("v_tip", "intensity")["uncertainty_fine"])


if __name__ == "__main__":
    sys.exit(main())
