"""Pre-registration of the singular benchmark with an exact solution (lshape_exact.py)
and of the graded-mesh remedy (remedy.py), written and committed BEFORE any loaded
solve (only a zero-load smoke test was run: trivial solution). Same protocol as the
earlier held-out problems: spec, the unchanged rules' predictions, the developer's
expectations, the scoring tests, the SHA-256 of the verdict-deciding files.

The exact values (lshape_exact_values.json) are for scoring only; the agent never sees
them.

    python -m omar_pfem.verification_agent.lshape_exact_preregister            (before)
    python -m omar_pfem.verification_agent.lshape_exact_preregister --compare  (after)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lshape_exact_preregistration.json")
RESULT = os.path.join(HERE, "lshape_exact_result.json")
FROZEN = ("gates.py", "trees.py", "orchestrate.py", "extra_adapters.py", "lshape_exact.py",
          "remedy.py")
GRADED_LEVELS = (2, 4, 8, 16, 32, 64)


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def main():
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import _default_spec
    from omar_pfem.verification_agent.extra_adapters import LShapeExactAdapter
    from omar_pfem.verification_agent.lshape_exact import CACHE, EXACT, exponents
    from omar_pfem.verification_agent.remedy import plan_grading
    if os.path.exists(CACHE):
        raise SystemExit("lshape_exact_ladder.json exists: not a pre-registration any more")
    sd = _default_spec("l_shape_exact").to_dict()
    rules = trees.apply_rules(sd)
    lam1, lam2 = exponents()
    plan = plan_grading(lam1, 2)
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "first SINGULAR problem with an exact solution: does the certified "
                   "uncertainty (post hoc singular-order extension) contain the true error? "
                   "And does the agent's graded-mesh remedy recover the optimal order?",
        "spec": sd, "adapter_description": LShapeExactAdapter.description,
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "exact_values_sha256": hashlib.sha256(open(EXACT, "rb").read()).hexdigest(),
        "exponents": [lam1, lam2],
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"]}
                             for q, r in rules.items()},
        "scoring_tests": {
            "T1 rule predictions": "default rules (and with VERIF_POSTHOC_SINGULAR=1); hits "
                                   "counted as before",
            "T2 coverage": "for every quantity the agent passes with RATE evidence (in each "
                           "mode): |exact - certified limit| <= U_fine x scale, scale = |finest "
                           "value|; no tolerance added",
            "T3 remedy": "graded family g from remedy.plan_grading, levels n = 2..64, the "
                         "same five-level convergence check (qoi_convergence_multilevel with "
                         "the class tolerances of trees.class_params, order window 0.5-5): fitted order, U, T2 coverage, and "
                         "true error against the uniform family at comparable unknowns",
        },
        "developer_expectations": {
            "default rules": "every singular_order check INCONCLUSIVE (degree-1 theory only), "
                             "so the global and displacement quantities are INCONCLUSIVE; the "
                             "peak FAILs (rule hits 1 of 4, as on the L-shape)",
            "posthoc strain_energy": "PASS with rate, order within 0.15 of 2 lam = 1.089 "
                                     "(MEDIUM: the second mode adds terms h^1.45 and h^1.82)",
            "posthoc displacements": "both PASS with rate (MEDIUM; on the L-shape one "
                                     "displacement fell just short)",
            "posthoc region averages": "far Gaussian average PASS (rate); the two corner "
                                       "averages PASS (LOW)",
            "peak": "FAIL in both modes (HIGH)",
            "T2": "the exact value lies inside the certified interval for every rate-certified "
                  "quantity (MEDIUM; this is the test the extension has not had)",
            "remedy plan": f"g = {plan['g']} (p / lam = {plan['g_min']:.3f})",
            "remedy energy order": "five-level fitted order between 3.0 and 4.6 (theory 4)",
            "remedy accuracy": "true energy error at graded n = 64 below 1/100 of the uniform "
                               "error at n = 128 (fewer unknowns)",
            "remedy coverage": "T2 holds for the graded energy and both displacements",
        },
        "remedy_plan": plan,
        "graded_levels": list(GRADED_LEVELS),
        "rules_changed_for_this_problem": False,
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps({k: pre[k] for k in ("rule_predictions", "remedy_plan")}, indent=1))


def _limit(gv):
    ml, c3 = gv.get("qoi_convergence_multilevel", {}), gv.get("qoi_convergence", {})
    if ml.get("evidence") == "rate":
        return ml["fit_limit"], ml["uncertainty_fine"], ml.get("fit_order")
    if c3.get("evidence") == "rate":
        return c3["richardson_extrapolate"], c3["gci_fine"], c3.get("observed_order")
    return None


def compare():
    import numpy as np
    from omar_pfem.verification_agent import gates, trees
    from omar_pfem.verification_agent.verify_report import rules
    from omar_pfem.verification_agent.lshape_exact import CACHE, EXACT
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    # AMENDMENT (2026-11-23, after the uniform ladder and before any scoring):
    # lshape_exact.py interpolates the displacement where the query point is not a node
    # (graded meshes); uniform meshes keep the node branch. Allowed only if every stored
    # uniform level is reproduced to round-off (relative 1e-12; the direct solver is not
    # bitwise repeatable across calls); any other change refuses scoring.
    # remedy.py gained the corner analysis after this scoring (round K, be9e643); the scoring
    # uses only plan_grading (through the stored plan), which must be unchanged since 338d7f1
    from omar_pfem.verification_agent.prereg_util import functions_unchanged
    if "remedy.py" in changed and functions_unchanged("remedy.py", "338d7f1", ("plan_grading",)):
        changed.remove("remedy.py")
        print("note: remedy.py changed after pre-registration; plan_grading is unchanged")
    # round M (post hoc switch): orchestrate.py gained blocks run only with VERIF_POSTHOC_DETF=1;
    # allowed iff the module equals its b850330 version with those blocks removed
    from omar_pfem.verification_agent.prereg_util import unchanged_except_switch
    if "orchestrate.py" in changed and unchanged_except_switch("orchestrate.py", "b850330",
                                                               "VERIF_POSTHOC_DETF"):
        changed.remove("orchestrate.py")
        print("note: orchestrate.py changed only by the off-by-default VERIF_POSTHOC_DETF switch")
    if set(changed) - {"lshape_exact.py"}:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    if changed:
        from omar_pfem.verification_agent import lshape_exact as LX
        st = {r["n"]: r for r in json.load(open(CACHE)) if r["g"] == 1.0}
        for n in (2, 4, 8, 16, 32, 64):
            r = LX.solve_level(n)
            assert all(abs(r[k] - st[n][k]) <= 1e-12 * max(abs(st[n][k]), 1e-300)
                       for k in r if isinstance(r[k], float) and k != "wall_clock_s"), n
        print("note: lshape_exact.py amended (interpolated query points); uniform levels "
              "reproduced to round-off (1e-12)")
    if hashlib.sha256(open(EXACT, "rb").read()).hexdigest() != pre["exact_values_sha256"]:
        raise SystemExit("exact values changed since pre-registration")
    ex = json.load(open(EXACT))
    out = {}
    for mode in ("default", "posthoc"):
        if mode == "posthoc":
            os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        R = rules.__wrapped__("l_shape_exact")
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        ad = R["adapter"]
        rows, hits, npred = {}, 0, 0
        for q, pr in pre["rule_predictions"].items():
            ov, ev = R["combine"][q][:2]
            gv = R["v"][q]
            sg = gv.get("singular_order", {})
            hs, vals, _ = ad.series(q)
            row = {"verdict": ov, "evidence": ev,
                   "singular_order": {k: sg.get(k) for k in ("verdict", "observed", "label")},
                   "finest": vals[-1], "exact": ex[q]}
            if np.isfinite(ex[q]):
                row["true_error_finest_rel"] = abs(vals[-1] - ex[q]) / abs(ex[q])
            if pr["predicted"] is not None:
                npred += 1
                row["rule_hit"] = pr["predicted"] == ov
                hits += row["rule_hit"]
            if ov == "PASS" and ev == "rate":
                lim, U, p = _limit(gv)
                sc = abs(vals[-1])
                row.update(limit=lim, U_rel=U, fit_order=p,
                           true_error_of_limit_rel=abs(ex[q] - lim) / sc,
                           T2_covered=abs(ex[q] - lim) <= U * sc,
                           effectivity=abs(ex[q] - lim) / (U * sc) if U else None)
            rows[q] = row
        cov = [r["T2_covered"] for r in rows.values() if "T2_covered" in r]
        out[mode] = {"rule_hits": hits, "rule_predictions": npred, "qois": rows,
                     "T2_covered": sum(cov), "T2_of": len(cov)}
    # T3: the remedy (graded family), same five-level check, no singular-order check
    allrows = json.load(open(CACHE))
    g = pre["remedy_plan"]["g"]
    gr = sorted((r for r in allrows if r["g"] == g), key=lambda r: r["n"])
    un = sorted((r for r in allrows if r["g"] == 1.0), key=lambda r: r["n"])
    rem = {"g": g, "levels": [r["n"] for r in gr], "n_dof": [r["n_dof"] for r in gr], "qois": {}}
    for q in ("strain_energy", "u_mid_left", "v_mid_right", "corner_gk_avg_von_mises",
              "corner_disc_avg_von_mises", "far_gk_avg_syy"):
        hs, vals = [r["h"] for r in gr], [r[q] for r in gr]
        cp = trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))
        gm = gates.multilevel_convergence_gate(hs, vals, band_tol=cp["band_tol"],
                                               gci_tol=cp["gci_tol"], order_window=(0.5, 5))
        row = {"evidence": gm.get("evidence"), "verdict": gm.get("verdict"),
               "fit_order": gm.get("fit_order"), "U_rel": gm.get("uncertainty_fine"),
               "graded_true_error_finest_rel": abs(vals[-1] - ex[q]) / abs(ex[q]),
               "uniform_true_error_finest_rel": abs(un[-1][q] - ex[q]) / abs(ex[q]),
               "graded_finest_dof": gr[-1]["n_dof"], "uniform_finest_dof": un[-1]["n_dof"]}
        if gm.get("evidence") == "rate":
            sc = abs(vals[-1])
            row["T2_covered"] = abs(ex[q] - gm["fit_limit"]) <= gm["uncertainty_fine"] * sc
        rem["qois"][q] = row
    E = rem["qois"]["strain_energy"]
    P = out["posthoc"]["qois"]
    rate = lambda q: P[q]["verdict"] == "PASS" and P[q]["evidence"] == "rate"
    lam2x = 2 * pre["exponents"][0]
    exp = {
        "default: rule hits 1 of 4": out["default"]["rule_hits"] == 1,
        "posthoc energy PASS rate, order within 0.15 of 2 lam": rate("strain_energy")
        and abs((P["strain_energy"].get("fit_order") or 0) - lam2x) <= 0.15,
        "posthoc both displacements PASS rate": rate("u_mid_left") and rate("v_mid_right"),
        "posthoc far Gaussian average PASS": P["far_gk_avg_syy"]["verdict"] == "PASS",
        "posthoc corner averages PASS": P["corner_gk_avg_von_mises"]["verdict"] == "PASS"
        and P["corner_disc_avg_von_mises"]["verdict"] == "PASS",
        "peak FAIL in both modes": out["default"]["qois"]["peak_von_mises"]["verdict"] == "FAIL"
        and P["peak_von_mises"]["verdict"] == "FAIL",
        "T2 holds for every rate-certified quantity (posthoc)":
            out["posthoc"]["T2_covered"] == out["posthoc"]["T2_of"],
        "remedy energy order 3.0-4.6": 3.0 <= (E["fit_order"] or 0) <= 4.6,
        "remedy energy error < 1/100 of uniform": E["graded_true_error_finest_rel"]
        < E["uniform_true_error_finest_rel"] / 100,
        "remedy T2 for energy and both displacements": all(
            rem["qois"][q].get("T2_covered") for q in ("strain_energy", "u_mid_left",
                                                         "v_mid_right")),
    }
    res = {"modes": out, "remedy": rem, "developer_expectations_met": exp}
    json.dump(res, open(RESULT, "w"), indent=1, default=float)
    for mode, o in out.items():
        print(f"{mode}: rule hits {o['rule_hits']}/{o['rule_predictions']}, "
              f"T2 {o['T2_covered']}/{o['T2_of']}")
        for q, r in o["qois"].items():
            print(f"   {q:<27}{r['verdict']}/{r['evidence']:<5} "
                  + (f"p={r['fit_order']:.3f} err(limit)={100 * r['true_error_of_limit_rel']:.4f}% "
                     f"U={100 * r['U_rel']:.4f}% covered={r['T2_covered']}"
                     if "T2_covered" in r else
                     f"err(finest)={100 * r.get('true_error_finest_rel', float('nan')):.4f}%"))
    print(f"remedy g = {g}:")
    for q, r in rem["qois"].items():
        print(f"   {q:<27}{r['verdict']}/{r['evidence']} p={r['fit_order']} U={r['U_rel']} "
              f"err graded {100 * r['graded_true_error_finest_rel']:.2e}% ({r['graded_finest_dof']}"
              f" dof) vs uniform {100 * r['uniform_true_error_finest_rel']:.2e}% "
              f"({r['uniform_finest_dof']} dof) T2={r.get('T2_covered')}")
    print(f"developer expectations {sum(exp.values())}/{len(exp)}")
    for k, v in exp.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    compare() if "--compare" in sys.argv else main()
