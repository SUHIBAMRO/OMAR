"""Pre-registration of the intensity-aware remedy on a HELD-OUT problem (the U-shaped
panel, ushape.py), written and committed BEFORE any loaded solve of it (round L).

The planner (remedy_intensity.plan_intensity) is fixed here: its rule constants, the
corner analysis it uses, and the variants it is compared with. The plan itself is
computed by code from the uniform family once that is solved; no choice is left to a
person after the data exist.

Variants, all with Q2 elements, n = 2, 4, ..., 64 elements per unit:
  uniform     no grading
  strongest   grade toward the lines through the strongest corners by exponent (both
              re-entrant corners tie at lam = 0.5445), one g = 3.7 (round K "strongest")
  all         grade toward the lines through every rate-limiting corner, one g = 3.7
              (round K "all corners")
  intensity   the intensity-aware plan (per-corner g, only corners with share >= 0.10)
  intensity_p2  the intensity plan with 6-node triangles (independent discretization)

    python -m omar_pfem.verification_agent.ushape_preregister            (pre-register)
    python -m omar_pfem.verification_agent.ushape_preregister --run      (solve)
    python -m omar_pfem.verification_agent.ushape_preregister --compare  (score)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ushape_preregistration.json")
LADDER = os.path.join(HERE, "ushape_ladder.json")
RESULT = os.path.join(HERE, "ushape_result.json")
FROZEN = ("ushape.py", "remedy_intensity.py", "remedy.py", "gates.py", "trees.py",
          "independent_p2.py")
LEVELS = (2, 4, 8, 16, 32, 64)
QOIS = ("strain_energy", "v_tip", "left_gk_von_mises", "right_gk_von_mises")
FIXED_PLANS = {"uniform": {"x": {}, "y": {}},
               "strongest": {"x": {1: 3.7, 2: 3.7}, "y": {1: 3.7}},
               "all": {"x": {0: 3.7, 1: 3.7, 2: 3.7}, "y": {0: 3.7, 1: 3.7, 2: 3.7}}}


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def main():
    from omar_pfem.verification_agent import ushape as U
    from omar_pfem.verification_agent.remedy import corner_analysis
    from omar_pfem.verification_agent.remedy_intensity import SHARE_MIN
    if os.path.exists(LADDER):
        raise SystemExit("ushape_ladder.json exists: not a pre-registration any more")
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "test the intensity-aware remedy (the lesson of round K) on a problem it "
                   "was not designed on",
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "corner_analysis": [c for c in corner_analysis(U.POLYGON, U.NU) if c["rate_limiting"]],
        "rule_constants": {"share_min": SHARE_MIN, "rho": U.RHO,
                           "indicator": "|E_c(64) - E_c(32)| of the uniform family"},
        "fixed_plans": {k: {a: {str(l): g for l, g in v[a].items()} for a in v}
                        for k, v in FIXED_PLANS.items()},
        "levels": list(LEVELS),
        "check": "five-level convergence check (qoi_convergence_multilevel) with the class "
                 "tolerances of trees.class_params, order window 0.5-5, on each variant",
        "predictions": {
            "P0 plan": "the plan grades the left re-entrant corner (1, 1), where the bending "
                       "moment is larger than at (2, 1) (LOW-MEDIUM)",
            "P1 smallest U": "the energy's five-level U at n = 64 is smallest for the "
                             "intensity variant among strongest, all and intensity (MEDIUM)",
            "P2 order": "the intensity variant's energy order is at least 2.5 (MEDIUM)",
            "P3 consistency": "its energy limit lies inside the uniform family's interval "
                              "(MEDIUM-HIGH)",
            "P4 cross-check": "the P2 discretization on the intensity plan cross-checks the "
                              "energy (MEDIUM)",
            "P5 gain": "its energy U is at least 10 times smaller than the uniform family's "
                       "at the same n (LOW-MEDIUM)",
        },
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps(pre["predictions"], indent=1))


def _plan(key, plans):
    p = plans[key]
    return {"x": {int(k): v for k, v in p["x"].items()}, "y": {int(k): v for k, v in p["y"].items()}}


def run():
    from omar_pfem.verification_agent import independent_p2 as I, ushape as U
    from omar_pfem.verification_agent.remedy_intensity import plan_intensity
    from omar_pfem.verification_agent.remedy_preregister import _relaxed_invF
    rows = json.load(open(LADDER)) if os.path.exists(LADDER) else []

    def solve(variant, plan, p2=False):
        done = {r["n"] for r in rows if r["variant"] == variant}
        for n in LEVELS:
            if n in done:
                continue
            U.MESH_PLAN = plan
            try:
                with _relaxed_invF():
                    r = I.solve_p2(U, n) if p2 else U.solve_level(n)
            finally:
                U.MESH_PLAN = {"x": {}, "y": {}}
            r["variant"] = variant
            rows.append(r)
            json.dump(rows, open(LADDER, "w"), indent=1)
            print(f"{variant} n={n} dof={r['n_dof']} E={r['strain_energy']:.12g}", flush=True)
    solve("uniform", FIXED_PLANS["uniform"])
    plan = plan_intensity(U.POLYGON, U.NU, [r for r in rows if r["variant"] == "uniform"])
    json.dump(plan, open(os.path.join(HERE, "ushape_plan.json"), "w"), indent=1)
    print("plan:", json.dumps(plan["plan"]), [(c["x"], c["y"], round(c["share"], 3))
                                              for c in plan["corners"]], flush=True)
    for v in ("strongest", "all"):
        solve(v, FIXED_PLANS[v])
    solve("intensity", plan["plan"])
    solve("intensity_p2", plan["plan"], p2=True)


def compare():
    from omar_pfem.verification_agent import gates, trees
    from omar_pfem.verification_agent.crosscheck import assess_independent, decide
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    rows = json.load(open(LADDER))
    plan = json.load(open(os.path.join(HERE, "ushape_plan.json")))
    res = {"plan": plan, "qois": {}}
    for q in QOIS:
        cp = trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))
        out = {}
        for v in ("uniform", "strongest", "all", "intensity", "intensity_p2"):
            rs = sorted((r for r in rows if r["variant"] == v), key=lambda r: r["n"])
            gm = gates.multilevel_convergence_gate([r["h"] for r in rs], [r[q] for r in rs],
                                                   band_tol=cp["band_tol"], gci_tol=cp["gci_tol"],
                                                   order_window=(0.5, 5))
            out[v] = {k: gm.get(k) for k in ("evidence", "verdict", "fit_order", "fit_limit",
                                             "uncertainty_fine")}
            out[v]["finest"] = rs[-1][q]
            out[v]["finest_dof"] = rs[-1]["n_dof"]
        a = out["intensity"]
        if a["evidence"] == "rate":
            sc = abs(a["finest"])
            rs = sorted((r for r in rows if r["variant"] == "intensity_p2"), key=lambda r: r["n"])
            ind = assess_independent([r["h"] for r in rs], [r[q] for r in rs],
                                     {"band_tol": cp["band_tol"], "gci_tol": cp["gci_tol"]}, sc)
            if ind.get("evidence") == "rate" and ind.get("verdict") == "PASS":
                st, gap = decide(a["fit_limit"], a["uncertainty_fine"], ind["fit_limit"],
                                 ind["uncertainty_fine"] * sc, sc)
            else:
                st, gap = "independent_unconverged", None
            out["p2_crosscheck"] = {"status": st, "gap_rel": gap}
        res["qois"][q] = out
    E = res["qois"]["strain_energy"]
    U_ = lambda v: E[v]["uncertainty_fine"] if E[v]["evidence"] == "rate" else None
    graded = {(c["x"], c["y"]) for c in plan["corners"] if c["graded"]}
    met = {
        "P0 plan grades (1, 1)": (1, 1) in graded,
        "P1 smallest U": U_("intensity") is not None and all(
            U_(v) is None or U_("intensity") <= U_(v) for v in ("strongest", "all")),
        "P2 order >= 2.5": (E["intensity"]["fit_order"] or 0) >= 2.5,
        "P3 limit inside uniform interval": U_("uniform") is not None
        and E["intensity"]["fit_limit"] is not None
        and abs(E["intensity"]["fit_limit"] - E["uniform"]["fit_limit"])
        <= U_("uniform") * abs(E["uniform"]["finest"]),
        "P4 P2 cross-checks the energy": E.get("p2_crosscheck", {}).get("status") == "cross_checked",
        "P5 U at least 10x below uniform": U_("intensity") is not None and U_("uniform") is not None
        and U_("intensity") * 10 <= U_("uniform"),
    }
    res["predictions_met"] = met
    json.dump(res, open(RESULT, "w"), indent=1, default=float)
    for q, o in res["qois"].items():
        print(q, {v: (o[v]["evidence"], o[v]["fit_order"] and round(o[v]["fit_order"], 2),
                      o[v]["uncertainty_fine"]) for v in ("uniform", "strongest", "all", "intensity")},
              o.get("p2_crosscheck", {}).get("status"))
    print(f"predictions met {sum(met.values())}/{len(met)}")
    for k, v in met.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    if "--compare" in sys.argv:
        compare()
    elif "--run" in sys.argv:
        run()
    else:
        main()
