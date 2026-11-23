"""Pre-registration of the automatic remedy on the two real singular problems (L-shape,
edge crack), written and committed BEFORE any graded solve of them (round K, 2026-11-23).

The agent analyses every corner of the polygon (remedy.corner_analysis: interior angle,
edge conditions, Williams exponents), finds the rate-limiting corners, and plans a graded
mesh family (remedy.plan_problem). Two variants are run, so that the corner analysis
itself is tested, not only the grading:

  strongest  graded only toward the lines through the strongest singular point (the
             problem module's own grading, g from the plan): the analysis predicts that the
             weaker clamped corners then limit the energy order to 2 x 0.7112 = 1.42
  all        graded toward the lines through every rate-limiting corner
             (remedy.graded_coords): predicted order 2p = 4

No exact solution exists; the checks are the agent's own five-level convergence check
(class tolerances, order window 0.5-5) and the P2 cross-check on the all-graded meshes.

    python -m omar_pfem.verification_agent.remedy_preregister            (before)
    python -m omar_pfem.verification_agent.remedy_preregister --run      (solve)
    python -m omar_pfem.verification_agent.remedy_preregister --compare  (score)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "remedy_preregistration.json")
RESULT = os.path.join(HERE, "remedy_result.json")
FROZEN = ("gates.py", "trees.py", "remedy.py", "lshape.py", "crack.py", "independent_p2.py",
          "crosscheck.py")
LEVELS = (2, 4, 8, 16, 32, 64)
MODULES = {"l_shape": "lshape", "edge_crack": "crack"}
QOIS = {"l_shape": ("strain_energy", "u_top_left", "v_right_top", "corner_gk_avg_von_mises",
                    "far_gk_avg_syy"),
        "edge_crack": ("strain_energy", "u_cod_mouth", "v_top_right", "tip_gk_avg_von_mises",
                       "far_gk_avg_syy")}


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def _ladder(geo):
    return os.path.join(HERE, f"remedy_{geo}_ladder.json")


def main():
    from omar_pfem.verification_agent import remedy as R
    for geo in MODULES:
        if os.path.exists(_ladder(geo)):
            raise SystemExit(f"{_ladder(geo)} exists: not a pre-registration any more")
    plans = {geo: R.plan_problem(R.POLYGONS[geo], 0.3) for geo in MODULES}
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "plans": plans, "levels": list(LEVELS),
        "targets_note": "both modules use one coordinate function for x and for y (shifted "
                        "by -1 for the crack); the all-graded variant grades toward the "
                        "union of the x and the shifted y targets, {0, 1, 2}",
        "predictions": {
            "P1 strongest-only energy order": "within 0.2 of 2 x 0.7112 = 1.42 for both "
                                              "problems (the weaker corners limit it)",
            "P2 all-graded energy order": "between 3.0 and 4.6 for both problems (theory 4)",
            "P3 uncertainty": "five-level U of the energy, all-graded n = 64, below 1/100 of "
                              "the uniform family's certified U (post hoc extension, n = 128)",
            "P4 consistency": "the all-graded energy limit lies within the uniform family's "
                              "certified interval",
            "P5 P2 cross-check": "on the all-graded meshes the P2 sequence cross-checks the "
                                 "energy (rule v2)",
            "confidence": "P1 MEDIUM (pre-asymptotic mixing); P2-P5 MEDIUM",
        },
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps({g: {k: p[k] for k in ("g", "x_targets", "y_targets",
                                            "predicted_energy_order_graded_strongest_only")}
                      for g, p in plans.items()}, indent=1))


class _relaxed_invF:
    """AMENDMENT (2026-11-23, after the first graded levels failed and before any scoring):
    scikit-fem inverts its isoparametric map with an ABSOLUTE tolerance of 1e-12 on the
    Newton step in reference coordinates. On elements of size ~1e-6..1e-8 the round-off of
    the physical coordinates alone, amplified by the inverse Jacobian, exceeds it, so the
    traction FacetBasis fails. During the graded solves the same Newton iteration is used
    with the tolerance scaled to that floor: max(1e-12, 1e3 x eps x |x| x |DF^-1|) per point.
    No criterion changed."""

    def __enter__(self):
        import numpy as np
        from skfem.mapping import MappingIsoparametric as M
        self.orig = M.invF
        # AMENDMENT 2 (same reason, found when the all-graded energy jumped at n = 32): the
        # problem modules select boundary facets with np.isclose (relative tolerance 1e-5,
        # i.e. 2e-5 at y = 2); on rows of elements thinner than that the short facets
        # across the row are selected too (the total top load came out 2.0 instead of 1.0).
        # During the graded solves the modules' np.isclose uses an absolute 1e-12.
        import types
        self.mods = []
        for name in ("lshape", "crack"):
            mod = __import__(f"omar_pfem.verification_agent.{name}", fromlist=["np"])
            proxy = types.ModuleType("np_strict_isclose")
            proxy.__dict__.update(np.__dict__)
            proxy.isclose = lambda a, b, **kw: np.isclose(a, b, rtol=0.0, atol=1e-12)
            self.mods.append((mod, mod.np))
            mod.np = proxy

        def invF(s_, x, tind=None, newton_max_iters=50, newton_tol=1e-12):
            X = np.zeros(x.shape) + .5
            for _ in range(newton_max_iters):
                F = s_.F(X, tind)
                invDF = s_.invDF(X, tind)
                dX = np.einsum("ijkl,jkl->ikl", invDF, x - F)
                X = np.clip(X + dX, 0., 1.)
                floor = 1e3 * np.finfo(float).eps * (np.abs(x).max() + 1) * np.abs(invDF).max(
                    axis=(0, 1))
                tol = np.maximum(newton_tol, floor.max(axis=-1))
                if (np.linalg.norm(dX, 1, (0, 2)) < tol).all():
                    return X
            raise Exception("Newton iteration did not reach the round-off floor")
        M.invF = invF

    def __exit__(self, *a):
        from skfem.mapping import MappingIsoparametric as M
        M.invF = self.orig
        for mod, orig_np in self.mods:
            mod.np = orig_np


def _solve(geo, n, g, variant):
    from omar_pfem.verification_agent import remedy as R
    mod = __import__(f"omar_pfem.verification_agent.{MODULES[geo]}", fromlist=["solve_level"])
    if variant == "strongest":
        return mod.solve_level(n, g)
    orig = mod._coords
    mod._coords = lambda n_, g_: R.graded_coords(n_, g_, {0, 1, 2})
    try:
        return mod.solve_level(n, g)
    finally:
        mod._coords = orig


def _solve_p2(geo, n, g):
    from omar_pfem.verification_agent import independent_p2 as I, remedy as R
    mod = __import__(f"omar_pfem.verification_agent.{MODULES[geo]}", fromlist=["solve_level"])
    orig = mod._coords
    mod._coords = lambda n_, g_: R.graded_coords(n_, g_, {0, 1, 2})
    q2_mesh = mod.mesh
    try:
        mod.mesh = lambda n_, g_=1.0: q2_mesh(n_, g)
        return I.solve_p2(mod, n)
    finally:
        mod._coords, mod.mesh = orig, q2_mesh


def run():
    pre = json.load(open(OUT))
    for geo in MODULES:
        g = pre["plans"][geo]["g"]
        path = _ladder(geo)
        rows = json.load(open(path)) if os.path.exists(path) else []
        done = {(r["variant"], r["n"]) for r in rows}
        for variant in ("strongest", "all", "all_p2"):
            for n in LEVELS:
                if (variant, n) in done:
                    continue
                with _relaxed_invF():
                    r = (_solve_p2(geo, n, g) if variant == "all_p2"
                         else _solve(geo, n, g, variant))
                r["variant"], r["g_plan"] = variant, g
                rows.append(r)
                json.dump(rows, open(path, "w"), indent=1)
                print(f"{geo} {variant} n={n} dof={r['n_dof']} E={r['strain_energy']:.12g}",
                      flush=True)


def compare():
    from omar_pfem.verification_agent import gates, trees
    from omar_pfem.verification_agent.crosscheck import assess_independent, decide
    from omar_pfem.verification_agent.verify_report import rules
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    # crosscheck.py gained a problem in its lists after this scoring (round K); the scoring
    # uses only assess_independent and decide, which must be unchanged since be9e643
    from omar_pfem.verification_agent.prereg_util import functions_unchanged
    exempt = {"crosscheck.py": ("be9e643", ("assess_independent", "decide"))}
    for f in list(changed):
        if f in exempt and functions_unchanged(f, *exempt[f]):
            changed.remove(f)
            print(f"note: {f} changed after pre-registration; the functions used here are unchanged")
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    res, met = {}, {}
    for geo in MODULES:
        rows = json.load(open(_ladder(geo)))
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        try:
            Ru = rules.__wrapped__(geo)
        finally:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        out = {}
        for q in QOIS[geo]:
            cp = trees.class_params("qoi_convergence_multilevel", trees.classify_qoi(q))
            row = {}
            for variant in ("strongest", "all", "all_p2"):
                rs = sorted((r for r in rows if r["variant"] == variant), key=lambda r: r["n"])
                gm = gates.multilevel_convergence_gate([r["h"] for r in rs], [r[q] for r in rs],
                                                       band_tol=cp["band_tol"],
                                                       gci_tol=cp["gci_tol"],
                                                       order_window=(0.5, 5))
                row[variant] = {k: gm.get(k) for k in ("evidence", "verdict", "fit_order",
                                                       "fit_limit", "uncertainty_fine")}
                row[variant]["finest"] = rs[-1][q]
                row[variant]["finest_dof"] = rs[-1]["n_dof"]
            mu = Ru["v"][q].get("qoi_convergence_multilevel", {})
            _, uvals, _ = Ru["adapter"].series(q)
            row["uniform"] = {"evidence": mu.get("evidence"), "fit_order": mu.get("fit_order"),
                              "fit_limit": mu.get("fit_limit"),
                              "uncertainty_fine": mu.get("uncertainty_fine"),
                              "finest": uvals[-1], "verdict": Ru["combine"][q][0]}
            a = row["all"]
            if a["evidence"] == "rate":
                sc = abs(a["finest"])
                ind = assess_independent([r["h"] for r in sorted(
                    (r for r in rows if r["variant"] == "all_p2"), key=lambda r: r["n"])],
                    [r[q] for r in sorted((r for r in rows if r["variant"] == "all_p2"),
                                          key=lambda r: r["n"])],
                    {"band_tol": cp["band_tol"], "gci_tol": cp["gci_tol"]}, sc)
                if ind.get("evidence") == "rate" and ind.get("verdict") == "PASS":
                    st, gap = decide(a["fit_limit"], a["uncertainty_fine"], ind["fit_limit"],
                                     ind["uncertainty_fine"] * sc, sc)
                else:
                    st, gap = "independent_unconverged", None
                row["p2_crosscheck"] = {"status": st, "gap_rel": gap,
                                        "independent": {k: ind.get(k) for k in
                                                        ("evidence", "fit_order",
                                                         "uncertainty_fine")}}
            out[q] = row
        res[geo] = out
        E = out["strain_energy"]
        u = E["uniform"]
        met[f"{geo} P1 strongest-only order within 0.2 of 1.42"] = abs(
            (E["strongest"]["fit_order"] or 0) - 1.4223) <= 0.2
        met[f"{geo} P2 all-graded order 3.0-4.6"] = 3.0 <= (E["all"]["fit_order"] or 0) <= 4.6
        met[f"{geo} P3 U below 1/100 of uniform"] = (
            E["all"]["uncertainty_fine"] is not None and u["uncertainty_fine"] is not None
            and E["all"]["uncertainty_fine"] < u["uncertainty_fine"] / 100)
        met[f"{geo} P4 limit inside uniform interval"] = (
            E["all"]["fit_limit"] is not None and u["fit_limit"] is not None
            and abs(E["all"]["fit_limit"] - u["fit_limit"])
            <= u["uncertainty_fine"] * abs(u["finest"]))
        met[f"{geo} P5 P2 cross-checks the energy"] = (
            E.get("p2_crosscheck", {}).get("status") == "cross_checked")
    res["predictions_met"] = met
    json.dump(res, open(RESULT, "w"), indent=1, default=float)
    for geo in MODULES:
        for q, r in res[geo].items():
            print(geo, q, {v: (r[v]["evidence"], r[v]["fit_order"] and round(r[v]["fit_order"], 3),
                               r[v]["uncertainty_fine"]) for v in ("uniform", "strongest", "all")},
                  r.get("p2_crosscheck", {}).get("status"))
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
