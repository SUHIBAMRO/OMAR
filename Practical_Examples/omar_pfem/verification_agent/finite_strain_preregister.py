"""Pre-registration: coverage of the certified interval on 12 finite-strain neo-Hookean
problems with exact (manufactured) solutions (2026-11-24). Written before the ladders are
solved; disclosed: exact values for all cases and a smoke test of one case
(s15_e20_nu45, n = 4 and 8; Newton converged, errors fell) were computed first.

Method: the framework's rule plan for a smooth 2D neo-Hookean problem, element Q9, mesh
refinement family n = 4..64, post hoc switches on as in calibration.py; class tolerances;
covered iff |exact - limit| <= U |finest|. GCI: conventional.conventional.

    python -m omar_pfem.verification_agent.finite_strain_preregister            (pre-register)
    python -m omar_pfem.verification_agent.finite_strain_preregister --score
"""
import datetime
import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "finite_strain_preregistration.json")
RESULT = os.path.join(HERE, "finite_strain_result.json")
FROZEN = ("finite_strain_exact.py", "gates.py", "trees.py", "orchestrate.py",
          "conventional.py", "calibration.py", "wedge_preregister.py")


def _sha(n):
    return hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest()


def main():
    from omar_pfem.verification_agent.finite_strain_exact import LADDER, case_list
    if os.path.exists(LADDER):
        raise SystemExit("ladder exists: not a pre-registration")
    pre = {"written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
           "cases": [c["id"] for c in case_list()],
           "predictions": {
               "F1 coverage": "of the quantities the framework certifies, the exact value lies "
                              "inside the interval for at least 90% (MEDIUM)",
               "F2 liveness": "the framework certifies at least 27 of the 36 quantities "
                              "(smooth fields, quadratic elements) (MEDIUM)",
               "F3 conservative": "median framework effectivity at most 0.5 (MEDIUM)",
               "F4 GCI": "GCI covers no larger a fraction of what it certifies than the "
                         "framework (LOW-MEDIUM)"}}
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps(pre["predictions"], indent=1))


class _Adapter:
    def __init__(self, fam):
        self.rows, self.hs = fam, [r["h"] for r in fam]

    @property
    def family(self):
        return [{"n_elem": r["n"] ** 2, "min_detF": r["min_J"]} for r in self.rows]

    def series(self, q):
        return self.hs, [r[q] for r in self.rows], None

    def singular_exponent(self):
        return None

    def strategies(self, q):
        return None


def score():
    import numpy as np
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.calibration import _limit
    from omar_pfem.verification_agent.conventional import conventional
    from omar_pfem.verification_agent.finite_strain_exact import EXACT, LADDER, QOIS
    from omar_pfem.verification_agent.orchestrate import combine, convergence_params, run_check
    from omar_pfem.verification_agent.wedge_preregister import _wilson
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    ex, lad = json.load(open(EXACT)), json.load(open(LADDER))
    rows = []
    for cid in pre["cases"]:
        fam = sorted((r for r in lad if r["case"] == cid), key=lambda r: r["n"])
        ad = _Adapter(fam)
        sd = dict(name=cid, geometry="fs_" + cid, dim=2, material="neo_hookean",
                  loading="body_force", regularity="smooth", element="Q9",
                  reference="mesh_refinement_family", qois=list(QOIS), well_posedness=[],
                  open_questions=[])
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        try:
            plan = trees.rule_plan(sd)
            cp = convergence_params(plan)
            v = {}
            for c in plan["checks"]:
                for q in c["qois"]:
                    v.setdefault(q, {})[c["check"]] = run_check(ad, c["check"], q, c["params"],
                                                                 element="Q9", conv_params=cp.get(q))
        finally:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        for q in QOIS:
            comb = combine(v[q])
            e = ex[cid]["exact"][q]
            sc = abs(fam[-1][q])
            row = {"case": cid, "qoi": q, "verdict": list(comb[:2]), "exact": e,
                   "finest_true_error": abs(fam[-1][q] - e) / sc}
            if comb[0] == "PASS" and comb[1] == "rate":
                lim, U = _limit(v[q])
                row["framework"] = {"limit": lim, "U": U, "effectivity": abs(e - lim) / (U * sc),
                                    "covered": abs(e - lim) <= U * sc}
            A, g = conventional(ad, q)
            if A == "PASS":
                lim, U = g["richardson_extrapolate"], g["gci_fine"]
                row["gci"] = {"limit": lim, "U": U, "effectivity": abs(e - lim) / (U * sc) if U else math.inf,
                              "covered": abs(e - lim) <= U * sc}
            rows.append(row)
    fw = [r["framework"] for r in rows if "framework" in r]
    gc = [r["gci"] for r in rows if "gci" in r]
    kf, kg = sum(f["covered"] for f in fw), sum(f["covered"] for f in gc)
    summ = {"quantities": len(rows), "framework_certified": len(fw), "framework_covered": kf,
            "framework_coverage_wilson95": _wilson(kf, len(fw)),
            "framework_effectivity_median": float(np.median([f["effectivity"] for f in fw])) if fw else None,
            "framework_effectivity_max": float(max(f["effectivity"] for f in fw)) if fw else None,
            "gci_certified": len(gc), "gci_covered": kg, "gci_coverage_wilson95": _wilson(kg, len(gc))}
    cf = kf / len(fw) if fw else None
    cg = kg / len(gc) if gc else None
    met = {"F1 coverage >= 90%": cf is not None and cf >= 0.9,
           "F2 certifies >= 27 of 36": len(fw) >= 27,
           "F3 median effectivity <= 0.5": summ["framework_effectivity_median"] is not None
           and summ["framework_effectivity_median"] <= 0.5,
           "F4 GCI coverage not above framework": cf is not None and (cg is None or cg <= cf)}
    res = {"summary": summ, "predictions_met": met, "rows": rows}
    json.dump(res, open(RESULT, "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    for k, val in met.items():
        print(f"  {'met ' if val else 'MISS'} {k}")
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    score() if "--score" in sys.argv else main()
