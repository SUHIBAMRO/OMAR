"""Pre-registration of the coverage study on 32 exact corner solutions (2026-11-24).

Written and committed BEFORE the ladders of wedge_exact.py are solved. Disclosed: the exact
values and self-checks of all 32 cases were computed first (no finite element solve), and
a smoke test solved levels 0-2 of two cases (cf_90_nu30, ff_270_nu45) to check the code
(the energy converged at about 2 lam). No other finite element result exists.

Method (fixed here): for each case the framework is run as `agent rules` runs it, with
the post hoc singular-order extension on (VERIF_POSTHOC_SINGULAR=1, as in calibration.py):
the rule plan from trees for a 2D linear-elastic problem with a corner, element T6,
reference = mesh refinement family; strategies at level 3 (g = 1.5, 2.0); class tolerances
from trees.class_params. A quantity is certified when its combined verdict is PASS with a
rate; its interval is [limit - U |finest|, limit + U |finest|] (calibration._limit).
Conventional GCI: conventional.conventional (three finest levels, class tolerance).
Covered: |exact - limit| <= U |finest|. Effectivity: |exact - limit| / (U |finest|).

    python -m omar_pfem.verification_agent.wedge_preregister            (pre-register)
    python -m omar_pfem.verification_agent.wedge_preregister --score    (after solve)
"""
import datetime
import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wedge_preregistration.json")
RESULT = os.path.join(HERE, "wedge_result.json")
FROZEN = ("wedge_exact.py", "gates.py", "trees.py", "orchestrate.py", "conventional.py",
          "calibration.py")


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def main():
    from omar_pfem.verification_agent.wedge_exact import LADDER, case_list
    if os.path.exists(LADDER):
        raise SystemExit("wedge_exact_ladder.json exists: not a pre-registration any more")
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "cases": [c["id"] for c in case_list()],
        "predictions": {
            "P1 coverage": "of the quantities the framework certifies with a rate, the exact "
                           "value lies inside the certified interval for at least 90% (MEDIUM)",
            "P2 GCI below": "conventional GCI covers a smaller fraction of what it certifies "
                            "than the framework does (MEDIUM)",
            "P3 GCI misses": "GCI certifies at least 3 quantities whose interval misses the "
                             "exact value (LOW-MEDIUM)",
            "P4 peaks": "no peak_von_mises is certified, by either method (HIGH)",
            "P5 conservative": "the framework's median effectivity is at most 0.5 (MEDIUM)",
            "P6 liveness": "the framework certifies at least half of the 128 verifiable "
                           "quantities (LOW-MEDIUM: quadratic elements on singular corners "
                           "often end pre-asymptotic or band-only)",
        },
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps(pre["predictions"], indent=1))


class WedgeAdapter:
    def __init__(self, cid, rows, lam):
        fam = sorted((r for r in rows if r["case"] == cid and r["g"] == 1.0),
                     key=lambda r: r["level"])
        self.rows, self.hs, self.lam = fam, [r["h"] for r in fam], lam
        from omar_pfem.verification_agent.wedge_exact import STRATEGY_LEVEL
        self.strategy_rows = sorted((r for r in rows if r["case"] == cid
                                     and r["level"] == STRATEGY_LEVEL), key=lambda r: r["g"])

    @property
    def family(self):
        return [{"n_elem": r["n_elem"], "min_detF": None} for r in self.rows]

    def series(self, qoi):
        return self.hs, [r[qoi] for r in self.rows], None

    def singular_exponent(self):
        return self.lam

    def strategies(self, qoi):
        if len(self.strategy_rows) < 2:
            return None
        return [r[qoi] for r in self.strategy_rows], None


def _wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def certify(cid, rows, case):
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.calibration import _limit
    from omar_pfem.verification_agent.conventional import conventional
    from omar_pfem.verification_agent.orchestrate import combine, convergence_params, run_check
    from omar_pfem.verification_agent.wedge_exact import QOIS
    reg = "clamped_free_corner" if case["bc0"] != case["bc1"] else "reentrant_corner"
    sd = dict(name=cid, geometry="wedge_" + cid, dim=2, material="linear_elastic",
              loading="prescribed_displacement", regularity=reg, element="T6",
              reference="mesh_refinement_family", qois=list(QOIS), well_posedness=[],
              open_questions=[])
    ad = WedgeAdapter(cid, rows, case["lams"][0])
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    try:
        plan = trees.rule_plan(sd)
        cp = convergence_params(plan)
        v = {}
        for c in plan["checks"]:
            for q in c["qois"]:
                v.setdefault(q, {})[c["check"]] = run_check(ad, c["check"], q, c["params"],
                                                             element="T6", conv_params=cp.get(q))
    finally:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    out = []
    for q in QOIS:
        comb = combine(v[q])
        e = case["exact"][q]
        _, vals, _ = ad.series(q)
        sc = abs(vals[-1])
        row = {"case": cid, "qoi": q, "class": trees.classify_qoi(q), "verdict": list(comb[:2]),
               "verifiable": math.isfinite(e)}
        if math.isfinite(e):
            row["exact"] = e
            row["finest_true_error"] = abs(vals[-1] - e) / sc
        if comb[0] == "PASS" and comb[1] == "rate":
            lim, U = _limit(v[q])
            row["framework"] = {"limit": lim, "U": U}
            if math.isfinite(e):
                row["framework"].update(true_error=abs(e - lim) / sc,
                                        effectivity=abs(e - lim) / (U * sc) if U else math.inf,
                                        covered=abs(e - lim) <= U * sc)
        A, g = conventional(ad, q)
        if A == "PASS":
            row["gci"] = {"limit": g["richardson_extrapolate"], "U": g["gci_fine"]}
            if math.isfinite(e):
                lim, U = g["richardson_extrapolate"], g["gci_fine"]
                row["gci"].update(true_error=abs(e - lim) / sc,
                                  effectivity=abs(e - lim) / (U * sc) if U else math.inf,
                                  covered=abs(e - lim) <= U * sc)
        out.append(row)
    return out


def score():
    import numpy as np
    from omar_pfem.verification_agent.wedge_exact import EXACT, LADDER
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    ex, rows = json.load(open(EXACT)), json.load(open(LADDER))
    allr = []
    for cid in pre["cases"]:
        allr += certify(cid, rows, ex[cid])
    ver = [r for r in allr if r["verifiable"]]
    fw = [r["framework"] for r in ver if "framework" in r]
    gc = [r["gci"] for r in ver if "gci" in r]
    k_fw, k_gc = sum(f["covered"] for f in fw), sum(f["covered"] for f in gc)
    peaks = [r for r in allr if not r["verifiable"]]
    summ = {
        "quantities": len(allr), "verifiable": len(ver), "peaks": len(peaks),
        "framework_certified": len(fw), "framework_covered": k_fw,
        "framework_coverage": k_fw / len(fw) if fw else None,
        "framework_coverage_wilson95": _wilson(k_fw, len(fw)),
        "framework_effectivity_median": float(np.median([f["effectivity"] for f in fw])) if fw else None,
        "framework_effectivity_max": float(max(f["effectivity"] for f in fw)) if fw else None,
        "gci_certified": len(gc), "gci_covered": k_gc,
        "gci_coverage": k_gc / len(gc) if gc else None,
        "gci_coverage_wilson95": _wilson(k_gc, len(gc)),
        "gci_not_covered": len(gc) - k_gc,
        "peaks_certified_framework": sum("framework" in r for r in peaks),
        "peaks_certified_gci": sum("gci" in r for r in peaks),
        "by_class": {},
    }
    for cls in sorted({r["class"] for r in ver}):
        rc = [r for r in ver if r["class"] == cls]
        f = [r["framework"] for r in rc if "framework" in r]
        g = [r["gci"] for r in rc if "gci" in r]
        summ["by_class"][cls] = {"n": len(rc), "fw_cert": len(f), "fw_cov": sum(x["covered"] for x in f),
                                 "gci_cert": len(g), "gci_cov": sum(x["covered"] for x in g)}
    fwc, gcc = summ["framework_coverage"], summ["gci_coverage"]
    met = {
        "P1 coverage >= 90%": fwc is not None and fwc >= 0.90,
        "P2 GCI coverage below framework": fwc is not None and gcc is not None and gcc < fwc,
        "P3 GCI misses >= 3": summ["gci_not_covered"] >= 3,
        "P4 no peak certified": summ["peaks_certified_framework"] == 0 and summ["peaks_certified_gci"] == 0,
        "P5 median effectivity <= 0.5": summ["framework_effectivity_median"] is not None
        and summ["framework_effectivity_median"] <= 0.5,
        "P6 certifies >= 64 of 128": summ["framework_certified"] >= 64,
    }
    res = {"summary": summ, "predictions_met": met, "rows": allr}
    json.dump(res, open(RESULT, "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    print(f"predictions met {sum(met.values())}/{len(met)}")
    for k, v in met.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    if "--score" in sys.argv:
        score()
    else:
        main()
