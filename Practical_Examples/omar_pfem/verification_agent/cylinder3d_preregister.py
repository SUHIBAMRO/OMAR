"""Pre-registration of the unseen 3D benchmark (cylinder3d.py), written and
committed BEFORE any loaded solve (only a zero-load smoke test was run, which
gives the trivial solution). Same protocol as cook/plate_hole/lshape/crack.

Frozen here: the specification, the unchanged rules' predictions, the
developer's expectations, the scoring tests and the SHA-256 of the files that
decide verdicts (gates.py, trees.py, orchestrate.py, crosscheck.py). compare()
refuses to score if any of them changed.

The exact (1D) solution is computed here too and stored, for scoring only; the
agent's verdicts never use it.

    python -m omar_pfem.verification_agent.cylinder3d_preregister            (before solving)
    python -m omar_pfem.verification_agent.cylinder3d_preregister --compare  (after solving)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cylinder3d_preregistration.json")
FROZEN = ("gates.py", "trees.py", "orchestrate.py", "crosscheck.py")


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def main():
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import DEFAULT_SPECS
    from omar_pfem.verification_agent.cylinder3d import CACHE, exact_qois
    from omar_pfem.verification_agent.orchestrate import Cylinder3DAdapter
    if os.path.exists(CACHE):
        raise SystemExit("cylinder3d_ladder.json exists: not a pre-registration any more")
    spec = dict(DEFAULT_SPECS["thick_cylinder_3d"])
    spec["qois"] = list(Cylinder3DAdapter.qois)
    rules = trees.apply_rules(spec)
    exact = exact_qois()
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "first held-out 3D problem; same element (Q20, reduced rule), material "
                   "(nu = 0.45) and curved iso-parametric boundaries as B3, but with an exact "
                   "solution, so the certified uncertainty can be compared with the TRUE error",
        "spec": spec, "adapter_description": Cylinder3DAdapter.description,
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "posthoc_switches": {"VERIF_POSTHOC_SINGULAR": "off (no singular feature here)"},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"], "role": r["role"]}
                             for q, r in rules.items()},
        "exact_for_scoring_only": exact,
        "scoring_tests": {
            "T1 rule predictions": "the unchanged rules predict PASS for strain_energy, "
                                   "reaction_inner_radial, u_outer_radial and nothing for the "
                                   "three stress quantities; hits counted as before",
            "T2 coverage of the true error": "for every quantity the agent passes with RATE "
                                             "evidence: |exact - certified limit| <= U_fine x "
                                             "scale (the agent's own uncertainty). Counted per "
                                             "quantity; no tolerance added",
            "T3 finest-level error inside U": "secondary: |exact - finest value| compared with "
                                              "|certified limit - finest value| + U x scale",
            "T4 final status": "crosscheck v2 has no independent discretization here, so every "
                               "rate-certified quantity is single_family -> INCONCLUSIVE "
                               "(final); T2 says whether that caution was needed",
        },
        "developer_expectations": {
            "strain_energy": "PASS rate; fitted order 3-5; T2 holds",
            "reaction_inner_radial": "PASS rate; T2 holds",
            "u_outer_radial": "PASS rate; T2 holds",
            "gk_avg_hoop": "PASS rate; T2 holds (smooth weight, smooth field); MEDIUM confidence",
            "disc_avg_hoop": "PASS, band or rate (sharp region cut by elements: irregular "
                             "convergence); if rate, T2 holds with LOW confidence",
            "peak_hoop": "PASS rate with an order near 1 (Gauss points approach the inner wall "
                         "at O(h)), limit below the exact surface value only by the fit error; "
                         "T2 holds with LOW confidence",
            "overall": "at least 5 of 6 quantities PASS; T2 holds for every rate-certified "
                       "quantity. A T2 failure would show that one-family certification can "
                       "miss the true error even on a smooth 3D problem with this element",
        },
        "rules_changed_for_this_problem": False,
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps(pre["rule_predictions"], indent=1))


def compare():
    from omar_pfem.verification_agent.verify_report import rules
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    R = rules.__wrapped__("thick_cylinder_3d")
    ad = R["adapter"]
    ex = pre["exact_for_scoring_only"]
    rows, hits, npred = {}, 0, 0
    for q, pr in pre["rule_predictions"].items():
        ov, ev = R["combine"][q][:2]
        ml = R["v"][q].get("qoi_convergence_multilevel", {})
        c3 = R["v"][q].get("qoi_convergence", {})
        hs, vals, scale = ad.series(q)
        sc = scale if scale else abs(vals[-1])
        row = {"verdict": ov, "evidence": ev, "fit_order": ml.get("fit_order"),
               "multilevel_evidence": ml.get("evidence"), "three_level_order":
               c3.get("observed_order"), "finest": vals[-1], "exact": ex[q],
               "true_error_finest_rel": abs(vals[-1] - ex[q]) / abs(ex[q])}
        if pr["predicted"] is not None:
            npred += 1
            row["rule_hit"] = pr["predicted"] == ov
            hits += row["rule_hit"]
        if ov == "PASS" and ev == "rate":
            if ml.get("evidence") == "rate":
                lim, U = ml["fit_limit"], ml["uncertainty_fine"]
            else:
                lim, U = c3["richardson_extrapolate"], c3["gci_fine"]
            row.update(limit=lim, U_rel_scale=U, scale=sc,
                       true_error_of_limit_rel_scale=abs(ex[q] - lim) / sc,
                       T2_covered=abs(ex[q] - lim) <= U * sc,
                       T3_finest_within=abs(ex[q] - vals[-1]) <= abs(lim - vals[-1]) + U * sc)
        rows[q] = row
    rate = [q for q, r in rows.items() if "T2_covered" in r]
    res = {"levels_k": [r["k"] for r in ad.rows], "n_dof": [r["n_dof"] for r in ad.rows],
           "rule_hits": hits, "rule_predictions": npred, "qois": rows,
           "T2_covered": sum(rows[q]["T2_covered"] for q in rate), "T2_of": len(rate),
           "strategies": [{k: r[k] for k in ("grading",) + tuple(pre["rule_predictions"])}
                          for r in ad.strategy_rows]}
    json.dump(res, open(os.path.join(HERE, "cylinder3d_result.json"), "w"), indent=1)
    print(f"rule predictions {hits}/{npred}; T2 coverage {res['T2_covered']}/{res['T2_of']}")
    for q, r in rows.items():
        print(f"  {q:<24}{r['verdict']}/{r['evidence']:<5} p={r['fit_order']} "
              f"err_finest={100 * r['true_error_finest_rel']:.4f}% "
              + (f"limit_err/scale={100 * r['true_error_of_limit_rel_scale']:.4f}% "
                 f"U={100 * r['U_rel_scale']:.4f}% covered={r['T2_covered']}"
                 if "T2_covered" in r else ""))
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    compare() if "--compare" in sys.argv else main()
