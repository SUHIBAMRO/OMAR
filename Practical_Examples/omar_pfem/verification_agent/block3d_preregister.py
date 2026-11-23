"""Pre-registration of the second unseen 3D benchmark (block3d.py), written and
committed BEFORE any loaded solve. Same protocol as crack_preregister.py: the rules'
predictions (unchanged), the theory, and the developer's expectations for the
default rules and for the post hoc singular-order extension (VERIF_POSTHOC_SINGULAR=1),
here tested for the first time in 3D. SHA-256 of the verdict-deciding files frozen.

    python -m omar_pfem.verification_agent.block3d_preregister            (before solving)
    python -m omar_pfem.verification_agent.block3d_preregister --compare  (after solving)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "block3d_preregistration.json")
FROZEN = ("gates.py", "trees.py", "orchestrate.py", "crosscheck.py")


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def main():
    from omar_pfem.verification_agent import trees
    from omar_pfem.verification_agent.agent import DEFAULT_SPECS
    from omar_pfem.verification_agent.block3d import CACHE, corner_exponent
    from omar_pfem.verification_agent.orchestrate import Block3DAdapter
    if os.path.exists(CACHE):
        raise SystemExit("block3d_ladder.json exists: not a pre-registration any more")
    spec = dict(DEFAULT_SPECS["clamped_block_3d"])
    spec["qois"] = list(Block3DAdapter.qois)
    lam = corner_exponent()
    rules = trees.apply_rules(spec)
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "second held-out 3D problem: singular clamped/free edges with quadratic "
                   "elements; first 3D test of the post hoc singular-order extension",
        "spec": spec, "adapter_description": Block3DAdapter.description,
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "edge_exponent_lam": lam,
        "theory": {"energy_order_if_edges_dominate": 2 * lam,
                   "peak_slope_if_edges_dominate": lam - 1,
                   "singular_order_tol": 0.15,
                   "caveat": "the four vertices where two free faces meet the clamped face carry "
                             "3D vertex singularities whose exponents are not computed; they act "
                             "on point neighbourhoods and are expected to affect the energy less "
                             "than the edges, but may dominate the PEAK"},
        "rule_predictions": {q: {"predicted": r["predicted"], "required": r["required"],
                                 "rules": r["rules"], "role": r["role"]}
                             for q, r in rules.items()},
        "stated_in_advance": {
            "default rules": "every singular_order check INCONCLUSIVE (theory for degree-1 "
                             "elements only); verdicts therefore INCONCLUSIVE where that check "
                             "is required (strain_energy, the two Gaussian averages)",
            "pre-asymptotic label": "not expected to trigger for strain_energy",
        },
        "developer_expectations_posthoc": {
            "strain_energy": "singular_order PASS (order 2 lam = 1.42 within 0.15) and "
                             "convergence PASS with rate; MEDIUM confidence (vertex effect)",
            "w_tip": "convergence PASS with rate, fitted order >= 1.27",
            "mid_gk_avg_sxx": "convergence PASS with rate; singular_order PASS (order >= 1.27, "
                              "pollution from the edges); MEDIUM confidence",
            "root_gk_avg_von_mises": "convergence PASS (band or rate); LOW confidence "
                                     "(weight centred on the singular edge)",
            "peak_von_mises": "convergence FAIL; located at the clamped face (x < 0.1) on the "
                              "finest three levels; log-log slope negative; slope = lam - 1 = "
                              "-0.289 within 0.15 with LOW confidence (vertices may give a "
                              "steeper growth)",
            "strategies": "both graded strategies solve with min det F > 0; LOW confidence "
                          "(the reduced rule failed on a strongly graded mesh in cylinder3d)",
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
    out = {}
    for mode in ("default", "posthoc"):
        if mode == "posthoc":
            os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        else:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        R = rules.__wrapped__("clamped_block_3d")
        rows, hits, npred = {}, 0, 0
        for q, pr in pre["rule_predictions"].items():
            ov, ev = R["combine"][q][:2]
            sg = R["v"][q].get("singular_order", {})
            ml = R["v"][q].get("qoi_convergence_multilevel", {})
            row = {"verdict": ov, "evidence": ev, "fit_order": ml.get("fit_order"),
                   "convergence": ml.get("evidence"), "convergence_verdict": ml.get("verdict"),
                   "singular_order": {k: sg.get(k) for k in ("verdict", "observed", "expected",
                                                             "label")}}
            if pr["predicted"] is not None:
                npred += 1
                row["rule_hit"] = pr["predicted"] == ov
                hits += row["rule_hit"]
            rows[q] = row
        out[mode] = {"rule_hits": hits, "rule_predictions": npred, "qois": rows}
    os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    ad = R["adapter"]
    P = out["posthoc"]["qois"]
    lam = pre["edge_exponent_lam"]
    sing = lambda q: P[q]["singular_order"]["verdict"] == "PASS"
    conv = lambda q, ok: P[q]["convergence"] in ok and P[q]["convergence_verdict"] == "PASS"
    import numpy as np
    hs, pv, _ = ad.series("peak_von_mises")
    slope = float(np.polyfit(np.log(hs[-4:]), np.log(pv[-4:]), 1)[0])
    exp = {
        "default: every singular_order INCONCLUSIVE": all(
            r["singular_order"]["verdict"] in ("INCONCLUSIVE", None)
            for r in out["default"]["qois"].values()),
        "strain_energy singular_order PASS and rate": sing("strain_energy")
        and conv("strain_energy", ("rate",)),
        "w_tip rate, order >= 1.27": conv("w_tip", ("rate",))
        and (P["w_tip"]["fit_order"] or 0) >= 1.27,
        "mid_gk_avg_sxx rate and singular_order PASS": conv("mid_gk_avg_sxx", ("rate",))
        and sing("mid_gk_avg_sxx"),
        "root_gk_avg_von_mises PASS (band or rate)": conv("root_gk_avg_von_mises",
                                                          ("rate", "band")),
        "peak convergence FAIL": P["peak_von_mises"]["convergence_verdict"] != "PASS",
        "peak at the clamped face on the finest three levels": all(
            r["peak_location"][0] < 0.1 for r in ad.rows[-3:]),
        "peak slope negative": slope < 0,
        "peak slope = lam - 1 within 0.15 (low confidence)": abs(slope - (lam - 1)) <= 0.15,
        "graded strategies min det F > 0": all((r.get("min_detF") or -1) > 0
                                               for r in ad.strategy_rows if r["grading"] != 1.0),
    }
    res = {"levels_k": [r["k"] for r in ad.rows], "n_dof": [r["n_dof"] for r in ad.rows],
           "lam": lam, "peak_slope_finest4": slope, "modes": out,
           "developer_expectations_met": exp,
           "strategies": [{k: r.get(k) for k in ("grading", "min_detF", "min_detF_location",
                                                  "strain_energy", "peak_von_mises")}
                          for r in ad.strategy_rows]}
    json.dump(res, open(os.path.join(HERE, "block3d_result.json"), "w"), indent=1)
    for mode, o in out.items():
        print(f"{mode}: rule predictions {o['rule_hits']}/{o['rule_predictions']}")
    print(f"developer expectations {sum(exp.values())}/{len(exp)}; peak slope {slope:.3f} "
          f"(lam - 1 = {lam - 1:.3f})")
    for k, v in exp.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    for q, r in P.items():
        print(f"  {q:<24} {r['verdict']}/{r['evidence']:<5} p={r['fit_order']} "
              f"singular {r['singular_order']}")
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    compare() if "--compare" in sys.argv else main()
