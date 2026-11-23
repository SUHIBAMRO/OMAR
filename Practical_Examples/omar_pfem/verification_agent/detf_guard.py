"""Admissibility guard for stored solves (2026-11-23, POST HOC: added after the
inverted g = 2 strategy of cylinder3d was found; no verdict rule is changed).

Finding: a Newton solve with too few load increments can converge to an inverted state
(min det F <= 0) without any error being raised (cylinder3d, g = 2.0, 10 increments:
min det F = -1.03, errors 13-27%). The full 3x3x3 rule does not prevent it; 40
increments do (graded_q20_study.json). The agent's rules had no check for this.

This guard
  1. scans every stored solve of every adapter (family levels and strategy meshes) and
     lists those with min det F <= 0 as INADMISSIBLE,
  2. states for each whether any verdict check consumed it (the strategy checks are the
     only consumers of strategy rows),
  3. records the remedy and, where the re-solve exists (graded_q20_study.json), its result.
guarded() implements that rule for the 3D solvers behind VERIF_POSTHOC_DETF=1.
The rule it proposes, for future runs (not applied to the frozen pre-registered scoring):
a solve with min det F <= 0 is re-solved with four times the increments; if it stays
inverted, it is excluded and every check that needs it is INCONCLUSIVE.

    python -m omar_pfem.verification_agent.detf_guard
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "detf_guard.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape", "edge_crack", "thick_cylinder_3d", "clamped_block_3d")
STRATEGY_CHECKS = ("strategy_robustness", "anchored_robustness")


def guarded(solve_fn, *args, nsteps=10, **kw):
    """POST HOC switch VERIF_POSTHOC_DETF=1 (off by default, so every stored and
    pre-registered result is unchanged): a converged solve with min det F <= 0 is
    re-solved with four times the increments; if it is still inverted, the row is marked
    inadmissible ("detf_inadmissible": True) so that no check may use it. The attempt is
    recorded in row["detf_guard"]."""
    row = solve_fn(*args, nsteps=nsteps, **kw)
    if os.environ.get("VERIF_POSTHOC_DETF") != "1" or not (row["min_detF"] <= 0):
        return row
    first = row["min_detF"]
    row = solve_fn(*args, nsteps=4 * nsteps, **kw)
    row["detf_guard"] = {"first_min_detF": first, "first_nsteps": nsteps,
                         "resolved_nsteps": 4 * nsteps, "min_detF": row["min_detF"]}
    row["detf_inadmissible"] = not (row["min_detF"] > 0)
    return row


GUARDED_RESOLVES = {"thick_cylinder_3d": "cylinder3d_strategy_guarded.json"}


def admissible_strategies(adapter, qoi, s):
    """POST HOC, round M (2026-11-23), active only with VERIF_POSTHOC_DETF=1: the check-level
    half of the guard. Round J guarded the solver and scanned the stored solves, but a plan
    chosen by the LLM (billed ablation, round M) ran the strategy checks on the cylinder and
    consumed the stored inverted g = 2 row. Here every strategy row with min det F <= 0 is
    replaced by its guarded re-solve (GUARDED_RESOLVES, four times the increments) when that
    exists and is admissible; otherwise None is returned and the check is INCONCLUSIVE.
    `s` is adapter.strategies(qoi); returns (values, scale), None, or a reason string."""
    rows = getattr(adapter, "strategy_rows", None)
    if s is None or not rows or len(rows) != len(s[0]):
        return s
    vals = list(s[0])
    for i, r in enumerate(rows):
        d = _det(r)
        if d is None or d > 0:
            continue
        fn = GUARDED_RESOLVES.get(getattr(adapter, "geometry", None))
        alt = None
        if fn and os.path.exists(os.path.join(HERE, fn)):
            alt = next((a for a in json.load(open(os.path.join(HERE, fn)))
                        if a.get("k") == r.get("k") and a.get("grading") == r.get("grading")
                        and _det(a) is not None and _det(a) > 0), None)
        if alt is None or alt.get(qoi) is None:
            return (f"strategy solve with min det F = {d:.3g} <= 0 is inadmissible and has no "
                    "admissible re-solve (VERIF_POSTHOC_DETF)")
        vals[i] = alt[qoi]
    return vals, s[1]


def _det(r):
    for k in ("min_detF", "posthoc_min_detF"):
        if r.get(k) is not None:
            return r[k]
    return None


def scan():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent.verify_report import rules
    out, recorded, missing = [], 0, []
    for geo in PROBLEMS:
        R = rules.__wrapped__(geo)
        ad = R["adapter"]
        fam = getattr(ad, "rows", None) or []
        if not any(_det(r) is not None for r in fam):       # B3 keeps them in .family
            fam = list(getattr(ad, "family", None) or [])
        strat = list(getattr(ad, "strategy_rows", None) or [])
        strat += [z for _, z in getattr(ad, "zbeta", [])]   # B3 local-refinement meshes
        used = sorted(q for q, v in R["v"].items() if any(c in v for c in STRATEGY_CHECKS))
        n_with = 0
        for role, rows in (("family", fam), ("strategy", strat)):
            for r in rows:
                d = _det(r)
                if d is None:
                    continue
                n_with += 1
                if d <= 0:
                    out.append({"problem": geo, "role": role, "min_detF": d,
                                "grading": r.get("grading"), "level": r.get("k", r.get("n")),
                                "consumed_by_a_verdict": role == "family" or bool(used),
                                "strategy_checks_on": used})
        recorded += n_with
        if not n_with:
            missing.append(geo)
    return out, recorded, missing


def main():
    bad, recorded, missing = scan()
    gq = json.load(open(os.path.join(HERE, "graded_q20_study.json")))
    resolve = [r for r in gq if r["grading"] == 2.0]
    res = {"posthoc": True, "solves_with_min_detF_recorded": recorded,
           "problems_without_min_detF_records": missing,
           "inadmissible": bad,
           "remedy_tested": {
               "case": "thick_cylinder_3d, k = 4, g = 2.0",
               "rows": [{k: r.get(k) for k in ("quadrature", "nsteps", "min_detF", "errors")}
                        for r in resolve],
               "conclusion": "10 increments: inverted with both rules (min det F -1.03 reduced, "
                             "-0.74 full); 40 increments: min det F 1.004, every global "
                             "quantity within 1.2e-5 of the exact value (reduced 2.6e-7)"},
           "proposed_rule_for_future_runs": "min det F <= 0 -> re-solve with 4x increments; "
                                            "still <= 0 -> exclude, dependent checks "
                                            "INCONCLUSIVE (not applied to the frozen scoring)"}
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "remedy_tested"}, indent=1))
