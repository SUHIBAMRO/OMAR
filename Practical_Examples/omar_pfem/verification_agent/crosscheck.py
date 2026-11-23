"""Rule: a rate-certified uncertainty is FINAL only if an independent
discretization agrees with the certified limit within the stated uncertainties.

Why (2026-11-22): the B3 Gaussian groove averages were certified with 0.11-0.18%
of the stress scale from the Q20 family, yet an independent spline
discretization lies about 2% (of the values) away. Convergence of one family
bounds that family's sequence, not the distance to the continuum value.

Two versions are computed and both are stored:

  v1 (2026-11-22, post hoc after the 2% finding): the independent value is the
     finest level of the independent sequence, its uncertainty the change over
     its last step.
  v2 (2026-11-23, post hoc refinement, stricter): the independent sequence must
     itself pass the agent's multilevel convergence check with RATE evidence,
     with the same band and GCI tolerances as the certified quantity and the
     order window of the independent element (quadratic: 0.5-5), and its own
     uncertainty must meet the quantity's tolerance (verdict PASS; otherwise a
     loose independent interval would "agree" with anything); its fit limit
     and fit uncertainty are then compared. An independent sequence with only
     band or no evidence gives `independent_unconverged`. Reason for v2: the
     spline Gaussian sequences behind the v1 "contradicted" verdicts do not
     themselves converge with a rate (three-level orders 0.47 and 0.75), so the
     last change does not bound their error and the v1 verdict was not sound in
     either direction.

Final status (v2), reported next to the agent's family verdict, never replacing it:
  CERTIFIED       cross_checked: |limit - independent| <= U + U_ind
  INCONCLUSIVE    contradicted | independent_unconverged | single_family
                  (an unresolved case is not a certification)

Independent sources:
  unit_square, quarter_ring  second FE code, Q2 (independent_solver.py; B1 Gaussian
                             average from crosscheck_extra.py)
  cook_membrane              same scikit-fem code, Q2 instead of Q4 (crosscheck_extra.py):
                             an independent discretization, not an independent code
  open_hole_lamina           splines p = 3, exact geometry (iga_lamina.py)
  thick_cylinder_3d          the exact 1D solution (cylinder3d.py; scoring only, never used
                             by the agent's verdicts)
  bushing_groove             moment, energy: splines p = 2 (iga_b3.py, displacement);
                             Gaussian averages: spline P1 and spline displacement; the
                             cylindrical zz average is identical to sigma_zz; rr and tt
                             from the local spline re-run (iga_b3_cyl_local.py ->
                             iga_b3_cyl_result.json, added 2026-11-23, round J)
  l_shape, edge_crack,       6-node triangles P2, the same problem code with each Q2
  plate_hole                 quadrilateral split in two (independent_p2.py, round J;
                             plate: hole mid-edge nodes on the exact circle)
  clamped_block_3d           8-node hexahedra Q8, k = 2..16 (block3d.run_q8, round J;
                             order window of a linear element 0.5-3); and a second code,
                             scikit-fem with 27-node hexahedra (independent_block3d.py). Its certifications
                             exist only with the post hoc singular-order extension, so
                             they are in rows_posthoc_only (below), not in the summary

Round J (2026-11-23): quantities certified only WITH the post hoc singular-order
extension (VERIF_POSTHOC_SINGULAR=1) are cross-checked too, in a separate list
rows_posthoc_only with its own summary; the main summary keeps its old meaning.

    python -m omar_pfem.verification_agent.crosscheck
"""
import json
import os

import numpy as np

from omar_pfem.verification_agent import gates

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "crosscheck_result.json")
PROBLEMS = ("unit_square", "quarter_ring", "bushing_groove", "cook_membrane", "plate_hole",
            "open_hole_lamina", "l_shape", "edge_crack", "thick_cylinder_3d",
            "clamped_block_3d")
QUAD_WINDOW = (0.5, 5)
LIN_WINDOW = (0.5, 3)


def _j(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def independent_series(geometry, qoi):
    """[(source, hs, values)] of independent sequences for this quantity (may be empty)."""
    out = []
    if geometry in ("unit_square", "quarter_ring"):
        tag = "b1" if geometry == "unit_square" else "b2"
        d = dict(_j("independent_solver_result.json"))
        d.update({k: v for k, v in _j("independent_extra_result.json").items()
                  if k.startswith(f"{tag}:")})
        rows = sorted((v for k, v in d.items() if k.startswith(f"{tag}:q2:") and qoi in v),
                      key=lambda r: r["N"])
        if rows:
            out.append(("second FE code, Q2", [1.0 / r["N"] for r in rows],
                        [r[qoi] for r in rows]))
    elif geometry == "cook_membrane":
        d = _j("independent_extra_result.json")
        rows = sorted((v for k, v in d.items() if k.startswith("cook:q2:") and qoi in v),
                      key=lambda r: r["N"])
        if rows:
            out.append(("Q2, same code (independent discretization)",
                        [r["h"] for r in rows], [r[qoi] for r in rows]))
    elif geometry == "open_hole_lamina":
        d = _j("iga_lamina_result.json").get("iga", [])
        rows = sorted((r for r in d if r["p"] == 3 and r["beta"] == 2.8 and qoi in r),
                      key=lambda r: r["m"])
        if rows:
            out.append(("splines p = 3, exact geometry", [1.0 / r["m"] for r in rows],
                        [r[qoi] for r in rows]))
    elif geometry in ("l_shape", "edge_crack", "plate_hole"):
        name = {"l_shape": "lshape", "edge_crack": "crack"}.get(geometry, geometry)
        rows = sorted(_j(f"{name}_p2_ladder.json"), key=lambda r: r.get("n", r.get("m")))
        rows = [r for r in rows if qoi in r]
        if rows:
            out.append(("6-node triangles P2 (independent discretization)",
                        [r["h"] for r in rows], [r[qoi] for r in rows]))
    elif geometry == "clamped_block_3d":
        rows = sorted(_j("block3d_q8_ladder.json"), key=lambda r: r["k"])
        rows = [r for r in rows if qoi in r]
        if rows:
            out.append(("8-node hexahedra Q8 (independent discretization)",
                        [r["h"] for r in rows], [r[qoi] for r in rows]))
        rows = [r for r in sorted(_j("block3d_hex27_ladder.json"), key=lambda r: r["k"])
                if qoi in r]
        if rows:
            out.append(("second code: scikit-fem, 27-node hexahedra",
                        [r["h"] for r in rows], [r[qoi] for r in rows]))
    elif geometry == "thick_cylinder_3d":
        ex = _j("cylinder3d_exact.json")
        if qoi in ex and isinstance(ex[qoi], float):
            out.append(("exact 1D solution", "exact", ex[qoi]))
    elif geometry == "bushing_groove":
        d = _j("iga_b3_result.json")
        for key, extra in _j("iga_b3_cyl_result.json").items():   # rr, tt (round J)
            if key in d:
                d[key] = dict(d[key], **{k: v for k, v in extra.items() if "cyl_" in k})
        if qoi in ("reaction_moment_y", "strain_energy"):
            keys = [("splines p = 2, displacement", "")]
            name = qoi
        elif qoi.startswith("gk2_"):
            comp = qoi[len("gk2_"):]
            if comp == "cyl_zz_avg":
                comp = "sigma_zz_avg"            # identical by definition (verify_report 8.3)
            name = "gk2trunc_" + comp
            keys = [("spline P1", ":fbar_p1"), ("splines p = 2, displacement", "")]
        else:
            return out
        for src, suffix in keys:
            rows = [d.get(f"L{lv}:p2{suffix}") for lv in (1, 2, 3, 4)]
            rows = [r for r in rows if r is not None and name in r]
            if len(rows) >= 2:
                out.append((src, [r["n_elem"] ** (-1.0 / 3.0) for r in rows],
                            [r[name] for r in rows]))
    return out


def independent_v1(geometry, qoi):
    """v1 exactly as on 2026-11-22 (kept to reproduce the v1 verdicts)."""
    """(value, uncertainty as an absolute number, source) or None."""
    if geometry in ("unit_square", "quarter_ring"):
        tag = "b1" if geometry == "unit_square" else "b2"
        d = _j("independent_solver_result.json")
        rows = sorted((v for k, v in d.items() if k.startswith(f"{tag}:q2:") and qoi in v),
                      key=lambda r: r["N"])
        if len(rows) < 2:
            return None
        return rows[-1][qoi], abs(rows[-1][qoi] - rows[-2][qoi]), "second FE code, Q2"
    if geometry == "open_hole_lamina":
        d = _j("iga_lamina_result.json")["iga"]
        pick = lambda m: next((r for r in d if r["p"] == 3 and r["m"] == m and r["beta"] == 2.8
                               and qoi in r), None)
        a, b = pick(16), pick(8)
        if a is None or b is None:
            return None
        return a[qoi], abs(a[qoi] - b[qoi]), "splines p = 3, exact geometry"
    if geometry == "bushing_groove":
        d = _j("iga_b3_result.json")
        for key, extra in _j("iga_b3_cyl_result.json").items():   # rr, tt (round J)
            if key in d:
                d[key] = dict(d[key], **{k: v for k, v in extra.items() if "cyl_" in k})
        if qoi in ("reaction_moment_y", "strain_energy"):
            rows = [d[f"L{l}:p2"] for l in (1, 2, 3, 4)]
            g = gates.multilevel_convergence_gate([1.0 / r["spans"][0] for r in rows],
                                                  [r[qoi] for r in rows], band_tol=0.01,
                                                  gci_tol=0.02, order_window=(0.5, 5))
            lim = g.get("fit_limit")
            return lim, g.get("uncertainty_fine", 0.0) * abs(rows[-1][qoi]), "splines p = 2"
        if qoi.startswith("gk2_"):
            key = "gk2trunc_" + qoi[len("gk2_"):]
            if f"L4:p2:fbar_p1" not in d or key not in d["L4:p2:fbar_p1"]:
                return None
            a, b = d["L4:p2:fbar_p1"][key], d["L3:p2:fbar_p1"][key]
            return a, abs(a - b), "spline P1, finest level (not converged)"
    return None


def decide(limit, U_rel, independent_value, U_ind_abs, scale):
    """The rule itself: cross_checked iff the gap is within both uncertainties."""
    if independent_value is None:
        return "single_family", None
    gap = abs(limit - independent_value) / scale
    return ("cross_checked" if gap <= U_rel + U_ind_abs / scale else "contradicted"), gap


def assess_independent(hs, vals, ml, scale, window=QUAD_WINDOW):
    """v2: run the agent's multilevel check on the independent sequence."""
    if len(vals) < 4:
        return {"evidence": "too_few_levels", "n_levels": len(vals)}
    g = gates.multilevel_convergence_gate(list(hs), list(vals), band_tol=ml["band_tol"],
                                          gci_tol=ml["gci_tol"], scale=scale,
                                          order_window=window)
    return {k: g.get(k) for k in ("evidence", "fit_limit", "fit_order", "uncertainty_fine",
                                  "verdict", "reason", "n_levels")}


def agent_limit(R, qoi):
    v = R["v"][qoi]
    ml = v.get("qoi_convergence_multilevel", {})
    if ml.get("evidence") == "rate":
        return ml["fit_limit"], ml["uncertainty_fine"], ml
    c3 = v.get("qoi_convergence", {})
    if c3.get("evidence") == "rate":
        return c3["richardson_extrapolate"], c3["gci_fine"], ml
    return None


def _row(geo, q, R, ad):
    al = agent_limit(R, q)
    if al is None:
        return None
    lim, U, ml = al
    _, vals, scale = ad.series(q)
    sc = scale if scale else abs(vals[-1])
    row = {"problem": geo, "qoi": q, "limit": lim, "U_rel": U, "scale": sc,
           "independent_sources": []}
    series = independent_series(geo, q)
    # v1: exactly the rule of 2026-11-22
    ind = independent_v1(geo, q)
    if ind is None or ind[0] is None:
        row["v1"] = {"status": "single_family"}
    else:
        st, gap = decide(lim, U, ind[0], ind[1], sc)
        row["v1"] = {"source": ind[2], "independent": ind[0], "U_ind_rel": ind[1] / sc,
                     "gap_rel": gap, "status": st}
    # v2: every source assessed; the quantity is cross_checked if a converged
    # source agrees, contradicted if a converged source disagrees (and none agrees)
    statuses = []
    for src, hs, vs in series:
        if hs == "exact":                  # an exact value: no uncertainty of its own
            st, gap = decide(lim, U, vs, 0.0, sc)
            row["independent_sources"].append(
                {"source": src, "levels": 0, "finest": vs, "assessment":
                 {"evidence": "exact"}, "status": st, "gap_rel": gap, "U_ind_rel": 0.0})
            statuses.append(st)
            continue
        a = assess_independent(hs, vs, ml if ml else {"band_tol": 0.01, "gci_tol": 0.05},
                               sc, LIN_WINDOW if "Q8" in src else QUAD_WINDOW)
        ent = {"source": src, "levels": len(vs), "finest": vs[-1], "assessment": a}
        if a.get("evidence") == "rate" and a.get("verdict") == "PASS":
            st, gap = decide(lim, U, a["fit_limit"], a["uncertainty_fine"] * sc, sc)
            ent.update(status=st, gap_rel=gap, U_ind_rel=a["uncertainty_fine"])
        else:
            ent.update(status="independent_unconverged",
                       gap_finest_rel=abs(lim - vs[-1]) / sc)
        row["independent_sources"].append(ent)
        statuses.append(ent["status"])
    if not series:
        row["status"] = "single_family"
    elif "cross_checked" in statuses:
        row["status"] = ("cross_checked" if "contradicted" not in statuses
                         else "conflicting_sources")
    elif "contradicted" in statuses:
        row["status"] = "contradicted"
    else:
        row["status"] = "independent_unconverged"
    row["final"] = "CERTIFIED" if row["status"] == "cross_checked" else "INCONCLUSIVE"
    return row


STATUSES = ("cross_checked", "contradicted", "independent_unconverged", "conflicting_sources",
            "single_family")


def run():
    from omar_pfem.verification_agent.verify_report import rules
    rows, rows_ph = [], []
    for geo in PROBLEMS:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        R = rules.__wrapped__(geo)
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        try:
            Rp = rules.__wrapped__(geo)
        finally:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        for q, comb in R["combine"].items():
            if comb[0] == "PASS" and comb[1] == "rate":
                r = _row(geo, q, R, R["adapter"])
                if r:
                    rows.append(r)
            elif Rp["combine"][q][0] == "PASS" and Rp["combine"][q][1] == "rate":
                r = _row(geo, q, Rp, Rp["adapter"])
                if r:
                    rows_ph.append(r)
    summary = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    summary_v1 = {s: sum(r["v1"]["status"] == s for r in rows)
                  for s in ("cross_checked", "contradicted", "single_family")}
    res = {"summary": summary, "summary_v1": summary_v1,
           "final": {s: sum(r["final"] == s for r in rows) for s in ("CERTIFIED",
                                                                       "INCONCLUSIVE")},
           "rows": rows,
           "summary_posthoc_only": {s: sum(r["status"] == s for r in rows_ph) for s in STATUSES},
           "rows_posthoc_only": rows_ph}
    json.dump(res, open(OUT, "w"), indent=1)
    return res

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    res = run()
    print("v2", res["summary"], "final", res["final"])
    print("v1", res["summary_v1"])
    print("posthoc-only", res["summary_posthoc_only"])
    for r in res["rows"] + res["rows_posthoc_only"]:
        srcs = "; ".join(f"{s['source']}: {s['status']} ({s['assessment'].get('evidence')})"
                         for s in r["independent_sources"]) or "-"
        print(f"{r['problem']:<17}{r['qoi']:<24}{r['status']:<24} v1 {r['v1']['status']:<14} {srcs}")
