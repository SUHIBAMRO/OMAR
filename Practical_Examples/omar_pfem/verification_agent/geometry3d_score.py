"""Score the pre-registered 3D predictions (geometry3d_preregistration.json, commit 46e4ab2)
against the stored runs.

    python -m omar_pfem.verification_agent.geometry3d_score
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _j(*p):
    return json.load(open(os.path.join(HERE, *p)))


def _rows(name, key):
    return sorted(_j("pipeline2d_runs", name, "ladders.json").get(key, []), key=lambda r: r["n"])


def _uabs(c):
    return abs(c.get("U_rel", c["U"])) * abs(c["limit"])


def score():
    from omar_pfem.verification_agent.geometry2d import make
    pe3, pe2, ck3 = (_j("autonomous_runs", n, "run.json") for n in
                     ("plate_hole_quarter_3d_pe", "plate_hole_quarter_pe", "cook_3d"))
    ck2 = _j("autonomous_runs", "cook_linear", "run.json")
    T = 1.0
    out = {}
    out["P1_mms"] = (all(r["verdict"] == "PASS" for run in (pe3, ck3) for r in run["mms"].values()),
                     {n: {el: (r["verdict"], r.get("rates", {}).get("L2", {}).get("observed"),
                               r.get("rates", {}).get("H1", {}).get("observed"), r.get("levels"))
                          for el, r in run["mms"].items()} for n, run in (("plate", pe3), ("cook", ck3))})
    a, b = _rows("plate_hole_quarter_3d_pe", "Q2 uniform"), _rows("plate_hole_quarter_pe", "Q2 uniform")
    b = {r["n"]: r for r in b}
    dev = {}
    for r in a:
        if r["n"] in b and r["n"] <= 16:
            for q, s in (("strain_energy", T), ("u_right", 1), ("v_hole_top", 1), ("peak_von_mises", 1)):
                dev.setdefault(q, []).append(abs(r[q] / s / b[r["n"]][q] - 1))
    out["P2_planestrain_identity"] = (bool(dev) and all(max(v) < 1e-8 for v in dev.values()),
                                      {q: max(v) for q, v in dev.items()})
    c3, c2 = pe3.get("certified_info", {}), pe2.get("certified_info", {})
    gaps = {}
    for q, c in c3.items():
        s = T if q == "strain_energy" else 1
        if q in c2:
            gaps[q] = (abs(c["limit"] / s - c2[q]["limit"]), _uabs(c) / s + _uabs(c2[q]))
    out["P3_planestrain_limits"] = (bool(gaps) and all(g <= u for g, u in gaps.values()), gaps)
    p3 = make(pe3.get("description") or _j("problems3d", "plate_hole_quarter_3d_pe.json"))
    out["P4_plate_loop"] = (not p3.corners and not p3.face_edges and len(pe3["final"]["certified"]) >= 3,
                            {"corners": len(p3.corners), "face_edges": len(p3.face_edges),
                             "certified": pe3["final"]["certified"]})
    k3 = make(ck3.get("description") or _j("problems3d", "cook_3d.json"))
    k2 = make({k: v for k, v in ck2["description"].items() if k != "assumed"})
    loc = lambda P: sorted((round(c["x"] * P.a, 6), round(c["y"] * P.a, 6)) for c in P.corners)
    fe = [e for e in k3.face_edges if e["side"] == "clamped" and min(e["rate_limiting"]) < 1]
    out["P5_cook_edges"] = (loc(k3) == loc(k2) and len(fe) == 2,
                            {"3d": loc(k3), "2d": loc(k2),
                             "face_edges": [(e["face"], e["side"], e["rate_limiting"]) for e in k3.face_edges]})
    ci = ck3.get("certified_info", {})
    rows = _rows("cook_3d", "Q2 uniform") if os.path.exists(
        os.path.join(HERE, "pipeline2d_runs", "cook_3d", "ladders.json")) else []
    if not rows and not ci:
        sup = _j("cook_3d_supplementary.json")["rows"]["Q2_16"]
        out["P6_cook_values"] = (False, {"not_evaluable": "the loop stopped at code verification; "
                                         "no certified value and no ladder",
                                         "supplementary_outside_loop_Q2_16": sup,
                                         "supplementary_rel_v_C": sup["v_C"] / 23.96 - 1,
                                         "supplementary_rel_energy": sup["strain_energy"] / 12.02 - 1})
    else:
        fin = rows[-1]
        vC = ci["v_C"]["limit"] if "v_C" in ci else fin["v_C"]
        en = ci["strain_energy"]["limit"] if "strain_energy" in ci else fin["strain_energy"]
        out["P6_cook_values"] = (abs(vC / 23.96 - 1) < 0.02 and abs(en / T / 12.02 - 1) < 0.02,
                                 {"v_C": vC, "rel_v_C": vC / 23.96 - 1, "energy": en,
                                  "rel_energy": en / T / 12.02 - 1})
    concl = [e for e in ck3.get("experiments", []) if e.get("quantity") == "peak_von_mises"]
    out["P7_cook_peak"] = (bool(concl) and "not verifiable" in concl[0]["conclusion"],
                           concl[0] if concl else "not reached: the loop stopped at code verification")
    try:
        k3.build_mesh(32, None, "Q2")
        refused, why = False, "built"
    except MemoryError as e:
        refused, why = True, str(e)
    out["P8_memory_guard"] = (refused, why)
    return out


if __name__ == "__main__":
    res = score()
    for k, (ok, ev) in res.items():
        print(f"{'HIT ' if ok else 'MISS'} {k}: {json.dumps(ev, default=float)[:400]}")
    print(f"{sum(ok for ok, _ in res.values())} of {len(res)} predictions confirmed")
    json.dump({k: {"confirmed": ok, "evidence": ev} for k, (ok, ev) in res.items()},
              open(os.path.join(HERE, "geometry3d_score.json"), "w"), indent=1, default=float)
