"""Score the pre-registered sensitivity predictions (sensitivity_preregistration.json) against
the stored runs (sensitivity_runs/*.json). Uses the pre-registered estimator S; the post hoc
S_log is printed alongside for information only.

    python -m omar_pfem.verification_agent.sensitivity_score
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(name):
    return json.load(open(os.path.join(HERE, "sensitivity_runs", name + ".json")))["params"]


def score():
    pl, ck, nh = _run("plate_hole_quarter"), _run("cook_linear"), _run("nh_l_bracket_demo")
    S = lambda r, p, q: r[p]["quantities"][q]["S"]
    out = {}
    disp = ["u_right", "v_hole_top", "strain_energy"]
    out["P1_plate_E"] = (all(abs(S(pl, "E", q) + 1) < 1e-6 for q in disp)
                         and abs(S(pl, "E", "peak_von_mises")) < 1e-6,
                         {q: S(pl, "E", q) for q in disp + ["peak_von_mises"]})
    t = "bcs[2].t"
    out["P2_plate_load"] = (all(abs(S(pl, t, q) - 1) < 1e-6 for q in
                                ("u_right", "v_hole_top", "peak_von_mises"))
                            and abs(S(pl, t, "strain_energy") - 2) < 1e-6,
                            {q: S(pl, t, q) for q in pl[t]["quantities"]})
    out["P3_plate_nu_michell"] = (abs(S(pl, "nu", "peak_von_mises")) < 1e-3
                                  and abs(S(pl, "nu", "u_right")) > 0.01
                                  and abs(S(pl, "nu", "v_hole_top")) > 0.01,
                                  {q: S(pl, "nu", q) for q in pl["nu"]["quantities"]})
    r = pl["hole[0].radius"]["quantities"]["peak_von_mises"]
    out["P4_plate_radius"] = (r["change_rel"] < 0.03, {"change_rel": r["change_rel"], "S": r["S"]})
    cq = list(ck["E"]["quantities"])
    corners = [q for q in cq if q.startswith("corner")]
    out["P5_cook_E"] = (all(abs(S(ck, "E", q) + 1) < 1e-6 for q in ("v_C", "strain_energy") if q in cq)
                        and all(abs(S(ck, "E", q)) < 1e-6 for q in corners),
                        {q: S(ck, "E", q) for q in cq})
    out["P6_cook_nu"] = (bool(corners) and all(abs(S(ck, "nu", q)) > 1e-3 for q in corners),
                         {q: S(ck, "nu", q) for q in corners})
    t = "bcs[1].t"
    nq = list(nh["E"]["quantities"])
    out["P7_nh_identity"] = (all(abs(S(nh, "E", q) + S(nh, t, q)) < 0.03 for q in nq)
                             and abs(S(nh, t, "strain_energy") - 2) > 0.02,
                             {q: (S(nh, "E", q), S(nh, t, q)) for q in nq})
    lin = [(n, p, r.get("levels")) for n, run in (("plate", pl), ("cook", ck))
           for p, r in run.items()]
    out["P8_selfcheck"] = (all(lv == [8, 16] for _, _, lv in lin), lin)
    return out


if __name__ == "__main__":
    res = score()
    for k, (ok, ev) in res.items():
        print(f"{'HIT ' if ok else 'MISS'} {k}: {json.dumps(ev, default=float)[:300]}")
    print(f"{sum(ok for ok, _ in res.values())} of {len(res)} predictions confirmed")
    json.dump({k: {"confirmed": ok, "evidence": ev} for k, (ok, ev) in res.items()},
              open(os.path.join(HERE, "sensitivity_runs", "score.json"), "w"), indent=1,
              default=float)
