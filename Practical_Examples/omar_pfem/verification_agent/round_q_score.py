"""Score the round Q predictions (round_q_preregistration.json, commit 3020da3) from the stored
runs.

    python -m omar_pfem.verification_agent.round_q_score
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _j(*p):
    return json.load(open(os.path.join(HERE, *p)))


def _run(n):
    f = os.path.join(HERE, "autonomous_runs", n, "run.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def _rows(n, key):
    f = os.path.join(HERE, "pipeline2d_runs", n, "ladders.json")
    return {r["n"]: r for r in (json.load(open(f)).get(key, []) if os.path.exists(f) else [])}


def score():
    from omar_pfem.verification_agent import pipeline2d as P
    from omar_pfem.verification_agent.geometry2d import make
    ck, pl, nh3, nh2 = (_run(n) for n in ("cook_3d", "plate_hole_quarter_3d_pe",
                                          "nh_plate_hole_3d_pe", "nh_plate_hole_pe"))
    out = {}
    prod = {el: r.get("verdict") for el, r in ck.get("mms", {}).items()}
    # the production verdicts as checked (before R2 relabels them PASS in the run record)
    rec = _j("code_verification_record.json")
    from omar_pfem.verification_agent import mms_auto
    pc = make(_j("problems3d", "cook_3d.json"))
    prod = {el: rec.get(mms_auto.key(pc, el), {}).get("verdict") for el in ("Q2", "P2")}
    out["Q1_cook_classes"] = (all(v == "INCONCLUSIVE" for v in prod.values()), prod)
    ref = {el: r.get("verdict") for el, r in ck.get("mms_reference_domain", {}).get("checks", {}).items()}
    ci = ck.get("certified_info", {})
    vC = ci.get("v_C", {}).get("limit")
    out["Q2_cook_reference"] = (bool(ref) and all(v == "PASS" for v in ref.values()) and vC is not None
                                and abs(vC / 23.96 - 1) < 0.01,
                                {"reference": ref, "v_C_limit": vC,
                                 "rel": None if vC is None else vC / 23.96 - 1})
    concl = [e for e in ck.get("experiments", []) if e.get("quantity") == "peak_von_mises"]
    out["Q3_cook_peak"] = (bool(concl) and "not verifiable" in concl[0]["conclusion"],
                           concl[0]["conclusion"] if concl else "not reached")
    m = {el: (r.get("verdict"), r.get("reference_pass")) for el, r in nh3.get("mms", {}).items()}
    out["Q4_nh3d_mms"] = (bool(m) and all(v[0] == "PASS" for v in m.values()), m)
    a, b = _rows("nh_plate_hole_3d_pe", "Q2 uniform"), _rows("nh_plate_hole_pe", "Q2 uniform")
    dev = {}
    for n in sorted(set(a) & set(b)):
        if n <= 16:
            for q in ("strain_energy", "u_right", "v_hole_top"):
                dev.setdefault(q, []).append(abs(a[n][q] / b[n][q] - 1))
    out["Q5_nh3d_identity"] = (bool(dev) and all(max(v) < 1e-7 for v in dev.values()),
                               {q: max(v) for q, v in dev.items()} or "no common levels")
    d = _j("problems3d", "nh_plate_hole_3d_pe.json")
    for bc in d["bcs"]:
        if bc["type"] == "traction":
            bc["t"] = [v * 1e-4 for v in bc["t"]]
    n_ = 4
    u_nh = make(d).solve(n_, element="Q2")["u_right"]
    u_li = make(dict(d, material="linear_elastic")).solve(n_, element="Q2")["u_right"]
    out["Q6_nh3d_small_load"] = (abs(u_nh / u_li - 1) < 1e-3,
                                 {"level": n_, "u_nh": u_nh, "u_linear": u_li, "rel": u_nh / u_li - 1})
    cert = (nh3.get("final") or {}).get("certified", [])
    out["Q7_nh3d_loop"] = (len(cert) >= 2, cert)
    jobs = pl.get("cluster_jobs", [])
    persons = [s for s in pl.get("steps", []) if s["actor"] == "person" and "cluster job" in s["detail"]]
    fp = P.fingerprint(make(_j("problems3d", "plate_hole_quarter_3d_pe.json")))
    out["Q8_cluster_job"] = (bool(jobs) and all(j["fingerprint"] == fp and j.get("bundle_md5") for j in jobs)
                             and len(persons) == 1,
                             {"jobs": [{k: j[k] for k in ("name", "bundle_md5", "levels")} for j in jobs],
                              "fingerprint_matches": [j["fingerprint"] == fp for j in jobs],
                              "person_points": len(persons)})
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    res = score()
    for k, (ok, ev) in res.items():
        print(f"{'HIT ' if ok else 'MISS'} {k}: {json.dumps(ev, default=float)[:400]}")
    print(f"{sum(ok for ok, _ in res.values())} of {len(res)} predictions confirmed")
    json.dump({k: {"confirmed": ok, "evidence": ev} for k, (ok, ev) in res.items()},
              open(os.path.join(HERE, "round_q_score.json"), "w"), indent=1, default=float)
