"""Cluster jobs for mesh levels this machine refuses (25 Nov, round Q, rule R3).

When the memory guard refuses a level, the agent does not stop at "too large": it writes a job
that solves exactly the refused levels on the TU Ilmenau cluster, for MANUAL submission (no agent
connects to the cluster). The returned rows are imported into the ladder cache only if

  * the md5 of the returned file matches the one the person prints on the cluster, and
  * the job ran the identical code and description: its fingerprint (pipeline2d.fingerprint)
    equals the local one at import time.

Then the loop is simply run again: the imported levels are taken from the cache.

    python -m omar_pfem.verification_agent.cluster_ladder prepare <description.json> \
           "Q2 uniform:32:Q2" ["P2 uniform:32:P2" ...] [--mem-mb 120000 --hours 24]
    python -m omar_pfem.verification_agent.cluster_ladder run <spec.json>        (on the cluster)
    python -m omar_pfem.verification_agent.cluster_ladder import <returned.json> <md5>
"""
import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "cluster_jobs")


def _prob(desc):
    from omar_pfem.verification_agent.geometry2d import make
    return make({k: v for k, v in desc.items() if k != "assumed"})


def _dirty():
    """Uncommitted changes to the code the job would ship (the bundle is the committed HEAD)."""
    out = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=HERE,
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def prepare(desc, levels, mem_mb=120000, hours=24, log=print):
    """levels: [(ladder key, n, element)]. Writes the spec and the job; returns its record."""
    from omar_pfem.verification_agent import pipeline2d as P
    from omar_pfem.verification_agent.job_factory import make_job
    prob = _prob(desc)
    fp = P.fingerprint(prob)
    name = f"ladder_{desc['name']}_" + "_".join(f"{el}{n}" for _, n, el in levels)
    os.makedirs(JOBS, exist_ok=True)
    spec = {"name": name, "description": desc, "levels": [list(l) for l in levels],
            "fingerprint": fp, "max_dof": 4_000_000, "mem_mb": mem_mb,
            "output": f"cluster_jobs/{name}_result.json",
            "note": "prepared automatically after the local memory guard refused these levels"}
    spec_f = os.path.join(JOBS, f"{name}_spec.json")
    json.dump(spec, open(spec_f, "w"), indent=1)
    dirty = _dirty()
    if dirty:
        # the bundle is the committed tree: the spec must be committed first, and the code must
        # be the committed code, or the fingerprint check on import would fail
        log(f"[cluster] uncommitted code: {dirty[:5]} - commit before submitting this job")
    job = make_job(name, f"-m omar_pfem.verification_agent.cluster_ladder run "
                         f"omar_pfem/verification_agent/cluster_jobs/{name}_spec.json",
                   mem_mb, hours, [spec["output"]], include=("Practical_Examples/omar_pfem",),
                   extra=[spec_f])
    rec = {"name": name, "spec": os.path.relpath(spec_f, HERE), "fingerprint": fp,
           "bundle_md5": job["md5"], "instructions": os.path.relpath(job["instructions"], HERE),
           "levels": spec["levels"], "uncommitted_code": dirty}
    json.dump(rec, open(os.path.join(JOBS, f"{name}_job.json"), "w"), indent=1)
    return rec


def prepare_mms(desc, elements=("Q2", "P2"), levels=(2, 4, 8, 16, 32), reference=True,
                mem_mb=64000, hours=24, log=print):
    """A cluster job for a manufactured-solution check this machine cannot settle (levels
    refused by the memory guard): the production domain and, if it has one, the reference
    domain, every element, the listed levels. Imported into code_verification_record.json
    only under the same record key (same code)."""
    from omar_pfem.verification_agent import mms_auto
    from omar_pfem.verification_agent.job_factory import make_job
    from omar_pfem.verification_agent.reference_domain import reference_description
    checks = [{"description": desc, "suffix": ""}]
    if reference:
        ref, _ = reference_description(desc)
        if ref is not None:
            checks.append({"description": ref, "suffix": mms_auto.REF})
    for ch in checks:
        p = _prob(ch["description"])
        ch["keys"] = {el: mms_auto.key(p, el) + ch["suffix"] for el in elements}
    name = f"mms_{desc['name']}_" + "_".join(elements) + f"_n{max(levels)}"
    os.makedirs(JOBS, exist_ok=True)
    spec = {"kind": "mms", "name": name, "checks": checks, "elements": list(elements),
            "levels": list(levels), "max_dof": 6_000_000, "mem_mb": mem_mb,
            "output": f"cluster_jobs/{name}_result.json",
            "note": "prepared automatically: the manufactured-solution check could not settle "
                    "within this machine's memory"}
    spec_f = os.path.join(JOBS, f"{name}_spec.json")
    json.dump(spec, open(spec_f, "w"), indent=1)
    job = make_job(name, f"-m omar_pfem.verification_agent.cluster_ladder run "
                         f"omar_pfem/verification_agent/cluster_jobs/{name}_spec.json",
                   mem_mb, hours, [spec["output"]], include=("Practical_Examples/omar_pfem",),
                   extra=[spec_f])
    rec = {"name": name, "kind": "mms", "spec": os.path.relpath(spec_f, HERE),
           "keys": [ch["keys"] for ch in checks], "bundle_md5": job["md5"],
           "instructions": os.path.relpath(job["instructions"], HERE), "levels": list(levels),
           "uncommitted_code": _dirty()}
    json.dump(rec, open(os.path.join(JOBS, f"{name}_job.json"), "w"), indent=1)
    return rec


def _run_mms(spec):
    from omar_pfem.verification_agent import mms_auto
    from omar_pfem.verification_agent.gates import mms_rate_gate
    out = {"kind": "mms", "name": spec["name"], "results": {}}
    for ch in spec["checks"]:
        p = _prob(ch["description"])
        mms = p.manufactured() if hasattr(p, "manufactured") else mms_auto.Manufactured(p)
        for el in spec["elements"]:
            k = mms_auto.key(p, el) + ch["suffix"]
            if k != ch["keys"][el]:
                out["results"][ch["keys"][el]] = {"error": "the code on the cluster differs"}
                continue
            rows = []
            for n in spec["levels"]:
                r = p.solve(int(n), None, el, mms=mms)
                if r.get("failed"):
                    break
                rows.append(r)
                print(f"{k} n={n} L2 {r['L2']:.3e} H1 {r['H1']:.3e}", flush=True)
            hs = [r["h"] for r in rows]
            errs = {q: [r[q] for r in rows] for q in mms_auto.EXPECTED}
            g = mms_rate_gate(hs[-3:], {q: e[-3:] for q, e in errs.items()}, mms_auto.EXPECTED, tol=0.2)
            out["results"][k] = {"verdict": mms_auto.classify(g["verdict"], errs),
                                 "gate_verdict": g["verdict"], "rule": "R1 (post hoc, round Q)",
                                 "rates": g["rates"], "levels": [r["n"] for r in rows],
                                 "levels_added": ["computed on the cluster"], "errors": errs,
                                 "field": mms.field, "problem": p.name,
                                 "reason": "cluster job " + spec["name"]}
            json.dump(out, open(os.path.join(HERE, spec["output"]), "w"), indent=1, default=float)
    return out


def run(spec_f):
    """On the cluster: preflight the tools, solve the levels, write the result."""
    import importlib
    spec = json.load(open(spec_f))
    missing = [m for m in ("gmsh", "skfem", "pypardiso", "sympy") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"missing Python packages on the cluster: {missing}. Install them once with:"
                         " pip install --user gmsh==4.13.1 scikit-fem==12.0.2 pypardiso==0.4.7 "
                         "sympy==1.14.0   and submit again.")
    os.environ["VERIF_MAX_DOF"] = str(spec["max_dof"])
    os.environ["VERIF_MAX_GB"] = str(spec.get("max_gb", 0.8 * spec.get("mem_mb", 120000) / 1000))
    if spec.get("kind") == "mms":
        out = _run_mms(spec)
        json.dump(out, open(os.path.join(HERE, spec["output"]), "w"), indent=1, default=float)
        return out
    from omar_pfem.verification_agent import pipeline2d as P
    prob = _prob(spec["description"])
    fp = P.fingerprint(prob)
    out = {"name": spec["name"], "fingerprint": fp, "spec_fingerprint": spec["fingerprint"],
           "rows": {}}
    if fp != spec["fingerprint"]:
        out["error"] = "the code on the cluster differs from the code the job was prepared with"
    else:
        for key, n, el in spec["levels"]:
            r = prob.solve(int(n), None, el)
            out["rows"].setdefault(key, []).append(r)
            print(f"{key} n={n}: " + (f"FAILED {r.get('reason')}" if r.get("failed")
                                      else f"dof {r['n_dof']}"), flush=True)
            json.dump(out, open(os.path.join(HERE, spec["output"]), "w"), indent=1, default=float)
    json.dump(out, open(os.path.join(HERE, spec["output"]), "w"), indent=1, default=float)
    return out


def import_result(returned, md5):
    """Merge the returned rows into the ladder cache after the md5 and fingerprint checks."""
    import hashlib
    from omar_pfem.verification_agent import pipeline2d as P
    got = hashlib.md5(open(returned, "rb").read()).hexdigest()
    if got != md5:
        raise SystemExit(f"md5 mismatch: {got} != {md5} (the copy is not identical)")
    res = json.load(open(returned))
    if res.get("error"):
        raise SystemExit(f"the job reported: {res['error']}")
    if res.get("kind") == "mms":
        return _import_mms(res)
    spec = json.load(open(os.path.join(JOBS, f"{res['name']}_spec.json")))
    prob = _prob(spec["description"])
    if P.fingerprint(prob) != res["fingerprint"]:
        raise SystemExit("the local code or description changed since the job was prepared: the "
                         "rows do not belong to this code and are NOT imported")
    lad = P.Ladders(prob, log=lambda s: None)
    added = []
    for key, rows in res["rows"].items():
        have = {r["n"] for r in lad.cache.setdefault(key, [])}
        for r in rows:
            if r.get("failed"):
                continue
            if r["n"] not in have:
                lad.cache[key].append(r)
                added.append((key, r["n"]))
        lad.cache[key].sort(key=lambda r: r["n"])
    json.dump(lad.cache, open(lad.f, "w"), indent=1)
    return {"added": added, "problem": prob.name}


def batch(names=None, mem_mb=64000, hours=48):
    """Combine pending jobs (cluster_jobs/*_job.json without a returned result) into ONE job:
    one download, one copy, one submission for the person."""
    import datetime
    from omar_pfem.verification_agent.job_factory import make_job
    recs = [json.load(open(os.path.join(JOBS, f))) for f in sorted(os.listdir(JOBS))
            if f.endswith("_job.json") and not f.startswith("batch_")]
    recs = [r for r in recs if (names is None or r["name"] in names) and not os.path.exists(
        os.path.join(JOBS, f"{r['name']}_result.json"))]
    if not recs:
        raise SystemExit("no pending cluster jobs")
    specs = [os.path.join(HERE, r["spec"]) for r in recs]
    outs = [json.load(open(s))["output"] for s in specs]
    name = f"batch_{datetime.date.today().isoformat().replace('-', '')}_{len(recs)}jobs"
    job = make_job(name, "-m omar_pfem.verification_agent.cluster_ladder run-many " + " ".join(
        "omar_pfem/verification_agent/" + os.path.relpath(s, HERE) for s in specs),
                   mem_mb, hours, outs, include=("Practical_Examples/omar_pfem",), extra=specs)
    rec = {"name": name, "kind": "batch", "jobs": [r["name"] for r in recs], "outputs": outs,
           "bundle_md5": job["md5"], "instructions": os.path.relpath(job["instructions"], HERE),
           "uncommitted_code": _dirty()}
    json.dump(rec, open(os.path.join(JOBS, f"{name}_job.json"), "w"), indent=1)
    return rec


def _import_mms(res):
    """Merge returned code-verification results under keys that the LOCAL code reproduces."""
    from omar_pfem.verification_agent import mms_auto
    spec = json.load(open(os.path.join(JOBS, f"{res['name']}_spec.json")))
    local = {}
    for ch in spec["checks"]:
        p = _prob(ch["description"])
        for el in spec["elements"]:
            local[mms_auto.key(p, el) + ch["suffix"]] = True
    rec = mms_auto._record()
    added, refused = [], []
    for k, r in res["results"].items():
        if k not in local or r.get("error"):
            refused.append(k)
            continue
        if rec.get(k, {}).get("verdict") == "PASS" and r["verdict"] != "PASS":
            refused.append(k + " (a PASS is on record)")
            continue
        rec[k] = r
        added.append((k, r["verdict"]))
    json.dump(rec, open(mms_auto.RECORD, "w"), indent=1)
    if refused and not added:
        raise SystemExit(f"nothing imported: the local code differs or the job failed: {refused}")
    return {"added": added, "refused": refused}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("description")
    p.add_argument("levels", nargs="+", help="'<ladder key>:<n>:<element>'")
    p.add_argument("--mem-mb", type=int, default=120000)
    p.add_argument("--hours", type=int, default=24)
    r = sub.add_parser("run")
    r.add_argument("spec")
    m = sub.add_parser("run-many")
    m.add_argument("specs", nargs="+")
    b = sub.add_parser("batch")
    b.add_argument("names", nargs="*")
    i = sub.add_parser("import")
    i.add_argument("returned")
    i.add_argument("md5")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        lv = [(s.rsplit(":", 2)[0], int(s.rsplit(":", 2)[1]), s.rsplit(":", 2)[2]) for s in a.levels]
        print(json.dumps(prepare(json.load(open(a.description)), lv, a.mem_mb, a.hours), indent=1))
    elif a.cmd == "run":
        run(a.spec)
    elif a.cmd == "run-many":
        for s in a.specs:                 # one failing job must not stop the others
            try:
                run(s)
            except BaseException as e:
                print(f"{s}: {e!s:.300}", flush=True)
    elif a.cmd == "batch":
        print(json.dumps(batch(a.names or None), indent=1))
    else:
        print(json.dumps(import_result(a.returned, a.md5), indent=1))


if __name__ == "__main__":
    main()
