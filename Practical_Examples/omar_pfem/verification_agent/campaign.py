"""Campaign: run the autonomous loop on many problems and tie the results together (25 Nov,
round Q).

  run      runs the loop on every listed description (cached solves are reused, so a repeated
           campaign only computes what is new) and writes each run's report
  index    autonomous_runs/index.html (+ index.pdf): one row per run with the code
           verification status, certified / open quantities, points needing a person, pending
           cluster jobs and a link to the run's report; then every open question collected from
           the runs (stored in open_questions.json) - built from the stored files only
  draft    advisor_feedback/<date>_new_results_DRAFT.md: the facts of the campaign and the open
           questions, written from the stored files, as a DRAFT for Omar (nothing is sent)

    python -m omar_pfem.verification_agent.campaign run problems3d/*.json problems2d/x.json
    python -m omar_pfem.verification_agent.campaign index
    python -m omar_pfem.verification_agent.campaign draft
"""
import argparse
import datetime
import html
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "autonomous_runs")
QUESTIONS = os.path.join(HERE, "open_questions.json")
ADVISOR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "advisor_feedback"))
# questions raised by the work itself (not by a single run); each with its evidence
STANDING = [
    {"id": "Q-sensitivity-threshold",
     "question": "When should the agent ask a person to confirm an assumed input? The "
                 "pre-registered rule (the input changes a certified quantity by more than its "
                 "numerical uncertainty) asks for every input on smooth problems; the post hoc "
                 "proposal asks only when the change exceeds the adopted accuracy target.",
     "evidence": "sensitivity_runs/*.json, sensitivity_preregistration.json"},
    {"id": "Q-earlier-certificate",
     "question": "Should a quantity certified at an earlier stage of a run keep its certificate "
                 "when a later state (more levels) no longer certifies it? Today the loop keeps it.",
     "evidence": "autonomous_runs/plate_hole_quarter/report.pdf (u_right)"},
    {"id": "Q-reference-domain",
     "question": "May code verification on a reference domain (same code paths, regular meshes) "
                 "replace an INCONCLUSIVE check on the production domain (rules R1/R2, post hoc)?",
     "evidence": "round_q_preregistration.json; autonomous_runs/cook_3d/report.pdf"},
]


def _j(p):
    return json.load(open(p))


def run(files, budget_minutes=20.0, max_experiments=4):
    from omar_pfem.verification_agent import autonomous
    done = []
    for f in files:
        desc = _j(f)
        print(f"=== {desc['name']} ({f})", flush=True)
        try:
            autonomous.run(desc, budget_minutes=budget_minutes, max_experiments=max_experiments)
            done.append((desc["name"], "ok"))
        except Exception as e:                      # one problem must not stop the campaign
            done.append((desc["name"], f"error: {e!s:.200}"))
            print(f"    error: {e!s:.300}", flush=True)
    index()
    return done


def _desc(name, r):
    if r.get("description"):
        return r["description"]
    for d in ("problems3d", "problems2d"):
        f = os.path.join(HERE, d, name + ".json")
        if os.path.exists(f):
            return _j(f)
    return {}


def _row(name):
    r = _j(os.path.join(RUNS, name, "run.json"))
    desc = _desc(name, r)
    mms = r.get("mms", {})
    cv = ("reference domain (R2)" if r.get("code_verification") else
          ", ".join(f"{k} {v.get('verdict')}" for k, v in mms.items()) or "-")
    final = r.get("final") or {}
    steps = r.get("steps", [])
    stop = [s for s in steps if s["status"] == "stop" and s["stage"] != "N"]
    persons = [s for s in steps if s["actor"] == "person" or s["status"] == "needs_person"]
    last = steps[-1] if steps else {}
    status = ("complete" if r.get("final") else
              f"stopped at {last.get('stage')} ({last.get('status')})" if last else "-")
    if not r.get("final") and stop:
        status = f"stopped at {stop[-1]['stage']}"
    return {"name": name, "dim": desc.get("dim", 2), "status": status,
            "material": desc.get("material", "linear_elastic"),
            "code_verification": cv, "certified": final.get("certified", []),
            "open": final.get("open", ["(not reached)"] if not r.get("final") else []),
            "persons": [s["detail"] for s in persons],
            "stopped": stop[-1]["detail"] if stop else None,
            "cluster_jobs": [j["name"] for j in r.get("cluster_jobs", [])],
            "report": os.path.exists(os.path.join(RUNS, name, "report.pdf"))}


def collect_questions(rows):
    qs = list(STANDING)
    for r in rows:
        for j in r["cluster_jobs"]:
            qs.append({"id": f"job-{j}", "question": f"Submit cluster job {j} (manual) - the "
                       f"levels this machine refused for {r['name']}.",
                       "evidence": f"cluster_jobs/{j}_job.json"})
        if r["stopped"]:
            qs.append({"id": f"stop-{r['name']}", "question": f"(status) {r['name']} stopped: "
                       f"{r['stopped']}" + (f"; cluster job {', '.join(r['cluster_jobs'])} prepared"
                                            if r["cluster_jobs"] else ""),
                       "evidence": f"autonomous_runs/{r['name']}/report.pdf"})
    json.dump(qs, open(QUESTIONS, "w"), indent=1)
    return qs


def index(pdf=True):
    names = sorted(d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "run.json")))
    rows = [_row(n) for n in names]
    qs = collect_questions(rows)
    e = html.escape
    t = ["<h1>Autonomous verification: all runs</h1>",
         f"<p class=meta>built {datetime.date.today()} from autonomous_runs/*/run.json; "
         f"{len(rows)} runs</p>",
         "<table><tr><th>run</th><th>status</th><th>dim</th><th>material</th><th>code verification</th>"
         "<th>certified</th><th>open</th><th>person points</th><th>stopped / jobs</th></tr>"]
    for r in rows:
        link = f"<a href='{e(r['name'])}/report.pdf'>{e(r['name'])}</a>" if r["report"] else e(r["name"])
        t.append(f"<tr><td>{link}</td><td>{e(r['status'])}</td><td>{r['dim']}D</td><td>{e(r['material'])}</td>"
                 f"<td>{e(r['code_verification'])}</td><td>{len(r['certified'])}: "
                 f"{e(', '.join(r['certified']))}</td><td>{e(', '.join(r['open']))}</td>"
                 f"<td>{len(r['persons'])}</td><td>{e(r['stopped'] or '')} "
                 f"{e(', '.join(r['cluster_jobs']))}</td></tr>")
    t.append("</table><h2>Open questions and actions</h2><ol>")
    t += [f"<li><b>{e(q['id'])}</b>: {e(q['question'])} <span class=meta>({e(q['evidence'])})"
          "</span></li>" for q in qs]
    t.append("</ol>")
    css = ("body{font-family:Arial,sans-serif;font-size:12px;max-width:1100px;margin:20px auto}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:3px 5px;"
           "vertical-align:top;text-align:left}th{background:#f2f4f4}.meta{color:#667;font-size:11px}"
           "@page{size:A4 landscape;margin:10mm}")
    page = f"<!doctype html><meta charset=utf-8><title>Runs index</title><style>{css}</style>{''.join(t)}"
    f = os.path.join(RUNS, "index.html")
    open(f, "w").write(page)
    if pdf:
        try:
            from omar_pfem.verification_agent.run_report import _chrome
            subprocess.run([_chrome(), "--headless", "--no-sandbox", "--disable-gpu",
                            "--no-pdf-header-footer", f"--print-to-pdf={f[:-5]}.pdf", "file://" + f],
                           check=True, capture_output=True, timeout=180)
        except Exception as ex:
            print(f"index PDF not written: {ex!s:.200}")
    return f


def draft(names=None):
    """A DRAFT summary for Omar from the stored files (facts only; nothing is sent)."""
    rows = [_row(n) for n in (names or sorted(
        d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "run.json"))))]
    qs = _j(QUESTIONS) if os.path.exists(QUESTIONS) else collect_questions(rows)
    out = [f"# New results - DRAFT ({datetime.date.today()}), generated from the stored runs",
           "", "Not sent. Every line below is written by campaign.py from autonomous_runs/*/run.json.",
           "", "| run | status | dim | material | code verification | certified | open | person points |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['status']} | {r['dim']}D | {r['material']} | {r['code_verification']} | "
                   f"{len(r['certified'])} ({', '.join(r['certified'])}) | {', '.join(r['open'])} | "
                   f"{len(r['persons'])} |")
    out += ["", "## Questions for Professor Rabczuk", ""]
    out += [f"{i + 1}. {q['question']} (evidence: {q['evidence']})" for i, q in
            enumerate(q for q in qs if not q["id"].startswith(("job-", "stop-")))]
    stops = [q for q in qs if q["id"].startswith("stop-")]
    if stops:
        out += ["", "## Runs waiting for the cluster", ""] + [f"- {q['question'][9:]}" for q in stops]
    os.makedirs(ADVISOR, exist_ok=True)
    f = os.path.join(ADVISOR, f"{datetime.date.today()}_new_results_DRAFT.md")
    open(f, "w").write("\n".join(out) + "\n")
    return f


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("files", nargs="+")
    sub.add_parser("index")
    d = sub.add_parser("draft")
    d.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(run(a.files))
    elif a.cmd == "index":
        print(index())
    else:
        print(draft(a.names or None))


if __name__ == "__main__":
    main()
