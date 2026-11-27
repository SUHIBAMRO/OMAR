"""Automatic per-run report (25 Nov; item 3 of "automate everything").

Built only from the stored files of one run (no number is typed by hand):
  autonomous_runs/<name>/run.json, pipeline2d_runs/<name>/ladders.json,
  experiments/<name>/*.json (the files named in the run's log), sensitivity record in run.json.
Writes autonomous_runs/<name>/report.html and, if Chromium is available, report.pdf.

    python -m omar_pfem.verification_agent.run_report plate_hole_quarter [--no-pdf]

The report states whether the stored ladders still match the current code (fingerprint) and
gives the sha256 of every input file, so a reader can tell exactly what it was built from.
"""
import argparse
import base64
import hashlib
import html
import io
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = {"L": "literature / input", "D": "problem definition", "M": "geometry and mesh",
          "V": "code verification (manufactured solution)", "S": "solve (mesh ladders)",
          "C": "convergence checks", "G": "diagnosis", "N": "next experiments",
          "A": "assumed inputs (sensitivity)"}


def fmt(x, sig=6):
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{sig}g}"
    except (TypeError, ValueError):
        return html.escape(str(x))


def pct(x):
    return "-" if x is None else f"{100 * float(x):.3g}%"


def esc(s):
    return html.escape(str(s))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# ------------------------------------------------------------------ figures ---------------
BC_COL = {"clamped": "#c0392b", "symmetry": "#2471a3", "traction": "#d68910",
          "pressure": "#7d3c98", "free": "#7f8c8d"}


def fig_domain(desc, prob):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    if "domain" in desc:
        o = np.array(desc["domain"]["outer"] + desc["domain"]["outer"][:1], float)
        ax.plot(o[:, 0], o[:, 1], color=BC_COL["free"], lw=1)
        for h in desc["domain"].get("holes", []):
            if "circle" in h:
                cx, cy, r = h["circle"]
                t = np.linspace(0, 2 * np.pi, 200)
                ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=BC_COL["free"], lw=1)
            else:
                p = np.array(h["polygon"] + h["polygon"][:1], float)
                ax.plot(p[:, 0], p[:, 1], color=BC_COL["free"], lw=1)
    else:
        from omar_pfem.verification_agent.pipeline2d import polygon
        a = float(desc.get("cell_size", 1.0))
        p = np.array([[x * a, y * a] for x, y, _ in polygon(desc)])
        p = np.vstack([p, p[:1]])
        ax.plot(p[:, 0], p[:, 1], color=BC_COL["free"], lw=1)
    seen = set()
    for b in desc["bcs"]:
        col = BC_COL.get(b["type"], "k")
        lab = b["type"] if b["type"] not in seen else None
        seen.add(b["type"])
        if "segment" in b:
            s = np.array(b["segment"], float)
            ax.plot(s[:, 0], s[:, 1], color=col, lw=3.5, label=lab)
        elif "hole" in b:
            cx, cy, r = desc["domain"]["holes"][b["hole"]]["circle"]
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=col, lw=3.5, label=lab)
    for name, pt in desc.get("points", {}).items():
        x, y, c = pt[0], pt[1], pt[-1]
        ax.plot([x], [y], "ko", ms=4)
        ax.annotate(f"{name} ({c})", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    for c in getattr(prob, "corners", []):
        ax.plot([c["x"] * prob.a], [c["y"] * prob.a], "r^", ms=7)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("domain, boundary conditions, points (triangles: rate-limiting corners)",
                 fontsize=9)
    return _fig(fig)


def fig_mesh(prob, n=8):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    try:
        msh, _ = prob.build_mesh(n, None, "Q2")
    except Exception as e:                                 # the report must not fail on a plot
        return f"<p>(mesh plot not available: {esc(e)})</p>"
    k = 4 if msh.t.shape[0] >= 4 and "Quad" in type(msh).__name__ else 3
    polys = [msh.p[:, msh.t[:k, i]].T for i in range(msh.t.shape[1])]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#34495e", lw=0.4))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(f"Q2 mesh at level n = {n} ({msh.t.shape[1]} elements; straight edges drawn)",
                 fontsize=9)
    return _fig(fig)


def _u_rel(c):
    """The framework's U is RELATIVE (gates.uncertainty_fine)."""
    return abs(c.get("U_rel", c["U"]))


def _u_abs(c):
    return _u_rel(c) * abs(c["limit"])


def fig_convergence(ladders, q, info):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for key, st in (("Q2 uniform", "o-"), ("P2 uniform", "s--"), ("Q2 graded", "^-"),
                    ("P2 graded", "v--")):
        rows = sorted(ladders.get(key, []), key=lambda r: r["n"])
        rows = [r for r in rows if q in r]
        if len(rows) < 2:
            continue
        dof = [r["n_dof"] for r in rows]
        val = [r[q] for r in rows]
        ax[0].semilogx(dof, val, st, ms=4, label=key)
        if info and info.get("limit"):
            err = [abs(v - info["limit"]) / abs(info["limit"]) for v in val]
            if all(e > 0 for e in err):
                ax[1].loglog(dof, err, st, ms=4, label=key)
    if info and info.get("limit"):
        ua = _u_abs(info)
        ax[0].axhspan(info["limit"] - ua, info["limit"] + ua, color="#27ae60", alpha=0.25,
                      label="verified limit +- U (numerical verification error)")
        ax[1].axhline(_u_rel(info), color="#27ae60", ls=":", label="U (relative)")
        ax[1].set_title("|value - verified limit| / |limit|", fontsize=9)
    else:
        ax[1].set_visible(False)
    ax[0].set_title(q, fontsize=9)
    for a in ax:
        a.set_xlabel("degrees of freedom", fontsize=8)
        a.tick_params(labelsize=7)
        if a.get_visible():
            a.legend(fontsize=7)
    return _fig(fig)


def _chrome():
    import glob
    import shutil
    for c in (os.environ.get("CHROME"), "/opt/pw-browsers/chromium/chrome-linux64/chrome",
              *sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux64/chrome")),
              shutil.which("chromium"), shutil.which("chromium-browser"),
              shutil.which("google-chrome")):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("no Chromium/Chrome found (set CHROME)")


# ------------------------------------------------------------------ report ----------------
def build(name, pdf=True):
    from omar_pfem.verification_agent import pipeline2d as P
    from omar_pfem.verification_agent.geometry2d import make
    rdir = os.path.join(HERE, "autonomous_runs", name)
    run_f = os.path.join(rdir, "run.json")
    run = json.load(open(run_f))
    if "description" not in run:                    # a run that stopped early
        src = [s for s in run["steps"] if s.get("description")]
        run["description"] = src[-1]["description"] if src else json.load(open(os.path.join(
            HERE, "problems3d" if os.path.exists(os.path.join(HERE, "problems3d", name + ".json"))
            else "problems2d", name + ".json")))
    desc = {k: v for k, v in run["description"].items() if k != "assumed"}
    prob = make(desc)
    lad_f = os.path.join(HERE, "pipeline2d_runs", name, "ladders.json")
    ladders = json.load(open(lad_f)) if os.path.exists(lad_f) else {}
    fp_now = P.fingerprint(prob)
    fp_ok = ladders.get("_fingerprint") == fp_now
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                          text=True).stdout.strip()
    final = run.get("final") or {"certified": [], "open": list(prob.qois)}
    cert = run.get("certified_info", {})
    steps = run["steps"]
    persons = [s for s in steps if s["actor"] == "person" or s["status"] == "needs_person"]
    by_actor = {a: sum(s["actor"] == a for s in steps) for a in ("code", "LLM", "person")}
    concl = {e["quantity"]: e for e in run.get("experiments", []) if "quantity" in e}
    V, Gd = run.get("verdicts", {}), run.get("graded") or {}
    out = []
    w = out.append
    w(f"<h1>Verification run: {esc(name)}</h1>")
    w(f"<p class=meta>built automatically from the stored files of the run; git {esc(head)}; "
      f"wall clock of the run {fmt(run.get('wall_clock_s', 0) / 60, 3)} min</p>")
    # summary
    n_q = len(final["certified"]) + len(final["open"])
    w("<h2>1. Summary</h2><ul>")
    w(f"<li><b>{len(final['certified'])} of {n_q}</b> quantities numerically verified: "
      f"{esc(', '.join(final['certified']) or 'none')}.</li>")
    if final["open"]:
        w(f"<li>Open: {esc(', '.join(final['open']))} (reasons in Section 4).</li>")
    mms = run.get("mms", {})
    if mms:
        w("<li>Code verification (manufactured solution): " + esc(", ".join(
            f"{k} {r['verdict']}" for k, r in mms.items())) + ".</li>")
    from omar_pfem.verification_agent.autonomous import KINDS, step_kind
    kinds = {k: sum(step_kind(s["stage"], s["actor"], s["status"]) == k for s in steps)
             for k in KINDS}
    w(f"<li>Steps (27 Nov classification): autonomous decisions {kinds['autonomous decision']}, "
      f"predefined procedures {kinds['predefined procedure']}, human interventions "
      f"<b>{kinds['human intervention']}</b> (by code {by_actor['code']}, by LLM "
      f"{by_actor['LLM']}).</li>")
    if run.get("code_verification"):
        w(f"<li class=warn><b>Code verification</b>: {esc(run['code_verification'])}.</li>")
    for job in run.get("cluster_jobs", []):
        w(f"<li class=warn>Cluster job prepared (manual submission): {esc(job['name'])}, levels "
          f"{esc(job['levels'])}, bundle md5 <code>{esc(job['bundle_md5'])}</code>, instructions "
          f"{esc(job['instructions'])}.</li>")
    stops = [s for s in steps if s["status"] == "stop"]
    if stops and stops[-1]["stage"] != "N":
        w(f"<li class=warn><b>The run stopped at stage {esc(stops[-1]['stage'])} "
          f"({esc(STAGES.get(stops[-1]['stage'], ''))})</b>: {esc(stops[-1]['detail'])}</li>")
    if not ladders:
        w("<li>No mesh ladders were solved in this run.</li></ul>")
    else:
        w(f"<li>Stored ladders {'match' if fp_ok else '<b>DO NOT match</b>'} the current solver and "
          f"mesh code (fingerprint {esc(str(ladders.get('_fingerprint', 'none'))[:12])}"
          f"{'' if fp_ok else ' vs current ' + fp_now[:12]}).</li></ul>")
    # problem
    w("<h2>2. Problem</h2>")
    w(f"<p>Material {esc(desc.get('material', 'linear_elastic'))}, E = {fmt(desc['E'])}, "
      f"&nu; = {fmt(desc['nu'])}, "
      f"{'3D' if desc.get('dim') == 3 else 'plane ' + esc(desc.get('plane', 'strain'))}. Source: "
      f"{esc(steps[0]['detail'])}.</p>")
    w("<table><tr><th>boundary</th><th>type</th><th>values</th></tr>")
    for k, b in enumerate(desc["bcs"]):
        where = (f"segment {b['segment']}" if "segment" in b else f"hole {b['hole']}"
                 if "hole" in b else f"face {b['face']} (z = "
                 f"{0 if b['face'] == 'z0' else desc['thickness']})")
        vals = ", ".join(f"{key} = {b[key]}" for key in ("t", "p", "u") if key in b)
        w(f"<tr><td>bcs[{k}]: {esc(where)}</td><td>{esc(b['type'])}</td><td>{esc(vals)}</td></tr>")
    w("</table>")
    if getattr(prob, "corners", []):
        w("<p>Rate-limiting corners (smallest exponent &lambda;; &lambda; &lt; 1 means unbounded "
          "stress): " + esc("; ".join(f"({fmt(c['x'] * prob.a, 4)}, {fmt(c['y'] * prob.a, 4)}) "
                                      f"λ = {fmt(min(c['rate_limiting']), 4)}"
                                      for c in prob.corners)) + "</p>")
    if getattr(prob, "dim", 2) == 3:
        w(f"<p>3D: the section below extruded to thickness {fmt(prob.T)} in n/2 x {prob.kz} layers; "
          "27-node hexahedra (Q2) and 10-node tetrahedra (P2); z faces free unless listed in the table. "
          "Singular edges: the extruded-edge "
          "approximation (vertex singularities not analysed). Edges where a side face meets a z face: "
          + esc("; ".join(f"{e['face']} / {e['side']}: λ = {fmt(min(e['rate_limiting']), 4)}"
                          for e in prob.face_edges) or "none singular") + ".</p>")
        prob.g2.corners = prob.corners
        w("<div class=row>" + fig_domain(desc, prob) + fig_mesh(prob.g2) + "</div>")
    else:
        w("<div class=row>" + fig_domain(desc, prob) + fig_mesh(prob) + "</div>")
    # code verification
    refc = run.get("mms_reference_domain")
    w("<h2>3. Code verification (manufactured solution, MMS): is the solver correct?</h2>")
    if refc:
        w("<p>Production domain INCONCLUSIVE (post hoc rule R1); the same check on the reference "
          f"domain {esc(refc['description']['domain'])} (post hoc rule R2): "
          + esc("; ".join(f"{el} {c['verdict']}" + ("" if c.get("skipped") else
                                                     f" (L2 {c['rates']['L2']['observed']:.2f}, H1 "
                                                     f"{c['rates']['H1']['observed']:.2f}, levels {c['levels']})")
                          for el, c in refc["checks"].items())) + "</p>")
    w("<table><tr><th>element</th><th>verdict</th><th>L2 rate "
      "(expected 3)</th><th>H1 rate (expected 2)</th><th>levels</th><th>note</th></tr>")
    for el, r in mms.items():
        if r.get("skipped"):
            w(f"<tr><td>{el}</td><td>{r['verdict']}</td><td colspan=3>not repeated</td>"
              f"<td>{esc(r['reason'])}</td></tr>")
        else:
            w(f"<tr><td>{el}</td><td>{r['verdict']}</td><td>{fmt(r['rates']['L2']['observed'], 3)}"
              f"</td><td>{fmt(r['rates']['H1']['observed'], 3)}</td><td>{esc(r['levels'])}</td>"
              f"<td>{esc('; '.join(r.get('levels_added', [])) or '')}</td></tr>")
    w("</table>")
    # results
    w("<h2>4. Solution verification: convergence of the problem itself</h2><table><tr><th>quantity</th><th>status</th><th>verified limit "
      "&plusmn; U</th><th>U (relative)</th><th>finest value</th><th>evidence / reason</th></tr>")
    later = []
    for q in prob.qois:
        if q in cert:
            c = cert[q]
            now = [s for s in (V.get(q, {}), Gd.get(q, {})) if s.get("final") == "CERTIFIED"]
            if now:
                st = now[0]
                ev = f"{st.get('evidence', '-')}; cross-check {st.get('crosscheck', '-')}"
            else:
                # the loop keeps a quantity certified at an earlier stage (certified_any); the
                # report shows that the latest state no longer certifies it
                st = V.get(q, {})
                ev = (f"numerically verified at an earlier stage of this run; LATEST STATE {st.get('final', '-')}"
                      f" ({st.get('evidence', '-')}; cross-check {st.get('crosscheck', '-')}) "
                      "- see the log")
                later.append(q)
            w(f"<tr class=ok><td>{esc(q)}</td><td>CERTIFIED</td><td>{fmt(c['limit'])} &plusmn; "
              f"{fmt(_u_abs(c), 3)}</td><td>{pct(_u_rel(c))}"
              f"</td><td>{fmt(c['finest'])}</td><td>{esc(ev)}</td></tr>")
        else:
            st = V.get(q, {})
            if q in concl:
                why = f"{concl[q]['conclusion']}: {concl[q]['justification']}"
            else:
                d = run.get("diagnosis", {}).get(q, {})
                why = (f"verdict {st.get('verdict', '-')} ({st.get('evidence', '-')}), cross-check "
                       f"{st.get('crosscheck', '-')}; diagnosis {', '.join(d.get('diagnosis', [])) or '-'}")
            w(f"<tr class=open><td>{esc(q)}</td><td>{esc(st.get('final', 'open'))}</td><td>-</td>"
              f"<td>-</td><td>{fmt(st.get('finest'))}</td><td>{esc(why)}</td></tr>")
    w("</table>")
    if later:
        w(f"<p class=warn>Note: {esc(', '.join(later))} were numerically verified at an earlier stage, but the "
          "latest state after the agent's experiments does not verify them. By the decision of 27 Nov the loop keeps "
          "an earlier estimate only without contradictory evidence and records the limitation "
          "(open question, see the change log).</p>")
    for q in prob.qois:
        w(fig_convergence(ladders, q, cert.get(q)))
    # experiments
    w("<h2>5. Experiments chosen by the agent</h2>")
    pre = [s for s in steps if s["stage"] == "N" and s["status"] == "pre-registered"]
    if not pre:
        w("<p>None.</p>")
    else:
        w("<table><tr><th>#</th><th>action and hypothesis</th><th>prediction</th><th>prior</th>"
          "<th>est. / observed min</th><th>outcome</th></tr>")
        for s in pre:
            rec = json.load(open(os.path.join(HERE, s["file"]))) if s.get("file") else {}
            p, o = rec.get("preregistration", {}), rec.get("outcome", {})
            out_s = ("failed: " + o["failed"][:80]) if o.get("failed") else (
                "confirmed" if o.get("confirmed") else "refuted" if o else "-")
            w(f"<tr><td>{esc(os.path.basename(s.get('file', '')))}</td><td>{esc(p.get('action'))}: "
              f"{esc(p.get('hypothesis'))}</td><td>{esc(p.get('prediction', {}).get('text'))}</td>"
              f"<td>{fmt(p.get('prior'), 2)}</td><td>{fmt(p.get('cost', {}).get('minutes'), 2)} / "
              f"{fmt(o.get('observed_minutes'), 2)}</td><td>{esc(out_s)}</td></tr>")
        w("</table><p class=meta>Each prediction was frozen (sha256) before the run; outcomes "
          "judged by code.</p>")
    # sensitivity
    w("<h2>6. Assumed inputs</h2>")
    sens = run.get("sensitivity")
    if not sens:
        w("<p>No assumed inputs were checked in this run.</p>")
    else:
        w(f"<p>Range: E, loads, prescribed displacement, hole radius &plusmn;"
          f"{pct(sens['range']['relative'])}; &nu; &plusmn;{fmt(sens['range']['nu_absolute'])}. "
          "Rule in force (decision of 27 Nov, QoI-dependent tolerances): a person is needed "
          "only where the change exceeds the adopted accuracy target of the quantity. The "
          "pre-registered rule (change above the numerical verification error U) is shown for "
          "comparison.</p>")
        w("<table><tr><th>input</th><th>quantity</th><th>change over range</th><th>U / |limit|"
          "</th><th>S</th><th>pre-registered rule</th><th>rule in force (27 Nov)</th></tr>")
        for name_, r in sens["params"].items():
            if "quantities" not in r:
                w(f"<tr><td>{esc(name_)}</td><td colspan=6>{esc(r.get('status'))}: person</td></tr>")
                continue
            for q, x in r["quantities"].items():
                ph = x.get("posthoc_person_needed")
                w(f"<tr><td>{esc(name_)}</td><td>{esc(q)}</td><td>{pct(x['change_rel'])}</td>"
                  f"<td>{pct(x.get('U_rel'))}</td><td>{fmt(x['S'], 4)}</td>"
                  f"<td>{esc(x.get('decision', '-'))}</td><td>"
                  f"{'-' if ph is None else 'person' if ph else 'no person'}</td></tr>")
        w("</table>")
    # people and log
    w("<h2>7. Where a person was needed</h2>")
    w("<ul>" + "".join(f"<li>[{esc(s['stage'])}] {esc(s['detail'])}</li>" for s in persons) + "</ul>"
      if persons else "<p>Nowhere.</p>")
    w("<h2>8. Full log</h2><table><tr><th>stage</th><th>by</th><th>status</th><th>detail</th></tr>")
    for s in steps:
        w(f"<tr><td>{esc(s['stage'])} {esc(STAGES.get(s['stage'], ''))}</td><td>{esc(s['actor'])}"
          f"</td><td>{esc(s['status'])}</td><td>{esc(s['detail'])}</td></tr>")
    w("</table>")
    # provenance
    w("<h2>9. Inputs of this report</h2><table><tr><th>file</th><th>sha256</th></tr>")
    files = [run_f] + ([lad_f] if os.path.exists(lad_f) else []) + [
        os.path.join(HERE, s["file"]) for s in pre if s.get("file")]
    for f in files:
        w(f"<tr><td>{esc(os.path.relpath(f, HERE))}</td><td><code>{_sha(f)[:16]}</code></td></tr>")
    w("</table><p class=meta>Regenerate: <code>python -m omar_pfem.verification_agent.run_report "
      f"{esc(name)}</code></p>")
    css = ("body{font-family:Arial,Helvetica,sans-serif;font-size:12px;max-width:980px;margin:24px "
           "auto;color:#1c2833}h1{font-size:20px}h2{font-size:15px;border-bottom:1px solid #aab7b8;"
           "margin-top:22px}table{border-collapse:collapse;width:100%;margin:6px 0}td,th{border:1px "
           "solid #d5d8dc;padding:3px 5px;vertical-align:top;text-align:left}th{background:#f2f4f4}"
           "tr.ok td:nth-child(2){color:#1e8449;font-weight:bold}tr.open td:nth-child(2){color:"
           "#b9770e;font-weight:bold}.meta{color:#566573;font-size:11px}.warn{color:#9a3412;background:#fff7ed;padding:5px}.row{display:flex;gap:8px;"
           "flex-wrap:wrap}img{max-width:100%}@page{size:A4;margin:12mm 10mm}tr{page-break-inside:avoid}code{font-size:11px}")
    page = (f"<!doctype html><html><head><meta charset=utf-8><title>Run report {esc(name)}</title>"
            f"<style>{css}</style></head><body>{''.join(out)}</body></html>")
    html_f = os.path.join(rdir, "report.html")
    open(html_f, "w").write(page)
    pdf_f = None
    if pdf:
        try:
            chrome = _chrome()
            pdf_f = os.path.join(rdir, "report.pdf")
            subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu",
                            "--no-pdf-header-footer", f"--print-to-pdf={pdf_f}",
                            "file://" + html_f], check=True, capture_output=True, timeout=180)
        except Exception as e:                       # the HTML is the report; PDF is a copy
            pdf_f = None
            print(f"PDF not written: {e!s:.200}")
    return html_f, pdf_f


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    print(build(a.name, pdf=not a.no_pdf))


if __name__ == "__main__":
    main()
