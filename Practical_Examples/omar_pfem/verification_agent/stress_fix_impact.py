"""Impact of the 2026-11-19 stress fix (double Cauchy conversion) on the B3
values and verdicts: the deterministic rules run on the OLD data
(pre_stressfix_20261119/) and on the re-solved data, side by side.

    python -m omar_pfem.verification_agent.stress_fix_impact
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "pre_stressfix_20261119")
OUT = os.path.join(HERE, "stress_fix_impact.json")


def _run(root):
    from omar_pfem.verification_agent import orchestrate, region_study
    from omar_pfem.verification_agent.verify_report import rules
    saved = orchestrate.HERE, region_study.DDIR
    orchestrate.HERE = root
    region_study.DDIR = os.path.join(root, "b3_gauss_near_groove")
    try:
        rules.cache_clear()                 # rules() is lru-cached: one result per root
        R = rules("bushing_groove")
        rules.cache_clear()
    finally:
        orchestrate.HERE, region_study.DDIR = saved
    out = {}
    for q, c in R["combine"].items():
        hs, vals, _ = R["adapter"].series(q)
        g5 = R["v"][q].get("qoi_convergence_multilevel", {})
        out[q] = {"verdict": c[0], "evidence": c[1], "finest": vals[-1],
                  "fit_limit": g5.get("fit_limit"), "n_levels": len(vals)}
    return out


def main():
    old, new = _run(OLD), _run(HERE)
    rows, changed = {}, []
    for q in new:
        o, n = old.get(q), new[q]
        rel = (abs(n["finest"] - o["finest"]) / max(abs(o["finest"]), 1e-12)) if o else None
        rows[q] = {"old": o, "new": n, "finest_rel_change": rel}
        if o and (o["verdict"], o["evidence"]) != (n["verdict"], n["evidence"]):
            changed.append(q)
    json.dump({"rows": rows, "verdict_changes": changed}, open(OUT, "w"), indent=1)
    print(f"{'QoI':<24}{'old':<14}{'new':<14}{'finest old':>12}{'finest new':>12}{'change':>9}")
    for q, r in rows.items():
        o, n = r["old"], r["new"]
        print(f"{q:<24}{(o['verdict'] + '/' + o['evidence']) if o else '-':<14}"
              f"{n['verdict'] + '/' + n['evidence']:<14}{(o['finest'] if o else float('nan')):>12.5g}"
              f"{n['finest']:>12.5g}{(r['finest_rel_change'] or 0):>9.2%}"
              + ("   <-- verdict changed" if q in changed else ""))
    print(f"\nverdict changes: {len(changed)} of {len(rows)}: {changed}")


if __name__ == "__main__":
    main()
