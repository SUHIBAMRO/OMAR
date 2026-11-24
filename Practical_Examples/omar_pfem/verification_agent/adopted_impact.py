"""What the adopted rule set (adopted.py, 24 Nov 2026) changes on every stored problem.

Same stored solves, labels and cross-check as three_way.py. Compares, quantity by quantity,
B+ (the framework with the former post hoc singular-order switch) with the adopted rules
(the same switch plus 'pre-asymptotic from below' and the absolute 1% / 5% targets), and the
final status B* (a rate certification counts only if cross-checked) for both.

    python -m omar_pfem.verification_agent.adopted_impact
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "adopted_impact.json")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent import adopted
    from omar_pfem.verification_agent.three_way import PROBLEMS, _j, label
    from omar_pfem.verification_agent.verify_report import rules
    X = _j("crosscheck_result.json")
    xc = {(r["problem"], r["qoi"]): r["status"] for r in X["rows"] + X.get("rows_posthoc_only", [])}
    rows = []
    for geo in PROBLEMS:
        os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
        try:
            Rp = rules.__wrapped__(geo)
        finally:
            os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
        with adopted.ruleset():
            Ra = rules.__wrapped__(geo)
        for q in Rp["combine"]:
            bp, ba = Rp["combine"][q][:2], Ra["combine"][q][:2]
            star = lambda b: ("PASS" if xc.get((geo, q)) == "cross_checked" else "INCONCLUSIVE") \
                if list(b) == ["PASS", "rate"] else b[0]
            sing = Ra["v"][q].get("singular_order", {})
            rows.append({"problem": geo, "qoi": q, "truth": label(geo, q),
                         "Bp": list(bp), "adopted": list(ba),
                         "Bstar": star(bp), "adopted_star": star(ba),
                         "below_rule": sing.get("label") == adopted.LABEL_BELOW,
                         "changed": list(bp) != list(ba)})

    def score(key):
        ver = [r for r in rows if r["truth"] == "verifiable"]
        bad = [r for r in rows if r["truth"] not in ("verifiable", "band_only")]
        return {"certified_of_verifiable": sum(r[key] == "PASS" for r in ver),
                "verifiable": len(ver),
                "missed_failures": sum(r[key] == "PASS" for r in bad)}
    out = {"ruleset": adopted.RULESET,
           "changed": [r for r in rows if r["changed"]],
           "below_rule_fired": sum(r["below_rule"] for r in rows),
           "Bstar": score("Bstar"), "adopted_star": score("adopted_star"),
           "rows": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("below_rule_fired", "Bstar", "adopted_star")}, indent=1))
    for r in out["changed"]:
        print(f"  {r['problem']:<22} {r['qoi']:<28} {r['Bp']} -> {r['adopted']}  ({r['truth']})")
    return out


if __name__ == "__main__":
    main()
