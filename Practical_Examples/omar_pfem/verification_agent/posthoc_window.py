"""POST HOC (2026-11-24, after seeing the finite-strain coverage result): drop the coarsest
level from the multilevel fit when the five-level fit gives no certified rate.

Seen first: on the steepest finite-strain cases (s = 0.08) the coarsest mesh (n = 4) is
far from the asymptotic range (observed order 1.2 between n = 4 and 8, then 4.0). The
five-level fit then returns p = 1.3 and an uncertainty of about 5%, above the 1% tolerance,
so the framework said FAIL although the error at n = 64 is 3e-5. The three-level GCI, which
never looks at n = 4, passed.

Variant (not part of the frozen framework; nothing in gates.py or orchestrate.py changes):
when qoi_convergence_multilevel on the finest five levels is not PASS with a rate, it is
run again on the finest four. Its result replaces the five-level one only if that is PASS
with a rate. Four levels give no leave-one-out check, so the variant is weaker by
construction; this script measures what it does to coverage on BOTH exact-solution studies
(the 32 corner problems, where it was not tuned, and the 12 finite-strain problems, where
the problem was seen). Outputs: posthoc_window_result.json; the pre-registered results
(wedge_result.json, finite_strain_result.json) are not touched.

    python -m omar_pfem.verification_agent.posthoc_window
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "posthoc_window_result.json")


class _Trim:
    """The same adapter without its coarsest level."""

    def __init__(self, ad):
        self._ad = ad

    def __getattr__(self, k):
        return getattr(self._ad, k)

    @property
    def family(self):
        return self._ad.family[1:]

    def series(self, q):
        hs, vals, extra = self._ad.series(q)
        return hs[1:], vals[1:], extra


def _patched(original, log):
    def run_check(ad, check, qoi, params, **kw):
        r = original(ad, check, qoi, params, **kw)
        if check != "qoi_convergence_multilevel" or (r.get("verdict") == "PASS"
                                                     and r.get("evidence") == "rate"):
            return r
        if len(ad.series(qoi)[1]) < 5:
            return r
        t = original(_Trim(ad), check, qoi, params, **kw)
        if t.get("verdict") == "PASS" and t.get("evidence") == "rate":
            log.append({"qoi": qoi, "five_level": r.get("reason"), "four_level": t.get("reason")})
            return {**t, "posthoc_dropped_coarsest": True}
        return r
    return run_check


def _run(module, attr_result):
    from omar_pfem.verification_agent import orchestrate
    original, log = orchestrate.run_check, []
    saved = getattr(module, attr_result)
    tmp = OUT + ".tmp"
    orchestrate.run_check = _patched(original, log)
    setattr(module, attr_result, tmp)
    try:
        res = module.score()
    finally:
        orchestrate.run_check = original
        setattr(module, attr_result, saved)
        if os.path.exists(tmp):
            os.remove(tmp)
    return res, log


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from omar_pfem.verification_agent import finite_strain_preregister as fs
    from omar_pfem.verification_agent import wedge_preregister as wp
    out = {"label": "POST HOC variant: finest-four refit when the five-level fit gives no rate"}
    for name, mod, base in (("finite_strain", fs, "finite_strain_result.json"),
                            ("wedge", wp, "wedge_result.json")):
        res, log = _run(mod, "RESULT")
        pre = json.load(open(os.path.join(HERE, base)))["summary"]
        keys = ("framework_certified", "framework_covered", "framework_effectivity_median",
                "framework_effectivity_max")
        out[name] = {"pre_registered": {k: pre.get(k) for k in keys},
                     "variant": {k: res["summary"].get(k) for k in keys},
                     "variant_wilson95": res["summary"].get("framework_coverage_wilson95"),
                     "refits_used": len(log),
                     "variant_not_covered": [
                         {k: r.get(k) for k in ("case", "qoi")} | {"effectivity": r["framework"]["effectivity"]}
                         for r in res["rows"] if "framework" in r and not r["framework"].get("covered", True)]}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
