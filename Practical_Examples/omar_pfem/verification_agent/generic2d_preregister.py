"""Pre-registration of the generic 2D pipeline on a held-out problem (T-bracket),
written before any solve of it (2026-11-24). The pipeline was validated beforehand on the
U-shape only (generic2d reproduces ushape.py's uniform ladder to 1e-12, levels 2-16).

    python -m omar_pfem.verification_agent.generic2d_preregister           (pre-register)
    python -m omar_pfem.verification_agent.generic2d_preregister --run     (solve + score)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generic2d_preregistration.json")
DESC = os.path.join(HERE, "problems2d", "t_bracket.json")
FROZEN = ("generic2d.py", "gates.py", "trees.py", "orchestrate.py", "crosscheck.py",
          "remedy.py", "remedy_intensity.py", "calibration.py", "problems2d/t_bracket.json")


def _sha(n):
    return hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest()


def main():
    if os.path.exists(os.path.join(HERE, "generic2d_runs", "t_bracket")):
        raise SystemExit("t_bracket already solved: not a pre-registration")
    pre = {"written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
           "predictions": {
               "P1 end to end": "the pipeline runs from the JSON description with no problem "
                                "code (HIGH)",
               "P2 energy": "the strain energy is certified with a rate on the uniform Q2 "
                            "ladder (MEDIUM)",
               "P3 cross-check": "the independent P2 ladder cross-checks the uniform energy "
                                 "(MEDIUM)",
               "P4 remedy": "the remedy loop is triggered and the graded energy U is at least "
                            "10 times below the uniform one (MEDIUM)",
               "P5 peaks": "the peak von Mises is certified on neither ladder (HIGH)",
               "P6 plan": "the intensity plan grades both re-entrant corners (1,1) and (2,1) "
                          "(LOW-MEDIUM: the load is symmetric, so their intensities are equal)"}}
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps(pre["predictions"], indent=1))


def _check_uniform_reproduced():
    from omar_pfem.verification_agent.generic2d import Problem, RUNS
    cache = json.load(open(os.path.join(RUNS, "t_bracket", "ladders.json")))
    prob = Problem(json.load(open(DESC)))
    for key, el in (("Q2 uniform", "Q2"), ("P2 uniform", "P2")):
        for row in cache.get(key, []):
            if row["n"] <= 16:
                new = prob.solve(row["n"], None, el)
                assert abs(new["strain_energy"] / row["strain_energy"] - 1) < 1e-12, key


def amend():
    pre = json.load(open(OUT))
    pre["amendment"] = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "why": "the first graded solve crashed (scikit-fem inverse map: Newton did not reach "
               "TOL = 1e-12 on tiny graded elements); graded solves now use the size-scaled "
               "tolerance of rounds K/L (remedy_preregister._relaxed_invF). Uniform solves are "
               "unchanged and must be reproduced to 1e-12; no criterion or prediction changes. "
               "At this point the uniform Q2, strategy and P2 ladders existed; no graded "
               "result and no score did.",
        "generic2d_sha256": _sha("generic2d.py")}
    json.dump(pre, open(OUT, "w"), indent=1)


def run_and_score():
    from omar_pfem.verification_agent.generic2d import run
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    am = pre.get("amendment")
    if am and "generic2d.py" in changed and _sha("generic2d.py") == am["generic2d_sha256"]:
        _check_uniform_reproduced()
        changed.remove("generic2d.py")
        print("note: generic2d.py amended (graded solves only); uniform levels reproduced")
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    res = run(json.load(open(DESC)))
    U, G = res["uniform"], res.get("graded", {})
    graded = {(c["x"], c["y"]) for c in res.get("plan", {}).get("corners", []) if c.get("graded")}
    met = {
        "P1 end to end": True,
        "P2 energy": "limit" in U["strain_energy"],
        "P3 cross-check": U["strain_energy"].get("crosscheck") == "cross_checked",
        "P4 remedy": bool(G) and "U" in U["strain_energy"] and "U" in G.get("strain_energy", {})
        and G["strain_energy"]["U"] * 10 <= U["strain_energy"]["U"],
        "P5 peaks": "limit" not in U["peak_von_mises"] and "limit" not in G.get("peak_von_mises", {}),
        "P6 plan": {(1, 1), (2, 1)} <= graded,
    }
    res["predictions_met"] = met
    json.dump(res, open(os.path.join(HERE, "generic2d_runs", "t_bracket", "result.json"), "w"),
              indent=1, default=float)
    print(f"predictions met {sum(met.values())}/{len(met)}")
    for k, v in met.items():
        print(f"  {'met ' if v else 'MISS'} {k}")
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    if "--amend" in sys.argv:
        amend()
    elif "--run" in sys.argv:
        run_and_score()
    else:
        main()
