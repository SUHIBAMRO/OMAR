"""Follow-up of the clamped block (round J, 2026-11-23): one finer level, k = 12.

The agent's follow-up (followup.py, extended to 3D) supports D4 for strain_energy:
the pairwise orders 2.56, 1.84, 1.53, 1.38 (levels k = 1..8) still fall toward the
edge-singularity order 2 lam = 1.42, so the post hoc singular-order check labels the
five-level fit (1.68) pre-asymptotic (INCONCLUSIVE). The agent chose "add a finer
level" (k = 12) and declined to run it within its budget (estimate about 980 s and
at least 15 GB, budget 300 s and 8 GB; followup_clamped_block_3d_strain_energy.json).

Predictions are written here BEFORE k = 12 is solved (`python -m ... block3d_followup`);
the level is run by the authorized manual job block3d_followup_job.sh (higher budget,
chosen by a person), stored in clamped_block_3d_followup.json (never in the
pre-registered ladder) and scored with `--score`. No check, tolerance or window changes.

    python -m omar_pfem.verification_agent.block3d_followup            (pre-register)
    python -m omar_pfem.verification_agent.block3d_followup --score    (after the job)
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "block3d_followup_preregistration.json")
EXTRA = os.path.join(HERE, "clamped_block_3d_followup.json")
FROZEN = ("gates.py", "trees.py", "orchestrate.py", "diagnose.py", "followup.py")


def _sha(name):
    return hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()


def main():
    if os.path.exists(EXTRA):
        raise SystemExit("clamped_block_3d_followup.json exists: not a pre-registration any more")
    from omar_pfem.verification_agent.block3d import corner_exponent
    lam = corner_exponent()
    dec = json.load(open(os.path.join(HERE, "followup_clamped_block_3d_strain_energy.json")))
    pre = {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent_decision": {k: dec.get(k) for k in ("diagnosis_supported", "plan", "action")},
        "frozen_files_sha256": {f: _sha(f) for f in FROZEN},
        "theory_2lam": 2 * lam, "lam_minus_1": lam - 1,
        "predictions": {
            "P1 pairwise order (k = 6, 8, 12) of strain_energy": "within 0.15 of 2 lam = 1.42 "
                                                                 "(medium confidence)",
            "P2 five-level fit k = 3..12 of strain_energy": "within 0.15 of 2 lam, so the post "
                                                            "hoc singular-order check PASSes and "
                                                            "the B+ verdict becomes PASS with rate "
                                                            "(low-medium confidence: the four "
                                                            "vertex singularities may keep the fit "
                                                            "away from 2 lam)",
            "P3 w_tip": "stays PASS with rate (post hoc)",
            "P4 peak_von_mises": "grows again; location at the clamped face (x < 0.1); "
                                 "slope over k = 4..12 within 0.15 of lam - 1 = -0.289 "
                                 "(low confidence)",
            "P5 root_gk_avg_von_mises": "stays FAIL: its strategy checks use the k = 4 meshes, "
                                        "which a finer family level does not change",
            "P6 cost": "under 2 h and under 48 GB on the cluster node",
        },
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    print(json.dumps(pre["predictions"], indent=1))


def score():
    import numpy as np
    from omar_pfem.verification_agent.diagnose import pairwise_orders
    from omar_pfem.verification_agent.followup import _rows, evaluate
    pre = json.load(open(OUT))
    changed = [f for f, h in pre["frozen_files_sha256"].items() if _sha(f) != h]
    if changed:
        raise SystemExit(f"frozen files changed since pre-registration: {changed}")
    os.environ["VERIF_POSTHOC_SINGULAR"] = "1"
    try:
        _, rows, extra, _ = _rows("clamped_block_3d")
        ext = sorted(rows + extra, key=lambda r: r["k"])
        before, after = evaluate("clamped_block_3d", rows), evaluate("clamped_block_3d", ext)
    finally:
        os.environ.pop("VERIF_POSTHOC_SINGULAR", None)
    hs = [r["h"] for r in ext]
    po = pairwise_orders(hs, [r["strain_energy"] for r in ext])
    ml = after["strain_energy"][1].get("qoi_convergence_multilevel", {})
    sg = after["strain_energy"][1].get("singular_order", {})
    lam2 = pre["theory_2lam"]
    k12 = next(r for r in extra if r["k"] == 12)
    pk = [r["peak_von_mises"] for r in ext if r["k"] >= 4]
    slope = float(np.polyfit(np.log([r["h"] for r in ext if r["k"] >= 4]), np.log(pk), 1)[0])
    met = {
        "P1": po[-1] is not None and abs(po[-1] - lam2) <= 0.15,
        "P2": abs((ml.get("fit_order") or 0) - lam2) <= 0.15 and sg.get("verdict") == "PASS"
        and after["strain_energy"][0] == ("PASS", "rate"),
        "P3": after["w_tip"][0] == ("PASS", "rate"),
        "P4": pk[-1] > pk[-2] and k12["peak_location"][0] < 0.1
        and abs(slope - pre["lam_minus_1"]) <= 0.15,
        "P5": after["root_gk_avg_von_mises"][0][0] == "FAIL",
        "P6": k12["seconds"] < 7200,
    }
    res = {"pairwise_orders": po, "fit_order_energy": ml.get("fit_order"),
           "singular_order_energy": {k: sg.get(k) for k in ("verdict", "observed", "label")},
           "peak_slope_k4_12": slope, "k12_seconds": k12["seconds"], "k12_n_dof": k12["n_dof"],
           "verdicts": {q: {"before": before[q][0], "after": after[q][0]} for q in before},
           "predictions_met": met}
    json.dump(res, open(os.path.join(HERE, "block3d_followup_result.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    print(json.dumps(score(), indent=1)) if "--score" in sys.argv else main()
