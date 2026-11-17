"""Test of the region-definition hypothesis for the B3 groove averages.

Hypothesis (2026-11-17): the region-averaged groove stresses oscillate
under refinement (no demonstrable rate) because the averaging region is a
small sharp-edged sphere (radius R0 = 2 x groove curvature radius, centred
on the groove-root surface; 2-36 Gauss points from 2,376 to 45,000
elements), so which Gauss points fall inside changes from mesh to mesh.

Falsifiable predictions:
  P1 a smooth Gaussian weight of the same nominal size damps the
     oscillation (smaller finest-three spread, more QoIs with a rate);
  P2 larger sharp spheres damp it too (more points inside).
If the smooth-kernel averages oscillate as much as the sharp ones, the
oscillation is a property of the stress field, and the hypothesis fails.

All region definitions are evaluated on the SAME solutions (Gauss-point
data dumped by b3_global.py dump), so differences are due to the region
definition alone.

    python -m omar_pfem.verification_agent.region_study
"""
import glob
import json
import os

import numpy as np

from omar_pfem.verification_agent import gates

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "b3_gauss_near_groove")
ORDER_WINDOW = (0.5, 5)        # Q20
BAND, GCI, ROBUST = 0.01, 0.02, 0.02   # region_average ceilings (trees.CLASS_CEILINGS)


def load():
    fam, strat = [], []
    for f in glob.glob(os.path.join(DDIR, "*.npz")):
        d = np.load(f)
        zb = os.path.basename(f).split("_zb")[1][:-4]
        rec = {k: d[k] for k in d.files}
        rec["n_elem"] = int(d["n_elem"])
        rec["z_beta"] = None if zb == "None" else float(zb)
        (fam if rec["z_beta"] is None else strat).append(rec)
    fam.sort(key=lambda r: r["n_elem"])
    strat.sort(key=lambda r: r["z_beta"])
    return fam, strat


def definitions(R0):
    defs = {f"sharp {k}R0": ("sharp", k * R0) for k in (1, 2, 3, 4)}
    defs.update({f"gauss l={k}R0": ("gauss", k * R0) for k in (1, 2)})
    return defs


def weights(rec, kind, size):
    d = np.linalg.norm(rec["pos"] - rec["ref"][None, :], axis=1)
    if kind == "sharp":
        return rec["w"] * (d < size), int((d < size).sum())
    k = np.exp(-(d / size) ** 2)
    return rec["w"] * k, float(k.sum() ** 2 / (k ** 2).sum())  # effective point count


def qois(rec, wt):
    s = rec["sigma"]
    W = wt.sum()
    avg = lambda v: float((v * wt).sum() / W)
    p = -np.trace(s, axis1=1, axis2=2) / 3.0
    dev = s + p[:, None, None] * np.eye(3)
    th = np.arctan2(rec["pos"][:, 1], rec["pos"][:, 0])
    c, sn = np.cos(th), np.sin(th)
    Q = np.zeros((len(th), 3, 3))
    Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1], Q[:, 2, 2] = c, sn, -sn, c, 1.0
    sc = np.einsum("nai,nij,nbj->nab", Q, s, Q)
    return {
        "sigma_xx_avg": avg(s[:, 0, 0]), "sigma_yy_avg": avg(s[:, 1, 1]),
        "sigma_zz_avg": avg(s[:, 2, 2]), "pressure_avg": avg(p),
        "cyl_rr_avg": avg(sc[:, 0, 0]), "cyl_tt_avg": avg(sc[:, 1, 1]),
        "cyl_rz_avg": avg(sc[:, 0, 2]),
        "frob_avg": avg(np.sqrt(np.einsum("nij,nij->n", s, s))),
        "von_mises_avg": avg(np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))),
    }


def main():
    fam, strat = load()
    if len(fam) < 3:
        raise SystemExit(f"only {len(fam)} family levels dumped")
    R0 = float(fam[0]["radius"])
    hs = [r["n_elem"] ** (-1.0 / 3.0) for r in fam]
    base = next((r for r in fam if r["n_elem"] == 10200), None)
    strategies = ([base] + strat) if (base is not None and len(strat) == 3) else None
    print(f"levels: {[r['n_elem'] for r in fam]}; strategies: "
          f"{'uniform + z_beta 1,2,3 at 10,200' if strategies else 'not yet available'}")
    out = {}
    for name, (kind, size) in definitions(R0).items():
        rows = [qois(r, weights(r, kind, size)[0]) for r in fam]
        npts = [weights(r, kind, size)[1] for r in fam]
        srows = [qois(r, weights(r, kind, size)[0]) for r in strategies] if strategies else None
        scale = rows[-1]["frob_avg"]
        res = {}
        for q in rows[0]:
            vals = [r[q] for r in rows]
            g = gates.qoi_convergence_gate(hs, vals, BAND, GCI, scale=scale,
                                           order_window=ORDER_WINDOW)
            rec = {"values": vals, "conv": g}
            if srows:
                sv = [r[q] for r in srows]
                rec["strategy"] = gates.strategy_robustness_gate(sv, ROBUST, scale=scale)
                anchor, kind_a = gates.family_anchor(g)
                rec["anchored"] = gates.anchored_robustness_gate(sv, anchor, ROBUST, scale=scale,
                                                                 anchor_kind=kind_a)
            vs = [rec[k]["verdict"] for k in ("conv", "strategy", "anchored") if k in rec]
            rec["overall"] = ("FAIL" if "FAIL" in vs else
                              "PASS" if all(v == "PASS" for v in vs) else "INCONCLUSIVE")
            res[q] = rec
        out[name] = {"kind": kind, "size": size, "points": npts, "scale": scale, "qois": res}

        # summary line per definition
        n_rate = sum(r["overall"] == "PASS" and r["conv"]["evidence"] == "rate" for r in res.values())
        n_band = sum(r["overall"] == "PASS" and r["conv"]["evidence"] == "band" for r in res.values())
        n_fail = sum(r["overall"] == "FAIL" for r in res.values())
        osc = np.median([r["conv"]["relative_spread"] for r in res.values()])
        pts = npts[-1]
        print(f"\n{name:<14} points at finest {pts:>8.1f}   PASS(rate) {n_rate}  PASS(band) "
              f"{n_band}  FAIL {n_fail}   median finest-three spread {osc:.3%}")
        for q, r in res.items():
            c = r["conv"]
            extra = "".join(f"  {k[:5]} {r[k]['verdict']}" for k in ("strategy", "anchored") if k in r)
            print(f"   {q:<15} {r['overall']:<5} {c['evidence']:<5} {c['reason'][:70]}{extra}")
    json.dump(out, open(os.path.join(HERE, "region_study_result.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
