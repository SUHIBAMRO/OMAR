"""Groove-surface peak stress for B3, from nodal (extrapolated) stresses.

The *_max QoIs of the B3 adapter are maxima over the Gauss points inside
the averaging sphere, which lie about 0.2 h inside the body; the true peak
of the finite-radius groove (rho = 0.0228) lies ON the groove surface. Here
the 2x2x2 Gauss stresses of every element touching the groove surface are
extrapolated to its corner nodes (b3_global.surface_nodal_stress, exact for
fields linear in the parent coordinates) and averaged over the elements
sharing a node; the surface peak is the maximum over the surface nodes
within R0 of the groove root.

    python -m omar_pfem.verification_agent.surface_study
"""
import glob
import json
import os

import numpy as np

from omar_pfem.verification_agent import gates

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER_WINDOW = (0.5, 5)
BAND, GCI = 0.01, 0.02        # region/peak class ceilings


def measures(sig):
    p = -np.trace(sig, axis1=1, axis2=2) / 3.0
    dev = sig + p[:, None, None] * np.eye(3)
    return {"sxx": sig[:, 0, 0], "szz": sig[:, 2, 2], "sxz": sig[:, 0, 2],
            "von_mises": np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev)), "pressure": p}


def level_values(path):
    d = np.load(path)
    dist = np.linalg.norm(d["pos"] - d["ref"][None, :], axis=1)
    inside = dist < float(d["radius"])
    m = measures(d["sigma"])
    root = int(np.argmin(dist))
    out = {"n_elem": int(d["n_elem"]), "n_surface_nodes": int(inside.sum()),
           "root_node_distance": float(dist[root])}
    for k, v in m.items():
        out[f"surface_max_abs_{k}"] = float(np.abs(v[inside]).max())
        out[f"root_{k}"] = float(v[root])
    return out


def main():
    rows = sorted((level_values(f) for f in glob.glob(os.path.join(HERE, "b3_surface", "*.npz"))),
                  key=lambda r: r["n_elem"])
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} levels with surface data")
    ext = json.load(open(os.path.join(HERE, "b3_extended.json")))
    fam = {r["n_elem"]: r for r in ext if r["z_beta"] is None and r["rtol"] == 1e-8}
    scale = fam[max(fam)]["region_avg_sigma_frob"]
    hs = [r["n_elem"] ** (-1.0 / 3.0) for r in rows]
    print(f"levels {[r['n_elem'] for r in rows]}; surface nodes within R0: "
          f"{[r['n_surface_nodes'] for r in rows]}; stress scale {scale:.4f}")
    print(f"root-node distance from the groove root: "
          f"{[round(r['root_node_distance'], 4) for r in rows]}")
    out = {"levels": rows, "scale": scale, "gates": {}}
    for k in [k for k in rows[0] if k.startswith(("surface_max", "root_"))]:
        vals = [r[k] for r in rows]
        g3 = gates.qoi_convergence_gate(hs, vals, BAND, GCI, scale=scale, order_window=ORDER_WINDOW)
        g5 = gates.multilevel_convergence_gate(hs, vals, BAND, GCI, scale=scale,
                                               order_window=ORDER_WINDOW)
        overall, evidence, _ = gates.combine_verdicts({"qoi_convergence": g3,
                                                       "qoi_convergence_multilevel": g5})
        out["gates"][k] = {"values": vals, "3level": g3, "5level": g5, "overall": overall,
                           "evidence": evidence}
        print(f"  {k:<26} {overall:<5} {evidence:<5} "
              f"{' '.join(f'{v:8.4f}' for v in vals)}\n{'':<30}3: {g3['reason'][:80]}\n"
              f"{'':<30}5: {g5['reason'][:80]}")
    # Gauss-point maxima of the stored QoIs, for comparison
    gp = [fam[r["n_elem"]] for r in rows if r["n_elem"] in fam]
    print("Gauss-point max|sigma_xx| in the sphere (stored QoI): "
          f"{[round(g['per_component']['xx']['max'], 4) for g in gp]}")
    print("Gauss-point max von Mises in the sphere (stored QoI): "
          f"{[round(g['region_max_von_mises'], 4) for g in gp]}")
    json.dump(out, open(os.path.join(HERE, "surface_study_result.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
