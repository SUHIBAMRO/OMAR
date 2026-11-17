"""Groove-surface peak stress for B3, from nodal (extrapolated) stresses.

The *_max QoIs of the B3 adapter are maxima over the Gauss points inside
the averaging sphere, which lie about 0.2 h inside the body; the true peak
of the finite-radius groove (rho = 0.0228) lies ON the groove surface. Here
the 2x2x2 Gauss stresses of every element touching the groove surface are
extrapolated to ALL 20 nodes (corner and midside) with the trilinear
interpolant through the 8 Gauss points (exact for fields trilinear in the
parent coordinates) and averaged over the elements sharing a node; the
surface peak is the maximum over the groove-surface nodes within R0 of the
groove root, and the root value is taken at the node AT the root.

Midside nodes matter: with an even number of z node layers (e.g. 25x20x36,
48x34x66) no CORNER node lies at z = 0.5, so a corner-only extrapolation
has no node at the root, and its "root" value moves by ~0.015 between
levels; a z-direction midside node then sits exactly at the root.

The Gauss-point stresses are taken post hoc from the b3_gauss_near_groove/
dumps (written by the solves), matched to the rebuilt (deterministic) mesh
by position (match distance must be < 1e-10); no re-solve. The
solver-time corner-only files b3_surface/ serve as a cross-check.

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


def extrapolation_matrix(etype, n_nodes=20):
    """X[n, q]: value at element node n of the trilinear interpolant through
    the 2x2x2 Gauss points q (Gauss coordinates +-g in each direction)."""
    ip = etype.ipoints.numpy()
    if ip.shape[0] != 8:
        raise ValueError("2x2x2 rule only")
    g = abs(float(ip[0, 0]))
    xi = etype.iso_coords[:n_nodes].numpy()
    return np.prod((1.0 + np.sign(ip)[None, :, :] * xi[:, None, :] / g) / 2.0, axis=2)


def surface_nodal_from_dump(res, dump_path, n_nodes=20):
    """Nodal Cauchy stress on the groove surface near the root, from the
    Gauss-point dump of the solve at resolution res = (Ntheta, Nr, Nz).
    Returns (pos (n,3), sigma (n,3,3), ref, radius, n_elem, match_distance)."""
    import torch
    from torchfem.elements import Hexa2
    from omar_pfem.verification_agent import b3_global as b
    from omar_pfem.data.solve_quadratic_hex_groove_stress import (
        build_quadratic_mesh_with_curved_surfaces)
    from scipy.spatial import cKDTree
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        nodes, el, inner, _, _ = build_quadratic_mesh_with_curved_surfaces(
            *res, groove_depth=b.GROOVE_DEPTH)
        et = Hexa2()
        N = et.N(et.ipoints).numpy()                       # (8, 20)
        X = extrapolation_matrix(et, n_nodes)               # (n_nodes, 8)
    finally:
        torch.set_default_dtype(old)
    d = np.load(dump_path)
    ref, radius = d["ref"], float(d["radius"])
    on_inner = np.zeros(len(nodes), bool)
    on_inner[inner] = True
    # every element containing a groove-surface node within R0 is used, so each
    # such node is averaged over ALL its elements; nodes farther out are dropped
    close = on_inner & (np.linalg.norm(nodes - ref[None, :], axis=1) < radius)
    sel = np.where(close[el].any(axis=1))[0]
    gp = np.einsum("gn,enD->egD", N, nodes[el[sel]])
    dist, idx = cKDTree(d["pos"]).query(gp.reshape(-1, 3))
    if dist.max() > 1e-10:
        raise ValueError(f"{dump_path}: Gauss points not matched (max distance {dist.max():.2e})")
    sig = d["sigma"][idx].reshape(len(sel), 8, 3, 3)
    node_sig = np.einsum("nq,eqij->enij", X, sig)          # (n_sel, n_nodes, 3, 3)
    acc = np.zeros((len(nodes), 3, 3))
    cnt = np.zeros(len(nodes))
    for k in range(n_nodes):
        np.add.at(acc, el[sel, k], node_sig[:, k])
        np.add.at(cnt, el[sel, k], 1.0)
    keep = close & (cnt > 0)
    return (nodes[keep], acc[keep] / cnt[keep][:, None, None], ref, radius,
            int(d["n_elem"]), float(dist.max()))


def level_values(pos, sigma, ref, radius, n_elem):
    dist = np.linalg.norm(pos - ref[None, :], axis=1)
    inside = dist < radius
    m = measures(sigma)
    root = int(np.argmin(dist))
    out = {"n_elem": int(n_elem), "n_surface_nodes": int(inside.sum()),
           "root_node_distance": float(dist[root])}
    for k, v in m.items():
        out[f"surface_max_abs_{k}"] = float(np.abs(v[inside]).max())
        out[f"root_{k}"] = float(v[root])
    return out


def cross_check(res, pos20, sig20):
    """Corner nodes: post-hoc values vs the solver-time corner-only file."""
    f = os.path.join(HERE, "b3_surface", "{}_{}_{}.npz".format(*res))
    if not os.path.exists(f):
        return None
    s = np.load(f)
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(pos20).query(s["pos"])
    ok = dist < 1e-12
    return float(np.abs(sig20[idx[ok]] - s["sigma"][ok]).max()) if ok.any() else None


def main():
    rows = []
    for f in glob.glob(os.path.join(HERE, "b3_gauss_near_groove", "*_zbNone.npz")):
        res = tuple(int(x) for x in os.path.basename(f).split("_zb")[0].split("_"))
        pos, sig, ref, radius, n_elem, mdist = surface_nodal_from_dump(res, f)
        r = level_values(pos, sig, ref, radius, n_elem)
        r["resolution"], r["gauss_match_distance"] = list(res), mdist
        r["corner_check_vs_solver_time"] = cross_check(res, pos, sig)
        rows.append(r)
    rows.sort(key=lambda r: r["n_elem"])
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} levels with surface data")
    print("corner nodes, post hoc vs solver-time extrapolation (max |diff|): "
          f"{[r['corner_check_vs_solver_time'] for r in rows]}")
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
