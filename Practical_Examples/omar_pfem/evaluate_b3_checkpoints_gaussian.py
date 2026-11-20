"""Surrogate errors of the B3 groove stresses under the GAUSSIAN-KERNEL region
definition (verification agent, `gk2_*` QoIs), for every u-only checkpoint,
against the cached 950,400-element reference. No new FEM solve.

Why: the verification agent rate-certifies the Gaussian-weighted groove
averages (weight exp(-(d / (2 R0))^2), R0 = 2 x groove curvature radius,
centred at the groove root; Gauss points within 0.25 of the root), while
the sharp-sphere averages are only band-stable. The surrogate errors in the
paper are for the sharp sphere. This script measures the surrogates' errors
for the Gaussian definition, so that `requirements.py` can record them
(per-QoI requirement = best surrogate error / 3).

Definition: identical to the agent's (verification_agent/region_study.py,
`weights(rec, "gauss", 2 * R0)` and `qois(rec, w)`), applied to the Gauss
points of the evaluation mesh. Cauchy stress per Gauss point from the
network's (or reference's) displacement field with the same autograd
P -> sigma = P F^T / det F construction as every other B3 evaluation
(evaluate_B3_qois.compute_region_sigma_full); the sharp-sphere averages are
reported next to them from the same points.

Errors per sample: |pred - true| / |true| (self-relative) and
|pred - true| / (Frobenius-average of the true stress, same weight)
(scale-relative: the normal stresses can be small; the agent judges the
Gaussian averages on that scale). Summary per checkpoint: mean and median
over samples, and pooled RMS.

Caveat: the reference here is the hex8 950,400-element mesh (the paper's
evaluation reference); the agent's certified Gaussian uncertainty
(0.11-0.18%) is for the Q20 family. The difference of the two references is
not part of this script.

Cluster usage (reuses the cache of the re-evaluation run):
  python -u -m omar_pfem.evaluate_b3_checkpoints_gaussian \\
      --run_dir /scratch/gole3157/pfem_run --n_samples 10 \\
      --out_json /scratch/gole3157/pfem_run/b3_rabczuk_reeval_v2.json \\
      --results_out /scratch/gole3157/pfem_run/b3_gaussian_checkpoints.json
CPU smoke test (small mesh, any run directory with the same layout):
  python -m omar_pfem.evaluate_b3_checkpoints_gaussian --run_dir <dir> --n_samples 1 \\
      --out_json <tmp>/x.json --fine_resolution 13 12 19 --cpu
"""
import argparse
import json
import os

import numpy as np
import torch

from omar_pfem.data.data_generate_B3 import groove_radius_of_curvature
from omar_pfem.data.data_generate_B3_dataset import GROOVE_DEPTH, GROOVE_HALF_WIDTH
from omar_pfem.data.mesh_convergence_B3 import LZ, R_IN0
from omar_pfem.evaluate_B3_qois import compute_region_sigma_full
from omar_pfem.evaluate_B3_qois_finer import (FINE_RESOLUTION_STRESS, apply_dirichlet_b3,
                                              build_model, generate_fine_reference)
from omar_pfem.run_b3_rabczuk_reeval_v2_local import U_ONLY_NAMES
from omar_pfem.train_B1 import _apply_input_norm, install_input_norm_for_checkpoint
from omar_pfem.verification_agent import region_study as RS

DUMP_RADIUS = 0.25          # as in verification_agent/b3_global.py (Gauss-point dumps)
MEASURES = ("sigma_xx_avg", "sigma_yy_avg", "sigma_zz_avg", "pressure_avg", "cyl_rr_avg",
            "cyl_tt_avg", "cyl_zz_avg", "cyl_rz_avg", "sigma_xz_avg", "frob_avg",
            "von_mises_avg")


def gauss_positions(geom):
    """(n_elem, n_gauss, 3) reference positions of the hex8 Gauss points, in
    the order of geom["B"] (torchfem Hexa1 ipoints)."""
    from torchfem.elements import Hexa1
    e = Hexa1()
    N = e.N(e.ipoints).to(geom["nodes"].dtype).to(geom["nodes"].device)   # (n_gauss, 8)
    X = geom["nodes"][geom["elements"]]                                   # (n_elem, 8, 3)
    return torch.einsum("ga,eai->egi", N, X)


def near_groove(geom):
    ref = np.array([R_IN0 - GROOVE_DEPTH, 0.0, LZ / 2.0])
    pos = gauss_positions(geom)
    d = torch.linalg.norm(pos - torch.tensor(ref, dtype=pos.dtype, device=pos.device), dim=-1)
    mask = d < DUMP_RADIUS
    return mask, pos[mask].cpu().numpy(), ref


def averages(sigma, w, pos, ref, R0):
    """{'gk2': {...}, 'sharp': {...}} with the agent's region_study definitions."""
    rec = {"pos": pos, "ref": ref, "sigma": sigma, "w": w}
    out = {}
    for kind, size in (("gk2", 2 * R0), ("sharp", R0)):
        wt, n_eff = RS.weights(rec, "gauss" if kind == "gk2" else "sharp", size)
        q = RS.qois(rec, wt)
        q["n_points_or_effective"] = float(n_eff)
        out[kind] = q
    return out


def predict(ckpt_path, fine_ref, device, dtype):
    geom = fine_ref["geom_fine"]
    install_input_norm_for_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(argparse.Namespace(**ckpt["args"]), device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds = []
    for s in fine_ref["samples"]:
        with torch.no_grad():
            fm = torch.stack([s["E_f_t"], s["nu_f_t"],
                              s["phi_t"][:, None].expand(-1, geom["n_nodes"])], dim=2)
            u = model(geom["nodes"].unsqueeze(0), _apply_input_norm(fm))
            preds.append(apply_dirichlet_b3(u, geom, s["phi_t"]))
    return preds


def evaluate(ckpt_path, fine_ref, mask, pos, ref, R0, device, dtype, true_cache):
    geom = fine_ref["geom_fine"]
    preds = predict(ckpt_path, fine_ref, device, dtype)
    rows = []
    for s, u_pred in zip(fine_ref["samples"], preds):
        i = s["sample_index"]
        if i not in true_cache:
            sig_t, w = compute_region_sigma_full(s["u_true_f"], s["E_f_t"], s["nu_f_t"], geom, mask)
            true_cache[i] = averages(sig_t[0].cpu().numpy(), w.cpu().numpy(), pos, ref, R0)
        sig_p, w = compute_region_sigma_full(u_pred, s["E_f_t"], s["nu_f_t"], geom, mask)
        pred = averages(sig_p[0].cpu().numpy(), w.cpu().numpy(), pos, ref, R0)
        row = {"sample_index": i, "true": true_cache[i], "pred": pred, "rel_self": {},
               "rel_scale": {}}
        for kind in ("gk2", "sharp"):
            scale = abs(true_cache[i][kind]["frob_avg"])
            for m in MEASURES:
                t, p = true_cache[i][kind][m], pred[kind][m]
                row["rel_self"][f"{kind}_{m}"] = abs(p - t) / max(abs(t), 1e-30)
                row["rel_scale"][f"{kind}_{m}"] = abs(p - t) / scale
        rows.append(row)
    summary = {}
    for key in rows[0]["rel_self"]:
        a = np.array([r["rel_self"][key] for r in rows])
        b = np.array([r["rel_scale"][key] for r in rows])
        kind, m = key.split("_", 1)
        t = np.array([r["true"][kind][m] for r in rows])
        p = np.array([r["pred"][kind][m] for r in rows])
        summary[key] = {"mean_rel_self": float(a.mean()), "median_rel_self": float(np.median(a)),
                        "mean_rel_scale": float(b.mean()),
                        "pooled_rms_rel": float(np.sqrt(np.mean((p - t) ** 2) / np.mean(t ** 2)))}
    return {"checkpoint": ckpt_path, "n_samples": len(rows), "summary": summary,
            "per_sample": rows}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--out_json", required=True,
                    help="the re-evaluation run's --out_json: its 950,400-element cache "
                         "(<out_json>.fine_950400.cache.pt) is reused")
    ap.add_argument("--results_out", default=None)
    ap.add_argument("--fine_resolution", type=int, nargs=3, default=list(FINE_RESOLUTION_STRESS))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    dtype = torch.float64
    res = tuple(a.fine_resolution)
    tag = "950400" if res == tuple(FINE_RESOLUTION_STRESS) else "x".join(map(str, res))
    results_out = a.results_out or a.out_json + f".gaussian_checkpoints_{tag}.json"
    fine_ref = generate_fine_reference(f"{a.run_dir}/b3_dataset_clean_holdout/dataset.h5", res,
                                       a.n_samples, device, dtype,
                                       cache_path=a.out_json + f".fine_{tag}.cache.pt",
                                       require_region_mask=False)
    R0 = 2.0 * groove_radius_of_curvature(GROOVE_DEPTH, GROOVE_HALF_WIDTH)
    mask, pos, ref = near_groove(fine_ref["geom_fine"])
    print(f"device {device}; mesh {res}; {int(mask.sum())} Gauss points within {DUMP_RADIUS} "
          f"of the groove root; R0 = {R0:.5f}, kernel length 2 R0 = {2 * R0:.5f}")
    results = json.load(open(results_out)) if os.path.exists(results_out) else {}
    results.setdefault("_definition", {
        "kernel": "w * exp(-(d / (2 R0))^2), d = distance to the groove root",
        "R0": R0, "dump_radius": DUMP_RADIUS, "groove_root": ref.tolist(),
        "fine_resolution": list(res), "n_gauss_near": int(mask.sum())})
    true_cache = {}
    for name, rel in U_ONLY_NAMES.items():
        if name in results:
            print(f"[skip, done] {name}")
            continue
        ckpt = os.path.join(a.run_dir, rel)
        if not os.path.exists(ckpt):
            print(f"[missing] {name}: {ckpt}")
            continue
        results[name] = evaluate(ckpt, fine_ref, mask, pos, ref, R0, device, dtype, true_cache)
        json.dump(results, open(results_out, "w"), indent=1)
        s = results[name]["summary"]
        print(f"{name:<62} gk2 xx {100 * s['gk2_sigma_xx_avg']['mean_rel_self']:9.1f}%  "
              f"zz {100 * s['gk2_sigma_zz_avg']['mean_rel_self']:9.1f}%  p "
              f"{100 * s['gk2_pressure_avg']['mean_rel_self']:9.1f}%  | sharp xx "
              f"{100 * s['sharp_sigma_xx_avg']['mean_rel_self']:9.1f}%", flush=True)
    best = {}
    for name, r in results.items():
        if name.startswith("_"):
            continue
        for key, v in r["summary"].items():
            if key.startswith("gk2_") and (key not in best or v["mean_rel_self"] < best[key][0]):
                best[key] = (v["mean_rel_self"], name)
    results["_best_gk2_mean_rel_self"] = {k: {"error": e, "checkpoint": n}
                                          for k, (e, n) in best.items()}
    json.dump(results, open(results_out, "w"), indent=1)
    print("\nbest (smallest) mean self-relative error per Gaussian QoI, for requirements.py:")
    for k, (e, n) in sorted(best.items()):
        print(f"  {k:<24} {100 * e:9.2f}%  ({n})")
    print(f"-> {results_out}")


if __name__ == "__main__":
    main()
