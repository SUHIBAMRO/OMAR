"""Why the operator degrades out of distribution -- the advisor's point 6.

`evaluate_ood.py` already answers *how much*: a 4-5x degradation factor.
Point 6 asks what is behind it, since that factor is "probably the biggest
obstacle to a strong physics-informed operator claim". A single ratio cannot
say whether the network has lost the shape of the solution or merely its
scale, and those call for completely different responses -- so this script
decomposes the same predictions instead of producing another aggregate.

It runs on the datasets `evaluate_ood.py` already uses and performs no new
FEM solves, so the first cut of the diagnosis is essentially free.

Three decompositions, per sample, in and out of distribution:

1. **Magnitude vs shape.** ||pred||/||ref|| says whether the prediction is
   the right size; the residual after the best possible rescaling,
   ||a*pred - ref||/||ref||, says how much error no calibration could remove.
   If OOD error is mostly magnitude the network still understands the field
   and is merely mis-scaled -- recoverable in principle by one scalar. If it
   is mostly shape, it does not, and rescaling cannot help. Those have very
   different implications for the paper's claim, and a single ratio hides
   which is happening.

   Magnitude and the rescaling factor are deliberately kept apart. A unit
   test caught the temptation to read |a - 1| as the magnitude error: noise
   orthogonal to the reference leaves the magnitude untouched yet still
   drags a below 1, because it enlarges <pred, pred>. So size is read from
   norm_ratio and a only answers "would rescaling help".

2. **Under- or over-prediction.** norm_ratio is reported signed, as a mean
   over samples rather than folded into an absolute value, so a systematic
   bias in one direction is visible as such rather than as scatter.

3. **Where the error sits.** Relative error split by distance from the
   constrained boundary, in four bands. Physics-informed training enforces
   equilibrium through an energy integral, so error concentrating far from
   the Dirichlet boundary means something different from error at the load.

Usage:
  python -m omar_pfem.ood_diagnosis \
      --geometry B1 --material neo_hookean \
      --checkpoint .../model_best.pt \
      --id_path .../dataset.npz --id_ntrain 800 \
      --ood_path .../ood.npz --ntest 200 \
      --out_json .../ood_diagnosis_B1_neo_hookean.json
"""
import os
import json
import time
import argparse

import numpy as np
import torch

from omar_pfem.run_manifest import write_manifest
from omar_pfem.model_dict import get_model


def decompose(pred, ref):
    """Split one sample's error into a magnitude part and a shape part.

    Two distinct quantities, and conflating them was the first thing a unit
    test caught here:

    `norm_ratio` = ||pred||/||ref|| is the magnitude answer. It is 1 exactly
    when the prediction has the right size, whatever its shape.

    `alpha` is the least-squares scalar best matching pred to ref, so a*pred
    is as close as rescaling alone can get and `shape_rel_L2` is the error
    that survives it -- the part no calibration could remove. Note alpha is
    NOT a magnitude measure: adding noise orthogonal to ref leaves the
    magnitude untouched but still pulls alpha below 1, since the noise
    enlarges the denominator <pred,pred>. Read norm_ratio for size and
    shape_rel_L2 for shape; alpha only answers "how much would rescaling
    help".
    """
    p = np.asarray(pred, dtype=float).ravel()
    r = np.asarray(ref, dtype=float).ravel()
    nr = np.linalg.norm(r)
    pp = float(p @ p)
    a = float(p @ r) / pp if pp > 0 else 0.0
    return {
        "rel_L2": float(np.linalg.norm(p - r) / nr) if nr > 0 else float("nan"),
        "norm_ratio": float(np.linalg.norm(p) / nr) if nr > 0 else float("nan"),
        "alpha": a,
        "shape_rel_L2": (float(np.linalg.norm(a * p - r) / nr)
                         if nr > 0 else float("nan")),
    }


def band_errors(pred, ref, dist, n_bands=4):
    """Relative error in bands of increasing distance from the constraint."""
    p = np.asarray(pred, dtype=float)
    r = np.asarray(ref, dtype=float)
    edges = np.quantile(dist, np.linspace(0, 1, n_bands + 1))
    out = {}
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        m = (dist >= lo) & (dist <= hi if b == n_bands - 1 else dist < hi)
        if not m.any():
            continue
        nr = np.linalg.norm(r[m])
        out[f"band{b}"] = {
            "dist_range": [float(lo), float(hi)],
            "n_nodes": int(m.sum()),
            "rel_L2": float(np.linalg.norm(p[m] - r[m]) / nr) if nr > 0 else float("nan"),
        }
    return out


def summarize(per_sample):
    keys = ["rel_L2", "norm_ratio", "alpha", "shape_rel_L2"]
    out = {}
    for k in keys:
        v = np.array([s[k] for s in per_sample], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max())}
    bands = {}
    for b in per_sample[0]["bands"]:
        v = np.array([s["bands"][b]["rel_L2"] for s in per_sample
                      if b in s["bands"]], dtype=float)
        bands[b] = {"mean": float(v.mean()),
                    "dist_range": per_sample[0]["bands"][b]["dist_range"]}
    out["bands"] = bands
    return out


def run_split(samples, model, args, device, dtype, predict, bnd, geo_kw, xy, quad, dist):
    per = []
    for s in samples:
        E = torch.tensor(s["E_node"][None], device=device, dtype=dtype)
        nu = torch.tensor(s["nu_node"][None], device=device, dtype=dtype)
        f = torch.tensor(s["node_forces"][None], device=device, dtype=dtype)
        with torch.no_grad():
            uv = predict(xy, quad, *bnd, model, E, nu, f,
                         use_soft_dirichlet=args.use_soft_dirichlet,
                         dtype=dtype, fun_dim=args.fun_dim, **geo_kw)
        pred = uv[0].cpu().numpy()
        ref = np.asarray(s["uv_exact"]).reshape(pred.shape)
        rec = decompose(pred, ref)
        rec["bands"] = band_errors(pred, ref, dist)
        per.append(rec)
    return per


def main():
    p = argparse.ArgumentParser("Why the operator degrades OOD (advisor point 6)")
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", required=True,
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--id_path", required=True)
    p.add_argument("--id_ntrain", type=int, default=800,
                   help="must match the checkpoint's own --ntrain, or the "
                        "'in-distribution' split would include training samples")
    p.add_argument("--ood_path", required=True)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--model", default="Transolver_Irregular_Mesh")
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unified_pos", type=int, default=0)
    p.add_argument("--ref", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=128)
    p.add_argument("--fun_dim", type=int, default=4)
    p.add_argument("--use_soft_dirichlet", type=int, default=1)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    args = p.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32
    if args.out_json is None:
        args.out_json = f"ood_diagnosis_{args.geometry}_{args.material}.json"

    if args.geometry == "B1":
        from omar_pfem.train_B1 import (
            predict_displacement_Q4_only as predict,
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds)
    else:
        from omar_pfem.train_B2 import (
            predict_displacement_Q4_only as predict,
            load_fem_dataset_Q4_with_materials_and_random_force as load_ds)

    model = get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, id_test = load_ds(args.id_path, args.id_ntrain, args.ntest)
    _, ood_test = load_ds(args.ood_path, 0, args.ntest)
    print(f"in-distribution: {len(id_test)} samples, OOD: {len(ood_test)}")

    s0 = id_test[0]
    xy = torch.tensor(s0["xy"], device=device, dtype=dtype)
    quad = torch.tensor(s0["quad"], device=device, dtype=torch.long)
    nodes = np.asarray(s0["xy"], dtype=float)
    if args.geometry == "B1":
        bnd = (torch.tensor(s0["top_edges"], device=device, dtype=torch.long),
               torch.tensor(s0["bottom_nodes"], device=device, dtype=torch.long))
        geo_kw = {"Ly": args.Ly}
        constrained = np.asarray(s0["bottom_nodes"], dtype=int)
    else:
        bnd = (torch.tensor(s0["inner_edges"], device=device, dtype=torch.long),
               torch.tensor(s0["theta0_nodes"], device=device, dtype=torch.long),
               torch.tensor(s0["thetahalfpi_nodes"], device=device, dtype=torch.long))
        geo_kw = {"R_out": args.R_out}
        constrained = np.unique(np.concatenate(
            [np.asarray(s0["theta0_nodes"], dtype=int),
             np.asarray(s0["thetahalfpi_nodes"], dtype=int)]))
    # distance of each node from the nearest constrained node
    dist = np.min(np.linalg.norm(
        nodes[:, None, :] - nodes[constrained][None, :, :], axis=2), axis=1)

    res = {}
    for name, split in (("in_distribution", id_test), ("out_of_distribution", ood_test)):
        per = run_split(split, model, args, device, dtype, predict, bnd, geo_kw,
                        xy, quad, dist)
        res[name] = summarize(per)

    idm, oodm = res["in_distribution"], res["out_of_distribution"]
    factor = oodm["rel_L2"]["mean"] / idm["rel_L2"]["mean"]

    print("\n" + "=" * 74)
    print(f"OOD DIAGNOSIS  ({args.geometry} x {args.material})")
    print("=" * 74)
    print(f"{'':22}{'in-dist':>14}{'OOD':>14}{'factor':>10}")
    for k, label in (("rel_L2", "total rel L2"),
                     ("shape_rel_L2", "  shape part (irreducible)")):
        a, b = idm[k]["mean"], oodm[k]["mean"]
        print(f"{label:22}{a:>14.4e}{b:>14.4e}{b / a if a > 0 else float('nan'):>9.2f}x")
    print(f"{'magnitude ||p||/||r||':22}{idm['norm_ratio']['mean']:>14.4f}"
          f"{oodm['norm_ratio']['mean']:>14.4f}")
    print(f"{'best-fit rescale a':22}{idm['alpha']['mean']:>14.4f}"
          f"{oodm['alpha']['mean']:>14.4f}")
    print("\n  magnitude < 1 means the network UNDER-predicts the displacement.")
    print("  shape part is what would remain after the best possible rescaling:")
    print("  close to the total means recalibration cannot help.")
    print(f"\n{'error by distance from the constraint':}")
    print(f"{'band':>8}{'dist range':>22}{'in-dist':>14}{'OOD':>14}")
    for b in sorted(idm["bands"]):
        lo, hi = idm["bands"][b]["dist_range"]
        print(f"{b:>8}{f'{lo:.3f}-{hi:.3f}':>22}"
              f"{idm['bands'][b]['mean']:>14.4e}{oodm['bands'][b]['mean']:>14.4e}")
    print("=" * 74)

    report = {"geometry": args.geometry, "material": args.material,
              "checkpoint": args.checkpoint, "id_path": args.id_path,
              "ood_path": args.ood_path, "ntest": args.ntest,
              "degradation_factor": factor, **res}
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written to {args.out_json}")

    write_manifest(
        os.path.dirname(os.path.abspath(args.out_json)) or ".",
        kind="ood_diagnosis", args=args, started_at=started,
        results=report, outputs=[args.out_json],
        notes=("Advisor point 6, first cut: decomposes the degradation factor "
               "evaluate_ood.py reports rather than restating it. Splits each "
               "sample's error into the part a single rescaling could remove "
               "(scale) and the part it could not (shape), reports the sign of "
               "the scale error so systematic under-prediction is visible, and "
               "breaks error down by distance from the constrained boundary. "
               "No new FEM solves -- it reuses the same datasets."))


if __name__ == "__main__":
    main()
