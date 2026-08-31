"""Two error metrics on the SAME validation samples, for one checkpoint.

WHY THIS EXISTS. The energy-vs-error run produced a contradiction that has
to be resolved before anything is concluded from it. On the four probe
samples of each resolution, the epoch-450 model beat the epoch-50 model on
EVERY ONE -- relative L2 0.5877 -> 0.0930 on one of them -- while the
trainer's own validation number said epoch 450 was 1.27x WORSE. The two
numbers rank the same two models in opposite directions.

They are not the same metric:

    the trainer      0.5 * ( rms(e_u)/rms(u) + rms(e_v)/rms(v) )
    the probe        rms(e) / rms(uv_exact)          both components at once

The first is an average of two per-component ratios, each normalised by its
OWN component. If one displacement component is much smaller than the other,
that component's ratio dominates the average and the metric mostly reports
the weaker component. The second weights the components by their actual
size. Neither is wrong as a definition; they answer different questions, and
the trainer's is what early stopping and every reported B2 zero-shot number
were built on.

So this script computes BOTH on the SAME samples -- all of them, not four --
plus the two per-component ratios separately and the size of each component,
which is what would show a small-component effect if there is one.

It settles which of these is true:

  * the trainer's metric ranks these models backwards, in which case early
    stopping has been keeping the WORSE model on every B2 run and the
    reported ~1.0 errors need re-reading;
  * or the four probe samples were unrepresentative and the trainer's number
    is right on the full set.

Usage:
  python -m omar_pfem.compare_val_metrics --geometry B2 \
      --cache <samples_cache.pt> --checkpoint <model.pt> --cpu
"""
import argparse
import json

import numpy as np
import torch

from omar_pfem.model_dict import get_model
from omar_pfem.resolution_invariance_zeroshot import (
    loss_and_pred, mesh_tensors_of)


def build_model(args, device):
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref,
        unified_pos=args.unified_pos).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", default="B2", choices=["B1", "B2"])
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--material", default="neo_hookean")
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--mode", default="plane_strain")
    p.add_argument("--use_soft_dirichlet", type=int, default=1)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_json", default=None)
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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    dtype = torch.float32

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    assert isinstance(cache, dict) and "val_samples" in cache, list(cache)[:8]
    buckets = {int(N): v for N, v in sorted(cache["val_samples"].items())}

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"checkpoint : {args.checkpoint}")
    print(f"cache      : {args.cache}")

    out = {"checkpoint": args.checkpoint, "cache": args.cache, "rows": []}
    for N, samples in buckets.items():
        mesh = mesh_tensors_of(args.geometry, samples[0], device, dtype)
        # Per-sample accumulators. The trainer averages its per-sample ratio
        # over samples, so the same per-sample-then-average order is kept
        # here -- averaging the other way round would be a different number
        # and the point is to reproduce the trainer's exactly.
        tr, comb, ru_all, rv_all, u_sz, v_sz = [], [], [], [], [], []
        for i0 in range(0, len(samples), args.batch):
            chunk = samples[i0:i0 + args.batch]
            E = torch.tensor(np.stack([s["E_node"] for s in chunk]),
                             device=device, dtype=dtype)
            nu = torch.tensor(np.stack([s["nu_node"] for s in chunk]),
                              device=device, dtype=dtype)
            f = torch.tensor(np.stack([s["node_forces"] for s in chunk]),
                             device=device, dtype=dtype)
            tgt = torch.tensor(np.stack([s["uv_exact"] for s in chunk]),
                               device=device, dtype=dtype)
            with torch.no_grad():
                _, _, _, pred, _ = loss_and_pred(
                    args.geometry, mesh, model, E, nu, f, args, dtype)
            err = pred - tgt
            l2_u = torch.sqrt(torch.mean(err[:, :, 0] ** 2, dim=1))
            l2_v = torch.sqrt(torch.mean(err[:, :, 1] ** 2, dim=1))
            ref_u = torch.sqrt(torch.mean(tgt[:, :, 0] ** 2, dim=1)) + 1e-12
            ref_v = torch.sqrt(torch.mean(tgt[:, :, 1] ** 2, dim=1)) + 1e-12
            # THE TRAINER'S METRIC, copied from evaluate_resolution so the
            # comparison cannot drift from what early stopping actually used.
            tr.append(0.5 * (l2_u / ref_u + l2_v / ref_v))
            # BOTH COMPONENTS AT ONCE, which is what the probe reports.
            comb.append(torch.sqrt(torch.mean(err ** 2, dim=(1, 2)))
                        / torch.sqrt(torch.mean(tgt ** 2, dim=(1, 2)))
                        .clamp_min(1e-30))
            ru_all.append(l2_u / ref_u)
            rv_all.append(l2_v / ref_v)
            u_sz.append(ref_u)
            v_sz.append(ref_v)

        def cat(xs):
            return torch.cat(xs).cpu().numpy()

        tr, comb = cat(tr), cat(comb)
        ru, rv, us, vs = cat(ru_all), cat(rv_all), cat(u_sz), cat(v_sz)
        row = {"N": N, "n_val": int(len(tr)),
               "trainer_metric": float(tr.mean()),
               "combined_rel_L2": float(comb.mean()),
               "rel_u": float(ru.mean()), "rel_v": float(rv.mean()),
               "rms_u": float(us.mean()), "rms_v": float(vs.mean()),
               "v_over_u": float((vs / us).mean())}
        out["rows"].append(row)
        print(f"\nresolution {N}   ({row['n_val']} val samples)")
        print(f"  trainer's metric  0.5*(rel_u + rel_v)   "
              f"{row['trainer_metric']:.4f}   <- early stopping used this")
        print(f"  both components at once, rms(e)/rms(uv) "
              f"{row['combined_rel_L2']:.4f}   <- what the probe prints")
        print(f"  rel_u {row['rel_u']:.4f}   rel_v {row['rel_v']:.4f}")
        print(f"  rms(u) {row['rms_u']:.4e}   rms(v) {row['rms_v']:.4e}"
              f"   rms(v)/rms(u) {row['v_over_u']:.4f}")

    if out["rows"]:
        m = out["rows"]
        out["mean_trainer_metric"] = float(
            np.mean([r["trainer_metric"] for r in m]))
        out["mean_combined_rel_L2"] = float(
            np.mean([r["combined_rel_L2"] for r in m]))
        print(f"\nmean over resolutions: trainer "
              f"{out['mean_trainer_metric']:.4f}   combined "
              f"{out['mean_combined_rel_L2']:.4f}")
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"written to {args.out_json}")


if __name__ == "__main__":
    main()
