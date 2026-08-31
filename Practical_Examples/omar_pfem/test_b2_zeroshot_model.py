"""What is the trained B2 zero-shot model actually doing, and what is it fed?

Where this stands. test_b2_zeroshot_functional.py settled the previous
question: Pi's minimum sits at s = 1.000 on every sample checked, at both
training resolutions, with |W|/U = 2.00 to three decimals. The data and the
functional agree. So the failure is not a mis-scaled work term and not the
cache -- it is in the training path, and this looks at that path with the
checkpoint that is already on disk. No training.

Four measurements, in the order they narrow things down.

  1. THE INPUT CHANNELS. The network is fed fun_material = (E, nu, f_x, f_y)
     RAW, with no normalization (train_B2.py, and the zero-shot trainer
     re-exports that function unchanged). For B2 the force is an inner-edge
     traction, so f is exactly zero on every node that is not on the inner
     boundary -- about 5% of the mesh at N=21. E is around 1000. If the two
     channels carrying the loading are orders of magnitude below the channel
     carrying stiffness, and nonzero on a twentieth of the nodes, the model
     may simply not see the load. This prints the scales rather than assuming
     them.

  2. WHAT IT PREDICTS. rms(pred)/rms(uv_exact) and the correlation between
     them. A combined error of 1.0 is what predicting zero scores, but it is
     also roughly what predicting noise of the right size scores, and those
     are different failures. This separates them.

  3. WHERE IT SITS ON Pi. Pi(pred) against Pi(0) = 0 and Pi(uv_exact). The
     trainer minimizes Pi. If Pi(pred) is essentially 0, the optimizer never
     descended at all. If it is part of the way down, it descended and
     stalled. Those point at different causes.

  4. DOES IT USE ITS INPUT? The same mesh, several different samples. If the
     outputs barely differ, the model has collapsed to a function of the
     coordinates alone and is ignoring (E, nu, f) -- which would explain an
     error that is flat in the mesh, flat in the material and flat in N, and
     would point at the input channels of measurement 1.

Usage:
  python -m omar_pfem.test_b2_zeroshot_model \
      --cache <samples_cache.pt> --checkpoint <model_best.pt>
"""
import argparse
import os

import torch

from omar_pfem.train_B2 import total_potential_energy_Q4_hyperelastic


def build_model(args, device):
    from omar_pfem.model_dict import get_model
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref,
        unified_pos=args.unified_pos).to(device)


def rel(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2))
                 / torch.sqrt(torch.mean(b ** 2)).clamp_min(1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--material", default="neo_hookean")
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--mode", default="plane_strain")
    p.add_argument("--use_soft_dirichlet", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    # the architecture the trainer defaults to; these must match or the
    # state dict will not load, which is itself the check
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

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    dtype = torch.float32
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    assert isinstance(cache, dict) and "val_samples" in cache, list(cache)[:8]
    buckets = {int(N): v for N, v in sorted(cache["val_samples"].items())}

    model = build_model(args, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"cache      : {args.cache}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"resolutions: " + ", ".join(f"{k} ({len(v)} val)"
                                       for k, v in buckets.items()))

    collapse = {}
    for N, samples in buckets.items():
        print("\n" + "=" * 74)
        print(f"resolution {N}")
        print("=" * 74)
        xy = torch.tensor(samples[0]["xy"], device=device, dtype=dtype)
        quad = torch.tensor(samples[0]["quad"], device=device, dtype=torch.long)
        ie = torch.tensor(samples[0]["inner_edges"], device=device, dtype=torch.long)
        t0 = torch.tensor(samples[0]["theta0_nodes"], device=device, dtype=torch.long)
        th = torch.tensor(samples[0]["thetahalfpi_nodes"], device=device, dtype=torch.long)
        preds, tgts = [], []
        for i in range(min(args.n_samples, len(samples))):
            s = samples[i]
            E = torch.tensor(s["E_node"], device=device, dtype=dtype)[None]
            nu = torch.tensor(s["nu_node"], device=device, dtype=dtype)[None]
            f = torch.tensor(s["node_forces"], device=device, dtype=dtype)[None]
            tgt = torch.tensor(s["uv_exact"], device=device, dtype=dtype)

            # ---- 1. the input channels, as the model receives them --------
            if i == 0:
                nz = (f.abs().sum(dim=2) > 0).float().mean().item()
                print("\n  the four input channels, exactly as fed "
                      "(no normalization anywhere in this path):")
                for name, ch in (("E    ", E[0]), ("nu   ", nu[0]),
                                 ("f_x  ", f[0, :, 0]), ("f_y  ", f[0, :, 1])):
                    print(f"    {name} rms {float(torch.sqrt(torch.mean(ch**2))):11.4e}"
                          f"   min {float(ch.min()):11.4e}"
                          f"   max {float(ch.max()):11.4e}")
                print(f"    nodes carrying any force: {nz * 100:.1f}% "
                      f"({int(nz * f.shape[1])} of {f.shape[1]})")
                print(f"    rms(f)/rms(E) = "
                      f"{float(torch.sqrt(torch.mean(f**2)) / torch.sqrt(torch.mean(E**2))):.3e}"
                      f"  -- how loud the loading is beside the stiffness")

            with torch.no_grad():
                Pi, U, W, uv, _ = total_potential_energy_Q4_hyperelastic(
                    xy, quad, ie, t0, th, model, E, nu, f,
                    use_soft_dirichlet=bool(args.use_soft_dirichlet),
                    R_out=args.R_out, mode=args.mode, dtype=dtype,
                    fun_dim=args.fun_dim, material=args.material)
            pred = uv[0]
            preds.append(pred)
            tgts.append(tgt)

            # ---- 2. what it predicts -------------------------------------
            num = float(torch.sum((pred - pred.mean()) * (tgt - tgt.mean())))
            den = float(torch.sqrt(torch.sum((pred - pred.mean()) ** 2)
                                   * torch.sum((tgt - tgt.mean()) ** 2)))
            corr = num / den if den > 1e-30 else float("nan")
            r_pred = float(torch.sqrt(torch.mean(pred ** 2)))
            r_tgt = float(torch.sqrt(torch.mean(tgt ** 2)))
            print(f"\n  sample {i}")
            print(f"    rms(pred) {r_pred:.4e}   rms(uv_exact) {r_tgt:.4e}"
                  f"   ratio {r_pred / max(r_tgt, 1e-30):8.4f}")
            print(f"    relative L2 vs uv_exact {rel(pred, tgt):.4f}"
                  f"   correlation {corr:+.4f}")
            print(f"    (a prediction of exactly zero scores 1.0000 and "
                  f"correlation nan)")

            # ---- 3. where it sits on Pi ----------------------------------
            print(f"    Pi(pred) {float(Pi[0]): .6e}   "
                  f"Pi(0) 0.000000e+00   "
                  f"U {float(U[0]):.4e}  W {float(W[0]):.4e}")

        # ---- 4. does the model use its input? ----------------------------
        if len(preds) >= 2:
            def variability(stack):
                S = torch.stack(stack)
                return float(torch.sqrt(torch.mean((S - S.mean(dim=0)) ** 2))
                             / torch.sqrt(torch.mean(S ** 2)).clamp_min(1e-30))
            collapse[N] = variability(preds)
            print(f"\n  across the {len(preds)} samples on this one mesh, "
                  f"sample-to-sample variability,")
            print(f"  measured as rms(deviation from the mean field) / "
                  f"rms(the fields):")
            print(f"    the model's predictions  {collapse[N]:.4f}")
            print(f"    the FEM targets          {variability(tgts):.4f}")
            print(f"  a model that reads its input should be near the second "
                  f"number, not near zero.")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if collapse and max(collapse.values()) < 0.05:
        print("The model gives essentially the SAME field whatever sample it is")
        print("shown. It has collapsed to a function of the coordinates alone")
        print("and is ignoring (E, nu, f). That is consistent with an error")
        print("flat in the mesh, flat in N and flat across materials, and it")
        print("points straight at the input channels printed above: read")
        print("rms(f)/rms(E) and the percentage of nodes carrying any force.")
    elif collapse:
        print("The model does respond to its input -- the predictions differ")
        print("across samples. So it is not ignoring the fields, and the")
        print("failure is in how far the optimizer got, not in what the model")
        print("can see. Compare Pi(pred) against Pi(uv_exact) from")
        print("test_b2_zeroshot_functional.py: that says how much of the")
        print("descent actually happened.")
    print("\nSend this block over. Nothing was written and nothing trained.")


if __name__ == "__main__":
    main()
