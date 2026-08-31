"""Is the FEM solution in the B2 zero-shot cache a minimum of the Pi the
trainer minimizes?

Why this and not another training run. Three B2 zero-shot cases were
retrained with --loss_force_norm on, and all three landed at a combined
validation error of about 1.0 and stayed there. A batch-size arm at
matched optimizer steps moved it from 0.9888 to 0.9444, which is noise on
curves that swing between 0.94 and 1.45.

A relative error of exactly 1.0 is not a random bad number. The metric is
0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v)); substitute uv_pred = 0 and it is
identically 1. So the network is very likely predicting approximately
nothing -- and if it is, it may be doing so CORRECTLY, because a network
that minimizes Pi will go wherever Pi's minimum is. The question is then
not about the optimizer at all. It is whether Pi's minimum, built from
this cache's node_forces, is anywhere near this cache's uv_exact.

That is what this checks, and it needs no training:

  1. Pi(uv_exact) against Pi(0). If Pi(0) is lower, the trainer is being
     asked to find zero and is finding it.
  2. A scale scan: Pi(s * uv_exact) for s from 0 to about 1.5. The
     minimum should sit at s = 1. Where it actually sits says how badly
     the work term is mis-scaled -- s near 0 means W is far too weak, and
     the ratio is a direct measure of the shortfall.
  3. The decomposition U and W at uv_exact, so the imbalance is visible
     as two numbers rather than inferred.

The same test on the MMS operator (test_mms_operator.py) is what caught a
mis-scaled W there before any training was wasted on it. This is the B2
version.

Usage:
  python -m omar_pfem.test_b2_zeroshot_functional --cache <samples_cache.pt>
"""
import argparse
import os

import numpy as np
import torch

from omar_pfem.train_B2 import total_potential_energy_Q4_hyperelastic


class ConstantField(torch.nn.Module):
    """Stands in for the trained network and returns a prescribed field.

    total_potential_energy_Q4_hyperelastic multiplies the model's output by
    the soft-Dirichlet ramp (x/R_out, y/R_out), so to make the masked
    result equal a target field this must return target/ramp. The ramp
    vanishes on the symmetry edges, where the target is zero too, so the
    quotient is taken with a floor and the numerator there is zero anyway.
    """

    def __init__(self, target, xy, R_out, use_soft_dirichlet):
        super().__init__()
        if use_soft_dirichlet:
            fu = (xy[:, 0] / R_out).clamp(0.0, 1.0)
            fv = (xy[:, 1] / R_out).clamp(0.0, 1.0)
        else:
            fu = torch.ones(xy.shape[0], device=xy.device, dtype=xy.dtype)
            fv = torch.ones_like(fu)
        floor = 1e-12
        self.register_buffer("raw", torch.stack([
            target[:, :, 0] / fu.clamp_min(floor)[None, :],
            target[:, :, 1] / fv.clamp_min(floor)[None, :]], dim=2))

    def forward(self, xy_domain, fun_material):
        return self.raw.to(xy_domain.dtype)


def pi_at(scale, sample, xy, quad, inner_edges, t0, thalf, args, dtype, device):
    tgt = torch.tensor(sample["uv_exact"], device=device, dtype=dtype)[None] * scale
    model = ConstantField(tgt, xy, args.R_out, bool(args.use_soft_dirichlet)).to(device)
    E = torch.tensor(sample["E_node"], device=device, dtype=dtype)[None]
    nu = torch.tensor(sample["nu_node"], device=device, dtype=dtype)[None]
    f = torch.tensor(sample["node_forces"], device=device, dtype=dtype)[None]
    with torch.no_grad():
        Pi, U, W, uv, _ = total_potential_energy_Q4_hyperelastic(
            xy, quad, inner_edges, t0, thalf, model, E, nu, f,
            use_soft_dirichlet=bool(args.use_soft_dirichlet), R_out=args.R_out,
            mode=args.mode, dtype=dtype, fun_dim=args.fun_dim,
            material=args.material)
    return float(Pi[0]), float(U[0]), float(W[0]), uv[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--material", default="neo_hookean")
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--mode", default="plane_strain")
    p.add_argument("--fun_dim", type=int, default=4)
    p.add_argument("--use_soft_dirichlet", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    dtype = torch.float32
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    # the cache is {resolution: {"train": [...], "val": [...]}} or a flat
    # dict of lists; accept either rather than guessing
    if "train" in cache:
        buckets = {"(flat)": cache["train"]}
    else:
        buckets = {str(k): v["train"] for k, v in sorted(cache.items())
                   if isinstance(v, dict) and "train" in v}
    assert buckets, f"could not find training samples in {args.cache}"
    print(f"cache: {args.cache}")
    print("resolutions:", ", ".join(f"{k} ({len(v)} train)"
                                    for k, v in buckets.items()))

    verdicts = []
    for key, samples in buckets.items():
        print("\n" + "=" * 74)
        print(f"resolution {key}")
        print("=" * 74)
        for i in range(min(args.n_samples, len(samples))):
            s = samples[i]
            xy = torch.tensor(s["xy"], device=device, dtype=dtype)
            quad = torch.tensor(s["quad"], device=device, dtype=torch.long)
            ie = torch.tensor(s["inner_edges"], device=device, dtype=torch.long)
            t0 = torch.tensor(s["theta0_nodes"], device=device, dtype=torch.long)
            th = torch.tensor(s["thetahalfpi_nodes"], device=device, dtype=torch.long)

            pi1, U1, W1, uv1 = pi_at(1.0, s, xy, quad, ie, t0, th, args, dtype, device)
            pi0, U0, W0, _ = pi_at(0.0, s, xy, quad, ie, t0, th, args, dtype, device)

            # does the constant-field stand-in actually reproduce uv_exact?
            # if not, nothing below means anything.
            tgt = torch.tensor(s["uv_exact"], device=device, dtype=dtype)
            rep = float(torch.max(torch.abs(uv1 - tgt)))
            scale_of = float(torch.max(torch.abs(tgt)))
            assert rep < 1e-4 * max(scale_of, 1e-12), (
                f"the stand-in does not reproduce uv_exact (max abs diff "
                f"{rep:.3e} against a field of size {scale_of:.3e}); the "
                f"Dirichlet ramp assumption in ConstantField is wrong and "
                f"every number below is meaningless")

            scan = [(sc, pi_at(sc, s, xy, quad, ie, t0, th, args, dtype, device)[0])
                    for sc in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0,
                               1.15, 1.3, 1.5)]
            best_s = min(scan, key=lambda t: t[1])[0]
            verdicts.append((key, i, best_s))

            print(f"\nsample {i}")
            print(f"  Pi(uv_exact) = {pi1:.6e}   (U {U1:.6e}, W {W1:.6e})")
            print(f"  Pi(0)        = {pi0:.6e}   (U {U0:.6e}, W {W0:.6e})")
            print(f"  |W|/U at uv_exact = {abs(W1) / max(U1, 1e-30):.4f}"
                  f"   -- for a solution of Pi=U-W this should be about 2")
            print("  scale scan, Pi(s * uv_exact):")
            for sc, v in scan:
                mark = "  <-- min" if sc == best_s else ""
                bar = "#" * int(60 * (v - min(x[1] for x in scan)) /
                                max(1e-30, max(x[1] for x in scan)
                                    - min(x[1] for x in scan)))
                print(f"    s={sc:<5.2f} Pi={v: .6e} {bar}{mark}")
            print(f"  minimum at s = {best_s}")
            if best_s < 0.5:
                print("  -> Pi is minimized far BELOW the true solution. A")
                print("     network minimizing this Pi is supposed to shrink")
                print("     toward zero, and the ~1.0 relative error is it")
                print("     doing that correctly. The fault is in the data or")
                print("     the work term, not the optimizer.")
            elif abs(best_s - 1.0) < 0.2:
                print("  -> Pi is minimized at the true solution. The")
                print("     functional and this cache agree, so the training")
                print("     failure is NOT a mis-scaled W and lies elsewhere.")
            else:
                print("  -> Pi is minimized away from 1 but not at 0.")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    ss = [v for _, _, v in verdicts]
    print(f"minimum of Pi along the solution ray, over {len(ss)} samples: "
          f"{min(ss)} to {max(ss)}")
    if max(ss) < 0.5:
        print("\nThe work term is too weak in EVERY sample checked. Pi's")
        print("minimizer is near zero, so no optimizer setting can recover the")
        print("solution from this data. Compare this cache's node_forces")
        print("against the force the ground-truth solve used -- the repair's")
        print("mesh-independence check verified the load's TOTAL is the same")
        print("on both meshes, which does not verify it matches the solve.")
    elif min(ss) > 0.8:
        print("\nThe functional agrees with the data. Look elsewhere: the")
        print("model's capacity to represent the field through the")
        print("x/R_out ramp, or the optimizer settings.")
    else:
        print("\nMixed across samples -- report the spread rather than a")
        print("single cause.")


if __name__ == "__main__":
    main()
