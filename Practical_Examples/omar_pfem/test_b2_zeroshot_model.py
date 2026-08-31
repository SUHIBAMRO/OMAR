"""What is the trained zero-shot model doing, and what is it fed?

Runs on either geometry, because the point of it is the COMPARISON: B2 sits
at a combined validation error of ~1.0 and B1, same trainer and same
protocol, reaches 0.066. Any account of B2's failure that would apply
equally to B1 is not an account of anything, so every number below is meant
to be read in pairs.

Where this stands. test_b2_zeroshot_functional.py established that Pi's
minimum is at the FEM solution -- s = 1.000 on six samples, |W|/U = 2.00 to
three decimals. The cache and the functional are fine. What is left is the
training path, and this looks at it with the checkpoint already on disk.
Nothing is trained and nothing is written.

The first version of this script had three faults and this one fixes them.
All three are worth naming, because each would have produced a confident
wrong reading, and the third was caught only because the B1 control arm was
added and the script then printed a sentence that contradicted itself.

  * It reported Pi(pred) with no Pi(uv_exact) for the SAME sample. The
    functional test's Pi values are on train_samples, this reads
    val_samples, and the two sample sets are different problems -- so
    "Pi(pred) = -2.7e-02" could not be compared with anything. Both are now
    computed here, per sample, and the fraction of the available descent is
    what gets printed.

  * It printed the input-channel scales at N=21 and N=33 side by side as
    though the difference between them were a mesh effect. It is not
    separable that way: the cache seeds each resolution differently
    (seed_base = 10_000 * N), so those are different draws as well as
    different meshes. This builds ONE fixed seed at BOTH resolutions and
    compares that, which is the controlled version, and prints the
    mesh-independent load total beside the per-node scale.

  * It applied a B2 check to B1. The quantity that must be mesh-invariant is
    the one the geometry's own work term uses, and the two are different:
    B1's W is sum(f*uv)/len(top_edges) over the RAW pointwise traction, while
    B2's is sum(f*uv) over the ASSEMBLED force. Printing sum(f) for both and
    asserting in the text that they "must" agree produced, on B1, the line
    "the TOTAL load agrees to 56.427% -- it must". Each geometry's own
    invariant is printed now. (B1's is 4.6556 against 4.5516, 2.3% apart.)
    The same confusion sat behind the "that is impossible" flag on
    Pi(pred) < Pi(uv_exact): where the trainer's Pi and the solver's Pi are
    not the same functional, as on B1, uv_exact does not minimise the
    trainer's Pi and a gap of order a per cent is just that quadrature
    difference. On B2 they ARE the same functional, which is why its W/U at
    uv_exact is 2.000 on every sample.

What it measures, per geometry:

  1. THE INPUT CHANNELS as the model receives them -- fun_material is
     (E, nu, f_x, f_y) fed RAW, with no normalization anywhere in this path.
     Plus the controlled mesh comparison described above.
  2. WHAT IT PREDICTS -- amplitude ratio, correlation, relative L2.
  3. HOW FAR DOWN Pi IT GOT -- Pi(pred) against Pi(uv_exact) and Pi(0) = 0,
     same sample, as a percentage of the available descent.
  4. WHETHER RESCALING WOULD HELP -- W/U at the prediction, and a scan of
     Pi(s * pred). W/U = 2 means the prediction is already stationary under
     rescaling, so a stalled descent along a ray is NOT the explanation.
  5. WHETHER IT USES ITS INPUT -- sample-to-sample variability of the
     prediction against that of the target, and of the strain energy U,
     which is the physical version of the same question.

Usage:
  python -m omar_pfem.test_b2_zeroshot_model --geometry B2 \
      --cache <samples_cache.pt> --checkpoint <model_best.pt>
"""
import argparse
import os

import numpy as np
import torch

from omar_pfem.model_dict import get_model
from omar_pfem.resolution_invariance_zeroshot import (
    build_sample_b1, build_sample_b2, loss_and_pred, mesh_tensors_of)


class Fixed(torch.nn.Module):
    """Stands in for the network and returns a prescribed raw field."""

    def __init__(self, raw):
        super().__init__()
        self.register_buffer("raw", raw)

    def forward(self, xy_domain, fun_material):
        return self.raw.to(xy_domain.dtype)


def build_model(args, device):
    return get_model(args).Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim, out_dim=2,
        slice_num=args.slice_num, ref=args.ref,
        unified_pos=args.unified_pos).to(device)


def rel(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2))
                 / torch.sqrt(torch.mean(b ** 2)).clamp_min(1e-30))


def rms(a):
    return float(torch.sqrt(torch.mean(a ** 2)))


def variability(stack):
    S = torch.stack(stack)
    return float(torch.sqrt(torch.mean((S - S.mean(dim=0)) ** 2))
                 / torch.sqrt(torch.mean(S ** 2)).clamp_min(1e-30))


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
    p.add_argument("--n_samples", type=int, default=4)
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
    print(f"geometry   : {args.geometry} x {args.material}")
    print(f"cache      : {args.cache}")
    print(f"checkpoint : {args.checkpoint}")
    print("resolutions: " + ", ".join(f"{k} ({len(v)} val)"
                                      for k, v in buckets.items()))

    def energy(sample, mesh, raw_or_model):
        E = torch.tensor(sample["E_node"], device=device, dtype=dtype)[None]
        nu = torch.tensor(sample["nu_node"], device=device, dtype=dtype)[None]
        f = torch.tensor(sample["node_forces"], device=device, dtype=dtype)[None]
        with torch.no_grad():
            Pi, U, W, uv, _ = loss_and_pred(
                args.geometry, mesh, raw_or_model, E, nu, f, args, dtype)
        return float(Pi[0]), float(U[0]), float(W[0]), uv[0]

    # ---- 1b. the controlled mesh comparison, one seed on both meshes -----
    print("\n" + "=" * 74)
    print("THE LOAD CHANNEL ACROSS THE TWO TRAINING MESHES, ONE FIXED SEED")
    print("=" * 74)
    print("The cache seeds each resolution differently, so its own samples")
    print("cannot answer this. One seed is rebuilt on both meshes instead.")
    build_fn = build_sample_b1 if args.geometry == "B1" else build_sample_b2
    ctrl = {}
    for N in buckets:
        s, _ = build_fn(N, seed=777_000, material=args.material,
                        solve_fem=False)
        fc = s["node_forces"]
        ctrl[N] = dict(
            rms_f=float(np.sqrt(np.mean(fc ** 2))),
            rms_E=float(np.sqrt(np.mean(s["E_node"] ** 2))),
            nz=float((np.abs(fc).sum(axis=1) > 0).mean()),
            total=float(np.linalg.norm(fc.sum(axis=0))),
            effective=float(np.linalg.norm(fc.sum(axis=0)))
            / max(len(s.get("top_edges", s.get("inner_edges", []))), 1),
            n_nodes=fc.shape[0])
    print(f"\n{'N':>6}{'nodes':>8}{'rms(f)':>12}{'rms(f)/rms(E)':>16}"
          f"{'nodes with f':>14}{'|sum f| (total load)':>22}")
    for N, c in ctrl.items():
        print(f"{N:>6}{c['n_nodes']:>8}{c['rms_f']:>12.4e}"
              f"{c['rms_f'] / c['rms_E']:>16.3e}"
              f"{c['nz'] * 100:>13.1f}%{c['total']:>22.6f}")
    Ns = sorted(ctrl)
    if len(Ns) >= 2:
        a, b = ctrl[Ns[0]], ctrl[Ns[-1]]
        # The quantity that must be mesh-invariant is the one the geometry's
        # OWN work term uses, and the two geometries do not use the same one.
        # An earlier version of this script printed sum(f) for both and
        # asserted in its own text that they "must" agree -- which read as
        # "the TOTAL load agrees to 56.427% -- it must" on B1, a sentence that
        # contradicts itself. B1's W divides by the top-edge count and its
        # node_forces are the raw pointwise traction, so sum(f)/n_edges is its
        # invariant; B2's W does not divide and its node_forces are assembled,
        # so sum(f) is.
        if args.geometry == "B1":
            key, label = "effective", "sum(f) / (number of loaded edges)"
            note = ("B1's W is sum(f*uv)/len(top_edges) over the RAW pointwise "
                    "traction, so this is the quantity its energy actually "
                    "sees.")
        else:
            key, label = "total", "|sum f|, the total applied load"
            note = ("B2's W is sum(f*uv) over the ASSEMBLED force, so the bare "
                    "total is the quantity its energy sees. This is the check "
                    "the load repair installed.")
        gap = abs(b[key] / a[key] - 1) * 100
        print(f"\n  {note}")
        print(f"  {label}: {a[key]:.6f} at N={Ns[0]}, {b[key]:.6f} at "
              f"N={Ns[-1]} -- {gap:.3f}% apart"
              + ("  OK" if gap < 5 else "  <-- NOT mesh-invariant, look at it"))
        print(f"  the PER-NODE scale the network is fed changes by "
              f"{a['rms_f'] / b['rms_f']:.2f}x between the two meshes: the same")
        print(f"  physical loading, two different numbers in the input "
              f"channel, and nothing")
        print(f"  normalises it unless --normalize_inputs was on.")

    # ---- the per-sample work -------------------------------------------
    collapse, energy_spread = {}, {}
    descent, roughness, demand_stats = [], [], []
    for N, samples in buckets.items():
        print("\n" + "=" * 74)
        print(f"resolution {N}")
        print("=" * 74)
        mesh = mesh_tensors_of(args.geometry, samples[0], device, dtype)
        n_nodes = samples[0]["xy"].shape[0]

        # the Dirichlet mask, discovered rather than reimplemented: feeding a
        # raw field of ones returns the mask itself, whatever the geometry
        ones = torch.ones(1, n_nodes, 2, device=device, dtype=dtype)
        _, _, _, mask = energy(samples[0], mesh, Fixed(ones))

        preds, tgts, Us = [], [], []
        for i in range(min(args.n_samples, len(samples))):
            s = samples[i]
            tgt = torch.tensor(s["uv_exact"], device=device, dtype=dtype)

            if i == 0:
                E = torch.tensor(s["E_node"], dtype=dtype)
                f = torch.tensor(s["node_forces"], dtype=dtype)
                nu = torch.tensor(s["nu_node"], dtype=dtype)
                print("\n  the four channels of this sample, exactly as fed "
                      "(nothing normalises them):")
                for nm, ch in (("E   ", E), ("nu  ", nu),
                               ("f_x ", f[:, 0]), ("f_y ", f[:, 1])):
                    print(f"    {nm} rms {rms(ch):11.4e}   min "
                          f"{float(ch.min()):11.4e}   max {float(ch.max()):11.4e}")
                print(f"    rms(f)/rms(E) = {rms(f) / rms(E):.3e}")

            pi_p, U_p, W_p, pred = energy(s, mesh, model)
            # Pi at the truth, SAME sample. The stand-in must reproduce it or
            # nothing below means anything.
            raw = torch.where(mask.abs() > 1e-12, tgt / mask.clamp_min(1e-12),
                              torch.zeros_like(tgt))[None]
            pi_t, U_t, W_t, back = energy(s, mesh, Fixed(raw))
            err = float(torch.max(torch.abs(back - tgt)))
            assert err < 1e-4 * max(rms(tgt), 1e-30), (
                f"the stand-in does not reproduce uv_exact (max abs diff "
                f"{err:.3e}); every Pi below is meaningless")

            frac = pi_p / pi_t if pi_t < 0 else float("nan")
            descent.append(frac)
            num = float(torch.sum((pred - pred.mean()) * (tgt - tgt.mean())))
            den = float(torch.sqrt(torch.sum((pred - pred.mean()) ** 2)
                                   * torch.sum((tgt - tgt.mean()) ** 2)))
            print(f"\n  sample {i}")
            print(f"    rms(pred) {rms(pred):.4e}   rms(uv_exact) "
                  f"{rms(tgt):.4e}   ratio {rms(pred) / max(rms(tgt), 1e-30):7.4f}")
            print(f"    relative L2 {rel(pred, tgt):.4f}   correlation "
                  f"{(num / den if den > 1e-30 else float('nan')):+.4f}"
                  f"   (predicting zero scores 1.0000)")
            print(f"    Pi(pred)     {pi_p: .6e}   U {U_p:.4e}  W {W_p:.4e}"
                  f"   W/U {W_p / max(U_p, 1e-30):.2f}")
            print(f"    Pi(uv_exact) {pi_t: .6e}   U {U_t:.4e}  W {W_t:.4e}"
                  f"   W/U {W_t / max(U_t, 1e-30):.2f}")
            # strain energy per unit amplitude.  (see below for the mask) A prediction as smooth as the
            # truth has U scaling with amplitude squared, so this ratio is 1.
            # Above 1 means the field carries more strain than its size
            # warrants -- it is rough, not merely small.
            amp = rms(pred) / max(rms(tgt), 1e-30)
            rough = (U_p / max(U_t, 1e-30)) / max(amp * amp, 1e-30)
            # THE RAW FIELD THE MASK DEMANDS. The network's output is
            # mask * raw, so to produce uv_exact it must emit uv_exact/mask.
            # Near an edge where the mask vanishes that quotient blows up, and
            # B2 has TWO ramps vanishing on DIFFERENT edges against B1's one.
            # The mask can represent uv_exact -- the assertion above proves it
            # -- so this is not about representability; it is about how large
            # and how uneven a field the network has to emit to get there.
            live = mask.abs() > 1e-9
            demand = torch.where(live, tgt / mask.clamp_min(1e-30),
                                 torch.zeros_like(tgt))
            dyn = (float(demand[live].abs().max())
                   / max(rms(demand[live]), 1e-30))
            raw_over_out = rms(demand[live]) / max(rms(tgt), 1e-30)
            demand_stats.append((dyn, raw_over_out))
            print(f"    Pi(0) = 0, so the model captured "
                  f"{frac * 100:5.1f}% of the available descent")
            print(f"    the raw field the mask demands for uv_exact: rms "
                  f"{rms(demand[live]):.4e} against the output's "
                  f"{rms(tgt):.4e} ({raw_over_out:5.2f}x),")
            print(f"      peak/rms {dyn:7.2f}  -- how uneven a field the "
                  f"network must emit to hit the target")
            print(f"    roughness (U ratio)/(amplitude ratio)^2 = {rough:5.2f}"
                  f"   -- 1.0 is as smooth as the truth")
            roughness.append(rough)
            if pi_p < pi_t - 1e-9 * abs(pi_t):
                over = (pi_t - pi_p) / abs(pi_t) * 100
                print(f"    note: Pi(pred) is {over:.2f}% BELOW Pi(uv_exact). "
                      f"That is not a contradiction where the")
                print("       trainer's Pi and the FEM solver's Pi are not the "
                       "same functional. On B1 they")
                print("       are not: the trainer's W is a trapezoid sum over "
                       "the raw pointwise traction,")
                print("       divided by the edge count, while the solver "
                       "integrates the traction exactly, so")
                print("       uv_exact minimises the solver's Pi and not quite "
                       "this one. A gap of order a")
                print("       per cent is that quadrature difference. On B2 "
                       "the two ARE the same functional")
                print("       (assembled force, no division), which is why its "
                       "W/U at uv_exact is 2.000.")

            # would rescaling the prediction help?
            scan = []
            for sc in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0):
                rp = torch.where(mask.abs() > 1e-12,
                                 pred * sc / mask.clamp_min(1e-12),
                                 torch.zeros_like(pred))[None]
                scan.append((sc, energy(s, mesh, Fixed(rp))[0]))
            best = min(scan, key=lambda t: t[1])
            print(f"    rescaling the prediction: Pi is lowest at s = "
                  f"{best[0]} ({best[1]:.4e}); "
                  + ("amplitude is not the problem"
                     if abs(best[0] - 1.0) < 0.3 else
                     "the amplitude alone is worth "
                     f"{(best[1] / pi_p - 1) * 100:.0f}% more descent"))

            preds.append(pred)
            tgts.append(tgt)
            Us.append(torch.tensor(U_p))

        if len(preds) >= 2:
            collapse[N] = variability(preds)
            energy_spread[N] = (max(float(u) for u in Us)
                                / max(min(float(u) for u in Us), 1e-30))
            print(f"\n  across the {len(preds)} samples on this one mesh:")
            print(f"    variability of the predictions  {collapse[N]:.4f}")
            print(f"    variability of the targets      {variability(tgts):.4f}")
            print(f"    spread of U(pred), max/min      {energy_spread[N]:.2f}x")
            print(f"  a model that reads its input tracks the second number.")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    d = [x for x in descent if x == x]
    if d:
        print(f"descent captured: {min(d) * 100:.0f}% to {max(d) * 100:.0f}% "
              f"of Pi(uv_exact), mean {sum(d) / len(d) * 100:.0f}%")
    if roughness:
        print(f"roughness: {min(roughness):.2f}x to {max(roughness):.2f}x, "
              f"mean {sum(roughness) / len(roughness):.2f}x")
    if demand_stats:
        dyns = [d for d, _ in demand_stats]
        amps = [a for _, a in demand_stats]
        print(f"the raw field the mask demands: {min(amps):.2f}x to "
              f"{max(amps):.2f}x the output in rms, peak/rms {min(dyns):.1f} "
              f"to {max(dyns):.1f}")
        print("  Read this one across the two geometries. If B2's numbers are")
        print("  far larger than B1's, the Dirichlet ramp is asking the")
        print("  network for a much harder field and is a real obstacle. If")
        print("  they are comparable, the ramp is exonerated and the last")
        print("  candidate is the parametric family.")
    if collapse:
        print(f"prediction variability: "
              + ", ".join(f"N={N} {v:.3f}" for N, v in collapse.items()))
    print("\nRead it against the other geometry. B1 reaches 0.066 on the same")
    print("trainer, so whatever separates the two runs has to show up as a")
    print("difference between these two printouts -- and anything that looks")
    print("the same in both is not the cause.")


if __name__ == "__main__":
    main()
