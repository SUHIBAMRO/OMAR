"""Speed-up and break-even against GPU FEM, at MATCHED batch sizes.

The advisor's round-5 point 3 asked for the break-even to be recomputed
against a GPU FEM baseline rather than a CPU one; point 4 asked for the two
to be benchmarked under identical batch sizes. Those are the same
calculation, and doing point 3 without point 4 gets it wrong: the earlier
figure compared GPU FEM at batch size 128 -- where a GPU solver amortises
its kernel launches -- against the Transolver at batch size 1. That is not
a like-for-like comparison, and it understates the operator by roughly the
factor batching buys it (measured: 4.80 ms/sample at bs=1 versus 0.29 at
bs=128, about 16x).

This script pairs the two benchmarks by batch size and reports both
figures, so the effect of the fair comparison is visible rather than
asserted:

  speed-up(B)   = fem_ms(B) / nn_ms(B)
  break-even(B) = training_seconds / ((fem_ms(B) - nn_ms(B)) / 1000)

Break-even is the number of new problems that must be solved before the
one-off training cost is repaid. Note the two quantities respond very
differently to fair batching. Speed-up changes by the full batching factor,
because it is a ratio of the two per-sample times. Break-even barely moves,
because FEM dominates the denominator either way -- shrinking a term that
was already about 1% of it changes little. Both are reported so that
neither is mistaken for the other.

The previously published figures (Transolver at bs=1 against FEM at bs=128)
are printed alongside, labelled as the unmatched comparison, so the report
can be corrected against a like-for-like pair rather than silently.

Usage:
  python -m omar_pfem.break_even_analysis \
      --fem_json  .../gpu_fem_solver/B1_neo_hookean_timing.json \
      --nn_json   .../inference_latency_by_batch_B1_neo_hookean.json \
      --training_seconds 2784 \
      --out_json  .../break_even_B1_neo_hookean.json
"""
import os
import json
import time
import argparse


def load_rows(path, label):
    with open(path) as f:
        data = json.load(f)
    rows = {int(r["batch_size"]): float(r["per_sample_ms"]) for r in data["rows"]}
    if not rows:
        raise SystemExit(f"{label}: {path} has no rows")
    return data, rows


def main():
    p = argparse.ArgumentParser("Speed-up and break-even at matched batch sizes")
    p.add_argument("--fem_json", required=True,
                   help="gpu_fem_benchmark.py output for this case")
    p.add_argument("--nn_json", required=True,
                   help="inference_latency_by_batch.py output for this case")
    p.add_argument("--training_seconds", type=float, default=None,
                   help="total one-off training wall-clock for this case. Omit to "
                        "report speed-ups only; break-even needs it.")
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()
    started = time.time()

    fem_data, fem = load_rows(args.fem_json, "GPU FEM")
    nn_data, nn = load_rows(args.nn_json, "Transolver")

    case = f"{nn_data.get('geometry')}_{nn_data.get('material')}"
    if (fem_data.get("geometry"), fem_data.get("material")) != \
       (nn_data.get("geometry"), nn_data.get("material")):
        raise SystemExit(
            f"the two files describe different cases -- FEM is "
            f"{fem_data.get('geometry')}_{fem_data.get('material')} but the network is "
            f"{case}. Comparing them would be meaningless; refusing to continue.")

    shared = sorted(set(fem) & set(nn))
    if not shared:
        raise SystemExit(
            f"no batch size appears in both benchmarks (FEM has {sorted(fem)}, "
            f"the network has {sorted(nn)}) -- there is nothing to match.")

    if args.out_json is None:
        args.out_json = f"break_even_{case}.json"
        print(f"[auto-save] --out_json not given; writing to {args.out_json}")

    rows = []
    for B in shared:
        saving_ms = fem[B] - nn[B]
        row = {"batch_size": B,
               "fem_ms_per_sample": fem[B],
               "nn_ms_per_sample": nn[B],
               "speedup": fem[B] / nn[B],
               "saving_ms_per_sample": saving_ms}
        if args.training_seconds is not None:
            row["break_even_samples"] = (args.training_seconds / (saving_ms / 1000.0)
                                         if saving_ms > 0 else None)
        rows.append(row)

    # The comparison the report currently makes: FEM batched as hard as it
    # goes, the network not batched at all.
    unmatched = None
    B_fem, B_nn = max(fem), min(nn)
    if B_fem != B_nn:
        saving = fem[B_fem] - nn[B_nn]
        unmatched = {"fem_batch_size": B_fem, "nn_batch_size": B_nn,
                     "fem_ms_per_sample": fem[B_fem], "nn_ms_per_sample": nn[B_nn],
                     "speedup": fem[B_fem] / nn[B_nn],
                     "break_even_samples": (args.training_seconds / (saving / 1000.0)
                                            if args.training_seconds and saving > 0
                                            else None)}

    print("\n" + "=" * 74)
    print(f"SPEED-UP AND BREAK-EVEN vs GPU FEM  ({case})")
    print("=" * 74)
    print(f"{'batch':>6}{'FEM ms/sample':>16}{'NN ms/sample':>15}"
          f"{'speed-up':>12}{'break-even':>14}")
    for r in rows:
        be = r.get("break_even_samples")
        print(f"{r['batch_size']:>6}{r['fem_ms_per_sample']:>16.3f}"
              f"{r['nn_ms_per_sample']:>15.4f}{r['speedup']:>11.1f}x"
              f"{('%14.0f' % be) if be else '             -'}")
    if unmatched:
        print("-" * 74)
        print(f"unmatched (what the report currently quotes): FEM at bs="
              f"{unmatched['fem_batch_size']} vs network at bs={unmatched['nn_batch_size']}")
        be = unmatched["break_even_samples"]
        print(f"{'':>6}{unmatched['fem_ms_per_sample']:>16.3f}"
              f"{unmatched['nn_ms_per_sample']:>15.4f}{unmatched['speedup']:>11.1f}x"
              f"{('%14.0f' % be) if be else '             -'}")
    if args.training_seconds is None:
        print("\n(no --training_seconds given, so break-even is not computed)")
    print("=" * 74)

    report = {"case": case, "geometry": nn_data.get("geometry"),
              "material": nn_data.get("material"),
              "training_seconds": args.training_seconds,
              "fem_json": args.fem_json, "nn_json": args.nn_json,
              "matched": rows, "unmatched": unmatched}
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written to {args.out_json}")

    try:
        from omar_pfem.run_manifest import write_manifest
        write_manifest(
            os.path.dirname(os.path.abspath(args.out_json)) or ".",
            kind="break_even_analysis", args=args, started_at=started,
            results=report, outputs=[args.out_json],
            notes=("Advisor points 3 and 4 together: the break-even against a GPU FEM "
                   "baseline, computed at batch sizes matched between the two. The "
                   "'unmatched' entry reproduces the comparison the report currently "
                   "quotes -- FEM batched at its best, the network at batch size 1 -- so "
                   "the two can be seen side by side. Speed-up moves by the full batching "
                   "factor; break-even barely moves, since FEM dominates the per-sample "
                   "saving either way."))
    except Exception as e:
        print(f"[manifest] not recorded: {e}")


if __name__ == "__main__":
    main()
