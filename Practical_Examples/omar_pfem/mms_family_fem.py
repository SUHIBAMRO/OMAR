"""Score Q4 and Q9 over the WHOLE manufactured family, not one member.

Timon's point 9, second email: "Ideally, we do it for a parametrised family
of solutions which will be a bit more time consuming." Half of that is done
and half is not:

  * the OPERATOR is trained on a 64-member family and its mean over the 16
    held-out test members is already recorded in each
    mms_operator_B1_neo_hookean*.json as `operator_mean_over_test_family`;

  * Q4 and Q9 have never been scored on the family at all. Tables 22, 23 and
    24 are every one of them the single member alpha = 0.05, beta = 0.7.

So there is no three-way family comparison, only a three-way single-member
one. This closes that: it runs mms_study.py once per test member and
aggregates, which needs no new code beyond the loop -- mms_study already
takes --alpha and --beta.

THE MEMBERS ARE THE OPERATOR'S OWN. They are drawn with
mms_operator.sample_family(ntest, seed + 1), the identical call the operator
run makes, so the FEM mean and the operator mean are over the SAME 16
problems and can be put in one table. Drawing a fresh set would give a
mean over a different family and the comparison would be silently wrong.

Cost: a Q4 and a Q9 solve per member per mesh. At N=17 that measured 14 s
and 73 s, so 16 members is about 23 minutes for that mesh; N=9 is far
cheaper and N=33 about 4x more. No training.

Usage:
  python -m omar_pfem.mms_family_fem --Ns 9,17,33 --ntest 16
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

from omar_pfem.mms_operator import sample_family

KEYS = ("L2_rel", "H1_semi_rel", "stress_rel_L2", "energy_rel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", default="neo_hookean")
    p.add_argument("--Ns", default="9,17,33")
    p.add_argument("--orders", default="Q4,Q9")
    p.add_argument("--ntest", type=int, default=16,
                   help="must match the operator run's --ntest, or the two "
                        "means are over different families")
    p.add_argument("--seed", type=int, default=31_000_000,
                   help="must match the operator run's --seed; the members "
                        "are drawn with sample_family(ntest, seed + 1), "
                        "exactly as mms_operator.py draws them")
    p.add_argument("--out_json", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    members = sample_family(args.ntest, args.seed + 1)
    print(f"{len(members)} test members, drawn exactly as the operator run "
          f"draws them (seed {args.seed} + 1):")
    for i, (a, b) in enumerate(members):
        print(f"  {i:2d}  alpha={a:.5f}  beta={b:.5f}")
    ref = (0.05, 0.7)
    print(f"\nthe single member Tables 22-24 use is alpha={ref[0]}, "
          f"beta={ref[1]}; it is NOT one of these unless it happens to be "
          f"drawn, and the point of this run is that the family mean is a "
          f"different number from it.")

    out = {"study": "MMS, Q4 and Q9 over the parametrised family",
           "material": args.material,
           "why": ("Timon's point 9, second email: 'Ideally, we do it for a "
                   "parametrised family of solutions'. The operator's family "
                   "mean was already recorded; Q4 and Q9 had only ever been "
                   "scored on the single member alpha=0.05, beta=0.7."),
           "members_drawn_as": (f"mms_operator.sample_family({args.ntest}, "
                                f"{args.seed} + 1) -- the operator's own call"),
           "members": [{"alpha": a, "beta": b} for a, b in members],
           "rows": []}

    for N in Ns:
        per = {o: {k: [] for k in KEYS} for o in args.orders.split(",")}
        for i, (a, b) in enumerate(members):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
                tmp = t.name
            cmd = [sys.executable, "-u", "-m", "omar_pfem.mms_study",
                   "--material", args.material, "--orders", args.orders,
                   "--Ns", str(N), "--alpha", f"{a:.10f}",
                   "--beta", f"{b:.10f}", "--out_json", tmp]
            if args.cpu:
                cmd.append("--cpu")
            print(f"\n[N={N}] member {i + 1}/{len(members)}  "
                  f"alpha={a:.5f} beta={b:.5f}", flush=True)
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            with open(tmp) as f:
                d = json.load(f)
            os.unlink(tmp)
            for r in d["rows"]:
                if r["N"] != N:
                    continue
                for k in KEYS:
                    per[r["order"]][k].append(r[k])
            got = {o: len(per[o]["L2_rel"]) for o in per}
            print("   " + "  ".join(f"{o} {v}/{i + 1}" for o, v in got.items()))

        row = {"N": N}
        for o, cols in per.items():
            assert len(cols["L2_rel"]) == len(members), (
                f"N={N} {o}: {len(cols['L2_rel'])} of {len(members)} members "
                f"produced a row; the aggregate would be over a subset")
            row[o] = {}
            for k in KEYS:
                v = cols[k]
                row[o][k] = {"mean": statistics.fmean(v),
                             "median": statistics.median(v),
                             "min": min(v), "max": max(v),
                             "stdev": statistics.stdev(v) if len(v) > 1 else 0.0}
        out["rows"].append(row)

        print(f"\n{'=' * 70}\n[N={N}] family mean over {len(members)} members"
              f"\n{'=' * 70}")
        print(f"{'order':<8}" + "".join(f"{k:>16}" for k in KEYS))
        for o in per:
            print(f"{o:<8}" + "".join(f"{row[o][k]['mean']:>16.4e}"
                                      for k in KEYS))
        print(f"{'':<8}" + "".join(f"{'+/- %.1e' % row[o][k]['stdev']:>16}"
                                   for k in KEYS)
              + f"   (stdev, {o})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out_json}")

    print("\n" + "=" * 70)
    print("HOW TO READ IT")
    print("=" * 70)
    print("Put each order's family MEAN beside the operator's")
    print("`operator_mean_over_test_family` from the matching")
    print("mms_operator_B1_neo_hookean*.json. Those are now means over the")
    print("SAME 16 problems, so the three-way comparison Timon asked for is")
    print("a family comparison and not a single-member one.")
    print("\nThe spread matters as much as the mean: if the FEM stdev is small")
    print("and the operator's is large, the operator is inconsistent across")
    print("the family even where its mean looks acceptable, and that is worth")
    print("saying in the report.")


if __name__ == "__main__":
    main()
