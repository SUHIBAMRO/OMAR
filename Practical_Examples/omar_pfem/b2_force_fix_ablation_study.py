"""
Ablation study for the three interventions discussed for B2/neo_hookean's
anomalously high validation error (accuracy_diagnostics.py found it
uncorrelated with E/nu/load magnitude but strongly localized: radial error
3.76x the circumferential error, and highest at the inner (pressure-loaded)
boundary, decreasing toward the outer boundary):

  1. FORCE-CONSISTENCY FIX (always applied, not optional -- see below):
     check_boundary_force_consistency.py confirmed convert_B2_quad.py's old
     "raw pressure value x exact radial direction" node_forces (no
     edge-length/quadrature weighting) differed from the FEM-consistent
     nodal force data_generate_B2.py's own solver actually used to produce
     the ground-truth displacements by a factor of ~4.4x in magnitude
     (direction agreed, cosine~0.9975) -- exactly the boundary-localized,
     direction-specific, load-magnitude-independent bias pattern found.
     data_generate_B2.py now saves that consistent force
     (inner_force_consistent) and convert_B2_quad.py now uses it
     unconditionally -- there is no "old broken" code path left to A/B
     against directly, so each variant below is compared against the OLD
     run's already-known numbers (REFERENCE_OLD_BASELINE below, from the
     accuracy_diagnostics_B2_neo_hookean run already completed) instead of
     re-running the bug.
  2. RADIAL MESH GRADING toward the inner boundary (data_generate_B2.py's
     new --r_grading), tested as a second, independent lever in case some
     of the radial/inner-boundary error is a coarse-discretization effect
     rather than (only) the force bug.
  3. INCREASED NETWORK CAPACITY (n_hidden/slice_num), tested as a third,
     independent lever in case the Transolver architecture's default
     capacity (carried over unchanged from B1) under-resolves B2's
     curved-boundary response.

Drives the full data_generate_B2 -> convert_B2_quad -> train_B2 ->
accuracy_diagnostics pipeline as subprocesses (same run_streaming pattern
as screening_study.py / batch_size_fair_comparison.py) for each variant,
then writes a comparison table against the old (pre-fix) run's numbers.

Usage (example -- adjust ntrain/ntest/epochs to match your original recipe):
  python -m omar_pfem.b2_force_fix_ablation_study \
      --material neo_hookean --num_samples 1000 --ntrain 800 --ntest 200 \
      --epochs 2000 --early_stop_patience 8 --validate_every 25 \
      --r_grading 2.5 --n_workers 8 \
      --out_dir /content/drive/MyDrive/pfem_run/B2_force_fix_ablation

Each variant's raw H5 dataset, converted NPZ, training run, and diagnostics
live under --out_dir/<variant_name>/{raw,q4,train,diagnostics}. A failure in
one variant is caught and recorded as FAILED -- it does not stop the sweep.
"""
import os
import sys
import json
import argparse
import traceback

from omar_pfem.screening_study import run_streaming

# From the accuracy_diagnostics_B2_neo_hookean run already completed on the
# OLD (pre-fix) pipeline, pasted from that run's accuracy_diagnostics.json --
# the fixed reference point every variant below is compared against.
REFERENCE_OLD_BASELINE = {
    "mean_rel_l2_error": 0.32462,
    "corr_error_vs_E": 0.014940992898197428,
    "corr_error_vs_nu": -0.03339209572320238,
    "corr_error_vs_load_magnitude": -0.04973139688298935,
    "radial_vs_circumferential_ratio": 3.764945415070387,
    "mean_rms_error_inner_boundary": 0.003933427404263057,
    "mean_rms_error_outer_boundary": 0.0023083212636993266,
    "mean_rms_error_interior": 0.0029696662569767794,
}


def run_cmd(cmd, log_path):
    returncode, output = run_streaming(cmd, log_path=log_path, raise_on_error=False)
    if returncode != 0:
        raise RuntimeError(f"Command failed (exit {returncode}): {' '.join(cmd)} -- see {log_path}")
    return output


def run_variant(name, args, r_grading, n_hidden, slice_num, out_root):
    variant_dir = os.path.join(out_root, name)
    raw_dir = os.path.join(variant_dir, "raw")
    q4_dir = os.path.join(variant_dir, "q4")
    train_dir = os.path.join(variant_dir, "train")
    diag_dir = os.path.join(variant_dir, "diagnostics")
    os.makedirs(variant_dir, exist_ok=True)

    print(f"\n{'='*90}\nVARIANT {name}: r_grading={r_grading}, n_hidden={n_hidden}, "
          f"slice_num={slice_num}\n{'='*90}")

    run_cmd([
        sys.executable, "-u", "-m", "omar_pfem.data.data_generate_B2",
        "--num_index", "1", "--num_samples", str(args.num_samples),
        "--Ntheta", str(args.Ntheta), "--Nr", str(args.Nr),
        "--r_grading", str(r_grading),
        "--material", args.material, "--n_workers", str(args.n_workers),
        "--out_dir", raw_dir,
    ], os.path.join(variant_dir, "data_generate.log"))

    run_cmd([
        sys.executable, "-u", "-m", "omar_pfem.data.convert_B2_quad",
        "--h5_dir", raw_dir, "--out_dir", q4_dir,
    ], os.path.join(variant_dir, "convert.log"))

    npz_path = os.path.join(q4_dir, "hyperelastic_training_data_q4.npz")

    train_cmd = [
        sys.executable, "-u", "-m", "omar_pfem.train_B2",
        "--path", npz_path, "--material", args.material,
        "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
        "--lr", str(args.lr), "--epochs", str(args.epochs),
        "--save_every", str(args.validate_every), "--validate_every", str(args.validate_every),
        "--early_stop_patience", str(args.early_stop_patience),
        "--n_hidden", str(n_hidden), "--slice_num", str(slice_num),
        "--print_every", "999999",
        "--out_dir", train_dir,
    ]
    if args.cpu:
        train_cmd.append("--cpu")
    run_cmd(train_cmd, os.path.join(variant_dir, "train.log"))

    ckpt = os.path.join(train_dir, "model_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(train_dir, "model_final.pt")

    diag_cmd = [
        sys.executable, "-u", "-m", "omar_pfem.accuracy_diagnostics",
        "--geometry", "B2", "--material", args.material,
        "--checkpoint", ckpt, "--dataset", npz_path,
        "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
        "--n_hidden", str(n_hidden), "--slice_num", str(slice_num),
        "--out_dir", diag_dir,
    ]
    if args.cpu:
        diag_cmd.append("--cpu")
    run_cmd(diag_cmd, os.path.join(variant_dir, "diagnostics.log"))

    with open(os.path.join(diag_dir, "accuracy_diagnostics.json")) as f:
        report = json.load(f)
    return report


def summarize(name, report):
    corr = report["correlations_and_breakdowns"]
    return {
        "variant": name,
        "mean_rel_l2_error": report["mean_rel_l2_error"],
        "corr_error_vs_E": corr["corr_error_vs_E"],
        "corr_error_vs_nu": corr["corr_error_vs_nu"],
        "corr_error_vs_load_magnitude": corr["corr_error_vs_load_magnitude"],
        "radial_vs_circumferential_ratio": corr["radial_vs_circumferential_ratio"],
        "mean_rms_error_inner_boundary": corr["mean_rms_error_inner_boundary"],
        "mean_rms_error_outer_boundary": corr["mean_rms_error_outer_boundary"],
        "mean_rms_error_interior": corr["mean_rms_error_interior"],
    }


def write_comparison_table(rows, out_dir):
    headers = ["variant", "mean_rel_l2_error", "radial_vs_circumferential_ratio",
               "mean_rms_error_inner_boundary", "mean_rms_error_outer_boundary",
               "mean_rms_error_interior", "corr_error_vs_E", "corr_error_vs_nu",
               "corr_error_vs_load_magnitude"]

    all_rows = [dict(REFERENCE_OLD_BASELINE, variant="OLD_BASELINE_pre_fix")] + rows

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in all_rows:
        cells = []
        for h in headers:
            v = r.get(h)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    md_path = os.path.join(out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# B2 force-fix ablation summary\n\n")
        f.write("\n".join(lines))
        f.write("\n")

    json_path = os.path.join(out_dir, "ablation_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    print("\n" + "\n".join(lines) + "\n")
    print(f"Summary written to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser("B2 force-consistency-fix ablation study")
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--Ntheta", type=int, default=21)
    parser.add_argument("--Nr", type=int, default=21)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--validate_every", type=int, default=25)
    parser.add_argument("--early_stop_patience", type=int, default=8)

    parser.add_argument("--r_grading", type=float, default=2.5,
                         help="Radial grading exponent used by the force_fixed_graded variant "
                              "(see data_generate_B2.py's generate_grid_Q4_ring)")
    parser.add_argument("--bigcap_n_hidden", type=int, default=384)
    parser.add_argument("--bigcap_slice_num", type=int, default=192)

    parser.add_argument("--variants", type=str, default="force_fixed,force_fixed_graded,force_fixed_bigcap",
                         help="Comma-separated subset of: force_fixed, force_fixed_graded, force_fixed_bigcap")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]

    variant_defs = {
        "force_fixed": dict(r_grading=1.0, n_hidden=256, slice_num=128),
        "force_fixed_graded": dict(r_grading=args.r_grading, n_hidden=256, slice_num=128),
        "force_fixed_bigcap": dict(r_grading=1.0, n_hidden=args.bigcap_n_hidden, slice_num=args.bigcap_slice_num),
    }

    rows = []
    for name in requested:
        if name not in variant_defs:
            print(f"Skipping unknown variant '{name}' (known: {list(variant_defs)})")
            continue
        cfg = variant_defs[name]
        try:
            report = run_variant(name, args, cfg["r_grading"], cfg["n_hidden"], cfg["slice_num"], args.out_dir)
            row = summarize(name, report)
            rows.append(row)
            print(f"\n----- {name}: mean_rel_l2_error={row['mean_rel_l2_error']:.4e} "
                  f"(old baseline was {REFERENCE_OLD_BASELINE['mean_rel_l2_error']:.4e}), "
                  f"radial/circ ratio={row['radial_vs_circumferential_ratio']:.3f} "
                  f"(old was {REFERENCE_OLD_BASELINE['radial_vs_circumferential_ratio']:.3f}) -----")
        except Exception:
            print(f"\n----- {name}: FAILED -----")
            traceback.print_exc()
            rows.append({"variant": name, "status": "FAILED"})

    write_comparison_table(rows, args.out_dir)


if __name__ == "__main__":
    main()
