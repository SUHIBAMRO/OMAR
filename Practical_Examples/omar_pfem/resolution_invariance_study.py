"""
Resolution-invariance study, per the advisor's request: "I would also
expect a discussion or numerical demonstration of how the method behaves
across different spatial resolutions ... train on two different meshes and
then training on further 5 different meshes (altogether 10) and showing
the results are always the same."

No new capability is needed in the data-generation or training scripts
for this: data_generate_B{1,2}.py already take --Nx/--Ny (B1) or --Ntheta/
--Nr (B2) as CLI arguments (this study's MESH_N=21 was always just one
choice of that argument, see the Colab notebook's config cell), and
Transolver is architecturally mesh-size-agnostic (it consumes per-node
coordinates + a material/load feature vector, not a fixed-size grid
input) -- train_B{1,2}.py trains on whatever mesh the NPZ dataset it is
given actually has. What was missing was a DRIVER that runs the existing
pipeline (generate dataset -> convert -> train) across a sequence of mesh
resolutions and collects the resulting validation errors side by side,
exactly analogous to screening_study.py's batch-size sweep -- this script
is that driver, swept over mesh resolution instead.

Cost note: this is the single most expensive item among the advisor's
requests -- generating a fresh ~1000-sample dataset and training a full
run at EACH of ~10 resolutions is roughly 10x the cost of training one
case. --epochs/--early_stop_patience here default to a REDUCED budget
(not Table 2's full protocol) specifically to keep the total wall-clock
cost of this study tractable; override them if the full protocol is
wanted at every resolution and the compute budget allows it.

Usage:
  python -m omar_pfem.resolution_invariance_study \
      --geometry B1 --material neo_hookean \
      --resolutions 9,13,17,21,25,29,33,37,41,45 \
      --ntrain 800 --ntest 200 --epochs 500 --batch_size 8 \
      --validate_every 25 --early_stop_patience 8 \
      --out_dir ./resolution_study_B1_neo_hookean
"""
import os
import sys
import json
import argparse

from omar_pfem.screening_study import run_streaming, read_metrics


def case_name_for_resolution(geometry, material, N):
    return f"{geometry}_{material}_N{N}"


def main():
    parser = argparse.ArgumentParser("Resolution-invariance study: train the same protocol at multiple mesh resolutions")
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--resolutions", type=str, default="9,13,17,21,25,29,33,37,41,45",
                         help="Comma-separated node-count-per-side values (~10, per the advisor's request); "
                              "21 is this study's normal operator-learning mesh, included by default "
                              "so this study's own result appears as one point in the comparison")
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--n_workers", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8,
                         help="Default 8 = this study's winning batch size from Table 2, "
                              "applied uniformly here too so resolution is the only thing varying")
    parser.add_argument("--epochs", type=int, default=500,
                         help="REDUCED from Table 2's 2000-epoch upper bound, to keep the total "
                              "cost of ~10 full training runs tractable -- see module docstring")
    parser.add_argument("--validate_every", type=int, default=25)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-3)

    parser.add_argument("--data_dir", type=str, default="./resolution_study_data",
                         help="Scratch directory for per-resolution generated datasets")
    parser.add_argument("--out_dir", type=str, default="./resolution_study")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    resolutions = [int(n) for n in args.resolutions.split(",") if n.strip()]

    gen_module = "omar_pfem.data.data_generate_B1" if args.geometry == "B1" else "omar_pfem.data.data_generate_B2"
    conv_module = "omar_pfem.data.convert_B1_quad" if args.geometry == "B1" else "omar_pfem.data.convert_B2_quad"
    train_module = "omar_pfem.train_B1" if args.geometry == "B1" else "omar_pfem.train_B2"
    mesh_flag_names = ("--Nx", "--Ny") if args.geometry == "B1" else ("--Ntheta", "--Nr")

    rows = []
    for N in resolutions:
        name = case_name_for_resolution(args.geometry, args.material, N)
        print(f"\n{'='*80}\nRESOLUTION N={N}  ({name})\n{'='*80}")

        h5_dir = os.path.join(args.data_dir, f"fem_{name}")
        npz_dir = os.path.join(args.data_dir, f"npz_{name}")
        npz_path = os.path.join(npz_dir, "hyperelastic_training_data_q4.npz")
        run_dir = os.path.join(args.out_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        if not os.path.exists(npz_path):
            run_streaming([
                sys.executable, "-u", "-m", gen_module,
                "--num_index", "1", "--num_samples", str(args.ntrain + args.ntest),
                mesh_flag_names[0], str(N), mesh_flag_names[1], str(N),
                "--material", args.material, "--n_workers", str(args.n_workers),
                "--out_dir", h5_dir,
            ], log_path=os.path.join(run_dir, "data_generation.log"))
            run_streaming([
                sys.executable, "-u", "-m", conv_module,
                "--h5_dir", h5_dir, "--out_dir", npz_dir,
            ], log_path=os.path.join(run_dir, "npz_conversion.log"))
        else:
            print(f"[{name}] NPZ already exists at {npz_path}, skipping generation.")

        train_cmd = [
            sys.executable, "-u", "-m", train_module,
            "--path", npz_path, "--material", args.material,
            "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
            "--lr", str(args.lr), "--print_every", "999999",
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--save_every", str(args.validate_every),
            "--validate_every", str(args.validate_every),
            "--early_stop_patience", str(args.early_stop_patience),
            "--early_stop_min_delta", str(args.early_stop_min_delta),
            "--out_dir", run_dir,
        ]
        if args.cpu:
            train_cmd.append("--cpu")

        already_done = os.path.exists(os.path.join(run_dir, "model_final.pt")) or \
            os.path.exists(os.path.join(run_dir, "EARLY_STOPPED"))
        if not already_done:
            run_streaming(train_cmd, log_path=os.path.join(run_dir, "training.log"))
        else:
            print(f"[{name}] already fully trained, skipping.")

        history = read_metrics(run_dir)
        if history:
            best_rec = min(history, key=lambda r: r["val_error"])
            final_rec = history[-1]
            rows.append({
                "N": N, "n_nodes": final_rec.get("n_nodes"),
                "best_val_error": best_rec["val_error"], "best_epoch": best_rec["epoch"],
                "final_epoch": final_rec["epoch"],
                "wall_clock_s": final_rec["cumulative_wall_clock_s"],
            })
        else:
            rows.append({"N": N, "best_val_error": None, "note": "no metrics_history.json produced"})

    print("\n" + "=" * 100)
    print("RESOLUTION-INVARIANCE STUDY SUMMARY")
    print("=" * 100)
    print(f"{'N':>5s} {'best_val_error':>16s} {'best_epoch':>12s} {'final_epoch':>12s} {'wall_clock_s':>14s}")
    for row in rows:
        if row.get("best_val_error") is not None:
            print(f"{row['N']:>5d} {row['best_val_error']:>16.4e} {row['best_epoch']:>12d} "
                  f"{row['final_epoch']:>12d} {row['wall_clock_s']:>14.1f}")
        else:
            print(f"{row['N']:>5d} {'FAILED':>16s}")
    errs = [r["best_val_error"] for r in rows if r.get("best_val_error") is not None]
    if len(errs) > 1:
        print(f"\nbest_val_error range across {len(errs)} resolutions: "
              f"[{min(errs):.4e}, {max(errs):.4e}]  (spread = {(max(errs)-min(errs))/min(errs):.1%} "
              f"of the smallest value)")
    print("=" * 100)

    summary_path = os.path.join(args.out_dir, "resolution_study_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"geometry": args.geometry, "material": args.material,
                   "resolutions": resolutions, "rows": rows}, f, indent=2)
    print(f"Full report written to {summary_path}")


if __name__ == "__main__":
    main()
