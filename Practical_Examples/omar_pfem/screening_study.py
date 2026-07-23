"""
Systematic batch-size screening study for the PFEM/Transolver training
protocol, per the advisor's request: before committing to a fixed
training recipe across all 6 (geometry x material) cases, run a short,
controlled comparison on one representative case (B1 x Neo-Hookean by
default) across a handful of batch sizes, then continue only the most
promising one to a larger epoch budget with early stopping.

This does NOT reimplement any training logic -- it drives train_B1.py /
train_B2.py as subprocesses (same pattern as PFEM_Training_Colab.ipynb)
and reads back the metrics_history.json each run already writes (val
error, wall-clock, GPU peak memory, optimizer steps -- all added to
train_B1.py alongside real mini-batch support, best-checkpoint tracking,
and early stopping).

Usage (example matching the advisor's protocol):
  python -m omar_pfem.screening_study \
      --path <npz path> --geometry B1 --material neo_hookean \
      --batch_sizes 4,8,16 --screen_epochs 400 --validate_every 25 \
      --continue_epochs 2000 --continue_early_stop_patience 8 \
      --out_dir ./screening_B1_neo_hookean

Produces, under --out_dir:
  screening_summary.json / screening_summary.md  -- the comparison table
  bs{N}/                                          -- one screening run per batch size
  winner_bs{N}/                                   -- the continued (best) run
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
import time


def run_streaming(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def read_metrics(out_dir):
    path = os.path.join(out_dir, "metrics_history.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def summarize_run(out_dir, batch_size):
    history = read_metrics(out_dir)
    if not history:
        return {
            "batch_size": batch_size, "epochs_run": 0, "opt_steps": 0,
            "wall_clock_s": 0.0, "gpu_peak_mem_mb": 0.0,
            "final_val_error": None, "best_val_error": None, "best_epoch": None,
        }

    best_rec = min(history, key=lambda r: r["val_error"])
    final_rec = history[-1]
    peak_mem = max(r.get("gpu_peak_mem_mb", 0.0) for r in history)

    return {
        "batch_size": batch_size,
        "epochs_run": final_rec["epoch"],
        "opt_steps": final_rec["opt_steps"],
        "wall_clock_s": final_rec["cumulative_wall_clock_s"],
        "gpu_peak_mem_mb": peak_mem,
        "final_val_error": final_rec["val_error"],
        "best_val_error": best_rec["val_error"],
        "best_epoch": best_rec["epoch"],
    }


def build_train_cmd(geometry, common_args, extra_args):
    train_module = "omar_pfem.train_B1" if geometry == "B1" else "omar_pfem.train_B2"
    cmd = [sys.executable, "-u", "-m", train_module] + common_args + extra_args
    return cmd


def write_summary_table(rows, out_dir, winner_bs):
    json_path = os.path.join(out_dir, "screening_summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    headers = ["batch_size", "epochs_run", "opt_steps", "wall_clock_s",
               "gpu_peak_mem_mb", "final_val_error", "best_val_error", "best_epoch"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        marker = " **(winner)**" if r["batch_size"] == winner_bs else ""
        lines.append("| " + " | ".join(cells) + f" |{marker}")

    md_path = os.path.join(out_dir, "screening_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Batch-size screening summary\n\n")
        f.write("\n".join(lines))
        f.write(f"\n\nWinner: batch_size={winner_bs} (lowest best_val_error)\n")

    print("\n" + "\n".join(lines) + "\n")
    print(f"Summary written to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser("Batch-size screening study for the PFEM/Transolver protocol")
    parser.add_argument("--path", type=str, required=True, help="NPZ dataset path")
    parser.add_argument("--geometry", type=str, default="B1", choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--out_dir", type=str, default="./screening_study")
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)

    parser.add_argument("--batch_sizes", type=str, default="4,8,16",
                         help="Comma-separated batch sizes to screen")
    parser.add_argument("--screen_epochs", type=int, default=400)
    parser.add_argument("--validate_every", type=int, default=25)

    parser.add_argument("--continue_epochs", type=int, default=2000)
    parser.add_argument("--continue_early_stop_patience", type=int, default=8)
    parser.add_argument("--continue_early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--final_out_dir", type=str, default=None,
                         help="If set, Phase 2 (the continued winning run) writes directly here "
                              "instead of out_dir/bs{N} -- point this at the case's real results "
                              "directory (e.g. RESULTS_DIR/B1_neo_hookean) so the screening study's "
                              "own continuation phase IS that case's training run, instead of a "
                              "separate copy that the main 6-case loop would then redundantly retrain.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    common_args = [
        "--path", args.path,
        "--material", args.material,
        "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
        "--lr", str(args.lr),
        "--print_every", "999999",
    ]
    if args.cpu:
        common_args.append("--cpu")

    # ---- Phase 1: screen each batch size for a short, fixed epoch budget ----
    print(f"\n{'='*80}\nPHASE 1: Screening batch sizes {batch_sizes} for {args.screen_epochs} "
          f"epochs each (validate_every={args.validate_every})\n{'='*80}")

    rows = []
    for bs in batch_sizes:
        run_dir = os.path.join(args.out_dir, f"bs{bs}")
        print(f"\n----- Screening batch_size={bs} -----")
        t0 = time.time()
        run_streaming(build_train_cmd(args.geometry, common_args, [
            "--batch_size", str(bs),
            "--epochs", str(args.screen_epochs),
            "--save_every", str(args.screen_epochs),
            "--validate_every", str(args.validate_every),
            "--early_stop_patience", "0",
            "--out_dir", run_dir,
        ]))
        print(f"----- Done batch_size={bs} in {time.time()-t0:.0f}s wall (driver-side) -----")
        rows.append(summarize_run(run_dir, bs))

    # ---- Pick the winner: lowest best_val_error across the screened runs ----
    scored = [r for r in rows if r["best_val_error"] is not None]
    if not scored:
        raise RuntimeError("No screening run produced any validation metrics -- check the logs above.")
    winner = min(scored, key=lambda r: r["best_val_error"])
    winner_bs = winner["batch_size"]

    write_summary_table(rows, args.out_dir, winner_bs)

    # ---- Phase 2: continue the winning config to a larger epoch budget,
    # resuming from its own screening checkpoint, with early stopping ----
    print(f"\n{'='*80}\nPHASE 2: Continuing batch_size={winner_bs} (best screening val_error="
          f"{winner['best_val_error']:.4g}) to {args.continue_epochs} epochs, "
          f"early_stop_patience={args.continue_early_stop_patience}\n{'='*80}")

    screen_dir = os.path.join(args.out_dir, f"bs{winner_bs}")

    if args.final_out_dir:
        # Seed the case's real results directory with the winner's screening
        # checkpoint/history so the continuation resumes from where
        # screening left off, instead of restarting from scratch.
        continue_dir = args.final_out_dir
        os.makedirs(continue_dir, exist_ok=True)
        for fname in os.listdir(screen_dir):
            if fname.startswith("model_epoch") or fname in (
                "metrics_history.json", "best_checkpoint_meta.json", "model_best.pt",
            ):
                shutil.copy2(os.path.join(screen_dir, fname), os.path.join(continue_dir, fname))
    else:
        continue_dir = screen_dir

    final_marker = os.path.join(continue_dir, "model_final.pt")
    if os.path.exists(final_marker):
        # The screening run completed its fixed epoch budget normally, which
        # writes model_final.pt -- remove it so the continuation run's resume
        # logic (based on the latest model_epoch{N}.pt) doesn't mistake the
        # screening run for an already-fully-trained case and skip entirely.
        os.remove(final_marker)

    run_streaming(build_train_cmd(args.geometry, common_args, [
        "--batch_size", str(winner_bs),
        "--epochs", str(args.continue_epochs),
        "--save_every", str(args.validate_every),
        "--validate_every", str(args.validate_every),
        "--early_stop_patience", str(args.continue_early_stop_patience),
        "--early_stop_min_delta", str(args.continue_early_stop_min_delta),
        "--out_dir", continue_dir,
    ]))

    final_summary = summarize_run(continue_dir, winner_bs)
    print(f"\n{'='*80}\nFINAL: batch_size={winner_bs} reached epoch {final_summary['epochs_run']}, "
          f"best_val_error={final_summary['best_val_error']:.4g} at epoch {final_summary['best_epoch']} "
          f"(wall_clock={final_summary['wall_clock_s']:.0f}s, opt_steps={final_summary['opt_steps']})\n"
          f"{'='*80}")
    print(f"\nProposed automatic protocol for the remaining 5 cases: batch_size={winner_bs}, "
          f"validate_every={args.validate_every}, early_stop_patience={args.continue_early_stop_patience}, "
          f"early_stop_min_delta={args.continue_early_stop_min_delta}, epochs={args.continue_epochs} "
          f"(as an upper bound -- early stopping decides the actual length per case).")


if __name__ == "__main__":
    main()
