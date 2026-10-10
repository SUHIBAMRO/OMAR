"""Early-stopping robustness check, per the advisor's follow-up review
(point 5): "Highlight that B1/B2 use early stopping ... add a short
early-stopping robustness check if feasible."

What this actually checks
--------------------------
Every B1/B2 result in this study was produced with ONE early-stopping
patience value (8 validation events with no improvement, i.e. 8 x
--validate_every epochs of no improvement before stopping -- see
PFEM_Training_Colab.ipynb's own EARLY_STOP_PATIENCE=8). "Early stopping"
here really means best-checkpoint selection over a long run, not a tight
stop: the question worth answering is whether the reported best_val_error
is a stable property of training, or an artifact of this one particular
patience choice.

This script re-runs train_B1.py/train_B2.py for ONE case multiple times,
changing ONLY --early_stop_patience (same seed, same architecture, same
dataset, same epoch ceiling, same everything else) via subprocess calls
to the SAME module every other script in this project uses, not a
reimplementation of the training loop. Each patience value trains into
its own --out_dir (train_B1.py refuses to resume/re-run a directory that
already holds model_final.pt or an EARLY_STOPPED marker, so separate
directories are required, not optional).

What "robust" means here: if best_val_error and best_epoch are close
across patience values, the reported number does not depend sensitively
on this hyperparameter. If they vary a lot, that is a real finding worth
reporting honestly, not a reason to rerun until the "right" patience
appears.

Usage (one case, three patience values, everything else matching the
production protocol):
  python -m omar_pfem.early_stopping_robustness_check \
      --geometry B1 --material neo_hookean \
      --path .../hyperelastic_training_data_q4.npz \
      --batch_size 16 --epochs 2000 --validate_every 25 \
      --patiences 4,8,16 \
      --out_dir .../early_stopping_robustness_B1_neo_hookean
"""
import argparse
import json
import os
import subprocess
import sys
import time


def run_streaming(cmd, log_path):
    """Same streaming-subprocess pattern every Colab-driven script in this
    project uses (python -u so output is not buffered away)."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def read_result(case_out_dir):
    meta_path = os.path.join(case_out_dir, "best_checkpoint_meta.json")
    hist_path = os.path.join(case_out_dir, "metrics_history.json")
    stop_marker = os.path.join(case_out_dir, "EARLY_STOPPED")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found -- run did not complete "
                                 f"or wrote to an unexpected path")
    with open(meta_path) as f:
        meta = json.load(f)
    with open(hist_path) as f:
        hist = json.load(f)
    last = hist[-1] if hist else {}
    return {
        "best_val_error": meta.get("best_val_error"),
        "best_epoch": meta.get("best_epoch"),
        "stopped_early": os.path.exists(stop_marker),
        "stopped_at_epoch": last.get("epoch"),
        "wall_clock_s": last.get("cumulative_wall_clock_s"),
        "opt_steps": last.get("opt_steps"),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--geometry", required=True, choices=["B1", "B2"])
    p.add_argument("--material", required=True,
                   choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    p.add_argument("--path", required=True)
    p.add_argument("--out_dir", required=True,
                   help="each patience value trains into out_dir/patience_<p>/")
    p.add_argument("--patiences", default="4,8,16",
                   help="comma list of --early_stop_patience values to compare; "
                        "8 is the value every reported B1/B2 result actually used")
    p.add_argument("--batch_size", type=int, required=True,
                   help="the case's OWN winning batch size from its screening "
                        "protocol (training_protocol.json) -- not guessed here, "
                        "since a different batch size would confound the "
                        "comparison with a second changed variable")
    p.add_argument("--epochs", type=int, default=2000,
                   help="production protocol's own epoch ceiling (CONTINUE_EPOCHS)")
    p.add_argument("--validate_every", type=int, default=25)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2025, help="train_B1/B2's own default")
    p.add_argument("--skip_existing", action="store_true",
                   help="skip a patience value whose out_dir already has a result")
    args = p.parse_args()
    started = time.time()

    os.makedirs(args.out_dir, exist_ok=True)
    train_module = "omar_pfem.train_B1" if args.geometry == "B1" else "omar_pfem.train_B2"
    patiences = [int(x) for x in args.patiences.split(",")]

    results = {}
    for patience in patiences:
        case_out_dir = os.path.join(args.out_dir, f"patience_{patience}")
        result_path = os.path.join(case_out_dir, "best_checkpoint_meta.json")
        if args.skip_existing and os.path.exists(result_path):
            print(f"[patience={patience}] already done -> {case_out_dir}, skipping run.")
        else:
            os.makedirs(case_out_dir, exist_ok=True)
            run_streaming([
                sys.executable, "-u", "-m", train_module,
                "--path", args.path, "--material", args.material,
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--validate_every", str(args.validate_every),
                "--early_stop_patience", str(patience),
                "--early_stop_min_delta", str(args.early_stop_min_delta),
                "--seed", str(args.seed),
                "--out_dir", case_out_dir,
            ], log_path=os.path.join(case_out_dir, "run.log"))
        results[patience] = read_result(case_out_dir)
        print(f"[patience={patience}] best_val_error={results[patience]['best_val_error']:.4f} "
              f"at epoch {results[patience]['best_epoch']} "
              f"(stopped_early={results[patience]['stopped_early']}, "
              f"stopped_at_epoch={results[patience]['stopped_at_epoch']})")

    vals = [r["best_val_error"] for r in results.values() if r["best_val_error"] is not None]
    spread = {
        "min": min(vals), "max": max(vals),
        "range": max(vals) - min(vals),
        "range_pct_of_min": 100.0 * (max(vals) - min(vals)) / min(vals) if min(vals) else None,
    } if vals else None

    report = {
        "geometry": args.geometry, "material": args.material,
        "patiences_tested": patiences,
        "production_patience": 8,
        "batch_size": args.batch_size, "epochs_ceiling": args.epochs,
        "validate_every": args.validate_every, "seed": args.seed,
        "per_patience": results,
        "best_val_error_spread": spread,
        "wall_clock_s": time.time() - started,
    }
    out_json = os.path.join(args.out_dir, "early_stopping_robustness_summary.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"EARLY-STOPPING ROBUSTNESS  {args.geometry} x {args.material}")
    for patience, r in results.items():
        print(f"  patience={patience:<3} best_val_error={r['best_val_error']:.4f} "
              f"(epoch {r['best_epoch']}, stopped_at={r['stopped_at_epoch']})")
    if spread:
        print(f"  spread across patience values: {spread['range']:.4f} absolute "
              f"({spread['range_pct_of_min']:.1f}% of the smallest value)")
    print("=" * 70)
    print(f"Written to {out_json}")


if __name__ == "__main__":
    main()
