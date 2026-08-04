"""
Fair batch-size comparison, per the advisor's third-round correction to the
existing screening_study.py: "The batch-size comparison is confounded by
the optimization budget. At batch size 8 the model receives 40,000
optimizer updates, whereas at batch size 256 it receives only 1,600 ...
repeat the comparison using equal optimizer steps or equal numbers of
processed samples, and retune or scale the learning rate appropriately.
Please plot the error against optimizer steps, processed samples, and
wall-clock time."

This does NOT change train_B1.py/train_B2.py's training loop (no new
instrumentation needed there: metrics_history.json already logs epoch,
opt_steps, cumulative_wall_clock_s, and val_error at every validation
event -- everything the three requested plots need). What was missing was
a DRIVER that:
  1. Targets a FIXED TOTAL OPTIMIZER-STEP BUDGET across all batch sizes
     (not a fixed epoch count, which is exactly the confound the advisor
     flagged), by computing --epochs = ceil(target_steps / steps_per_epoch)
     separately for each batch size, where steps_per_epoch =
     ceil(ntrain / batch_size).
  2. For EACH batch size, sweeps more than one learning rate -- the
     baseline (unscaled) lr, a linearly-scaled lr (lr * B/B0), and a
     sqrt-scaled lr (lr * sqrt(B/B0)) -- since the advisor explicitly
     warned that blindly applying a single scaling rule is not enough.
  3. Chooses each batch size's --validate_every (in epochs, the only unit
     train_B1.py/B2.py support) so validation events land at roughly the
     same optimizer-step spacing across batch sizes, keeping the resulting
     curves comparably resolved.
  4. Collects every run's full metrics_history.json into one combined
     table and renders the three requested plots (error vs. optimizer
     steps, vs. processed samples = opt_steps * batch_size, vs. wall-clock
     time), one line per (batch_size, lr_rule) combination.

Usage:
  python -m omar_pfem.batch_size_fair_comparison \
      --path <npz path> --geometry B1 --material neo_hookean \
      --batch_sizes 8,16,32,64,128,256 --base_batch_size 8 --base_lr 2e-3 \
      --lr_rules baseline,linear,sqrt --target_opt_steps 40000 \
      --target_validate_every_steps 1000 \
      --out_dir ./batch_size_fair_B1_neo_hookean
"""
import os
import sys
import json
import math
import argparse

from omar_pfem.screening_study import run_streaming, read_metrics, is_oom_error


def lr_for_rule(base_lr, base_bs, bs, rule):
    if rule == "baseline":
        return base_lr
    elif rule == "linear":
        return base_lr * (bs / base_bs)
    elif rule == "sqrt":
        return base_lr * math.sqrt(bs / base_bs)
    raise ValueError(rule)


def build_train_cmd(geometry, extra_args):
    train_module = "omar_pfem.train_B1" if geometry == "B1" else "omar_pfem.train_B2"
    return [sys.executable, "-u", "-m", train_module] + extra_args


def main():
    parser = argparse.ArgumentParser("Fair batch-size comparison: equal optimizer-step budget + LR sweep")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--geometry", type=str, default="B1", choices=["B1", "B2"])
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)

    parser.add_argument("--batch_sizes", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--base_batch_size", type=int, default=8,
                         help="Reference batch size the LR-scaling rules are anchored to "
                              "(this study's original winning batch size)")
    parser.add_argument("--base_lr", type=float, default=2e-3,
                         help="Learning rate AT base_batch_size (this study's original lr)")
    parser.add_argument("--lr_rules", type=str, default="baseline,linear,sqrt",
                         help="Comma-separated: baseline (same lr at every batch size), "
                              "linear (lr*B/B0), sqrt (lr*sqrt(B/B0)) -- run per the advisor's "
                              "explicit warning not to apply one rule blindly without comparison")

    parser.add_argument("--target_opt_steps", type=int, default=40000,
                         help="Total optimizer-step budget EVERY batch size is trained to "
                              "(default 40,000 = this study's original batch_size=8 budget at "
                              "400 epochs, so this comparison is a strict superset of the "
                              "earlier, confounded one)")
    parser.add_argument("--target_validate_every_steps", type=int, default=1000,
                         help="Approximate optimizer-step spacing between validation events, "
                              "converted to --validate_every epochs per batch size")

    parser.add_argument("--lr", type=float, default=None, help=argparse.SUPPRESS)  # unused, avoid confusion
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./batch_size_fair_comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    lr_rules = [r.strip() for r in args.lr_rules.split(",") if r.strip()]

    common_args = [
        "--path", args.path, "--material", args.material,
        "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
        "--print_every", "999999",
    ]
    if args.cpu:
        common_args.append("--cpu")

    plan = []
    for bs in batch_sizes:
        steps_per_epoch = math.ceil(args.ntrain / bs)
        epochs = max(1, math.ceil(args.target_opt_steps / steps_per_epoch))
        validate_every = max(1, round(args.target_validate_every_steps / steps_per_epoch))
        for rule in lr_rules:
            lr = lr_for_rule(args.base_lr, args.base_batch_size, bs, rule)
            plan.append({
                "batch_size": bs, "lr_rule": rule, "lr": lr,
                "steps_per_epoch": steps_per_epoch, "epochs": epochs,
                "validate_every": validate_every,
                "actual_opt_steps_at_end": epochs * steps_per_epoch,
            })

    print(f"Plan: {len(plan)} runs (batch_sizes={batch_sizes} x lr_rules={lr_rules}), "
          f"each targeting ~{args.target_opt_steps} optimizer steps")
    for p in plan:
        print(f"  bs={p['batch_size']:>4d} rule={p['lr_rule']:<8s} lr={p['lr']:.2e} "
              f"epochs={p['epochs']:>5d} validate_every={p['validate_every']:>3d} "
              f"(-> {p['actual_opt_steps_at_end']} opt_steps at completion)")

    all_rows = []  # one row per validation event, across every run
    run_summaries = []
    for p in plan:
        run_name = f"bs{p['batch_size']}_{p['lr_rule']}"
        run_dir = os.path.join(args.out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n----- {run_name}: lr={p['lr']:.4e}, epochs={p['epochs']}, "
              f"validate_every={p['validate_every']} -----")

        returncode, output = run_streaming(build_train_cmd(args.geometry, common_args + [
            "--batch_size", str(p["batch_size"]),
            "--lr", str(p["lr"]),
            "--epochs", str(p["epochs"]),
            "--save_every", str(p["validate_every"]),
            "--validate_every", str(p["validate_every"]),
            "--early_stop_patience", "0",  # no early stopping: every run must reach the SAME step budget
            "--out_dir", run_dir,
        ]), log_path=os.path.join(run_dir, "run.log"), raise_on_error=False)

        status = "OK" if returncode == 0 else ("OOM" if is_oom_error(output) else "FAILED")
        history = read_metrics(run_dir)
        for rec in history:
            all_rows.append({
                "batch_size": p["batch_size"], "lr_rule": p["lr_rule"], "lr": p["lr"],
                "epoch": rec["epoch"], "opt_steps": rec["opt_steps"],
                "processed_samples": rec["opt_steps"] * p["batch_size"],
                "wall_clock_s": rec["cumulative_wall_clock_s"], "val_error": rec["val_error"],
            })
        best = min(history, key=lambda r: r["val_error"]) if history else None
        run_summaries.append({
            **p, "status": status,
            "best_val_error": best["val_error"] if best else None,
            "best_opt_steps": best["opt_steps"] if best else None,
            "final_val_error": history[-1]["val_error"] if history else None,
            "final_wall_clock_s": history[-1]["cumulative_wall_clock_s"] if history else None,
        })
        print(f"----- {run_name}: status={status}, "
              f"best_val_error={best['val_error']:.4e} at opt_steps={best['opt_steps']}"
              if best else f"----- {run_name}: status={status}, no metrics produced -----")

    with open(os.path.join(args.out_dir, "fair_comparison_all_validation_events.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(os.path.join(args.out_dir, "fair_comparison_run_summaries.json"), "w") as f:
        json.dump(run_summaries, f, indent=2)

    print(f"\n{'='*100}\nFAIR BATCH-SIZE COMPARISON SUMMARY (equal optimizer-step budget "
          f"~{args.target_opt_steps} for every run)\n{'='*100}")
    print(f"{'batch_size':>10s} {'lr_rule':>8s} {'lr':>10s} {'status':>7s} "
          f"{'best_val_error':>15s} {'best_opt_steps':>15s} {'final_wall_clock_s':>19s}")
    for r in run_summaries:
        print(f"{r['batch_size']:>10d} {r['lr_rule']:>8s} {r['lr']:>10.2e} {r['status']:>7s} "
              f"{str(r['best_val_error']):>15s} {str(r['best_opt_steps']):>15s} "
              f"{str(r['final_wall_clock_s']):>19s}")

    _make_plots(all_rows, args.out_dir)


def _make_plots(all_rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot generation (JSON data is still written).")
        return

    series = {}
    for r in all_rows:
        key = (r["batch_size"], r["lr_rule"])
        series.setdefault(key, []).append(r)
    for key in series:
        series[key].sort(key=lambda r: r["opt_steps"])

    x_fields = [("opt_steps", "Optimizer steps"),
                ("processed_samples", "Processed samples (opt_steps x batch_size)"),
                ("wall_clock_s", "Wall-clock time (s)")]

    for x_field, x_label in x_fields:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (bs, rule), rows in sorted(series.items()):
            ax.plot([r[x_field] for r in rows], [r["val_error"] for r in rows],
                    marker="o", markersize=3, label=f"bs={bs}, {rule}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Validation error (mean rel. L2, u & v)")
        ax.set_yscale("log")
        ax.set_title("Fair batch-size comparison (equal optimizer-step budget)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"fair_comparison_vs_{x_field}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Plot written: {out_path}")


if __name__ == "__main__":
    main()
