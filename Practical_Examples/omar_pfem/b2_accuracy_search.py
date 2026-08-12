"""
Bounded, resumable automated search over B2/neo_hookean training configs,
built on top of the force-consistency fix (data_generate_B2.py's
inner_force_consistent, convert_B2_quad.py, and train_B2.py's
--loss_force_norm; see b2_force_fix_ablation_study.py for the original
single-fix ablation this extends).

Motivation: the force-consistency fix alone regressed accuracy (32.46% ->
94.08%, see ablation_summary.json) instead of improving it.
--loss_force_norm (train_B2.py) normalizes the Pi=U-W training objective
by each sample's own boundary-force scale -- a fixed, uv-independent
positive constant, so the true minimizer is unchanged, only the
optimizer's gradient conditioning is restored -- to counter the ~13-16x
smaller |W| the correct force produces relative to the old, uncorrected
approximation. This driver tries that fix, plus a small number of
additional escalating levers (higher LR, radial mesh grading, larger
network capacity) ONLY as needed, stopping as soon as one variant's
mean_rel_l2_error drops below --target, or after --max_trials full-scale
attempts, whichever comes first -- each full-scale attempt takes roughly
an hour on a single GPU (see Table 5 of the report: B2 x Neo-Hookean's
own prior run took 3257.1s wall-clock), so an unbounded search is not
acceptable.

Resumability (point this --out_dir at a Google Drive path):
  - Trial-level: each trial writes a TRIAL_DONE.json marker only once its
    full data_generate -> convert -> train -> accuracy_diagnostics
    pipeline has completed; on restart, already-marked trials are
    skipped entirely (no recomputation), matching this codebase's
    existing checkpoint/resume convention used throughout (see
    train_B2.py's own find_latest_checkpoint, high_dof_convergence_study.py).
  - Within a trial: data_generate_B2.py / convert_B2_quad.py / train_B2.py
    each already resume from their own partial progress if interrupted
    (train_B2.py's model_epoch*.pt checkpoints in particular), so a
    Colab disconnect mid-trial does not restart that trial from scratch --
    just re-run this exact same command and it picks back up.

Usage (example):
  python -m omar_pfem.b2_accuracy_search \\
      --material neo_hookean --target 0.09 --max_trials 5 \\
      --out_dir /content/drive/MyDrive/pfem_run/B2_accuracy_search
"""
import os
import sys
import json
import argparse
import traceback

from omar_pfem.screening_study import run_streaming


def run_cmd(cmd, log_path):
    returncode, output = run_streaming(cmd, log_path=log_path, raise_on_error=False)
    if returncode != 0:
        raise RuntimeError(f"Command failed (exit {returncode}): {' '.join(cmd)} -- see {log_path}")
    return output


# Escalating candidate list: the theoretically-motivated fix
# (loss_force_norm=1) is common to every candidate; each subsequent
# candidate adds exactly one additional lever on top of the previous
# ones' full set, so a later success also tells us which combination of
# levers mattered, not just "some combination works".
CANDIDATES = [
    {"name": "lossnorm",                 "lr": None, "r_grading": 1.0, "n_hidden": 256, "slice_num": 128},
    {"name": "lossnorm_lr5e3",            "lr": 5e-3, "r_grading": 1.0, "n_hidden": 256, "slice_num": 128},
    {"name": "lossnorm_graded",           "lr": 5e-3, "r_grading": 2.5, "n_hidden": 256, "slice_num": 128},
    {"name": "lossnorm_bigcap",           "lr": 5e-3, "r_grading": 2.5, "n_hidden": 384, "slice_num": 192},
    {"name": "lossnorm_bigcap_lr1e2",     "lr": 1e-2, "r_grading": 2.5, "n_hidden": 384, "slice_num": 192},
]


def run_trial(cand, args, out_root):
    name = cand["name"]
    lr = cand["lr"] if cand["lr"] is not None else args.lr
    trial_dir = os.path.join(out_root, name)
    raw_dir = os.path.join(trial_dir, "raw")
    q4_dir = os.path.join(trial_dir, "q4")
    train_dir = os.path.join(trial_dir, "train")
    diag_dir = os.path.join(trial_dir, "diagnostics")
    done_marker = os.path.join(trial_dir, "TRIAL_DONE.json")
    os.makedirs(trial_dir, exist_ok=True)

    if os.path.exists(done_marker):
        with open(done_marker) as f:
            result = json.load(f)
        print(f"\n[skip] {name}: already completed (mean_rel_l2_error="
              f"{result['mean_rel_l2_error']:.4f}) -- see {done_marker}")
        return result

    print(f"\n{'='*90}\nTRIAL {name}: lr={lr}, r_grading={cand['r_grading']}, "
          f"n_hidden={cand['n_hidden']}, slice_num={cand['slice_num']}, "
          f"loss_force_norm=1\n{'='*90}")

    run_cmd([
        sys.executable, "-u", "-m", "omar_pfem.data.data_generate_B2",
        "--num_index", "1", "--num_samples", str(args.num_samples),
        "--Ntheta", str(args.Ntheta), "--Nr", str(args.Nr),
        "--r_grading", str(cand["r_grading"]),
        "--material", args.material, "--n_workers", str(args.n_workers),
        "--out_dir", raw_dir,
    ], os.path.join(trial_dir, "data_generate.log"))

    run_cmd([
        sys.executable, "-u", "-m", "omar_pfem.data.convert_B2_quad",
        "--h5_dir", raw_dir, "--out_dir", q4_dir,
    ], os.path.join(trial_dir, "convert.log"))

    npz_path = os.path.join(q4_dir, "hyperelastic_training_data_q4.npz")

    train_cmd = [
        sys.executable, "-u", "-m", "omar_pfem.train_B2",
        "--path", npz_path, "--material", args.material,
        "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
        "--lr", str(lr), "--epochs", str(args.epochs),
        "--save_every", str(args.validate_every), "--validate_every", str(args.validate_every),
        "--early_stop_patience", str(args.early_stop_patience),
        "--loss_force_norm", "1",
        "--n_hidden", str(cand["n_hidden"]), "--slice_num", str(cand["slice_num"]),
        "--print_every", "999999",
        "--out_dir", train_dir,
    ]
    if args.cpu:
        train_cmd.append("--cpu")
    # train_B2.py resumes on its own from model_epoch*.pt if this exact
    # command is re-run after an interruption -- nothing extra needed here.
    run_cmd(train_cmd, os.path.join(trial_dir, "train.log"))

    ckpt = os.path.join(train_dir, "model_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(train_dir, "model_final.pt")

    diag_cmd = [
        sys.executable, "-u", "-m", "omar_pfem.accuracy_diagnostics",
        "--geometry", "B2", "--material", args.material,
        "--checkpoint", ckpt, "--dataset", npz_path,
        "--ntrain", str(args.ntrain), "--ntest", str(args.ntest),
        "--n_hidden", str(cand["n_hidden"]), "--slice_num", str(cand["slice_num"]),
        "--out_dir", diag_dir,
    ]
    if args.cpu:
        diag_cmd.append("--cpu")
    run_cmd(diag_cmd, os.path.join(trial_dir, "diagnostics.log"))

    with open(os.path.join(diag_dir, "accuracy_diagnostics.json")) as f:
        report = json.load(f)

    result = {
        "trial": name,
        "lr": lr,
        "r_grading": cand["r_grading"],
        "n_hidden": cand["n_hidden"],
        "slice_num": cand["slice_num"],
        "loss_force_norm": 1,
        "mean_rel_l2_error": report["mean_rel_l2_error"],
        "checkpoint": ckpt,
    }
    with open(done_marker, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser("Bounded automated search for a B2 accuracy fix")
    parser.add_argument("--material", type=str, default="neo_hookean",
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--target", type=float, default=0.09,
                         help="Stop as soon as a trial's mean_rel_l2_error drops below this")
    parser.add_argument("--max_trials", type=int, default=5,
                         help="Stop after this many full-scale trials even if --target is never reached")

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--Ntheta", type=int, default=21)
    parser.add_argument("--Nr", type=int, default=21)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-3, help="LR for candidates that don't override it")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--validate_every", type=int, default=25)
    parser.add_argument("--early_stop_patience", type=int, default=8)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Point this at a Google Drive path so progress/resume state survives "
                              "a Colab disconnect (e.g. /content/drive/MyDrive/pfem_run/B2_accuracy_search)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "search_summary.json")

    results = []
    success = None
    for i, cand in enumerate(CANDIDATES[:args.max_trials]):
        try:
            result = run_trial(cand, args, args.out_dir)
        except Exception:
            print(f"\n----- {cand['name']}: FAILED -----")
            traceback.print_exc()
            result = {"trial": cand["name"], "status": "FAILED"}
        results.append(result)

        with open(summary_path, "w") as f:
            json.dump({"target": args.target, "max_trials": args.max_trials,
                       "trials_run": len(results), "results": results}, f, indent=2)

        err = result.get("mean_rel_l2_error")
        if err is not None:
            print(f"\n----- {cand['name']}: mean_rel_l2_error={err:.4f} "
                  f"(target < {args.target:.4f}) -----")
            if err < args.target:
                success = result
                print(f"\n{'#'*90}\nTARGET REACHED at trial {i+1}/{len(CANDIDATES[:args.max_trials])}: "
                      f"'{cand['name']}' -> mean_rel_l2_error={err:.4f} < {args.target:.4f}\n"
                      f"Checkpoint: {result['checkpoint']}\n{'#'*90}")
                break

    if success is None:
        print(f"\n{'!'*90}\nTarget {args.target:.4f} NOT reached within {len(results)} trial(s). "
              f"See {summary_path} for every trial's result.\n{'!'*90}")

    print(f"\nFull search summary written to {summary_path}")


if __name__ == "__main__":
    main()
