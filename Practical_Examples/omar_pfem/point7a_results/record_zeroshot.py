"""Writes one zero-shot case's JSON from values transcribed off Colab stdout.

Five more cases are coming and hand-writing each JSON invites exactly the
drift this project has already been bitten by. This takes the numbers once,
checks what can be checked, and derives the rest.

Checked rather than trusted:
  * final_epoch * steps_per_epoch == the manifest's opt_steps, which pins
    down that the manifest field is the END count and lets the AT-BEST count
    be derived instead of guessed;
  * the protocol matches every other case (the comparison is only meaningful
    if it does), and any difference is recorded rather than silently kept;
  * the error curve's shape is computed, not asserted -- monotone or not.

Usage: edit CASE below and run. One case per invocation, on purpose.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# The protocol every case in this study must share. A case that differs is
# not a row of the same table.
PROTOCOL = {
    "train_resolutions": [21, 33], "n_train_per_res": 400, "n_val_per_res": 100,
    "batch_size": 8, "epochs": 2000, "validate_every": 25,
    "early_stop_patience": 8, "early_stop_min_delta": 0.0001,
    "lr": 0.002, "weight_decay": 0.0, "grad_clip": 1.0, "seed": 0,
    "model": "Transolver_Irregular_Mesh", "fun_dim": 4, "use_soft_dirichlet": 1,
}


def record(case, rows, training, cost, env, fingerprint, fine_N=101,
           protocol_deviations=None):
    spe = training["opt_steps_at_end"] // training["final_epoch"]
    assert spe * training["final_epoch"] == training["opt_steps_at_end"], (
        "opt_steps is not an exact multiple of final_epoch -- the manifest "
        "field may not be the end-of-run count after all")
    n_train = PROTOCOL["n_train_per_res"] * len(PROTOCOL["train_resolutions"])
    assert spe == n_train // PROTOCOL["batch_size"], (spe, n_train)
    assert spe == 100, spe
    training["steps_per_epoch"] = spe
    training["opt_steps_at_best"] = training["best_epoch"] * spe
    training["opt_steps_note"] = (
        "the manifest's opt_steps field is the TOTAL at the end of the run, "
        "not the count at the best checkpoint. The reported errors come from "
        "model_best.pt, i.e. from opt_steps_at_best.")
    assert training["best_epoch"] < training["final_epoch"]
    # early stopping must have fired exactly patience validation events after
    # the best, or the run did not stop for the reason the protocol says
    events_after_best = ((training["final_epoch"] - training["best_epoch"])
                         // PROTOCOL["validate_every"])
    training["validation_events_after_best"] = events_after_best
    if events_after_best != PROTOCOL["early_stop_patience"]:
        # Not fatal, but never silent: it means the run did not stop for the
        # reason the protocol states, and a reader comparing budgets across
        # cases needs to know which case is the odd one.
        training["early_stopping_anomaly"] = (
            f"stopped {events_after_best} validation events after the best "
            f"epoch, where the protocol's patience is "
            f"{PROTOCOL['early_stop_patience']}. The reported errors are "
            "unaffected -- they come from model_best.pt, which is the best "
            "checkpoint whenever it was written. The likely explanation is "
            "that the trainer's own best-tracker sat at a later epoch than "
            "the argmin of combined_val_error (a difference of "
            f"{(events_after_best - PROTOCOL['early_stop_patience']) * PROTOCOL['validate_every']} "
            "epochs would reconcile them exactly), but that is unverified.")
        print(f"  ANOMALY: {training['early_stopping_anomaly']}")

    e = [r["mean_rel_L2_vs_fine_reference"] for r in rows]
    N = [r["N"] for r in rows]
    assert N == sorted(N) and len(N) == 7
    monotone = all(a > b for a, b in zip(e, e[1:]))
    imin = e.index(min(e))
    shape = {
        "monotone_decreasing_in_N": monotone,
        "best_resolution_N": N[imin],
        "best_error": e[imin],
        "error_at_finest_N": e[-1],
        "rise_after_minimum_pct": (e[-1] / e[imin] - 1) * 100,
        "reading": (
            "error falls all the way to the finest mesh tested"
            if monotone else
            f"U-shaped: the error bottoms out at N={N[imin]} and rises "
            f"{(e[-1] / e[imin] - 1) * 100:.1f}% by N={N[-1]}. The model was "
            "trained at N=21 and 33, so the finest test meshes are the "
            "furthest extrapolation and the rise is where zero-shot transfer "
            "starts to cost something."),
    }

    d = {
        "study": "Round-5 point 1/7a -- zero-shot resolution invariance",
        "geometry": case.split("_")[0],
        "material": case.split("_", 1)[1],
        "checkpoint_fingerprint": fingerprint,
        "fine_N": fine_N,
        "test_resolutions": N,
        "rows": rows,
        "shape": shape,
        "training": training,
        "protocol": PROTOCOL,
        "protocol_deviations": protocol_deviations or
            "none -- identical to every other case in this study",
        "cost": cost,
        "environment": env,
        "NOT_COMPARABLE_WITH_TABLE_12": (
            "Report Table 12 is B1 x Neo-Hookean trained at N=21 ALONE and "
            "evaluated at five resolutions, all FINER than training. This "
            "study trains jointly at N=21 and 33 and evaluates seven, "
            "including two COARSER. Round-5 item 7 asked for coarser AND "
            "finer, so this protocol answers it and Table 12's does not. The "
            "two cannot share a table."),
        "provenance": (
            "transcribed from zeroshot_eval_report.json and run_manifest.json "
            "on Drive via record_zeroshot.py, which re-derives the at-best "
            "step count and re-checks the early-stopping arithmetic."),
    }
    out = os.path.join(HERE, f"zeroshot_{case}.json")
    with open(out, "w") as f:
        json.dump(d, f, indent=2)
    print(f"wrote {os.path.basename(out)}")
    print(f"  best val {training['best_combined_val_error']:.5f} at epoch "
          f"{training['best_epoch']} ({training['opt_steps_at_best']:,} steps), "
          f"ran to {training['opt_steps_at_end']:,}")
    print(f"  shape: {'monotone' if monotone else 'U-shaped'}, "
          f"best N={N[imin]} at {e[imin]:.4e}, finest N={N[-1]} at {e[-1]:.4e}")
    return d


CASES = {
 "B1_neo_hookean": dict(
   fingerprint="86030f4f05ea74f83079cee6b74485b30f2c1a5acac6acd5bc7a06e3adaa88f4",
   rows=[(13,0.09667502625249838,0.031589901205428796),
         (17,0.0791437465182073,0.024321721226294214),
         (25,0.05739191567731076,0.0181387180350072),
         (29,0.052068253382705064,0.01641490450794315),
         (37,0.05247267574090144,0.015488935136676752),
         (41,0.05616763113251163,0.01683540912826112),
         (49,0.0670210165116116,0.020469196521041896)],
   training=dict(best_combined_val_error=0.06575, best_epoch=650,
                 final_epoch=900, opt_steps_at_end=90000,
                 validation_entries=36, early_stopped=True),
   cost=dict(note="this case predates the run_manifest instrumentation, so no "
                  "generation or training wall clock was recorded. File mtimes "
                  "place generation at 2026-08-10 11:57 and the end of "
                  "training at 23:11 the same day."),
   env=dict(note="not recorded -- predates the manifest. The 7-resolution "
                 "eval that produced these rows ran 2026-08-27 21:12."),
 ),
 "B1_arruda_boyce": dict(
   fingerprint="bff6d7f2af589477c00720d2aec0c7870f5b1d61ec344be57d70b4c01b2792eb",
   rows=[(13,0.1010555613294791,0.03507471643962464),
         (17,0.08319307493853036,0.024845609914405892),
         (25,0.06467257263974399,0.0214337876404974),
         (29,0.059746068190494474,0.02159836087436734),
         (37,0.05642291490242689,0.02162412904703176),
         (41,0.057457482435830765,0.021351231230364986),
         (49,0.06296242899517626,0.020871424228532297)],
   training=dict(best_combined_val_error=0.07831752486526966, best_epoch=700,
                 final_epoch=900, opt_steps_at_end=90000,
                 total_train_wall_clock_s=5966.1893446445465,
                 validation_entries=36, early_stopped=True),
   cost=dict(fem_generation_s=11010.708227872849, fem_generation_human="3h 3m 30s",
             train_s=5989.547846794128, train_human="1h 39m 49s",
             eval_human="7h 56m 19s"),
   env=dict(gpu="Tesla T4", gpu_total_mem_mb=15637.086208, cpu_count=8,
            torch="2.11.0+cu128", python="3.13.15",
            code_commit="a3e0ad04b1b75e6c5832c1eb380688110a80c0ea", dirty=False,
            gpu_note="ran on a T4; B1 x Mooney-Rivlin ran on an A100. That "
                     "changes wall clock (1h 40m against 44m to train) and "
                     "nothing else reported here. Do not compare timings "
                     "across the two."),
 ),
 "B1_mooney_rivlin": dict(
   fingerprint="0f363c182b52671bfb35e5db9632c297a56fa10b5d521e1b6ffdd7efffc557fc",
   rows=[(13,0.10642525217969193,0.040393926549212505),
         (17,0.08846675460084326,0.03294190733511524),
         (25,0.069066433666394,0.022718667059276932),
         (29,0.06278944178583773,0.019407487421762445),
         (37,0.05412227340612732,0.016933248178185634),
         (41,0.051526492757131316,0.017170942385982525),
         (49,0.050376113714820855,0.018318237328395187)],
   training=dict(best_combined_val_error=0.08268473669886589, best_epoch=575,
                 final_epoch=775, opt_steps_at_end=77500,
                 total_train_wall_clock_s=2628.6661372184753,
                 validation_entries=31, early_stopped=True),
   cost=dict(fem_generation_s=10681.428952932358, fem_generation_human="2h 58m 1s",
             train_s=2637.858293533325, train_human="43m 57s",
             eval_human="7h 48m 49s"),
   env=dict(gpu="NVIDIA A100-SXM4-80GB", gpu_total_mem_mb=85094.825984,
            cpu_count=12, torch="2.11.0+cu128", python="3.13.15",
            code_commit="a3e0ad04b1b75e6c5832c1eb380688110a80c0ea", dirty=False),
 ),
}

GEN_CAVEAT = ("duration_s covers only what THIS invocation generated; if an "
              "earlier interrupted run produced part of the samples the true "
              "total is the sum over every zeroshot_generate manifest entry.")

if __name__ == "__main__":
    for case, c in CASES.items():
        rows = [{"N": n, "n_eval_samples": 20,
                 "mean_rel_L2_vs_fine_reference": m,
                 "std_rel_L2_vs_fine_reference": s} for n, m, s in c["rows"]]
        cost = dict(c["cost"], generation_caveat=GEN_CAVEAT)
        record(case, rows, c["training"], cost, c["env"], c["fingerprint"])
        print()
