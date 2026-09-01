"""Records the B2 x Neo-Hookean rerun whose checkpoint selection was fixed.

WHY THIS FILE EXISTS. Every B2 number this project has published came from a
run that stopped at its first or second validation event, because early
stopping and `model_best.pt` selection used

    per_component = 0.5 * ( rms(e_u)/rms(u) + rms(e_v)/rms(v) )

which divides each component by its OWN size. On B2 that quantity RISES while
the model improves, so patience ran out immediately. Re-running the identical
configuration with `--selection_metric both_components` -- the metric
rms(e)/rms(uv_exact), which does not invert -- took the same case from 0.9986
to 0.0330 on the very metric that had condemned it.

SOURCE. The run wrote its own JSONs to Google Drive
(`pfem_run/zeroshot_B2_neo_hookean_fixedsel/`); only the Colab stdout came
back into the repo. Every number below is transcribed from that stdout at the
precision it was printed, and `provenance` says so. Nothing here is inferred,
averaged, or rounded by this script -- the two eval columns are copied
verbatim from the eval's own table, and the old columns are copied from the
same table's before/after block, which the cell printed by reading the
superseded run's JSON on Drive.

Usage: python3 record_b2_fixedselection.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "B2_zeroshot_fixedselection.json")

# The seven unseen meshes, each row exactly as the eval printed it:
#   N, old per_component, new per_component, old both_components, new both
EVAL = [
    (13, 8.7137e-01, 2.3142e-01, 7.2109e-01, 2.0462e-01, 5.0334e-02),
    (17, 8.7176e-01, 1.1422e-01, 7.2133e-01, 1.0914e-01, 2.8221e-02),
    (25, 8.7222e-01, 7.2015e-02, 7.2143e-01, 5.9540e-02, 3.7909e-02),
    (29, 8.7235e-01, 7.1113e-02, 7.2144e-01, 6.5495e-02, 3.0029e-02),
    (37, 8.7254e-01, 8.0817e-02, 7.2142e-01, 7.3770e-02, 1.5242e-02),
    (41, 8.7259e-01, 1.5113e-01, 7.2141e-01, 1.4319e-01, 1.8423e-02),
    (49, 8.7270e-01, 2.6912e-01, 7.2138e-01, 2.5963e-01, 3.2771e-02),
]

# Validation events the run printed near its best, verbatim. Kept because the
# SHAPE of this curve is itself a finding: the run reached 0.0369
# both_components at epoch 950, rose and fell repeatedly for 1,800 epochs, and
# only then reached its best. A patience of 8 events would have stopped it in
# one of those dips -- which is why the three B1 runs, all patience 8, are
# recorded as "not known to be wrong, not known to be converged".
LATE_CURVE = [
    (2700, {21: 0.08795422315597534, 33: 0.06182359531521797}, 7.4889e-02, 5.1398e-02),
    (2750, {21: 0.033085793256759644, 33: 0.03295707330107689}, 3.3021e-02, 2.1370e-02),
    (2800, {21: 0.09390207380056381, 33: 0.06394565850496292}, 7.8924e-02, 4.7616e-02),
    (2850, {21: 0.09261499345302582, 33: 0.08719110488891602}, 8.9903e-02, 5.9085e-02),
    (2900, {21: 0.05957557633519173, 33: 0.048785608261823654}, 5.4181e-02, 3.4731e-02),
    (2950, {21: 0.04854102060198784, 33: 0.0545775480568409}, 5.1559e-02, 3.4248e-02),
    (3000, {21: 0.03824955224990845, 33: 0.03959283232688904}, 3.8921e-02, 2.4701e-02),
    (3050, {21: 0.052507780492305756, 33: 0.06338551640510559}, 5.7947e-02, 3.5514e-02),
    (3100, {21: 0.13836173713207245, 33: 0.14821457862854004}, 1.4329e-01, 8.2517e-02),
    (3150, {21: 0.0678519457578659, 33: 0.06598378717899323}, 6.6918e-02, 3.9123e-02),
    (3200, {21: 0.08327015489339828, 33: 0.10811640322208405}, 9.5693e-02, 6.2912e-02),
    (3250, {21: 0.07963953912258148, 33: 0.09635469317436218}, 8.7997e-02, 6.1150e-02),
    (3300, {21: 0.03940986841917038, 33: 0.0485902763903141}, 4.4000e-02, 2.5035e-02),
    (3350, {21: 0.05115897208452225, 33: 0.04500778391957283}, 4.8083e-02, 3.2137e-02),
    (3400, {21: 0.04710771515965462, 33: 0.051361244171857834}, 4.9234e-02, 3.0654e-02),
    (3450, {21: 0.056995704770088196, 33: 0.0597023107111454}, 5.8349e-02, 4.1949e-02),
    (3500, {21: 0.05001213774085045, 33: 0.04654787480831146}, 4.8280e-02, 3.2074e-02),
]

BEST_EPOCH, FINAL_EPOCH = 2750, 3500
BEST_BOTH, BEST_PER = 2.1370e-02, 3.3021e-02
OLD_BEST_PER = 0.9986          # the superseded run, same case, same protocol
OPT_STEPS_AT_END = 350_000
WALL_CLOCK_S = 3 * 3600 + 18 * 60 + 8   # "3h 18m 8s" from the run manifest

# --- checks, so a typo in the block above does not become a published number
assert len(EVAL) == 7
assert [r[0] for r in EVAL] == [13, 17, 25, 29, 37, 41, 49]
for N, old_pc, new_pc, old_bc, new_bc, _sd in EVAL:
    assert new_pc < old_pc and new_bc < old_bc, (
        f"N={N}: the rerun is not better on both metrics, which is the whole "
        f"claim of this file")
best = [e for e in LATE_CURVE if e[0] == BEST_EPOCH]
assert len(best) == 1 and best[0][3] == BEST_BOTH and best[0][2] == BEST_PER
assert min(e[3] for e in LATE_CURVE) == BEST_BOTH, (
    "epoch 2750 is not the minimum of the recorded window, so it cannot be "
    "described as the selected checkpoint")
assert OPT_STEPS_AT_END % FINAL_EPOCH == 0
STEPS_PER_EPOCH = OPT_STEPS_AT_END // FINAL_EPOCH
assert STEPS_PER_EPOCH == 100, STEPS_PER_EPOCH

new_pc = [r[2] for r in EVAL]
new_bc = [r[4] for r in EVAL]
old_pc = [r[1] for r in EVAL]

report = {
    "geometry": "B2",
    "material": "neo_hookean",
    "study": "zero-shot resolution invariance, checkpoint selection corrected",
    "checkpoint": "pfem_run/zeroshot_B2_neo_hookean_fixedsel/model_best.pt",
    "checkpoint_fingerprint_prefix": "1afbcd250a2933e2",
    "fine_N": 101,
    "n_eval_samples": 20,
    "train_resolutions": [21, 33],
    "test_resolutions": [r[0] for r in EVAL],

    "the_defect": {
        "what": "early stopping and model_best.pt selection used "
                "per_component = 0.5*(rms(e_u)/rms(u) + rms(e_v)/rms(v)), "
                "which divides each component by its own size",
        "why_it_inverts_on_B2": "the per-sample ratio rms(v)/rms(u) averages "
                                "1.90 while the ratio of the AVERAGED "
                                "components is 0.90 -- a skewed distribution, "
                                "so the mean of the per-sample ratios reports "
                                "its tail. The quantity rises while the model "
                                "improves.",
        "consequence": "every B2 run in this project stopped at its first or "
                       "second validation event (epochs 25, 25 and 225 for "
                       "neo_hookean, mooney_rivlin and arruda_boyce), so every "
                       "downstream B2 diagnosis was measuring a model trained "
                       "for 25-50 epochs",
        "fix": "--selection_metric both_components, i.e. "
               "rms(e)/rms(uv_exact), which does not invert; it is now the "
               "default in resolution_invariance_zeroshot.py",
        "B1_is_unaffected": "B1's two metrics differ by a stable 1.36-1.71x "
                            "offset and all three B1 cases agree, in both "
                            "metrics, that model_best.pt is the better "
                            "checkpoint. Every published B1 number stands.",
    },

    "training": {
        "epochs_requested": 4000,
        "final_epoch": FINAL_EPOCH,
        "best_epoch": BEST_EPOCH,
        "early_stopped": True,
        "early_stop_patience": 15,
        "validate_every": 50,
        "selection_metric": "both_components",
        "best_both_components_val_error": BEST_BOTH,
        "per_component_val_error_at_that_checkpoint": BEST_PER,
        "superseded_run_best_per_component_val_error": OLD_BEST_PER,
        "opt_steps_at_end": OPT_STEPS_AT_END,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "opt_steps_at_best": BEST_EPOCH * STEPS_PER_EPOCH,
        "wall_clock_s": WALL_CLOCK_S,
        "late_validation_curve": [
            {"epoch": e, "per_resolution_per_component": pr,
             "per_component": pc, "both_components": bc}
            for e, pr, pc, bc in LATE_CURVE],
    },

    "rows": [
        {"N": N,
         "mean_rel_L2_vs_fine_reference": new_pc_,
         "std_rel_L2_vs_fine_reference": sd,
         "mean_combined_rel_L2_vs_fine_reference": new_bc_,
         "superseded_mean_rel_L2_vs_fine_reference": old_pc_,
         "superseded_mean_combined_rel_L2_vs_fine_reference": old_bc_}
        for N, old_pc_, new_pc_, old_bc_, new_bc_, sd in EVAL],

    "reading": {
        "headline": "B2 x Neo-Hookean does not fail. Under a selection metric "
                    "that orders its checkpoints correctly it goes from 0.9986 "
                    "to 0.0330 validation per_component, and from a flat "
                    "0.8714-0.8727 zero-shot band to 0.0711-0.2691 over the "
                    "same seven unseen meshes.",
        "against_B1": "B1 spans 0.050-0.106 per_component on these same seven "
                      "meshes. The rerun is comparable to B1 in the middle of "
                      "the range -- 0.0711 to 0.0808 at N=25, 29 and 37 "
                      "against B1's 0.052-0.067 -- and degrades at both ends, "
                      "0.2314 at N=13 and 0.2691 at N=49.",
        "spread": {
            "fixed_max_over_min": round(max(new_pc) / min(new_pc), 3),
            "note": "3.784x across the seven meshes, against B1's 2.1x. The "
                    "old run's 0.153% spread was not resolution invariance; a "
                    "model whose output barely responds to its input is flat "
                    "everywhere. Insensitivity and invariance look identical "
                    "in that column and are not the same property.",
        },
        "what_this_does_NOT_establish": "the other two B2 materials. "
                                        "mooney_rivlin (old best 0.9752) and "
                                        "arruda_boyce (old best 1.0267) carry "
                                        "the identical defect and have not "
                                        "been rerun. No B2 row may be quoted "
                                        "for them.",
        "why_the_ends_degrade_is_open": "not measured. N=13 and N=17 are "
                                        "coarser than either training mesh and "
                                        "N=41, 49 are far finer; both ends are "
                                        "extrapolation in mesh size. Which of "
                                        "the two effects dominates was not "
                                        "tested and is not asserted here.",
    },

    "cost": "A100. Training 3 h 18 m 8 s to epoch 3500 (350,000 optimiser "
            "steps); the eval reused 20 cached N=101 solves and took 10 s. The "
            "sample cache and the fine-reference cache were COPIED from the "
            "superseded run's directory, so neither the 7+ hours of FEM behind "
            "the samples nor the N=101 references were regenerated, and "
            "nothing was written to that directory.",

    "provenance": "transcribed from the Colab stdout of "
                  "Round6_B2_FixedSelection (cell_b2_fixed_selection.py) on "
                  "2026-09-01. The run's own JSONs are on Drive at "
                  "pfem_run/zeroshot_B2_neo_hookean_fixedsel/. Every value "
                  "carries the PRINTED precision (4-5 significant digits) "
                  "except the late validation curve, which the trainer prints "
                  "at full float precision. The superseded columns were "
                  "printed by the same cell from "
                  "pfem_run/zeroshot_B2_neo_hookean/zeroshot_eval.json.",
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"wrote {OUT}")
print(f"  val per_component  {OLD_BEST_PER} -> {BEST_PER}")
print(f"  zero-shot per_component  "
      f"{min(old_pc):.4f}-{max(old_pc):.4f} -> {min(new_pc):.4f}-{max(new_pc):.4f}")
print(f"  zero-shot both_components  {min(new_bc):.4f}-{max(new_bc):.4f}")
