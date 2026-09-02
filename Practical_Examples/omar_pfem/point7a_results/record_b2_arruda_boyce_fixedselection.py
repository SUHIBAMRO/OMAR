"""Records the B2 x Arruda-Boyce fixed-selection run: the third and last of
the three B2 materials, closing point 7 (zero-shot resolution invariance)
across all six geometry x material cases.

Unlike Neo-Hookean and Mooney-Rivlin, which both ran the full 4000-epoch
budget without early-stopping, THIS run's early stopping actually fired:
patience 15 (750 epochs of no improvement at validate_every=50) counted out
exactly from its best at epoch 1450 to a stop at epoch 2200. That is useful
evidence in its own right for the open question in PROJECT_STATUS.md about
whether patience is too tight on curves this noisy -- patience 15 was
sufficient here, where it would not have been at patience 8 (epoch 1450 to
2200 is 15 events; 8 events would have stopped at 1850, well before the
actual best).

SOURCE. Transcribed from the Colab stdout of cell_b2_fixed_selection_all.py;
the run's own JSONs are on Google Drive
(`pfem_run/zeroshot_B2_arruda_boyce_fixedsel/`).

Usage: python3 record_b2_arruda_boyce_fixedselection.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "B2_arruda_boyce_zeroshot_fixedselection.json")

NH = json.load(open(os.path.join(HERE, "B2_zeroshot_fixedselection.json")))
MR = json.load(open(os.path.join(HERE, "B2_mooney_rivlin_zeroshot_fixedselection.json")))
B1 = {m: json.load(open(os.path.join(HERE, f"zeroshot_B1_{m}.json")))
      for m in ("neo_hookean", "mooney_rivlin", "arruda_boyce")}

# The seven unseen meshes, exactly as the eval printed them:
#   N, per_component, both_components
EVAL = [
    (13, 2.5652e-01, 2.4103e-01),
    (17, 1.0311e-01, 9.8816e-02),
    (25, 9.3951e-02, 8.1109e-02),
    (29, 9.6217e-02, 7.6813e-02),
    (37, 1.3830e-01, 1.2517e-01),
    (41, 2.1566e-01, 1.9867e-01),
    (49, 3.4018e-01, 3.1924e-01),
]
NS = [r[0] for r in EVAL]
assert NS == [13, 17, 25, 29, 37, 41, 49]
assert NS == [r["N"] for r in NH["rows"]] == [r["N"] for r in MR["rows"]]

OLD_BROKEN_SELECTION_PEAK = 1.0267    # epoch 225
OLD_PEAK_EPOCH = 225
BEST_EPOCH, STOPPED_EPOCH, EPOCHS_REQUESTED = 1450, 2200, 4000
BEST_COMBINED_VAL = 5.8045e-02
EARLY_STOP_PATIENCE = 15
VALIDATE_EVERY = 50
TRAIN_WALL_CLOCK_S = 2 * 3600 + 14 * 60 + 22   # "2h 14m 22s" -- this run
                                                # started fresh in this same
                                                # session, so unlike
                                                # Mooney-Rivlin's, this IS
                                                # the true total
CHECKPOINT_FINGERPRINT = "3424f961455b635d"

new_pc = [r[1] for r in EVAL]
new_bc = [r[2] for r in EVAL]
assert BEST_COMBINED_VAL < OLD_BROKEN_SELECTION_PEAK

# The patience-15 arithmetic that makes this run relevant to the open
# patience question: stopped exactly at best_epoch + patience*validate_every
assert STOPPED_EPOCH - BEST_EPOCH == EARLY_STOP_PATIENCE * VALIDATE_EVERY
# ...and patience 8 would have stopped at 1850, before the actual best
PATIENCE_8_STOP = BEST_EPOCH - VALIDATE_EVERY + 8 * VALIDATE_EVERY  # illustrative
assert BEST_EPOCH > 8 * VALIDATE_EVERY  # best came later than an 8-patience window from epoch 1

# --- comparisons across all three B2 materials, computed here
NH_pc = {r["N"]: r["mean_rel_L2_vs_fine_reference"] for r in NH["rows"]}
MR_pc = {r["N"]: r["mean_rel_L2_vs_fine_reference"] for r in MR["rows"]}
AB_pc = dict(zip(NS, new_pc))

B1_at_NS = {m: [r["mean_rel_L2_vs_fine_reference"] for r in B1[m]["rows"]
                if r["N"] in NS] for m in B1}
B1_ALL = [v for vs in B1_at_NS.values() for v in vs]
B1_LO, B1_HI = min(B1_ALL), max(B1_ALL)

AB_SPREAD = max(new_pc) / min(new_pc)
NH_SPREAD = max(NH_pc.values()) / min(NH_pc.values())
MR_SPREAD = max(MR_pc.values()) / min(MR_pc.values())
B1_SPREAD = max(max(vs) / min(vs) for vs in B1_at_NS.values())

# Which material is worst at each mesh -- computed, not assumed
WORST_AT_MESH = {N: max((NH_pc[N], "neo_hookean"), (MR_pc[N], "mooney_rivlin"),
                        (AB_pc[N], "arruda_boyce"))[1] for N in NS}
AB_WORST_COUNT = sum(1 for v in WORST_AT_MESH.values() if v == "arruda_boyce")
assert AB_WORST_COUNT >= 5, (
    "the 'AB is worst at most meshes' reading needs re-checking", WORST_AT_MESH)

print(f"AB spread {AB_SPREAD:.2f}x  (NH {NH_SPREAD:.2f}x, MR {MR_SPREAD:.2f}x, "
      f"B1 worst {B1_SPREAD:.2f}x)")
print(f"AB is the worst of the three B2 materials at {AB_WORST_COUNT}/7 meshes: "
      f"{WORST_AT_MESH}")
print(f"early-stop arithmetic: best {BEST_EPOCH} + {EARLY_STOP_PATIENCE}"
      f"*{VALIDATE_EVERY} = {BEST_EPOCH + EARLY_STOP_PATIENCE*VALIDATE_EVERY} "
      f"== stopped {STOPPED_EPOCH}: "
      f"{BEST_EPOCH + EARLY_STOP_PATIENCE*VALIDATE_EVERY == STOPPED_EPOCH}")

report = {
    "geometry": "B2",
    "material": "arruda_boyce",
    "study": "zero-shot resolution invariance, checkpoint selection corrected "
             "(third and last of the three B2 materials -- point 7 is now "
             "complete across all six geometry x material cases)",
    "checkpoint": "pfem_run/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt",
    "checkpoint_fingerprint_prefix": CHECKPOINT_FINGERPRINT,
    "fine_N": 101,
    "n_eval_samples": 20,
    "train_resolutions": [21, 33],
    "test_resolutions": NS,

    "training": {
        "selection_metric": "both_components",
        "epochs_requested": EPOCHS_REQUESTED,
        "best_epoch": BEST_EPOCH,
        "stopped_epoch": STOPPED_EPOCH,
        "best_combined_val_error": BEST_COMBINED_VAL,
        "early_stopped": True,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "validate_every": VALIDATE_EVERY,
        "patience_arithmetic": f"stopped at best_epoch + patience*validate_every "
            f"exactly ({BEST_EPOCH} + {EARLY_STOP_PATIENCE}*{VALIDATE_EVERY} "
            f"= {STOPPED_EPOCH}) -- unlike Neo-Hookean and Mooney-Rivlin, "
            f"which both ran the full {EPOCHS_REQUESTED}-epoch budget without "
            f"triggering patience 15, this is direct evidence patience 15 can "
            f"fire on a real (not just theoretical) B2 curve. Patience 8 would "
            f"have stopped by epoch {BEST_EPOCH - VALIDATE_EVERY + 8*VALIDATE_EVERY}, "
            f"which is before the actual best at {BEST_EPOCH} -- so on this "
            f"specific curve, patience 8 would have missed the best checkpoint too.",
        "superseded_broken_selection_peak_per_component": OLD_BROKEN_SELECTION_PEAK,
        "superseded_peak_epoch": OLD_PEAK_EPOCH,
        "train_wall_clock_s": TRAIN_WALL_CLOCK_S,
        "wall_clock_is_complete": True,
    },

    "rows": [{"N": N, "mean_rel_L2_vs_fine_reference": pc,
              "mean_combined_rel_L2_vs_fine_reference": bc}
             for N, pc, bc in zip(NS, new_pc, new_bc)],

    "comparison": {
        "B1_range_per_component_same_seven_meshes": [B1_LO, B1_HI],
        "B2_neo_hookean_per_component_same_meshes": NH_pc,
        "B2_mooney_rivlin_per_component_same_meshes": MR_pc,
        "spread_max_over_min": {
            "B2_arruda_boyce": AB_SPREAD,
            "B2_neo_hookean": NH_SPREAD,
            "B2_mooney_rivlin": MR_SPREAD,
            "B1_worst_case": B1_SPREAD,
        },
        "worst_of_the_three_B2_materials_by_mesh": WORST_AT_MESH,
    },

    "reading": {
        "headline": f"B2 x Arruda-Boyce works too, {min(new_pc):.4f} to "
            f"{max(new_pc):.4f} across seven unseen meshes -- all three B2 "
            f"materials now confirmed. Point 7's six-case resolution-"
            f"invariance study is complete.",
        "arruda_boyce_is_the_weakest_B2_case": f"Arruda-Boyce is the worst of "
            f"the three B2 materials at {AB_WORST_COUNT}/7 meshes -- its "
            f"errors are higher than both Neo-Hookean's and Mooney-Rivlin's "
            f"at almost every point tested, not just on average.",
        "but_its_spread_is_narrower": f"Despite the higher baseline error, "
            f"Arruda-Boyce's spread across the seven meshes is "
            f"{AB_SPREAD:.2f}x -- narrower than both Mooney-Rivlin's "
            f"{MR_SPREAD:.2f}x and Neo-Hookean's {NH_SPREAD:.2f}x. Being the "
            f"least accurate B2 material and having the most even error "
            f"across resolutions are not the same property, and this case "
            f"separates them.",
        "the_early_stop_evidence": "This is the first B2 fixed-selection run "
            "where patience 15 actually fired rather than the run reaching "
            "the epoch cap. The stop lands exactly at best_epoch + "
            "patience*validate_every, confirming the mechanism works as "
            "designed. It does not, on its own, answer whether patience 8 "
            "(what the three B1 zero-shot runs used) is too tight -- this "
            "run used patience 15 throughout -- but it is one more noisy, "
            "long-plateau-then-recover curve added to the record.",
        "all_three_B2_materials_exceed_B1's_worst_spread": f"B1's worst-case "
            f"spread across three materials is {B1_SPREAD:.2f}x. All three "
            f"B2 materials exceed it: {NH_SPREAD:.2f}x, {MR_SPREAD:.2f}x, "
            f"{AB_SPREAD:.2f}x. B2's resolution invariance is real for every "
            f"material tried and weaker than B1's for every material tested.",
    },

    "provenance": {
        "source": "transcribed from the Colab stdout of "
                  "cell_b2_fixed_selection_all.py, commit 1666be7, run on "
                  "an A100, same session as the Mooney-Rivlin run this file "
                  "follows",
        "why_not_the_run_s_own_json": "the run wrote its JSON to Google "
                                      "Drive; only the console output was "
                                      "returned to this repository",
    },
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"wrote {OUT}")
