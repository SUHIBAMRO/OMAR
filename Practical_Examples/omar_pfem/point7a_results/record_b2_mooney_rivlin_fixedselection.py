"""Records the B2 x Mooney-Rivlin fixed-selection run: the second of the two
B2 materials the Neo-Hookean rerun (record_b2_fixedselection.py) was gating.

Unlike Neo-Hookean, there is no clean "before" zero-shot table at these same
seven meshes to compare against: the earlier B2 x Mooney-Rivlin attempts were
invalidated by the mesh-dependent load bug (INVALID_B2_zeroshot.json) and
then, after that repair, stopped at epoch 25 under the broken selection
metric (0.9752, per B2_zeroshot_retrain_status.json) -- a checkpoint from a
few dozen steps, not a comparable trained model. So this file records the
one number that IS comparable across attempts (the broken-selection peak)
and otherwise reports this run on its own terms, against B1's range and
against B2 x Neo-Hookean's already-recorded fixed-selection result.

SOURCE. The run wrote its own JSONs to Google Drive
(`pfem_run/zeroshot_B2_mooney_rivlin_fixedsel/`); only the Colab stdout came
back into the repo. Every number below is transcribed from that stdout at
the precision it was printed.

Usage: python3 record_b2_mooney_rivlin_fixedselection.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "B2_mooney_rivlin_zeroshot_fixedselection.json")

NH = json.load(open(os.path.join(HERE, "B2_zeroshot_fixedselection.json")))
B1 = {m: json.load(open(os.path.join(HERE, f"zeroshot_B1_{m}.json")))
      for m in ("neo_hookean", "mooney_rivlin", "arruda_boyce")}

# The seven unseen meshes, exactly as the eval printed them:
#   N, per_component, both_components
EVAL = [
    (13, 1.9462e-01, 1.9607e-01),
    (17, 7.9047e-02, 7.7937e-02),
    (25, 6.6098e-02, 5.6481e-02),
    (29, 7.3031e-02, 6.4001e-02),
    (37, 1.2197e-01, 1.0735e-01),
    (41, 1.9973e-01, 1.8152e-01),
    (49, 3.2457e-01, 3.0283e-01),
]
NS = [r[0] for r in EVAL]
assert NS == [13, 17, 25, 29, 37, 41, 49]
assert NS == [r["N"] for r in NH["rows"]], "must be the same seven meshes as NH"

OLD_BROKEN_SELECTION_PEAK = 0.9752   # epoch 25, per_component -- the last
                                     # attempt before this rerun, and the
                                     # only earlier number that is comparable
BEST_EPOCH, FINAL_EPOCH = 3350, 4000
BEST_COMBINED_VAL = 3.4406e-02
TRAIN_WALL_CLOCK_S_THIS_SEGMENT = 2 * 3600 + 24 * 60 + 23   # "2h 24m 23s",
                                     # this segment only -- the run resumed
                                     # from epoch 1651 after a container
                                     # restart, so the true total wall clock
                                     # includes an earlier, unrecorded
                                     # segment and is not claimed here
CHECKPOINT_FINGERPRINT = "4d76646d14674ad0"

new_pc = [r[1] for r in EVAL]
new_bc = [r[2] for r in EVAL]
assert BEST_COMBINED_VAL < OLD_BROKEN_SELECTION_PEAK

# --- comparisons, computed here rather than asserted from memory
NH_pc = {r["N"]: r["mean_rel_L2_vs_fine_reference"] for r in NH["rows"]}
B1_at_NS = {m: [r["mean_rel_L2_vs_fine_reference"] for r in B1[m]["rows"]
                if r["N"] in NS] for m in B1}
for m in B1:
    assert [r["N"] for r in B1[m]["rows"] if r["N"] in NS] == NS
B1_ALL = [v for vs in B1_at_NS.values() for v in vs]
B1_LO, B1_HI = min(B1_ALL), max(B1_ALL)

MR_SPREAD = max(new_pc) / min(new_pc)
NH_SPREAD = max(NH_pc.values()) / min(NH_pc.values())
B1_SPREAD = max(max(vs) / min(vs) for vs in B1_at_NS.values())
assert MR_SPREAD > NH_SPREAD > B1_SPREAD, (
    "the ordering this file's reading depends on", MR_SPREAD, NH_SPREAD, B1_SPREAD)

# Where MR beats NH and where it does not, mesh by mesh
MR_VS_NH = {N: (pc, NH_pc[N], "MR better" if pc < NH_pc[N] else "NH better")
            for N, pc in zip(NS, new_pc)}
COARSE = [13, 17, 25]   # the three meshes nearest the low end of the sweep
FINE = [37, 41, 49]     # the three meshes nearest the high end
mr_coarse_better = sum(1 for N in COARSE if MR_VS_NH[N][2] == "MR better")
mr_fine_better = sum(1 for N in FINE if MR_VS_NH[N][2] == "MR better")
assert mr_coarse_better == 3 and mr_fine_better == 0, (
    "the coarse/fine split this file reports does not hold", MR_VS_NH)

print(f"MR spread {MR_SPREAD:.2f}x vs NH {NH_SPREAD:.2f}x vs B1 worst {B1_SPREAD:.2f}x")
print(f"MR beats NH at all {len(COARSE)} coarse meshes {COARSE}, "
      f"loses at all {len(FINE)} fine meshes {FINE}")

report = {
    "geometry": "B2",
    "material": "mooney_rivlin",
    "study": "zero-shot resolution invariance, checkpoint selection corrected "
             "(second of the two B2 materials gated behind the Neo-Hookean "
             "rerun)",
    "checkpoint": "pfem_run/zeroshot_B2_mooney_rivlin_fixedsel/model_best.pt",
    "checkpoint_fingerprint_prefix": CHECKPOINT_FINGERPRINT,
    "fine_N": 101,
    "n_eval_samples": 20,
    "train_resolutions": [21, 33],
    "test_resolutions": NS,

    "training": {
        "selection_metric": "both_components",
        "epochs_requested": FINAL_EPOCH,
        "best_epoch": BEST_EPOCH,
        "final_epoch": FINAL_EPOCH,
        "best_combined_val_error": BEST_COMBINED_VAL,
        "early_stopped": False,
        "note_on_early_stopped": "ran the full 4000-epoch budget without "
            "triggering patience 15 -- unlike Neo-Hookean (stopped at 3500) "
            "the best checkpoint (epoch 3350) was close to the end, so "
            "whether a longer budget would have found a better one further "
            "out is not known, same open question raised for B1 in "
            "PROJECT_STATUS.md",
        "superseded_broken_selection_peak_per_component": OLD_BROKEN_SELECTION_PEAK,
        "superseded_peak_epoch": 25,
        "train_wall_clock_s_this_segment_only": TRAIN_WALL_CLOCK_S_THIS_SEGMENT,
        "wall_clock_caveat": "this run resumed from epoch 1651/4000 after a "
            "container restart lost track of an earlier segment; the "
            "reported wall clock covers only epoch 1651 onward, not the "
            "true total training time",
    },

    "rows": [{"N": N, "mean_rel_L2_vs_fine_reference": pc,
              "mean_combined_rel_L2_vs_fine_reference": bc}
             for N, pc, bc in zip(NS, new_pc, new_bc)],

    "comparison": {
        "B1_range_per_component_same_seven_meshes": [B1_LO, B1_HI],
        "B2_neo_hookean_fixedsel_per_component_same_meshes": NH_pc,
        "spread_max_over_min": {
            "B2_mooney_rivlin": MR_SPREAD,
            "B2_neo_hookean": NH_SPREAD,
            "B1_worst_case": B1_SPREAD,
        },
        "mesh_by_mesh_vs_neo_hookean": MR_VS_NH,
    },

    "reading": {
        "headline": f"B2 x Mooney-Rivlin works, {min(new_pc):.4f} to "
            f"{max(new_pc):.4f} across seven unseen meshes -- comparable to "
            f"Neo-Hookean's {min(NH_pc.values()):.4f} to "
            f"{max(NH_pc.values()):.4f} and to B1's range "
            f"{B1_LO:.4f}-{B1_HI:.4f}, confirming the fixed-selection recipe "
            f"is not specific to Neo-Hookean.",
        "the_crossover": f"Mooney-Rivlin is BETTER than Neo-Hookean at all "
            f"three coarse meshes ({COARSE}) and WORSE at all three fine "
            f"meshes ({FINE}) -- the two materials' errors cross over "
            f"somewhere between N=29 and N=37. Training was at N=21 and 33 "
            f"for both materials, so this is not explained by one material "
            f"training closer to the fine end.",
        "the_spread_is_wider_than_neo_hookean's": f"Mooney-Rivlin's spread "
            f"across the seven meshes is {MR_SPREAD:.2f}x, wider than "
            f"Neo-Hookean's {NH_SPREAD:.2f}x, both far wider than B1's worst "
            f"case {B1_SPREAD:.2f}x. B2's resolution invariance is real for "
            f"both materials and weaker than B1's for both, and weakest yet "
            f"for Mooney-Rivlin.",
        "what_this_does_not_establish": "Arruda-Boyce, the third B2 case, "
            "has not been evaluated yet (training just started, epoch 200 "
            "of 4000 as this file is written). Whether the coarse/fine "
            "crossover and the widening spread are a B2 pattern or "
            "Mooney-Rivlin-specific is not known until it finishes.",
    },

    "provenance": {
        "source": "transcribed from the Colab stdout of "
                  "cell_b2_fixed_selection_all.py, commit 1666be7, run on "
                  "an A100",
        "why_not_the_run_s_own_json": "the run wrote its JSON to Google "
                                      "Drive; only the console output was "
                                      "returned to this repository",
    },
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"wrote {OUT}")
