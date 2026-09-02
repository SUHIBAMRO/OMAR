"""Records the B2 x Neo-Hookean accuracy/cost Pareto -- the first of the
three B2 materials to finish (B1's three are already recorded in this
directory).

SOURCE: fetched directly from Google Drive (file id
1hjWfVmpXDmNdrTjrwk7E8QVecweNzTOc, pfem_run/zeroshot_B2_neo_hookean_fixedsel/
pareto_B2_neo_hookean.json), not transcribed from stdout.

FINDING THIS FILE EXISTS TO RECORD: the operator's error is not smooth in
N the way every B1 Pareto curve is. It has sharp LOCAL MINIMA exactly at
N=21 and N=33 -- the two resolutions this checkpoint was jointly trained
on -- 3.6x-4.3x lower than the neighbouring untrained meshes on each side.
B1's Neo-Hookean Pareto (pareto_B1_neo_hookean.json), evaluated from a
checkpoint trained at N=21 only, shows no such dip at N=21: its curve is
smooth and monotone through that point. The candidate explanation -- joint
training at two specific resolutions leaves two visible anchor points that
single-resolution training does not -- is stated as a candidate, not
established; nothing here rules out a B2-geometry effect instead of a
training-protocol one, since the two checkpoints differ in both ways at
once.

Usage: python3 record_pareto_b2_neo_hookean.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pareto_B2_neo_hookean.json")

NH_B1 = json.load(open(os.path.join(HERE, "pareto_B1_neo_hookean.json")))

# Verbatim from the Drive JSON.
RAW_ROWS = [
    {"N": 13, "n_nodes": 169, "fem_rel_L2": 0.019436165724500097,
     "fem_ms_per_sample": 10108.463829000357, "operator_rel_L2": 0.19858644450756793,
     "operator_ms_per_sample": 5.814663500132156},
    {"N": 17, "n_nodes": 289, "fem_rel_L2": 0.01120957031236559,
     "fem_ms_per_sample": 18024.733190000006, "operator_rel_L2": 0.10512962639262244,
     "operator_ms_per_sample": 5.786646999695222},
    {"N": 21, "n_nodes": 441, "fem_rel_L2": 0.00727896126910976,
     "fem_ms_per_sample": 27916.097556998466, "operator_rel_L2": 0.017859230263987506,
     "operator_ms_per_sample": 6.192653998368769},
    {"N": 25, "n_nodes": 625, "fem_rel_L2": 0.00519442099718464,
     "fem_ms_per_sample": 40024.49283900023, "operator_rel_L2": 0.048375295048719416,
     "operator_ms_per_sample": 5.714047500077868},
    {"N": 29, "n_nodes": 841, "fem_rel_L2": 0.003901066813714277,
     "fem_ms_per_sample": 54559.490818000995, "operator_rel_L2": 0.06777962732237545,
     "operator_ms_per_sample": 5.80108650137845},
    {"N": 33, "n_nodes": 1089, "fem_rel_L2": 0.0030595788773517913,
     "fem_ms_per_sample": 71714.15642650027, "operator_rel_L2": 0.01863013705974454,
     "operator_ms_per_sample": 5.875323997315718},
    {"N": 37, "n_nodes": 1369, "fem_rel_L2": 0.002478235967796337,
     "fem_ms_per_sample": 90885.42874499944, "operator_rel_L2": 0.06810450491313229,
     "operator_ms_per_sample": 5.87725550030882},
    {"N": 41, "n_nodes": 1681, "fem_rel_L2": 0.002040399562636437,
     "fem_ms_per_sample": 111848.18404900034, "operator_rel_L2": 0.13225479899557527,
     "operator_ms_per_sample": 5.98004349922121},
    {"N": 49, "n_nodes": 2401, "fem_rel_L2": 0.0015178359073748852,
     "fem_ms_per_sample": 161601.54347700154, "operator_rel_L2": 0.24045019626956768,
     "operator_ms_per_sample": 5.899513998883776},
]
CHECKPOINT_FINGERPRINT = "1afbcd250a2933e201ac3cafc5ec51cdbad45e39703499a3508547a9ead40b21"
TRAIN_RESOLUTIONS = [21, 33]
WALL_CLOCK_S = 7 * 3600 + 11 * 60 + 17   # "7h 11m 17s", run manifest

NS = [r["N"] for r in RAW_ROWS]
assert NS == [13, 17, 21, 25, 29, 33, 37, 41, 49]

rows = [dict(r, speedup=r["fem_ms_per_sample"] / r["operator_ms_per_sample"])
        for r in RAW_ROWS]
op = {r["N"]: r["operator_rel_L2"] for r in rows}
fem = {r["N"]: r["fem_rel_L2"] for r in rows}
speedup = {r["N"]: r["speedup"] for r in rows}

# --- the training-resolution anchoring effect
NEIGHBORS = {21: (17, 25), 33: (29, 37)}
anchor_ratio = {}
for trainN, (lo, hi) in NEIGHBORS.items():
    neighbor_mean = (op[lo] + op[hi]) / 2
    anchor_ratio[trainN] = neighbor_mean / op[trainN]
    assert op[trainN] < op[lo] and op[trainN] < op[hi], (
        f"expected a local minimum at N={trainN}", trainN, op[trainN], op[lo], op[hi])
    assert anchor_ratio[trainN] > 3, (trainN, anchor_ratio[trainN])

# --- B1's Neo-Hookean Pareto, trained at N=21 only, shows no such dip
b1_op = {r["N"]: r["operator_rel_L2"] for r in NH_B1["rows"]}
b1_neighbor_mean = (b1_op[17] + b1_op[25]) / 2
b1_ratio_at_21 = b1_neighbor_mean / b1_op[21]
assert 0.8 < b1_ratio_at_21 < 1.3, (
    "expected B1's curve to show NO anchoring dip at N=21 (ratio near 1); "
    "re-check the contrast this file states", b1_ratio_at_21)

print(f"B2xNH anchoring ratio at N=21: {anchor_ratio[21]:.2f}x, "
      f"at N=33: {anchor_ratio[33]:.2f}x")
print(f"B1xNH (trained at N=21 only) neighbor-mean/actual ratio at N=21: "
      f"{b1_ratio_at_21:.2f}x -- no dip")
print(f"speed-up {min(speedup.values()):,.0f}x-{max(speedup.values()):,.0f}x")

report = {
    "geometry": "B2",
    "material": "neo_hookean",
    "checkpoint": "pfem_run/zeroshot_B2_neo_hookean_fixedsel/model_best.pt",
    "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
    "fine_N": 101,
    "batch_size": 1,
    "n_samples": 20,
    "train_resolutions": TRAIN_RESOLUTIONS,
    "solver_side": "the CPU reference solver, the cost of producing a new "
        "solution today; the GPU-native solver of section 8.5 is "
        "71.7-171.5x faster and shifts the speed-up column down by about "
        "two orders of magnitude without touching accuracy",
    "metric": "combined relative L2 over both displacement components, "
        "||e||/||u||, every row scored against the SAME N=101 reference -- "
        "the convergence-study convention of section 4.4",
    "rows": rows,

    "shape": {
        "training_resolution_anchoring": {
            "note": f"the operator error has sharp LOCAL MINIMA exactly at "
                f"N=21 and N=33, the two meshes this checkpoint was jointly "
                f"trained on: {anchor_ratio[21]:.2f}x and "
                f"{anchor_ratio[33]:.2f}x better than the mean of each "
                f"point's immediate neighbours. B1 x Neo-Hookean's Pareto, "
                f"scored from a checkpoint trained at N=21 only, shows NO "
                f"such dip at N=21 (ratio {b1_ratio_at_21:.2f}x, "
                f"effectively flat).",
            "candidate_explanation_not_established": "joint training at "
                "two specific resolutions may leave two visible anchor "
                "points that single-resolution training does not -- stated "
                "as a candidate, not established. The two checkpoints "
                "being compared differ in geometry (B1 vs B2) AND training "
                "protocol (one vs two resolutions) at once, so this file "
                "cannot separate which difference causes the contrast.",
            "anchor_ratios": anchor_ratio,
            "b1_neo_hookean_ratio_at_N21_for_comparison": b1_ratio_at_21,
        },
    },

    "reading": {
        "headline": f"B2 x Neo-Hookean's operator/FEM Pareto is measured, "
            f"speed-up {min(speedup.values()):,.0f}x-{max(speedup.values()):,.0f}x. "
            f"Unlike every B1 Pareto curve, this one is not smooth in N -- "
            f"see training_resolution_anchoring.",
        "cost_not_yet_established_for_B2": "PROJECT_STATUS.md and the "
            "cell's own header say B2's assembly cost has not been "
            "separately measured; B1's numbers were used as the closest "
            "guide before this run. This file's fem_ms_per_sample values "
            "are the first direct B2 measurement.",
    },

    "cost": f"A100, {WALL_CLOCK_S}s wall clock end to end "
        "(7h 11m 17s, run manifest). First of the three B2 materials to "
        "finish its Pareto sweep; Mooney-Rivlin started immediately after "
        "in the same session, Arruda-Boyce has not started.",

    "provenance": "fetched directly from Google Drive "
        "(pfem_run/zeroshot_B2_neo_hookean_fixedsel/pareto_B2_neo_hookean.json, "
        "file id 1hjWfVmpXDmNdrTjrwk7E8QVecweNzTOc), not transcribed from "
        "Colab stdout.",
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {OUT}")
