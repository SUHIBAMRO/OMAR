"""Records the B2 x Mooney-Rivlin accuracy/cost Pareto -- the second of
the three B2 materials to finish (Neo-Hookean already recorded in this
directory; Arruda-Boyce still running as this is written).

SOURCE: fetched directly from Google Drive (file id
1uN8oJRbcEFzFrr7bnPHpCUQKokwdt0sX, pfem_run/zeroshot_B2_mooney_rivlin_fixedsel/
pareto_B2_mooney_rivlin.json), not transcribed from Colab stdout. The N=41
and N=49 rows were also visible in the run's stdout and match this file
exactly (239277.51 / 345074.72 ms per sample), an independent cross-check
on top of the Drive fetch.

FINDING THIS FILE EXISTS TO RECORD: the training-resolution anchoring
effect found in B2 x Neo-Hookean's Pareto (pareto_B2_neo_hookean.json)
REPLICATES here. The operator error again has local minima close to N=21
and N=33 -- the two resolutions this checkpoint was jointly trained on --
though the effect is smaller than Neo-Hookean's (~3.1x here against
3.6-4.3x there). Two of three B2 materials now show the same shape;
Arruda-Boyce is the deciding case for whether this is a general B2
property or a per-material coincidence.

Usage: python3 record_pareto_b2_mooney_rivlin.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pareto_B2_mooney_rivlin.json")

NH_B2 = json.load(open(os.path.join(HERE, "pareto_B2_neo_hookean.json")))

# Verbatim from the Drive JSON.
RAW_ROWS = [
    {"N": 13, "n_nodes": 169, "fem_rel_L2": 0.019525635046158986,
     "fem_ms_per_sample": 21566.057666001143, "operator_rel_L2": 0.165039340638395,
     "operator_ms_per_sample": 5.65918350184802},
    {"N": 17, "n_nodes": 289, "fem_rel_L2": 0.011258209446253106,
     "fem_ms_per_sample": 38272.3334350012, "operator_rel_L2": 0.0709803986791039,
     "operator_ms_per_sample": 5.612317498162156},
    {"N": 21, "n_nodes": 441, "fem_rel_L2": 0.007308095151896258,
     "fem_ms_per_sample": 59901.1025769978, "operator_rel_L2": 0.02079486322189907,
     "operator_ms_per_sample": 5.621075499220751},
    {"N": 25, "n_nodes": 625, "fem_rel_L2": 0.005213180585585557,
     "fem_ms_per_sample": 86515.03317150127, "operator_rel_L2": 0.05680641892930135,
     "operator_ms_per_sample": 5.572944999585161},
    {"N": 29, "n_nodes": 841, "fem_rel_L2": 0.003913633024101612,
     "fem_ms_per_sample": 117666.67327300092, "operator_rel_L2": 0.057309975076120026,
     "operator_ms_per_sample": 5.564422499446664},
    {"N": 33, "n_nodes": 1089, "fem_rel_L2": 0.0030682487014455665,
     "fem_ms_per_sample": 153728.79927699978, "operator_rel_L2": 0.024097411886594762,
     "operator_ms_per_sample": 5.686479002179112},
    {"N": 37, "n_nodes": 1369, "fem_rel_L2": 0.0024842581954679656,
     "fem_ms_per_sample": 194649.7673019985, "operator_rel_L2": 0.09539975094729769,
     "operator_ms_per_sample": 5.715952000173274},
    {"N": 41, "n_nodes": 1681, "fem_rel_L2": 0.0020446816743412825,
     "fem_ms_per_sample": 239277.51179250015, "operator_rel_L2": 0.16472587217132187,
     "operator_ms_per_sample": 5.639642000005551},
    {"N": 49, "n_nodes": 2401, "fem_rel_L2": 0.0015198860891366783,
     "fem_ms_per_sample": 345074.7159174994, "operator_rel_L2": 0.2790471304336879,
     "operator_ms_per_sample": 5.6894879999163095},
]
CHECKPOINT_FINGERPRINT = "4d76646d14674ad009b38a5165b88261192968c719b37687953c0cf872686a6a"
TRAIN_RESOLUTIONS = [21, 33]
# Only the resumed tail (N=41, N=49) ran in one continuous tracked session
# (run_manifest: "3h 15m 6s"); N=13-37 finished in an earlier session whose
# own wall clock was not captured. The total across both is NOT 3h15m6s --
# it is at least that plus whatever the first session took, which is not
# on record. Do not quote 3h15m6s as the full 9-resolution cost.
RESUMED_TAIL_WALL_CLOCK_S = 3 * 3600 + 15 * 60 + 6  # N=41, N=49 only

NS = [r["N"] for r in RAW_ROWS]
assert NS == [13, 17, 21, 25, 29, 33, 37, 41, 49]

# Cross-check against the values visible in the run's own stdout for the
# resumed tail -- independent of the Drive fetch.
STDOUT_SEEN = {41: 239277.5, 49: 345074.7}
for N, ms in STDOUT_SEEN.items():
    row = next(r for r in RAW_ROWS if r["N"] == N)
    assert abs(row["fem_ms_per_sample"] - ms) < 1.0, (
        N, row["fem_ms_per_sample"], ms)

rows = [dict(r, speedup=r["fem_ms_per_sample"] / r["operator_ms_per_sample"])
        for r in RAW_ROWS]
op = {r["N"]: r["operator_rel_L2"] for r in rows}
fem = {r["N"]: r["fem_rel_L2"] for r in rows}
speedup = {r["N"]: r["speedup"] for r in rows}

# FEM side should be monotonically decreasing in N regardless of material --
# a basic sanity check independent of the anchoring question below.
fem_seq = [fem[N] for N in NS]
assert all(fem_seq[i] > fem_seq[i + 1] for i in range(len(fem_seq) - 1)), (
    "FEM error is not monotonically decreasing in N", fem_seq)

# --- does the training-resolution anchoring effect replicate here?
NEIGHBORS = {21: (17, 25), 33: (29, 37)}
anchor_ratio = {}
for trainN, (lo, hi) in NEIGHBORS.items():
    neighbor_mean = (op[lo] + op[hi]) / 2
    anchor_ratio[trainN] = neighbor_mean / op[trainN]
    assert op[trainN] < op[lo] and op[trainN] < op[hi], (
        f"expected a local minimum at N={trainN}", trainN, op[trainN], op[lo], op[hi])
    assert anchor_ratio[trainN] > 2.5, (trainN, anchor_ratio[trainN])

nh_anchor = NH_B2["shape"]["training_resolution_anchoring"]["anchor_ratios"]
nh_anchor = {int(k): v for k, v in nh_anchor.items()}

print(f"B2xMR anchoring ratio at N=21: {anchor_ratio[21]:.2f}x "
      f"(Neo-Hookean: {nh_anchor[21]:.2f}x)")
print(f"B2xMR anchoring ratio at N=33: {anchor_ratio[33]:.2f}x "
      f"(Neo-Hookean: {nh_anchor[33]:.2f}x)")
print(f"speed-up {min(speedup.values()):,.0f}x-{max(speedup.values()):,.0f}x")

report = {
    "geometry": "B2",
    "material": "mooney_rivlin",
    "checkpoint": "pfem_run/zeroshot_B2_mooney_rivlin_fixedsel/model_best.pt",
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
            "note": f"the anchoring effect found in B2 x Neo-Hookean "
                f"REPLICATES here: local minima at N=21 and N=33, "
                f"{anchor_ratio[21]:.2f}x and {anchor_ratio[33]:.2f}x better "
                f"than the mean of each point's immediate neighbours -- "
                f"smaller than Neo-Hookean's {nh_anchor[21]:.2f}x / "
                f"{nh_anchor[33]:.2f}x but the same shape, at the same two "
                f"resolutions. Two of three B2 materials now show it.",
            "anchor_ratios": anchor_ratio,
            "neo_hookean_anchor_ratios_for_comparison": nh_anchor,
            "still_open": "Arruda-Boyce, the third and last B2 material, "
                "decides whether this is a general property of B2's "
                "two-resolution training protocol or coincidental to these "
                "two materials.",
        },
    },

    "reading": {
        "headline": f"B2 x Mooney-Rivlin's operator/FEM Pareto is measured, "
            f"speed-up {min(speedup.values()):,.0f}x-{max(speedup.values()):,.0f}x. "
            f"The training-resolution anchoring effect replicates from "
            f"Neo-Hookean, at a smaller magnitude.",
    },

    "cost": f"A100. Only the resumed tail (N=41, N=49) has a captured wall "
        f"clock: {RESUMED_TAIL_WALL_CLOCK_S}s (3h 15m 6s, run manifest). "
        f"N=13-37 finished in an earlier session whose own wall clock was "
        f"not recorded, so the full 9-resolution cost is NOT 3h15m6s -- it "
        f"is at least that much more, unmeasured.",

    "provenance": "fetched directly from Google Drive "
        "(pfem_run/zeroshot_B2_mooney_rivlin_fixedsel/pareto_B2_mooney_rivlin.json, "
        "file id 1uN8oJRbcEFzFrr7bnPHpCUQKokwdt0sX), not transcribed from "
        "Colab stdout. N=41/N=49 cross-checked against the run's own stdout "
        "independently.",
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {OUT}")
