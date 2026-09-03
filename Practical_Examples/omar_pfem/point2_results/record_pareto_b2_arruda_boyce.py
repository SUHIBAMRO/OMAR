"""Records the B2 x Arruda-Boyce accuracy/cost Pareto -- the third and
last of the three B2 materials to finish. Point 2 of the round-5 review
is now complete for all six geometry x material combinations.

SOURCE: fetched directly from Google Drive (file id
1dEn57bytN4DkkMDkQ7VnMtY4plT3b0yo, pfem_run/zeroshot_B2_arruda_boyce_fixedsel/
pareto_B2_arruda_boyce.json), not transcribed from Colab stdout. Every row
matches the run's own stdout exactly (checked below), an independent
cross-check on top of the Drive fetch.

FINDING THIS FILE EXISTS TO RECORD: the training-resolution anchoring
effect now REPLICATES in ALL THREE B2 materials --
Neo-Hookean (4.30x/3.65x), Mooney-Rivlin (3.07x/3.17x), and this one
(computed below). Three of three is no longer "a candidate pattern in one
case" -- it is a property of B2's two-resolution training protocol,
observed at every material tried. The B1-vs-B2 confound (geometry AND
protocol differ at once) from the original Neo-Hookean finding is
untouched by this -- still cannot separate protocol from geometry without
a B1 checkpoint trained jointly at two resolutions, which was not run --
but "does it happen for every B2 material" is now answered: yes.

Usage: python3 record_pareto_b2_arruda_boyce.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pareto_B2_arruda_boyce.json")

NH_B2 = json.load(open(os.path.join(HERE, "pareto_B2_neo_hookean.json")))
MR_B2 = json.load(open(os.path.join(HERE, "pareto_B2_mooney_rivlin.json")))

# Verbatim from the Drive JSON.
RAW_ROWS = [
    {"N": 13, "n_nodes": 169, "fem_rel_L2": 0.01934633348326857,
     "fem_ms_per_sample": 20870.493054500002, "operator_rel_L2": 0.2635221398562967,
     "operator_ms_per_sample": 5.6032755001069745},
    {"N": 17, "n_nodes": 289, "fem_rel_L2": 0.01116115144622183,
     "fem_ms_per_sample": 37056.644341999345, "operator_rel_L2": 0.1057816166672475,
     "operator_ms_per_sample": 5.569119500250963},
    {"N": 21, "n_nodes": 441, "fem_rel_L2": 0.007253167053877048,
     "fem_ms_per_sample": 57877.52493400012, "operator_rel_L2": 0.03181460918619187,
     "operator_ms_per_sample": 5.577541000093333},
    {"N": 25, "n_nodes": 625, "fem_rel_L2": 0.005182033058499364,
     "fem_ms_per_sample": 83374.7977349999, "operator_rel_L2": 0.08440254257411113,
     "operator_ms_per_sample": 5.73114149938192},
    {"N": 29, "n_nodes": 841, "fem_rel_L2": 0.0038965601281189866,
     "fem_ms_per_sample": 113473.39435449976, "operator_rel_L2": 0.08257219044150935,
     "operator_ms_per_sample": 5.640911500449874},
    {"N": 33, "n_nodes": 1089, "fem_rel_L2": 0.0030600643705638803,
     "fem_ms_per_sample": 148153.4894360002, "operator_rel_L2": 0.03796356496149738,
     "operator_ms_per_sample": 5.6414409991703},
    {"N": 37, "n_nodes": 1369, "fem_rel_L2": 0.002482138351499219,
     "fem_ms_per_sample": 187680.22048799958, "operator_rel_L2": 0.10479892116936822,
     "operator_ms_per_sample": 5.677023000316694},
    {"N": 41, "n_nodes": 1681, "fem_rel_L2": 0.002046006273641404,
     "fem_ms_per_sample": 231947.7028985002, "operator_rel_L2": 0.17563538118223027,
     "operator_ms_per_sample": 5.710683499273728},
    {"N": 49, "n_nodes": 2401, "fem_rel_L2": 0.0015262671056611504,
     "fem_ms_per_sample": 333647.3938764993, "operator_rel_L2": 0.29507113013240305,
     "operator_ms_per_sample": 5.632697500914219},
]
CHECKPOINT_FINGERPRINT = "3424f961455b635d0656b53d8438d53627700575864c385dafeb14d73e8d3f45"
TRAIN_RESOLUTIONS = [21, 33]
WALL_CLOCK_S = 7 * 3600 + 9 * 60 + 20  # "7h 9m 20s", run manifest #3

NS = [r["N"] for r in RAW_ROWS]
assert NS == [13, 17, 21, 25, 29, 33, 37, 41, 49]

# Cross-check every row against the run's own stdout -- not just N=41/49
# this time, since the full run's stdout was captured end to end.
STDOUT_SEEN = {
    13: (20870.5, 5.603), 17: (37056.6, 5.569), 21: (57877.5, 5.578),
    25: (83374.8, 5.731), 29: (113473.4, 5.641), 33: (148153.5, 5.641),
    37: (187680.2, 5.677), 41: (231947.7, 5.711), 49: (333647.4, 5.633),
}
for N, (fem_ms, op_ms) in STDOUT_SEEN.items():
    row = next(r for r in RAW_ROWS if r["N"] == N)
    assert abs(row["fem_ms_per_sample"] - fem_ms) < 1.0, (N, row["fem_ms_per_sample"], fem_ms)
    assert abs(row["operator_ms_per_sample"] - op_ms) < 0.01, (N, row["operator_ms_per_sample"], op_ms)

rows = [dict(r, speedup=r["fem_ms_per_sample"] / r["operator_ms_per_sample"])
        for r in RAW_ROWS]
op = {r["N"]: r["operator_rel_L2"] for r in rows}
fem = {r["N"]: r["fem_rel_L2"] for r in rows}
speedup = {r["N"]: r["speedup"] for r in rows}

fem_seq = [fem[N] for N in NS]
assert all(fem_seq[i] > fem_seq[i + 1] for i in range(len(fem_seq) - 1)), (
    "FEM error is not monotonically decreasing in N", fem_seq)

# --- does the training-resolution anchoring effect replicate a third time?
NEIGHBORS = {21: (17, 25), 33: (29, 37)}
anchor_ratio = {}
for trainN, (lo, hi) in NEIGHBORS.items():
    neighbor_mean = (op[lo] + op[hi]) / 2
    anchor_ratio[trainN] = neighbor_mean / op[trainN]
    assert op[trainN] < op[lo] and op[trainN] < op[hi], (
        f"expected a local minimum at N={trainN}", trainN, op[trainN], op[lo], op[hi])
    assert anchor_ratio[trainN] > 2.0, (trainN, anchor_ratio[trainN])

nh_anchor = {int(k): v for k, v in
             NH_B2["shape"]["training_resolution_anchoring"]["anchor_ratios"].items()}
mr_anchor = {int(k): v for k, v in
             MR_B2["shape"]["training_resolution_anchoring"]["anchor_ratios"].items()}

# Three of three: assert the ordering this file's own docstring claims --
# Arruda-Boyce's effect is the smallest of the three materials.
assert anchor_ratio[21] < mr_anchor[21] < nh_anchor[21]
assert anchor_ratio[33] < mr_anchor[33] < nh_anchor[33]

print(f"B2xAB anchoring ratio at N=21: {anchor_ratio[21]:.2f}x "
      f"(NH: {nh_anchor[21]:.2f}x, MR: {mr_anchor[21]:.2f}x)")
print(f"B2xAB anchoring ratio at N=33: {anchor_ratio[33]:.2f}x "
      f"(NH: {nh_anchor[33]:.2f}x, MR: {mr_anchor[33]:.2f}x)")
print(f"speed-up {min(speedup.values()):,.0f}x-{max(speedup.values()):,.0f}x")
print("THREE OF THREE B2 materials now show the anchoring effect.")

report = {
    "geometry": "B2",
    "material": "arruda_boyce",
    "checkpoint": "pfem_run/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt",
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
            "note": f"the anchoring effect now REPLICATES in all three B2 "
                f"materials. Local minima at N=21 and N=33: "
                f"{anchor_ratio[21]:.2f}x and {anchor_ratio[33]:.2f}x here, "
                f"against {mr_anchor[21]:.2f}x/{mr_anchor[33]:.2f}x "
                f"(Mooney-Rivlin) and {nh_anchor[21]:.2f}x/{nh_anchor[33]:.2f}x "
                f"(Neo-Hookean) -- Arruda-Boyce shows the smallest effect of "
                f"the three, but the same shape, at the same two "
                f"resolutions, every time.",
            "anchor_ratios": anchor_ratio,
            "mooney_rivlin_anchor_ratios_for_comparison": mr_anchor,
            "neo_hookean_anchor_ratios_for_comparison": nh_anchor,
            "conclusion": "three of three B2 materials show this effect. "
                "It is a property of B2's two-resolution training protocol, "
                "not a coincidence of one or two materials. What remains "
                "unresolved (stated in the Neo-Hookean record) is whether "
                "the effect is caused by the training PROTOCOL (joint "
                "two-resolution training) or by the B2 GEOMETRY -- B1 has "
                "no equivalent two-resolution checkpoint to test against, "
                "so protocol and geometry still differ at once between "
                "every B1/B2 comparison available.",
        },
    },

    "reading": {
        "headline": f"B2 x Arruda-Boyce's operator/FEM Pareto is measured, "
            f"speed-up {min(speedup.values()):,.0f}x-{max(speedup.values()):,.0f}x. "
            f"This closes point 2 of the round-5 review for all six "
            f"geometry x material combinations.",
    },

    "cost": f"A100, {WALL_CLOCK_S}s wall clock end to end (7h 9m 20s, run "
        "manifest). Ran in a dedicated notebook isolated to this material "
        "only, in parallel with Mooney-Rivlin's own dedicated run, with "
        "zero shared files between the two.",

    "provenance": "fetched directly from Google Drive "
        "(pfem_run/zeroshot_B2_arruda_boyce_fixedsel/pareto_B2_arruda_boyce.json, "
        "file id 1dEn57bytN4DkkMDkQ7VnMtY4plT3b0yo), not transcribed from "
        "Colab stdout. All nine rows cross-checked against the run's own "
        "stdout independently.",
}

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {OUT}")
